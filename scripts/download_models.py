#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# HF offline-mode override
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# I/O constants
# ---------------------------------------------------------------------------

CHUNK_SIZE = 512 * 1024  # 512KB — optimal for modern networks (30%+ gain over 32KB)


@contextlib.contextmanager
def temporary_online_mode():
    """Temporarily lift HF_HUB_OFFLINE so the setup script can download.

    Every HF download in this script MUST run inside this context manager.
    At runtime the server honours the .env offline flag; here we override it
    because the explicit purpose of this script is to populate caches.
    """
    import huggingface_hub.constants as hf_constants

    saved_env: dict[str, str] = {}
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        val = os.environ.pop(key, None)
        if val is not None:
            saved_env[key] = val

    _was_offline = getattr(hf_constants, "HF_HUB_OFFLINE", False)
    hf_constants.HF_HUB_OFFLINE = False

    try:
        yield
    finally:
        hf_constants.HF_HUB_OFFLINE = _was_offline
        for key, val in saved_env.items():
            os.environ[key] = val


# ---------------------------------------------------------------------------
# HF cache probe — zero-network existence check
# ---------------------------------------------------------------------------

def _hf_file_cached(repo_id: str, filename: str, *, repo_type: str = "model") -> bool:
    """Return True if *filename* is already in the local HF hub cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        result = try_to_load_from_cache(repo_id, filename, repo_type=repo_type)
        return isinstance(result, str)  # str path → cached; None → not cached
    except Exception:
        return False


def _hf_snapshot_cached(repo_id: str, sentinel: str = "config.json") -> bool:
    """Fast probe: True if a snapshot_download repo is already locally cached."""
    return _hf_file_cached(repo_id, sentinel)


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

_results: list[tuple[str, bool, str]] = []


def _record(label: str, ok: bool, detail: str = "") -> None:
    _results.append((label, ok, detail))


def _streaming_sha256(path: Path) -> str:
    """Compute SHA-256 of a file via streaming (safe for multi-GB files)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


# Expected checksums — None means skip verification (first download populates)
EXPECTED_CHECKSUMS: dict[str, str | None] = {
    "liberteRedmond_v10.safetensors": None,  # Populated after first verified download
}


def _print_summary() -> None:
    if not _results:
        return
    failures = [(l, d) for l, ok, d in _results if not ok]
    if not failures:
        print("\n    [OK] All downloads complete.")
        return
    ok_count = sum(1 for _, ok, _ in _results if ok)
    print(f"\n    [WARN] {len(failures)}/{len(_results)} download(s) failed "
          f"({ok_count} succeeded):")
    for label, detail in failures:
        short = (detail[:90] + "…") if len(detail) > 90 else detail
        print(f"      - {label}: {short}")
    print("    Re-run with network access or provide files manually.")


# ---------------------------------------------------------------------------
# Individual downloaders
# ---------------------------------------------------------------------------

def download_civitai_model() -> None:
    """SD1.5 checkpoint from Civitai (direct HTTP, no HF). Supports HTTP Range resume."""
    import time
    import urllib.request
    from urllib.error import URLError

    label = "SD1.5 Checkpoint (Liberte.Redmond)"
    print(f"    [DL] {label} ...")
    url = "https://civitai.com/api/download/models/100409"
    checkpoints_dir = _PROJECT_ROOT / "server" / "models" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    dest = checkpoints_dir / "liberteRedmond_v10.safetensors"
    part_dest = dest.with_suffix(".safetensors.part")

    if dest.exists():
        print(f"      [SKIP] {dest.name} already exists.")
        _record(label, True, "cached")
        return

    # Check for existing partial download (resume support)
    existing_size = 0
    if part_dest.exists():
        existing_size = part_dest.stat().st_size
        print(f"      [INFO] Resuming from {existing_size // (1024 * 1024)}MB ...")

    max_retries = 3
    total_size = 0
    for attempt in range(max_retries):
        try:
            req_headers = {"User-Agent": "SDDj/1.0 SOTA-Downloader"}
            if existing_size > 0:
                req_headers["Range"] = f"bytes={existing_size}-"
            req = urllib.request.Request(url, headers=req_headers)

            with urllib.request.urlopen(req, timeout=30) as response:
                status = response.status
                content_length = int(response.headers.get("content-length", 0))

                if status == 206 and existing_size > 0:
                    # Server supports Range — append to existing .part file
                    total_size = existing_size + content_length
                    downloaded = existing_size
                    mode = "ab"
                else:
                    # Fresh download (server ignored Range or first attempt)
                    total_size = content_length
                    downloaded = 0
                    existing_size = 0
                    mode = "wb"

                with open(part_dest, mode) as f:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            mb_dl = downloaded // (1024 * 1024)
                            mb_tot = total_size // (1024 * 1024)
                            sys.stdout.write(
                                f"\r      [DL] {pct:.1f}% ({mb_dl}MB / {mb_tot}MB)"
                            )
                            sys.stdout.flush()
                if total_size > 0:
                    print()
                break
        except (URLError, TimeoutError, OSError) as e:
            # Update existing_size for next resume attempt
            if part_dest.exists():
                existing_size = part_dest.stat().st_size
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"\n      [WARN] Network error ({e}), retrying in {wait}s...")
                time.sleep(wait)
                continue
            # Keep .part file for resume on next script invocation
            print(f"\n      [WARN] Network failure after {max_retries} attempts: {e}")
            print(f"      [INFO] Partial download preserved ({existing_size // (1024 * 1024)}MB) — re-run to resume.")
            _record(label, False, str(e))
            return

    # Validate size before committing
    if total_size > 0:
        actual_size = part_dest.stat().st_size
        if actual_size != total_size:
            # Keep .part for resume rather than deleting
            msg = f"Size mismatch: expected {total_size}, got {actual_size}"
            print(f"      [WARN] {msg}")
            print(f"      [INFO] Partial download preserved — re-run to resume.")
            _record(label, False, msg)
            return

    part_dest.rename(dest)

    # Checksum verification (if known)
    expected_hash = EXPECTED_CHECKSUMS.get(dest.name)
    if expected_hash:
        print("      [INFO] Verifying checksum...")
        actual_hash = _streaming_sha256(dest)
        if actual_hash != expected_hash:
            dest.unlink()
            msg = f"Checksum mismatch: expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
            print(f"      [WARN] {msg}")
            _record(label, False, msg)
            return

    print("      [OK] Liberte.Redmond safely secured.")
    _record(label, True)


def download_base_configs() -> None:
    label = "SD1.5 architecture configs"
    print(f"    [DL] Precaching {label} for offline generation\u2026")
    if _hf_snapshot_cached("runwayml/stable-diffusion-v1-5", "model_index.json"):
        print("      [SKIP] configs already cached.")
        _record(label, True, "cached")
        return
    try:
        from huggingface_hub import snapshot_download

        with temporary_online_mode():
            snapshot_download(
                "runwayml/stable-diffusion-v1-5",
                allow_patterns=[
                    "*.json", "tokenizer/*", "scheduler/*",
                    "text_encoder/*", "unet/*", "vae/*",
                ],
                ignore_patterns=["*.bin", "*.safetensors", "*.h5", "*.msgpack"],
            )
        print("      [OK] Architecture configs cached.")
        _record(label, True)
    except (OSError, ValueError) as e:
        print(f"      [WARN] {label}: {e}")
        _record(label, False, str(e))


def download_hyper_sd_lora() -> None:
    label = "Hyper-SD LoRA"
    print(f"    [DL] {label}: ByteDance/Hyper-SD ...")
    if _hf_file_cached("ByteDance/Hyper-SD", "Hyper-SD15-8steps-CFG-lora.safetensors"):
        print("      [SKIP] Hyper-SD LoRA already cached.")
        _record(label, True, "cached")
        return
    try:
        from huggingface_hub import hf_hub_download

        with temporary_online_mode():
            hf_hub_download(
                "ByteDance/Hyper-SD",
                filename="Hyper-SD15-8steps-CFG-lora.safetensors",
            )
        print("      [OK] Hyper-SD LoRA cached.")
        _record(label, True)
    except (OSError, ValueError) as e:
        print(f"      [WARN] {label}: {e}")
        _record(label, False, str(e))


def download_pixel_loras() -> None:
    label = "Pixel art LoRAs"
    print(f"    [DL] {label} ...")
    loras_dir = _PROJECT_ROOT / "server" / "models" / "loras"
    loras_dir.mkdir(parents=True, exist_ok=True)

    loras = [
        {
            "repo": "artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5",
            "file": "PixelArtRedmond15V-PixelArt-PIXARFK.safetensors",
            "local": "pixelart_redmond.safetensors",
        },
    ]

    all_ok = True
    for lora in loras:
        dest = loras_dir / lora["local"]
        if dest.exists():
            print(f"      [SKIP] {lora['local']} already exists.")
            continue
        print(f"      [DL] {lora['repo']} -> {lora['local']}")
        try:
            from huggingface_hub import hf_hub_download

            with temporary_online_mode():
                downloaded = hf_hub_download(lora["repo"], filename=lora["file"])
            shutil.copy2(downloaded, dest)
            print(f"      [OK] {lora['local']}")
        except (OSError, ValueError) as e:
            print(f"      [WARN] {lora['local']}: {e}")
            all_ok = False

    if all_ok:
        print("      [OK] Pixel art LoRAs ready.")
    _record(label, all_ok)


def download_embeddings() -> None:
    label = "Negative TI embeddings"
    print(f"    [DL] {label} ...")
    embeddings_dir = _PROJECT_ROOT / "server" / "models" / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    embeddings = [
        {
            "repo": "gsdf/EasyNegative",
            "file": "EasyNegative.safetensors",
            "local": "EasyNegative.safetensors",
            "repo_type": "dataset",
        },
    ]

    all_ok = True
    for emb in embeddings:
        dest = embeddings_dir / emb["local"]
        if dest.exists():
            print(f"      [SKIP] {emb['local']} already exists.")
            continue
        print(f"      [DL] {emb['repo']} -> {emb['local']}")
        try:
            from huggingface_hub import hf_hub_download

            with temporary_online_mode():
                downloaded = hf_hub_download(
                    emb["repo"],
                    filename=emb["file"],
                    repo_type=emb["repo_type"],
                )
            shutil.copy2(downloaded, dest)
            print(f"      [OK] {emb['local']}")
        except (OSError, ValueError) as e:
            print(f"      [WARN] {emb['local']}: {e}")
            all_ok = False

    if all_ok:
        print("      [OK] TI embeddings ready.")
    _record(label, all_ok)


def download_controlnets() -> None:
    label = "ControlNet v1.1"
    print(f"    [DL] {label} models ...")
    models = {
        "openpose": "lllyasviel/control_v11p_sd15_openpose",
        "canny": "lllyasviel/control_v11p_sd15_canny",
        "scribble": "lllyasviel/control_v11p_sd15_scribble",
        "lineart": "lllyasviel/control_v11p_sd15_lineart",
    }
    all_ok = True
    for name, repo in models.items():
        if _hf_snapshot_cached(repo):
            print(f"      [SKIP] {name} already cached.")
            continue
        print(f"      [DL] {repo} ...")
        try:
            from huggingface_hub import snapshot_download

            with temporary_online_mode():
                snapshot_download(repo, ignore_patterns=["*.bin"])
            print(f"      [OK] {name}")
        except (OSError, ValueError) as e:
            print(f"      [WARN] {name}: {e}")
            all_ok = False

    if all_ok:
        print("      [OK] ControlNet models cached.")
    _record(label, all_ok)


def download_qrcode_monster() -> None:
    label = "ControlNet QR Code Monster v2"
    print(f"    [DL] {label} ...")
    if _hf_snapshot_cached("monster-labs/control_v1p_sd15_qrcode_monster", "v2/config.json"):
        print(f"      [SKIP] {label} already cached.")
        _record(label, True, "cached")
        return
    try:
        from huggingface_hub import snapshot_download

        with temporary_online_mode():
            snapshot_download(
                "monster-labs/control_v1p_sd15_qrcode_monster",
                allow_patterns=["v2/*", "*.json"],
                ignore_patterns=["*.bin"],
            )
        print(f"      [OK] {label} cached.")
        _record(label, True)
    except (OSError, ValueError) as e:
        print(f"      [WARN] {label}: {e}")
        _record(label, False, str(e))


def download_animatediff() -> None:
    label = "AnimateDiff motion adapter"
    print(f"    [DL] {label} ...")
    if _hf_snapshot_cached("guoyww/animatediff-motion-adapter-v1-5-2"):
        print(f"      [SKIP] {label} already cached.")
        _record(label, True, "cached")
        return
    try:
        from huggingface_hub import snapshot_download

        with temporary_online_mode():
            snapshot_download(
                "guoyww/animatediff-motion-adapter-v1-5-2",
                ignore_patterns=["*.bin"],
            )
        print(f"      [OK] {label} cached.")
        _record(label, True)
    except (OSError, ValueError) as e:
        print(f"      [WARN] {label}: {e}")
        _record(label, False, str(e))


def download_animatediff_lightning() -> None:
    label = "AnimateDiff-Lightning checkpoints"
    print(f"    [DL] {label} ...")
    repo = "ByteDance/AnimateDiff-Lightning"
    try:
        from huggingface_hub import hf_hub_download

        all_cached = True
        with temporary_online_mode():
            for step in (2, 4, 8):
                fn = f"animatediff_lightning_{step}step_diffusers.safetensors"
                if _hf_file_cached(repo, fn):
                    print(f"      [SKIP] {fn} already cached.")
                    continue
                all_cached = False
                print(f"      [DL] {fn} ...")
                hf_hub_download(repo, filename=fn)
                print(f"      [OK] {fn}")
        print(f"      [OK] {label} cached in HF hub.")
        _record(label, True)
    except (OSError, ValueError) as e:
        print(f"      [WARN] {label}: {e}")
        _record(label, False, str(e))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description="Download SDDj models")
    parser.add_argument("--all", action="store_true", help="Download everything")
    parser.add_argument("--checkpoint", action="store_true",
                        help="SD1.5 base checkpoint")
    parser.add_argument("--hyper-sd", action="store_true", dest="hyper_sd",
                        help="Hyper-SD acceleration LoRA")
    parser.add_argument("--loras", action="store_true",
                        help="Pixel art LoRAs")
    parser.add_argument("--embeddings", action="store_true",
                        help="Negative TI embeddings")
    parser.add_argument("--controlnets", action="store_true",
                        help="ControlNet v1.1 models")
    parser.add_argument("--qrcode-monster", action="store_true",
                        dest="qrcode_monster",
                        help="ControlNet QR Code Monster v2")
    parser.add_argument("--animatediff", action="store_true",
                        help="AnimateDiff motion adapter")
    parser.add_argument("--animatediff-lightning", action="store_true",
                        dest="animatediff_lightning",
                        help="AnimateDiff-Lightning checkpoints (2/4/8 step)")
    args = parser.parse_args()

    if not any([
        args.all, args.checkpoint, args.hyper_sd, args.loras,
        args.embeddings, args.controlnets, args.qrcode_monster,
        args.animatediff, args.animatediff_lightning,
    ]):
        args.all = True

    # Sequential: base checkpoint and configs must complete first
    if args.all or args.checkpoint:
        download_civitai_model()
        download_base_configs()

    # Collect independent downloads for parallel execution
    parallel_tasks: list[tuple[str, callable]] = []
    if args.all or args.hyper_sd:
        parallel_tasks.append(("Hyper-SD LoRA", download_hyper_sd_lora))
    if args.all or args.loras:
        parallel_tasks.append(("Pixel LoRAs", download_pixel_loras))
    if args.all or args.embeddings:
        parallel_tasks.append(("Embeddings", download_embeddings))
    if args.all or args.controlnets:
        parallel_tasks.append(("ControlNets", download_controlnets))
    if args.all or args.controlnets or args.qrcode_monster:
        parallel_tasks.append(("QR Monster", download_qrcode_monster))
    if args.animatediff:
        parallel_tasks.append(("AnimateDiff", download_animatediff))
    if args.all or args.animatediff_lightning:
        parallel_tasks.append(("AnimateDiff Lightning", download_animatediff_lightning))

    if parallel_tasks:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for name, func in parallel_tasks:
                futures[executor.submit(func)] = name

            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"    [FAIL] {name}: {e}")
                    _record(name, False, str(e))

    _print_summary()


if __name__ == "__main__":
    main()
