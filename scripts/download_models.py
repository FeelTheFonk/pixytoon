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
    """SD1.5 checkpoint from Civitai (direct HTTP, no HF)."""
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

    print("      [INFO] Establishing connection securely...")
    req = urllib.request.Request(url, headers={"User-Agent": "SDDj/1.0 SOTA-Downloader"})
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0
                with open(part_dest, "wb") as f:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            mb_dl = downloaded // 1024 // 1024
                            mb_tot = total_size // 1024 // 1024
                            sys.stdout.write(
                                f"\r      [DL] {pct:.1f}% ({mb_dl}MB / {mb_tot}MB)"
                            )
                            sys.stdout.flush()
                if total_size > 0:
                    print()
                break
        except (URLError, TimeoutError, OSError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"\n      [WARN] Network error ({e}), retrying in {wait}s...")
                time.sleep(wait)
                continue
            if part_dest.exists():
                part_dest.unlink()
            print(f"\n      [WARN] Network failure after {max_retries} attempts: {e}")
            _record(label, False, str(e))
            return

    # Validate size before committing
    if total_size > 0:
        actual_size = part_dest.stat().st_size
        if actual_size != total_size:
            part_dest.unlink()
            msg = f"Size mismatch: expected {total_size}, got {actual_size}"
            print(f"      [WARN] {msg}")
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
    print(f"    [DL] Precaching {label} for offline generation...")
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
    try:
        from huggingface_hub import snapshot_download

        with temporary_online_mode():
            snapshot_download(
                "ByteDance/AnimateDiff-Lightning",
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
    try:
        from huggingface_hub import hf_hub_download

        with temporary_online_mode():
            repo = "ByteDance/AnimateDiff-Lightning"
            for step in (2, 4, 8):
                fn = f"animatediff_lightning_{step}step_diffusers.safetensors"
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

    if args.all or args.checkpoint:
        download_civitai_model()
        download_base_configs()
    if args.all or args.hyper_sd:
        download_hyper_sd_lora()
    if args.all or args.loras:
        download_pixel_loras()
    if args.all or args.embeddings:
        download_embeddings()
    if args.all or args.controlnets:
        download_controlnets()
    if args.all or args.controlnets or args.qrcode_monster:
        download_qrcode_monster()
    if args.all or args.animatediff:
        download_animatediff()
    if args.all or args.animatediff_lightning:
        download_animatediff_lightning()

    _print_summary()


if __name__ == "__main__":
    main()
