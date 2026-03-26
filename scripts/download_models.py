
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

import contextlib

@contextlib.contextmanager
def temporary_online_mode():
    import os
    import huggingface_hub.constants as hf_constants
    from huggingface_hub.utils._http import reset_sessions

    saved_env: dict[str, str] = {}
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        val = os.environ.pop(key, None)
        if val is not None:
            saved_env[key] = val

    _was_offline = getattr(hf_constants, "HF_HUB_OFFLINE", False)
    hf_constants.HF_HUB_OFFLINE = False
    reset_sessions()

    try:
        yield
    finally:
        hf_constants.HF_HUB_OFFLINE = _was_offline
        reset_sessions()
        for key, val in saved_env.items():
            os.environ[key] = val


def download_civitai_model() -> None:
    import shutil
    import sys
    import urllib.request
    from urllib.error import URLError
    from urllib.error import URLError
    
    print("    [DL] SD1.5 Checkpoint: Liberte.Redmond (Civitai) ...")
    url = "https://civitai.com/api/download/models/100409"
    checkpoints_dir = _PROJECT_ROOT / "server" / "models" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    dest = checkpoints_dir / "liberteRedmond_v10.safetensors"
    part_dest = dest.with_suffix(".safetensors.part")
    
    if dest.exists():
        print(f"      [SKIP] {dest.name} already exists.")
        return

    print("      [INFO] Establishing connection securely...")
    req = urllib.request.Request(url, headers={"User-Agent": "SDDj/1.0 SOTA-Downloader"})
    import time
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0
                with open(part_dest, "wb") as f:
                    while True:
                        chunk = response.read(8192 * 4)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            sys.stdout.write(f"\\r      [DL] {percent:.1f}% ({downloaded//1024//1024}MB / {total_size//1024//1024}MB)")
                            sys.stdout.flush()
                if total_size > 0:
                    print()
                break
        except (URLError, TimeoutError) as e:
            if attempt < max_retries - 1:
                print(f"\\n  [WARN] Network error ({e}), retrying in {2**attempt}s...")
                time.sleep(2**attempt)
                continue
            if part_dest.exists():
                part_dest.unlink()
            print(f"\\n  [ERROR] Network failure after {max_retries} attempts: {e}")
            sys.exit(1)
        
    part_dest.rename(dest)
    print("      [OK] Liberte.Redmond safely secured.")


def download_base_configs() -> None:
    print("    [DL] Precaching SD1.5 architecture configs for offline generation...")
    from huggingface_hub import snapshot_download
    with temporary_online_mode():
        snapshot_download(
            "runwayml/stable-diffusion-v1-5",
            allow_patterns=["*.json", "tokenizer/*", "scheduler/*", "text_encoder/*", "unet/*", "vae/*"],
            ignore_patterns=["*.bin", "*.safetensors", "*.h5", "*.msgpack"],
        )
    print("      [OK] Architecture configs cached.")


def download_hyper_sd_lora() -> None:
    from huggingface_hub import hf_hub_download
    print("    [DL] Hyper-SD LoRA: ByteDance/Hyper-SD ...")
    hf_hub_download("ByteDance/Hyper-SD", filename="Hyper-SD15-8steps-CFG-lora.safetensors")
    print("      [OK] Hyper-SD LoRA cached.")


def download_pixel_loras() -> None:
    from huggingface_hub import hf_hub_download
    print("    [DL] Pixel art LoRAs ...")
    loras_dir = _PROJECT_ROOT / "server" / "models" / "loras"
    loras_dir.mkdir(parents=True, exist_ok=True)

    loras = [
        {
            "repo": "artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5",
            "file": "PixelArtRedmond15V-PixelArt-PIXARFK.safetensors",
            "local": "pixelart_redmond.safetensors",
        },
    ]

    for lora in loras:
        dest = loras_dir / lora["local"]
        if dest.exists():
            print(f"      [SKIP] {lora['local']} already exists, skipping.")
            continue
        print(f"      [DL] {lora['repo']} -> {lora['local']}")
        downloaded = hf_hub_download(lora["repo"], filename=lora["file"])
        shutil.copy2(downloaded, dest)
        print(f"      [OK] {lora['local']}")

    print("      [OK] Pixel art LoRAs downloaded.")


def download_embeddings() -> None:
    from huggingface_hub import hf_hub_download
    print("    [DL] Negative TI embeddings ...")
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

    for emb in embeddings:
        dest = embeddings_dir / emb["local"]
        if dest.exists():
            print(f"      [SKIP] {emb['local']} already exists, skipping.")
            continue
        print(f"      [DL] {emb['repo']} -> {emb['local']}")
        downloaded = hf_hub_download(
            emb["repo"],
            filename=emb["file"],
            repo_type=emb["repo_type"],
        )
        shutil.copy2(downloaded, dest)
        print(f"      [OK] {emb['local']}")

    print("      [OK] TI embeddings downloaded.")


def download_controlnets() -> None:
    from huggingface_hub import snapshot_download
    print("    [DL] ControlNet v1.1 models ...")
    models = {
        "openpose": "lllyasviel/control_v11p_sd15_openpose",
        "canny": "lllyasviel/control_v11p_sd15_canny",
        "scribble": "lllyasviel/control_v11p_sd15_scribble",
        "lineart": "lllyasviel/control_v11p_sd15_lineart",
    }
    for name, repo in models.items():
        print(f"      [DL] {repo} ...")
        # Pre-cache in HF cache — engine loads by ID
        snapshot_download(repo, ignore_patterns=["*.bin"])
        print(f"      [OK] {name}")
    print("      [OK] ControlNet models cached.")


def download_qrcode_monster() -> None:
    from huggingface_hub import snapshot_download
    print("    [DL] ControlNet QR Code Monster v2 ...")
    with temporary_online_mode():
        snapshot_download(
            "monster-labs/control_v1p_sd15_qrcode_monster",
            allow_patterns=["v2/*", "*.json"],
            ignore_patterns=["*.bin"],
        )
    print("      [OK] QR Code Monster v2 cached.")



def download_animatediff() -> None:
    from huggingface_hub import snapshot_download
    print("    [DL] AnimateDiff motion adapter ...")
    snapshot_download(
        "ByteDance/AnimateDiff-Lightning",
        ignore_patterns=["*.bin"],
    )
    print("      [OK] AnimateDiff motion adapter cached.")


def download_animatediff_lightning() -> None:
    from huggingface_hub import hf_hub_download

    with temporary_online_mode():
        print("    [DL] AnimateDiff-Lightning checkpoints ...")
        repo = "ByteDance/AnimateDiff-Lightning"
        for step in (2, 4, 8):
            fn = f"animatediff_lightning_{step}step_diffusers.safetensors"
            print(f"      [DL] {fn} ...")
            hf_hub_download(repo, filename=fn)
            print(f"      [OK] {fn}")
        print("      [OK] AnimateDiff-Lightning cached in HF hub.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SDDj models")
    parser.add_argument("--all", action="store_true", help="Download everything")
    parser.add_argument("--checkpoint", action="store_true", help="SD1.5 base checkpoint")
    parser.add_argument("--hyper-sd", action="store_true", dest="hyper_sd", help="Hyper-SD acceleration LoRA")
    parser.add_argument("--loras", action="store_true", help="Pixel art LoRAs")
    parser.add_argument("--embeddings", action="store_true", help="Negative TI embeddings")
    parser.add_argument("--controlnets", action="store_true", help="ControlNet v1.1 models")
    parser.add_argument("--qrcode-monster", action="store_true", dest="qrcode_monster",
                        help="ControlNet QR Code Monster v2")
    parser.add_argument("--animatediff", action="store_true", help="AnimateDiff motion adapter")
    parser.add_argument("--animatediff-lightning", action="store_true", dest="animatediff_lightning",
                        help="AnimateDiff-Lightning checkpoints (2/4/8 step)")
    args = parser.parse_args()

    if not any([args.all, args.checkpoint, args.hyper_sd, args.loras, args.embeddings,
                args.controlnets, args.qrcode_monster, args.animatediff, args.animatediff_lightning]):
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

    print("\n    [OK] All downloads complete.")


if __name__ == "__main__":
    main()
