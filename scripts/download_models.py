"""Download all required models: SD1.5 checkpoint, Hyper-SD LoRA, pixel art LoRAs, ControlNets, AnimateDiff.

Usage:
    python scripts/download_models.py [--all] [--checkpoint] [--hyper-sd] [--loras] [--controlnets] [--animatediff]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def download_checkpoint() -> None:
    from huggingface_hub import snapshot_download
    print("[1/6] Downloading SD1.5 checkpoint: Lykon/dreamshaper-8 ...")
    # Pre-cache in HF cache — engine loads by repo ID at startup
    snapshot_download(
        "Lykon/dreamshaper-8",
        ignore_patterns=["*.ckpt", "*.bin"],  # Prefer safetensors
    )
    print("  [OK] Checkpoint cached.")


def download_hyper_sd_lora() -> None:
    from huggingface_hub import hf_hub_download
    print("[2/6] Downloading Hyper-SD LoRA: ByteDance/Hyper-SD ...")
    hf_hub_download("ByteDance/Hyper-SD", filename="Hyper-SD15-8steps-CFG-lora.safetensors")
    print("  [OK] Hyper-SD LoRA cached.")


def download_pixel_loras() -> None:
    from huggingface_hub import hf_hub_download
    print("[3/6] Downloading pixel art LoRAs ...")
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
            print(f"  [SKIP] {lora['local']} already exists, skipping.")
            continue
        print(f"  [DL] {lora['repo']} -> {lora['local']}")
        downloaded = hf_hub_download(lora["repo"], filename=lora["file"])
        shutil.copy2(downloaded, dest)
        print(f"  [OK] {lora['local']}")

    print("  [OK] Pixel art LoRAs downloaded.")


def download_embeddings() -> None:
    from huggingface_hub import hf_hub_download
    print("[4/6] Downloading negative TI embeddings ...")
    embeddings_dir = _PROJECT_ROOT / "server" / "models" / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    embeddings = [
        {
            "repo": "gsdf/EasyNegative",
            "file": "EasyNegative.safetensors",
            "local": "EasyNegative.safetensors",
            "repo_type": "dataset",
        },
        {
            "repo": "navmesh/Embeddings",
            "file": "FastNegativeV2.pt",
            "local": "FastNegativeV2.pt",
            "repo_type": "model",
        },
    ]

    for emb in embeddings:
        dest = embeddings_dir / emb["local"]
        if dest.exists():
            print(f"  [SKIP] {emb['local']} already exists, skipping.")
            continue
        print(f"  [DL] {emb['repo']} -> {emb['local']}")
        downloaded = hf_hub_download(
            emb["repo"],
            filename=emb["file"],
            repo_type=emb["repo_type"],
        )
        shutil.copy2(downloaded, dest)
        print(f"  [OK] {emb['local']}")

    print("  [OK] TI embeddings downloaded.")


def download_controlnets() -> None:
    from huggingface_hub import snapshot_download
    print("[5/6] Downloading ControlNet v1.1 models ...")
    models = {
        "openpose": "lllyasviel/control_v11p_sd15_openpose",
        "canny": "lllyasviel/control_v11p_sd15_canny",
        "scribble": "lllyasviel/control_v11p_sd15_scribble",
        "lineart": "lllyasviel/control_v11p_sd15_lineart",
    }
    for name, repo in models.items():
        print(f"  [DL] {repo} ...")
        # Pre-cache in HF cache — engine loads by ID
        snapshot_download(repo, ignore_patterns=["*.bin"])
        print(f"  [OK] {name}")
    print("  [OK] ControlNet models cached.")


def download_animatediff() -> None:
    from huggingface_hub import snapshot_download
    print("[6/6] Downloading AnimateDiff motion adapter ...")
    snapshot_download(
        "guoyww/animatediff-motion-adapter-v1-5-3",
        ignore_patterns=["*.bin"],
    )
    print("  [OK] AnimateDiff motion adapter cached.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PixyToon models")
    parser.add_argument("--all", action="store_true", help="Download everything")
    parser.add_argument("--checkpoint", action="store_true", help="SD1.5 base checkpoint")
    parser.add_argument("--hyper-sd", action="store_true", dest="hyper_sd", help="Hyper-SD acceleration LoRA")
    parser.add_argument("--loras", action="store_true", help="Pixel art LoRAs")
    parser.add_argument("--embeddings", action="store_true", help="Negative TI embeddings")
    parser.add_argument("--controlnets", action="store_true", help="ControlNet v1.1 models")
    parser.add_argument("--animatediff", action="store_true", help="AnimateDiff motion adapter")
    args = parser.parse_args()

    if not any([args.all, args.checkpoint, args.hyper_sd, args.loras, args.embeddings, args.controlnets, args.animatediff]):
        args.all = True

    if args.all or args.checkpoint:
        download_checkpoint()
    if args.all or args.hyper_sd:
        download_hyper_sd_lora()
    if args.all or args.loras:
        download_pixel_loras()
    if args.all or args.embeddings:
        download_embeddings()
    if args.all or args.controlnets:
        download_controlnets()
    if args.all or args.animatediff:
        download_animatediff()

    print("\n[OK] All downloads complete.")


if __name__ == "__main__":
    main()
