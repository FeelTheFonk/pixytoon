"""Build Aseprite extension package (.aseprite-extension = ZIP)."""

from __future__ import annotations

import zipfile
from pathlib import Path


def build() -> None:
    ext_dir = Path(__file__).resolve().parent.parent / "extension"
    out = Path(__file__).resolve().parent.parent / "dist"
    out.mkdir(exist_ok=True)
    target = out / "sddj.aseprite-extension"

    _exclude = {".DS_Store", "Thumbs.db", ".gitkeep"}
    _exclude_dirs = {"__pycache__", ".git", "node_modules"}

    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in ext_dir.rglob("*"):
            if item.is_file() and item.name not in _exclude:
                if not any(p in _exclude_dirs for p in item.relative_to(ext_dir).parts):
                    arcname = item.relative_to(ext_dir)
                    zf.write(item, arcname)
                    print(f"  + {arcname}")

    print(f"\n[OK] Built: {target}")
    print(f"  Size: {target.stat().st_size / 1024:.1f} KB")
    print(f"  Install: double-click the file in Aseprite")


if __name__ == "__main__":
    build()
