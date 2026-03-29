"""Build Aseprite extension package (.aseprite-extension = ZIP)."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path


def build() -> None:
    root = Path(__file__).resolve().parent.parent
    ext_dir = root / "extension"
    out = root / "dist"

    if not ext_dir.is_dir():
        print(f"[FAIL] Extension source not found: {ext_dir}", file=sys.stderr)
        sys.exit(1)

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

    # Validate non-empty ZIP
    with zipfile.ZipFile(target, "r") as zf:
        if not zf.namelist():
            print("[FAIL] Extension ZIP is empty — no files matched", file=sys.stderr)
            sys.exit(1)

    print(f"\n[OK] Built: {target}")
    print(f"  Size: {target.stat().st_size / 1024:.1f} KB")
    print(f"  Install: double-click the file in Aseprite")


if __name__ == "__main__":
    build()
