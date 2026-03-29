"""Standalone test: /health HTTP endpoint validation.

Usage:
    cd C:\\Users\\CleS\\Desktop\\SDDj\\server
    uv run python ..\\scripts\\test_health.py

Requires the SDDj server to be running on http://127.0.0.1:9876.
"""

from __future__ import annotations

import json
import sys
import urllib.request
import urllib.error


HEALTH_URL = "http://127.0.0.1:9876/health"


def main() -> bool:
    print(f"[TEST] Checking {HEALTH_URL}...")
    try:
        req = urllib.request.Request(HEALTH_URL, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                print(f"[FAIL] HTTP {resp.status}")
                return False
            data = json.loads(resp.read())
    except urllib.error.URLError as e:
        print(f"[FAIL] Cannot reach server: {e}")
        print("       Make sure the server is running (start.ps1 or uv run python run.py)")
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error: {e}")
        return False

    # Validate response structure
    ok = True

    if data.get("status") != "ok":
        print(f"[FAIL] Expected status='ok', got status='{data.get('status')}'")
        ok = False

    if "version" not in data:
        print("[FAIL] Missing 'version' in response")
        ok = False

    if "loaded" not in data:
        print("[FAIL] Missing 'loaded' in response")
        ok = False

    if ok:
        print(f"[OK]   /health — status={data['status']}, version={data['version']}, "
              f"loaded={data.get('loaded')}")
        vram = data.get("vram_free_mb")
        if vram is not None:
            print(f"       VRAM: {data.get('vram_used_mb', '?')}MB used / "
                  f"{data.get('vram_total_mb', '?')}MB total "
                  f"({vram}MB free)")

    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
