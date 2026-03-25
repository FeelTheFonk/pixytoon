"""Download OBP CSVs + merge with existing hand-written data → deduplicated JSON files.

This script:
1. Downloads relevant CSVs from OneButtonPrompt GitHub
2. Cleans OBP-specific template variables (-material-, OR(...), etc.)
3. Loads existing SDDj JSON files to avoid duplication
4. Merges OBP data with hand-written entries
5. Deduplicates (case-insensitive)
6. Writes final JSON files
"""

import json
import re
import urllib.request
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent.parent / "data" / "prompts"
OBP_RAW = "https://raw.githubusercontent.com/AIrjen/OneButtonPrompt/main/csvfiles"

# OBP CSV → our JSON mapping
SOURCES = {
    "pose.json": [f"{OBP_RAW}/poses.csv"],
    "outfit.json": [f"{OBP_RAW}/outfits.csv"],
    "accessory.json": [f"{OBP_RAW}/accessories.csv"],
    "material.json": [f"{OBP_RAW}/materials.csv"],
    "background.json": [f"{OBP_RAW}/backgrounds.csv"],
    "descriptor.json": [f"{OBP_RAW}/descriptors.csv"],
}

# OBP template variable patterns to filter out
OBP_TEMPLATE_RE = re.compile(r"-\w+-")  # matches -material-, -color-, etc.
OBP_OR_RE = re.compile(r"OR\([^)]+\)")  # matches OR(Cat;Bunny;Fox)
OBP_GENDER_RE = re.compile(r"\?(male|female|both)$")  # matches ?female suffix


def download_csv(url: str) -> list[str]:
    """Download a CSV and return cleaned lines."""
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        return lines
    except Exception as e:
        print(f"  WARN: Failed to download {url}: {e}")
        return []


def clean_entry(entry: str) -> str | None:
    """Clean an OBP CSV entry. Returns None if unusable."""
    # Remove gender suffix (?female, ?both, ?male)
    entry = OBP_GENDER_RE.sub("", entry).strip()
    
    # Skip entries with OBP template variables
    if OBP_TEMPLATE_RE.search(entry):
        return None
    
    # Skip entries with OR() constructs
    if OBP_OR_RE.search(entry):
        return None
    
    # Skip empty or too-short entries
    if len(entry) < 2:
        return None
    
    # Skip entries that are purely numeric or era-like with apostrophes only
    if re.match(r"^\d{4}'?S?$", entry, re.IGNORECASE):
        return None
    
    # Normalize: lowercase first char, strip trailing punctuation
    entry = entry.strip().rstrip(",;.")
    
    return entry if entry else None


def load_existing_json(path: Path) -> list[str]:
    """Load existing JSON items if file exists."""
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("items", [])
    except Exception:
        return []


def load_all_existing_items() -> set[str]:
    """Load ALL items from ALL existing JSON files to prevent cross-file duplication."""
    all_items: set[str] = set()
    for jf in PROMPTS_DIR.glob("*.json"):
        if jf.name in ("templates.json", "negatives.json"):
            continue
        items = load_existing_json(jf)
        for item in items:
            all_items.add(item.lower().strip())
    return all_items


def deduplicate(items: list[str], existing_lower: set[str]) -> list[str]:
    """Deduplicate items case-insensitively, also against existing data."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.lower().strip()
        if key not in seen and key not in existing_lower:
            seen.add(key)
            result.append(item)
    return result


def write_json(path: Path, items: list[str]) -> None:
    """Write items to JSON in our standard format."""
    items_sorted = sorted(items, key=str.lower)
    data = {"items": items_sorted}
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"  → Wrote {len(items_sorted)} items to {path.name}")


def main():
    print("Loading all existing SDDj data for cross-file dedup...")
    existing_global = load_all_existing_items()
    print(f"  Found {len(existing_global)} existing items across all JSON files\n")

    for target_name, urls in SOURCES.items():
        target_path = PROMPTS_DIR / target_name
        print(f"Processing {target_name}:")
        
        # Load existing hand-written data for this file
        existing_items = load_existing_json(target_path)
        print(f"  Existing hand-written entries: {len(existing_items)}")
        
        # Download and clean OBP data
        obp_items: list[str] = []
        for url in urls:
            csv_name = url.split("/")[-1]
            print(f"  Downloading {csv_name}...")
            raw_lines = download_csv(url)
            print(f"    Raw lines: {len(raw_lines)}")
            
            cleaned = 0
            skipped = 0
            for line in raw_lines:
                result = clean_entry(line)
                if result:
                    obp_items.append(result)
                    cleaned += 1
                else:
                    skipped += 1
            print(f"    Cleaned: {cleaned}, Skipped (template vars/too short): {skipped}")
        
        # Merge: existing first, then OBP additions
        all_items = existing_items + obp_items
        
        # Build dedup set excluding this file's own items from global check
        # (we want cross-file dedup but not self-dedup against existing)
        cross_file_existing = existing_global - {i.lower().strip() for i in existing_items}
        
        # Deduplicate
        final_items = deduplicate(all_items, cross_file_existing)
        print(f"  After dedup: {len(final_items)} (from {len(all_items)} merged)")
        
        write_json(target_path, final_items)
        print()

    print("Done! All JSON files updated with OBP data, zero duplicates.")


if __name__ == "__main__":
    main()
