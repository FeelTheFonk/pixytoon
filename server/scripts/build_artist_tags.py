"""Build artist tag system from OBP's artists_and_category.csv.

OBP CSV format (proper CSV with header):
  Artist,Tags,Medium,Description,popular,greg mode,3D,abstract,...
  "Alvar Aalto","architecture, high contrast",Architecture,...
"""

import csv
import io
import json
import urllib.request
from pathlib import Path

ARTIST_PATH = Path(__file__).parent.parent / "data" / "prompts" / "artist.json"
OBP_URL = "https://raw.githubusercontent.com/AIrjen/OneButtonPrompt/main/csvfiles/artists_and_category.csv"


def download(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def main():
    with open(ARTIST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    our_artists = data["items"]
    print(f"Our artists: {len(our_artists)}")
    
    # Build lookup: lowercase "artist name" (without "by ") → original "by Artist Name"
    our_lookup: dict[str, str] = {}
    for a in our_artists:
        clean = a.lower().replace("by ", "", 1).strip()
        our_lookup[clean] = a
    
    # Download and parse OBP artist categories
    print("Downloading OBP artists_and_category.csv...")
    raw = download(OBP_URL)
    
    reader = csv.reader(io.StringIO(raw))
    header = next(reader)  # Skip header
    print(f"  Header columns: {len(header)}")
    print(f"  First few: {header[:5]}")
    
    tagged: dict[str, list[str]] = {}
    matched = 0
    total = 0
    
    # Popular artists (top SD1.5 community)
    popular_set = {
        "greg rutkowski", "artgerm", "alphonse mucha", "ross tran",
        "wlop", "sakimichan", "peter mohrbacher", "ilya kuvshinov",
        "krenz cushart", "ruan jia", "john singer sargent",
        "beeple", "james jean", "makoto shinkai", "hayao miyazaki",
        "frank frazetta", "boris vallejo", "yoshitaka amano",
        "moebius", "h.r. giger", "zdzislaw beksinski",
        "ivan aivazovsky", "caspar david friedrich", "thomas cole",
        "ansel adams", "albert bierstadt", "claude monet",
    }
    
    for row in reader:
        if len(row) < 2:
            continue
        total += 1
        
        obp_name = row[0].strip()
        tags_str = row[1].strip() if len(row) > 1 else ""
        
        # Parse tags from the Tags column
        tags = [t.strip().lower() for t in tags_str.split(",") if t.strip()]
        
        # Match against our list
        obp_clean = obp_name.lower().strip()
        if obp_clean in our_lookup:
            our_name = our_lookup[obp_clean]
            if obp_clean in popular_set:
                tags = list(set(tags + ["popular"]))
            tagged[our_name] = sorted(set(t for t in tags if t))
            matched += 1
    
    # Add "popular" tag to known popular artists not matched in OBP
    for name in popular_set:
        full = f"by {name}"
        for our_name in our_artists:
            if our_name.lower() == full.lower():
                if our_name not in tagged:
                    tagged[our_name] = ["popular"]
                elif "popular" not in tagged[our_name]:
                    tagged[our_name].append("popular")
                    tagged[our_name].sort()
                break
    
    print(f"  OBP total: {total}, Matched to our list: {matched}")
    print(f"  Tagged artists: {len(tagged)}")
    
    # Count categories
    all_cats: dict[str, int] = {}
    for tags in tagged.values():
        for t in tags:
            all_cats[t] = all_cats.get(t, 0) + 1
    
    print(f"  Unique categories: {len(all_cats)}")
    top15 = sorted(all_cats.items(), key=lambda x: x[1], reverse=True)[:15]
    for cat, count in top15:
        print(f"    {cat}: {count}")
    
    # Write back
    data["tagged"] = tagged
    
    with open(ARTIST_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    
    print(f"\nWrote tagged metadata to {ARTIST_PATH.name}")


if __name__ == "__main__":
    main()
