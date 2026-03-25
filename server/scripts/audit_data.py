"""Exhaustive audit script for prompt data files and generator."""

import json
import sys
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent.parent / "data" / "prompts"
errors = []
warnings = []
stats = {}

def check(cond, msg, critical=True):
    if not cond:
        (errors if critical else warnings).append(msg)

# ── 1. Validate all JSON files ──
print("=== 1. JSON FILE VALIDATION ===")
for f in sorted(PROMPTS_DIR.glob("*.json")):
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        
        if f.stem in ("templates", "negatives"):
            if f.stem == "templates":
                templates = data.get("templates", {})
                stats[f.stem] = len(templates)
                print(f"  {f.name}: {len(templates)} templates")
                # Check all template placeholders
                import re
                for name, tmpl in templates.items():
                    placeholders = re.findall(r"\{(\w+)\}", tmpl)
                    for ph in placeholders:
                        if ph not in ("quality","subject","style","artist","lighting",
                                      "mood","camera","colors","details",
                                      "pose","outfit","accessory","material",
                                      "background","descriptor"):
                            warnings.append(f"Template '{name}' has unknown placeholder '{{{ph}}}'")
            else:
                sets = data.get("sets", {})
                stats[f.stem] = len(sets)
                print(f"  {f.name}: {len(sets)} negative sets")
        else:
            items = data.get("items", [])
            stats[f.stem] = len(items)
            print(f"  {f.name}: {len(items)} items, {f.stat().st_size/1024:.1f}KB")
            
            # Check for duplicates within file
            lower_items = [i.lower().strip() for i in items]
            dupes = len(lower_items) - len(set(lower_items))
            check(dupes == 0, f"{f.name}: {dupes} internal duplicates", critical=False)
            
            # Check for empty items
            empties = sum(1 for i in items if not i.strip())
            check(empties == 0, f"{f.name}: {empties} empty items")
            
            # Subject-specific checks
            if f.stem == "subject":
                typed = data.get("typed", {})
                typed_total = sum(len(v) for v in typed.values())
                check(typed_total == len(items), 
                      f"subject.json: typed total ({typed_total}) != items ({len(items)})")
                # Check no subject appears in multiple types
                all_typed = []
                for v in typed.values():
                    all_typed.extend(v)
                check(len(all_typed) == len(set(all_typed)),
                      f"subject.json: {len(all_typed)-len(set(all_typed))} subjects in multiple types")
                # Check every typed subject exists in items
                items_set = set(items)
                for stype, sitems in typed.items():
                    for s in sitems:
                        check(s in items_set, 
                              f"subject.json: typed '{stype}' entry not in items: '{s[:50]}'")
            
            # Artist-specific checks
            if f.stem == "artist":
                tagged = data.get("tagged", {})
                print(f"    tagged: {len(tagged)} artists")
                # Check all tagged artists exist in items
                items_set = set(items)
                for artist in tagged:
                    check(artist in items_set,
                          f"artist.json: tagged artist not in items: '{artist}'")
                    
    except json.JSONDecodeError as e:
        errors.append(f"{f.name}: INVALID JSON - {e}")
    except Exception as e:
        errors.append(f"{f.name}: ERROR - {e}")

# ── 2. Cross-file duplicate check ──
print("\n=== 2. CROSS-FILE DUPLICATE CHECK ===")
all_by_file = {}
for f in sorted(PROMPTS_DIR.glob("*.json")):
    if f.stem in ("templates", "negatives"):
        continue
    data = json.loads(f.read_text(encoding="utf-8"))
    items = data.get("items", [])
    all_by_file[f.stem] = {i.lower().strip() for i in items}

checked = set()
for f1, s1 in all_by_file.items():
    for f2, s2 in all_by_file.items():
        if f1 >= f2:
            continue
        pair = (f1, f2)
        if pair in checked:
            continue
        checked.add(pair)
        overlap = s1 & s2
        if overlap:
            # Some overlap is acceptable between certain pairs
            if not ({f1, f2} & {"subject"}):  # subject can overlap with specialized categories
                warnings.append(f"Cross-file overlap {f1}↔{f2}: {len(overlap)} items (e.g. '{list(overlap)[:3]}')")
            print(f"  {f1} ↔ {f2}: {len(overlap)} overlapping items")
        else:
            print(f"  {f1} ↔ {f2}: clean")

# ── 3. Expected files present ──
print("\n=== 3. EXPECTED FILES ===")
expected = ["subject","artist","style","quality","lighting","mood","camera","colors","details",
            "pose","outfit","accessory","material","background","descriptor",
            "templates","negatives"]
for name in expected:
    path = PROMPTS_DIR / f"{name}.json"
    check(path.exists(), f"Missing expected file: {name}.json")
    if path.exists():
        print(f"  ✓ {name}.json")
    else:
        print(f"  ✗ {name}.json MISSING")

# ── 4. Stats summary ──
print("\n=== 4. STATS ===")
total_items = sum(v for k,v in stats.items() if k not in ("templates","negatives"))
print(f"  Total data items: {total_items}")
print(f"  Templates: {stats.get('templates', 0)}")
print(f"  Negative sets: {stats.get('negatives', 0)}")
for k, v in sorted(stats.items()):
    if k not in ("templates", "negatives"):
        print(f"  {k}: {v}")

# ── 5. Results ──
print("\n=== RESULTS ===")
print(f"  Errors: {len(errors)}")
for e in errors:
    print(f"    ✗ {e}")
print(f"  Warnings: {len(warnings)}")
for w in warnings:
    print(f"    ⚠ {w}")

if errors:
    print("\n❌ AUDIT FAILED")
    sys.exit(1)
else:
    print(f"\n✅ AUDIT PASSED ({len(warnings)} warnings)")
