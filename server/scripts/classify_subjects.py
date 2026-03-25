"""Add subject type classification to subject.json using keyword heuristics.

Classifies each subject entry into: humanoid, animal, landscape, object, concept.
Adds a "typed" dict alongside existing "items" for backward compatibility.
"""

import json
import re
from pathlib import Path

SUBJECT_PATH = Path(__file__).parent.parent / "data" / "prompts" / "subject.json"

# Keyword patterns for classification (order matters: first match wins)
HUMANOID_PATTERNS = [
    r"\b(warrior|knight|princess|prince|queen|king|wizard|witch|mage|sorcerer|sorceress)\b",
    r"\b(woman|man|girl|boy|lady|lord|maiden|monk|nun|priest|priestess)\b",
    r"\b(hero|archer|assassin|bard|druid|ranger|rogue|thief|paladin|cleric)\b",
    r"\b(soldier|captain|admiral|general|commander|pirate|viking|gladiator)\b",
    r"\b(dancer|musician|painter|artist|sculptor|cook|baker|chef)\b",
    r"\b(blacksmith|alchemist|merchant|vendor|trader|shopkeeper)\b",
    r"\b(child|elder|grandmother|grandfather|old man|old woman)\b",
    r"\b(angel|demon|devil|god|goddess|titan|vampire|zombie|werewolf)\b",
    r"\b(samurai|ronin|ninja|geisha|shogun)\b",
    r"\b(explorer|traveler|nomad|wanderer|pilgrim)\b",
    r"\b(fisherman|fishmonger|farmer|shepherd|hunter|trapper)\b",
    r"\b(conductor|worker|craftsman|artisan|toymaker|clockmaker|watchmaker)\b",
    r"\b(astronaut|diver|climber|rider)\b",
    r"\b(couple|lovers|dancing|person|figure|portrait|character|humanoid|jester|fool)\b",
    r"\b(golem|colossus|automaton|scarecrow|puppet)\b",  # humanoid-shaped constructs
    r"\b(cyborg|android|robot\b.*(human|person))\b",
    r"\b(troll|ogre|dwarf|elf|fairy|pixie|gnome|hobbit)\b",
    r"\b(sphinx|centaur|minotaur|cyclops|harpy|mermaid|siren|nymph|satyr)\b",
    r"\b(necromancer|warlock|shaman|oracle|seer|prophet)\b",
    r"\b(dervish|griot|nonna|keeper|carver|dyer|weaver|player|window washer)\b",
]

ANIMAL_PATTERNS = [
    r"\b(wolf|fox|bear|deer|stag|horse|cat|dog|lion|tiger|eagle|hawk|falcon|owl)\b",
    r"\b(dragon|phoenix|griffin|wyvern|serpent|basilisk|hydra|kraken|leviathan)\b",
    r"\b(whale|shark|fish|octopus|jellyfish|seahorse|turtle|tortoise)\b",
    r"\b(butterfly|moth|bee|spider|insect|beetle|dragonfly|firefly)\b",
    r"\b(raven|crow|swan|heron|crane|peacock|parrot|flamingo)\b",
    r"\b(unicorn|cerberus|behemoth|chimera|manticore)\b",
    r"\b(snake|cobra|viper|lizard|chameleon|gecko|iguana|salamander)\b",
    r"\b(frog|toad|newt|axolotl)\b",
    r"\b(yeti|bigfoot|wendigo)\b",
    r"\b(armadillo|pangolin|hedgehog|porcupine|badger|otter|ferret|rabbit|hare)\b",
    r"\b(elephant|rhino|hippo|giraffe|zebra|gorilla|chimpanzee|orangutan)\b",
    r"\b(bat|mice|mouse|rat|squirrel|chipmunk)\b",
    r"\b(moth|anglerfish|crab|lobster|shrimp|starfish|coral)\b",
    r"\b(treant|elemental.*animal|spirit.*animal|familiar)\b",
]

LANDSCAPE_PATTERNS = [
    r"\b(landscape|vista|panorama|scenery|horizon)\b",
    r"\b(forest|jungle|woodland|grove|thicket|clearing)\b",
    r"\b(mountain|peak|summit|cliff|canyon|ravine|gorge|valley)\b",
    r"\b(ocean|sea|coast|beach|shore|island|archipelago|reef|lagoon)\b",
    r"\b(desert|dune|oasis|wasteland|badlands|tundra|steppe|savanna)\b",
    r"\b(city|town|village|metropolis|cityscape|skyline|alley|street|plaza)\b",
    r"\b(castle|fortress|palace|cathedral|temple|ruins|tower)\b",
    r"\b(cave|cavern|grotto|underground|subterranean|tunnel)\b",
    r"\b(garden|meadow|field|farmland|vineyard|orchard)\b",
    r"\b(river|lake|waterfall|stream|pond|swamp|marsh|bog)\b",
    r"\b(volcano|crater|geyser|hot spring)\b",
    r"\b(aurora|sunset|sunrise|moonlit|starry|eclipse)\b",
    r"\b(village built|treehouse village|floating island|sky.*castle)\b",
]

OBJECT_PATTERNS = [
    r"\b(sword|dagger|axe|hammer|bow|shield|spear|lance|staff|wand)\b",
    r"\b(book|tome|scroll|map|letter|manuscript)\b",
    r"\b(clock|watch|hourglass|compass|astrolabe|orrery|armillary)\b",
    r"\b(potion|flask|bottle|vial|cauldron|chalice|goblet|grail)\b",
    r"\b(gem|crystal|diamond|ruby|emerald|sapphire|pearl|amber)\b",
    r"\b(crown|throne|scepter|medallion|amulet|ring|necklace)\b",
    r"\b(ship|boat|vessel|gondola|canoe|raft)\b",
    r"\b(telescope|microscope|lantern|lamp|candle|torch|chandelier)\b",
    r"\b(violin|piano|guitar|flute|drum|harp|lute|organ)\b",
    r"\b(painting|statue|sculpture|tapestry|mosaic|fresco)\b",
    r"\b(motorcycle|locomotive|car|wagon|carriage|cart)\b",
    r"\b(still life|bento|charcuterie|food.*table|meal)\b",
]

CONCEPT_PATTERNS = [
    r"\bthe (duality|passage|weight|moment|four seasons|sun surface)\b",
    r"\b(allegory|personified|duality|paradox|metamorphosis)\b",
    r"\b(abstract|geometric|patterns|fractal|mandala|kaleidoscope)\b",
    r"\b(infinity|reflected|reality dissolving|gravity reversing)\b",
    r"\b(visualized|magnified|viewed at|quantum|photosynthesis)\b",
    r"\b(neurons|atoms|blood cells|magnetic field|sound waves)\b",
    r"\b(creation and|light and shadow|sleeping and waking)\b",
    r"\b(tears crystallizing|thoughts solidifying|ocean waves frozen)\b",
    r"\b(time zones|snowflakes magnified|tree rings telling)\b",
]


def classify_subject(text: str) -> str:
    """Classify a subject into one of 5 types."""
    lower = text.lower()
    
    # Check concept first (these are abstract / non-representational)
    for pat in CONCEPT_PATTERNS:
        if re.search(pat, lower):
            return "concept"
    
    # Check humanoid (most entries feature human-like characters)
    for pat in HUMANOID_PATTERNS:
        if re.search(pat, lower):
            return "humanoid"
    
    # Check animal
    for pat in ANIMAL_PATTERNS:
        if re.search(pat, lower):
            return "animal"
    
    # Check landscape
    for pat in LANDSCAPE_PATTERNS:
        if re.search(pat, lower):
            return "landscape"
    
    # Check object
    for pat in OBJECT_PATTERNS:
        if re.search(pat, lower):
            return "object"
    
    # Default: concept (abstract/uncategorized)
    return "concept"


def main():
    with open(SUBJECT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    items = data["items"]
    print(f"Classifying {len(items)} subjects...")
    
    typed: dict[str, list[str]] = {
        "humanoid": [],
        "animal": [],
        "landscape": [],
        "object": [],
        "concept": [],
    }
    
    for item in items:
        category = classify_subject(item)
        typed[category].append(item)
    
    # Sort each type
    for key in typed:
        typed[key].sort(key=str.lower)
    
    # Report
    for key, vals in typed.items():
        print(f"  {key}: {len(vals)}")
    
    # Write back: keep items intact, add typed
    data["typed"] = typed
    
    with open(SUBJECT_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    
    print(f"\nWrote typed metadata to {SUBJECT_PATH.name}")


if __name__ == "__main__":
    main()
