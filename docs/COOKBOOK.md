# PixyToon Cookbook

> Tested recipes by creative intention. No theory — just settings that work.

**[README](../README.md)** | **[Guide](GUIDE.md)** | **[Cookbook](COOKBOOK.md)** | **[Live Paint](LIVE-PAINT.md)** | **[Audio Reactivity](AUDIO-REACTIVITY.md)** | **[API Reference](API-REFERENCE.md)** | **[Configuration](CONFIGURATION.md)** | **[Troubleshooting](TROUBLESHOOTING.md)**

---

## Table of Contents

- [How to Read a Recipe](#how-to-read-a-recipe)
- [Characters](#characters)
- [Environments](#environments)
- [Items and Icons](#items-and-icons)
- [Portraits](#portraits)
- [Inpaint](#inpaint)
- [Animation](#animation)
- [Output Mode: Sequence](#output-mode-sequence)
- [Loop and Random Loop](#loop-and-random-loop)
- [Live Paint Recipes](#live-paint-recipes)
- [ControlNet Recipes](#controlnet-recipes)
- [Palette Craft](#palette-craft)
- [Parameter Matrix](#parameter-matrix)
- [Presets](#presets)
- [Anti-Patterns](#anti-patterns)

---

## How to Read a Recipe

Each recipe shows:

- **Prompt** — what to type
- **Settings** — the key parameters to change (everything not listed stays at default)
- **Post-Process** — pixel art pipeline settings
- **Why** — what makes this combination work

Default settings (if not overridden in a recipe): steps=8, CFG=5.0, clip_skip=2, denoise=1.0, pixelate=128, colors=32, quantize=kmeans, dither=none, palette=auto.

---

## Characters

### RPG Hero Sprite

A classic top-down or side-view character sprite for a 2D RPG.

**Prompt:**
```
pixel art, PixArFK, RPG hero character, warrior, sword, cape,
fantasy armor, game sprite, front view, sharp pixels, clean outline
```

**Settings:**

| Parameter | Value | Why |
|-----------|-------|-----|
| Mode | txt2img | Starting from scratch |
| Size | 512x512 | Standard SD 1.5 resolution |
| Steps | 8 | Hyper-SD default is enough |
| CFG | 5.0 | Balanced prompt following |

**Post-Process:**

| Parameter | Value | Why |
|-----------|-------|-----|
| Pixelate | 64px | Classic RPG sprite size |
| Colors | 16-24 | Tight palette, clean reads |
| Quantize | kmeans | Best color grouping |
| Remove BG | Yes | Transparent background for sprites |

---

### Chibi / Small Character

Tiny characters for overworld maps or UI elements.

**Prompt:**
```
pixel art, chibi character, simple, cute, game sprite,
front view, sharp pixels, clean outline, tiny
```

**Settings:**

| Parameter | Value | Why |
|-----------|-------|-----|
| Size | 256x256 | Smaller canvas = less detail to manage |
| CFG | 6.0 | Slightly stricter for small sprites |

**Post-Process:**

| Parameter | Value | Why |
|-----------|-------|-----|
| Pixelate | 32px | Tiny target |
| Colors | 8-12 | Minimal palette for small sprites |

---

### Enemy / Monster

Creatures with more freedom in shape and color.

**Prompt:**
```
pixel art, monster, slime creature, green, game enemy,
side view, pixel art style, sharp pixels, flat colors
```

**Settings:**

| Parameter | Value | Why |
|-----------|-------|-----|
| Seed | Fix one you like | Iterate on the same creature |
| CFG | 4.0-5.0 | Slightly loose = more creative shapes |

**Post-Process:**

| Parameter | Value | Why |
|-----------|-------|-----|
| Pixelate | 48-64px | Room for detail in the shape |
| Colors | 12-20 | Enough for shading |
| Remove BG | Yes | Clean extraction |

---

### NPC Variations

Generate multiple similar-but-different NPCs from the same base.

**Technique:** Use a fixed seed, then change one detail in the prompt each time.

```
Seed 42: pixel art, townsperson, blue shirt, brown hair, game sprite
Seed 42: pixel art, townsperson, red shirt, brown hair, game sprite
Seed 42: pixel art, townsperson, blue shirt, blonde hair, game sprite
Seed 43: pixel art, townsperson, blue shirt, brown hair, game sprite
```

The first three share the same composition but differ in details. The last one (seed+1) gives a related but different pose.

---

## Environments

### Side-Scroller Background

Parallax-ready background layers.

**Prompt:**
```
pixel art, side scrolling game background, fantasy forest,
trees, grass, path, vibrant colors, game scenery, flat colors
```

**Settings:**

| Parameter | Value | Why |
|-----------|-------|-----|
| Size | 768x512 | Wide aspect for side-scrollers |
| CFG | 4.0-5.0 | Less strict = more natural landscapes |

**Post-Process:**

| Parameter | Value | Why |
|-----------|-------|-----|
| Pixelate | 128-192px | More detail for backgrounds |
| Colors | 32-48 | Richer palette for scenery |
| Dither | bayer_4x4 | Smooth sky gradients without too many colors |

---

### Top-Down Tilemap Chunk

A section of a top-down game world.

**Prompt:**
```
pixel art, top down game tile, grass field, dirt path,
small rocks, game tilemap, orthographic, flat lighting
```

**Settings:**

| Parameter | Value | Why |
|-----------|-------|-----|
| Size | 512x512 | Square for tiles |
| CFG | 6.0 | Stricter to get consistent flat shapes |

**Post-Process:**

| Parameter | Value | Why |
|-----------|-------|-----|
| Pixelate | 64-128px | Crisp tile boundaries |
| Colors | 16-24 | Consistent palette across tiles |
| Palette | Preset or Custom | Use the same palette for all tiles in a set |

> [!TIP]
> Generate multiple tiles with the same palette preset to ensure they look coherent as a tileset.

---

### Interior / Room

Dungeon rooms, houses, shops.

**Prompt:**
```
pixel art, interior room, medieval tavern, wooden tables,
fireplace, warm lighting, top down view, game scene
```

**Settings:**

| Parameter | Value | Why |
|-----------|-------|-----|
| Size | 512x512 | Standard |
| CFG | 5.0-6.0 | Needs structure |

**Post-Process:**

| Parameter | Value | Why |
|-----------|-------|-----|
| Pixelate | 128px | Detail for furniture |
| Colors | 24-32 | Warm palette variety |
| Dither | floyd_steinberg | Smooth warm lighting transitions |

---

## Items and Icons

### Inventory Item

Small icons for RPG inventories.

**Prompt:**
```
pixel art, game item icon, magical sword, glowing blue,
simple background, centered, sharp pixels, clean outline
```

**Settings:**

| Parameter | Value | Why |
|-----------|-------|-----|
| Size | 256x256 | Small canvas for small items |
| CFG | 6.0-7.0 | Strict = clean, recognizable shape |

**Post-Process:**

| Parameter | Value | Why |
|-----------|-------|-----|
| Pixelate | 32px | Classic 32x32 icon |
| Colors | 8-12 | Very tight palette |
| Remove BG | Yes | Transparent for UI overlay |

---

### UI Element

Buttons, frames, health bars.

**Prompt:**
```
pixel art, game UI button, stone texture, rectangular,
beveled edge, fantasy style, clean flat design
```

**Settings:**

| Parameter | Value | Why |
|-----------|-------|-----|
| Size | 256x256 | |
| CFG | 7.0 | Very strict for geometric shapes |

**Post-Process:**

| Parameter | Value | Why |
|-----------|-------|-----|
| Pixelate | 48-64px | Readable at UI scale |
| Colors | 6-10 | Minimal |
| Palette | Custom | Match your game's UI palette |

---

## Portraits

### Character Portrait from Sketch

Transform a rough Aseprite sketch into a polished pixel art portrait.

**Prompt:**
```
pixel art, character portrait, fantasy elf, detailed face,
pointed ears, green eyes, pixel art style, sharp pixels
```

**Settings:**

| Parameter | Value | Why |
|-----------|-------|-----|
| Mode | img2img | Uses your sketch as base |
| Strength | 0.5-0.6 | Keeps your composition, adds detail |
| Size | 512x512 | Room for facial detail |

**Post-Process:**

| Parameter | Value | Why |
|-----------|-------|-----|
| Pixelate | 64-96px | Enough for expressive faces |
| Colors | 16-24 | Skin tones + details |

**Iteration workflow:**

1. First pass at strength 0.6 — get the overall look
2. If you like the direction, lower strength to 0.3-0.4
3. Make manual edits in Aseprite on the result
4. Run img2img again at 0.2-0.3 to polish

---

## Inpaint

### Fix a Face

Generated sprite has a bad face? Inpaint just that region.

1. Select the face area with the selection tool
2. Mode: **inpaint**

**Prompt:**
```
pixel art, detailed face, expressive eyes, game character face,
sharp pixels, clean
```

**Settings:**

| Parameter | Value | Why |
|-----------|-------|-----|
| Strength | 0.6-0.8 | Enough to regenerate the face, keep surroundings |
| CFG | 5.0-6.0 | Follow the prompt closely |

> [!TIP]
> Use a feathered selection for smoother blending at the inpaint boundary.

---

### Add Missing Detail

Sprite is missing a weapon, accessory, or background element.

1. Select the empty area where you want the new element
2. Mode: **inpaint**

**Prompt:**
```
pixel art, (element to add), game sprite, matching style,
sharp pixels, flat colors
```

**Settings:**

| Parameter | Value | Why |
|-----------|-------|-----|
| Strength | 0.9-1.0 | Full generation in the selected area |
| CFG | 5.0 | Balanced interpretation |

---

## Animation

### Walk Cycle (Chain)

4-8 frame walk animation using frame chaining.

**Prompt:**
```
pixel art, character walk cycle, side view, game sprite,
walking animation, sharp pixels, consistent style
```

**Settings:**

| Parameter | Value | Why |
|-----------|-------|-----|
| Method | chain | Frame-by-frame control |
| Frames | 4-8 | Standard walk cycle |
| Duration | 100-120ms | Natural walking speed |
| Strength | 0.25-0.35 | Low = consistent between frames |
| Seed Mode | increment | Slight variation per frame |

> [!TIP]
> Lower strength (0.20-0.30) = smoother animation with less frame-to-frame variation. Higher (0.40+) = more dynamic but less consistent.

---

### Idle Animation (AnimateDiff)

Subtle breathing/shifting idle animation.

**Prompt:**
```
pixel art, character idle animation, slight movement,
breathing, game sprite, pixel art style
```

**Settings:**

| Parameter | Value | Why |
|-----------|-------|-----|
| Method | animatediff | Temporal consistency for subtle motion |
| Frames | 8-16 | Loopable at these counts |
| Strength | 0.20-0.30 | Subtle changes only |
| FreeInit | On | Better temporal consistency |
| FreeInit Iters | 2 | Good balance |

---

### Effect Animation (Fire, Water, Sparkle)

Elemental effects with more motion freedom.

**Prompt:**
```
pixel art, fire animation, campfire, flickering flames,
orange yellow, game effect, pixel art style
```

**Settings:**

| Parameter | Value | Why |
|-----------|-------|-----|
| Method | animatediff | Good for fluid motion |
| Frames | 8-12 | Loopable |
| Strength | 0.35-0.50 | More variation for effects |
| Remove BG | Yes | Overlay on game scenes |

---

## Output Mode: Sequence

### Img2Img Iteration Timeline

Generate multiple img2img passes and review them as timeline frames.

1. Draw your base on a layer
2. Set mode to **img2img**, strength 0.3-0.5
3. Set **Output** to **sequence**
4. Enable **Loop Mode**, set Loop Seed to `increment`
5. Click **Generate** — each iteration becomes a new frame
6. Click **Cancel** when you have enough — scrub the timeline to pick the best

**Why sequence?** In "layer" mode, 20 generations stack as 20 layers on one frame — hard to compare. In "sequence" mode, they become 20 frames you can scrub through in Aseprite's timeline.

> [!TIP]
> Combine with different Loop Seed modes: `increment` for subtle variations of the same composition, `random` for diverse explorations.

---

### Txt2Img Reference Sheet

Generate many variations of a prompt and browse them as an animation.

1. Set mode to **txt2img**
2. Set **Output** to **sequence**
3. Enable **Loop Mode**, Loop Seed `random`
4. Click **Generate** — each result becomes a frame
5. Cancel after 10-20 frames, then scrub to find your favorite

---

## Loop and Random Loop

### Rapid Variation Exploration (Loop Mode)

Generate 20+ variations of the same prompt to find the perfect one.

1. Set your prompt and all generation settings
2. Check **Loop Mode**
3. Set **Loop Seed** to `random` (different each time) or `increment` (+1 each iteration)
4. Click **Generate** and watch variations flow

> [!TIP]
> Use `increment` seed mode to get subtle variations — each seed nearby produces similar but slightly different results.

> [!TIP]
> Set **Output** to `sequence` before starting a loop. Each generation becomes a timeline frame — much easier to compare than stacked layers. Cancel anytime; partial results are kept and finalized.

---

### Automated Creative Discovery (Random Loop)

Let the model surprise you with fully randomized prompts and images.

1. Check **Loop Mode** and **Random Loop**
2. Optionally check **Lock Subject** and enter a fixed subject (e.g., "warrior character")
3. Click **Generate**
4. Each iteration: new random prompt is generated, then the image is generated, repeat

**Example locked subject workflow:**

| Setting | Value |
|---------|-------|
| Lock Subject | On |
| Fixed Subject | "dragon" |
| Loop Seed | random |

Result: endless dragon variations — pixel art dragon, chibi dragon, dragon boss sprite, dragon idle animation... all automatically generated with diverse styles, moods, and compositions.

> [!TIP]
> Random Loop is perfect for mood boards and inspiration gathering. Let it run for 5 minutes, then review the results and pick your favorites.

---

## Live Paint Recipes

For detailed Live Paint technique explanations, see the [Live Paint guide](LIVE-PAINT.md).

### Quick Concept Sketch

Sketch rough shapes and let SD fill in details in real-time.

**Live Tab Settings:**

| Parameter | Value | Why |
|-----------|-------|-----|
| Strength | 0.50-0.60 | Strong SD interpretation |
| Steps | 4 | Real-time speed |
| CFG | 2.0-3.0 | Creative, loose |
| Preview Opacity | 70% | See both your sketch and SD output |

**Workflow:**
1. Start Live, draw broad shapes and colors
2. The model interprets them immediately
3. Adjust prompt to steer the direction
4. Lower strength to 0.30 to refine details
5. Accept when satisfied

---

### Style Exploration

Try different art directions on the same drawing.

**Live Tab Settings:**

| Parameter | Value | Why |
|-----------|-------|-----|
| Strength | 0.40-0.50 | Enough to see style changes |
| Steps | 4 | Fast feedback |
| CFG | 2.5 | Balanced |

**Technique:** Keep drawing the same thing, change the prompt:

```
pixel art, dark fantasy, grim, desaturated
pixel art, cute, colorful, chibi style
pixel art, sci-fi, neon lights, cyberpunk
```

The Live mode updates instantly when the prompt changes.

---

## ControlNet Recipes

### Line Art to Pixel Art (Canny)

Convert clean line drawings into pixel art.

1. Draw clean outlines on a layer in Aseprite
2. Select that layer as active
3. Mode: **controlnet_canny**

**Prompt:**
```
pixel art, colored game sprite, clean pixel art, sharp pixels,
flat colors, (matching your subject description)
```

**Settings:**

| Parameter | Value | Why |
|-----------|-------|-----|
| CFG | 5.0-6.0 | Follow the lines closely |
| Steps | 8 | Standard quality |

---

### Rough Sketch to Sprite (Scribble)

The most forgiving mode — rough drawings become detailed sprites.

1. Scribble a rough shape (stick figures work)
2. Mode: **controlnet_scribble**

**Prompt:**
```
pixel art, (your subject), game sprite, detailed,
sharp pixels, flat shading
```

**Settings:**

| Parameter | Value | Why |
|-----------|-------|-----|
| CFG | 5.0 | Balanced interpretation |
| Steps | 8 | Standard quality |

> [!TIP]
> Scribble mode is great for prototyping. Draw 10 stick figures, generate 10 character concepts in minutes.

---

### Pose to Character (OpenPose)

Draw a stick figure skeleton, get a posed character.

1. Draw a simple stick figure (head circle, body line, arm lines, leg lines)
2. Mode: **controlnet_openpose**

**Prompt:**
```
pixel art, (character description), (action/pose),
game sprite, sharp pixels
```

**Settings:**

| Parameter | Value | Why |
|-----------|-------|-----|
| CFG | 5.0-6.0 | Follow the pose closely |
| Steps | 8 | Standard quality |

**Post-Process:**

| Setting | Value |
|---------|-------|
| Pixelate | 64-128px |
| Colors | 16-24 |

---

### Coloring Line Art (Lineart)

You have clean line art, want pixel art coloring.

1. Draw your line art (or use existing)
2. Mode: **controlnet_lineart**

**Prompt:**
```
pixel art, colored version, vibrant colors, flat shading,
(subject description), sharp pixels
```

**Settings:**

| Parameter | Value | Why |
|-----------|-------|-----|
| CFG | 6.0 | Respect the line art structure |
| Steps | 8 | Standard quality |

---

## Palette Craft

### Using Built-in Palettes

Set **Palette** to **Preset** in the Post-Process tab. Available palettes:

| Palette | Colors | Character |
|---------|--------|-----------|
| **PICO-8** | 16 | The quintessential fantasy console palette — vibrant, limited, instantly recognizable |
| **Game Boy** | 4 | Green monochrome. Maximum constraint = maximum creativity |
| **NES** | 54 | Classic 8-bit console colors with that specific warmth |
| **SNES** | 256 | Full 16-bit range — more freedom while staying retro |
| **C64** | 16 | Commodore 64 — earthy, muted, nostalgic |
| **Endesga 32** | 32 | Modern pixel art standard — warm, expressive, versatile |
| **Endesga 64** | 64 | Extended Endesga for when 32 isn't enough |

> [!TIP]
> PICO-8 and Endesga 32 are the most versatile. Start with these if unsure.

### Custom Palette from Aseprite

If you have a carefully crafted palette in Aseprite:

1. Export your palette colors as hex codes
2. Set Palette to **Custom**
3. Paste hex codes: `#1a1c2c #5d275d #b13e53 #ef7d57 #ffcd75 ...`

### When Palette Enforcement Shifts Colors Too Much

Palette enforcement maps each pixel to the nearest perceptual match (CIELAB color space). If results feel "off":

- **Increase color count first** (quantize to 48+ colors), then enforce palette
- **Use dithering** — Floyd-Steinberg after enforcement smooths harsh transitions
- **Try a larger palette** — Endesga 64 or SNES give more room than PICO-8

---

## Parameter Matrix

Quick reference: recommended settings by creative intention.

| Intent | Steps | CFG | Strength | Pixelate | Colors | Quantize | Dither |
|--------|-------|-----|----------|----------|--------|----------|--------|
| **Character sprite** | 8 | 5.0 | 1.0 | 64 | 16-24 | kmeans | none |
| **Tiny icon** | 8 | 6.0-7.0 | 1.0 | 32 | 8-12 | kmeans | none |
| **Environment** | 8 | 4.0-5.0 | 1.0 | 128-192 | 32-48 | kmeans | bayer_4x4 |
| **Portrait** | 8 | 5.0 | 0.5-0.6 | 64-96 | 16-24 | kmeans | none |
| **Walk cycle** | 8 | 5.0 | 0.25-0.35 | 64 | 16-24 | kmeans | none |
| **Idle anim** | 8 | 5.0 | 0.20-0.30 | 64-128 | 16-32 | kmeans | none |
| **Effect anim** | 8 | 5.0 | 0.35-0.50 | 64 | 12-20 | kmeans | none |
| **Live concept** | 4 | 2.0-3.0 | 0.50-0.60 | 128 | 32 | kmeans | none |
| **Retro (GB)** | 8 | 5.0 | 1.0 | 32-64 | 4 | kmeans | bayer_2x2 |
| **Retro (NES)** | 8 | 5.0 | 1.0 | 64 | 12-16 | kmeans | none |
| **Retro (PICO-8)** | 8 | 5.0 | 1.0 | 64-128 | 16 | kmeans | none |
| **Hi-fi pixel art** | 10-12 | 5.0 | 1.0 | 192-256 | 48-64 | kmeans | floyd_steinberg |
| **Loop + sequence** | 8 | 5.0 | 0.3-0.5 | 64-128 | 16-32 | kmeans | none |

---

## Presets

### Using Built-in Presets

PixyToon ships with built-in presets for common use cases. Select one from the dropdown to instantly load tuned settings.

| Preset | Best for |
|--------|----------|
| **pixel_art** | General pixel art sprites (default LoRA + post-process) |
| **anime** | Anime-style characters |
| **character** | Detailed character art |
| **landscape** | Environments and backgrounds |
| **concept_art** | Concept exploration |
| **illustration** | Detailed illustrations |
| **realistic** | Photo-realistic output |

### Saving Your Own Presets

Found a combination of settings that works? Save it:

1. Configure all parameters (prompt, steps, CFG, post-process, etc.)
2. Click **Save** and enter a name
3. Your preset appears in the dropdown for future use
4. Click **Del** next to a user preset to remove it

> [!TIP]
> Save presets for each project or art style you work on. Switching between "RPG sprites" and "UI icons" becomes instant.

---

## Anti-Patterns

Things that don't work well — and what to do instead.

### CFG too high (>10)

**Symptom:** Over-saturated colors, artifacts, "deep-fried" look.

**Fix:** Keep CFG at 3.0-7.0. With Hyper-SD, the sweet spot is around 5.0.

---

### Steps too low without Hyper-SD

**Symptom:** Blurry, unformed images.

This shouldn't happen with the default configuration (Hyper-SD is always active), but if you disabled it: use 20-25 steps minimum.

---

### High strength on img2img when you want to preserve detail

**Symptom:** Your carefully drawn sketch is completely overwritten.

**Fix:** Start with strength 0.3-0.4 for img2img. Only go above 0.7 if you want radical reinterpretation.

---

### Generating above 768x768

**Symptom:** Duplicated faces, repeated patterns, incoherent composition.

**Fix:** SD 1.5 was trained on 512x512. Generate at 512x512, then upscale manually in Aseprite if needed.

---

### Too many colors + palette enforcement

**Symptom:** Palette enforcement makes everything muddy because there are too many close colors mapping to the same palette entry.

**Fix:** Quantize to fewer colors (8-16) before palette enforcement. Let quantization do the heavy lifting.

---

### Dithering without palette

**Symptom:** Dithering creates patterns between colors that are too similar, resulting in visual noise rather than smooth gradients.

**Fix:** Always set a palette (preset or custom) or at least reduce colors significantly (8-32) before enabling dithering.

---

### Animation strength too high

**Symptom:** Each frame looks completely different — no visual continuity.

**Fix:** For chain animations, keep strength at 0.20-0.35. For AnimateDiff, 0.20-0.40.

---

### Negative prompt is empty

**Symptom:** Smooth gradients, anti-aliased edges, photorealistic elements leaking into pixel art.

**Fix:** Always keep the default negative prompt. It specifically blocks anti-aliasing, smooth gradients, and photorealism — the three biggest enemies of pixel art generation.

---

**[README](../README.md)** | **[Guide](GUIDE.md)** | **[Cookbook](COOKBOOK.md)** | **[Live Paint](LIVE-PAINT.md)** | **[Audio Reactivity](AUDIO-REACTIVITY.md)** | **[API Reference](API-REFERENCE.md)** | **[Configuration](CONFIGURATION.md)** | **[Troubleshooting](TROUBLESHOOTING.md)**
