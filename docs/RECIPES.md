# Recipes

Tested settings by creative intention. No theory — just what works.

---

## Parameter Matrix

One-line recipes. Everything not listed stays at default (steps=8, CFG=5.0, clip_skip=2, denoise=1.0, pixelate=128, colors=32, quantize=kmeans, dither=none).

> [!NOTE]
> Non-pixel-art styles (anime, illustration, realistic) use: Pixelate **OFF**, Colors **256**, no quantize, no dither.

### Generation

| Intent | Steps | CFG | Strength | Pixelate | Colors | Extras |
|--------|-------|-----|----------|----------|--------|--------|
| **Character sprite** | 8 | 5.0 | 1.0 | 64 | 16–24 | Remove BG |
| **Tiny icon (32×32)** | 8 | 6.0–7.0 | 1.0 | 32 | 8–12 | Remove BG |
| **Chibi / small** | 8 | 6.0 | 1.0 | 32 | 8–12 | Size 256×256 |
| **Enemy / monster** | 8 | 4.0–5.0 | 1.0 | 48–64 | 12–20 | Remove BG |
| **UI element** | 8 | 7.0 | 1.0 | 48–64 | 6–10 | Custom palette |
| **Side-scroller BG** | 8 | 4.0–5.0 | 1.0 | 128–192 | 32–48 | Size 768×512, Bayer 4×4 |
| **Top-down tile** | 8 | 6.0 | 1.0 | 64–128 | 16–24 | Same palette for set |
| **Interior / room** | 8 | 5.0–6.0 | 1.0 | 128 | 24–32 | Floyd-Steinberg |
| **Portrait (sketch→sprite)** | 8 | 5.0 | 0.5–0.6 | 64–96 | 16–24 | img2img mode |
| **Hi-fi pixel art** | 10–12 | 5.0 | 1.0 | 192–256 | 48–64 | Floyd-Steinberg |
| **Retro (Game Boy)** | 8 | 5.0 | 1.0 | 32–64 | 4 | Bayer 2×2, GB palette |
| **Retro (NES)** | 8 | 5.0 | 1.0 | 64 | 12–16 | NES palette |
| **Retro (PICO-8)** | 8 | 5.0 | 1.0 | 64–128 | 16 | PICO-8 palette |
| **Anime character** | 10 | 6.0 | 1.0 | OFF | 256 | clip_skip=2 |
| **Anime portrait** | 10 | 6.0 | 1.0 | OFF | 256 | clip_skip=2 |
| **Anime background** | 10 | 5.0 | 1.0 | OFF | 256 | Size 768×512 |
| **Illustration** | 12 | 7.0 | 1.0 | OFF | 256 | — |
| **Concept art** | 12 | 6.5 | 0.95 | OFF | 256 | — |
| **Watercolor** | 10 | 5.5 | 1.0 | OFF | 128 | Preserve texture |
| **Realistic portrait** | 15 | 7.5 | 1.0 | OFF | 256 | clip_skip=1, neg: `cartoon, anime` |
| **Painterly portrait** | 10 | 5.5 | 1.0 | OFF | 256 | — |
| **Modern UI icon** | 8 | 7.0 | 1.0 | OFF | 256 | Size 256×256, Remove BG |
| **Illustrated item** | 12 | 7.0 | 1.0 | OFF | 256 | Remove BG |
| **Abstract** | 8 | 4.0 | 1.0 | OFF | 256 | — |

### Animation

| Intent | Method | Frames | Strength | Seed Mode | Extras |
|--------|--------|--------|----------|-----------|--------|
| **Walk cycle** | chain | 4–8 | 0.25–0.35 | increment | 100–120ms duration |
| **Idle anim** | animatediff | 8–16 | 0.20–0.30 | — | FreeInit ON |
| **Effect (fire/water)** | animatediff | 8–12 | 0.35–0.50 | — | Remove BG |
| **Anime walk** | chain | 4–8 | 0.35 | increment | Non-pixel-art PP |
| **Abstract motion** | animatediff | 16 | — | — | CFG 4.0, FreeInit ON |
| **Illustration morph** | chain | 8–12 | 0.40 | increment | Colors 128 |
| **Audio-reactive** | chain or animatediff | — | — | — | See [Audio](AUDIO.md) |

### Quick Reference (Top 5 Workflows)

| Goal | Mode | Key Settings | Post-Process |
|------|------|-------------|--------------|
| **Tiny Sprite** | txt2img | CFG 6–7, Steps 8 | Pixelate 32, Colors 8–12 |
| **Lineart → Sprite** | controlnet_canny/lineart | CFG 5.5–6, Strength 1.0 | Pixelate 64, Colors 16, Preset palette |
| **Hi-Fi Illustration** | txt2img / inpaint | CFG 7, Steps 12 | Pixelate OFF, Colors 256 |
| **Rapid Variation** | txt2img + Loop | Output: sequence | Cancel after 10–20 frames |
| **Audio Animation** | Audio tab | Lightning (CFG 2, Steps 4) | Load preset (e.g., `cinematic_sweep`) |

---

## Techniques

### NPC Variations (Seed Technique)

Use a fixed seed and change one prompt detail at a time:

```
Seed 42: pixel art, townsperson, blue shirt, brown hair
Seed 42: pixel art, townsperson, red shirt, brown hair  ← same composition, different detail
Seed 43: pixel art, townsperson, blue shirt, brown hair  ← different pose
```

### Portrait Iteration (img2img)

1. First pass: img2img at strength **0.6** — get the overall look
2. If direction is good, lower to **0.3–0.4**
3. Make manual edits in Aseprite
4. Run again at **0.2–0.3** to polish

### ControlNet Workflows

| Workflow | Mode | Input | Prompt tip |
|----------|------|-------|-----------|
| **Lineart → pixel art** | controlnet_canny | Clean outlines | Start with `pixel art, colored game sprite` |
| **Sketch → sprite** | controlnet_scribble | Rough shapes/stick figures | Most forgiving mode |
| **Pose → character** | controlnet_openpose | Stick figure skeleton | Include `(action/pose)` in prompt |
| **Coloring lineart** | controlnet_lineart | Line drawing | `colored version, vibrant colors, flat shading` |

### Sequence + Loop

1. Set **Output** to `sequence`, enable **Loop Mode**
2. Each generation becomes a timeline frame you can scrub
3. `increment` seed → subtle variations. `random` seed → diverse exploration
4. Cancel anytime — partial results are kept

### Random Loop Discovery

1. Enable **Loop** + **Random Loop**
2. Optionally **Lock Subject** (e.g., "dragon")
3. Result: endless variations — pixel art dragon, chibi dragon, boss dragon… all auto-generated
4. Let it run 5 minutes, then pick favorites

---

## Color Control

### Built-in Palettes

Set **Palette** to **Preset** in Post-Process.

| Palette | Colors | Character |
|---------|--------|-----------|
| **PICO-8** | 16 | Fantasy console — vibrant, limited, recognizable |
| **Game Boy** | 4 | Green monochrome. Maximum constraint |
| **NES** | 54 | Classic 8-bit warmth |
| **SNES** | 256 | Full 16-bit range |
| **C64** | 16 | Earthy, muted, nostalgic |
| **Endesga 32** | 32 | Modern pixel art standard — warm, versatile |
| **Endesga 64** | 64 | Extended Endesga |

> [!TIP]
> **PICO-8** and **Endesga 32** are the most versatile starting points.

### Custom Palette

1. Export hex codes from Aseprite palette
2. Set palette to **Custom**
3. Paste: `#1a1c2c #5d275d #b13e53 #ef7d57 #ffcd75 ...`

### Palette CRUD

- **Save Palette**: click to save current custom colors with a name
- **Del Palette**: remove a saved palette
- Saved palettes appear in the preset dropdown alongside built-in palettes (stored as JSON in `server/palettes/`)

### When Colors Shift Too Much

- Increase color count first (quantize to 48+), then enforce palette
- Add **dithering** (Floyd-Steinberg) after enforcement
- Try a larger palette (Endesga 64 or SNES)

---

## QR Code / Illusion Art

Use `controlnet_qrcode` mode to embed scannable QR codes or hidden patterns inside generated imagery.

### Recipe: Scannable QR Art

| Parameter | Value | Note |
|-----------|-------|------|
| **Mode** | `controlnet_qrcode` | QR Code Monster ControlNet |
| **Steps** | 20 | Higher than default — QR needs resolution |
| **CFG** | 7–10 | Structure enforcement |
| **Conditioning Scale** | 1.3–1.5 | Lower = more artistic, higher = more scannable |
| **Guidance End** | 0.7–0.85 | Stop conditioning late but not at 100% — allows artistic finishing |
| **Denoise** | 1.0 | Full generation |

**Tips:**
- The **control image** should be a high-contrast QR code (black on white, centered, with quiet zone)
- Test scannability with a phone camera after generation
- Lower `conditioning_scale` for more artistic blending, raise it if QR fails to scan
- Works well with architectural/landscape prompts: `"a medieval castle, detailed stone walls, dramatic lighting"`
- Pair with pixelate post-processing for stylized scannable art

**Environment overrides** (optional):
```
SDDJ_QR_CONTROLNET_CONDITIONING_SCALE=1.5
SDDJ_QR_CONTROL_GUIDANCE_END=0.8
SDDJ_QR_DEFAULT_STEPS=20
```

---

## Anti-Patterns

| Mistake | Symptom | Fix |
|---------|---------|-----|
| **CFG > 10** | Over-saturated, artifacts, "deep-fried" | Keep at 3–7. Sweet spot ~5.0 with Hyper-SD |
| **Too few steps without Hyper-SD** | Blurry, unformed | Default config has Hyper-SD always active. If disabled: 20–25 steps |
| **High strength on img2img** | Sketch completely overwritten | Start at 0.3–0.4 for preservation |
| **Size > 768×768** | Duplicated faces, repeated patterns | Generate at 512, upscale in Aseprite |
| **Many colors + palette enforcement** | Muddy colors | Quantize to 8–16 first, then enforce |
| **Dithering without palette** | Visual noise instead of smooth gradients | Set a palette or reduce colors (8–32) first |
| **Animation strength > 0.40** | No visual continuity | Chain: 0.20–0.35. AnimateDiff: 0.20–0.40 |
| **Empty negative prompt** | Anti-aliased edges, smooth gradients leak in | Keep the default negative prompt |
| **Realistic style + clip_skip=2** | Cartoonish output | Switch to clip_skip=1, add `cartoon, anime` to negative |
| **Colors washed out** | Desaturated, flat | Increase CFG (6–7), add "vibrant colors" to prompt |
| **Anime features distorted** | Misshapen eyes, broken proportions | Keep clip_skip=2, CFG 5–6, add "detailed eyes" |
