# SDDj Prompt Schedule DSL — Language Specification v1.0

## Overview

The Prompt Schedule DSL is a domain-specific text format for defining multi-prompt
animation sequences in SDDj. It maps time positions to prompts with configurable
transitions, enabling smooth visual evolution across frame sequences.

## Syntax

### EBNF Grammar

```ebnf
schedule      = { line_or_blank } ;
line_or_blank = blank_line | comment_line | auto_directive | keyframe_block ;
blank_line    = { whitespace } , newline ;
comment_line  = { whitespace } , "#" , { any_char } , newline ;
auto_directive = { whitespace } , "{auto}" , { whitespace } , newline ;

keyframe_block = time_marker , newline , { directive_line | prompt_line | negative_line } ;

time_marker   = { whitespace } , "[" , time_spec , "]" , { whitespace } , newline ;
time_spec     = frame_abs | percent_spec | seconds_spec ;
frame_abs     = integer ;                         (* absolute frame: [0], [24], [100] *)
percent_spec  = number , "%" ;                    (* percentage of total: [0%], [50%], [85.5%] *)
seconds_spec  = number , "s" ;                    (* seconds: [0s], [2.5s], [10s] *)

directive_line = { whitespace } , directive , newline ;
directive      = transition_dir | blend_dir | weight_dir | denoise_dir | cfg_dir | steps_dir ;
transition_dir = "transition:" , { whitespace } , transition_type ;
transition_type = "hard_cut" | "blend" | "linear_blend" | "ease_in" | "ease_out"
               | "ease_in_out" | "cubic" | "slerp" ;
blend_dir      = "blend:" , { whitespace } , integer ;      (* transition window in frames *)
weight_dir     = "weight:" , { whitespace } , weight_spec ;
weight_spec    = number                                       (* static: weight: 1.2 *)
               | number , "->" , number ;                     (* animated: weight: 1.0->1.5 *)
denoise_dir    = "denoise:" , { whitespace } , number ;       (* 0.0 - 1.0 *)
cfg_dir        = "cfg:" , { whitespace } , number ;           (* 1.0 - 30.0 *)
steps_dir      = "steps:" , { whitespace } , integer ;        (* 1 - 150 *)

prompt_line    = { whitespace } , prompt_text , newline ;     (* any text not matching above *)
negative_line  = { whitespace } , "--" , { whitespace } , prompt_text , newline ;

(* Terminals *)
integer       = digit , { digit } ;
number        = integer , [ "." , { digit } ] ;
digit         = "0" | "1" | ... | "9" ;
whitespace    = " " | "\t" ;
newline       = "\n" | "\r\n" ;
prompt_text   = { any_char - newline } ;
any_char      = ? any Unicode character ? ;
```

### File Reference

A schedule file can reference an external file:

```
file: relative/path/to/schedule.txt
```

The path is resolved relative to the Aseprite document directory. Path traversal
(`..\`) is rejected. Only files within the document directory tree are permitted.

---

## Semantics

### Time Resolution

| Format | Resolution Rule |
|--------|----------------|
| `[N]` (integer) | Absolute frame index (0-based) |
| `[N%]` (percent) | `floor(N / 100 * total_frames)`, clamped to `[0, total_frames - 1]` |
| `[Ns]` (seconds) | `floor(N * fps)`, clamped to `[0, total_frames - 1]` |

The first keyframe MUST start at frame 0. If it doesn't, frame 0 is implicit with
an empty prompt (inherits the global prompt from the dialog).

### Keyframe Structure

A keyframe consists of:

| Field | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `frame` | int | (required) | 0 – total_frames-1 | Start frame |
| `prompt` | string | `""` (inherit global) | — | Positive prompt text |
| `negative_prompt` | string | `""` | — | Negative prompt text |
| `transition` | enum | `hard_cut` | see grammar | Transition type into this keyframe |
| `transition_frames` | int | `0` | 0 – 120 | Frames over which to blend from previous |
| `weight` | float or float→float | `1.0` | 0.1 – 5.0 | Prompt embedding weight |
| `denoise_strength` | float | (inherit) | 0.0 – 1.0 | Per-keyframe denoise override |
| `cfg_scale` | float | (inherit) | 1.0 – 30.0 | Per-keyframe CFG override |
| `steps` | int | (inherit) | 1 – 150 | Per-keyframe steps override |

### Transition Types

| Type | Behavior |
|------|----------|
| `hard_cut` | Instant switch at keyframe boundary. No blending. |
| `blend` / `linear_blend` | Linear SLERP interpolation between prompt embeddings over `blend:` frames. |
| `ease_in` | Slow start, accelerating crossfade (quadratic ease-in). |
| `ease_out` | Fast start, decelerating crossfade (quadratic ease-out). |
| `ease_in_out` | S-curve crossfade (cubic ease-in-out). |
| `cubic` | Cubic Bézier crossfade (smooth, balanced). |
| `slerp` | Spherical linear interpolation in embedding space (alias for `blend`). |

All non-`hard_cut` transitions produce a blend weight ∈ [0.0, 1.0] that is used
to SLERP/LERP the CLIP text embeddings of the outgoing and incoming prompts.
The easing function shapes how this weight progresses across the transition window.

### Blend Weight Computation

For a frame `f` within the transition window `[start, start + transition_frames]`:

```
t = (f - start) / transition_frames   # normalized [0, 1]
blend_weight = easing_function(t)      # shaped [0, 1]
```

The engine encodes both prompts and produces:
```
final_embedding = slerp(embed_outgoing, embed_incoming, blend_weight)
```

### Per-Keyframe Parameter Overrides

When `denoise:`, `cfg:`, or `steps:` directives are present, they override the
base parameters for frames within that keyframe's region. Parameters are
interpolated between keyframes using the same easing curve as the prompt
transition, unless the next keyframe specifies a different value.

### The `{auto}` Directive

When `{auto}` appears in the schedule, the server's auto-fill engine generates
prompt variations for keyframes with empty prompts, using the locked subject
and randomness settings. Only empty prompts are filled; user-written prompts
are preserved.

### Prompt Text Rules

- Lines not matching any directive pattern are treated as prompt text
- Multiple prompt lines within one keyframe are concatenated with `, `
- Leading/trailing whitespace is trimmed from prompt text
- Empty prompts (or omitted) inherit the global prompt from the dialog

---

## Validation Rules

### Errors (generation blocked)

| Code | Condition |
|------|-----------|
| `E001` | Time marker out of range (negative, >100%, or exceeds total frames) |
| `E002` | Duplicate time marker (two keyframes at same frame) |
| `E003` | Keyframes not in chronological order |
| `E004` | Transition window exceeds distance to previous keyframe |
| `E005` | Invalid transition type |
| `E006` | Weight out of range (< 0.1 or > 5.0) |
| `E007` | Denoise out of range (< 0.0 or > 1.0) |
| `E008` | CFG out of range (< 1.0 or > 30.0) |
| `E009` | Steps out of range (< 1 or > 150) |
| `E010` | File reference path traversal rejected |
| `E011` | File reference not found |
| `E012` | Unrecognized directive syntax |

### Warnings (generation proceeds)

| Code | Condition |
|------|-----------|
| `W001` | First keyframe not at frame 0 (implicit keyframe inserted) |
| `W002` | Empty schedule (no keyframes defined) |
| `W003` | Very short transition window (< 2 frames) may be visually imperceptible |
| `W004` | Very high weight (> 2.0) may cause artifacts |
| `W005` | `{auto}` with all prompts already filled (auto-fill has nothing to do) |

---

## Examples

### Minimal: Two-Prompt Scene Change

```
[0]
a cyberpunk cityscape at night, neon lights, rain

[50%]
transition: hard_cut
a serene mountain landscape at sunrise, golden light
```

### Blended Evolution with Negative Prompts

```
[0]
a dark forest, mysterious shadows, moonlight
-- daylight, sunny, bright

[30%]
transition: ease_in_out
blend: 12
an enchanted forest clearing, bioluminescent plants, fairy lights
-- dark, scary, horror

[70%]
transition: blend
blend: 8
weight: 1.3
a celestial garden above the clouds, ethereal glow, floating islands
-- underground, cave, dark
```

### Audio-Synced with Auto-Fill

```
{auto}

[0]
a warrior standing in shadows

[4s]
transition: ease_in
blend: 8
weight: 1.2

[8s]
transition: ease_out
blend: 6

[12s]
transition: hard_cut
weight: 1.5
```

### Per-Keyframe Parameter Overrides

```
[0]
a calm ocean at sunset
denoise: 0.35
cfg: 5.0

[50%]
transition: ease_in_out
blend: 16
denoise: 0.65
cfg: 7.5
a stormy ocean, massive waves, lightning
-- calm, peaceful

[85%]
transition: ease_out
blend: 8
denoise: 0.40
cfg: 5.5
steps: 25
the storm passing, rainbow emerging, golden light breaking through clouds
```

### Multi-Line Prompts

```
[0]
a majestic dragon perched on a cliff,
intricate scales reflecting moonlight,
dark fantasy art style, highly detailed
-- cute, cartoon, simple

[60%]
transition: blend
blend: 10
the dragon taking flight,
wings spread wide against a starlit sky,
dynamic pose, dramatic lighting
```
