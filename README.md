# SDDj v0.9.94

Stable Diffusion pixel art generation and animation directly inside Aseprite, 100% offline.

## Quick Start

```powershell
.\setup.ps1          # deps, models, extension
.\start.ps1          # launch server + Aseprite
```

## What It Does

- Generate, inpaint, and animate pixel art from text or existing artwork via SD 1.5 + AnimateDiff
- Audio-reactive generation mapping DSP features to diffusion parameters in real time
- ControlNet spatial conditioning (OpenPose, Canny, Scribble, Lineart, QR Code)
- Automated post-processing: background removal, downscale, palette quantization, dithering

## Requirements

- **GPU**: NVIDIA, 4 GB VRAM minimum (8 GB+ for AnimateDiff / ControlNet)
- **CUDA**: 12.x
- **Aseprite**: 1.3+
- **Windows**: 10 / 11

## Documentation

| Doc | Content |
|-----|---------|
| [Guide](docs/GUIDE.md) | Setup, modes, parameters, animation, performance |
| [Reference](docs/REFERENCE.md) | Architecture, WebSocket API, environment variables |
| [Prompt Schedule DSL](docs/PROMPT_SCHEDULE_DSL.md) | Keyframe grammar, transitions, validation rules |
| [Sources](docs/SOURCES.md) | Papers and technical references |

## License

MIT
