# Troubleshooting

Quick fixes for common issues. For detailed configuration, see [Configuration](CONFIGURATION.md).

**[README](../README.md)** | **[Guide](GUIDE.md)** | **[Cookbook](COOKBOOK.md)** | **[Live Paint](LIVE-PAINT.md)** | **[Audio Reactivity](AUDIO-REACTIVITY.md)** | **[API Reference](API-REFERENCE.md)** | **[Configuration](CONFIGURATION.md)** | **[Troubleshooting](TROUBLESHOOTING.md)**

---

## Connection

| Problem | Solution |
|---------|----------|
| Server won't start | Check `uv run python run.py` output for errors |
| Port already in use | Change `PIXYTOON_PORT` or kill existing process |
| Aseprite can't connect | Ensure server is running, check firewall on 127.0.0.1:9876 |
| "Server unresponsive" | Heartbeat watchdog detected no pong for 90s; auto-reconnect kicks in |
| "Reconnecting in Xs" | Normal: exponential backoff (2s to 30s); server is unreachable, will auto-retry |

## GPU / Performance

| Problem | Solution |
|---------|----------|
| CUDA OOM | Reduce resolution, disable torch.compile (`PIXYTOON_ENABLE_TORCH_COMPILE=False`), enable VAE tiling, close other GPU apps |
| torch.compile fails | Install Visual Studio 2022 with C++ Desktop Development workload; ensure Triton installed |
| "Not enough SMs" | Harmless Triton warning on consumer GPUs, can be ignored |
| CUDAGraphs tensor overwrite | Uses `default` compile mode. If using `reduce-overhead`, disable DeepCache |
| Slow first generation | Normal: torch.compile + Numba JIT warm up on first run (~30-60s) |

## Generation

| Problem | Solution |
|---------|----------|
| Generation timed out | Increase `PIXYTOON_GENERATION_TIMEOUT` or reduce steps/resolution |
| Cancel doesn't stop immediately | Server-side cancel ACK + 30s safety timer auto-unlocks UI; v0.7.3 concurrent receive handles cancel during long-running generations; check server terminal |
| Blurry or non-pixel results | Ensure post-processing is enabled (pixelate + quantize). Check denoise_strength isn't too low |
| Wrong colors | Try palette enforcement in CIELAB mode or adjust quantize_colors |

## LoRA / Models

| Problem | Solution |
|---------|----------|
| LoRA not found | Place `.safetensors` file in `server/models/loras/` |
| LoRA change is slow | Expected: LoRA weight change triggers recompilation (~30-60s once per change) |
| TI embedding not working | Place in `server/models/embeddings/`, use exact filename (without extension) in prompt |

## Animation

| Problem | Solution |
|---------|----------|
| AnimateDiff OOM | AnimateDiff needs ~8-10GB VRAM; reduce `frame_count` or resolution |
| AnimateDiff slow first run | Motion adapter downloads on first use (~97MB); subsequent runs use cache |
| Chain animation flicker | Lower denoise_strength (0.20-0.35) for more frame coherence |

## Live Paint

| Problem | Solution |
|---------|----------|
| Live Paint not starting | Ensure no generation is in progress (GPU_BUSY); check server logs |
| Live Paint high latency | Reduce steps (2-3), reduce resolution, ensure no other GPU load |
| Live Paint auto-stopped | Session times out after 5min of inactivity (configurable via `PIXYTOON_REALTIME_TIMEOUT`) |
| Brush strokes not detected | Ensure Auto mode is enabled, or use F5 (Manual mode) |

## Audio Reactivity

| Problem | Solution |
|---------|----------|
| Audio file not found | Use absolute path; supported: .wav, .mp3, .flac, .ogg, .m4a, .aac |
| Stem separation unavailable | Install demucs: `pip install demucs>=4.0` (heavy dependency, CPU only) |
| Modulation too subtle | Increase min/max range in slot, or try a preset with wider range (e.g., `glitch_chaos`) |
| MP4 export fails | Install ffmpeg and ensure it's in PATH. Set `PIXYTOON_FFMPEG_PATH` if non-standard location |
| Modulation too aggressive | Increase release frames (smoother decay), decrease max range |
| "Analysis failed" | Check server logs; ensure librosa installed, audio file not corrupted |

## Server Logs

The server logs to stdout with timestamps. Key log patterns:

- `INFO: Audio loaded: Xs, Y Hz, Z frames` — Successful audio analysis
- `INFO: BPM detected: X.X` — BPM extracted from audio
- `WARNING: Expression error at frame N` — Custom expression syntax error
- `ERROR: CUDA out of memory` — Reduce resolution or steps

---

**[README](../README.md)** | **[Guide](GUIDE.md)** | **[Cookbook](COOKBOOK.md)** | **[Live Paint](LIVE-PAINT.md)** | **[Audio Reactivity](AUDIO-REACTIVITY.md)** | **[API Reference](API-REFERENCE.md)** | **[Configuration](CONFIGURATION.md)** | **[Troubleshooting](TROUBLESHOOTING.md)**
