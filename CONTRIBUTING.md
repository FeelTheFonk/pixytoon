## Development Setup

```powershell
git clone https://github.com/FeelTheFonk/sddj.git
cd sddj/server
uv sync --all-extras   # all deps including demucs, sageattention, etc.
uv run python run.py   # dev server on ws://localhost:9999
```

## Repository Structure

| Directory | Stack | Role |
|-----------|-------|------|
| `server/` | Python (PyTorch, Diffusers, Librosa) | Generation engine, WebSocket API |
| `extension/` | Lua (Aseprite scripting API) | UI, image capture, canvas injection |
| `docs/` | Markdown | User guide, API reference, DSL spec, sources |
| `scripts/` | PowerShell/Python | Model download, extension build, integration tests |

Architecture: [docs/REFERENCE.md](docs/REFERENCE.md#architecture)

## Code Style

### Python
- **Ruff** for lint + format. Line length: 100. Strict type hints (`from __future__ import annotations`).
- Google-style docstrings for classes and public functions.
```powershell
uv run ruff check . && uv run ruff format .
```

### Lua (Aseprite Extension)
- Code split across `sddj_*.lua` modules — do not merge into `sddj.lua`.
- UI state centralized in `sddj_state.lua`. Use `pcall` for Aseprite API calls that may fail mid-generation.

## Testing

```powershell
cd server && uv run pytest -v
```

New features/fixes require `pytest` tests. `scripts/test_generate.py` runs end-to-end pipeline tests.

## Pull Request Process

1. Branch: `feat/your-feature` or `fix/your-bug`
2. Code + tests + docs (if user-facing)
3. `pytest` passes, `ruff` zero warnings
4. PR with problem description + approach

## Architecture Philosophy

- **Offline by default** — no runtime downloads, `download_models.py` fetches everything upfront
- **SOTA Performance** — torch.compile, DeepCache, distilled models over slower alternatives
- **Resilience** — server never crashes Aseprite, WebSockets auto-reconnect, generations timeout
