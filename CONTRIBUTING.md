# Contributing to SDDj

First off, thank you for considering contributing to SDDj! The project aims to provide the ultimate SOTA generative AI experience within Aseprite.

## Development Setup

SDDj uses `uv` for lightning-fast Python dependency management and builds.

1. **Clone the repository**
   ```bash
   git clone https://github.com/FeelTheFonk/sddj.git
   cd sddj
   ```
2. **Install dependencies**
   ```bash
   cd server
   uv sync --all-extras
   ```
3. **Run the server in dev mode**
   ```bash
   uv run python run.py
   ```

## Repository Structure

SDDj is split into two halves that communicate over WebSockets:

* `server/` — Python backend (FastAPI, PyTorch, Diffusers, Librosa). Does the heavy lifting.
* `extension/` — Lua frontend (Aseprite scripting API). Handles the UI, image extraction, and canvas injection.

Detailed architecture diagrams are available in **[docs/REFERENCE.md](docs/REFERENCE.md#architecture)**.

## Python Code Style

We use [Ruff](https://astral.sh/ruff) for linting and formatting. 

* **Line length**: 100 characters.
* **Typing**: Strict type hints required for all new code. Use `from __future__ import annotations`.
* **Docstrings**: Google style docstrings for classes and public functions.
* **Imports**: Grouped and sorted by Ruff.

Run Ruff before committing:
```bash
uv run ruff check .
uv run ruff format .
```

## Lua Code Style

The Aseprite extension is written in standard Lua 5.3.

* **Modularity**: Code is split across `sddj_*.lua` files (dialog, state, network, generic utils). Do not put everything in `sddj.lua`.
* **State Management**: UI state is centralized in `sddj_state.lua` to ensure the dialog can be closed and reopened safely.
* **Error Handling**: Use `pcall` when interacting with Aseprite's API if there's a risk of the user having closed the sprite/layer mid-generation.

## Testing

New features or bug fixes in the Python server must include `pytest` tests.

```bash
cd server
uv run pytest -v
```

* **Core Logic**: Test prompt generation, metadata parsing, file routing, and DSP (audio) math heavily.
* **Integration**: `tests/test_generation.py` ensures the diffusion pipeline actually compiles and runs end-to-end. We mock network calls but run the actual PyTorch operations.

## Pull Request Process

1. **Branch**: Create a feature branch (`feat/your-feature` or `fix/your-bug`).
2. **Code**: Write your code. Keep commits logical.
3. **Docs**: If you add a user-facing feature, update the relevant `docs/*.md` files. **This project prides itself on SOTA documentation.**
4. **Test**: Ensure `pytest` passes and `ruff` reports zero warnings.
5. **PR**: Submit a Pull Request describing the problem solved and the approach taken.

## Architecture Philosophy

- **Offline by default**: No runtime downloads. `download_models.py` fetches everything upfront.
- **SOTA Performance**: Always favor `torch.compile`, VAE slicing, DeepCache, and distilled models over slower native equivalents if quality is preserved.
- **Resilience**: The server must never crash Aseprite. WebSockets must auto-reconnect. Generations must timeout if abandoned.
