@echo off
title PixyToon — Setup
echo.
echo  ╔═══════════════════════════════════════╗
echo  ║   PixyToon — One-Click Setup          ║
echo  ╚═══════════════════════════════════════╝
echo.

set ROOT=%~dp0
cd /d "%ROOT%"

:: ── 1. Check uv ───────────────────────────────────────────────
where uv >nul 2>&1
if errorlevel 1 (
    echo [X] uv not found. Install it: https://docs.astral.sh/uv/
    pause
    exit /b 1
)
echo [OK] uv found

:: ── 2. Create venv + install deps (torch CUDA 12.8 + Triton) ─
echo.
echo [1/4] Installing Python dependencies (torch CUDA 12.8 + Triton)...
cd /d "%ROOT%server"
uv venv --python 3.13 2>nul
uv pip install -e .
if errorlevel 1 (
    echo [X] Dependency install failed
    pause
    exit /b 1
)
echo [OK] Dependencies installed (torch, diffusers, triton-windows, etc.)
cd /d "%ROOT%"

:: ── 3. Download models ────────────────────────────────────────
echo.
echo [2/4] Downloading models (~10 GB total)...
cd /d "%ROOT%server"
uv run python "%ROOT%scripts\download_models.py" --all
if errorlevel 1 (
    echo [X] Model download failed
    pause
    exit /b 1
)
echo [OK] Models downloaded
cd /d "%ROOT%"

:: ── 4. Build extension ────────────────────────────────────────
echo.
echo [3/4] Packaging Aseprite extension...
cd /d "%ROOT%server"
uv run python "%ROOT%scripts\build_extension.py"
if errorlevel 1 (
    echo [X] Extension build failed
    pause
    exit /b 1
)
echo [OK] Extension built
cd /d "%ROOT%"

:: ── 5. Install extension into Aseprite ────────────────────────
echo.
echo [4/4] Installing scripts into Aseprite...
set ASEPRITE_SCRIPTS=%APPDATA%\Aseprite\scripts
set ASEPRITE_EXT=%APPDATA%\Aseprite\extensions\pixytoon
rem -- Scripts dir (compiled Aseprite)
mkdir "%ASEPRITE_SCRIPTS%" 2>nul
copy /y "%ROOT%extension\scripts\json.lua" "%ASEPRITE_SCRIPTS%\" >nul
copy /y "%ROOT%extension\scripts\pixytoon.lua" "%ASEPRITE_SCRIPTS%\" >nul
rem -- Extensions dir (purchased Aseprite)
if exist "%ASEPRITE_EXT%" rmdir /s /q "%ASEPRITE_EXT%"
mkdir "%ASEPRITE_EXT%\scripts" 2>nul
copy /y "%ROOT%extension\package.json" "%ASEPRITE_EXT%\" >nul
copy /y "%ROOT%extension\scripts\json.lua" "%ASEPRITE_EXT%\scripts\" >nul
copy /y "%ROOT%extension\scripts\pixytoon.lua" "%ASEPRITE_EXT%\scripts\" >nul
echo [OK] Scripts installed to %ASEPRITE_SCRIPTS%

:: ── Done ──────────────────────────────────────────────────────
echo.
echo  ╔═══════════════════════════════════════╗
echo  ║   Setup complete!                     ║
echo  ║   Run start.bat to launch PixyToon    ║
echo  ╚═══════════════════════════════════════╝
echo.
echo NOTE: torch.compile requires Visual Studio 2022
echo       C++ Desktop Development workload for
echo       optimal performance (Triton backend).
echo.
pause
