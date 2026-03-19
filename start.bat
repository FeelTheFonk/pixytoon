@echo off
title PixyToon — Server + Aseprite
echo.
echo  ╔═══════════════════════════════════════╗
echo  ║   PixyToon — Start                    ║
echo  ╚═══════════════════════════════════════╝
echo.

set ROOT=%~dp0

:: ── 1. Launch server in background ────────────────────────────
echo Starting PixyToon server on ws://127.0.0.1:9876/ws ...
cd /d "%ROOT%server"
start "PixyToon Server" cmd /k "uv run python run.py"

:: ── 2. Wait for server to be ready ────────────────────────────
echo Waiting for engine to load...
echo   (first launch: ~30s load + ~30s torch.compile warmup on first generate)
set MAX_WAIT=120
set WAITED=0

:wait_server
curl -s http://127.0.0.1:9876/health >nul 2>&1
if not errorlevel 1 goto server_ready
if %WAITED% geq %MAX_WAIT% (
    echo [WARN] Server not responding after %MAX_WAIT%s — launching Aseprite anyway.
    goto launch_aseprite
)
timeout /t 3 /nobreak >nul
set /a WAITED+=3
echo   ... waiting (%WAITED%s)
goto wait_server

:server_ready
echo [OK] Server is ready.

:: ── 3. Launch Aseprite ────────────────────────────────────────
:launch_aseprite
echo Launching Aseprite...
start "" "%ROOT%bin\aseprite\aseprite.exe"

echo.
echo  ╔═══════════════════════════════════════╗
echo  ║   Server running in background        ║
echo  ║   In Aseprite:                        ║
echo  ║     File ^> Scripts ^> PixyToon         ║
echo  ║     Connect ^> Generate                ║
echo  ╚═══════════════════════════════════════╝
echo.
echo Press any key to stop the server...
pause >nul

:: ── Kill server ───────────────────────────────────────────────
:: Try graceful first, then force
taskkill /fi "WINDOWTITLE eq PixyToon Server*" >nul 2>&1
timeout /t 2 /nobreak >nul
taskkill /fi "WINDOWTITLE eq PixyToon Server*" /f >nul 2>&1
echo Server stopped.
