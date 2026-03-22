#!/usr/bin/env pwsh
#Requires -Version 7.0
$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "PixyToon"
$PSStyle.OutputRendering = "Ansi"

# --- Helpers -----------------------------------------------------------------
$e = [char]27
$R  = "$e[0m";  $D = "$e[90m";  $W = "$e[97m"; $B = "$e[1m"
$G  = "$e[92m"; $Y = "$e[93m";  $C = "$e[96m"

function Ok($msg)   { Write-Host "  ${G}OK${R}  $D$msg$R" }
function Warn($msg) { Write-Host "  ${Y}!${R}  $msg" }

$Root = $PSScriptRoot

Write-Host ""
Write-Host "  ${B}${W}PixyToon${R}  ${D}Launch${R}"
Write-Host "  ${D}$('-' * 36)${R}"
Write-Host ""

# --- Check server already running --------------------------------------------
$running = $false
try {
    $resp = Invoke-WebRequest -Uri "http://127.0.0.1:9876/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
    if ($resp.StatusCode -eq 200) { $running = $true }
} catch {}

$serverProc = $null

if ($running) {
    Ok "Server already running"
} else {
    # --- Start server --------------------------------------------------------
    Write-Host "  ${D}Starting server...${R}"
    $serverProc = Start-Process pwsh -ArgumentList "-NoExit", "-Command", "`$Host.UI.RawUI.WindowTitle='PixyToon Server'; Set-Location '$Root/server'; uv run python run.py" -WindowStyle Minimized -PassThru

    # --- Wait for ready ------------------------------------------------------
    Write-Host "  ${D}Waiting for engine to load...${R}"
    Write-Host "  ${D}(first launch: ~30s load + ~30s torch.compile warmup)${R}"

    $maxWait = 120; $waited = 0; $ready = $false

    while ($waited -lt $maxWait) {
        Start-Sleep -Seconds 3
        $waited += 3
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:9876/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
            if ($r.Content -match "true") { $ready = $true; break }
        } catch {}
        Write-Host "`r  ${D}Waiting... (${waited}s)${R}" -NoNewline
    }
    Write-Host ""

    if ($ready) { Ok "Server ready (${waited}s)" }
    else { Warn "Server not responding after ${maxWait}s - launching Aseprite anyway" }
}

# --- Launch Aseprite ---------------------------------------------------------
Write-Host ""
$asePath = "$Root/bin/aseprite/aseprite.exe"
if (Test-Path $asePath) {
    Write-Host "  ${D}Launching Aseprite...${R}"
    Start-Process $asePath
    Ok "Aseprite launched"
} else {
    Warn "Aseprite not found at bin/aseprite/"
    Write-Host "  ${D}Launch Aseprite manually - PixyToon opens on startup.${R}"
}

# --- Running -----------------------------------------------------------------
Write-Host ""
Write-Host "  ${D}$('-' * 36)${R}"
Write-Host "  ${W}Server:${R}  ${C}ws://127.0.0.1:9876/ws${R}"
Write-Host "  ${W}Action:${R}  Connect in PixyToon dialog"
Write-Host "  ${D}$('-' * 36)${R}"
Write-Host ""

Read-Host "  Press Enter to stop the server"

# --- Shutdown ----------------------------------------------------------------
Write-Host ""
Write-Host "  ${D}Stopping server...${R}"
if ($serverProc -and -not $serverProc.HasExited) {
    # Kill the pwsh host and all its child processes (uv, python)
    $id = $serverProc.Id
    Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $id } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
} else {
    # Fallback: find by window title (server was already running or PID lost)
    Get-Process -Name "pwsh" -ErrorAction SilentlyContinue |
        Where-Object { $_.MainWindowTitle -eq "PixyToon Server" } |
        Stop-Process -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 1
Ok "Server stopped"
Write-Host ""
