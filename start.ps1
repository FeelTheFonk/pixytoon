#!/usr/bin/env pwsh
#Requires -Version 7.0
$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "SDDj"
$PSStyle.OutputRendering = "Ansi"

# --- Helpers -----------------------------------------------------------------
$e = [char]27
$R  = "$e[0m";  $D = "$e[90m";  $W = "$e[97m"; $B = "$e[1m"
$G  = "$e[92m"; $Y = "$e[93m";  $C = "$e[96m"

function Ok($msg)   { Write-Host "  ${G}OK${R}  $D$msg$R" }
function Warn($msg) { Write-Host "  ${Y}!${R}  $msg" }

$Root = $PSScriptRoot

# --- Force offline mode: never contact HuggingFace Hub at runtime ------------
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:DO_NOT_TRACK = "1"

Write-Host ""
Write-Host "  ${B}${W}SDDj${R}  ${D}Launch${R}"
Write-Host "  ${D}$('-' * 36)${R}"
Write-Host ""

# --- Preflight: verify venv + sddj package installed ------------------------
$venvPython = "$Root/server/.venv/Scripts/python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "  ${Y}!${R}  Virtual environment not found. Run ${C}./setup.ps1${R} first."
    Read-Host "`n  Press Enter to exit"
    exit 1
}
# Quick sanity: can we import sddj?
$importCheck = & $venvPython -c "import sddj" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ${Y}!${R}  sddj package not installed in venv. Run ${C}./setup.ps1${R} first."
    Read-Host "`n  Press Enter to exit"
    exit 1
}

# --- Preflight: verify models exist (offline mode safety) -------------------
$modelsDir = "$Root/server/models"
$modelFiles = Get-ChildItem "$modelsDir" -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "\.(safetensors|ckpt|bin)$" }
if (-not $modelFiles) {
    Write-Host "  ${Y}!${R}  Model weights missing in server/models/. Offline mode cannot start."
    Write-Host "  ${Y}!${R}  Run ${C}./setup.ps1${R} to download required weights."
    Read-Host "`n  Press Enter to exit"
    exit 1
}

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
    # --- Start server (direct venv Python — zero network, zero uv resolve) ---
    Write-Host "  ${D}Starting server...${R}"
    $serverProc = Start-Process pwsh -ArgumentList "-NoExit", "-Command", "`$Host.UI.RawUI.WindowTitle='SDDj Server'; `$env:HF_HUB_OFFLINE='1'; `$env:TRANSFORMERS_OFFLINE='1'; `$env:HF_HUB_DISABLE_TELEMETRY='1'; `$env:DO_NOT_TRACK='1'; Set-Location '$Root/server'; & '$venvPython' run.py" -WindowStyle Minimized -PassThru

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
    Write-Host "  ${D}Launch Aseprite manually - SDDj opens on startup.${R}"
}

# --- Running -----------------------------------------------------------------
Write-Host ""
Write-Host "  ${D}$('-' * 36)${R}"
Write-Host "  ${W}Server:${R}  ${C}ws://127.0.0.1:9876/ws${R}"
Write-Host "  ${W}Action:${R}  Connect in SDDj dialog"
Write-Host "  ${D}$('-' * 36)${R}"
Write-Host ""

Read-Host "  Press Enter to stop the server"

# --- Shutdown ----------------------------------------------------------------
Write-Host ""
Write-Host "  ${D}Stopping server...${R}"

# 1) Graceful: ask the server to shut down via HTTP
$graceful = $false
try {
    Invoke-WebRequest -Uri "http://127.0.0.1:9876/shutdown" -Method POST -TimeoutSec 3 -ErrorAction SilentlyContinue | Out-Null
    Start-Sleep -Seconds 2
    # Verify it stopped
    try {
        Invoke-WebRequest -Uri "http://127.0.0.1:9876/health" -TimeoutSec 2 -ErrorAction SilentlyContinue | Out-Null
    } catch {
        $graceful = $true
    }
} catch {}

if ($graceful) {
    Ok "Server stopped (graceful)"
} else {
    # 2) Force: kill entire process tree recursively
    if ($serverProc -and -not $serverProc.HasExited) {
        # taskkill /T kills the process and ALL descendants recursively
        taskkill /T /F /PID $serverProc.Id 2>$null | Out-Null
    }
    # 3) Fallback: find any Python process listening on port 9876
    $pyPids = (Get-NetTCPConnection -LocalPort 9876 -ErrorAction SilentlyContinue).OwningProcess | Select-Object -Unique
    foreach ($pid in $pyPids) {
        if ($pid -and $pid -ne 0) {
            taskkill /T /F /PID $pid 2>$null | Out-Null
        }
    }
    # 4) Fallback: kill pwsh with SDDj Server title
    Get-Process -Name "pwsh" -ErrorAction SilentlyContinue |
        Where-Object { $_.MainWindowTitle -eq "SDDj Server" } |
        ForEach-Object { taskkill /T /F /PID $_.Id 2>$null | Out-Null }
    Start-Sleep -Seconds 1
    Ok "Server stopped (forced)"
}
Write-Host ""
