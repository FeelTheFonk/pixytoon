#!/usr/bin/env pwsh
#Requires -Version 7.0

[CmdletBinding()]
param(
    [switch]$Online
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSStyle.OutputRendering = "Ansi"

# --- UI ---
$e = [char]27
$R  = "$e[0m";  $D = "$e[90m"; $W = "$e[97m"; $B = "$e[1m"
$G  = "$e[92m"; $Re = "$e[91m"; $Y = "$e[93m"; $C = "$e[96m"
$CL = "$e[2K"  # ANSI: clear entire line
function Ok($msg)   { Write-Host "  ${G}OK${R}  $D$msg$R" }
function Warn($msg) { Write-Host "  ${Y}!${R}  $msg" }
try { $Host.UI.RawUI.WindowTitle = "SDDj" } catch {}

$Root = [System.IO.Path]::GetFullPath($PSScriptRoot)
$ServerDir = Join-Path $Root "server"
$ModelsDir = Join-Path $ServerDir "models"

# --- Health check helper (DRY: used for pre-check + readiness poll) ---
function Test-ServerHealth {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:9876/health" -TimeoutSec 2 -ErrorAction Stop
        if ($r.StatusCode -eq 200) { return ($r.Content | ConvertFrom-Json) }
    } catch {}
    return $null
}

# --- Offline mode (unless -Online) ---
if (-not $Online) {
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
}
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:DO_NOT_TRACK = "1"

Write-Host "`n  ${B}${W}SDDj${R}  ${D}Launch${R}`n  ${D}$('-' * 36)${R}`n"

# --- Preflight ---
$VenvPython = Join-Path $ServerDir ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Warn "Virtual environment not found. Run .\setup.ps1 first."
    Read-Host "`n  Press Enter to exit"
    exit 1
}

# Package check: metadata-only (avoids torch/CUDA import chain, ~50ms vs ~500ms)
& $VenvPython -c "from importlib.metadata import version; version('sddj-server')" 2>$null
if ($LASTEXITCODE -ne 0) {
    Warn "sddj package not installed in venv. Run .\setup.ps1 first."
    Read-Host "`n  Press Enter to exit"
    exit 1
}

# Model check: short-circuit on first match (no full recursive enumeration)
$hasModel = Get-ChildItem -Path $ModelsDir -Recurse -Depth 2 -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match "\.(safetensors|ckpt|bin)$" } |
    Select-Object -First 1
if (-not $hasModel) {
    Warn "Model weights missing in server\models\. Offline mode cannot start."
    Warn "Run .\setup.ps1 to download required weights."
    Read-Host "`n  Press Enter to exit"
    exit 1
}

# --- Check if server already running ---
$h = Test-ServerHealth
$alreadyRunning = ($null -ne $h -and $h.status -eq "ok")
$serverProc = $null

if ($alreadyRunning) {
    Ok "Server already running"
} else {
    Write-Host "  ${D}Starting server...${R}"

    # EncodedCommand: avoids quoting issues, sets window title, uses --frozen (skip sync)
    $EncodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes(
        "`$Host.UI.RawUI.WindowTitle='SDDj Server'; Set-Location -LiteralPath `"$ServerDir`"; uv run --frozen run.py"
    ))
    $serverProc = Start-Process pwsh -ArgumentList "-EncodedCommand", $EncodedCommand -WindowStyle Minimized -PassThru

    Write-Host "  ${D}Waiting for engine to load (~10-30s)...${R}"

    $maxWait = 120; $waited = 0; $ready = $false
    while ($waited -lt $maxWait) {
        # Progressive polling: 2s for first 20s (fast detection), then 3s (reduce overhead)
        $interval = if ($waited -lt 20) { 2 } else { 3 }
        Start-Sleep -Seconds $interval
        $waited += $interval
        # Detect server crash before timeout
        if ($null -ne $serverProc -and $serverProc.HasExited) {
            Warn "Server process exited (code $($serverProc.ExitCode)) before becoming ready"
            Warn "Check server window or run 'cd server && uv run run.py' for details"
            Read-Host "`n  Press Enter to exit"
            exit 1
        }
        $h = Test-ServerHealth
        if ($null -ne $h -and $h.status -eq "ok" -and $h.loaded -eq $true) {
            $ready = $true; break
        }
        Write-Host "`r${CL}  ${D}Waiting... (${waited}s)${R}" -NoNewline
    }
    Write-Host "`r${CL}" -NoNewline  # Clear progress line cleanly
    if ($ready) { Ok "Server ready (${waited}s)" }
    else { Warn "Server not responding after ${maxWait}s — launching Aseprite anyway" }
}

# --- Launch Aseprite ---
Write-Host ""
$aseProc = $null  # Initialize unconditionally (StrictMode requires it before monitor loop)
$asePath = Join-Path $Root "bin\aseprite\aseprite.exe"
if (Test-Path $asePath) {
    Write-Host "  ${D}Launching Aseprite...${R}"
    $aseProc = Start-Process -FilePath $asePath -PassThru
    Ok "Aseprite launched"
} else {
    Warn "Aseprite not found at bin\aseprite"
    Write-Host "  ${D}Launch Aseprite manually — SDDj connects on startup.${R}"
}

Write-Host "`n  ${D}$('-' * 36)${R}`n  ${W}Server:${R}  ${C}ws://127.0.0.1:9876/ws${R}"
Write-Host "  ${W}Action:${R}  Connect in SDDj dialog`n  ${D}$('-' * 36)${R}`n"

# --- Monitor: auto-shutdown when Aseprite exits ---
Write-Host "  ${D}Monitoring... (close Aseprite or press any key to exit)${R}"
while ($true) {
    Start-Sleep -Milliseconds 500
    # Aseprite exited?
    if ($null -ne $aseProc -and $aseProc.HasExited) {
        Write-Host "`n  ${D}Aseprite closed — shutting down...${R}"
        break
    }
    # Server crashed?
    if ($null -ne $serverProc -and $serverProc.HasExited) {
        Warn "Server process exited unexpectedly (code $($serverProc.ExitCode))"
        break
    }
    # User keypress? (guarded: [Console]::KeyAvailable throws if stdin is redirected)
    try {
        if ([Console]::KeyAvailable) {
            $null = [Console]::ReadKey($true)
            break
        }
    } catch {}
}

# --- Shutdown: only stop server if WE started it ---
if ($null -ne $serverProc -and -not $serverProc.HasExited) {
    Write-Host "`n  ${D}Stopping server...${R}"
    $graceful = $false
    try {
        $null = Invoke-WebRequest -Uri "http://127.0.0.1:9876/shutdown" -Method POST -TimeoutSec 3 -ErrorAction SilentlyContinue
    } catch {}
    # Poll process exit: 10x 500ms = 5s max (server call_later(0.5) + uvicorn shutdown ≈ 1-2s)
    for ($i = 0; $i -lt 10; $i++) {
        Start-Sleep -Milliseconds 500
        if ($serverProc.HasExited) { $graceful = $true; break }
    }
    if ($graceful) {
        Ok "Server stopped (graceful)"
    } else {
        Warn "Graceful shutdown timed out — killing process tree"
        taskkill /T /F /PID $serverProc.Id 2>$null | Out-Null
        Ok "Server stopped (forced)"
    }
} elseif ($null -ne $serverProc -and $serverProc.HasExited) {
    # Server crashed during monitor loop — already dead, no false "stopped" message
    Write-Host ""
} elseif ($alreadyRunning) {
    Write-Host "`n  ${D}Server was pre-existing — left running${R}"
}
Write-Host ""
