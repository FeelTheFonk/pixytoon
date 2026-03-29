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
function Ok($msg)   { Write-Host "  ${G}OK${R}  $D$msg$R" }
function Warn($msg) { Write-Host "  ${Y}!${R}  $msg" }
try { $Host.UI.RawUI.WindowTitle = "SDDj" } catch {}

$Root = [System.IO.Path]::GetFullPath($PSScriptRoot)
$ServerDir = Join-Path -Path $Root -ChildPath "server"
$ModelsDir = Join-Path -Path $ServerDir -ChildPath "models"

# --- Force offline mode (unless -Online) ---
if (-not $Online) {
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
}
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:DO_NOT_TRACK = "1"

Write-Host "`n  ${B}${W}SDDj${R}  ${D}Launch${R}`n  ${D}$('-' * 36)${R}`n"

# --- Preflight ---
$VenvPython = Join-Path -Path $ServerDir -ChildPath ".venv\Scripts\python.exe"
if (-not (Test-Path -Path $VenvPython)) {
    Warn "Virtual environment not found. Run .\setup.ps1 first."
    Read-Host "`n  Press Enter to exit"
    exit 1
}

$importCheck = & $VenvPython -c "import sddj" 2>&1
if ($LASTEXITCODE -ne 0) {
    Warn "sddj package not installed in venv. Run .\setup.ps1 first."
    Read-Host "`n  Press Enter to exit"
    exit 1
}

$modelFiles = Get-ChildItem -Path $ModelsDir -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "\.(safetensors|ckpt|bin)$" }
if (-not $modelFiles) {
    Warn "Model weights missing in server\models\. Offline mode cannot start."
    Warn "Run .\setup.ps1 to download required weights."
    Read-Host "`n  Press Enter to exit"
    exit 1
}

# --- Check server running ---
$running = $false
try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:9876/health" -TimeoutSec 2 -ErrorAction Stop
    if ($r.StatusCode -eq 200) {
        $h = $r.Content | ConvertFrom-Json
        if ($h.status -eq "ok") { $running = $true }
    }
} catch {}

$serverProc = $null

if ($running) {
    Ok "Server already running"
} else {
    Write-Host "  ${D}Starting server...${R}"
    
    # Safe argument encoding avoiding quotes hell, leveraging purely uv orchestration
    $EncodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes(
        "`$Host.UI.RawUI.WindowTitle='SDDj Server'; Set-Location -LiteralPath `"$ServerDir`"; uv run run.py"
    ))
    
    $serverProc = Start-Process pwsh -ArgumentList "-NoExit", "-EncodedCommand", "$EncodedCommand" -WindowStyle Minimized -PassThru

    Write-Host "  ${D}Waiting for engine to load (~30s load + ~30s warmup)...${R}"

    $maxWait = 120; $waited = 0; $ready = $false
    while ($waited -lt $maxWait) {
        Start-Sleep -Seconds 3
        $waited += 3
        # Detect child process crash before timeout
        if ($null -ne $serverProc -and $serverProc.HasExited) {
            Warn "Server process exited (code $($serverProc.ExitCode)) before becoming ready"
            Warn "Check server\run.py output for errors"
            Read-Host "`n  Press Enter to exit"
            exit 1
        }
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:9876/health" -TimeoutSec 2 -ErrorAction Stop
            if ($null -ne $r -and $r.StatusCode -eq 200) {
                $h = $r.Content | ConvertFrom-Json
                if ($h.status -eq "ok" -and $h.loaded -eq $true) { $ready = $true; break }
            }
        } catch {}
        Write-Host "`r  ${D}Waiting... (${waited}s)${R}" -NoNewline
    }
    Write-Host ""
    if ($ready) { Ok "Server ready (${waited}s)" }
    else { Warn "Server not responding after ${maxWait}s - launching Aseprite anyway" }
}

# --- Launch Aseprite ---
Write-Host ""
$asePath = Join-Path -Path (Join-Path -Path $Root -ChildPath "bin\aseprite") -ChildPath "aseprite.exe"
if (Test-Path -Path $asePath) {
    Write-Host "  ${D}Launching Aseprite...${R}"
    $null = Start-Process -FilePath $asePath
    Ok "Aseprite launched"
} else {
    Warn "Aseprite not found at bin\aseprite"
    Write-Host "  ${D}Launch Aseprite manually - SDDj opens on startup.${R}"
}

Write-Host "`n  ${D}$('-' * 36)${R}`n  ${W}Server:${R}  ${C}ws://127.0.0.1:9876/ws${R}"
Write-Host "  ${W}Action:${R}  Connect in SDDj dialog`n  ${D}$('-' * 36)${R}`n"
Read-Host "  Press Enter to stop the server"

# --- Shutdown (SOTA Graceful + Surgical Kill) ---
Write-Host "`n  ${D}Stopping server...${R}"

$graceful = $false
try {
    # Send HTTP shutdown
    $null = Invoke-WebRequest -Uri "http://127.0.0.1:9876/shutdown" -Method POST -TimeoutSec 3 -ErrorAction SilentlyContinue
    
    # Wait 1s for server-side call_later(0.5) to fire
    Start-Sleep -Seconds 1

    # True exponential backoff: 1, 2, 4, 8
    $polls = 0; $delay = 1
    while ($polls -lt 5) {
        Start-Sleep -Seconds $delay
        $delay = [Math]::Min($delay * 2, 8)
        try {
            $null = Invoke-WebRequest -Uri "http://127.0.0.1:9876/health" -TimeoutSec 1 -ErrorAction Stop
        } catch {
            $graceful = $true
            break
        }
        $polls++
    }
} catch {}

if ($graceful) {
    Ok "Server stopped (graceful)"
} else {
    Warn "Graceful shutdown failed, initiating surgical process termination"
    if ($null -ne $serverProc -and -not $serverProc.HasExited) {
        taskkill /T /F /PID $serverProc.Id 2>$null | Out-Null
    } else {
        # Server was pre-existing (not started by us) — find by port
        $portPid = (Get-NetTCPConnection -LocalPort 9876 -State Listen -ErrorAction SilentlyContinue).OwningProcess | Select-Object -First 1
        if ($portPid) {
            taskkill /T /F /PID $portPid 2>$null | Out-Null
        }
    }
    Ok "Server stopped (fallback process tree kill)"
}
Write-Host ""
