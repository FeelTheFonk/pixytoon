#!/usr/bin/env pwsh
#Requires -Version 7.0

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "SDDj"
$PSStyle.OutputRendering = "Ansi"

# --- Helpers ---
$e = [char]27
$R  = "$e[0m";  $D = "$e[90m";  $W = "$e[97m"; $B = "$e[1m"
$G  = "$e[92m"; $Y = "$e[93m";  $C = "$e[96m"; $Re = "$e[91m"

function Ok($msg)   { Write-Host "  ${G}OK${R}  $D$msg$R" }
function Warn($msg) { Write-Host "  ${Y}!${R}  $msg" }

$Root = [System.IO.Path]::GetFullPath($PSScriptRoot)
$ServerDir = Join-Path -Path $Root -ChildPath "server"
$ModelsDir = Join-Path -Path $ServerDir -ChildPath "models"

# --- Force offline mode ---
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
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
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:9876/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
    if ($null -ne $r -and $r.StatusCode -eq 200) { $running = $true }
} catch {}

$serverProc = $null

if ($running) {
    Ok "Server already running"
} else {
    Write-Host "  ${D}Starting server...${R}"
    
    # Safe argument encoding avoiding quotes hell, leveraging purely uv orchestration
    $EncodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes(
        "`$Host.UI.RawUI.WindowTitle='SDDj Server'; Set-Location -LiteralPath `"$ServerDir`"; uv run --frozen run.py"
    ))
    
    $serverProc = Start-Process pwsh -ArgumentList "-NoExit", "-EncodedCommand", "$EncodedCommand" -WindowStyle Minimized -PassThru

    Write-Host "  ${D}Waiting for engine to load (~30s load + ~30s warmup)...${R}"

    $maxWait = 120; $waited = 0; $ready = $false
    while ($waited -lt $maxWait) {
        Start-Sleep -Seconds 3
        $waited += 3
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:9876/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
            if ($null -ne $r -and $r.Content -match "true") { $ready = $true; break }
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
    
    # Polling exponential decay for shutdown validation
    $polls = 0
    while ($polls -lt 6) {
        Start-Sleep -Seconds 2
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
        # SOTA surgical fallback: kill exactly the root PID and its children
        taskkill /T /F /PID $serverProc.Id 2>$null | Out-Null
    }
    Ok "Server stopped (fallback process tree kill)"
}
Write-Host ""
