#!/usr/bin/env pwsh
#Requires -Version 7.0

[CmdletBinding()]
param(
    [switch]$SkipModels
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSStyle.OutputRendering = "Ansi"

# --- UI ---
$e = [char]27
$R  = "$e[0m";  $D = "$e[90m"; $W = "$e[97m"; $B = "$e[1m"
$G  = "$e[92m"; $Re = "$e[91m"; $Y = "$e[93m"; $C = "$e[96m"
function Step($n, $total, $msg) { Write-Host "  ${D}[$n/$total]${R} $msg${D}...${R}" }
function Ok($msg)   { Write-Host "  ${G}OK${R}  $D$msg$R" }
function Fail($msg) { Write-Host "  ${Re}FAIL${R}  $msg"; Read-Host "`n  Press Enter to exit"; exit 1 }
function Warn($msg) { Write-Host "  ${Y}!${R}  $msg" }
try { $Host.UI.RawUI.WindowTitle = "SDDj - Setup" } catch {}

$Root = [System.IO.Path]::GetFullPath($PSScriptRoot)
Set-Location -Path $Root

Write-Host "`n  ${B}${W}SDDj${R}  ${D}Setup${R}`n  ${D}$('-' * 36)${R}`n"

# --- Paths ---
$ServerDir = Join-Path -Path $Root -ChildPath "server"
$ScriptsDir = Join-Path -Path $Root -ChildPath "scripts"
$VenvPython = Join-Path -Path $ServerDir -ChildPath ".venv\Scripts\python.exe"

# --- 1. Check uv ---
Step 1 7 "Checking uv package manager"
if (-not (Get-Command -Name "uv" -ErrorAction Ignore)) {
    Fail "uv not found - install from https://docs.astral.sh/uv/"
}
$uvVer = (uv --version 2>$null) -join ""
Ok $uvVer

# --- 2. Install dependencies ---
Step 2 7 "Installing dependencies"
Push-Location -Path $ServerDir
try {
    $null = uv sync --locked 2>&1
    if ($LASTEXITCODE -ne 0) {
        Warn "Lockfile mismatch, syncing with resolution..."
        $uvOut = uv sync 2>&1
        if ($LASTEXITCODE -ne 0) {
            Fail "Dependency install failed:`n$($uvOut | Out-String)"
        }
    }
} finally {
    Pop-Location
}
Ok "Dependencies installed"

# --- 3. Validate Env ---
if (-not (Test-Path -Path $VenvPython)) {
    Fail "Python executable not found in .venv"
}

# --- 4. Download models ---
Step 3 7 "Provisioning models"
if ($SkipModels) {
    Ok "Skipped (--SkipModels)"
} else {
    $dlScript = Join-Path -Path $ScriptsDir -ChildPath "download_models.py"
    # Execute natively to preserve \r buffer flushing and prevent console newline spam
    Write-Host ""
    & $VenvPython $dlScript --all
    Write-Host ""
    if ($LASTEXITCODE -ne 0) {
        Warn "Model download had errors. Re-run: $VenvPython $dlScript --all"
    } else {
        Ok "Models ready"
    }
}

# --- 5. Build extension ---
Step 4 7 "Building Aseprite extension"
$buildScript = Join-Path -Path $ScriptsDir -ChildPath "build_extension.py"
$buildOut = & $VenvPython $buildScript 2>&1
if ($LASTEXITCODE -ne 0) { Fail "Extension build failed:`n$($buildOut | Out-String)" }
Ok "Extension built"

# --- 6. Install extension ---
Step 5 7 "Deploying extension"
if (-not $env:APPDATA) { Fail "APPDATA not set (required for Aseprite extension deploy)" }
$AseData = Join-Path -Path $env:APPDATA -ChildPath "Aseprite"
$AseExt = Join-Path -Path $AseData -ChildPath "extensions\sddj"
$AseScripts = Join-Path -Path $AseData -ChildPath "scripts"

foreach ($f in @("sddj.lua", "json.lua")) {
    $p = Join-Path -Path $AseScripts -ChildPath $f
    if (Test-Path -Path $p) { Remove-Item -Path $p -Force }
}

if (Test-Path -Path $AseExt) { Remove-Item -Path $AseExt -Recurse -Force }
$null = New-Item -Path (Join-Path -Path $AseExt -ChildPath "scripts") -ItemType Directory -Force
$null = New-Item -Path (Join-Path -Path $AseExt -ChildPath "keys") -ItemType Directory -Force

$ExtSrc = Join-Path -Path $Root -ChildPath "extension"
Copy-Item -Path (Join-Path -Path $ExtSrc -ChildPath "package.json") -Destination $AseExt -Force
Copy-Item -Path (Join-Path -Path (Join-Path -Path $ExtSrc -ChildPath "scripts") -ChildPath "*.lua") -Destination (Join-Path -Path $AseExt -ChildPath "scripts") -Force
Copy-Item -Path (Join-Path -Path (Join-Path -Path $ExtSrc -ChildPath "keys") -ChildPath "*") -Destination (Join-Path -Path $AseExt -ChildPath "keys") -Force

$luaFiles = @(Get-ChildItem -Path (Join-Path -Path $AseExt -ChildPath "scripts") -Filter "*.lua" -ErrorAction SilentlyContinue)
$count = $luaFiles.Count
Ok "$count Lua files deployed"

# --- 7. Environment config ---
Step 6 7 "Checking environment config"
$EnvFile = Join-Path -Path $ServerDir -ChildPath ".env"
$EnvExample = Join-Path -Path $ServerDir -ChildPath ".env.example"

if (-not (Test-Path -Path $EnvFile)) {
    if (Test-Path -Path $EnvExample) {
        Copy-Item -Path $EnvExample -Destination $EnvFile -Force
        Ok "Created .env from template"
    } else {
        Ok "Using default settings"
    }
} else {
    Ok ".env exists - keeping config"
}

# --- Verify ---
Step 7 7 "Verifying installation"
try {
    $ver = & $VenvPython -c "import sddj; print(sddj.__version__)" 2>$null
    if ($LASTEXITCODE -eq 0) { Ok "SDDj v$ver" } else { Warn "Package import check failed" }
} catch {
    Warn "Package import check failed"
}

Write-Host "`n  ${D}$('-' * 36)${R}`n  ${G}${B}Setup complete.${R}`n"
Write-Host "  ${W}Next:${R}  Run ${C}.\start.ps1${R} to launch SDDj"
Write-Host "  ${W}Edit:${R}  ${D}server\.env${R} to customize settings`n"
Read-Host "  Press Enter to exit"
