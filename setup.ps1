#!/usr/bin/env pwsh
#Requires -Version 7.0
$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "SDDj - Setup"
$PSStyle.OutputRendering = "Ansi"

# --- Helpers -----------------------------------------------------------------
$e = [char]27
$R  = "$e[0m";  $D = "$e[90m";  $W = "$e[97m"; $B = "$e[1m"
$G  = "$e[92m"; $Re = "$e[91m"; $Y = "$e[93m"; $C = "$e[96m"

function Step($n, $total, $msg) { Write-Host "  ${D}[$n/$total]${R} $msg${D}...${R}" }
function Ok($msg)   { Write-Host "  ${G}OK${R}  $D$msg$R" }
function Fail($msg) { Write-Host "  ${Re}FAIL${R}  $msg"; Read-Host "`n  Press Enter to exit"; exit 1 }
function Warn($msg) { Write-Host "  ${Y}!${R}  $msg" }

$Root = $PSScriptRoot
Set-Location $Root

Write-Host ""
Write-Host "  ${B}${W}SDDj${R}  ${D}Setup${R}"
Write-Host "  ${D}$('-' * 36)${R}"
Write-Host ""

# --- 1. Check uv ------------------------------------------------------------
Step 1 6 "Checking uv package manager"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Fail "uv not found - install from https://docs.astral.sh/uv/"
}
$uvVer = (uv --version 2>$null) -join ""
Ok $uvVer

# --- 2. Install dependencies ------------------------------------------------
Step 2 6 "Installing dependencies (torch CUDA + triton)"
Push-Location "$Root/server"
uv venv 2>$null | Out-Null
$null = uv pip install -e . 2>&1
if ($LASTEXITCODE -ne 0) {
    Warn "Retrying with verbose output..."
    uv pip install -e .
    if ($LASTEXITCODE -ne 0) { Pop-Location; Fail "Dependency install failed" }
}
Pop-Location
Ok "torch, diffusers, triton installed"

# --- 3. Download models -----------------------------------------------------
Step 3 6 "Checking / downloading models (~10 GB first time)"

# If --skip-models flag was passed, skip download
$skipModels = $args -contains "--skip-models"
if ($skipModels) {
    Ok "Skipped (--skip-models)"
} else {
    Push-Location "$Root/server"
    uv run python "$Root/scripts/download_models.py" --all
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        Warn "Model download had errors — you can retry later or place models manually"
        Warn "Re-run: uv run python scripts/download_models.py --all"
    } else {
        Pop-Location
        Ok "Models ready"
    }
}

# --- 4. Build extension ------------------------------------------------------
Step 4 6 "Building Aseprite extension"
Push-Location "$Root/server"
$null = uv run python "$Root/scripts/build_extension.py" 2>&1
if ($LASTEXITCODE -ne 0) { Pop-Location; Fail "Extension build failed" }
Pop-Location
Ok "Extension built"

# --- 5. Install extension ----------------------------------------------------
Step 5 6 "Installing extension into Aseprite"
$aseExt = "$env:APPDATA/Aseprite/extensions/sddj"
$aseScripts = "$env:APPDATA/Aseprite/scripts"

# Clean stale global scripts (current + legacy)
foreach ($f in "sddj.lua", "pixytoon.lua", "json.lua") {
    $p = Join-Path $aseScripts $f
    if (Test-Path $p) { Remove-Item $p -Force }
}

# Remove legacy extension (pre-0.7.5 rename)
$legacyExt = "$env:APPDATA/Aseprite/extensions/pixytoon"
if (Test-Path $legacyExt) { Remove-Item $legacyExt -Recurse -Force }

# Deploy extension
if (Test-Path $aseExt) { Remove-Item $aseExt -Recurse -Force }
New-Item "$aseExt/scripts" -ItemType Directory -Force | Out-Null
New-Item "$aseExt/keys" -ItemType Directory -Force | Out-Null
Copy-Item "$Root/extension/package.json" "$aseExt/" -Force
Copy-Item "$Root/extension/scripts/*.lua" "$aseExt/scripts/" -Force
Copy-Item "$Root/extension/keys/*" "$aseExt/keys/" -Force

$count = (Get-ChildItem "$aseExt/scripts/*.lua").Count
Ok "$count Lua files + keys -> $aseExt"

# --- 6. Environment config ---------------------------------------------------
Step 6 6 "Checking environment config"
if (-not (Test-Path "$Root/server/.env")) {
    if (Test-Path "$Root/server/.env.example") {
        Copy-Item "$Root/server/.env.example" "$Root/server/.env" -Force
        Ok "Created .env from template"
    } else { Ok "Using default settings" }
} else { Ok ".env exists - keeping current config" }

# --- Verify ------------------------------------------------------------------
Write-Host ""
Write-Host "  ${D}Verifying...${R}"
Push-Location "$Root/server"
try {
    $ver = uv run python -c "import sddj; print(sddj.__version__)" 2>$null
    Ok "SDDj v$ver"
} catch { Warn "Package import check failed" }
Pop-Location

# --- Done --------------------------------------------------------------------
Write-Host ""
Write-Host "  ${D}$('-' * 36)${R}"
Write-Host "  ${G}${B}Setup complete.${R}"
Write-Host ""
Write-Host "  ${W}Next:${R}  Run ${C}./start.ps1${R} to launch SDDj"
Write-Host "  ${W}Edit:${R}  ${D}server/.env${R} to customize settings"
Write-Host ""
Write-Host "  ${D}Note: torch.compile requires VS 2022 C++ workload${R}"
Write-Host ""
Read-Host "  Press Enter to exit"
