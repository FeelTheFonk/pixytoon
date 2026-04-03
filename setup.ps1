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
$ServerDir  = Join-Path $Root "server"
$ScriptsDir = Join-Path $Root "scripts"
$VenvPython = Join-Path $ServerDir ".venv\Scripts\python.exe"

# --- 1. Check uv ---
Step 1 9 "Checking uv package manager"
if (-not (Get-Command -Name "uv" -ErrorAction Ignore)) {
    Fail "uv not found — install from https://docs.astral.sh/uv/"
}
$uvVer = (uv --version 2>$null) -join ""
Ok $uvVer

# --- 2. Install dependencies ---
Step 2 9 "Installing dependencies (Python 3.11, this may take a few minutes)"
Push-Location $ServerDir
try {
    # Stream output to console (no capture) so the user sees real-time progress.
    # --python 3.11 is belt-and-suspenders with .python-version.
    uv sync --locked --python 3.11
    if ($LASTEXITCODE -ne 0) {
        Warn "Locked sync failed — retrying with full resolution..."
        uv sync --python 3.11
        if ($LASTEXITCODE -ne 0) {
            Fail "Dependency install failed — run 'uv sync --python 3.11' in server/ for details"
        }
    }
} finally {
    Pop-Location
}
# Verify the venv was created with the expected Python version
$actualPy = (& $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null).Trim()
if ($actualPy -ne "3.11") {
    Fail "Expected Python 3.11 but venv has $actualPy — delete .venv and re-run"
}
Ok "Dependencies installed (Python $actualPy)"

# --- 3. Validate environment ---
Step 3 9 "Validating environment"
if (-not (Test-Path $VenvPython)) {
    Fail "Python executable not found in .venv"
}
Ok "Virtual environment valid"

# --- 4. SageAttention (optional, pre-built wheel) ---
Step 4 9 "SageAttention (optional)"
try {
    # Detect Python ABI tag (e.g. cp311)
    $pyTag = (& $VenvPython -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')" 2>$null).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $pyTag) { throw "Python version detection failed" }

    # Detect torch base version + CUDA build tag
    $torchInfo = (& $VenvPython -c @"
import torch
base = torch.__version__.split('+')[0]
cuda = torch.version.cuda or ''
cu = ('cu' + cuda.replace('.', '')) if cuda else ''
print(f'{cu} {base}')
"@ 2>$null).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $torchInfo) { throw "torch/CUDA detection failed" }

    $cuTag, $torchVer = $torchInfo -split ' ', 2
    if (-not $cuTag) { throw "No CUDA build detected" }

    $wheelGlob = "sageattention-*+${cuTag}torch${torchVer}-${pyTag}-${pyTag}-win_amd64.whl"
    $wheelsDir = Join-Path $ServerDir "wheels"
    if (-not (Test-Path $wheelsDir)) { $null = New-Item -Path $wheelsDir -ItemType Directory -Force }

    # Check cached wheels, then repo root (manual download)
    $wheel = Get-ChildItem -Path $wheelsDir -Filter $wheelGlob -ErrorAction SilentlyContinue |
             Select-Object -First 1
    if (-not $wheel) {
        $rootWheel = Get-ChildItem -Path $Root -Filter $wheelGlob -ErrorAction SilentlyContinue |
                     Select-Object -First 1
        if ($rootWheel) {
            Move-Item $rootWheel.FullName (Join-Path $wheelsDir $rootWheel.Name) -Force
            $wheel = Get-Item (Join-Path $wheelsDir $rootWheel.Name)
        }
    }

    if (-not $wheel) {
        # Query GitHub releases for a matching wheel (anonymous: 60 req/h limit)
        try {
            $releases = Invoke-RestMethod `
                -Uri "https://api.github.com/repos/sdbds/SageAttention-for-windows/releases" `
                -Headers @{ "User-Agent" = "SDDj-Setup"; "Accept" = "application/vnd.github+json" } `
                -ErrorAction Stop
        } catch {
            $status = $_.Exception.Response.StatusCode.value__
            if ($status -eq 403 -or $status -eq 429) {
                throw "GitHub API rate limit hit — try again later or place wheel in server/wheels/"
            }
            throw "GitHub API error ($status): $($_.Exception.Message)"
        }

        $dlUrl = $null; $dlName = $null
        foreach ($rel in $releases) {
            foreach ($a in $rel.assets) {
                if ($a.name -match "^sageattention-[\d\.]+\+${cuTag}torch${torchVer}-${pyTag}-${pyTag}-win_amd64\.whl$") {
                    $dlUrl = $a.browser_download_url; $dlName = $a.name; break
                }
            }
            if ($dlUrl) { break }
        }
        if (-not $dlUrl) { throw "No wheel for ${pyTag} + torch ${torchVer} + ${cuTag}" }

        Warn "Downloading $dlName..."
        Invoke-WebRequest -Uri $dlUrl -OutFile (Join-Path $wheelsDir $dlName) -UseBasicParsing -ErrorAction Stop
        $wheel = Get-Item (Join-Path $wheelsDir $dlName)
    }

    # --no-build-isolation avoids re-resolving; explicit --python targets the managed venv
    uv pip install --python $VenvPython --no-build-isolation $wheel.FullName
    if ($LASTEXITCODE -ne 0) { throw "Wheel install failed — run the command above manually for details" }
    Ok "SageAttention $(($wheel.Name -split '-')[1]) ($cuTag, torch $torchVer)"
} catch {
    Warn "SageAttention skipped — SDP fallback ($($_.Exception.Message))"
}

# --- 5. Download models ---
Step 5 9 "Provisioning models"
if ($SkipModels) {
    Ok "Skipped (-SkipModels)"
} else {
    $dlScript = Join-Path $ScriptsDir "download_models.py"
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

# --- 6. Build extension ---
Step 6 9 "Building Aseprite extension"
$buildScript = Join-Path $ScriptsDir "build_extension.py"
$buildOut = & $VenvPython $buildScript 2>&1
if ($LASTEXITCODE -ne 0) { Fail "Extension build failed:`n$($buildOut | Out-String)" }
Ok "Extension built"

# --- 7. Deploy extension ---
Step 7 9 "Deploying extension"
if (-not $env:APPDATA) { Fail "APPDATA not set (required for Aseprite extension deploy)" }
$AseData    = Join-Path $env:APPDATA "Aseprite"
$AseExt     = Join-Path $AseData "extensions\sddj"
$AseScripts = Join-Path $AseData "scripts"

# Remove legacy flat scripts (pre-extension era)
foreach ($f in @("sddj.lua", "json.lua")) {
    $p = Join-Path $AseScripts $f
    if (Test-Path $p) { Remove-Item $p -Force }
}

# Remove old extension (guarded: Aseprite file locks cause access-denied)
if (Test-Path $AseExt) {
    try {
        Remove-Item $AseExt -Recurse -Force
    } catch {
        Fail "Cannot remove old extension — close Aseprite first"
    }
}

$null = New-Item -Path (Join-Path $AseExt "scripts") -ItemType Directory -Force
$null = New-Item -Path (Join-Path $AseExt "keys") -ItemType Directory -Force

$ExtSrc = Join-Path $Root "extension"
Copy-Item (Join-Path $ExtSrc "package.json") $AseExt -Force
Copy-Item (Join-Path $ExtSrc "scripts\*.lua") (Join-Path $AseExt "scripts") -Force
Copy-Item (Join-Path $ExtSrc "keys\*") (Join-Path $AseExt "keys") -Force

$luaFiles = @(Get-ChildItem -Path (Join-Path $AseExt "scripts") -Filter "*.lua" -ErrorAction SilentlyContinue)
if ($luaFiles.Count -eq 0) { Fail "No Lua files deployed — extension source may be missing" }
Ok "$($luaFiles.Count) Lua files deployed"

# --- 8. Environment config ---
Step 8 9 "Checking environment config"
$EnvFile    = Join-Path $ServerDir ".env"
$EnvExample = Join-Path $ServerDir ".env.example"

if (-not (Test-Path $EnvFile)) {
    if (Test-Path $EnvExample) {
        Copy-Item $EnvExample $EnvFile -Force
        Ok "Created .env from template"
    } else {
        Ok "Using default settings"
    }
} else {
    Ok ".env exists — keeping config"
}

# --- 9. Verify installation ---
Step 9 9 "Verifying installation"
try {
    # Metadata-only check (avoids torch/CUDA import chain, ~50ms vs ~500ms)
    $ver = & $VenvPython -c "from importlib.metadata import version; print(version('sddj-server'))" 2>$null
    if ($LASTEXITCODE -eq 0 -and $ver) { Ok "SDDj v$ver" } else { Warn "Package verification failed" }
} catch {
    Warn "Package verification failed"
}

Write-Host "`n  ${D}$('-' * 36)${R}`n  ${G}${B}Setup complete.${R}`n"
Write-Host "  ${W}Next:${R}  Run ${C}.\start.ps1${R} to launch SDDj"
Write-Host "  ${W}Edit:${R}  ${D}server\.env${R} to customize settings`n"
Read-Host "  Press Enter to exit"
