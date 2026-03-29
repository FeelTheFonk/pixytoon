#!/usr/bin/env pwsh
#Requires -Version 7.0

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true, Position=0)]
    [ValidatePattern('^\d+\.\d+\.\d+(-\w+)?$')]
    [string]$Version
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = [System.IO.Path]::GetFullPath($PSScriptRoot)

# ── Validate all targets exist before modifying any ──
$PyProject = Join-Path -Path (Join-Path -Path $Root -ChildPath "server") -ChildPath "pyproject.toml"
$PkgJson   = Join-Path -Path (Join-Path -Path $Root -ChildPath "extension") -ChildPath "package.json"
$ScriptsDir = Join-Path -Path (Join-Path -Path $Root -ChildPath "extension") -ChildPath "scripts"
$Lua       = Join-Path -Path $ScriptsDir -ChildPath "sddj_state.lua"
$Readme    = Join-Path -Path $Root -ChildPath "README.md"

foreach ($f in @($PyProject, $PkgJson, $Lua, $Readme)) {
    if (-not (Test-Path -Path $f)) {
        Write-Host "[FAIL] Required file not found: $f" -ForegroundColor Red
        exit 1
    }
}

# ── 1. pyproject.toml ──
$Content = Get-Content -Path $PyProject -Raw
$Content = $Content -replace '(?m)^version\s*=\s*"[^"]+"', "version = ""$Version"""
Set-Content -Path $PyProject -Value $Content -NoNewline
Write-Host "[OK] pyproject.toml -> $Version" -ForegroundColor Green

# ── 2. package.json ──
$Content = Get-Content -Path $PkgJson -Raw
# Strict multiline Lookaround replacement keeping exact spacing and trailing characters
$Content = $Content -replace '(?m)^(\s*"version"\s*:\s*)"[^"]+"(.*)$', "`$1""$Version""`$2"
Set-Content -Path $PkgJson -Value $Content -NoNewline
Write-Host "[OK] package.json   -> $Version" -ForegroundColor Green

# ── 3. sddj_state.lua ──
$Content = Get-Content -Path $Lua -Raw
$Content = $Content -replace 'PT\.VERSION\s*=\s*"[^"]+"', "PT.VERSION = ""$Version"""
Set-Content -Path $Lua -Value $Content -NoNewline
Write-Host "[OK] sddj_state.lua -> $Version" -ForegroundColor Green

# ── 4. README.md ──
$Content = Get-Content -Path $Readme -Raw
$Content = $Content -replace '(?m)^(# SDDj v)\S+', "`${1}$Version"
Set-Content -Path $Readme -Value $Content -NoNewline
Write-Host "[OK] README.md       -> $Version" -ForegroundColor Green

# ── 5. uv.lock ──
Push-Location -Path (Join-Path -Path $Root -ChildPath "server")
try {
    $lockOut = uv lock --quiet 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] uv lock failed: $($lockOut | Out-String)" -ForegroundColor Yellow
    } else {
        Write-Host "[OK] uv.lock        -> synced" -ForegroundColor Green
    }
} catch {
    Write-Host "[WARN] uv lock failed (run manually)" -ForegroundColor Yellow
} finally {
    Pop-Location
}

# ── 6. Post-bump verification ──
$verifyPy  = (Get-Content -Path $PyProject -Raw) -match "version\s*=\s*`"$([regex]::Escape($Version))`""
$verifyPkg = (Get-Content -Path $PkgJson -Raw)   -match "`"version`"\s*:\s*`"$([regex]::Escape($Version))`""
$verifyLua = (Get-Content -Path $Lua -Raw)        -match "PT\.VERSION\s*=\s*`"$([regex]::Escape($Version))`""

if (-not ($verifyPy -and $verifyPkg -and $verifyLua)) {
    Write-Host "[WARN] Version replacement may have failed in one or more files. Verify manually." -ForegroundColor Yellow
}

Write-Host "`nVersion bumped to $Version across all files." -ForegroundColor Cyan
Write-Host "Next: update CHANGELOG.md, then: git add -A && git commit -m 'v$Version' && git tag v$Version && git push --follow-tags"
