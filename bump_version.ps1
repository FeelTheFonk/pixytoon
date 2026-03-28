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

# ── 1. pyproject.toml ──
$PyProject = Join-Path -Path (Join-Path -Path $Root -ChildPath "server") -ChildPath "pyproject.toml"
$Content = Get-Content -Path $PyProject -Raw
$Content = $Content -replace '(?m)^version\s*=\s*"[^"]+"', "version = ""$Version"""
Set-Content -Path $PyProject -Value $Content -NoNewline
Write-Host "[OK] pyproject.toml -> $Version" -ForegroundColor Green

# ── 2. package.json ──
$PkgJson = Join-Path -Path (Join-Path -Path $Root -ChildPath "extension") -ChildPath "package.json"
$Content = Get-Content -Path $PkgJson -Raw
# Strict multiline Lookaround replacement keeping exact spacing and trailing characters
$Content = $Content -replace '(?m)^(\s*"version"\s*:\s*)"[^"]+"(.*)$', "`$1""$Version""`$2"
Set-Content -Path $PkgJson -Value $Content -NoNewline
Write-Host "[OK] package.json   -> $Version" -ForegroundColor Green

# ── 3. sddj_state.lua ──
$ScriptsDir = Join-Path -Path (Join-Path -Path $Root -ChildPath "extension") -ChildPath "scripts"
$Lua = Join-Path -Path $ScriptsDir -ChildPath "sddj_state.lua"
$Content = Get-Content -Path $Lua -Raw
$Content = $Content -replace 'PT\.VERSION\s*=\s*"[^"]+"', "PT.VERSION = ""$Version"""
Set-Content -Path $Lua -Value $Content -NoNewline
Write-Host "[OK] sddj_state.lua -> $Version" -ForegroundColor Green

# ── 4. README.md ──
$Readme = Join-Path -Path $Root -ChildPath "README.md"
$Content = Get-Content -Path $Readme -Raw
$Content = $Content -replace '(?m)^(# SDDj v)\S+', "`${1}$Version"
Set-Content -Path $Readme -Value $Content -NoNewline
Write-Host "[OK] README.md       -> $Version" -ForegroundColor Green

# ── 5. uv.lock ──
Push-Location -Path (Join-Path -Path $Root -ChildPath "server")
try {
    uv lock --quiet 2>$null
    Write-Host "[OK] uv.lock        -> synced" -ForegroundColor Green
} catch {
    Write-Host "[WARN] uv lock failed (run manually)" -ForegroundColor Yellow
} finally {
    Pop-Location
}

Write-Host "`nVersion bumped to $Version across all files." -ForegroundColor Cyan
Write-Host "Next: update CHANGELOG.md, then: git add -A && git commit -m 'v$Version' && git tag v$Version && git push --follow-tags"
