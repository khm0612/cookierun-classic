$ErrorActionPreference = "Stop"

$Root = $PSScriptRoot
if (-not $Root) {
    $Root = (Get-Location).Path
}

$VenvDir = Join-Path $Root ".venv"
$Python = Join-Path $VenvDir "Scripts\python.exe"

if (-not (Test-Path $Python)) {
    python -m venv $VenvDir
}

& $Python -m pip install -r (Join-Path $Root "requirements.txt")
& $Python -m pip install --editable $Root --no-deps

# scrcpy-client bundles the scrcpy server jar, but its declared adbutils pin is stale.
# The bot uses adbutils 2.x directly, so install scrcpy-client without its old dependency.
& $Python -m pip install --no-deps "scrcpy-client>=0.4.0"

Write-Host "Installed. Launch with .\CookieGame.bat"
