$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPath = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvPath "Scripts\\python.exe"
$PipExe = Join-Path $VenvPath "Scripts\\pip.exe"
$RequirementsPath = Join-Path $ProjectRoot "requirements.txt"
$StampPath = Join-Path $VenvPath "requirements.sha256"

function Get-RequirementsHash {
    return (Get-FileHash -Algorithm SHA256 -Path $RequirementsPath).Hash
}

if (-not (Test-Path $PythonExe)) {
    Write-Host "Creating virtual environment at .venv ..."
    python -m venv $VenvPath
}

$CurrentHash = Get-RequirementsHash
$SavedHash = if (Test-Path $StampPath) { (Get-Content $StampPath -Raw).Trim() } else { "" }

if ($CurrentHash -ne $SavedHash) {
    Write-Host "Installing project dependencies into .venv ..."
    & $PythonExe -m pip install --upgrade pip
    & $PipExe install -r $RequirementsPath
    Set-Content -Path $StampPath -Value $CurrentHash -Encoding ascii
} else {
    Write-Host "Dependencies already match requirements.txt"
}

Write-Host "Starting Flask app on http://127.0.0.1:5000"
& $PythonExe (Join-Path $ProjectRoot "app.py")
