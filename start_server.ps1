$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$venvActivate = Join-Path $root 'venv\Scripts\Activate.ps1'
$frontendDir = Join-Path $root 'frontend'

Write-Host 'Starting backend API on http://localhost:8000...'
Start-Process powershell -ArgumentList @(
    '-NoExit',
    '-Command',
    "& '$venvActivate'; Set-Location '$root'; uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload"
) | Out-Null

Write-Host 'Starting frontend app on http://localhost/login...'
Start-Process powershell -ArgumentList @(
    '-NoExit',
    '-Command',
    "Set-Location '$frontendDir'; npm run dev -- --host 0.0.0.0 --port 80"
) | Out-Null

Write-Host ''
Write-Host 'Open the app here:'
Write-Host '  http://localhost/'
Write-Host 'It will redirect to the login screen first.'
Write-Host 'API is available at:'
Write-Host '  http://localhost:8000/api/v1/health'