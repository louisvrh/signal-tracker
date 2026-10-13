<#
.SYNOPSIS
    Start the Signal Tracker dashboard and open it in the browser.

.DESCRIPTION
    Runs `streamlit run dashboard\app.py` with the project's .venv Python from
    the project folder, so .streamlit\config.toml applies (127.0.0.1 only,
    port 8551, no first-run prompt). Streamlit opens the default browser at
    http://localhost:8551 once the server is up. Press Ctrl+C in this window to
    stop it. The dashboard only reads data\; it never writes (presets aside).

    If something already serves the port (normally the "SignalTracker
    Dashboard" scheduled task, scripts\register_dashboard_task.ps1), this just
    opens it in the browser.

    To run it:
        powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_dashboard.ps1

.PARAMETER Port
    Port to serve on (default 8551).

.PARAMETER NoBrowser
    Don't open a browser window (the dashboard is still served).
#>
[CmdletBinding()]
param(
    [int]$Port = 8551,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) {
    throw "Python not found at $Python. Create the virtual environment first (docs/SETUP.md)."
}

$Url = "http://localhost:$Port"
if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) {
    Write-Host "Something already serves port $Port (normally the 'SignalTracker Dashboard' task): opening $Url"
    if (-not $NoBrowser) { Start-Process $Url }
    exit 0
}

Set-Location $Root
$env:PYTHONUTF8 = '1'
$Headless = if ($NoBrowser) { 'true' } else { 'false' }
Write-Host "Starting the dashboard at $Url (Ctrl+C to stop)..."
& $Python -m streamlit run (Join-Path $Root 'dashboard\app.py') --server.address 127.0.0.1 --server.port $Port --server.headless $Headless
exit $LASTEXITCODE
