<#
.SYNOPSIS
    Keep the Signal Tracker dashboard running: start Streamlit, restart it if it stops.

.DESCRIPTION
    Run by the scheduled task "SignalTracker Dashboard" (scripts\register_dashboard_task.ps1)
    at logon, in a hidden window. It starts the dashboard with the project's .venv
    Python, bound to 127.0.0.1 only (.streamlit\config.toml), headless (no browser),
    and restarts it whenever it exits: after 10 s, or after a longer wait (up to
    5 minutes) if it keeps stopping within a minute of starting. It never stops or
    replaces another program: if the port is taken by something else it waits and
    checks again. When it starts, it first stops a Streamlit left over from an
    earlier run of this script (one whose command line names this project's
    dashboard\app.py; stopping the task doesn't stop Streamlit). Everything goes
    to logs\dashboard.log.

.PARAMETER Port
    Local port (default 8551, not Streamlit's default 8501).
#>
[CmdletBinding()]
param(
    [int]$Port = 8551
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$App = Join-Path $Root 'dashboard\app.py'
$LogDir = Join-Path $Root 'logs'
$Log = Join-Path $LogDir 'dashboard.log'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-Log([string]$Message) {
    $line = '{0:yyyy-MM-dd HH:mm:ss} {1}' -f (Get-Date), $Message
    Add-Content -Path $Log -Value $line -Encoding UTF8
}

function Test-PortInUse([int]$P) {
    [bool](Get-NetTCPConnection -State Listen -LocalPort $P -ErrorAction SilentlyContinue)
}

function Stop-LeftoverDashboard {
    # Stopping the scheduled task ends this script but not the Streamlit it started.
    # Clear such leftovers: only processes whose command line names this project's app.py.
    Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -and $_.CommandLine.Contains($App) -and $_.CommandLine.Contains('streamlit') } |
        ForEach-Object {
            Write-Log "stopping a leftover dashboard process $($_.ProcessId)"
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        }
}

Set-Location $Root
$env:PYTHONUTF8 = '1'  # Python writes UTF-8 to the pipe...
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }  # ...and PowerShell reads UTF-8
$delay = 10
Write-Log "service started (pid $PID), port $Port"
Stop-LeftoverDashboard
Start-Sleep -Seconds 2
while ($true) {
    if ((Test-Path $Log) -and (Get-Item $Log).Length -gt 5MB) {
        Move-Item -Force $Log "$Log.1"
    }
    if (Test-PortInUse $Port) {
        Write-Log "port $Port is already in use by another program; checking again in 60 s"
        Start-Sleep -Seconds 60
        continue
    }
    $started = Get-Date
    Write-Log "starting streamlit on 127.0.0.1:$Port"
    $ErrorActionPreference = 'Continue'  # in PowerShell 5.1 a stderr line would otherwise stop the script
    try {
        & $Python -m streamlit run $App --server.address 127.0.0.1 --server.port $Port --server.headless true 2>&1 |
            ForEach-Object { Write-Log ('    ' + $_.ToString()) }
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = 'Stop'
    }
    $ran = [int]((Get-Date) - $started).TotalSeconds
    $delay = if ($ran -lt 60) { [Math]::Min($delay * 2, 300) } else { 10 }
    Write-Log "streamlit exited with code $code after $ran s; restarting in $delay s"
    Start-Sleep -Seconds $delay
}
