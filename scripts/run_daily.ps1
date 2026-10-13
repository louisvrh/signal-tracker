<#
.SYNOPSIS
    Daily pipeline run: dashboard check, fetch, quality checks, STL precompute, signals, backup, Telegram digest.

.DESCRIPTION
    Runs each step with the project's .venv Python and appends all output to
    logs\run_daily.log (moved to run_daily.log.old above 5 MB). Every step
    runs even if an earlier one failed (fetch already isolates each source),
    and the script exits with 1 if any step failed, so Task Scheduler shows
    the run as failed.

    It starts with scripts\check_dashboard.ps1: if nothing is listening on the
    dashboard's port, that restarts the "SignalTracker Dashboard" task.

    It ends with the Telegram digest, which always sends a message: "all
    quiet", the day's news, or, if any step failed, a failure message naming
    the steps (python -m tracker digest --failed-steps ...). A step that throws
    a PowerShell error counts as failed, so the digest still goes out.

    Registered as a daily task by scripts\register_task.ps1. To run it by hand:
        powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_daily.ps1

.PARAMETER SkipBackup
    Don't run scripts\backup.ps1.
.PARAMETER SkipDigest
    Don't send the Telegram digest.
.PARAMETER DashboardPort
    The dashboard's local port (default 8551, as in scripts\dashboard_service.ps1).
#>
[CmdletBinding()]
param(
    [switch]$SkipBackup,
    [switch]$SkipDigest,
    [int]$DashboardPort = 8551
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$LogDir = Join-Path $Root 'logs'
$Log = Join-Path $LogDir 'run_daily.log'

# Steps in order (the dashboard check comes first, backup and the digest last).
# decompose precomputes STL seasonal components for the dashboard (data\signals\stl.parquet, ~1 minute).
# signals computes composites, divergences and flags (data\signals\signals.parquet, flags.parquet, ~15 s).
$Steps = @(
    @{ Name = 'fetch';     Arguments = @('fetch') },
    @{ Name = 'quality';   Arguments = @('quality') },
    @{ Name = 'decompose'; Arguments = @('decompose') },
    @{ Name = 'signals';   Arguments = @('signals') }
)

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
if ((Test-Path $Log) -and ((Get-Item $Log).Length -gt 5MB)) {
    Move-Item -Force -Path $Log -Destination "$Log.old"
}
Set-Location $Root
$env:PYTHONUTF8 = '1'  # Python writes UTF-8 to the pipe...
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }  # ...and PowerShell reads UTF-8

function Write-Log([string]$Message) {
    $line = '{0:yyyy-MM-dd HH:mm:ss} {1}' -f (Get-Date), $Message
    Add-Content -Path $Log -Value $line -Encoding UTF8
    Write-Host $line
}

$script:Output = @()

function Invoke-Logged([string]$Name, [string]$Exe, [string[]]$Arguments) {
    # Runs a program, copies its stdout and stderr into the log (and $script:Output), returns its exit code.
    # A PowerShell error while doing so is logged and counts as exit code 1.
    Write-Log ('--- {0}: {1} {2}' -f $Name, (Split-Path -Leaf $Exe), ($Arguments -join ' '))
    $script:Output = @()
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'  # in PowerShell 5.1 a stderr line would otherwise stop the script
    try {
        & $Exe @Arguments 2>&1 | ForEach-Object {
            $text = $_.ToString()
            $script:Output += $text
            Write-Log ('    ' + $text)
        }
        $code = $LASTEXITCODE
    } catch {
        Write-Log ('    PowerShell error: {0}' -f $_.Exception.Message)
        $code = 1
    } finally {
        $ErrorActionPreference = $previous
    }
    Write-Log ('--- {0}: exit code {1}' -f $Name, $code)
    return $code
}

Write-Log "=== run_daily start (user $env:USERNAME, pid $PID)"
if (-not (Test-Path $Python)) {
    Write-Log "ERROR: $Python not found; create the virtual environment first (docs/SETUP.md). No digest can be sent."
    exit 1
}

$failed = @()
$check = Join-Path $PSScriptRoot 'check_dashboard.ps1'
$code = Invoke-Logged 'dashboard' 'powershell.exe' @('-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', $check, '-Port', $DashboardPort)
$dashboard = 'unknown'
$stateLine = $script:Output | Where-Object { $_ -match '^dashboard: (\S+)' } | Select-Object -Last 1
if ($stateLine -match '^dashboard: (\S+)') { $dashboard = $Matches[1] }
if ($code -ne 0) { $failed += 'dashboard' }

foreach ($step in $Steps) {
    $code = Invoke-Logged $step.Name $Python (@('-m', 'tracker') + $step.Arguments)
    if ($code -ne 0) { $failed += $step.Name }
}

if ($SkipBackup) {
    Write-Log '--- backup: skipped (-SkipBackup)'
} else {
    $backup = Join-Path $PSScriptRoot 'backup.ps1'
    $code = Invoke-Logged 'backup' 'powershell.exe' @('-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', $backup)
    if ($code -ne 0) { $failed += 'backup' }
}

if ($SkipDigest) {
    Write-Log '--- digest: skipped (-SkipDigest)'
} else {
    $digestArgs = @('-m', 'tracker', 'digest', '--daily-run', '--dashboard', $dashboard)
    if ($failed.Count -gt 0) { $digestArgs += @('--failed-steps', ($failed -join ',')) }
    $code = Invoke-Logged 'digest' $Python $digestArgs
    if ($code -ne 0) { $failed += 'digest' }
}

if ($failed.Count -gt 0) {
    Write-Log ('=== run_daily finished with failures: {0} (details above and in logs\tracker.log)' -f ($failed -join ', '))
    exit 1
}
Write-Log '=== run_daily finished OK'
exit 0
