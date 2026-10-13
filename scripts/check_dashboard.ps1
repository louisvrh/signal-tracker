<#
.SYNOPSIS
    Check that the dashboard is listening on its port; if not, restart its scheduled task.

.DESCRIPTION
    Run by scripts\run_daily.ps1 as its first step (its output goes to logs\run_daily.log).
    If nothing listens on the dashboard's port, it starts the scheduled task
    "SignalTracker Dashboard" (scripts\register_dashboard_task.ps1), stopping it
    first if Task Scheduler still shows it running, and waits up to a minute for
    the port. A disabled task (register_dashboard_task.ps1 -Stop) is left alone.
    It never touches another program or another scheduled task.

    The last line of output is "dashboard: <state>", where state is up, restarted,
    off or DOWN. Exit code 0 for up, restarted and off; 1 for DOWN.

    By hand:
        powershell -NoProfile -ExecutionPolicy Bypass -File scripts\check_dashboard.ps1

.PARAMETER Port
    The dashboard's local port (default 8551, as in scripts\dashboard_service.ps1).
.PARAMETER TaskName
    The dashboard's scheduled task (default "SignalTracker Dashboard").
.PARAMETER WaitSeconds
    How long to wait for the port after starting the task (default 60).
#>
[CmdletBinding()]
param(
    [int]$Port = 8551,
    [string]$TaskName = 'SignalTracker Dashboard',
    [int]$WaitSeconds = 60
)

$ErrorActionPreference = 'Stop'

function Test-Listening([int]$P) {
    [bool](Get-NetTCPConnection -State Listen -LocalPort $P -ErrorAction SilentlyContinue)
}

if (Test-Listening $Port) {
    Write-Output "dashboard: up (listening on port $Port)"
    exit 0
}
$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if (-not $task) {
    Write-Output "nothing listens on port $Port and there is no task '$TaskName' (scripts\register_dashboard_task.ps1)"
    Write-Output "dashboard: DOWN"
    exit 1
}
if ($task.State -eq 'Disabled') {
    Write-Output "task '$TaskName' is disabled (register_dashboard_task.ps1 -Start turns it on again)"
    Write-Output "dashboard: off"
    exit 0
}
Write-Output "nothing listens on port $Port (task state $($task.State)); restarting '$TaskName'"
try {
    if ($task.State -eq 'Running') { Stop-ScheduledTask -TaskName $TaskName | Out-Null }
    Start-ScheduledTask -TaskName $TaskName | Out-Null
} catch {
    Write-Output "could not start the task: $($_.Exception.Message)"
    Write-Output "dashboard: DOWN"
    exit 1
}
$waited = 0
while ($waited -lt $WaitSeconds) {
    Start-Sleep -Seconds 2
    $waited += 2
    if (Test-Listening $Port) {
        Write-Output "listening on port $Port after $waited s"
        Write-Output "dashboard: restarted"
        exit 0
    }
}
Write-Output "still nothing on port $Port after $WaitSeconds s (see logs\dashboard.log)"
Write-Output "dashboard: DOWN"
exit 1
