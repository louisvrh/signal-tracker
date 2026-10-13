<#
.SYNOPSIS
    Register (or remove) the scheduled task that keeps the dashboard running.

.DESCRIPTION
    Creates or updates the task "SignalTracker Dashboard" for the current user:
      * starts at logon (yours only), in a hidden PowerShell window, running
        scripts\dashboard_service.ps1, which starts Streamlit on 127.0.0.1 and
        restarts it whenever it stops (logs\dashboard.log);
      * a second trigger fires every 5 minutes, all day, every day. While the
        service runs, these starts are ignored (one instance at a time); if the
        service has been killed, the next one starts it again. This is the
        task-level restart: Task Scheduler's own restart-on-failure (also set:
        every minute, up to 999 times) only covers a task that fails to start,
        not one whose running process is killed (result 0xC000013A, 2026-09-30);
      * no time limit (the default would stop it after 3 days), also on battery;
      * runs only while you are logged on, so no password is stored and no
        administrator rights are needed (like "SignalTracker Daily", PLAN D26).
    Running the script again updates the task. -Start also (re)starts it now,
    -Stop stops and disables the task and stops its dashboard (until -Start;
    scripts\run_daily.ps1 leaves a disabled task alone), and -Unregister stops
    and removes it.

    Run once from PowerShell in the project folder:
        powershell -NoProfile -ExecutionPolicy Bypass -File scripts\register_dashboard_task.ps1 -Start

.PARAMETER TaskName
    Name shown in Task Scheduler (default "SignalTracker Dashboard").
.PARAMETER Port
    Local port for the dashboard (default 8551).
.PARAMETER Start
    Start (or restart) the task right away instead of waiting for the next logon.
.PARAMETER Stop
    Stop and disable the task and stop the dashboard it runs, then exit (the task
    stays registered; -Start enables it again).
.PARAMETER WatchMinutes
    How often the watchdog trigger checks that the service is running (default 5).
.PARAMETER Unregister
    Stop and remove the task instead of creating it.
#>
[CmdletBinding()]
param(
    [string]$TaskName = 'SignalTracker Dashboard',
    [int]$Port = 8551,
    [int]$WatchMinutes = 5,
    [switch]$Start,
    [switch]$Stop,
    [switch]$Unregister
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Service = Join-Path $PSScriptRoot 'dashboard_service.ps1'

function Stop-Dashboard {
    # Only this project's Streamlit: its command line names this project's app.py.
    $app = Join-Path $Root 'dashboard\app.py'
    Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -and $_.CommandLine.Contains($app) -and $_.CommandLine.Contains('streamlit') } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; Write-Host "Stopped dashboard process $($_.ProcessId)." }
}

if ($Stop) {
    # Disabled, so neither the watchdog trigger nor run_daily.ps1 starts it again.
    Disable-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue | Out-Null
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Stop-Dashboard
    Write-Host "Stopped and disabled '$TaskName' and its dashboard; -Start enables and starts it again."
    exit 0
}

if ($Unregister) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Stop-Dashboard
        Write-Host "Stopped and removed the scheduled task '$TaskName'."
    } else {
        Write-Host "There is no scheduled task named '$TaskName'."
    }
    exit 0
}

$user = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$actionArgs = @{
    Execute          = 'powershell.exe'
    Argument         = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Service`" -Port $Port"
    WorkingDirectory = $Root
}
$settingsArgs = @{
    StartWhenAvailable         = $true
    AllowStartIfOnBatteries    = $true
    DontStopIfGoingOnBatteries = $true
    MultipleInstances          = 'IgnoreNew'
    ExecutionTimeLimit         = [TimeSpan]::Zero   # no limit
    RestartCount               = 999
    RestartInterval            = (New-TimeSpan -Minutes 1)
}
$logon = New-ScheduledTaskTrigger -AtLogOn -User $user
# Repeats every $WatchMinutes minutes with no end date; with MultipleInstances IgnoreNew a
# start while the service runs does nothing, so this only restarts a killed service.
$watchdog = New-ScheduledTaskTrigger -Once -At (Get-Date).Date -RepetitionInterval (New-TimeSpan -Minutes $WatchMinutes)
$taskArgs = @{
    TaskName    = $TaskName
    Description = "Signal tracker dashboard on http://127.0.0.1:$Port (restarted if it stops, checked every $WatchMinutes minutes; log: logs\dashboard.log)"
    Action      = New-ScheduledTaskAction @actionArgs
    Trigger     = @($logon, $watchdog)
    Settings    = New-ScheduledTaskSettingsSet @settingsArgs
    Principal   = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
    Force       = $true
}
Register-ScheduledTask @taskArgs | Out-Null
Write-Host "Registered '$TaskName' for ${user}: starts at logon and is restarted within $WatchMinutes minutes if it stops; dashboard on http://127.0.0.1:$Port."
if ($Start) {
    Enable-ScheduledTask -TaskName $TaskName | Out-Null
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue  # a restart if it was running
    Start-ScheduledTask -TaskName $TaskName
    Write-Host "Started it. The dashboard is up within ~10 seconds (log: logs\dashboard.log)."
} else {
    Write-Host "Start it now with:  Start-ScheduledTask -TaskName '$TaskName'"
}
