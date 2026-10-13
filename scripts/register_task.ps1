<#
.SYNOPSIS
    Register (or remove) the Windows scheduled task that runs run_daily.ps1 every day.

.DESCRIPTION
    Creates or updates the task "SignalTracker Daily" for the current user:
      * daily at 07:00 local time (-At to change);
      * runs as soon as possible after a missed start (laptop off or asleep);
      * starts only when a network connection is available;
      * may start on battery and keeps running if unplugged;
      * one instance at a time, stopped if still running after 3 hours;
      * runs only while you are logged on, so no password is stored and no
        administrator rights are needed. PowerShell starts hidden, so a
        window may flash briefly.
    Running the script again updates the task. -Unregister removes it.

    Run once from PowerShell in the project folder:
        powershell -NoProfile -ExecutionPolicy Bypass -File scripts\register_task.ps1

.PARAMETER TaskName
    Name shown in Task Scheduler (default "SignalTracker Daily").
.PARAMETER At
    Local start time, HH:mm (default 07:00).
.PARAMETER Unregister
    Remove the task instead of creating it.
#>
[CmdletBinding()]
param(
    [string]$TaskName = 'SignalTracker Daily',
    [string]$At = '07:00',
    [switch]$Unregister
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$RunDaily = Join-Path $PSScriptRoot 'run_daily.ps1'

if ($Unregister) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed the scheduled task '$TaskName'."
    } else {
        Write-Host "There is no scheduled task named '$TaskName'."
    }
    exit 0
}

$user = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$actionArgs = @{
    Execute          = 'powershell.exe'
    Argument         = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$RunDaily`""
    WorkingDirectory = $Root
}
$settingsArgs = @{
    StartWhenAvailable         = $true
    RunOnlyIfNetworkAvailable  = $true
    AllowStartIfOnBatteries    = $true
    DontStopIfGoingOnBatteries = $true
    MultipleInstances          = 'IgnoreNew'
    ExecutionTimeLimit         = (New-TimeSpan -Hours 3)
}
$taskArgs = @{
    TaskName    = $TaskName
    Description = "Signal tracker daily run: $RunDaily (log: logs\run_daily.log)"
    Action      = New-ScheduledTaskAction @actionArgs
    Trigger     = New-ScheduledTaskTrigger -Daily -At $At
    Settings    = New-ScheduledTaskSettingsSet @settingsArgs
    Principal   = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
    Force       = $true
}
Register-ScheduledTask @taskArgs | Out-Null

$info = Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo
Write-Host "Registered '$TaskName' for ${user}: daily at $At, next run $($info.NextRunTime)."
Write-Host "Check it in Task Scheduler (taskschd.msc, Task Scheduler Library), or run it now with:"
Write-Host "    Start-ScheduledTask -TaskName '$TaskName'"
