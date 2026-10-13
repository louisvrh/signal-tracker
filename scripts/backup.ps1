<#
.SYNOPSIS
    Copy data\lake, data\meta and data\raw to BACKUP_DIR (set in .env).

.DESCRIPTION
    Additive copy with robocopy into <BACKUP_DIR>\signal-tracker\{lake,meta,raw}:
    new and changed files are copied, and nothing is ever deleted from the
    backup. A local mistake (a deleted data folder, say) can therefore never
    wipe the backup. The price is that lake files merged by `compact` stay in
    the backup next to the merged file; see docs\NOTES.md ("Backups") before
    restoring.

    BACKUP_DIR is read through tracker.settings; this script never reads .env
    itself. Writes robocopy's detailed log to logs\backup.log and the time of
    the last successful backup to logs\last_backup.txt (shown by `status`).

    Exit codes: 0 done, 1 robocopy failed, 2 BACKUP_DIR not set,
    3 backup folder missing (e.g. drive not connected), 4 backup folder is
    inside the project.

.PARAMETER Destination
    Back up here instead of BACKUP_DIR.
#>
[CmdletBinding()]
param(
    [string]$Destination
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$LogDir = Join-Path $Root 'logs'
$RobocopyLog = Join-Path $LogDir 'backup.log'
$Marker = Join-Path $LogDir 'last_backup.txt'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location $Root
$env:PYTHONUTF8 = '1'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

if (-not $Destination) {
    $lookup = "from tracker import settings; print(settings.get_setting('BACKUP_DIR') or '')"
    $Destination = ((& $Python -c $lookup) | Out-String).Trim()
}
if (-not $Destination) {
    Write-Host 'BACKUP_DIR is not set in .env, so nothing was backed up.'
    exit 2
}
if (-not (Test-Path -LiteralPath $Destination -PathType Container)) {
    Write-Host "Backup folder '$Destination' not found (drive not connected?), so nothing was backed up."
    exit 3
}
$resolved = (Resolve-Path -LiteralPath $Destination).ProviderPath.TrimEnd('\') + '\'
if ($resolved.StartsWith($Root.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
    Write-Host "Backup folder '$Destination' is inside the project; choose a folder elsewhere."
    exit 4
}

$Target = Join-Path $Destination 'signal-tracker'
Write-Host "Backing up to $Target"
Add-Content -Path $RobocopyLog -Value ('===== {0:yyyy-MM-dd HH:mm:ss} backup to {1}' -f (Get-Date), $Target) -Encoding UTF8
$failed = $false
foreach ($name in @('lake', 'meta', 'raw')) {
    $source = Join-Path $Root "data\$name"
    if (-not (Test-Path $source)) {
        Write-Host "  ${name}: no local folder, skipped"
        continue
    }
    # /E subfolders, never /MIR or /PURGE (never delete); skip half-written *.tmp files.
    robocopy $source (Join-Path $Target $name) /E /COPY:DAT /DCOPY:T /R:2 /W:5 /XF *.tmp /NP /NFL /NDL "/LOG+:$RobocopyLog" | Out-Null
    $code = $LASTEXITCODE
    if ($code -ge 8) {
        Write-Host "  ${name}: FAILED (robocopy exit code $code, see logs\backup.log)"
        $failed = $true
    } elseif ($code -band 1) {
        Write-Host "  ${name}: new or changed files copied"
    } else {
        Write-Host "  ${name}: already up to date"
    }
}
if ($failed) { exit 1 }

$files = Get-ChildItem -LiteralPath $Target -Recurse -File
$megabytes = ($files | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host ('Backup complete: {0} files, {1:N1} MB in {2}' -f $files.Count, $megabytes, $Target)
Set-Content -Path $Marker -Value ('{0:yyyy-MM-ddTHH:mm:ssZ} {1}' -f (Get-Date).ToUniversalTime(), $Target) -Encoding UTF8
exit 0
