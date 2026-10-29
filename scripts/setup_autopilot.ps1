# Registers Windows Task Scheduler tasks so MoneyPrinter posts on its own.
#
#   MoneyPrinter Worker     - starts the worker when you log in (generates queued videos)
#   MoneyPrinter Autopilot  - runs `autopilot.py tick` at logon and every 30 minutes
#
# `tick` picks random posting times each day (see AUTOPILOT_WINDOWS and
# AUTOPILOT_WINDOW_MINUTES in .env) and queues any post that is due. If the PC
# was off when a post was due, it is caught up once the PC is back on.
#
# Usage (from the project root):
#   powershell -ExecutionPolicy Bypass -File scripts\setup_autopilot.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\setup_autopilot.ps1 -Remove

param(
    [switch]$Remove
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\pythonw.exe"
$WorkerTask = "MoneyPrinter Worker"
$AutopilotTask = "MoneyPrinter Autopilot"

if ($Remove) {
    foreach ($name in @($WorkerTask, $AutopilotTask)) {
        if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
            Unregister-ScheduledTask -TaskName $name -Confirm:$false
            Write-Host "Removed '$name'"
        }
    }
    return
}

if (-not (Test-Path $Python)) {
    throw "Python venv not found at $Python. Run 'uv sync' first."
}

$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5)

# Worker: runs continuously from logon
$WorkerAction = New-ScheduledTaskAction -Execute $Python `
    -Argument "Backend\worker.py" -WorkingDirectory $ProjectRoot
$WorkerTrigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
Register-ScheduledTask -TaskName $WorkerTask -Action $WorkerAction `
    -Trigger $WorkerTrigger -Settings $Settings -Force | Out-Null
Write-Host "Registered '$WorkerTask' (starts at logon)"

# Autopilot: check for due posts at logon and every 30 minutes
$AutopilotAction = New-ScheduledTaskAction -Execute $Python `
    -Argument "Backend\autopilot.py tick" -WorkingDirectory $ProjectRoot
$AutopilotTriggers = @(
    (New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME),
    (New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 30))
)
Register-ScheduledTask -TaskName $AutopilotTask -Action $AutopilotAction `
    -Trigger $AutopilotTriggers -Settings $Settings -Force | Out-Null
Write-Host "Registered '$AutopilotTask' (checks every 30 minutes while the PC is on)"

Write-Host ""
Write-Host "Done. Posts happen at random times inside your posting windows;"
Write-Host "missed posts are caught up when the PC is back on."
