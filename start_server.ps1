# Start the arm server cleanly: stop any old copy first, then run one in this window.
# Two copies fight over COM10 and port 8765 and give "No reply from servo 1", so always start it this way.
#
#   .\start_server.ps1                 real arm on COM10 (starts disarmed; press G in the viewer to arm)
#   .\start_server.ps1 -Port COM7      a different serial port
#   .\start_server.ps1 -Sim            simulation only, no serial port
param([string]$Port = "COM10", [switch]$Sim)

Set-Location $PSScriptRoot
$old = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*sim_server.py*" -and $_.ProcessId -ne $PID }
if ($old) {
    # The venv python.exe is a launcher that starts the real python as a child; stop both.
    $old | Sort-Object ProcessId -Descending | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Write-Host "Stopped $($old.Count) old server process(es)."
    Start-Sleep -Seconds 2
}

if ($Sim) { & .\.venv\Scripts\python.exe sim_server.py }
else      { & .\.venv\Scripts\python.exe sim_server.py --hardware $Port }
