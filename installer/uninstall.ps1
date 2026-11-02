# Meeting Island uninstaller. Removes the program, shortcuts and the "Apps & features" entry.
# Your saved transcripts are never touched. Settings/glossary are kept unless you ask to remove them.
param([switch]$Silent, [switch]$RemoveData)
$ErrorActionPreference = 'Continue'
$Product = 'Meeting Island'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Desktop = if ($env:MI_DESKTOP) { $env:MI_DESKTOP } else { [Environment]::GetFolderPath('Desktop') }
$StartMenu = if ($env:MI_STARTMENU) { $env:MI_STARTMENU } else { Join-Path ([Environment]::GetFolderPath('Programs')) $Product }
$Data = Join-Path $env:LOCALAPPDATA 'MeetingIsland'
Add-Type -AssemblyName System.Windows.Forms

if (-not (Test-Path (Join-Path $Root 'MeetingIsland.exe'))) { exit 0 }     # safety: only ever delete a Meeting Island folder

if (-not $Silent) {
    $r = [System.Windows.Forms.MessageBox]::Show("Remove $Product from this computer?`n`nYour saved transcripts are not deleted.", "Uninstall $Product", 'YesNo', 'Question')
    if ($r -ne 'Yes') { exit 0 }
    $r2 = [System.Windows.Forms.MessageBox]::Show("Also delete $Product's settings, glossary and any unsaved sessions?`n($Data)", "Uninstall $Product", 'YesNo', 'Question')
    if ($r2 -eq 'Yes') { $RemoveData = $true }
}
Get-Process -Name 'MeetingIsland' -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Milliseconds 800
Remove-Item -Force (Join-Path $Desktop "$Product.lnk") -ErrorAction SilentlyContinue
if (Test-Path $StartMenu) { Remove-Item -Recurse -Force $StartMenu -ErrorAction SilentlyContinue }
Remove-Item -Recurse -Force 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\MeetingIsland' -ErrorAction SilentlyContinue
if ($RemoveData -and (Test-Path $Data)) { Remove-Item -Recurse -Force $Data -ErrorAction SilentlyContinue }

# this script lives inside the folder it deletes: finish from a temporary command so the folder can go
$cmd = Join-Path $env:TEMP 'mi-remove.cmd'
"@echo off`r`nping 127.0.0.1 -n 3 >nul`r`nrmdir /s /q `"$Root`"`r`ndel `"%~f0`"" | Set-Content -Encoding ASCII $cmd
Start-Process -FilePath $cmd -WindowStyle Hidden
if (-not $Silent) { [void][System.Windows.Forms.MessageBox]::Show("$Product was removed.", "Uninstall $Product", 'OK', 'Information') }
