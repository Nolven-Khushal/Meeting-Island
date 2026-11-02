# Meeting Island installer. Runs from the folder the package unpacked to; app.zip sits next to this script.
# Per-user install: no administrator rights, nothing outside your own profile is touched.
param([switch]$Silent)
$ErrorActionPreference = 'Stop'
$Product = 'Meeting Island'
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = if ($env:MI_TARGET) { $env:MI_TARGET } else { Join-Path $env:LOCALAPPDATA 'Programs\Meeting Island' }
$Desktop = if ($env:MI_DESKTOP) { $env:MI_DESKTOP } else { [Environment]::GetFolderPath('Desktop') }
$StartMenu = if ($env:MI_STARTMENU) { $env:MI_STARTMENU } else { Join-Path ([Environment]::GetFolderPath('Programs')) $Product }
$Log = Join-Path $env:TEMP 'MeetingIsland-install.log'
Start-Transcript -Path $Log -Force | Out-Null

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

function Say([string]$text, [string]$title = $Product, [string]$icon = 'Information') {
    if ($Silent) { Write-Host $text; return }
    [void][System.Windows.Forms.MessageBox]::Show($text, $title, 'OK', $icon)
}

# ---------------------------------------------------------------- requirements
if (-not [Environment]::Is64BitOperatingSystem) {
    Say "$Product needs a 64-bit version of Windows 10 or Windows 11." "$Product - cannot install" 'Error'; exit 1
}
if ([Environment]::OSVersion.Version.Major -lt 10) {
    Say "$Product needs Windows 10 or Windows 11." "$Product - cannot install" 'Error'; exit 1
}
$freeGb = [math]::Round((Get-PSDrive -Name ($env:LOCALAPPDATA.Substring(0, 1))).Free / 1GB, 1)
if ($freeGb -lt 3) {
    Say "Not enough free disk space ($freeGb GB free). $Product needs about 2 GB. Please free some space and run the installer again." "$Product - cannot install" 'Error'; exit 1
}

# ---------------------------------------------------------------- progress window
$form = $null
if (-not $Silent) {
    $form = New-Object System.Windows.Forms.Form
    $form.Text = "Installing $Product"
    $form.StartPosition = 'CenterScreen'
    $form.FormBorderStyle = 'FixedDialog'
    $form.ControlBox = $false
    $form.TopMost = $true
    $form.ClientSize = New-Object System.Drawing.Size(460, 120)
    $label = New-Object System.Windows.Forms.Label
    $label.AutoSize = $false
    $label.Location = New-Object System.Drawing.Point(20, 18)
    $label.Size = New-Object System.Drawing.Size(420, 40)
    $label.Font = New-Object System.Drawing.Font('Segoe UI', 10)
    $label.Text = 'Getting ready...'
    $bar = New-Object System.Windows.Forms.ProgressBar
    $bar.Style = 'Marquee'
    $bar.MarqueeAnimationSpeed = 25
    $bar.Location = New-Object System.Drawing.Point(20, 70)
    $bar.Size = New-Object System.Drawing.Size(420, 22)
    $form.Controls.Add($label); $form.Controls.Add($bar)
    $form.Show()
    [System.Windows.Forms.Application]::DoEvents()
}
function Step([string]$text) {
    Write-Host $text
    if ($form) { $label.Text = $text; [System.Windows.Forms.Application]::DoEvents() }
}
function WaitFor($proc) {
    while (-not $proc.HasExited) { if ($form) { [System.Windows.Forms.Application]::DoEvents() }; Start-Sleep -Milliseconds 120 }
}

try {
    # ------------------------------------------------------------ stop an older copy and replace the program files only
    Step 'Closing any running copy...'
    Get-Process -Name 'MeetingIsland' -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Milliseconds 800
    if ((Test-Path $Root) -and (Test-Path (Join-Path $Root 'MeetingIsland.exe'))) {
        Step 'Removing the previous version (your settings and glossary are kept)...'
        Remove-Item -Recurse -Force $Root
    }
    New-Item -ItemType Directory -Force $Root | Out-Null

    # ------------------------------------------------------------ unpack
    Step 'Copying the program and the speech models. This takes a minute or two...'
    $zip = Join-Path $Here 'app.zip'
    if (-not (Test-Path $zip)) { throw "app.zip is missing from the installer." }
    $tar = Join-Path $env:SystemRoot 'System32\tar.exe'
    if (Test-Path $tar) {
        $p = Start-Process -FilePath $tar -ArgumentList @('-xf', "`"$zip`"", '-C', "`"$Root`"") -PassThru -WindowStyle Hidden
        WaitFor $p
        if ($p.ExitCode -ne 0) { throw "Unpacking failed (code $($p.ExitCode))." }
    } else {
        Expand-Archive -LiteralPath $zip -DestinationPath $Root -Force
    }
    if (-not (Test-Path (Join-Path $Root 'MeetingIsland.exe'))) { throw 'The program was not unpacked correctly.' }

    # ------------------------------------------------------------ shortcuts
    Step 'Creating shortcuts...'
    $exe = Join-Path $Root 'MeetingIsland.exe'
    $ws = New-Object -ComObject WScript.Shell
    function MakeLink([string]$path, [string]$target, [string]$desc, [string]$args2 = '') {
        $dir = Split-Path -Parent $path
        New-Item -ItemType Directory -Force $dir | Out-Null
        $lnk = $ws.CreateShortcut($path)
        $lnk.TargetPath = $target
        if ($args2) { $lnk.Arguments = $args2 }
        $lnk.WorkingDirectory = $Root
        $lnk.IconLocation = (Join-Path $Root 'island.ico')
        $lnk.Description = $desc
        $lnk.Save()
    }
    MakeLink (Join-Path $Desktop "$Product.lnk") $exe 'Live transcript of meetings: English, Hindi and Hinglish'
    MakeLink (Join-Path $StartMenu "$Product.lnk") $exe 'Live transcript of meetings: English, Hindi and Hinglish'
    $guide = $ws.CreateShortcut((Join-Path $StartMenu 'Quick start guide.lnk'))
    $guide.TargetPath = (Join-Path $Root 'QUICK-START.txt'); $guide.Save()
    $unins = $ws.CreateShortcut((Join-Path $StartMenu "Uninstall $Product.lnk"))
    $unins.TargetPath = (Join-Path $Root 'Uninstall Meeting Island.cmd'); $unins.IconLocation = (Join-Path $Root 'island.ico'); $unins.Save()

    # ------------------------------------------------------------ "Apps & features" entry (this user only)
    Step 'Registering the uninstaller...'
    $key = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\MeetingIsland'
    New-Item -Path $key -Force | Out-Null
    $sizeKb = [int]((Get-ChildItem -Recurse -File $Root | Measure-Object Length -Sum).Sum / 1KB)
    $ver = (Get-Content (Join-Path $Root 'VERSION.txt') -ErrorAction SilentlyContinue | Select-Object -First 1)
    Set-ItemProperty $key DisplayName $Product
    Set-ItemProperty $key DisplayVersion $(if ($ver) { $ver } else { '1.0' })
    Set-ItemProperty $key Publisher 'Meeting Island'
    Set-ItemProperty $key InstallLocation $Root
    Set-ItemProperty $key DisplayIcon (Join-Path $Root 'island.ico')
    Set-ItemProperty $key UninstallString ('"' + (Join-Path $Root 'Uninstall Meeting Island.cmd') + '"')
    Set-ItemProperty $key EstimatedSize $sizeKb
    Set-ItemProperty $key NoModify 1
    Set-ItemProperty $key NoRepair 1

    if ($form) { $form.Close() }
    Stop-Transcript | Out-Null
    if ($Silent -or $env:MI_NOLAUNCH) { Write-Host 'Installed.'; exit 0 }
    $r = [System.Windows.Forms.MessageBox]::Show("$Product is installed.`n`nThere is now a '$Product' icon on your Desktop. Open it, press 'Start session', and the live transcript begins.`n`nOpen it now?", "$Product is ready", 'YesNo', 'Information')
    if ($r -eq 'Yes') { Start-Process -FilePath $exe -WorkingDirectory $Root }
    exit 0
}
catch {
    if ($form) { $form.Close() }
    Stop-Transcript | Out-Null
    Say "Sorry, the installation did not finish.`n`n$($_.Exception.Message)`n`nA log was saved to:`n$Log" "$Product - problem" 'Error'
    exit 1
}
