# Meeting Island updater. Runs from the folder the update package unpacked to.
# It changes only the files that differ, keeps your settings, glossary and saved transcripts, checks every file when done,
# and puts the old version back automatically if anything goes wrong.
param([switch]$Silent)
$ErrorActionPreference = 'Stop'
if ($env:MI_SILENT) { $Silent = $true }                  # (automated tests: no dialogs)
$Product = 'Meeting Island'
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Log = Join-Path $env:TEMP 'MeetingIsland-update.log'
Start-Transcript -Path $Log -Force | Out-Null
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

function Say([string]$text, [string]$title = "$Product update", [string]$icon = 'Information') {
    if ($Silent) { Write-Host $text; return }
    [void][System.Windows.Forms.MessageBox]::Show($text, $title, 'OK', $icon)
}

# ---------------------------------------------------------------- what to update
$plan = @{}
Get-Content (Join-Path $Here 'plan.txt') | ForEach-Object { $k, $v = $_ -split '=', 2; if ($k) { $plan[$k.Trim()] = $v.Trim() } }
$To = $plan['to']; $From = $plan['from']

$Root = $env:MI_TARGET
if (-not $Root) {
    $key = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\MeetingIsland'
    if (Test-Path $key) { $Root = (Get-ItemProperty $key).InstallLocation }
    if (-not $Root -or -not (Test-Path (Join-Path $Root 'MeetingIsland.exe'))) { $Root = Join-Path $env:LOCALAPPDATA 'Programs\Meeting Island' }
}
if (-not (Test-Path (Join-Path $Root 'MeetingIsland.exe'))) {
    Say "$Product is not installed on this computer, so there is nothing to update.`n`nInstall it first with MeetingIsland-Setup.exe." "$Product update" 'Warning'
    Stop-Transcript | Out-Null; exit 1
}
$Installed = ''
$vf = Join-Path $Root 'VERSION.txt'
if (Test-Path $vf) { $Installed = (Get-Content $vf -TotalCount 1).Trim() }

$freeGb = [math]::Round((Get-PSDrive -Name ($Root.Substring(0, 1))).Free / 1GB, 1)
if ($freeGb -lt 2) {
    Say "Not enough free disk space ($freeGb GB free). Please free some space and run the update again." "$Product update" 'Error'
    Stop-Transcript | Out-Null; exit 1
}

# ---------------------------------------------------------------- the target file list: relative path, size, sha256
$target = [ordered]@{}
Get-Content (Join-Path $Here 'manifest.tsv') | ForEach-Object {
    $p = $_ -split "`t"
    if ($p.Count -ge 3) { $target[$p[0]] = @{ size = [int64]$p[1]; sha = $p[2] } }
}
$changed = @(Get-Content (Join-Path $Here 'changed.txt') | Where-Object { $_ })
$removed = @(Get-Content (Join-Path $Here 'removed.txt') -ErrorAction SilentlyContinue | Where-Object { $_ })

# ---------------------------------------------------------------- progress window
$form = $null
if (-not $Silent) {
    $form = New-Object System.Windows.Forms.Form
    $form.Text = "Updating $Product"
    $form.StartPosition = 'CenterScreen'; $form.FormBorderStyle = 'FixedDialog'; $form.ControlBox = $false; $form.TopMost = $true
    $form.ClientSize = New-Object System.Drawing.Size(460, 120)
    $label = New-Object System.Windows.Forms.Label
    $label.AutoSize = $false; $label.Location = New-Object System.Drawing.Point(20, 18); $label.Size = New-Object System.Drawing.Size(420, 40)
    $label.Font = New-Object System.Drawing.Font('Segoe UI', 10); $label.Text = 'Getting ready...'
    $bar = New-Object System.Windows.Forms.ProgressBar
    $bar.Style = 'Marquee'; $bar.MarqueeAnimationSpeed = 25
    $bar.Location = New-Object System.Drawing.Point(20, 70); $bar.Size = New-Object System.Drawing.Size(420, 22)
    $form.Controls.Add($label); $form.Controls.Add($bar)
    $form.Show(); [System.Windows.Forms.Application]::DoEvents()
}
function Step([string]$text) {
    Write-Host $text
    if ($form) { $label.Text = $text; [System.Windows.Forms.Application]::DoEvents() }
}

$Stage = Join-Path $Root '.update_new'
$Backup = Join-Path $Root '.update_backup'
$moved = New-Object System.Collections.ArrayList        # files put in place (for undo)
$saved = New-Object System.Collections.ArrayList        # old files moved to the backup folder (for undo)
$success = $false
try {
    Step 'Closing Meeting Island...'
    Get-Process -Name 'MeetingIsland' -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Milliseconds 900
    foreach ($d in @($Stage, $Backup)) { if (Test-Path $d) { Remove-Item -Recurse -Force $d } }
    New-Item -ItemType Directory -Force $Stage, $Backup | Out-Null

    # ------------------------------------------------------------ unpack the new files and check they arrived intact
    if ($changed.Count -gt 0) {
        Step "Unpacking $($changed.Count) updated files..."
        $zip = Join-Path $Here 'update.zip'
        $tar = Join-Path $env:SystemRoot 'System32\tar.exe'
        if (Test-Path $tar) {
            $p = Start-Process -FilePath $tar -ArgumentList @('-xf', "`"$zip`"", '-C', "`"$Stage`"") -PassThru -WindowStyle Hidden
            while (-not $p.HasExited) { if ($form) { [System.Windows.Forms.Application]::DoEvents() }; Start-Sleep -Milliseconds 120 }
            if ($p.ExitCode -ne 0) { throw "Unpacking failed (code $($p.ExitCode))." }
        } else { Expand-Archive -LiteralPath $zip -DestinationPath $Stage -Force }
        Step 'Checking the new files...'
        foreach ($rel in $changed) {
            $f = Join-Path $Stage ($rel -replace '/', '\')
            if (-not (Test-Path -LiteralPath $f)) { throw "The update is incomplete: $rel is missing." }
            if ((Get-FileHash -LiteralPath $f -Algorithm SHA256).Hash.ToLower() -ne $target[$rel].sha) { throw "The update is damaged: $rel does not match. Copy it again." }
            if ($form) { [System.Windows.Forms.Application]::DoEvents() }
        }
    }

    # ------------------------------------------------------------ swap: old files go to the backup folder, new ones in
    Step 'Installing the update...'
    foreach ($rel in $changed) {
        $dst = Join-Path $Root ($rel -replace '/', '\')
        $new = Join-Path $Stage ($rel -replace '/', '\')
        New-Item -ItemType Directory -Force (Split-Path -Parent $dst) | Out-Null
        if (Test-Path -LiteralPath $dst) {
            $bk = Join-Path $Backup ($rel -replace '/', '\')
            New-Item -ItemType Directory -Force (Split-Path -Parent $bk) | Out-Null
            Move-Item -LiteralPath $dst -Destination $bk -Force
            [void]$saved.Add($rel)
        }
        Move-Item -LiteralPath $new -Destination $dst -Force
        [void]$moved.Add($rel)
    }
    foreach ($rel in $removed) {
        $dst = Join-Path $Root ($rel -replace '/', '\')
        if (Test-Path -LiteralPath $dst) {
            $bk = Join-Path $Backup ($rel -replace '/', '\')
            New-Item -ItemType Directory -Force (Split-Path -Parent $bk) | Out-Null
            Move-Item -LiteralPath $dst -Destination $bk -Force
            [void]$saved.Add($rel)
        }
    }

    # ------------------------------------------------------------ every file of the new version must now be there, right size
    Step 'Checking the installation...'
    $bad = New-Object System.Collections.ArrayList
    foreach ($rel in $target.Keys) {
        $f = Join-Path $Root ($rel -replace '/', '\')
        if (-not (Test-Path -LiteralPath $f) -or (Get-Item -LiteralPath $f).Length -ne $target[$rel].size) { [void]$bad.Add($rel) }
    }
    if ($bad.Count -gt 0) {
        if ($Installed -eq $From) {
            throw "Some program files are missing or damaged ($($bad.Count) files, for example $($bad[0])). Please reinstall with the full MeetingIsland-Setup.exe - your settings and transcripts are kept."
        }
        throw "This update is for version $From, but this PC has version '$Installed' ($($bad.Count) files differ, for example $($bad[0])). Install the full MeetingIsland-Setup.exe instead, or apply the updates in order."
    }
    Copy-Item -LiteralPath (Join-Path $Here 'manifest.tsv') -Destination (Join-Path $Root 'manifest.tsv') -Force
    $success = $true
}
catch {
    $err = $_.Exception.Message
    Write-Host "FAILED: $err"
    Step 'Something went wrong - putting the old version back...'
    foreach ($rel in $moved) { Remove-Item -LiteralPath (Join-Path $Root ($rel -replace '/', '\')) -Force -ErrorAction SilentlyContinue }
    foreach ($rel in $saved) {
        $dst = Join-Path $Root ($rel -replace '/', '\')
        $bk = Join-Path $Backup ($rel -replace '/', '\')
        if (Test-Path -LiteralPath $bk) {
            New-Item -ItemType Directory -Force (Split-Path -Parent $dst) | Out-Null
            Move-Item -LiteralPath $bk -Destination $dst -Force -ErrorAction SilentlyContinue
        }
    }
    if ($form) { $form.Close() }
    foreach ($d in @($Stage, $Backup)) { if (Test-Path $d) { Remove-Item -Recurse -Force $d -ErrorAction SilentlyContinue } }
    Stop-Transcript | Out-Null
    Say "The update could not be installed. Your previous version was put back and still works.`n`n$err`n`nA log was saved to:`n$Log" "$Product update" 'Error'
    exit 1
}

foreach ($d in @($Stage, $Backup)) { if (Test-Path $d) { Remove-Item -Recurse -Force $d -ErrorAction SilentlyContinue } }
try {
    $key = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\MeetingIsland'
    if (Test-Path $key) {
        Set-ItemProperty $key DisplayVersion $To
        $sizeKb = [int]((Get-ChildItem -Recurse -File $Root | Measure-Object Length -Sum).Sum / 1KB)
        Set-ItemProperty $key EstimatedSize $sizeKb
    }
} catch { }
if ($form) { $form.Close() }
Stop-Transcript | Out-Null
if ($Silent -or $env:MI_NOLAUNCH) { Write-Host "Updated to $To."; exit 0 }
$r = [System.Windows.Forms.MessageBox]::Show("$Product was updated to version $To.`n`nYour settings, glossary and saved transcripts are untouched.`n`nOpen it now?", "$Product is up to date", 'YesNo', 'Information')
if ($r -eq 'Yes') { Start-Process -FilePath (Join-Path $Root 'MeetingIsland.exe') -WorkingDirectory $Root }
exit 0
