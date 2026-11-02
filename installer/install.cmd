@echo off
rem Launched by the self-extracting package (or double-click it from the unzipped folder).
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0install.ps1"
exit /b %errorlevel%
