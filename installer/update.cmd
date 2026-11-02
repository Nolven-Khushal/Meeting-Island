@echo off
rem Launched by the self-extracting update package.
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0update.ps1"
exit /b %errorlevel%
