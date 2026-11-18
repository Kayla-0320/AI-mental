@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-package.ps1" %*
endlocal
