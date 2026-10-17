@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
if not exist ".runtime\Scripts\python.exe" (
  echo Please run setup.cmd first.
  pause
  exit /b 1
)
".runtime\Scripts\python.exe" -c "import sys" >nul 2>&1
if errorlevel 1 (
  echo Local Python environment cannot start. If this folder was moved, recreate .runtime with setup.cmd.
  pause
  exit /b 1
)
".runtime\Scripts\python.exe" run_app.py %*
if errorlevel 1 pause
