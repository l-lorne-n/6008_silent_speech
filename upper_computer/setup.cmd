@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
set "SEMG_NO_PAUSE=0"
set "SEMG_CHECK_ONLY=0"
for %%A in (%*) do (
  if /i "%%~A"=="--no-pause" set "SEMG_NO_PAUSE=1"
  if /i "%%~A"=="--check" set "SEMG_CHECK_ONLY=1"
)
if "%SEMG_CHECK_ONLY%"=="1" goto :check
if exist ".runtime\Scripts\python.exe" goto :install
if exist ".runtime" goto :broken
if exist ".python\cpython-3.12-windows-x86_64-none\python.exe" goto :bundled
py -3.12 check_environment.py --python-only >nul 2>&1
if not errorlevel 1 goto :launcher
python check_environment.py --python-only >nul 2>&1
if not errorlevel 1 goto :path_python
echo No compatible Python found. Install 64-bit Python 3.12 with the Python launcher or add it to PATH.
goto :failed
:bundled
".python\cpython-3.12-windows-x86_64-none\python.exe" check_environment.py --python-only
if errorlevel 1 goto :failed
".python\cpython-3.12-windows-x86_64-none\python.exe" -m venv .runtime
if errorlevel 1 goto :failed
goto :install
:launcher
py -3.12 -m venv .runtime
if errorlevel 1 goto :failed
goto :install
:path_python
python -m venv .runtime
if errorlevel 1 goto :failed
:install
".runtime\Scripts\python.exe" check_environment.py --python-only
if errorlevel 1 goto :broken
".runtime\Scripts\python.exe" -m ensurepip
if errorlevel 1 goto :failed
".runtime\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :failed
:check
if not exist ".runtime\Scripts\python.exe" (
  echo No local environment. Run setup.cmd first.
  goto :failed
)
".runtime\Scripts\python.exe" check_environment.py
if errorlevel 1 goto :failed
".runtime\Scripts\python.exe" -m pip check
if errorlevel 1 goto :failed
echo Environment ready. Run start.cmd. Use simulation mode to check without hardware.
if "%SEMG_NO_PAUSE%"=="0" pause
exit /b 0
:broken
echo The existing .runtime is incomplete, incompatible, or points to a moved Python installation.
echo Close the app, rename .runtime to .runtime-old, then run setup.cmd to recreate it.
echo Keep recordings and models. Setup will not remove any folders automatically.
:failed
echo Setup/check failed. Read the error above. Installation needs network access and 64-bit Python 3.12 is recommended.
if "%SEMG_NO_PAUSE%"=="0" pause
exit /b 1
