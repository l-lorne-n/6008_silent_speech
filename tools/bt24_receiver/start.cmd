@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
if defined BT24_PYTHON goto validate_python
for /f "delims=" %%P in ('py -3.12 -c "import sys; print(sys.executable)" 2^>nul') do set "BT24_PYTHON=%%P"
if defined BT24_PYTHON goto validate_python
for /f "delims=" %%P in ('python -c "import sys; assert sys.version_info[:2] == (3,12); print(sys.executable)" 2^>nul') do set "BT24_PYTHON=%%P"
if defined BT24_PYTHON goto validate_python
set "BT24_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%BT24_PYTHON%" goto validate_python
echo Python 3.12 was not found.
echo Install Python 3.12 from https://www.python.org/downloads/windows/
echo Enable the Python launcher and Tcl/Tk during installation, then try again.
pause
exit /b 1

:validate_python
"%BT24_PYTHON%" -c "import sys, tkinter; assert sys.version_info[:2] == (3,12)" >nul 2>&1
if errorlevel 1 (
  echo Python 3.12 with Tcl/Tk is required. Check your Python installation.
  pause
  exit /b 1
)
"%BT24_PYTHON%" -c "import sys; sys.path.insert(0,'.deps'); from bleak.backends.winrt.client import BleakClientWinRT" >nul 2>&1
if not errorlevel 1 goto run_receiver
echo Installing Bluetooth dependencies into this folder. Internet is required.
"%BT24_PYTHON%" -m pip install --disable-pip-version-check --only-binary=:all: --no-cache-dir --upgrade --target ".deps" -r requirements.txt
if errorlevel 1 (
  echo Installation failed. Check your Internet connection and see README.md.
  pause
  exit /b 1
)

:run_receiver
"%BT24_PYTHON%" receiver.py %*
if errorlevel 1 pause
