@echo off
setlocal
cd /d "%~dp0"
echo Train FIVE independent personal CNNs: each person's own data, grouped 80/20.
echo This is a NEW experiment; the earlier cross-person models are preserved.
echo Logs: models\cnn_individual_80_20_20261030\S0xx\fit.log
echo Results: models\cnn_individual_80_20_20261030\results.txt
"%~dp0.runtime\Scripts\python.exe" -u -X utf8 "%~dp0experiments\run_cnn_individual_80_20.py" --jobs 2 --threads 4
set "cnn_exit=%errorlevel%"
if not "%cnn_exit%"=="0" echo Run failed or interrupted. Inspect the log. Completed training stages can be resumed.
if "%cnn_exit%"=="0" echo Finished. Open models\cnn_individual_80_20_20261030\results.txt
pause
exit /b %cnn_exit%
