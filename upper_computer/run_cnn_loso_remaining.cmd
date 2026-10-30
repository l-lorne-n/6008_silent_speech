@echo off
setlocal
cd /d "%~dp0"
echo Resume current CNN LOSO: keep completed S011/S012, train S013/S014/S015.
echo Logs: models\cnn_current_loso_20261030\S0xx\continue_fit.log
echo Final results: models\cnn_current_loso_20261030\results.txt
echo You can rerun this file after interruption. Completed stages are verified and reused.
"%~dp0.runtime\Scripts\python.exe" -u -X utf8 "%~dp0experiments\continue_cnn_current_loso.py" --jobs 2 --threads 4
set "cnn_exit=%errorlevel%"
if not "%cnn_exit%"=="0" echo Run failed or interrupted. Keep the output directory and inspect the logs before retrying.
if "%cnn_exit%"=="0" echo Finished. Open models\cnn_current_loso_20261030\results.txt
pause
exit /b %cnn_exit%
