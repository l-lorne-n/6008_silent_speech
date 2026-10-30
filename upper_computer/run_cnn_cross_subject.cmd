@echo off
setlocal
cd /d "%~dp0"
echo Current mid64_temporal3 five-person LOSO: fresh weights, train four, test one.
echo Published results are in model_releases\cnn_20261030\cross_subject.
echo This run writes independent reproducibility outputs under models\cnn_cross_subject_reproduction.
set "cnn_resume="
if exist "models\cnn_cross_subject_reproduction\prepared_sha256.json" set "cnn_resume=--resume"
"%~dp0.runtime\Scripts\python.exe" -u -X utf8 "%~dp0experiments\run_cnn_current_loso.py" --output "%~dp0models\cnn_cross_subject_reproduction" --jobs 2 --threads 4 %cnn_resume%
set "cnn_exit=%errorlevel%"
if not "%cnn_exit%"=="0" echo Failed or interrupted. Keep logs and read model_releases\cnn_20261030\README.md.
pause
exit /b %cnn_exit%
