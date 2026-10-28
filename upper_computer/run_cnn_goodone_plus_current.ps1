param(
    [ValidateSet('all', 'prepare', 'check')]
    [string]$Phase = 'all',
    [string]$OutputDir = '',
    [switch]$Resume,
    [ValidateRange(1, 64)]
    [int]$Threads = 4
)
$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path $PSScriptRoot '.runtime\Scripts\python.exe'
$scriptPath = Join-Path $PSScriptRoot 'experiments\run_cnn_goodone_plus_current.py'
$runArguments = @('-u', $scriptPath, '--phase', $Phase, '--threads', "$Threads")
if ($OutputDir) { $runArguments += @('--output', $OutputDir) }
if ($Resume) { $runArguments += '--resume' }
$env:PYTHONIOENCODING = 'utf-8'
& $pythonPath @runArguments
if ($LASTEXITCODE -ne 0) {
    throw "CNN run failed (exit $LASTEXITCODE). Inspect run.log in the OUTPUT directory."
}
