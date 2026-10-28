param(
    [ValidateSet('all', 'train', 'export', 'replay', 'check')]
    [string]$Phase = 'all',
    [string]$OutputDir = '',
    [ValidateRange(1, 64)]
    [int]$Threads = 4
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.runtime\Scripts\python.exe'
$scriptPath = Join-Path $projectRoot 'experiments\run_speechnet_comparison.py'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Project Python is missing: $pythonPath"
}
$runArguments = @('-u', $scriptPath, '--phase', $Phase, '--threads', "$Threads")
if ($OutputDir) {
    $runArguments += @('--output', $OutputDir)
}
& $pythonPath @runArguments
if ($LASTEXITCODE -ne 0) {
    throw "SpeechNet run failed (exit $LASTEXITCODE). Inspect run.log in the OUTPUT directory."
}
