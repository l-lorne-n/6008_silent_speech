$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$demoUrl = 'http://127.0.0.1:8765'
try {
    $demoStatus = Invoke-RestMethod -Uri "$demoUrl/api/status" -TimeoutSec 2
    if ($demoStatus.app -eq 'quietly') {
        if ($demoStatus.version -lt 3) {
            Write-Host 'An older Quietly server is still running. Close its terminal, then run start.cmd again.'
            exit 1
        }
        Start-Process $demoUrl
        exit 0
    }
} catch { }
$demoPython = $null
$demoCandidates = @(
    (Join-Path $PSScriptRoot '..\upper_computer\.runtime\Scripts\python.exe'),
    (Join-Path $PSScriptRoot '..\upper_computer\.python\python.exe'),
    (Join-Path $PSScriptRoot '..\upper_computer\.venv\Scripts\python.exe'),
    'D:\python\python.exe'
)
foreach ($demoCandidate in $demoCandidates) {
    if (Test-Path -LiteralPath $demoCandidate) { $demoPython = $demoCandidate; break }
}
if (-not $demoPython) {
    $demoCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($demoCommand) { $demoPython = $demoCommand.Source }
}
if (-not $demoPython) {
    Write-Host 'Python 3.9+ is required. Install Python or open index.html for manual preview.'
    exit 1
}
& $demoPython (Join-Path $PSScriptRoot 'server.py') --open
exit $LASTEXITCODE
