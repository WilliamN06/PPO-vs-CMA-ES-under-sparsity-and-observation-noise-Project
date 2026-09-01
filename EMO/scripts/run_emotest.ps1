$ErrorActionPreference = "Stop"
$emoDir = "C:\Users\Willi\Documents\Opencode Projects\Research Internship\EMO"
Set-Location $emoDir
Write-Host "=== Running EMO basic-trends pilot ==="
python tests/test_basic_trends.py
Write-Host "`nDone. See test_run/trend_summary.txt"
