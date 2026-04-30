# Run benchmark in background, survive terminal close
# API key is read from .env file (kept locally, never committed)
$root = Split-Path $PSScriptRoot -Parent
$envFile = Join-Path $root ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^OPENROUTER_API_KEY=(.+)$') { $apiKey = $matches[1] }
    }
}
if (-not $apiKey) { throw "OPENROUTER_API_KEY not found in .env file" }
$logFile = Join-Path $PSScriptRoot "benchmark_output.log"

"=== Benchmark started at $(Get-Date) ===" | Out-File $logFile

Push-Location $root
try {
    python benchmark/compare_models.py --api-key $apiKey *>> $logFile
    "=== Benchmark done at $(Get-Date), generating plot ===" | Out-File $logFile -Append
    python benchmark/plot_convergence.py *>> $logFile
    "=== Plot done at $(Get-Date) ===" | Out-File $logFile -Append
} finally {
    Pop-Location
}
