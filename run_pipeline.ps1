# Runs the validated end-to-end pipeline (Phases 3-7) from the project root.
# Deterministic: every stage is seeded and idempotent, artifacts are regenerated
# identically. Stops immediately on the first failing stage.

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Virtual environment not found at $Python. Run: python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
}

$env:PYTHONUTF8 = "1"

$Stages = @(
    @{ Name = "Phase 3 - Data Cleaning & Validation";     Script = "src\data_cleaning.py" },
    @{ Name = "Phase 4 - Exploratory Data Analysis";      Script = "src\eda.py" },
    @{ Name = "Phase 5 - Feature Engineering & Selection"; Script = "src\feature_engineering.py" },
    @{ Name = "Phase 6 - Model Training & Evaluation";    Script = "src\train.py" },
    @{ Name = "Phase 7 - Risk Scoring & Power BI Prep";   Script = "src\risk_scoring.py" }
)

Write-Host "=== CUSTOMER CHURN PIPELINE - FULL RUN ===" -ForegroundColor Cyan
$Started = Get-Date

foreach ($Stage in $Stages) {
    Write-Host ""
    Write-Host (">>> {0}  ({1})" -f $Stage.Name, $Stage.Script) -ForegroundColor Yellow
    & $Python $Stage.Script
    if ($LASTEXITCODE -ne 0) {
        throw "Stage FAILED: $($Stage.Name) (exit code $LASTEXITCODE). Pipeline stopped."
    }
    Write-Host ("<<< {0} - OK" -f $Stage.Name) -ForegroundColor Green
}

$Elapsed = (Get-Date) - $Started
Write-Host ""
Write-Host ("=== ALL {0} STAGES COMPLETED SUCCESSFULLY in {1:mm\:ss} ===" -f $Stages.Count, $Elapsed) -ForegroundColor Cyan
Write-Host "Next: .\.venv\Scripts\python.exe src\project_validation.py   (final integrity check)"
