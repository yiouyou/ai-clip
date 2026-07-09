$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$SmokeDir = Join-Path $Root ".testdata\smoke"
New-Item -ItemType Directory -Force -Path $SmokeDir | Out-Null

$DataDir = ($SmokeDir -replace "\\", "/")
$ConfigPath = Join-Path $SmokeDir "config.yaml"
@"
data_dir: $DataDir
"@ | Set-Content -Path $ConfigPath -Encoding UTF8

Write-Host "== doctor =="
uv run ai-clip doctor --config $ConfigPath

Write-Host "== status json =="
$StatusJson = uv run ai-clip status -p empty --config $ConfigPath --json
$Status = $StatusJson | ConvertFrom-Json
if ($Status.project -ne "empty") {
    throw "unexpected status project: $($Status.project)"
}
if ($Status.storyboard.exists -ne $false) {
    throw "empty smoke project should not have storyboard.json"
}

Write-Host "== targeted tests =="
uv run pytest `
    tests/test_core.py `
    tests/test_cli.py::test_project_status_json_handles_missing_storyboard `
    tests/test_research.py::test_run_source_draft_injects_existing_research `
    -q

Write-Host "smoke passed"
