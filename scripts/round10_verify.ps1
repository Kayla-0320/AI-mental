#Requires -Version 5.1
param(
    [ValidateSet('speech_final', 'robust_final', 'large_final', 'holdout_final', 'direct_final', 'refined_final')]
    [string]$Stage = 'refined_final',
    [string]$PythonPath = 'D:\Develop\AIC\.venv\Scripts\python.exe'
)

$ErrorActionPreference = 'Stop'
$TaskRepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath (Join-Path $TaskRepoRoot 'algorithm')
try {
    & $PythonPath -X utf8 -m pytest tests\test_round10_protocol.py tests\test_round9_addition_models.py tests\test_knowledge_moe.py -q
    if ($LASTEXITCODE -ne 0) { throw '第十轮回归验证失败。' }
}
finally {
    Pop-Location
}
& $PythonPath -X utf8 (Join-Path $PSScriptRoot 'round10_audit.py') --stage $Stage
if ($LASTEXITCODE -ne 0) { throw '独立复算或模型重放失败。' }
& $PythonPath -X utf8 (Join-Path $PSScriptRoot 'round10_report.py')
if ($LASTEXITCODE -ne 0) { throw '报告生成失败。' }
