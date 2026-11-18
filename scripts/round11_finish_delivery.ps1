$ErrorActionPreference = 'Stop'
$taskPython = 'D:\Develop\AIC\.venv\Scripts\python.exe'
$taskOutput = 'D:\Develop\AIC\_国赛优化\round11'
$env:PYTHONPATH = 'algorithm'
while (-not ((Test-Path -LiteralPath "$taskOutput\emotiontalk_features\signal_features\manifest.json") -and
    (Test-Path -LiteralPath "$taskOutput\emotiontalk_features\face_features\manifest.json") -and
    (Test-Path -LiteralPath "$taskOutput\historical_diagnostic\results.json"))) {
    Start-Sleep -Seconds 10
}
& $taskPython -m experiments.round11_external_finish evaluate *> "$taskOutput\external_evaluation.log"
if ($LASTEXITCODE -ne 0) { throw '外部锁定M3评测失败' }
& $taskPython scripts\round11_quality_strata.py *> "$taskOutput\quality_strata.log"
if ($LASTEXITCODE -ne 0) { throw '可观测质量分层失败' }
& $taskPython scripts\round11_replay_bundle.py --retrained "$taskOutput\validation\refit7b\model.pkl" *> "$taskOutput\validation\final_replay.log"
if ($LASTEXITCODE -ne 0) { throw '权重/选型摘要与独立重训重放不符' }
& $taskPython scripts\round11_runtime.py *> "$taskOutput\runtime.log"
if ($LASTEXITCODE -ne 0) { throw '轻量原媒体成本实测失败' }
& $taskPython scripts\round11_runtime_staged.py *> "$taskOutput\runtime_staged.log"
if ($LASTEXITCODE -ne 0) { throw '3B/7B分阶段实际请求失败' }
& $taskPython scripts\round11_forecast_replay.py *> "$taskOutput\forecast_replay.log"
if ($LASTEXITCODE -ne 0) { throw '原媒体至未来预测重放失败' }
& $taskPython scripts\round11_finalize_state.py
if ($LASTEXITCODE -ne 0) { throw '最终交接记录失败' }
& $taskPython scripts\round11_final_report.py *> "$taskOutput\final_report.log"
if ($LASTEXITCODE -ne 0) { throw '最终报告尚有未通过验收项' }
& $taskPython scripts\round11_pack_review.py *> "$taskOutput\review_bundle.log"
if ($LASTEXITCODE -ne 0) { throw '可审阅压缩包CRC失败' }
Write-Output 'All Round11 evidence, reports and defense materials completed'
