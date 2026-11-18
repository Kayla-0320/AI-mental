$ErrorActionPreference = 'Stop'
$taskPython = 'D:\Develop\AIC\.venv\Scripts\python.exe'
$taskOutput = 'D:\Develop\AIC\_国赛优化\round11'
$env:PYTHONPATH = 'algorithm'
# 只等待本轮已核实的CLIP任务；不修改其他用户进程。
while (-not (Test-Path -LiteralPath "$taskOutput\emotiontalk_features\clip_features\manifest.json")) {
    $taskProcess = Get-Process -Id 57776 -ErrorAction SilentlyContinue
    if ($null -eq $taskProcess) { throw 'CLIP未完成且进程已退出，需检查而不能继续推断完成。' }
    Start-Sleep -Seconds 10
}
& $taskPython scripts\round11_ch_complete.py sense *> "$taskOutput\ch_sense.log"
if ($LASTEXITCODE -ne 0) { throw '中文语音最终输入失败' }
& $taskPython scripts\round11_ch_complete.py msp *> "$taskOutput\ch_msp.log"
if ($LASTEXITCODE -ne 0) { throw 'MSP最终输入失败' }
& $taskPython scripts\round11_wave_stress.py *> "$taskOutput\wave_stress.log"
if ($LASTEXITCODE -ne 0) { throw '真实波形压力失败' }
foreach ($taskMode in @('7b','3b','light')) {
    & $taskPython -m experiments.round11_final fit --mode $taskMode *> "$taskOutput\fit_$taskMode.log"
    if ($LASTEXITCODE -ne 0) { throw "最终模型拟合失败 $taskMode" }
}
& $taskPython -m experiments.round11_robustness *> "$taskOutput\robustness.log"
if ($LASTEXITCODE -ne 0) { throw '鲁棒性诊断失败' }
& $taskPython -m experiments.round11_final evaluate *> "$taskOutput\historical.log"
if ($LASTEXITCODE -ne 0) { throw '历史一次诊断失败' }
& $taskPython -m experiments.round11_external_finish adapt *> "$taskOutput\adaptation.log"
if ($LASTEXITCODE -ne 0) { throw '域适应开发选型失败' }
Write-Output 'CH final models, robustness and historical diagnostic complete'
