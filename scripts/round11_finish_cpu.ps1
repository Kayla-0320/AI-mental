$ErrorActionPreference = 'Stop'
$taskPython = 'D:\Develop\AIC\.venv\Scripts\python.exe'
$taskOutput = 'D:\Develop\AIC\_国赛优化\round11'
while (-not (Test-Path -LiteralPath "$taskOutput\wave_stress\complete.json")) {
    Start-Sleep -Seconds 10
}
& $taskPython scripts\round11_cpu_resume.py faces --workers 8 *> "$taskOutput\faces_finish.log"
if ($LASTEXITCODE -ne 0) { throw '人脸提取失败' }
& $taskPython scripts\round11_cpu_resume.py signals --workers 8 *> "$taskOutput\signals_finish.log"
if ($LASTEXITCODE -ne 0) { throw '原始信号提取失败' }
Write-Output 'All external CPU media features complete'
