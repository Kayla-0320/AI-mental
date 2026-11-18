# MindGarden launcher: set bundled model paths, then start the whole stack.
# Usage:  powershell -NoProfile -ExecutionPolicy Bypass -File .\start-package.ps1
#         .\start-package.cmd            (double-click friendly)
# Optional args are forwarded to start-all.mjs, e.g. --no-tts / --status / --force
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if ($root -match '[^\x00-\x7F]') {
    Write-Host '[warn] The package path contains non-ASCII characters.' -ForegroundColor Yellow
    Write-Host '       sherpa-onnx VITS (kaldifst) cannot open .fst files under such a path,' -ForegroundColor Yellow
    Write-Host '       so TTS will report available=false. Move the folder to an ASCII path,' -ForegroundColor Yellow
    Write-Host '       for example D:\MindGarden, and run this script again.' -ForegroundColor Yellow
}

$clip = Join-Path $root 'external_models\chinese-clip-vit-base-patch16'
$bge  = Join-Path $root 'external_models\bge-base-zh-v1.5'
if (Test-Path -LiteralPath $clip) { $env:ROUND3_VLM_ENCODER = $clip }
if (Test-Path -LiteralPath $bge)  { $env:SCENE_EMBED_MODEL  = $bge }

$py = Join-Path $root '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $py) {
    $env:AIC_PYTHON = $py
    $env:ALGORITHM_PYTHON = $py
} else {
    Write-Host '[warn] .venv not found; falling back to the launchers default interpreter probing.' -ForegroundColor Yellow
    Write-Host '       Run: python -m venv .venv ; .\.venv\Scripts\python.exe -m pip install -r algorithm\requirements.txt' -ForegroundColor Yellow
}

$envFile = Join-Path $root '.env'
if (-not (Test-Path -LiteralPath $envFile)) {
    Copy-Item -LiteralPath (Join-Path $root '.env.example') -Destination $envFile
    Write-Host '[info] created .env from .env.example (algorithm side).' -ForegroundColor Yellow
}

# Node backend reads server/.env (dotenv, cwd=server), NOT the repo-root .env.
$serverEnv = Join-Path $root 'server\.env'
if (-not (Test-Path -LiteralPath $serverEnv)) {
    Copy-Item -LiteralPath (Join-Path $root 'server\.env.example') -Destination $serverEnv
    Write-Host '[info] created server/.env from server/.env.example.' -ForegroundColor Yellow
    Write-Host '       Fill the SAME DASHSCOPE_API_KEY into both .env and server/.env.' -ForegroundColor Yellow
}

Write-Host '[start] node start-all.mjs' -ForegroundColor Cyan
& node (Join-Path $root 'start-all.mjs') @args
exit $LASTEXITCODE
