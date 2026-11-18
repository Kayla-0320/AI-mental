# Start the five services DETACHED from the DSH session, with timestamped logs.
#
# Why: running `node start-all.mjs` as a DSH background job means the whole
# process tree (all five services) is killed when that job is reclaimed.
# That is exactly what happened on 2026-10-11 16:58 -- the platform went down
# silently. Start-Process detaches it and redirects output into <repo>/logs.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start-detached.ps1

param(
  [int]$KeepLogs = 10
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $root 'logs'
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$outLog = Join-Path $logDir ('start-all-' + $stamp + '.out.log')
$errLog = Join-Path $logDir ('start-all-' + $stamp + '.err.log')

# Log rotation: keep only the newest $KeepLogs pairs.
Get-ChildItem $logDir -Filter 'start-all-*.out.log' -ErrorAction SilentlyContinue |
  Sort-Object LastWriteTime -Descending | Select-Object -Skip $KeepLogs | ForEach-Object {
    $base = $_.FullName -replace '\.out\.log$', ''
    Remove-Item ($base + '.out.log') -Force -ErrorAction SilentlyContinue
    Remove-Item ($base + '.err.log') -Force -ErrorAction SilentlyContinue
  }

$node = (Get-Command node).Source
Write-Host 'Starting five services (detached) ...'
Write-Host ('  stdout: ' + $outLog)
Write-Host ('  stderr: ' + $errLog)

$p = Start-Process -FilePath $node `
  -ArgumentList 'start-all.mjs', '--force' `
  -WorkingDirectory $root `
  -RedirectStandardOutput $outLog `
  -RedirectStandardError $errLog `
  -WindowStyle Hidden `
  -PassThru

$pidText = [string]$p.Id
Write-Host ('  started, PID ' + $pidText)
Write-Host ''
Write-Host 'status : node start-all.mjs --status'
Write-Host ('log    : Get-Content ' + $outLog + ' -Tail 40 -Wait')
Write-Host ('stop   : taskkill /PID ' + $pidText + ' /T /F')
