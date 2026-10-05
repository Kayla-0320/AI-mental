#!/usr/bin/env node
/**
 * 启动 CosyVoice3 语音合成微服务（本机 8002），并**在启动前**挡住会让它崩掉的情况。
 *
 * 为什么需要这个脚本（真实事故，2026-10-05）：
 *   算法服务用 `TTS_BACKEND=cosyvoice` 指着 8002。8002 不在时，算法侧的
 *   引擎会报 unavailable、**一句话都不出声** —— 与「音色没接上」表现完全一样。
 *   现场就是：改了 .env、代码也接了，但没人起 8002，于是听起来还是旧的。
 *
 * 更麻烦的是这台机器的内存配置：页面文件被固定成 4 GB（提交上限 ~35.4 GB），
 * 跑满整套服务后余量只剩几百 MB。CosyVoice3 加载模型峰值约需 9 GB，
 * 不够时的表现**不是**一条清晰的报错，而是：
 *   - ONNX 报 `bad allocation`、`OSError: 页面文件太小 (os error 1455)`
 *   - 或者进程直接原生崩溃（c10.dll / fbgemm.dll 访问冲突），**没有任何 Python 回溯**
 *   - 严重时同一分钟连 dwm.exe、msedge.exe 一起崩
 *   - 每次失败还会残留数 GB 提交内存，把下一次尝试推得更死
 * 所以这里必须先看余量再启动。
 *
 * 三道闸门：
 *   1. 解释器必须是 CosyVoice 自己的 venv（它依赖 torch 2.7 + numpy<2，
 *      与算法服务的 torch 2.11 + numpy 2.5 直接冲突，不能混用）。
 *   2. 依赖齐全（fastapi / uvicorn / torch / onnxruntime）。
 *   3. 提交内存余量足够，且 8002 上没有已经在跑的服务。
 *
 * 用法：
 *   npm run dev:tts            # 检查后启动（前台，Ctrl+C 结束）
 *   npm run dev:tts -- --check # 只检查，不启动
 */
import { spawn, spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..');
const SERVICE_DIR = path.join(REPO, 'algorithm', 'tts', 'cosyvoice_service');
const HOST = '127.0.0.1';
const PORT = 8002;

/** 模型权重与 CosyVoice 仓库的根目录 */
const ROOT = process.env.COSYVOICE_ROOT || 'D:\\Develop\\AIC\\_tts_candidates\\cosyvoice';

//: 加载峰值实测约 9.6 GB，但**不能按峰值卡**：提交内存是全系统共享的，
//: 其他进程在这 30 秒里也会涨。标定依据（这是**服务进程内**的读数，
//: 它在 `import torch` 之后 —— 光导入 torch+onnxruntime 就吃 ~1.4 GB，
//: 整个进程起来 4-6 GB，所以本脚本在启动**前**读到的值会明显偏高）：
//:   观测到原生崩溃时 服务内读数 = 10997 MB
//:   成功加载时       服务内读数 ≈ 12000 MB 以上
//: 本脚本在启动前检查，因此用同样的 12 GB 作为下限，宁可少一点余量也不要漏挡。
const MIN_COMMIT_MB = parseInt(process.env.COSYVOICE_MIN_COMMIT_MB || '12288', 10);

const REQUIRED_MODULES = ['fastapi', 'uvicorn', 'torch', 'onnxruntime'];
const CHECK_ONLY = process.argv.includes('--check');

const RED = '\x1b[31m';
const GREEN = '\x1b[32m';
const YELLOW = '\x1b[33m';
const OFF = '\x1b[0m';

function fail(title, lines) {
  console.error(`\n${RED}✗ ${title}${OFF}`);
  for (const l of lines) console.error(`  ${l}`);
  console.error('');
  process.exit(1);
}

function resolvePython() {
  const isWin = process.platform === 'win32';
  const rel = isWin ? ['Scripts', 'python.exe'] : ['bin', 'python'];
  const exe = path.join(ROOT, '.venv', ...rel);
  return { exe, found: existsSync(exe) };
}

/** 读取剩余可提交内存（MB）。用 PowerShell 读计数器，避免引入新依赖。 */
function commitHeadroomMb() {
  if (process.platform !== 'win32') return null;
  const r = spawnSync('powershell.exe', ['-NoProfile', '-Command',
    '$p = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory -ErrorAction SilentlyContinue;' +
    'if ($p) { [int](($p.CommitLimit - $p.CommittedBytes)/1MB) }',
  ], { encoding: 'utf8' });
  const n = parseInt((r.stdout || '').trim(), 10);
  return Number.isFinite(n) ? n : null;
}

/** 8002 上是否已经有服务在跑（服务的自述状态才是事实）。 */
async function probeHealth(timeoutMs = 3000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`http://${HOST}:${PORT}/health`, { signal: ctrl.signal });
    if (!res.ok) return { ok: false, status: res.status };
    return { ok: true, status: res.status, json: await res.json() };
  } catch (err) {
    return { ok: false, error: String(err?.message || err) };
  } finally {
    clearTimeout(timer);
  }
}

// ── 闸门 1：解释器
const py = resolvePython();
if (!py.found) {
  fail('找不到 CosyVoice 的虚拟环境', [
    `期望位置：${py.exe}`,
    '',
    'CosyVoice3 依赖 torch==2.7.0+cu128 与 numpy<2，**不能**用算法服务的解释器',
    '（那边是 torch 2.11 + numpy 2.5，直接冲突）。见 service 目录下 README.md。',
    '',
    `若权重装在别处，设置 COSYVOICE_ROOT 指向含 .venv/ 与 pretrained_models/ 的目录。`,
  ]);
}
console.log(`解释器: ${py.exe}`);
console.log(`模型根目录: ${ROOT}`);

// ── 闸门 2：依赖
const probe = spawnSync(py.exe, ['-c', [
  'import importlib.util, sys',
  `missing = [m for m in ${JSON.stringify(REQUIRED_MODULES)} if importlib.util.find_spec(m) is None]`,
  'print("MISSING:" + ",".join(missing))',
  'print("PY:" + sys.version.split()[0])',
].join('; ')], { encoding: 'utf8' });

if (probe.error) fail('无法运行该解释器', [String(probe.error.message), `解释器: ${py.exe}`]);
const probeOut = `${probe.stdout || ''}${probe.stderr || ''}`;
if (probe.status !== 0) fail('依赖探测失败', probeOut.trim().split('\n').slice(0, 6));

const missing = (/MISSING:(.*)/.exec(probeOut)?.[1] || '').trim();
const pyver = (/PY:(.*)/.exec(probeOut)?.[1] || '').trim();
if (missing) {
  fail(`解释器缺少依赖: ${missing}`, [
    `解释器: ${py.exe}  (Python ${pyver})`,
    '',
    '按 service 目录 README.md 里的安装命令补齐（注意 onnxruntime 与 torch 的',
    'CUDA 版本必须匹配本机驱动；本机是 torch 2.7.0+cu128）。',
  ]);
}
console.log(`依赖: ${REQUIRED_MODULES.join(' / ')} 齐全  (Python ${pyver})`);

// ── 闸门 3a：端口 —— 已经在跑就别再起一份
// ⚠️ 同时起两份会在**加载阶段**把提交内存吃穿并原生崩溃（实测两次），所以必须先问。
const health = await probeHealth();
if (health.ok && health.json) {
  const s = health.json;
  console.log(`${YELLOW}端口: ${PORT} 已有 CosyVoice 服务在跑${OFF}`);
  console.log(`  available=${s.available} engine=${s.engine} speed=${s.speed} error='${s.error || ''}'`);
  if (s.available) {
    console.log(`  ${GREEN}可以直接用，不需要再起一份。${OFF}`);
    process.exit(0);
  }
  fail('8002 上的服务在跑，但模型没加载成功', [
    `  error = ${s.error}`,
    '',
    '停掉它再重试：先找到并结束占用 8002 的进程，然后重新运行本脚本。',
    '若 error 提到「可提交内存余量不足 / 页面文件太小 / os error 1455」，',
    '见下面的说明。',
  ]);
}

// ── 闸门 3b：提交内存余量
const headroom = commitHeadroomMb();
if (headroom !== null) {
  if (headroom < MIN_COMMIT_MB) {
    fail(`可提交内存余量不足：${headroom} MB（需要约 ${MIN_COMMIT_MB} MB）`, [
      'CosyVoice3 加载模型峰值约 9 GB。余量不足时**不是**一条干净报错，而是：',
      '  - ONNX bad allocation / OSError: 页面文件太小 (os error 1455)',
      '  - 或原生崩溃（c10.dll / fbgemm.dll 访问冲突），没有 Python 回溯',
      '  - 严重时整机其他进程（含 dwm.exe）一起崩，且失败残留数 GB 提交内存',
      '',
      '两种处置，任选其一：',
      '',
      '  A) 把系统页面文件改成「系统托管」（根治，推荐，需要管理员）：',
      '     设置 → 系统 → 关于 → 高级系统设置 → 性能「设置」',
      '       → 高级 → 虚拟内存「更改」→ 勾选「自动管理所有驱动器的分页文件大小」→ 重启',
      '     当前是固定 4096 MB。',
      '',
      '  B) 临时腾内存：关掉 Docker Desktop/WSL、浏览器、QQ/微信等大占用程序，',
      '     并确保算法服务(8001)不在加载/运行中，再运行本脚本。',
    ]);
  }
  console.log(`提交内存余量: ${headroom} MB（下限 ${MIN_COMMIT_MB} MB）`);
} else {
  console.log(`${YELLOW}提交内存余量: 读不到（非 Windows 或计数器不可用），跳过此项检查${OFF}`);
}

console.log(`端口: ${PORT} 空闲`);

if (CHECK_ONLY) {
  console.log(`\n${GREEN}✓ 检查通过（--check 模式，未启动服务）${OFF}\n`);
  process.exit(0);
}

// ── 启动：stdio inherit —— 日志直接进当前终端
console.log(`\n启动: http://${HOST}:${PORT}/docs   （首次请求才会加载模型，约 20 s）\n`);
console.log('提示：算法服务需设置 TTS_BACKEND=cosyvoice（已写在仓库根 .env）。');
console.log('      核对实际生效引擎：GET http://127.0.0.1:8001/api/v1/tts/status\n');

const child = spawn(
  py.exe,
  ['-u', '-m', 'uvicorn', 'server:app', '--host', HOST, '--port', String(PORT)],
  { cwd: SERVICE_DIR, stdio: 'inherit', env: { ...process.env, COSYVOICE_ROOT: ROOT } },
);
child.on('exit', (code, signal) => process.exit(code ?? (signal ? 1 : 0)));
