#!/usr/bin/env node
/**
 * 启动算法服务（本机 8001），并**在启动前**挡住会让语音功能静默失效的情况。
 *
 * 为什么需要这个脚本（真实事故）：
 *   算法服务原本没有固定启动方式，`algorithm/README.md` 写的是
 *   `uvicorn main:app --reload --port 8000`。于是用 PATH 上任意一个 Python 都能把它
 *   启起来占住端口 —— 而且**不会报错**。实际发生过：解释器里没装 sherpa-onnx，
 *   服务照常启动、`/health` 照常 200，但 `/api/v1/asr/ws` 直接 404，
 *   前端「语音输入」从此静默失效，只有翻 `/api/v1/asr/status` 才看得出原因。
 *
 * 三道闸门：
 *   1. 解释器只用项目自己的 venv（可用 ALGORITHM_PYTHON 覆盖），不用 PATH 上的。
 *   2. 启动前验证依赖 —— 缺 sherpa-onnx / fastapi 立刻失败，绝不让一个
 *      "能起来但功能是坏的"服务占住端口。
 *   3. 启动前探测 8001：如果已经有一份服务在跑，**直接问它健不健康**
 *      （`/api/v1/asr/status` 的 `streaming_available`）。
 *
 *   第 3 条刻意不看进程的 `ExecutablePath`/`CommandLine`：Windows 上 uv 建的 venv
 *   里 `Scripts\python.exe` 是硬链接，WMI 报出来的是**链接目标**的路径，
 *   据此判断"用错了解释器"会误报 —— 这一点已经踩过。
 *   服务的自述状态才是 ground truth。
 *
 * 用法：
 *   npm run dev:algorithm          # 检查后启动
 *   npm run dev:algorithm:check    # 只检查，不启动
 */

import { spawn, spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..');
const ALGORITHM_DIR = path.join(REPO, 'algorithm');
const HOST = '127.0.0.1';
const PORT = 8001;
const REQUIRED_MODULES = ['fastapi', 'uvicorn', 'sherpa_onnx'];

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

/** 按优先级找解释器：环境变量 → 仓库自身/各级上级目录的 .venv → 报错。 */
function resolvePython() {
  if (process.env.ALGORITHM_PYTHON) {
    return { exe: process.env.ALGORITHM_PYTHON, why: 'ALGORITHM_PYTHON 环境变量', found: true };
  }

  const isWin = process.platform === 'win32';
  const rel = isWin ? ['Scripts', 'python.exe'] : ['bin', 'python'];

  // 逐级向上找 —— 本机的 venv 其实在仓库上面三层，写死层数很脆
  const tried = [];
  let dir = REPO;
  for (let i = 0; i < 6; i++) {
    const exe = path.join(dir, '.venv', ...rel);
    tried.push(exe);
    if (existsSync(exe)) {
      return { exe, why: i === 0 ? '仓库内 .venv' : `上级 ${i} 层的 .venv`, found: true, tried };
    }
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }

  return { exe: isWin ? 'python' : 'python3', why: 'PATH 上的默认解释器', found: false, tried };
}

async function httpJson(url, timeoutMs = 4000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(url, { signal: ctrl.signal });
    if (!res.ok) return { ok: false, status: res.status };
    return { ok: true, status: res.status, json: await res.json() };
  } catch (err) {
    return { ok: false, error: String(err?.message || err) };
  } finally {
    clearTimeout(timer);
  }
}

/** 占用 8001 的进程信息（仅用于给出"怎么停掉它"的命令）。 */
function portOwner(port) {
  if (process.platform !== 'win32') return null;
  const r = spawnSync('powershell.exe', ['-NoProfile', '-Command',
    `$c = Get-NetTCPConnection -LocalPort ${port} -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1;` +
    `if ($c) { $p = Get-CimInstance Win32_Process -Filter "ProcessId=$($c.OwningProcess)"; ` +
    `"$($c.OwningProcess)|$($p.CommandLine)" }`,
  ], { encoding: 'utf8' });
  const s = (r.stdout || '').trim();
  if (!s) return null;
  const idx = s.indexOf('|');
  return { pid: s.slice(0, idx), cmd: s.slice(idx + 1) };
}

// ── 闸门 1：解释器
const py = resolvePython();
if (!py.found) {
  fail('没有找到项目虚拟环境', [
    '从仓库目录向上找了 6 级，都没有 .venv/Scripts/python.exe：',
    ...py.tried.map((t) => `  ${t}`),
    '',
    '算法服务的依赖（sherpa-onnx 等）通常只装在项目 venv 里，',
    '用 PATH 上的解释器很可能缺依赖 —— 那会让「语音输入」静默失效。',
    '',
    '请显式指定它，例如：',
    '  set ALGORITHM_PYTHON=D:\\path\\to\\.venv\\Scripts\\python.exe',
  ]);
}
console.log(`解释器: ${py.exe}  (${py.why})`);

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
    '缺 sherpa-onnx 的后果特别隐蔽：服务照常启动、/health 照常 200，',
    '但 /api/v1/asr/ws 会 404，前端「语音输入」从此静默失效。',
    '',
    '安装：',
    `  "${py.exe}" -m pip install -r "${path.join(ALGORITHM_DIR, 'requirements.txt')}"`,
    '',
    '（本机若用 uv：uv pip install --python "<上面的解释器>" -r requirements.txt）',
  ]);
}
console.log(`依赖: ${REQUIRED_MODULES.join(' / ')} 齐全  (Python ${pyver})`);

// ── 闸门 3：端口 —— 直接问已经在跑的那份服务健不健康
const owner = portOwner(PORT);
if (owner) {
  const health = await httpJson(`http://${HOST}:${PORT}/api/v1/asr/status`);
  const stopHint = [
    '',
    `停止它：Stop-Process -Id ${owner.pid} -Force`,
    `  （命令行：${(owner.cmd || '').trim().slice(0, 160)}）`,
  ];

  if (health.ok && health.json) {
    const s = health.json;
    if (s.streaming_available) {
      console.log(
        `${YELLOW}端口: ${PORT} 已有服务在跑，且健康${OFF}` +
        ` （streaming_available=true, engine=${s.streaming_engine}）`,
      );
      console.log(`  /api/v1/asr/status 正常 —— 语音输入可用，不需要再起一份。`);
      console.log(`  若要重启，先停掉：Stop-Process -Id ${owner.pid} -Force`);
      process.exit(0);
    }

    fail(`${PORT} 上的算法服务**功能不完整**（这就是语音输入坏掉的样子）`, [
      `  streaming_available = ${s.streaming_available}`,
      `  streaming_error     = '${s.streaming_error || ''}'`,
      `  available/engine    = ${s.available} / ${s.engine}  (Python 侧的离线引擎也是坏的)`,
      ...stopHint,
      '',
      '停掉它，再用本脚本启动（本脚本会用带 sherpa-onnx 的解释器）。',
    ]);
  }

  fail(`${PORT} 被占用，但占用者不是可用的算法服务`, [
    `  /api/v1/asr/status 探测结果：${health.status ? `HTTP ${health.status}` : health.error}`,
    ...stopHint,
  ]);
}

console.log(`端口: ${PORT} 空闲`);

if (CHECK_ONLY) {
  console.log(`\n${GREEN}✓ 检查通过（--check 模式，未启动服务）${OFF}\n`);
  process.exit(0);
}

// ── 启动：stdio inherit —— 日志直接进当前终端，也避免沙箱下管道捕获的问题
console.log(`\n启动: http://${HOST}:${PORT}   文档: http://${HOST}:${PORT}/docs\n`);
const child = spawn(
  py.exe,
  ['-m', 'uvicorn', 'main:app', '--host', HOST, '--port', String(PORT), ...process.argv.slice(2)],
  { cwd: ALGORITHM_DIR, stdio: 'inherit' },
);
child.on('exit', (code, signal) => process.exit(code ?? (signal ? 1 : 0)));
