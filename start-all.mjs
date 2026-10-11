#!/usr/bin/env node
/**
 * 全功能启动器 —— 按**内存约束**决定的顺序拉起五端。
 *
 * 为什么需要它（本机实测）：
 *   CosyVoice3（:8002，.env 里 TTS_BACKEND=cosyvoice）加载模型需要约 12 GB 可提交内存，
 *   而本机页面文件被固定成 4 GB（提交上限约 36 GB）。算法服务(8001) 常驻约 3.1 GB、
 *   WSL 的 vmmem 约 1.7 GB 且**不会自动归还**，于是：
 *     - 先起 8001 再起 8002 → 8002 的内存闸门（下限 12288 MB）直接拒绝加载，
 *       算法侧引擎降级为 unavailable，表现是「语音一句话都不出声」；
 *     - 反过来先起 8002（此时内存最宽松）再起 8001，两边都能正常加载。
 *   `node server.js` 的固定顺序是 db→algo→api→web，正好踩中第一种，所以这里做编排。
 *
 * 顺序：PostgreSQL → TTS(8002，等模型就绪) → 算法(8001) → 后端(3000) → 前端(5173)
 *
 * 用法：
 *   node start-all.mjs              # 全量拉起（已在跑的服务自动跳过）
 *   node start-all.mjs --force      # 端口被占用时先结束占用进程再启动
 *   node start-all.mjs --status     # 只打印五端状态
 *   node start-all.mjs --no-tts     # 跳过 TTS 微服务
 */
import { spawn, spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import net from 'node:net';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const PORTS = { db: 5432, algo: 8001, tts: 8002, api: 3000, web: 5173 };
const HOST = '127.0.0.1';
const argv = new Set(process.argv.slice(2));
const FORCE = argv.has('--force');
const C = {
  grn: (s) => `\x1b[32m${s}\x1b[0m`,
  red: (s) => `\x1b[31m${s}\x1b[0m`,
  yel: (s) => `\x1b[33m${s}\x1b[0m`,
  gry: (s) => `\x1b[90m${s}\x1b[0m`,
  bold: (s) => `\x1b[1m${s}\x1b[0m`,
};
const say = (s) => console.log(`${C.gry(' ·')} ${s}`);
const ok = (s) => console.log(`${C.grn(' ✓')} ${s}`);
const bad = (s) => console.log(`${C.red(' ✗')} ${s}`);
const step = (s) => console.log(`\n${C.bold('▶ ' + s)}`);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function portOpen(port, host = HOST) {
  return new Promise((resolve) => {
    const s = net.connect({ port, host });
    const done = (v) => { s.destroy(); resolve(v); };
    s.setTimeout(700);
    s.once('connect', () => done(true));
    s.once('timeout', () => done(false));
    s.once('error', () => done(false));
  });
}

async function anyOpen(port) {
  for (const h of [HOST, '::1']) if (await portOpen(port, h)) return true;
  return false;
}

async function waitPort(port, timeoutMs, label) {
  const end = Date.now() + timeoutMs;
  while (Date.now() < end) {
    if (await anyOpen(port)) return true;
    await sleep(1000);
  }
  bad(`${label} :${port} ${Math.round(timeoutMs / 1000)}s 内未就绪`);
  return false;
}

async function getJson(url, timeoutMs = 5000) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const r = await fetch(url, { signal: ctrl.signal });
    return { ok: r.ok, status: r.status, json: r.ok ? await r.json() : null };
  } catch (e) {
    return { ok: false, error: String(e?.message || e) };
  } finally { clearTimeout(t); }
}

/** 剩余可提交内存（MB） */
function headroomMb() {
  const r = spawnSync('powershell.exe', ['-NoProfile', '-Command',
    '$p = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory -ErrorAction SilentlyContinue;' +
    'if ($p) { [int](($p.CommitLimit - $p.CommittedBytes)/1MB) }'], { encoding: 'utf8' });
  const n = parseInt((r.stdout || '').trim(), 10);
  return Number.isFinite(n) ? n : null;
}

function pidOnPort(port) {
  const out = spawnSync('netstat', ['-ano', '-p', 'TCP'], { encoding: 'utf8', windowsHide: true }).stdout || '';
  const pids = new Set();
  for (const line of out.split(/\r?\n/)) {
    const c = line.trim().split(/\s+/);
    if (c.length < 5 || c[3] !== 'LISTENING') continue;
    if (!c[1].endsWith(`:${port}`)) continue;
    if (/^\d+$/.test(c[4])) pids.add(Number(c[4]));
  }
  pids.delete(process.pid);
  return [...pids];
}

/** 需要腾出内存时，可安全结束的大占用程序（浏览器/聊天/加速器，不影响平台与数据库） */
const RECLAIMABLE = ['msedge', 'msedgewebview2', 'WeChatAppEx', 'Weixin', 'copilotapp',
  'SmartConnect', 'wpscloudsvr', 'XLAUI', 'SystemSettings'];

function reclaimMemory() {
  let freed = 0;
  for (const name of RECLAIMABLE) {
    const r = spawnSync('powershell.exe', ['-NoProfile', '-Command',
      `$x = Get-Process -Name '${name}' -ErrorAction SilentlyContinue;` +
      'if ($x) { $mb = [int](($x | Measure-Object PrivateMemorySize64 -Sum).Sum/1MB);' +
      'try { $x | Stop-Process -Force -ErrorAction Stop; Write-Output $mb } catch {} }'],
      { encoding: 'utf8', windowsHide: true });
    const n = parseInt((r.stdout || '').trim(), 10);
    if (Number.isFinite(n)) freed += n;
  }
  return freed;
}

const children = [];

function launch(name, cmd, args, cwd, extraEnv = {}) {
  const child = spawn(cmd, args, {
    cwd, stdio: ['ignore', 'inherit', 'inherit'], windowsHide: true,
    env: { ...process.env, ...extraEnv },
  });
  child.on('exit', (code) => { if (!shuttingDown) say(`${name} 进程退出（code=${code}）`); });
  children.push({ name, child });
  return child;
}

/**
 * 算法服务进程的额外环境变量。
 *
 * `CUDA_VISIBLE_DEVICES=-1` —— 内存减法，**不损失任何功能**：
 *   算法侧实际推理全在 CPU（见 `algorithm/intervention/scene_retrieval/embedder.py:84`
 *   注释「本机无 CUDA，默认 cpu」，用的是短文本 RoBERTa / BGE / sherpa-onnx）。
 *   但 torch 是 2.11.0+cu128 全量 CUDA 构建，会带上
 *   torch_cuda.dll(788MB) + cublasLt(723MB) + cudnn(461+258MB) + cufft(266MB)…
 *   约 2.9 GB 的 CUDA 运行库，而算法侧**一点都用不到**。
 *
 * ⚠️ 必须用 `-1`，**不能用空串**（2026-10-10 实测）：
 *     CUDA_VISIBLE_DEVICES=''    → torch.cuda.is_available()=True, device_count=0
 *                                  「说可用但没有设备」，代码若走
 *                                  `if is_available(): model.to("cuda")` 会直接报错
 *     CUDA_VISIBLE_DEVICES='-1'  → is_available()=False, device_count=0   ← 正确
 *   实测四种取值：不设=True/1、''=True/0（危险）、-1=False/0、none=False/0。
 *   用 `-1` 才能让所有 `is_available()` 判断如实落到 CPU 分支。
 *
 * 想恢复 GPU（例如以后换机器用 GPU 跑算法侧）：设 AIC_ALGO_USE_GPU=1 再启动。
 * 注意：**不要**给 TTS 服务加这个 —— 它真的在用 GPU（server.py:882 把 fp16 模型
 * 整体 .to("cuda")，nvidia-smi 显示其占用约 2.3–3.6 GB 显存），关掉会让合成明显变慢。
 */
function algoEnv() {
  if (process.env.AIC_ALGO_USE_GPU === '1') {
    say('算法服务：按 AIC_ALGO_USE_GPU=1 保留 CUDA（内存占用会多约 2.9 GB）');
    return {};
  }
  return { CUDA_VISIBLE_DEVICES: '-1' };
}

let shuttingDown = false;

function shutdown() {
  if (shuttingDown) return;
  shuttingDown = true;
  console.log('\n正在停止本脚本拉起的进程…');
  for (const { child } of children) {
    if (child.pid) spawnSync('taskkill', ['/PID', String(child.pid), '/T', '/F'], { stdio: 'ignore', windowsHide: true });
  }
  console.log('已停止（数据库容器仍在运行）。');
  process.exit(0);
}

// ───────────────────────── 各端 ─────────────────────────

async function ensureDb() {
  step('① PostgreSQL :5432');
  if (await anyOpen(PORTS.db)) { ok('已在监听'); return true; }
  const info = spawnSync('docker', ['info', '--format', '{{.ServerVersion}}'], { encoding: 'utf8', windowsHide: true });
  if (info.status !== 0) {
    bad('Docker daemon 未运行 —— 请先启动 Docker Desktop，再重跑本脚本');
    return false;
  }
  spawnSync('docker', ['start', 'mh-postgres'], { encoding: 'utf8', windowsHide: true });
  if (!(await waitPort(PORTS.db, 90000, '数据库'))) return false;
  ok('已就绪（容器 mh-postgres）');
  return true;
}

/**
 * TTS 微服务 —— 必须在算法服务之前起：它是内存大户，此时内存最宽松。
 * 启动脚本自身的闸门（下限 12288 MB，可用 COSYVOICE_MIN_COMMIT_MB 覆盖）会先拦一道；
 * 这里在它之前主动腾内存，并帮它把模型预热加载（不加载的话算法侧仍报 unavailable）。
 */
async function ensureTts() {
  step('② CosyVoice3 TTS :8002（内存大户，必须最早起）');
  const health = await getJson(`http://${HOST}:${PORTS.tts}/health`, 4000);
  if (health.ok && health.json?.available) {
    ok(`已在运行，模型已就绪（engine=${health.json.engine} speed=${health.json.speed}）`);
    return true;
  }
  if (await anyOpen(PORTS.tts)) {
    say('端口已占用但模型不可用，先结束占用者…');
    for (const pid of pidOnPort(PORTS.tts)) {
      spawnSync('taskkill', ['/PID', String(pid), '/T', '/F'], { stdio: 'ignore', windowsHide: true });
    }
    await sleep(3000);
  }

  // 腾内存后再起 —— 闸门在启动前读一次余量，所以必须先把内存让出来。
  // 目标不是「刚好过闸门」：TTS 加载后算法服务还要吃 ~6 GB，本机提交上限只有 ~36 GB，
  // 贴线启动会让 8001 在随后加载模型时以 0xC0000005（内存访问冲突）原生崩溃。
  const WANT_MB = parseInt(process.env.AIC_WANT_HEADROOM_MB || '15000', 10);
  let hr = headroomMb();
  say(`可提交内存余量 ${hr} MB（TTS 需 ≥12288 MB，本脚本目标 ≥${WANT_MB} MB）`);
  for (let round = 1; round <= 3 && hr !== null && hr < WANT_MB; round++) {
    const freed = reclaimMemory();
    await sleep(4000);
    hr = headroomMb();
    say(`第 ${round} 轮释放 ${freed} MB，余量 ${hr} MB`);
    if (freed === 0) break; // 没有可释放的了，不再空转
  }
  if (hr !== null && hr >= 12288 && hr < WANT_MB) {
    console.log(`  ${C.yel('!')} 余量仍低于目标 ${WANT_MB} MB —— 先把服务拉起来，但请尽快按下面的办法扩容页面文件：`);
    console.log(`    当前页面文件被固定为 4 GB（提交上限 ~36 GB），整套平台 + 浏览器等常驻程序贴近上限，`);
    console.log(`    表现是某个服务突然原生崩溃（无 Python 回溯）。根治：`);
    console.log(`    设置 → 系统 → 关于 → 高级系统设置 → 性能「设置」→ 高级 → 虚拟内存「更改」`);
    console.log(`      → 勾选「自动管理所有驱动器的分页文件大小」→ 重启（需要管理员）。`);
  }

  launch('tts', process.execPath, [path.join(ROOT, 'scripts', 'run-tts.mjs')], ROOT);

  if (!(await waitPort(PORTS.tts, 60000, 'TTS'))) return false;

  // 预热：健康检查会触发模型加载（约 20 s）
  say('预热加载模型（首次约 20 s）…');
  for (let i = 1; i <= 12; i++) {
    const h = await getJson(`http://${HOST}:${PORTS.tts}/health`, 240000);
    if (h.ok && h.json?.available) {
      ok(`模型已加载（engine=${h.json.engine}，${h.json.model_name}）`);
      return true;
    }
    const err = h.json?.error || h.error || '';
    say(`第 ${i} 次预热未成功：${String(err).slice(0, 90)}`);
    if (String(err).includes('内存')) {
      const freed = reclaimMemory();
      await sleep(5000);
      say(`再次释放 ${freed} MB，余量 ${headroomMb()} MB，重试…`);
    } else {
      await sleep(4000);
    }
  }
  bad('TTS 模型预热失败 —— 语音功能将不可用（算法侧会报 engine=unavailable）');
  return false;
}

async function ensureAlgo() {
  step('③ 算法服务 :8001');
  const st = await getJson(`http://${HOST}:${PORTS.algo}/api/v1/asr/status`, 5000);
  if (st.ok && st.json?.streaming_available) { ok('已在运行且健康'); return true; }
  if (await anyOpen(PORTS.algo)) {
    say('端口被占用，先结束占用者…');
    for (const pid of pidOnPort(PORTS.algo)) {
      spawnSync('taskkill', ['/PID', String(pid), '/T', '/F'], { stdio: 'ignore', windowsHide: true });
    }
    await sleep(3000);
  }
  const py = process.env.ALGORITHM_PYTHON || path.join(ROOT, '..', '..', '..', '.venv', 'Scripts', 'python.exe');
  if (!existsSync(py)) { bad(`找不到解释器 ${py}`); return false; }
  say(`解释器 ${py}`);
  launch('algo', py, ['-m', 'uvicorn', 'main:app', '--host', HOST, '--port', String(PORTS.algo)],
    path.join(ROOT, 'algorithm'), algoEnv());
  if (!(await waitPort(PORTS.algo, 180000, '算法服务'))) return false;
  const openapi = await getJson(`http://${HOST}:${PORTS.algo}/openapi.json`, 15000);
  const n = openapi.json ? Object.keys(openapi.json.paths || {}).length : 0;
  ok(`已就绪（${n} 条路由）`);
  const tts = await getJson(`http://${HOST}:${PORTS.algo}/api/v1/tts/status`, 20000);
  if (tts.json?.available) ok(`TTS 引擎已接通：engine=${tts.json.engine}`);
  else bad(`TTS 引擎未接通：${String(tts.json?.error || '').slice(0, 90)}`);
  return true;
}

async function ensureApi() {
  step('④ Node 后端 :3000');
  const h = await getJson(`http://${HOST}:${PORTS.api}/api/health`, 4000);
  if (h.ok && h.json?.status === 'ok') { ok('已在运行'); return true; }
  if (await anyOpen(PORTS.api)) {
    for (const pid of pidOnPort(PORTS.api)) {
      spawnSync('taskkill', ['/PID', String(pid), '/T', '/F'], { stdio: 'ignore', windowsHide: true });
    }
    await sleep(2500);
  }
  const tsx = path.join(ROOT, 'server', 'node_modules', 'tsx', 'dist', 'cli.mjs');
  if (!existsSync(tsx)) { bad(`缺少 tsx：${tsx}`); return false; }
  launch('api', process.execPath, [tsx, 'watch', 'src/app.ts'], path.join(ROOT, 'server'));
  const end = Date.now() + 90000;
  while (Date.now() < end) {
    const r = await getJson(`http://${HOST}:${PORTS.api}/api/health`, 3000);
    if (r.ok && r.json?.status === 'ok') { ok('已就绪（/api/health → 200）'); return true; }
    await sleep(1500);
  }
  bad(':3000 90 秒内未通过健康检查');
  return false;
}

async function ensureWeb() {
  step('⑤ Vite 前端 :5173');
  if (await anyOpen(PORTS.web)) { ok('已在监听'); return true; }
  const vite = path.join(ROOT, 'client', 'node_modules', 'vite', 'bin', 'vite.js');
  if (!existsSync(vite)) { bad(`缺少 vite：${vite}`); return false; }
  launch('web', process.execPath, [vite, '--port', String(PORTS.web), '--strictPort'], path.join(ROOT, 'client'));
  if (!(await waitPort(PORTS.web, 90000, '前端'))) return false;
  ok(`已就绪（http://localhost:${PORTS.web}）`);
  return true;
}

// ───────────────────────── 入口 ─────────────────────────

async function status() {
  const rows = [
    ['数据库', PORTS.db, await anyOpen(PORTS.db)],
    ['TTS   ', PORTS.tts, await anyOpen(PORTS.tts)],
    ['算法  ', PORTS.algo, await anyOpen(PORTS.algo)],
    ['后端  ', PORTS.api, await anyOpen(PORTS.api)],
    ['前端  ', PORTS.web, await anyOpen(PORTS.web)],
  ];
  console.log('');
  for (const [name, port, up] of rows) {
    console.log(`  ${name} :${String(port).padEnd(6)} ${up ? C.grn('监听中') : C.gry('未监听')}`);
  }
  console.log('');
}

/**
 * 健康巡检 —— 本机内存极紧时，算法服务可能以 0xC0000005（原生崩溃，无回溯）退出，
 * 而症状是「前端一切正常、对话/语音突然 503」，很难当场定位。
 * 这里每 20 s 探一次，发现掉了就腾内存后自动拉回（有次数上限，避免无限重启风暴）。
 */
async function healthMonitor() {
  const MAX_RESTARTS = Number(process.env.AIC_MAX_RESTARTS || 5);
  const restarts = { algo: 0, tts: 0, api: 0, web: 0 };

  const restart = async (key) => {
    restarts[key]++;
    console.log(`\n${C.yel('!')} 巡检：${key} 无响应，尝试第 ${restarts[key]} 次自动拉起（上限 ${MAX_RESTARTS}）`);
    const freed = reclaimMemory();
    if (freed > 0) say(`先释放 ${freed} MB，余量 ${headroomMb()} MB`);
    if (key === 'algo') {
      const py = process.env.ALGORITHM_PYTHON || path.join(ROOT, '..', '..', '..', '.venv', 'Scripts', 'python.exe');
      launch('algo', py, ['-m', 'uvicorn', 'main:app', '--host', HOST, '--port', String(PORTS.algo)],
        path.join(ROOT, 'algorithm'), algoEnv());
      await waitPort(PORTS.algo, 120000, '算法服务');
    } else if (key === 'api') {
      const tsx = path.join(ROOT, 'server', 'node_modules', 'tsx', 'dist', 'cli.mjs');
      launch('api', process.execPath, [tsx, 'watch', 'src/app.ts'], path.join(ROOT, 'server'));
      await sleep(8000);
    } else if (key === 'web') {
      const vite = path.join(ROOT, 'client', 'node_modules', 'vite', 'bin', 'vite.js');
      launch('web', process.execPath, [vite, '--port', String(PORTS.web), '--strictPort'], path.join(ROOT, 'client'));
      await sleep(5000);
    } else if (key === 'tts') {
      launch('tts', process.execPath, [path.join(ROOT, 'scripts', 'run-tts.mjs')], ROOT);
      await waitPort(PORTS.tts, 60000, 'TTS');
      await getJson(`http://${HOST}:${PORTS.tts}/health`, 240000);
    }
  };

  // 需要看「服务自述」而不是端口是否监听
  const probes = [
    ['algo', async () => (await getJson(`http://${HOST}:${PORTS.algo}/health`, 6000)).ok],
    ['tts', async () => {
      const h = await getJson(`http://${HOST}:${PORTS.tts}/health`, 6000);
      return h.ok && h.json?.available === true;
    }],
    ['api', async () => (await getJson(`http://${HOST}:${PORTS.api}/api/health`, 6000)).ok],
    ['web', async () => anyOpen(PORTS.web)],
  ];

  for (;;) {
    await sleep(20000);
    if (shuttingDown) return;
    for (const [key, probe] of probes) {
      let up = false;
      try { up = await probe(); } catch { up = false; }
      if (!up && restarts[key] < MAX_RESTARTS) {
        await restart(key);
      }
    }
    const hr = headroomMb();
    if (hr !== null && hr < 500) {
      console.log(`${C.red(' !')} 可提交内存仅剩 ${hr} MB —— 系统已到页面文件上限，建议立即扩容页面文件或关闭大占用程序`);
    }
  }
}

async function main() {
  if (argv.has('--status')) return status();
  if (argv.has('--check')) {
    const hr = headroomMb();
    console.log(`可提交内存余量: ${hr} MB`);
    return;
  }
  console.log(`\n${'─'.repeat(66)}`);
  console.log(`  ${C.bold('青少年 AI 心理健康平台')}  ${C.gry('全功能启动 · node start-all.mjs')}`);
  console.log(`${'─'.repeat(66)}`);

  const res = { db: false, tts: false, algo: false, api: false, web: false };
  res.db = await ensureDb();
  if (!res.db) say('数据库未就绪 —— 涉及登录/落库的接口会 500，继续拉起其余服务');
  res.tts = argv.has('--no-tts') ? true : await ensureTts();
  res.algo = await ensureAlgo();
  res.api = await ensureApi();
  res.web = await ensureWeb();

  console.log(`\n${'─'.repeat(66)}`);
  const label = { db: '数据库', tts: 'TTS  ', algo: '算法  ', api: '后端  ', web: '前端  ' };
  for (const k of ['web', 'api', 'algo', 'tts', 'db']) {
    console.log(`  ${res[k] ? C.grn('✓') : C.red('✗')} ${label[k]} ${res[k] ? '' : '未就绪'}`);
  }
  console.log(`${'─'.repeat(66)}`);
  console.log(`  ${C.bold('浏览器打开')} http://localhost:${PORTS.web}`);
  console.log(`  ${C.gry('演示账号')} admin@mental.com / consultant@mental.com / patient@mental.com   密码 123456`);
  console.log(`  ${C.gry('可提交内存余量')} ${headroomMb()} MB`);
  if (children.length) {
    console.log(`  ${C.gry('停止：Ctrl+C（会收掉本脚本拉起的进程）')}`);
    console.log(`  ${C.gry('健康巡检：每 20 s 探一次，异常自动拉起（可用 AIC_MAX_RESTARTS 调整上限）')}\n`);
    healthMonitor();
  } else {
    console.log(`  ${C.gry('五端均已在运行，本脚本无需托管进程，退出。')}\n`);
    process.exit(0);
  }
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

main().catch((e) => { bad(`启动出错：${e?.stack || e}`); shutdown(); });
