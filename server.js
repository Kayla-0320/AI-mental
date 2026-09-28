#!/usr/bin/env node
/**
 * 青少年 AI 心理健康平台 —— 一键启动脚本
 *
 * 用法：
 *     node server.js
 *
 * 按依赖顺序拉起平台四端：
 *     [db]   PostgreSQL（Docker Desktop + 容器 mh-postgres，:5432）
 *     [algo] Python 算法服务（uvicorn main:app，:8001）
 *     [api]  Node 后端（tsx watch src/app.ts，:3000）
 *     [web]  Vite 前端（:5173）
 *
 * 设计要点：
 *   - **不经过 npm / shell**：直接 `node <tsx|vite 的 JS 入口>`。Node 24 在 Windows 上
 *     spawn `.cmd` 会抛 EINVAL（CVE-2024-27980 之后的收紧），绕开它同时也让 Ctrl+C
 *     的信号语义变得干净。
 *   - **幂等**：每个端口先探活，已在监听就跳过，可反复执行、不会 EADDRINUSE。
 *   - **自愈**：Docker daemon 没起就拉起 Docker Desktop；容器不存在就按原配置新建。
 *   - **可清理**：Ctrl+C 一次收掉本脚本拉起的全部进程树（taskkill /T /F）。
 *
 * 可选参数：
 *     --force           端口被占用时，先结束占用进程再启动（接管上次没退干净的实例）
 *     --app-only        只启动三个应用服务，跳过 Docker / 数据库
 *     --no-db           同上
 *     --no-algo         跳过 Python 算法服务
 *     --no-api          跳过 Node 后端
 *     --no-web          跳过前端
 *     --status          只打印四个端口的当前状态后退出
 *     --no-color        关闭彩色输出
 *     -h, --help        显示帮助
 *
 * 可用环境变量覆盖：
 *     AIC_DB_PORT / AIC_ALGO_PORT / AIC_API_PORT / AIC_WEB_PORT   端口
 *     AIC_DB_CONTAINER / AIC_DB_NAME / AIC_DB_PASSWORD            容器与库
 *     AIC_PYTHON / AIC_DOCKER_DESKTOP                             依赖路径
 *     注意：改端口只影响监听端口，client/vite.config.ts 里的 proxy 目标
 *     （3000 / 8001）是写死的，改端口后需要同步改它。
 */
'use strict';

const { spawn, spawnSync } = require('child_process');
const net = require('net');
const fs = require('fs');
const path = require('path');
const http = require('http');

// ───────────────────────────── 路径与常量 ─────────────────────────────

const ROOT = __dirname;
const SERVER_DIR = path.join(ROOT, 'server');
const CLIENT_DIR = path.join(ROOT, 'client');
const ALGO_DIR = path.join(ROOT, 'algorithm');
/** 本目录的三级父目录，即 D:\Develop\AIC —— venv 都在这一层 */
const VENV_HOME = path.resolve(ROOT, '..', '..', '..');

const PORTS = {
  db: Number(process.env.AIC_DB_PORT || 5432),
  algorithm: Number(process.env.AIC_ALGO_PORT || 8001),
  api: Number(process.env.AIC_API_PORT || 3000),
  web: Number(process.env.AIC_WEB_PORT || 5173),
};

const DB_CONTAINER = process.env.AIC_DB_CONTAINER || 'mh-postgres';
const DB_NAME = process.env.AIC_DB_NAME || 'mental_health';
const DB_PASSWORD = process.env.AIC_DB_PASSWORD || 'postgres123';

const argv = new Set(process.argv.slice(2));
const NO_COLOR = argv.has('--no-color') || Boolean(process.env.NO_COLOR) || !process.stdout.isTTY;
/** --force：端口被占用时不跳过，而是先结束占用进程再启动 */
const FORCE = argv.has('--force');

// ───────────────────────────── 输出工具 ─────────────────────────────

const paint = (code) => (s) => (NO_COLOR ? s : `\x1b[${code}m${s}\x1b[0m`);
const C = {
  cyan: paint(36),
  magenta: paint(35),
  green: paint(32),
  blue: paint(34),
  gray: paint(90),
  red: paint(31),
  yellow: paint(33),
  bold: paint(1),
};

/** 各服务日志前缀的配色（也用于启动标题） */
const LANES = {
  db: { tag: 'db', label: '数据库', color: C.cyan },
  algo: { tag: 'algo', label: '算法', color: C.magenta },
  api: { tag: 'api', label: '后端', color: C.green },
  web: { tag: 'web', label: '前端', color: C.blue },
  root: { tag: ' ·', label: '启动器', color: C.gray },
};

function logline(tag, color, text) {
  const prefix = NO_COLOR ? `[${tag}]` : color(`[${tag}]`);
  process.stdout.write(`${prefix} ${text}\n`);
}

const say = (msg) => logline(LANES.root.tag, LANES.root.color, msg);
const step = (title) => process.stdout.write(`\n${C.bold('▶ ' + title)}\n`);
const ok = (msg) => say(`${C.green('✓')} ${msg}`);
const info = (msg) => say(`${C.gray('…')} ${msg}`);
const warn = (msg) => say(`${C.yellow('!')} ${msg}`);
const fail = (msg) => say(`${C.red('✗')} ${msg}`);

function banner() {
  const line = '─'.repeat(64);
  process.stdout.write(`\n${C.gray(line)}\n`);
  process.stdout.write(`  ${C.bold('青少年 AI 心理健康平台')}  ${C.gray('一键启动 · node server.js')}\n`);
  process.stdout.write(`${C.gray(line)}\n`);
}

// ───────────────────────────── 基础工具 ─────────────────────────────

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** 单次 TCP 探测 */
function probeOnce(port, host, timeout = 800) {
  return new Promise((resolve) => {
    const sock = net.connect({ port, host });
    let settled = false;
    const done = (v) => {
      if (settled) return;
      settled = true;
      sock.destroy();
      resolve(v);
    };
    sock.setTimeout(timeout);
    sock.once('connect', () => done(true));
    sock.once('timeout', () => done(false));
    sock.once('error', () => done(false));
  });
}

/** 端口是否已被监听（同时探 IPv4 / IPv6，vite 只绑 ::1 的情况很常见） */
async function isPortOpen(port) {
  for (const host of ['127.0.0.1', '::1']) {
    if (await probeOnce(port, host)) return true;
  }
  return false;
}

/** 轮询等端口就绪 */
async function waitForPort(port, timeoutMs, intervalMs = 1000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await isPortOpen(port)) return true;
    await sleep(intervalMs);
  }
  return false;
}

/** 轮询等任意条件成立 */
async function waitUntil(fn, timeoutMs, intervalMs = 2000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await fn()) return true;
    await sleep(intervalMs);
  }
  return false;
}

/** 最小 HTTP GET（带超时，只取文本） */
function httpGet(url, timeout = 5000) {
  return new Promise((resolve) => {
    const req = http.get(url, { timeout }, (res) => {
      let body = '';
      res.setEncoding('utf8');
      res.on('data', (c) => (body += c));
      res.on('end', () => resolve({ ok: true, status: res.statusCode, body }));
    });
    req.on('timeout', () => req.destroy(new Error('timeout')));
    req.on('error', (err) => resolve({ ok: false, error: err.message }));
  });
}

/** 同步跑一个外部命令并收集输出 */
function run(cmd, args, options = {}) {
  const r = spawnSync(cmd, args, {
    encoding: 'utf8',
    windowsHide: true,
    timeout: options.timeout || 30000,
  });
  return {
    ok: r.status === 0,
    status: r.status,
    out: `${r.stdout || ''}${r.stderr || ''}`,
  };
}

// ───────────────────────── 依赖（解释器 / 容器）探测 ─────────────────────────

/** 找可用的 Python：必须能同时 import fastapi 与 uvicorn */
function resolvePython() {
  const candidates = [
    process.env.AIC_PYTHON,
    path.join(VENV_HOME, '_verify_venv', 'Scripts', 'python.exe'),
    path.join(VENV_HOME, '.venv', 'Scripts', 'python.exe'),
    path.join(ROOT, '.venv', 'Scripts', 'python.exe'),
    'python',
  ].filter(Boolean);

  for (const py of candidates) {
    if (py !== 'python' && !fs.existsSync(py)) continue;
    const r = spawnSync(py, ['-c', 'import fastapi, uvicorn'], { stdio: 'ignore', windowsHide: true });
    if (r.status === 0) return py;
  }
  return null;
}

function resolveDocker() {
  if (process.env.AIC_DOCKER) return process.env.AIC_DOCKER;
  if (run('docker', ['--version']).ok) return 'docker';
  const guess = process.env.LOCALAPPDATA
    ? path.join(process.env.LOCALAPPDATA, 'Programs', 'DockerDesktop', 'resources', 'bin', 'docker.exe')
    : null;
  return guess && fs.existsSync(guess) ? guess : null;
}

function resolveDockerDesktop() {
  const candidates = [
    process.env.AIC_DOCKER_DESKTOP,
    process.env.LOCALAPPDATA
      ? path.join(process.env.LOCALAPPDATA, 'Programs', 'DockerDesktop', 'Docker Desktop.exe')
      : null,
    'C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe',
  ].filter(Boolean);
  return candidates.find((p) => fs.existsSync(p)) || null;
}

const dockerDaemonReady = (docker) => run(docker, ['info', '--format', '{{.ServerVersion}}'], { timeout: 15000 }).ok;

// ───────────────────────────── 子进程管理 ─────────────────────────────

/** 本脚本拉起的服务 [{ key, child, exitCode }] */
const managed = [];
let shuttingDown = false;

/** 按行切分并加前缀输出 */
function attachStream(stream, tag, color) {
  let buffer = '';
  stream.setEncoding('utf8');
  stream.on('data', (chunk) => {
    buffer += chunk;
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop();
    for (const line of lines) logline(tag, color, line);
  });
  stream.on('end', () => {
    if (buffer.trim()) logline(tag, color, buffer);
    buffer = '';
  });
}

/**
 * 启动一个受管子进程
 * @returns {{key:string, child:import('child_process').ChildProcess}}
 */
function launch(key, cmd, args, cwd, extraEnv = {}) {
  const lane = LANES[key];
  const child = spawn(cmd, args, {
    cwd,
    windowsHide: true,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, ...extraEnv },
  });
  attachStream(child.stdout, lane.tag, lane.color);
  attachStream(child.stderr, lane.tag, lane.color);
  child.on('error', (err) => fail(`${lane.label}进程启动失败：${err.message}`));
  child.on('exit', (code, signal) => {
    if (shuttingDown) return;
    logline(lane.tag, C.red, `进程已退出（code=${code ?? signal}）`);
  });
  managed.push({ key, child });
  return child;
}

/** 结束一个进程树（Windows 上必须 /T，否则 tsx/python 的孙进程会残留） */
function killTree(pid) {
  if (!pid) return;
  if (process.platform === 'win32') {
    spawnSync('taskkill', ['/PID', String(pid), '/T', '/F'], { stdio: 'ignore', windowsHide: true });
  } else {
    try {
      process.kill(-pid, 'SIGTERM');
    } catch {
      try {
        process.kill(pid, 'SIGTERM');
      } catch {
        /* 进程已退出 */
      }
    }
  }
}

/** 找出监听某端口的进程 PID（Windows 用 netstat，POSIX 用 lsof） */
function findPortOwnerPids(port) {
  const pids = new Set();
  if (process.platform === 'win32') {
    const out = run('netstat', ['-ano', '-p', 'TCP']).out;
    for (const line of out.split(/\r?\n/)) {
      const cols = line.trim().split(/\s+/);
      // Proto  Local Address  Foreign Address  State  PID
      if (cols.length < 5 || cols[3] !== 'LISTENING') continue;
      if (!cols[1].endsWith(`:${port}`)) continue;
      if (/^\d+$/.test(cols[4])) pids.add(Number(cols[4]));
    }
  } else {
    const out = run('lsof', ['-ti', `tcp:${port}`, '-sTCP:LISTEN']).out;
    for (const token of out.split(/\s+/)) {
      if (/^\d+$/.test(token)) pids.add(Number(token));
    }
  }
  pids.delete(process.pid);
  pids.delete(process.ppid);
  return [...pids];
}

/**
 * 处理端口占用。
 * @returns {Promise<boolean>} true = 端口上已有服务在跑、无需启动；false = 端口空着，可以启动
 */
async function handleOccupied(port, label) {
  if (!(await isPortOpen(port))) return false;

  if (!FORCE) {
    ok(`:${port} 已在监听，跳过`);
    return true;
  }

  const pids = findPortOwnerPids(port);
  if (pids.length === 0) {
    warn(`:${port} 被占用但定位不到 PID，跳过`);
    return true;
  }

  info(`--force：${label} :${port} 被 PID ${pids.join(', ')} 占用，先结束它…`);
  for (const pid of pids) killTree(pid);

  if (!(await waitUntil(async () => !(await isPortOpen(port)), 15000, 500))) {
    fail(`:${port} 15 秒内未释放，跳过 ${label}`);
    return true;
  }
  ok(`:${port} 已释放`);
  return false;
}

// ───────────────────────────── 四端启动流程 ─────────────────────────────

/** 1. 数据库 */
async function startDatabase() {
  step('① PostgreSQL 数据库');

  if (await isPortOpen(PORTS.db)) {
    ok(`:${PORTS.db} 已在监听，跳过`);
    return true;
  }

  const docker = resolveDocker();
  if (!docker) {
    fail('找不到 docker 命令，跳过数据库（涉及落库/登录的接口会 500）');
    return false;
  }

  if (!dockerDaemonReady(docker)) {
    const desktop = resolveDockerDesktop();
    if (!desktop) {
      fail('Docker daemon 未运行，且找不到 Docker Desktop，跳过数据库');
      return false;
    }
    info('Docker daemon 未运行，正在启动 Docker Desktop（首次约 30–90 秒）…');
    spawn(desktop, [], { detached: true, stdio: 'ignore', windowsHide: true }).unref();
    const ready = await waitUntil(() => dockerDaemonReady(docker), 180000, 3000);
    if (!ready) {
      fail('等待 Docker daemon 超时，跳过数据库（可手动启动 Docker Desktop 后重跑）');
      return false;
    }
  }
  ok('Docker daemon 就绪');

  const started = run(docker, ['start', DB_CONTAINER]);
  if (!started.ok && /no such container/i.test(started.out)) {
    info(`容器 ${DB_CONTAINER} 不存在，按原配置新建…`);
    const created = run(docker, [
      'run', '-d', '--name', DB_CONTAINER,
      '-e', `POSTGRES_PASSWORD=${DB_PASSWORD}`,
      '-e', `POSTGRES_DB=${DB_NAME}`,
      '-p', `${PORTS.db}:5432`,
      'postgres:16',
    ]);
    if (!created.ok) {
      fail(`创建容器失败：${created.out.trim()}`);
      return false;
    }
  } else if (!started.ok) {
    fail(`docker start ${DB_CONTAINER} 失败：${started.out.trim()}`);
    return false;
  }

  info(`等待 :${PORTS.db} 接受连接…`);
  if (!(await waitForPort(PORTS.db, 60000))) {
    fail(`:${PORTS.db} 60 秒内未就绪，排查：${docker} logs ${DB_CONTAINER}`);
    return false;
  }
  ok(`:${PORTS.db} 已就绪（容器 ${DB_CONTAINER}，库 ${DB_NAME}）`);
  return true;
}

/** 2. Python 算法服务 */
async function startAlgorithm() {
  step('② Python 算法服务');

  if (await handleOccupied(PORTS.algorithm, '算法服务')) return true;

  const py = resolvePython();
  if (!py) {
    fail('未找到装了 fastapi + uvicorn 的 Python 解释器（可用 AIC_PYTHON 指定）');
    return false;
  }
  info(`解释器：${py}`);

  launch('algo', py, ['-m', 'uvicorn', 'main:app', '--host', '127.0.0.1', '--port', String(PORTS.algorithm)], ALGO_DIR);

  info(`等待 :${PORTS.algorithm} 就绪（首次要加载感知模型，可能要几十秒）…`);
  if (!(await waitForPort(PORTS.algorithm, 180000))) {
    fail(`:${PORTS.algorithm} 180 秒内未就绪`);
    return false;
  }
  const routes = await httpGet(`http://127.0.0.1:${PORTS.algorithm}/openapi.json`, 10000);
  if (routes.ok) {
    try {
      const count = Object.keys(JSON.parse(routes.body).paths || {}).length;
      ok(`:${PORTS.algorithm} 已就绪（${count} 条路由）`);
    } catch {
      ok(`:${PORTS.algorithm} 已就绪`);
    }
  } else {
    warn(`:${PORTS.algorithm} 端口已监听，但 /openapi.json 无响应`);
  }
  return true;
}

/** 3. Node 后端 */
async function startApi() {
  step('③ Node 后端');

  if (await handleOccupied(PORTS.api, 'Node 后端')) return true;

  const tsxCli = path.join(SERVER_DIR, 'node_modules', 'tsx', 'dist', 'cli.mjs');
  if (!fs.existsSync(tsxCli)) {
    fail(`缺少 tsx（${tsxCli}），请先在 server/ 下执行 npm install`);
    return false;
  }

  launch('api', process.execPath, [tsxCli, 'watch', 'src/app.ts'], SERVER_DIR, {
    SERVER_PORT: String(PORTS.api),
  });

  info(`等待 :${PORTS.api}/api/health 就绪…`);
  const up = await waitUntil(async () => {
    const r = await httpGet(`http://127.0.0.1:${PORTS.api}/api/health`, 3000);
    return r.ok && r.status === 200;
  }, 90000, 1500);

  if (!up) {
    fail(`:${PORTS.api} 90 秒内未通过健康检查`);
    return false;
  }
  ok(`:${PORTS.api} 已就绪（/api/health → 200）`);
  return true;
}

/** 4. Vite 前端 */
async function startWeb() {
  step('④ Vite 前端');

  if (await handleOccupied(PORTS.web, 'Vite 前端')) return true;

  const viteBin = path.join(CLIENT_DIR, 'node_modules', 'vite', 'bin', 'vite.js');
  if (!fs.existsSync(viteBin)) {
    fail(`缺少 vite（${viteBin}），请先在 client/ 下执行 npm install`);
    return false;
  }

  launch('web', process.execPath, [viteBin, '--port', String(PORTS.web), '--strictPort'], CLIENT_DIR);

  info(`等待 :${PORTS.web} 就绪…`);
  if (!(await waitForPort(PORTS.web, 90000))) {
    fail(`:${PORTS.web} 90 秒内未就绪`);
    return false;
  }
  ok(`:${PORTS.web} 已就绪（http://localhost:${PORTS.web}）`);
  return true;
}

// ───────────────────────────── 汇总与退出 ─────────────────────────────

async function printSummary(results) {
  const rows = [
    ['前端', `http://localhost:${PORTS.web}`, results.web],
    ['后端', `http://127.0.0.1:${PORTS.api}/api/health`, results.api],
    ['算法', `http://127.0.0.1:${PORTS.algorithm}/docs`, results.algorithm],
    ['数据库', `localhost:${PORTS.db}（容器 ${DB_CONTAINER}）`, results.db],
  ];

  process.stdout.write(`\n${C.gray('─'.repeat(64))}\n`);
  for (const [name, where, state] of rows) {
    const mark = state === true ? C.green('✓') : state === false ? C.red('✗') : C.gray('·');
    process.stdout.write(`  ${mark} ${name.padEnd(4, '　')} ${where}\n`);
  }
  process.stdout.write(`${C.gray('─'.repeat(64))}\n`);

  const values = Object.values(results);
  const failed = values.filter((v) => v === false).length;
  const skipped = values.filter((v) => v === null).length; // null = 按参数跳过，不算失败

  if (failed === 0 && skipped === 0) {
    process.stdout.write(`  ${C.bold('浏览器打开')} http://localhost:${PORTS.web}\n`);
    process.stdout.write(`  ${C.gray('演示账号')} admin@mental.com / consultant@mental.com / patient@mental.com   密码 123456\n`);
  } else if (failed > 0) {
    process.stdout.write(`  ${C.yellow(`有 ${failed} 项未就绪，详见上方 [✗] 行`)}\n`);
  } else {
    process.stdout.write(`  ${C.gray(`本次为部分启动：有 ${skipped} 项按参数跳过（标记为 ·）`)}\n`);
  }

  if (managed.length > 0) {
    process.stdout.write(`  ${C.gray('停止：在本终端按 Ctrl+C（会一并收掉本脚本拉起的进程）')}\n`);
  } else {
    process.stdout.write(`  ${C.gray('四端均已在运行，本脚本没有需要托管的新进程，直接退出。')}\n`);
  }
  process.stdout.write(`${C.gray('─'.repeat(64))}\n\n`);
}

function shutdown(reason) {
  if (shuttingDown) return;
  shuttingDown = true;
  process.stdout.write('\n');
  say(`收到 ${reason}，正在停止本脚本拉起的 ${managed.length} 个进程…`);
  for (const { child } of managed) killTree(child.pid);
  say('已全部停止。数据库容器仍在后台运行（docker stop ' + DB_CONTAINER + ' 可停）。');
  process.exit(0);
}

async function printStatus() {
  const checks = await Promise.all([
    isPortOpen(PORTS.db),
    isPortOpen(PORTS.algorithm),
    isPortOpen(PORTS.api),
    isPortOpen(PORTS.web),
  ]);
  const rows = [
    ['数据库', PORTS.db, checks[0]],
    ['算法', PORTS.algorithm, checks[1]],
    ['后端', PORTS.api, checks[2]],
    ['前端', PORTS.web, checks[3]],
  ];
  process.stdout.write('\n');
  for (const [name, port, up] of rows) {
    const mark = up ? C.green('监听中') : C.gray('未监听');
    process.stdout.write(`  ${name.padEnd(4, '　')} :${String(port).padEnd(6)} ${mark}\n`);
  }
  process.stdout.write('\n');
}

function printHelp() {
  process.stdout.write(`
${C.bold('用法：')} node server.js [选项]

  一键拉起 数据库 → 算法服务 → 后端 → 前端 四端，已在运行的服务自动跳过。

${C.bold('选项：')}
  --force         端口被占用时，先结束占用进程再启动（接管上次没退干净的实例）
  --app-only      只启动三个应用服务，跳过 Docker / 数据库
  --no-db         同上
  --no-algo       跳过 Python 算法服务
  --no-api        跳过 Node 后端
  --no-web        跳过前端
  --status        只打印四个端口的当前状态后退出
  --no-color      关闭彩色输出
  -h, --help      显示本帮助

${C.bold('环境变量：')}
  AIC_DB_PORT / AIC_ALGO_PORT / AIC_API_PORT / AIC_WEB_PORT
  AIC_DB_CONTAINER / AIC_DB_NAME / AIC_DB_PASSWORD
  AIC_PYTHON / AIC_DOCKER / AIC_DOCKER_DESKTOP

${C.bold('端口约定：')} 数据库 5432 · 算法 8001 · 后端 3000 · 前端 5173
  （client/vite.config.ts 的 proxy 目标写死为 3000 / 8001）
`);
}

// ───────────────────────────── 入口 ─────────────────────────────

async function main() {
  if (argv.has('-h') || argv.has('--help')) return printHelp();

  banner();

  if (argv.has('--status')) {
    await printStatus();
    return;
  }

  const results = { db: null, algorithm: null, api: null, web: null };

  if (argv.has('--app-only') || argv.has('--no-db')) {
    step('① PostgreSQL 数据库');
    info('已指定 --app-only / --no-db，跳过（登录等落库接口可能 500）');
  } else {
    results.db = await startDatabase();
  }

  if (argv.has('--no-algo')) {
    step('② Python 算法服务');
    info('已指定 --no-algo，跳过');
  } else {
    results.algorithm = await startAlgorithm();
  }

  if (argv.has('--no-api')) {
    step('③ Node 后端');
    info('已指定 --no-api，跳过');
  } else {
    results.api = await startApi();
  }

  if (argv.has('--no-web')) {
    step('④ Vite 前端');
    info('已指定 --no-web，跳过');
  } else {
    results.web = await startWeb();
  }

  await printSummary(results);

  // 所有端口本来就在监听时，没有任何子进程需要托管，直接退出。
  if (managed.length === 0) process.exit(0);
}

process.on('SIGINT', () => shutdown('Ctrl+C'));
process.on('SIGTERM', () => shutdown('SIGTERM'));

main().catch((err) => {
  fail(`启动过程出错：${err && err.stack ? err.stack : err}`);
  shutdown('异常');
});
