/**
 * 浏览器探针公共库：启动无头浏览器 + 最小 CDP 客户端 + 登录注入。
 *
 * 被 `probe-growth-page.mjs` / `probe-checkin-flow.mjs` 共用。
 * 只用仓库里已有的 `ws`（server/node_modules），不额外装 puppeteer/playwright。
 */

import { spawn } from 'child_process';
import fs from 'fs';
import http from 'http';
import path from 'path';
import { fileURLToPath } from 'url';
import { createRequire } from 'module';

const require = createRequire(import.meta.url);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const REPO = path.resolve(__dirname, '..', '..');
const WebSocket = require(path.join(REPO, 'server', 'node_modules', 'ws'));

export const EDGE_CANDIDATES = [
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
];

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export function getJson(url) {
  return new Promise((resolve, reject) => {
    const req = http.get(url, (res) => {
      let body = '';
      res.on('data', (c) => { body += c; });
      res.on('end', () => { try { resolve(JSON.parse(body)); } catch (e) { reject(e); } });
    });
    req.on('error', reject);
    req.setTimeout(3000, () => req.destroy(new Error('timeout')));
  });
}

/** 最小 CDP 客户端（页面级会话）：顺手收集 console error / 未捕获异常。 */
export class Cdp {
  constructor(ws) {
    this.ws = ws;
    this.seq = 0;
    this.pending = new Map();
    this.consoleErrors = [];
    /** 全量 console（含 log/warn），排查"摄像头为什么没起来"这类链路问题时需要 */
    this.consoleAll = [];
    ws.on('message', (raw) => {
      let msg;
      try { msg = JSON.parse(raw.toString()); } catch { return; }
      if (msg.method === 'Runtime.consoleAPICalled') {
        const text = (msg.params.args || [])
          .map((a) => (a.value !== undefined ? String(a.value) : (a.description || a.type || '')))
          .join(' ');
        this.consoleAll.push({ type: msg.params.type, text });
      }
      if (msg.method === 'Runtime.consoleAPICalled' && msg.params.type === 'error') {
        this.consoleErrors.push((msg.params.args || []).map((a) => a.value || a.description || '').join(' '));
      }
      if (msg.method === 'Runtime.exceptionThrown') {
        this.consoleErrors.push(msg.params?.exceptionDetails?.exception?.description || 'exception');
      }
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (msg.error) reject(new Error(`${msg.error.message}`));
        else resolve(msg.result);
      }
    });
  }

  /** 打印一段时间的 console（排查链路问题用） */
  dumpConsole(pattern) {
    const re = pattern ? new RegExp(pattern) : null;
    for (const m of this.consoleAll) {
      if (!re || re.test(m.text)) console.log(`  [${m.type}] ${m.text.slice(0, 300)}`);
    }
  }

  send(method, params = {}) {
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
      setTimeout(() => {
        if (this.pending.has(id)) { this.pending.delete(id); reject(new Error(`CDP timeout: ${method}`)); }
      }, 60000);
    });
  }

  async eval(expression) {
    const r = await this.send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true, userGesture: true });
    if (r.exceptionDetails) throw new Error(`页面内异常: ${JSON.stringify(r.exceptionDetails).slice(0, 400)}`);
    return r.result.value;
  }

  /** 轮询页面文本直到满足条件（或超时），返回最后一次文本。 */
  async waitForText(predicate, { timeoutMs = 15000, intervalMs = 300 } = {}) {
    const start = Date.now();
    let text = '';
    while (Date.now() - start < timeoutMs) {
      text = await this.eval('document.body.innerText').catch(() => '');
      if (predicate(text)) return text;
      await sleep(intervalMs);
    }
    return text;
  }

  /** 整页截图并落盘 */
  async screenshot(file, { maxHeight = 6000 } = {}) {
    const metrics = await this.send('Page.getLayoutMetrics');
    const width = Math.ceil(metrics.cssContentSize.width);
    const height = Math.min(Math.ceil(metrics.cssContentSize.height), maxHeight);
    const shot = await this.send('Page.captureScreenshot', {
      format: 'png', captureBeyondViewport: true, clip: { x: 0, y: 0, width, height, scale: 1 },
    });
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, Buffer.from(shot.data, 'base64'));
    return { width, height, file };
  }
}

export async function login(api, email, password) {
  const res = await fetch(`${api}/api/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
  const body = await res.json();
  if (body.code !== 0) throw new Error(`登录失败：${JSON.stringify(body)}`);
  return body.data; // { user, accessToken, refreshToken }
}

/**
 * 启动无头浏览器并连上 page target。
 * 返回 { cdp, close }；close 会杀掉浏览器并清理临时 profile 目录。
 */
export async function launchBrowser({ port, width = 1400, height = 1200, extraArgs = [] } = {}) {
  const bin = EDGE_CANDIDATES.find((p) => fs.existsSync(p));
  if (!bin) throw new Error('找不到 Edge/Chrome');

  const profileDir = path.join(process.env.TEMP || '.', `dsh-probe-${port}-${Date.now()}`);
  const child = spawn(bin, [
    `--remote-debugging-port=${port}`,
    `--user-data-dir=${profileDir}`,
    '--no-first-run', '--no-default-browser-check', '--disable-extensions',
    '--headless=new', `--window-size=${width},${height}`,
    // 摄像头/麦克风探针用：`extraArgs` 传
    // `--use-fake-ui-for-media-stream --use-fake-device-for-media-stream`
    ...extraArgs,
    'about:blank',
  ], { stdio: 'ignore' });

  let ready = false;
  for (let i = 0; i < 60 && !ready; i += 1) {
    try { await getJson(`http://127.0.0.1:${port}/json/version`); ready = true; } catch { await sleep(300); }
  }
  if (!ready) { child.kill(); throw new Error('DevTools 端口未就绪'); }

  let page = null;
  for (let i = 0; i < 40 && !page; i += 1) {
    const list = await getJson(`http://127.0.0.1:${port}/json/list`).catch(() => []);
    page = list.find((t) => t.type === 'page' && t.webSocketDebuggerUrl);
    if (!page) await sleep(250);
  }
  if (!page) { child.kill(); throw new Error('没有可用的 page target'); }

  const ws = new WebSocket(page.webSocketDebuggerUrl, { maxPayload: 256 * 1024 * 1024 });
  await new Promise((res, rej) => { ws.once('open', res); ws.once('error', rej); });
  const cdp = new Cdp(ws);
  await cdp.send('Page.enable');
  await cdp.send('Runtime.enable');

  const close = async () => {
    try { ws.close(); } catch { /* ignore */ }
    child.kill();
    await sleep(500);
    try { fs.rmSync(profileDir, { recursive: true, force: true }); } catch { /* 占用则交给系统清理 */ }
  };
  return { cdp, close, bin };
}

/**
 * 注入登录态：必须在页面脚本之前执行，否则会被路由守卫弹回 /login。
 * 守卫读的是 zustand authStore（初始化自 localStorage），且 RoleRoute 还要 user.role，
 * 所以 token 和 user 都得塞。
 */
export async function injectAuth(cdp, { accessToken, refreshToken, user }) {
  await cdp.send('Page.addScriptToEvaluateOnNewDocument', {
    source: `try{localStorage.setItem('accessToken',${JSON.stringify(accessToken)});`
      + `localStorage.setItem('refreshToken',${JSON.stringify(refreshToken)});`
      + `localStorage.setItem('user',${JSON.stringify(JSON.stringify(user))});}catch(e){}`,
  });
}

export async function goto(cdp, url, { waitMs = 1200 } = {}) {
  await cdp.send('Page.navigate', { url });
  await sleep(waitMs);
}
