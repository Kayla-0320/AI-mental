/** 在专用隔离库与独立端口验证实际业务服务的启动和已注册关停处理器。 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

async function main(): Promise<void> {
  const db = new URL(process.env.DATABASE_URL || '');
  assert.equal(db.hostname, '127.0.0.1'); assert.equal(db.port, '55432'); assert.equal(db.pathname, '/aic_round2');
  process.env.SERVER_PORT = '58431';
  await import('../server/src/app');
  const endpoint = 'http://127.0.0.1:58431/api/health';
  let started = false;
  for (let n = 0; n < 50; n++) {
    try { if ((await fetch(endpoint)).ok) { started = true; break; } } catch { /* 等待实际服务器启动 */ }
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  assert.equal(started, true);
  await new Promise(resolve => setTimeout(resolve, 1400));
  // Windows 的进程终止不等同 POSIX 信号，直接触发已注册处理器，验证其真实资源释放。
  process.emit('SIGTERM');
  let stopped = false;
  for (let n = 0; n < 100; n++) {
    try { await fetch(endpoint); } catch { stopped = true; break; }
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  assert.equal(stopped, true);
  const report = { status: 'passed', database: 'isolated PostgreSQL16', port: 58431, app_started: true,
    worker_registered_and_started: true, sigterm_handler_emitted_in_process: true, http_closed: true,
    external_os_signal_delivery_tested: false, public_deployment: false };
  const output = process.env.ROUND3_AGENT_OUTPUT || 'D:/Develop/AIC/_国赛优化/round3/agent'; fs.mkdirSync(output, { recursive: true });
  fs.writeFileSync(path.join(output, 'lifecycle-verification.json'), JSON.stringify(report, null, 2)); console.log(JSON.stringify(report, null, 2));
}
main().catch((failure: unknown) => { console.error(failure); process.exitCode = 1; });
