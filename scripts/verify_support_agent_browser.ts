/** 真实 React、Express 鉴权、PostgreSQL 与浏览器工具；其他页面接口为替身，不调用在线大模型。 */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import express from '../server/node_modules/express';
import jwt from '../server/node_modules/jsonwebtoken';
import prisma from '../server/src/config/database';
import { config } from '../server/src/config';
import { errorHandler } from '../server/src/middlewares/errorHandler';
import { createSupportAgentRouter } from '../server/src/routes/support-agent.routes';
import { createPostgresAgentFixture } from './verify_support_agent_postgres';

async function main(): Promise<void> {
  const fixture = await createPostgresAgentFixture();
  const output = process.env.ROUND2_AGENT_OUTPUT || 'D:/Develop/AIC/_国赛优化/round2/agent';
  fs.mkdirSync(output, { recursive: true });
  const app = express(); app.use(express.json()); app.use('/api/support-agent', createSupportAgentRouter(fixture.service)); app.use(errorHandler);
  const server = app.listen(0, '127.0.0.1');
  await new Promise<void>(resolve => server.on('listening', resolve));
  const address = server.address() as { port: number };
  const apiBase = `http://127.0.0.1:${address.port}`;
  const repo = path.resolve(__dirname, '..');
  const vite = spawn(process.execPath, [path.join(repo, 'client/node_modules/vite/bin/vite.js'), '--host', '127.0.0.1', '--port', '5191', '--strictPort'], { cwd: path.join(repo, 'client'), windowsHide: true, stdio: 'pipe' });
  let viteError = ''; vite.stderr?.on('data', chunk => { viteError += chunk.toString(); });
  let browser: any;
  let failNext = false;
  const operations: string[] = [];
  const interceptErrors: string[] = [];
  try {
    for (let n = 0; n < 60; n++) {
      try { if ((await fetch('http://127.0.0.1:5191')).ok) break; } catch { /* 等待本地 Vite */ }
      if (n === 59) throw new Error(`Vite未启动: ${viteError}`);
      await new Promise(resolve => setTimeout(resolve, 250));
    }
    const cdpLib = await import('./lib/cdp.mjs');
    browser = await cdpLib.launchBrowser({ port: 9341, width: 1440, height: 1100 });
    const cdp = browser.cdp;
    cdp.ws.on('message', async (raw: Buffer): Promise<void> => {
      const event = JSON.parse(raw.toString());
      if (event.method !== 'Fetch.requestPaused') return;
      const { requestId, request } = event.params;
      try {
        const url = new URL(request.url);
        let status = 200;
        let data: unknown = { code: 0, data: {} };
        if (url.pathname.startsWith('/api/support-agent')) {
          {
            if (request.postData) operations.push(JSON.parse(request.postData).op);
            const response = await fetch(`${apiBase}${url.pathname}`, { method: request.method,
              headers: { 'Content-Type': 'application/json', Authorization: request.headers.Authorization || request.headers.authorization || '' },
              body: request.method === 'POST' ? request.postData : undefined });
            status = response.status; data = await response.json();
            // 故意在真实事务提交后丢弃成功响应，覆盖客户端无法判断是否已保存的情形。
            if (failNext && request.method === 'POST') { failNext = false; status = 503; data = { code: 503, message: '工程故障注入：回执是否保存尚未确认' }; }
          }
        } else if (url.pathname.includes('/notifications')) data = { code: 0, data: { notifications: [], unreadCount: 0 } };
        await cdp.send('Fetch.fulfillRequest', { requestId, responseCode: status,
          responseHeaders: [{ name: 'Content-Type', value: 'application/json; charset=utf-8' }], body: Buffer.from(JSON.stringify(data)).toString('base64') });
      } catch (failure: unknown) { interceptErrors.push(String(failure)); await cdp.send('Fetch.failRequest', { requestId, errorReason: 'Failed' }); }
    });
    await cdp.send('Fetch.enable', { patterns: [{ urlPattern: '*/api/*', requestStage: 'Request' }] });
    const token = jwt.sign({ userId: fixture.userId, role: 'PATIENT' }, config.jwt.secret, { expiresIn: '10m' });
    await cdpLib.injectAuth(cdp, { accessToken: token, refreshToken: '', user: { id: fixture.userId, role: 'PATIENT', email: 'round2-browser@example.invalid', nickname: '隔离测试用户' } });
    await cdp.send('Page.addScriptToEvaluateOnNewDocument', { source: `(() => {const OriginalAudio=window.Audio; window.__supportAudio=[]; window.Audio=function(...args){const audio=new OriginalAudio(...args);window.__supportAudio.push(audio);if(window.__failAudio){audio.play=()=>Promise.reject(new Error('engineered playback failure'));}return audio;};})();` });
    await cdpLib.goto(cdp, 'http://127.0.0.1:5191/chat?panel=tracking');
    const waitText = async (text: string, timeoutMs = 18000): Promise<void> => {
      const result = await cdp.waitForText((value: string) => value.includes(text), { timeoutMs }); assert.ok(result.includes(text), `未出现文字: ${text}; ${result.slice(0, 300)}`);
    };
    const click = async (text: string): Promise<void> => { await cdp.eval(`(() => {const card=document.querySelector('[data-testid="personal-support-agent"]'); const button=[...card.querySelectorAll('button')].find(b=>b.textContent.replace(/\\s/g,'')===${JSON.stringify(text)});if(!button || button.disabled)throw Error('button unavailable: '+${JSON.stringify(text)});button.click();})()`); };
    const scroll = async (): Promise<void> => { await cdp.eval(`document.querySelector('[data-testid="personal-support-agent"]').scrollIntoView({block:'center'})`); };
    await waitText('开启持续陪伴'); await click('开启持续陪伴');
    await waitText('接受建议'); await scroll(); await cdp.screenshot(path.join(output, '01-proactive-offer.png'));
    assert.ok((await fixture.service.overview(fixture.userId)).state.actions.length === 1);
    await click('接受建议'); await waitText('开始练习'); await click('开始练习'); await waitText('轻轻吸气');
    await scroll(); await cdp.screenshot(path.join(output, '02-real-breathing.png'));
    await waitText('这次对你有帮助吗', 36000); await click('没有帮助');
    await new Promise(resolve => setTimeout(resolve, 350));
    assert.equal((await fixture.service.overview(fixture.userId)).state.actions[0].feedback, 'unhelpful');
    await cdp.send('Page.reload'); await waitText('这次对你有帮助吗');
    const finished = await prisma.healingSession.findMany({ where: { userId: fixture.userId, type: 'breathing' } });
    assert.equal(finished.length, 1); assert.equal(finished[0].completed, true);
    await cdp.send('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
    await scroll(); await cdp.screenshot(path.join(output, '03-mobile-feedback.png'));
    const bounds = await cdp.eval(`(() => {const r=document.querySelector('[data-testid="personal-support-agent"]').getBoundingClientRect();return {left:r.left,right:r.right,viewport:document.documentElement.clientWidth,document:document.documentElement.scrollWidth};})()`);
    assert.ok(bounds.left >= 0 && bounds.right <= 391 && bounds.document <= 391, JSON.stringify(bounds));
    await cdp.send('Emulation.clearDeviceMetricsOverride');
    fixture.advance(6 * 3600);
    let overview = await fixture.service.overview(fixture.userId);
    await fixture.service.command(fixture.userId, { op: 'configure', requestId: randomUUID(), expectedRevision: overview.state.revision, enabled: true, preferredTool: 'music' });
    await cdp.send('Page.reload'); await waitText('接受建议'); await click('接受建议'); await waitText('开始播放');
    await cdp.eval('window.__failAudio=true'); await click('开始播放'); await waitText('这次没成功');
    overview = await fixture.service.overview(fixture.userId);
    assert.equal(overview.state.actions[1].status, 'failed');
    assert.equal(await prisma.healingSession.count({ where: { userId: fixture.userId, type: 'music', completed: true } }), 0);
    await scroll(); await cdp.screenshot(path.join(output, '04-playback-failure.png'));
    // 再开一条真实自然音活动，验证取消立即停止；不把这一段当疗效实验。
    fixture.advance(6 * 3600);
    await cdp.send('Page.reload'); await waitText('接受建议'); await click('接受建议'); await waitText('开始播放');
    failNext = true;
    await click('开始播放');
    await waitText('回执是否保存尚未确认', 25000); await click('重试保存');
    await waitText('此页没有正在运行的活动');
    await click('停止并取消'); await waitText('已经停下');
    // 页面隐藏事件注入，用真实播放器检查工具停止与取消回执；不模拟为真人使用。
    fixture.advance(6 * 3600);
    await cdp.send('Page.reload'); await waitText('接受建议'); await click('接受建议'); await waitText('开始播放'); await click('开始播放');
    await waitText('雨声在响', 25000);
    const nativeAudio = await cdp.eval(`(() => {const a=window.__supportAudio[window.__supportAudio.length-1];return {paused:a.paused,readyState:a.readyState};})()`);
    assert.equal(nativeAudio.paused, false); assert.ok(nativeAudio.readyState >= 2);
    await cdp.eval(`Object.defineProperty(document,'hidden',{configurable:true,value:true});document.dispatchEvent(new Event('visibilitychange'));`);
    await waitText('已经停下');
    assert.equal(await cdp.eval('window.__supportAudio[window.__supportAudio.length-1].paused'), true);
    await cdp.eval(`delete document.hidden`);
    fixture.advance(6 * 3600);
    await cdp.send('Page.reload'); await waitText('接受建议'); await click('接受建议'); await waitText('开始播放'); await click('开始播放');
    await waitText('雨声在响', 25000);
    overview = await fixture.service.overview(fixture.userId);
    await fixture.service.command(fixture.userId, { op: 'configure', requestId: randomUUID(), expectedRevision: overview.state.revision, enabled: false, preferredTool: 'music' });
    await waitText('开启持续陪伴', 6000);
    assert.equal(await cdp.eval('window.__supportAudio[window.__supportAudio.length-1].paused'), true);
    await click('开启持续陪伴'); await waitText('关闭持续陪伴');
    await click('关闭持续陪伴'); await waitText('开启持续陪伴');
    assert.equal((await fixture.service.overview(fixture.userId)).state.enabled, false);
    await cdp.send('Page.reload'); await waitText('开启持续陪伴');
    assert.equal(await cdp.eval('window.__supportAudio.length'), 0);
    assert.deepEqual(interceptErrors, []);
    const known = cdp.consoleErrors.filter((message: string) => message.includes('[antd: Modal]') || message.includes('[antd: Card]'));
    const unexpected = cdp.consoleErrors.filter((message: string) => !known.includes(message) && !message.includes('503') && !message.includes('favicon'));
    assert.deepEqual(unexpected, []);
    const report = { status: 'passed', scope: 'real React + authenticated Express HTTP + isolated PostgreSQL16 + real browser breathing timer; other layout APIs mocked',
      synthetic_user_and_mood_records: true, clinical_efficacy_tested: false, breathing_completed: true, participant_feedback_persisted: true,
      refresh_persisted: true, playback_failure_injected: true, failed_request_retry: true, uncertain_started_receipt_recovered: true,
      native_audio_play_resolved: nativeAudio, page_visibility_event_injected_cancelled: true, remote_opt_out_stopped_playback_within_6s: true,
      opt_out_reload_no_replay: true, cancel_and_opt_out: true,
      mobile_bounds: bounds, operations, unexpected_console_errors: unexpected, known_component_warnings: known };
    fs.writeFileSync(path.join(output, 'browser-verification.json'), JSON.stringify(report, null, 2)); console.log(JSON.stringify(report, null, 2));
  } finally {
    if (browser) await browser.close(); vite.kill(); await new Promise<void>(resolve => server.close(() => resolve())); await fixture.cleanup();
  }
}
main().catch((failure: unknown) => { console.error(failure); process.exitCode = 1; });
