/** 第三轮真实浏览器、HTTP鉴权、后台worker与PG；心情/确认状态/模型回应均为明确测试夹具。 */
import assert from 'node:assert/strict';
import { randomUUID, createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import express from '../server/node_modules/express';
import jwt from '../server/node_modules/jsonwebtoken';
import prisma from '../server/src/config/database';
import { config } from '../server/src/config';
import { errorHandler } from '../server/src/middlewares/errorHandler';
import { createSupportAgentRouter } from '../server/src/routes/support-agent.routes';
import { SupportAgentService } from '../server/src/services/support-agent.service';
import { SupportAgentWorker } from '../server/src/services/support-agent-worker';
import { createPostgresAgentFixture } from './verify_support_agent_postgres';

async function main(): Promise<void> {
  const fixture = await createPostgresAgentFixture();
  const quote = '我担心被忽视，不是觉得对方讨厌我'; const feedbackId = randomUUID();
  const conversation = await prisma.conversation.create({ data: { userId: fixture.userId, title: '第三轮浏览器确认内容夹具' } });
  await prisma.message.create({ data: { userId: fixture.userId, conversationId: conversation.id, role: 'user', content: '我担心没有人听我说话' } });
  const supportState = { schema_version: 'support.v1', session_id: conversation.id, revision: 1, goal: 'act', goal_source: 'user_feedback', declined: false,
    items: [{ id: 'e1', quote, source_id: `feedback:${feedbackId}`, source_sha256: createHash('sha256').update(quote).digest('hex'), start: 0, end: [...quote].length, status: 'corrected', origin: 'user_correction' }],
    feedback_log: [{ feedback_id: feedbackId, action: 'correct', item_id: 'e1', text: quote }] };
  const assistant = await prisma.message.create({ data: { conversationId: conversation.id, role: 'assistant', content: '收到你本人确认的说明。', sentiment: JSON.stringify({ support_state: supportState }) } });
  const agent = new SupportAgentService(prisma, () => new Date(), async evidence => ({ available: true, task_id: 'sentiment_polarity_3',
    label_schema: ['negative', 'neutral', 'positive'], model_version: 'browser-contract-fixture', artifact_sha256: 'a'.repeat(64),
    calibration: { status: 'calibrated', method: 'contract_fixture' }, evidence_source: { source_message_id: evidence.sourceId, source_sha256: createHash('sha256').update(evidence.quote).digest('hex') },
    probabilities: [0.2, 0.3, 0.5], observed_at: evidence.observedAt, limitations: ['injected fixture, no real inference'] }));
  await agent.command(fixture.userId, { op: 'configure', requestId: randomUUID(), expectedRevision: 0, enabled: true, preferredTool: 'breathing' });
  const localHour = new Date(Date.now() + 480 * 60000).getUTCHours();
  await agent.command(fixture.userId, { op: 'background_settings', requestId: randomUUID(), expectedRevision: 1, enabled: false, intervalHours: 12, quietStart: (localHour + 10) % 24, quietEnd: (localHour + 11) % 24 });
  const worker = new SupportAgentWorker(agent, 2000);
  const output = process.env.ROUND3_AGENT_OUTPUT || 'D:/Develop/AIC/_国赛优化/round3/agent'; fs.mkdirSync(output, { recursive: true });
  const app = express(); app.use(express.json()); app.use('/api/support-agent', createSupportAgentRouter(agent)); app.use(errorHandler);
  const server = app.listen(0, '127.0.0.1'); await new Promise<void>(resolve => server.on('listening', resolve));
  const endpoint = `http://127.0.0.1:${(server.address() as { port: number }).port}`;
  const repo = path.resolve(__dirname, '..');
  const vite = spawn(process.execPath, [path.join(repo, 'client/node_modules/vite/bin/vite.js'), '--host', '127.0.0.1', '--port', '5193', '--strictPort'], { cwd: path.join(repo, 'client'), windowsHide: true, stdio: 'ignore' });
  let browser: any;
  const operations: string[] = []; const failures: string[] = [];
  try {
    for (let n = 0; n < 60; n++) { try { if ((await fetch('http://127.0.0.1:5193')).ok) break; } catch { /* 等待Vite */ } await new Promise(resolve => setTimeout(resolve, 250)); }
    const lib = await import('./lib/cdp.mjs'); browser = await lib.launchBrowser({ port: 9343, width: 1440, height: 1100 }); const cdp = browser.cdp;
    cdp.ws.on('message', async (raw: Buffer): Promise<void> => {
      const event = JSON.parse(raw.toString()); if (event.method !== 'Fetch.requestPaused') return;
      const { requestId, request } = event.params;
      try {
        const url = new URL(request.url); let status = 200; let data: unknown = { code: 0, data: {} };
        if (url.pathname.startsWith('/api/support-agent')) {
          if (request.postData && !url.pathname.endsWith('/model-reference')) operations.push(JSON.parse(request.postData).op);
          const response = await fetch(`${endpoint}${url.pathname}`, { method: request.method, headers: { 'Content-Type': 'application/json', Authorization: request.headers.Authorization || request.headers.authorization || '' }, body: request.method === 'POST' ? request.postData : undefined });
          status = response.status; data = await response.json();
        } else if (url.pathname.includes('/notifications')) data = { code: 0, data: { notifications: [], unreadCount: 0 } };
        await cdp.send('Fetch.fulfillRequest', { requestId, responseCode: status, responseHeaders: [{ name: 'Content-Type', value: 'application/json; charset=utf-8' }], body: Buffer.from(JSON.stringify(data)).toString('base64') });
      } catch (error: unknown) { failures.push(String(error)); await cdp.send('Fetch.failRequest', { requestId, errorReason: 'Failed' }); }
    });
    await cdp.send('Fetch.enable', { patterns: [{ urlPattern: '*/api/*', requestStage: 'Request' }] });
    const token = jwt.sign({ userId: fixture.userId, role: 'PATIENT' }, config.jwt.secret, { expiresIn: '10m' });
    await lib.injectAuth(cdp, { accessToken: token, refreshToken: '', user: { id: fixture.userId, role: 'PATIENT', nickname: '第三轮隔离测试', email: 'round3-browser@example.invalid' } });
    const wait = async (text: string): Promise<void> => { const value = await cdp.waitForText((raw: string) => raw.includes(text), { timeoutMs: 20000 }); assert.ok(value.includes(text), `未出现${text}: ${value.slice(0, 350)}`); };
    const click = async (text: string): Promise<void> => { await cdp.eval(`(() => {const root=document.querySelector('[data-testid="personal-support-agent"]');const b=[...root.querySelectorAll('button')].find(b=>b.textContent.replace(/\\s/g,'')===${JSON.stringify(text)});if(!b||b.disabled)throw Error('button missing');b.click();})()`); };
    await lib.goto(cdp, 'http://127.0.0.1:5193/chat?panel=tracking');
    await wait('你告诉过我的事'); assert.ok((await cdp.eval('document.body.innerText')).includes(quote));
    // 首次进入的页面建议是第二轮能力；后台开关开启前取消它，确保离页建议来自worker。
    if ((await cdp.eval('document.body.innerText')).includes('接受建议')) { await click('停止并取消'); await wait('已经停下'); }
    // 冷却期测试须明确放行：清除仅属于本测试用户的前台夹具建议，不触碰其他数据。
    const rawState = await prisma.healingSession.findFirstOrThrow({ where: { userId: fixture.userId, type: 'support_agent_state' } });
    const metadata = JSON.parse(rawState.metadata!); metadata.actions = []; await prisma.healingSession.update({ where: { id: rawState.id }, data: { metadata: JSON.stringify(metadata) } });
    await cdp.eval(`document.querySelector('button[aria-label="后台持续建议"]').click()`);
    for (let n = 0; n < 30 && !(await agent.overview(fixture.userId)).state.background.enabled; n++) await new Promise(resolve => setTimeout(resolve, 100));
    assert.equal((await agent.overview(fixture.userId)).state.background.enabled, true);
    for (let n = 0; n < 20 && await cdp.eval(`document.querySelector('button[aria-label="站内建议提醒"]').disabled`); n++) await new Promise(resolve => setTimeout(resolve, 50));
    await cdp.eval(`document.querySelector('button[aria-label="站内建议提醒"]').click()`);
    for (let n = 0; n < 30 && !(await agent.overview(fixture.userId)).state.background.notifyInApp; n++) await new Promise(resolve => setTimeout(resolve, 100));
    assert.equal((await agent.overview(fixture.userId)).state.background.notifyInApp, true);
    await cdp.send('Page.navigate', { url: 'about:blank' }); worker.start();
    for (let n = 0; n < 50 && !(await agent.overview(fixture.userId)).state.actions.some(item => item.origin === 'background'); n++) await new Promise(resolve => setTimeout(resolve, 100));
    await worker.stop();
    assert.equal((await agent.overview(fixture.userId)).state.actions.filter(item => item.origin === 'background').length, 1);
    assert.equal(await prisma.healingSession.count({ where: { userId: fixture.userId, type: 'breathing' } }), 0);
    assert.equal(await prisma.notification.count({ where: { userId: fixture.userId, type: 'support_agent', isRead: false } }), 1);
    await lib.goto(cdp, 'http://127.0.0.1:5193/chat?panel=tracking'); await wait('我在后台记下的');
    await cdp.eval(`document.querySelector('[data-testid="personal-support-agent"]').scrollIntoView({block:'center'})`);
    await cdp.screenshot(path.join(output, '01-background-persisted-offer.png'));
    await click('接受建议'); await wait('开始练习'); await click('开始练习'); await wait('轻轻吸气'); await click('停止并取消'); await wait('已经停下');
    assert.equal(await prisma.notification.count({ where: { userId: fixture.userId, type: 'support_agent', isRead: false } }), 0);
    await click('看看这段文字读起来是什么感觉'); await wait('这段话读起来偏正向');
    await cdp.screenshot(path.join(output, '02-confirmed-context-model-reference.png'));
    const changed = structuredClone(supportState); changed.revision = 2; changed.goal = 'listen'; changed.items[0].status = 'rejected';
    await prisma.message.update({ where: { id: assistant.id }, data: { sentiment: JSON.stringify({ support_state: changed }) } });
    await cdp.send('Page.reload'); await wait('现在希望先被倾听');
    assert.equal((await cdp.eval('document.body.innerText')).includes(quote), false);
    assert.equal((await agent.overview(fixture.userId)).modelReference, null);
    await cdp.send('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
    await cdp.eval(`document.querySelector('[data-testid="personal-support-agent"]').scrollIntoView({block:'start'})`);
    await cdp.screenshot(path.join(output, '03-mobile-background-controls.png'));
    const bounds = await cdp.eval(`(()=>{const r=document.querySelector('[data-testid="personal-support-agent"]').getBoundingClientRect();return {left:r.left,right:r.right,viewport:document.documentElement.clientWidth,document:document.documentElement.scrollWidth};})()`);
    assert.ok(bounds.left >= 0 && bounds.right <= 391 && bounds.document <= 391, JSON.stringify(bounds));
    await cdp.eval(`document.querySelector('button[aria-label="后台持续建议"]').click()`);
    for (let n = 0; n < 30 && (await agent.overview(fixture.userId)).state.background.enabled; n++) await new Promise(resolve => setTimeout(resolve, 100));
    assert.equal((await agent.overview(fixture.userId)).state.background.enabled, false);
    await click('关闭持续陪伴'); await wait('开启持续陪伴'); await cdp.send('Page.reload'); await wait('开启持续陪伴');
    assert.deepEqual(failures, []);
    const errors = cdp.consoleErrors.filter((text: string) => !text.includes('[antd:') && !text.includes('favicon'));
    assert.deepEqual(errors, []);
    const result = { status: 'passed', real_components: ['React', 'authenticated Express HTTP', 'PostgreSQL16', 'background worker', 'breathing runner'],
      fixtures: ['synthetic user and self-reports', 'stored confirmed correction', 'model response contract fixture', 'other layout APIs'],
      background_opt_in: true, independent_in_app_notification_opt_in: true, persistent_notification_created_once: true, accepted_notification_invalidated: true,
      generated_after_leaving_page: true, persisted_offer_after_return: true, no_auto_tool_execution: true,
      explicit_accept_and_cancel: true, source_revocation_removed_quote_and_model_reference: true, listen_goal_respected: true,
      independent_background_opt_out: true, total_opt_out_persisted: true, model_reference_is_explanation_only: true,
      mobile_bounds: bounds, operations, unexpected_console_errors: errors, human_participants: 0, clinical_efficacy_tested: false };
    fs.writeFileSync(path.join(output, 'browser-verification.json'), JSON.stringify(result, null, 2)); console.log(JSON.stringify(result, null, 2));
  } finally { await worker.stop(); if (browser) await browser.close(); vite.kill(); await new Promise<void>(resolve => server.close(() => resolve())); await fixture.cleanup(); }
}
main().catch((failure: unknown) => { console.error(failure); process.exitCode = 1; });
