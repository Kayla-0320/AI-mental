/** 真实 React + Node 业务服务 + Python HTTP；数据库使用内存替身。全部样例为自编工程夹具。 */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { SupportFeedbackService } from '../server/src/services/support-feedback.service';

async function main() {
  const repo = path.resolve(__dirname, '..');
  const output = process.env.ROUND1_OUTPUT || 'D:/Develop/AIC/_国赛优化/round1/browser';
  const python = process.env.ROUND1_PYTHON || 'D:/Develop/AIC/.venv/Scripts/python.exe';
  const vite = process.env.ROUND1_VITE || 'http://127.0.0.1:5173';
  process.env.ALGORITHM_API_URL = 'http://127.0.0.1:8279';
  const child = spawn(python, ['-X', 'utf8', 'scripts/round1_support_http_app.py', '8279'], {
    cwd: repo, stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true,
    env: { ...process.env, OPENBLAS_NUM_THREADS: '1', OMP_NUM_THREADS: '1', MKL_NUM_THREADS: '1' },
  });
  let serverError = '';
  child.stderr.on('data', chunk => { serverError += chunk.toString(); });
  let browser: any;
  try {
    let fixture: any;
    for (let n = 0; n < 80; n++) {
      try { fixture = await (await fetch('http://127.0.0.1:8279/fixture')).json(); break; }
      catch { await new Promise(resolve => setTimeout(resolve, 250)); }
    }
    assert.ok(fixture?.support_state, `Python HTTP startup failed: ${serverError.slice(-600)}`);
    const userMessage = { id: 'user-1', role: 'user', content: fixture.text, createdAt: '2026-11-15T00:00:00Z' };
    const assistant = { id: 'assistant-1', conversationId: 'round1-demo', role: 'assistant',
      content: '你愿意说说等消息时发生的事吗？', createdAt: '2026-11-15T00:00:01Z',
      sentiment: JSON.stringify({ support_state: fixture.support_state, audit_passed: true }) };
    let writes = 0, failNext = false;
    const actions: string[] = [];
    const db: any = {
      healingSession: { findFirst: async () => null },
      conversation: { findFirst: async ({ where }: any) => where.id === 'round1-demo' && where.userId === 'qa-user' ? { id: where.id } : null },
      message: {
        findFirst: async ({ where }: any) => where.id ? (where.id === assistant.id ? { ...assistant } : null) : { id: assistant.id },
        updateMany: async ({ where, data }: any) => {
          if (where.sentiment !== assistant.sentiment) return { count: 0 };
          assistant.sentiment = data.sentiment; writes++; return { count: 1 };
        },
      },
    };
    const service = new SupportFeedbackService(db);
    db.$queryRaw = async () => [{ id: 'round1-demo' }];
    db.$transaction = async (run: any) => run(db);
    const cdpLib = await import('./lib/cdp.mjs');
    browser = await cdpLib.launchBrowser({ port: Number(process.env.ROUND1_CDP_PORT || 9339), width: 1440, height: 1100 });
    const cdp = browser.cdp;
    const interceptErrors: string[] = [];
    cdp.ws.on('message', async (raw: Buffer) => {
      const event = JSON.parse(raw.toString());
      if (event.method !== 'Fetch.requestPaused') return;
      const { requestId, request } = event.params;
      try {
        const url = new URL(request.url);
        let data: any = {}, status = 200;
        if (url.pathname.endsWith('/support-feedback')) {
          if (failNext) { failNext = false; status = 503; data = { code: 503, message: '测试中断：反馈未保存' }; }
          else {
            const input = JSON.parse(request.postData);
            actions.push(input.action);
            try { data = { code: 0, data: await service.submit('round1-demo', 'qa-user', 'assistant-1', input) }; }
            catch (error: any) { status = error.statusCode || 500; data = { code: status, message: error.message }; }
          }
        } else if (url.pathname === '/api/support-agent') data = { code: 0, data: {
          state: { schema: 'support-agent.v1', revision: 0, enabled: false, preferredTool: 'none', actions: [],
            background: { enabled: false, notifyInApp: false, intervalHours: 12, quietStart: 22, quietEnd: 8, nextRunAt: null, events: [] } },
          trend: { observations: 0, latestScore: null, mean: null, change: null, stale: true, label: '自编工程夹具', points: [] },
          suggestionEligible: false, context: { goal: 'unspecified', status: 'unavailable', evidence: [], revoked: 0 }, modelReference: null,
        } };
        else if (url.pathname.includes('/messages')) data = { code: 0, data: { messages: [userMessage, assistant], total: 2 } };
        else if (url.pathname.includes('/consultation/conversations')) data = { code: 0, data: { conversations: [{ id: 'round1-demo', title: '工程测试对话', updatedAt: assistant.createdAt }], total: 1 } };
        else if (url.pathname.includes('/notifications')) data = { code: 0, data: { notifications: [], unreadCount: 0 } };
        else data = { code: 0, data: {} };
        await cdp.send('Fetch.fulfillRequest', { requestId, responseCode: status,
          responseHeaders: [{ name: 'Content-Type', value: 'application/json; charset=utf-8' }],
          body: Buffer.from(JSON.stringify(data)).toString('base64') });
      } catch (error: any) { interceptErrors.push(error.message); await cdp.send('Fetch.failRequest', { requestId, errorReason: 'Failed' }); }
    });
    await cdp.send('Fetch.enable', { patterns: [{ urlPattern: '*/api/*', requestStage: 'Request' }] });
    await cdpLib.injectAuth(cdp, { accessToken: 'synthetic-local-test-token', refreshToken: '',
      user: { id: 'qa-user', nickname: '测试用户', role: 'PATIENT', email: 'qa@example.invalid' } });
    const url = `${vite}/chat/round1-demo`;
    await cdpLib.goto(cdp, url, { waitMs: 500 });
    const loaded = await cdp.waitForText((text: string) => text.includes('我听到你说的是') || text.includes('页面出错了'), { timeoutMs: 30000 });
    assert.ok(loaded.includes('我听到你说的是'), `echo did not load: ${JSON.stringify(cdp.consoleErrors)}`);
    const click = (text: string, slot = '') => cdp.eval(`(() => { const card=document.querySelector('[data-testid="support-understanding"]'); const root=${slot ? `card.querySelector('[data-evidence-slot="${slot}"]')` : 'card'}; const button=[...root.querySelectorAll('button')].find(b=>b.textContent.replace(/\\s/g,'')===${JSON.stringify(text)}); if(!button)throw Error('button missing'); button.click(); })()`);
    const settle = () => new Promise(resolve => setTimeout(resolve, 800));
    // 点某一句原话，就地变成输入框；不再有「改一改」二级按钮。
    const editSlot = async (slot: string) => {
      await cdp.eval(`(() => { const b=document.querySelector('[data-testid="support-understanding"] [data-evidence-slot="${slot}"]'); if(!b)throw Error('evidence slot missing'); b.click(); })()`);
      await cdp.waitForText((text: string) => text.includes('我可能没听准'), { timeoutMs: 8000 });
    };
    fs.mkdirSync(output, { recursive: true });
    // 默认态：只陈述「我听到你说的是」，不要求用户逐条确认。
    await cdp.screenshot(path.join(output, '01-proposed.png'));
    await editSlot('thought');
    await cdp.eval(`(() => {const el=document.querySelector('textarea[aria-label="修正我的意思"]'); const set=Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set; set.call(el,'我担心自己被忽视，不是觉得他讨厌我'); el.dispatchEvent(new Event('input',{bubbles:true}));})()`);
    await click('改好了'); await settle();
    const thought = () => JSON.parse(assistant.sentiment).support_state.items.find((item: any) => item.slot === 'thought');
    assert.equal(thought().status, 'corrected');
    await editSlot('thought');
    await cdp.eval(`(() => {const el=document.querySelector('textarea[aria-label="修正我的意思"]'); Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set.call(el,'这段修正草稿还没有保存'); el.dispatchEvent(new Event('input',{bubbles:true}));})()`);
    // 陪伴方式仍是三个轻量胶囊，点一下即保存，不打断正在编辑的修正。
    await click('一起想办法'); await settle();
    assert.equal(JSON.parse(assistant.sentiment).support_state.goal, 'act');
    assert.equal(JSON.parse(assistant.sentiment).support_state.plan.action, 'collaborate_action');
    assert.equal(await cdp.eval(`document.querySelector('textarea[aria-label="修正我的意思"]').value`), '这段修正草稿还没有保存');
    await click('先不改了');
    await cdp.screenshot(path.join(output, '02-corrected-action-goal.png'));
    await cdp.send('Page.reload');
    await cdp.waitForText((text: string) => text.includes('我担心自己被忽视'));
    const beforeFailure = assistant.sentiment, beforeWrites = writes;
    failNext = true;
    await editSlot('thought');
    await click('别记这句');
    await cdp.waitForText((text: string) => text.includes('测试中断：反馈未保存'));
    assert.equal(assistant.sentiment, beforeFailure); assert.equal(writes, beforeWrites);
    await cdp.screenshot(path.join(output, '03-error-not-saved.png'));
    await click('这次先不记了'); await settle();
    assert.ok(JSON.parse(assistant.sentiment).support_state.declined);
    await click('还是记一下吧'); await settle();
    assert.ok(!JSON.parse(assistant.sentiment).support_state.declined);
    // 短开场没有重点时只剩一个可选的陪伴方式，不再有“待确认的重点”。
    const withEvidence = assistant.sentiment;
    assistant.sentiment = JSON.stringify({ ...JSON.parse(withEvidence), support_state: {
      ...fixture.support_state, revision: 1, items: [], feedback_log: [],
    } });
    await cdp.send('Page.reload');
    await cdp.waitForText((text: string) => text.includes('这次，你想我怎么陪你？'));
    const emptyText = await cdp.eval(`document.querySelector('[data-testid="support-understanding"]').innerText`);
    assert.ok(emptyText.includes('这次，你想我怎么陪你？'));
    assert.ok(!emptyText.includes('这些片段来自你的原话'));
    assert.ok(!emptyText.includes('我听到你说的是'));
    await cdp.screenshot(path.join(output, '05-empty-state.png'));
    await click('先听我说'); await settle();
    assert.equal(JSON.parse(assistant.sentiment).support_state.goal, 'listen');
    assistant.sentiment = withEvidence;
    await cdp.send('Page.reload');
    await cdp.waitForText((text: string) => text.includes('我担心自己被忽视'));
    await cdp.send('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
    await new Promise(resolve => setTimeout(resolve, 400));
    await cdp.eval(`document.querySelector('[data-testid="support-understanding"]').scrollIntoView({block:'center'})`);
    await cdp.screenshot(path.join(output, '04-mobile.png'));
    const cardBounds = await cdp.eval(`(() => {const r=document.querySelector('[data-testid="support-understanding"]').getBoundingClientRect();return {left:r.left,right:r.right,width:document.documentElement.clientWidth,document_width:document.documentElement.scrollWidth};})()`);
    assert.ok(cardBounds.right <= cardBounds.width + 1 && cardBounds.left >= 0, `card overflows viewport: ${JSON.stringify(cardBounds)}`);
    assert.ok(cardBounds.document_width <= 391, `mobile page overflows viewport: ${JSON.stringify(cardBounds)}`);
    assert.deepEqual(interceptErrors, []);
    // 记录已有组件的两种弃用提示；未知错误仍使回归失败。
    const knownDeprecations = cdp.consoleErrors.filter((message: string) =>
      message.startsWith('Warning: [antd: Modal] `destroyOnClose` is deprecated.')
      || message.startsWith('Warning: [antd: Card] `bodyStyle` is deprecated.'));
    const relevantErrors = cdp.consoleErrors.filter((message: string) => !message.includes('503') && !message.includes('favicon') && !knownDeprecations.includes(message));
    assert.deepEqual(relevantErrors, []);
    const report = { status: 'passed', scope: 'real React, Node service, Python HTTP; in-memory database and synthetic messages; no external LLM or human participants',
      actions, writes, persisted_after_reload: true, failed_write_preserved_state: true, mobile_card_bounds: cardBounds,
      unsaved_correction_survives_goal_change: true,
      empty_state_explained_and_preferences_saved: true,
      unexpected_console_errors: relevantErrors, known_component_deprecations: knownDeprecations,
      console_errors: cdp.consoleErrors, synthetic_fixture: fixture.source };
    fs.writeFileSync(path.join(output, 'browser-verification.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report));
  } finally { if (browser) await browser.close(); child.kill(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
