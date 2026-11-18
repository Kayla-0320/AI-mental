/** RQI 回归：真实聊天页面，自编流式夹具；消息和状态上报不写入数据库。 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { launchBrowser, injectAuth, login, goto, sleep, REPO } from './lib/cdp.mjs';

const origin = process.env.RQI_VITE || 'http://localhost:5173';
const auth = await login('http://127.0.0.1:3000', 'patient@mental.com', '123456');
const browser = await launchBrowser({ port: 9358, width: 1280, height: 900,
  extraArgs: ['--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows'] });
const output = path.join(REPO, 'output/rqi-verification');
const stamp = new Date().toISOString();
const message = (id, role, content, rqi) => ({ id, role, content, createdAt: stamp,
  ...(rqi === undefined ? {} : { sentiment: JSON.stringify({ tamer_result: { rqi } }) }) });
const conversations = new Map([
  ['rqi-a', { id: 'rqi-a', title: 'RQI 渐进回归', messages: [] }],
  ['rqi-b', { id: 'rqi-b', title: 'RQI 历史会话', messages: [
    message('b-u', 'user', '一段历史对话'), message('b-a', 'assistant', '历史回复', .24),
    { ...message('b-bad', 'assistant', '损坏元数据'), sentiment: '{broken' },
    message('b-invalid', 'assistant', '越界元数据', 8),
  ] }],
  ['rqi-new', { id: 'rqi-new', title: 'RQI 新对话', messages: [
    message('new-u', 'user', '朋友，我相信你，只有你知道我的秘密。我最近焦虑又难受。'),
    message('new-a', 'assistant', '没有 RQI 的旧格式回复'),
  ] }],
]);
const interceptionErrors = [];
let unexpectedWrites = 0;
let historyGate = null;

try {
  const cdp = browser.cdp;
  await injectAuth(cdp, auth);
  // 真正的 ReadableStream 分开下发 user / done，重现“已发消息、后端尚未完成”。
  await cdp.send('Page.addScriptToEvaluateOnNewDocument', { source: `(() => {
    const realFetch = window.fetch.bind(window);
    window.__rqiHistory = [];
    window.fetch = (input, init) => {
      const url = new URL(typeof input === 'string' ? input : input.url, location.origin);
      if (!url.pathname.endsWith('/messages/stream')) return realFetch(input, init);
      const content = JSON.parse(init.body).content;
      const turn = window.__rqiHistory.filter(m => m.role === 'user').length + 1;
      const userMessage = { id: 'user-' + turn, role: 'user', content, createdAt: new Date().toISOString() };
      window.__rqiHistory.push(userMessage);
      const encoder = new TextEncoder();
      return Promise.resolve(new Response(new ReadableStream({ start(controller) {
        const emit = (event, data) => controller.enqueue(encoder.encode('event: ' + event + '\\ndata: ' + JSON.stringify(data) + '\\n\\n'));
        emit('user', { userMessage });
        emit('delta', { text: '自编流式回复' });
        window.__finishRQITurn = () => {
          const tamerResult = { rqi: [.06, .11, .15][turn - 1], relationship_depth: '陌生期', mis: 0, trajectory: 'stable' };
          const aiMessage = { id: 'assistant-' + turn, role: 'assistant', content: '自编流式回复 ' + turn,
            createdAt: new Date().toISOString(), sentiment: JSON.stringify({ tamer_result: tamerResult }) };
          window.__rqiHistory.push(aiMessage);
          emit('done', { aiMessage, tamerResult });
          controller.close();
          window.__finishRQITurn = null;
        };
      } }), { headers: { 'Content-Type': 'text/event-stream' } }));
    };
  })()` });
  cdp.ws.on('message', async raw => {
    const event = JSON.parse(raw.toString());
    if (event.method !== 'Fetch.requestPaused') return;
    const { requestId, request } = event.params;
    const url = new URL(request.url);
    let body;
    try {
      if (url.pathname === '/api/consultation/conversations') {
        assert.equal(request.method, 'GET');
        body = { code: 0, data: { conversations: [...conversations.values()], total: conversations.size } };
      } else if (url.pathname.startsWith('/api/consultation/conversations/') && url.pathname.endsWith('/messages')) {
        assert.equal(request.method, 'GET');
        const conversation = conversations.get(url.pathname.split('/')[4]);
        assert.ok(conversation);
        if (conversation.id === 'rqi-b' && historyGate) await historyGate;
        body = { code: 0, data: { messages: conversation.messages, total: conversation.messages.length } };
      } else if (url.pathname === '/api/profile/anxiety-report' || url.pathname === '/api/algorithm/baseline/sync') {
        body = { code: 0, data: {} };
      } else if (url.pathname.endsWith('/perception/analyze')) {
        body = { text_emotion_probs: [0, 0, 0, 0, 1], confidence: 1, evidence: [] };
      } else if (request.method === 'GET') {
        await cdp.send('Fetch.continueRequest', { requestId });
        return;
      } else {
        unexpectedWrites++;
        throw new Error('未预期写入，已拦截：' + request.method + ' ' + url.pathname);
      }
      await cdp.send('Fetch.fulfillRequest', { requestId, responseCode: 200,
        responseHeaders: [{ name: 'Content-Type', value: 'application/json' }],
        body: Buffer.from(JSON.stringify(body)).toString('base64') });
    } catch (error) {
      interceptionErrors.push(error.message);
      await cdp.send('Fetch.failRequest', { requestId, errorReason: 'Failed' }).catch(() => {});
    }
  });
  await cdp.send('Fetch.enable', { patterns: [{ urlPattern: '*/api/*' }] });
  const waitUntil = async (predicate, label) => {
    const deadline = Date.now() + 10000;
    while (Date.now() < deadline) {
      if (await predicate()) return;
      await sleep(50);
    }
    throw new Error(label + ': ' + JSON.stringify({ interceptionErrors, errors: cdp.consoleErrors }));
  };
  const readRqi = () => cdp.eval(`Number(document.querySelector('[data-testid="relationship-stage"]')?.innerText.match(/RQI (\\d+\\.\\d+)/)?.[1] ?? NaN)`);
  const send = async content => {
    await cdp.eval(`(() => {
      const input = document.querySelector('.chat-composer input');
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(input, ${JSON.stringify(content)});
      input.dispatchEvent(new Event('input', { bubbles: true }));
    })()`);
    await sleep(50);
    await cdp.eval(`[...document.querySelectorAll('.chat-composer button')].find(b => b.textContent === '发送').click()`);
    await waitUntil(() => cdp.eval(`typeof window.__finishRQITurn === 'function'`), '流式用户消息未到达');
    await sleep(100);
  };
  const select = title => cdp.eval(`[...document.querySelectorAll('.chat-history .ant-list-item')].find(row => row.textContent.includes(${JSON.stringify(title)})).click()`);

  await goto(cdp, origin + '/chat/rqi-a');
  await waitUntil(() => cdp.eval(`!!document.querySelector('.chat-composer input')`), '聊天页面未加载');
  assert.equal(await cdp.eval(`!!document.querySelector('[data-testid="relationship-stage"]')`), false);
  await send('朋友，我相信你，只有你知道我的秘密。我最近焦虑又难受。');
  assert.equal(await readRqi(), 0, '首轮等待时不能显示偏高的本地估算');
  await sleep(350);
  assert.equal(await readRqi(), 0);

  const trajectories = [];
  for (const [turn, target] of [[1, .06], [2, .11]]) {
    if (turn === 2) {
      await send('嗯');
      assert.equal(await readRqi(), .06, '等待下一轮时应保持已确认读数');
    }
    await cdp.eval(`(() => {
      window.__rqiSamples = [];
      window.__rqiMissing = [];
      const until = performance.now() + 1200;
      const sample = () => {
        const node = document.querySelector('[data-testid="relationship-stage"]');
        const match = node?.innerText.match(/RQI (\\d+\\.\\d+)/);
        if (match) window.__rqiSamples.push(Number(match[1]));
        else window.__rqiMissing.push(node?.innerText ?? 'missing');
        if (performance.now() < until) requestAnimationFrame(sample);
      };
      requestAnimationFrame(sample);
      window.__finishRQITurn();
    })()`);
    await sleep(1300);
    const samples = await cdp.eval('window.__rqiSamples');
    assert.deepEqual(await cdp.eval('window.__rqiMissing'), [], '动画过程中不能出现负数或无效读数');
    assert.ok(samples.length > 10, JSON.stringify({ samples, missing: await cdp.eval('window.__rqiMissing'), errors: cdp.consoleErrors }));
    assert.ok(new Set(samples).size > 2, '应有多个中间值，逐步上升');
    assert.ok(samples.every((v, i) => v <= target && (i === 0 || v >= samples[i - 1])), JSON.stringify(samples));
    assert.equal(await readRqi(), target);
    trajectories.push([...new Set(samples)]);
  }
  fs.mkdirSync(output, { recursive: true });
  await cdp.screenshot(path.join(output, 'rqi-gradual.png'));
  conversations.get('rqi-a').messages = await cdp.eval('window.__rqiHistory');

  // 切历史时加载窗口不显示旧会话；读取有效保存值，忽略损坏和越界记录。
  let releaseHistory;
  historyGate = new Promise(resolve => { releaseHistory = resolve; });
  await select('RQI 历史会话');
  await sleep(150);
  assert.equal(await cdp.eval(`!!document.querySelector('[data-testid="relationship-stage"]')`), false);
  releaseHistory(); historyGate = null;
  await waitUntil(async () => await readRqi() === .24, '历史 RQI 未恢复');
  await select('RQI 新对话');
  await waitUntil(async () => await readRqi() === 0, '新对话沿用了上个会话的读数');
  await select('RQI 渐进回归');
  await waitUntil(async () => await readRqi() === .11, '重新打开会话未恢复保存值');
  await goto(cdp, origin + '/chat/rqi-a');
  await waitUntil(async () => await readRqi() === .11, '刷新会话未恢复保存值');
  assert.equal(interceptionErrors.length, 0, JSON.stringify(interceptionErrors));
  assert.equal(unexpectedWrites, 0);
  const errors = cdp.consoleErrors.filter(text => !text.startsWith('[vite] failed to connect to websocket.')
    && !/^Warning: \[antd: (Modal|Card)\] `(destroyOnClose|bodyStyle)` is deprecated\./.test(text));
  assert.equal(errors.length, 0, JSON.stringify(errors));
  const report = { status: 'passed', scope: 'actual Chat UI with synthetic streamed messages; no fixture writes',
    starts_at_zero: true, trajectories, pending_keeps_previous: true, conversation_isolation: true,
    history_and_reload_restored: true, invalid_metadata_ignored: true, application_errors: errors };
  fs.writeFileSync(path.join(output, 'verification.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
} finally { await browser.close(); }
