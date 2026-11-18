/** 回归验证：新对话发送、连续新建、重复提交和创建失败后的草稿恢复。 */
import assert from 'node:assert/strict';
import { launchBrowser, injectAuth, login, goto, sleep } from './lib/cdp.mjs';

const auth = await login('http://127.0.0.1:3000', 'patient@mental.com', '123456');
const browser = await launchBrowser({ port: 9357, width: 1440, height: 1000 });
const conversations = new Map();
const sends = [];
const interceptionErrors = [];
let createAttempts = 0;
let failNextCreate = false;
let createGate = null;
let streamGate = null;
let unexpectedWrites = 0;

try {
  const cdp = browser.cdp;
  await injectAuth(cdp, auth);
  cdp.ws.on('message', async raw => {
    const event = JSON.parse(raw.toString());
    if (event.method !== 'Fetch.requestPaused') return;
    const { requestId, request } = event.params;
    const url = new URL(request.url);
    let status = 200;
    let type = 'application/json';
    let body;
    try {
      if (url.pathname === '/api/consultation/conversations') {
        if (request.method === 'POST') {
          createAttempts++;
          if (createGate) await createGate;
          if (failNextCreate) {
            failNextCreate = false;
            status = 503;
            body = { code: 503, message: '测试：创建暂时失败' };
          } else {
            const conversation = { id: `send-fixture-${createAttempts}`, title: '发送回归测试', messages: [] };
            conversations.set(conversation.id, conversation);
            body = { code: 0, data: conversation };
          }
        } else {
          body = { code: 0, data: { conversations: [...conversations.values()], total: conversations.size } };
        }
      } else if (url.pathname.startsWith('/api/consultation/conversations/')) {
        const conversation = conversations.get(url.pathname.split('/')[4]);
        assert.ok(conversation, `未知会话：${url.pathname}`);
        if (url.pathname.endsWith('/messages/stream')) {
          const { content } = JSON.parse(request.postData);
          sends.push({ conversationId: conversation.id, content });
          const createdAt = new Date().toISOString();
          const userMessage = { id: `user-${sends.length}`, role: 'user', content, createdAt };
          const aiMessage = { id: `assistant-${sends.length}`, role: 'assistant', content: `测试回复 ${sends.length}`, createdAt };
          conversation.messages.push(userMessage, aiMessage);
          if (streamGate) await streamGate;
          type = 'text/event-stream';
          body = `event: user\ndata: ${JSON.stringify({ userMessage })}\n\n`
            + `event: done\ndata: ${JSON.stringify({ aiMessage })}\n\n`;
        } else {
          assert.equal(request.method, 'GET');
          body = { code: 0, data: { messages: conversation.messages, total: conversation.messages.length } };
        }
      } else if (url.pathname === '/api/profile/anxiety-report' || url.pathname === '/api/algorithm/baseline/sync') {
        body = { code: 0, data: {} };
      } else if (url.pathname === '/api/v1/perception/analyze') {
        body = { text_emotion_probs: [0, 0, 0, 0, 1], confidence: 1, evidence: [] };
      } else if (request.method === 'GET') {
        await cdp.send('Fetch.continueRequest', { requestId });
        return;
      } else {
        unexpectedWrites++;
        throw new Error(`未预期写入，已拦截：${request.method} ${url.pathname}`);
      }
      await cdp.send('Fetch.fulfillRequest', {
        requestId, responseCode: status,
        responseHeaders: [{ name: 'Content-Type', value: type }],
        body: Buffer.from(typeof body === 'string' ? body : JSON.stringify(body)).toString('base64'),
      });
    } catch (error) {
      interceptionErrors.push(error.message);
      await cdp.send('Fetch.failRequest', { requestId, errorReason: 'Failed' }).catch(() => {});
    }
  });
  // 所有浏览器写请求均拦截，测试消息和状态上报不进入真实数据库。
  await cdp.send('Fetch.enable', { patterns: [{ urlPattern: '*/api/*' }] });
  await goto(cdp, 'http://localhost:5173/chat');

  const waitUntil = async (predicate, label, timeout = 8000) => {
    const deadline = Date.now() + timeout;
    while (Date.now() < deadline) {
      if (await predicate()) return;
      await sleep(50);
    }
    throw new Error(`${label}：${JSON.stringify({ createAttempts, sends, interceptionErrors, errors: cdp.consoleErrors })}`);
  };
  await waitUntil(() => cdp.eval(`!!document.querySelector('.chat-composer input')`), '输入框未显示');
  const typeText = async content => {
    await cdp.eval(`(() => {
      const input = document.querySelector('.chat-composer input');
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(input, ${JSON.stringify(content)});
      input.dispatchEvent(new Event('input', { bubbles: true }));
      input.focus();
    })()`);
    await sleep(50);
  };
  const clickSend = () => cdp.eval(`Array.from(document.querySelectorAll('.chat-composer button')).find(button => button.textContent === '发送').click()`);
  const pressEnter = async () => {
    await cdp.send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13 });
    await cdp.send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13 });
  };
  const waitForReply = number => waitUntil(
    () => cdp.eval(`document.querySelector('.chat-main').innerText.includes('测试回复 ${number}') && !document.querySelector('.chat-composer input').disabled`),
    `第 ${number} 条回复未完成`,
  );
  const fresh = async () => {
    await cdp.eval(`document.querySelector('.chat-history .anticon-plus').closest('button').click()`);
    await waitUntil(() => cdp.eval(`location.pathname === '/chat'`), '未返回新对话');
  };

  await typeText('第一次输入后点击发送');
  await clickSend();
  await waitForReply(1);
  assert.equal(conversations.size, 1);

  await typeText('在同一段对话里按回车');
  await pressEnter();
  await waitForReply(2);
  assert.equal(conversations.size, 1);
  assert.equal(sends[0].conversationId, sends[1].conversationId);

  // 核心回归：曾经成功建过会话后，再开新对话不能被旧的创建标记吞掉。
  await fresh();
  await typeText('第二段新对话仍能发送');
  await clickSend();
  await waitForReply(3);
  assert.equal(conversations.size, 2);
  assert.notEqual(sends[1].conversationId, sends[2].conversationId);

  await fresh();
  await cdp.eval(`document.querySelector('[data-topic="想找人说说"]').click()`);
  await waitForReply(4);
  assert.equal(conversations.size, 3);
  assert.equal(sends[3].content, '我有些话憋在心里很久了，想找个人说说。');

  await fresh();
  failNextCreate = true;
  const draft = '创建失败后保留的草稿';
  await typeText(draft);
  await clickSend();
  await waitUntil(() => cdp.eval(`document.querySelector('.chat-composer input').value === ${JSON.stringify(draft)} && !document.querySelector('.chat-composer input').disabled`), '创建失败后草稿未恢复');
  assert.equal(sends.length, 4);
  await clickSend();
  await waitForReply(5);
  assert.equal(sends[4].content, draft);

  await fresh();
  let releaseCreate;
  createGate = new Promise(resolve => { releaseCreate = resolve; });
  const attemptsBefore = createAttempts;
  await typeText('重复点击只创建一次');
  await clickSend();
  await waitUntil(() => createAttempts === attemptsBefore + 1, '创建请求未发出');
  assert.equal(await cdp.eval(`!!document.querySelector('.chat-composer .ant-btn-loading')`), true);
  await clickSend();
  await pressEnter();
  await sleep(200);
  assert.equal(createAttempts, attemptsBefore + 1);
  releaseCreate();
  createGate = null;
  await waitForReply(6);

  let releaseStream;
  streamGate = new Promise(resolve => { releaseStream = resolve; });
  await typeText('回复生成中不会重复发送');
  await pressEnter();
  await waitUntil(() => sends.length === 7, '已有会话未发送');
  await pressEnter();
  await clickSend();
  await sleep(200);
  assert.equal(sends.length, 7);
  releaseStream();
  streamGate = null;
  await waitForReply(7);

  assert.equal(unexpectedWrites, 0, JSON.stringify(interceptionErrors));
  assert.deepEqual(interceptionErrors, []);
  const errors = cdp.consoleErrors.filter(error => !error.startsWith('Warning: [antd:') && !error.startsWith('Failed to load resource: the server responded with a status of 503'));
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ status: 'passed', sends: sends.length, conversations: conversations.size,
    checked: ['首次按钮发送', '已有对话回车发送', '连续新建对话', '话题发送', '创建失败保留草稿并重试', '创建中重复提交', '回复中重复提交'], realChatWrites: 0 }, null, 2));
} finally {
  await browser.close();
}
// CDP 客户端的超时计时器会保留句柄，检查完成后显式结束。
process.exit(0);
