/**
 * 「HTTP 状态码必须能被调用方看见」的**离线回归**（不需要浏览器、不需要真机）。
 *
 * 为什么需要它 —— 用户截图（2026-10-11 通话页）：界面上弹出一句
 *   「对话不存在」
 * 追问下去，根因是一条**旧的/写错的会话链接**：
 *
 *   1. `VoiceCall` 原来对地址栏里的 `:conversationId` **不做任何校验**，
 *      直接把它交给 `useVoiceCall`；
 *   2. 用户按"开始通话"→ 说话 → 该轮 `POST .../messages/stream`；
 *   3. 服务端 `sendMessageStream` 先 `writeHead(200)`（SSE 已经开流），
 *      之后服务层才抛 `AppError('对话不存在', 404)`，只能作为 `error` 事件下发；
 *   4. 客户端把 `errorMessage` 原样当成提示条 → 界面显示「对话不存在」，
 *      而**每一轮**都会重复一遍，看着就是"AI 不回复"。
 *
 * 修法是通话页在进入时用一次只读拉取校验会话；**只有 404 才自动换新会话**，
 * 网络抖动/500 绝不能换（那会莫名丢掉用户原本的会话）。
 * 这个判断成立的前提就是本文件守的东西：`api.ts` 的响应拦截器必须把
 * **HTTP 状态码**附在抛出的错误上 —— 它原来只透传响应体 `{code,message}`，
 * 调用方只能靠中文文案猜，文案一改就失效。
 *
 * 用法（仓库根目录）：
 *   node "server/node_modules/tsx/dist/cli.mjs" scripts/verify_api_error_status.ts
 * 退出码 1 表示任一条断言失败（可直接接 CI）。
 */

import api from '../client/src/services/api';

let failed = 0;
function check(ok: boolean, what: string, detail = ''): void {
  if (ok) {
    console.log(`  [OK]   ${what}${detail ? `  ${detail}` : ''}`);
  } else {
    failed += 1;
    console.log(`  [FAIL] ${what}${detail ? `  ${detail}` : ''}`);
  }
}

/** 让这一次请求直接抛出带 `response` 的错误（等价于 axios 收到 4xx/5xx） */
function rejectWith(status: number, data: unknown): void {
  api.defaults.adapter = async (config) => {
    const err = Object.assign(new Error(`Request failed with status code ${status}`), {
      config,
      response: { status, data, headers: {}, config },
    });
    throw err;
  };
}

/** 让这一次请求成功返回 */
function resolveWith(status: number, data: unknown): void {
  api.defaults.adapter = async (config) => ({
    status,
    statusText: 'OK',
    headers: {},
    config,
    data,
  });
}

/** 让这一次请求以"压根没连上"告终（没有 response） */
function rejectNetwork(): void {
  api.defaults.adapter = async () => {
    throw Object.assign(new Error('Network Error'), { code: 'ERR_NETWORK' });
  };
}

async function main(): Promise<void> {
  // 拦截器里会读 `localStorage`（拿 accessToken）；Node 里没有这个全局对象，
  // 请求阶段会直接抛 ReferenceError —— 补一个最小替身（在发出任何请求之前）。
  (globalThis as { localStorage?: unknown }).localStorage = {
    getItem: () => null,
    setItem: () => undefined,
    removeItem: () => undefined,
  };

  console.log('group 1 · 404 必须能被识别（通话页据此自动换会话）');
  rejectWith(404, { code: 404, message: '对话不存在' });
  try {
    await api.get('/consultation/conversations/nope/messages');
    check(false, '404 应该 reject');
  } catch (e) {
    const err = e as { status?: number; message?: string; code?: number };
    check(err.status === 404, 'HTTP 状态码被带上（err.status === 404）', `status=${String(err.status)}`);
    check(
      err.message === '对话不存在',
      '响应体字段原样保留（err.message）',
      `message=${String(err.message)}`,
    );
    // 关键：判断只能看 `status`。`code` 是业务码，服务端现在恰好也填 404，
    // 但它不是 HTTP 状态的合同，别拿它当依据。
    check(err.code === 404, '业务码 code 仍在（仅作参考，不作判据）');
  }

  console.log('group 2 · 其它失败**不能**被误判成 404');
  rejectWith(500, { code: 500, message: '数据库炸了' });
  try {
    await api.get('/x');
    check(false, '500 应该 reject');
  } catch (e) {
    const err = e as { status?: number; message?: string };
    check(err.status === 500, 'err.status === 500', `status=${String(err.status)}`);
    check(err.status !== 404, '不会被误判为"会话不存在"');
  }

  rejectNetwork();
  try {
    await api.get('/x');
    check(false, '断网应该 reject');
  } catch (e) {
    const err = e as { status?: number };
    check(err?.status === undefined, '断网没有 status（调用方必须据此放弃自动换会话）');
  }

  rejectWith(404, 'plain text body');
  try {
    await api.get('/x');
    check(false, '非对象响应体也应 reject');
  } catch (e) {
    check(
      typeof e === 'string' && e === 'plain text body',
      '非对象响应体保持原样抛出（不硬塞 status 字段）',
      `got=${JSON.stringify(e)}`,
    );
  }

  console.log('group 3 · 正常响应仍然只回传 data（既有契约不变）');
  resolveWith(200, { code: 200, data: { ok: true } });
  const okRes = (await api.get('/x')) as { data?: { ok?: boolean } };
  check(okRes?.data?.ok === true, '拦截器仍然把 response.data 直接返回');

  console.log('');
  if (failed > 0) {
    console.log(`✗ ${failed} 条断言失败`);
    process.exit(1);
  }
  console.log('✓ 全部通过');
}

main().catch((e: unknown) => {
  console.error('harness 自身出错：', e);
  process.exit(1);
});
