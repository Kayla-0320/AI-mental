/**
 * `/algorithm` 反向代理（HTTP + WebSocket）—— 施工单 §S8
 *
 * ## 为什么需要它
 *
 * 语音链路的两个关键通道都挂在 `/algorithm` 前缀下：
 *
 * ```
 *   ws://<host>/algorithm/api/v1/asr/ws     ← 边说边出字（流式语音识别）
 *   ws://<host>/algorithm/api/v1/tts/ws     ← 逐句合成出声
 *   http://<host>/algorithm/api/v1/asr/status
 * ```
 *
 * 开发时这条前缀由 **vite 的 dev server** 代理到 8001（`client/vite.config.ts:13-20`）。
 * 但 `npm run build` 出来的生产包里没有 vite —— 页面由 Node 服务提供，
 * 而 Node 服务**既没有 `/algorithm` 代理，也没有 `express.static`**。
 * 后果是：页面正常、按钮正常、**就是没声音 / 打不出字**。
 *
 * ⚠️ 这个失败模式与本次会话已经踩过的"8001 用错解释器导致 `/asr/ws` 404"
 * **完全同型**：都是"HTTP 层看起来一切正常，只有 WS 静默失败"。
 * 所以代理必须**同时**处理 `upgrade`，只做 HTTP 代理等于没做。
 *
 * ## 为什么不装 `http-proxy-middleware`
 *
 * `ws: true` 的支持在 `http-proxy` 里，而它不在依赖表里。为了一个 20 行的转发
 * 去加一个运行时依赖（还带一串传递依赖），不如直接用 `http` / `net`：
 * 转发规则只有一条 —— **去掉 `/algorithm` 前缀，其余原样转发到 8001**。
 *
 * ## 与 Socket.IO 的关系（这一条最容易写错）
 *
 * Socket.IO 也挂在同一个 http server 的 `upgrade` 事件上。两个监听器会**都**被调用，
 * 所以这里必须严格按前缀判断：不是 `/algorithm` 开头就**立刻返回**，
 * 一个字节都别写进 socket，否则会把 Socket.IO 的握手弄坏（表现为"聊天页实时消息时好时坏"）。
 */

import type { Express } from 'express';
import { request as httpRequest, type IncomingMessage, type Server } from 'http';
import { connect as netConnect, type Socket } from 'net';
import type { Duplex } from 'stream';

/** 算法服务地址；与 `scripts/run-algorithm.mjs` 的默认端口一致 */
const ALGORITHM_HOST = process.env.ALGORITHM_HOST || '127.0.0.1';
const ALGORITHM_PORT = parseInt(process.env.ALGORITHM_PORT || '8001', 10);

/** 前缀；本模块只处理以它开头的请求 */
const PREFIX = '/algorithm';

/** 上游请求超时（毫秒）。本地服务，正常都是毫秒级 —— 给足余量但不无限等 */
const UPSTREAM_TIMEOUT_MS = 120_000;

/** 把 `/algorithm/api/v1/asr/status` 变成 `/api/v1/asr/status` */
function upstreamPath(url: string): string {
  const stripped = url.slice(PREFIX.length);
  return stripped.startsWith('/') ? stripped : `/${stripped}`;
}

/**
 * 挂载代理。
 *
 * @param app    express 应用（处理普通 HTTP）
 * @param server 同一个 http server（处理 WebSocket 的 `upgrade`）
 */
export function mountAlgorithmProxy(app: Express, server: Server): void {
  // ── 普通 HTTP：/algorithm/... → 8001 的同名路径 ──────────────────────────
  app.use(PREFIX, (req, res) => {
    const path = upstreamPath(req.originalUrl || req.url);
    const proxied = httpRequest(
      {
        host: ALGORITHM_HOST,
        port: ALGORITHM_PORT,
        method: req.method,
        path,
        headers: { ...req.headers, host: `${ALGORITHM_HOST}:${ALGORITHM_PORT}` },
        timeout: UPSTREAM_TIMEOUT_MS,
      },
      (upstream) => {
        // 透传状态码与响应头（WS 握手不走这里，见下）
        res.writeHead(upstream.statusCode ?? 502, upstream.headers);
        upstream.pipe(res);
      },
    );

    proxied.on('timeout', () => {
      proxied.destroy(new Error('上游超时'));
    });
    proxied.on('error', (err) => {
      // 算法服务没起来时，必须给一个**能看懂**的错，而不是挂起或 500 空响应。
      // 这个提示会直接显示在通话页/聊天页的错误条里。
      if (res.headersSent) {
        res.destroy();
        return;
      }
      res.status(502).json({
        code: 502,
        message: `算法服务不可用（${ALGORITHM_HOST}:${ALGORITHM_PORT}）：${err.message}。请先运行 npm run dev:algorithm`,
      });
    });

    req.pipe(proxied);
  });

  // ── WebSocket：/algorithm/... 的 upgrade 原样转发 ───────────────────────
  server.on('upgrade', (req: IncomingMessage, socket: Duplex, head: Buffer) => {
    const url = req.url || '';
    // ⚠️ 不是我们的路径就立刻返回，别碰 socket（Socket.IO 的握手也在同一事件上）
    if (!url.startsWith(`${PREFIX}/`)) return;

    const upstream: Socket = netConnect(ALGORITHM_PORT, ALGORITHM_HOST, () => {
      // 手写请求行 + 头：等价于 `http-proxy` 的 `ws: true`，但只有一条规则
      const headers = { ...req.headers, host: `${ALGORITHM_HOST}:${ALGORITHM_PORT}` };
      const lines = [`GET ${upstreamPath(url)} HTTP/1.1`];
      for (const [key, value] of Object.entries(headers)) {
        if (value === undefined) continue;
        if (Array.isArray(value)) {
          for (const v of value) lines.push(`${key}: ${v}`);
        } else {
          lines.push(`${key}: ${value}`);
        }
      }
      upstream.write(`${lines.join('\r\n')}\r\n\r\n`);
      // 客户端在握手包里已经带上来的字节要一起转发，否则首个音频块会丢
      if (head && head.length > 0) upstream.write(head);
      // 双向对拷（先挂在 connect 回调里，避免建立前的数据丢失）
      socket.pipe(upstream).pipe(socket);
    });

    const cleanup = (): void => {
      socket.destroy();
      upstream.destroy();
    };
    upstream.on('error', cleanup);
    socket.on('error', cleanup);
    // 通话是长连接：两边的空闲超时必须关掉，否则一段安静就会被掐断
    // （`Duplex` 上没有 `setTimeout` 的类型声明，所以先窄化到 `net.Socket` 再调）
    if ('setTimeout' in socket && typeof socket.setTimeout === 'function') {
      (socket as Socket).setTimeout(0);
    }
    upstream.setTimeout(0);
  });

  console.log(
    `[algorithm-proxy] /algorithm → http://${ALGORITHM_HOST}:${ALGORITHM_PORT}（HTTP + WebSocket）`,
  );
}

export { ALGORITHM_HOST, ALGORITHM_PORT, PREFIX as ALGORITHM_PREFIX };
