import { io, Socket } from 'socket.io-client';

const SOCKET_URL = (import.meta as any).env?.VITE_API_URL || 'http://localhost:3000';

let socket: Socket | null = null;

/**
 * 解析用于握手认证的 JWT。
 *
 * 全站唯一存放访问令牌的键是 `accessToken`（见 store/authStore.ts、services/api.ts），
 * 因此这里必须读 `accessToken`。此前读的 `token` 键在全代码库中从未被写入过，
 * 导致 9 处 `getSocket()` 调用都以 null 令牌握手，被服务端
 * （server/src/socket/index.ts 认证中间件）以「认证失败：未提供令牌」拒绝，
 * 实时通道（WebRTC 信令 / 多模态同步 / 专家消息）全部不可用。
 */
const resolveToken = (token?: string): string | null =>
  token || localStorage.getItem('accessToken');

export const getSocket = (token?: string): Socket => {
  const authToken = resolveToken(token);

  if (socket) {
    const currentToken = (socket.auth as { token?: string | null } | undefined)?.token ?? null;
    // 仅在「连接仍存活且令牌未变」时复用；否则重建，
    // 避免登录/登出换令牌后继续复用带着旧令牌或已失效的连接。
    if (socket.connected && currentToken === authToken) return socket;

    socket.removeAllListeners();
    socket.disconnect();
    socket = null;
  }

  socket = io(SOCKET_URL, {
    auth: { token: authToken },
    transports: ['websocket', 'polling'],
  });

  socket.on('connect', () => {
    console.log('[Socket] 已连接:', socket?.id);
  });

  socket.on('disconnect', () => {
    console.log('[Socket] 已断开');
  });

  socket.on('error', (err) => {
    console.error('[Socket] 错误:', err);
  });

  return socket;
};

export const disconnectSocket = () => {
  if (socket) {
    socket.disconnect();
    socket = null;
  }
};
