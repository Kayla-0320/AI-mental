import { io, Socket } from 'socket.io-client';

const SOCKET_URL = (import.meta as any).env?.VITE_API_URL || 'http://localhost:3001';

let socket: Socket | null = null;

export const getSocket = (token?: string): Socket => {
  if (socket?.connected) return socket;

  socket = io(SOCKET_URL, {
    auth: { token: token || localStorage.getItem('token') },
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
