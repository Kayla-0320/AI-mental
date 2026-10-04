import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:3000',
        changeOrigin: true,
      },
      '/algorithm': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        // 流式语音识别走 WebSocket：/algorithm/api/v1/asr/ws
        // 少了 ws:true，握手会被 vite 当成普通 HTTP 处理而失败
        ws: true,
        rewrite: (path) => path.replace(/^\/algorithm/, ''),
      },
      '/socket.io': {
        target: 'http://localhost:3000',
        ws: true,
      },
    },
  },
  resolve: {
    alias: {
      '@': '/src',
    },
  },
});
