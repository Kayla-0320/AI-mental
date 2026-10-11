import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 90000,
  headers: { 'Content-Type': 'application/json' },
});

// 请求拦截器 - 自动添加 Token
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('accessToken');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// 响应拦截器 - 处理 Token 过期
api.interceptors.response.use(
  (response) => response.data,
  async (error) => {
    if (error.response?.status === 401) {
      const refreshToken = localStorage.getItem('refreshToken');
      if (refreshToken) {
        try {
          const res = await axios.post('/api/auth/refresh-token', { refreshToken });
          const { accessToken, refreshToken: newRefreshToken } = res.data.data;
          localStorage.setItem('accessToken', accessToken);
          localStorage.setItem('refreshToken', newRefreshToken);
          // 重试原请求
          error.config.headers.Authorization = `Bearer ${accessToken}`;
          return api(error.config);
        } catch {
          localStorage.removeItem('accessToken');
          localStorage.removeItem('refreshToken');
          window.location.href = '/login';
        }
      } else {
        window.location.href = '/login';
      }
    }
    // 拒绝时把 HTTP 状态码**附在响应体上一起抛出**。
    //
    // 原来只透传 `error.response.data`，调用方拿到 `{code, message}` 后无法区分
    // "404 会话不存在" 与 "网络断了/500"（两种情况都只是 catch 到一个对象）。
    // 于是只能拿中文文案做判断 —— 文案一改判断就失效（2026-10-11：通话页要按
    // 404 决定"自动换一个新会话"，不能再靠猜）。
    // `code` 只当业务码用，**不要**拿它代替 HTTP 状态。
    const payload = error.response?.data;
    if (payload && typeof payload === 'object') {
      return Promise.reject(Object.assign(payload, { status: error.response?.status }));
    }
    return Promise.reject(payload || error);
  }
);

export default api;
