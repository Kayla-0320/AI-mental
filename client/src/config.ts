/**
 * 全局配置 —— 算法服务（Python FastAPI）地址
 *
 * 优先读取环境变量 VITE_ALGORITHM_API_URL，便于部署时切换；
 * 本地开发默认指向 http://localhost:8001。
 */
export const ALGORITHM_API_URL: string =
  (import.meta as any).env?.VITE_ALGORITHM_API_URL || 'http://localhost:8001';

// 兼容旧命名
export const PERCEPTION_API = ALGORITHM_API_URL;
