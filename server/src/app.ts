import express from 'express';
import cors from 'cors';
import { createServer } from 'http';
import { Server } from 'socket.io';
import rateLimit from 'express-rate-limit';
import { config } from './config';
import { errorHandler } from './middlewares/errorHandler';
import { setupSocketIO } from './socket';
import { mountAlgorithmProxy } from './services/algorithmProxy';
import { AchievementService } from './services/achievement.service';

// 路由
import authRoutes from './routes/auth.routes';
import consultationRoutes from './routes/consultation.routes';
import profileRoutes from './routes/profile.routes';
import healingRoutes from './routes/healing.routes';
import expertRoutes from './routes/expert.routes';
import treatmentRoutes from './routes/treatment.routes';
import learningRoutes from './routes/learning.routes';
import crisisRoutes from './routes/crisis.routes';
import moodRoutes from './routes/mood.routes';
import reviewRoutes from './routes/review.routes';
import communityRoutes from './routes/community.routes';
import extraRoutes from './routes/extra.routes';
import notificationRoutes from './routes/notification.routes';
import algorithmRoutes from './routes/algorithm.routes';

const app = express();
const httpServer = createServer(app);

// Socket.IO
const io = new Server(httpServer, {
  cors: {
    origin: `http://localhost:${config.clientPort}`,
    methods: ['GET', 'POST'],
  },
});

// 中间件
app.use(cors({
  origin: `http://localhost:${config.clientPort}`,
  credentials: true,
}));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// 限流
app.use('/api/', rateLimit({
  windowMs: 15 * 60 * 1000, // 15分钟
  max: 1000, // 开发阶段放宽
  message: { code: 429, message: '请求过于频繁，请稍后再试' },
}));

// API 路由
app.use('/api/auth', authRoutes);
app.use('/api/consultation', consultationRoutes);
app.use('/api/profile', profileRoutes);
app.use('/api/healing', healingRoutes);
app.use('/api/expert', expertRoutes);
app.use('/api/treatment', treatmentRoutes);
app.use('/api/learning', learningRoutes);
app.use('/api/crisis', crisisRoutes);
app.use('/api/mood', moodRoutes);
app.use('/api/reviews', reviewRoutes);
app.use('/api/community', communityRoutes);
app.use('/api/extra', extraRoutes);
app.use('/api/notifications', notificationRoutes);
app.use('/api/algorithm', algorithmRoutes);

// 健康检查
app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// ── `/algorithm` 反向代理（施工单 §S8）─────────────────────────────────
// 开发时这条前缀由 vite 代理；生产包里没有 vite，缺了它 TTS/ASR 的 WebSocket
// 会直接 404 —— 而症状是"页面正常、按钮正常、就是没声音"。
// 必须挂在 errorHandler **之前**，且同时接管 http server 的 `upgrade`。
mountAlgorithmProxy(app, httpServer);

// 错误处理
app.use(errorHandler);

// 启动服务
httpServer.listen(config.port, () => {
  console.log(`服务器运行在 http://localhost:${config.port}`);
  // 徽章表由本方法幂等补齐（prisma/seed.ts 不覆盖 achievements）
  new AchievementService().initDefaultAchievements()
    .then((n) => console.log(`默认徽章已就绪：${n} 枚`))
    .catch((err) => console.error('默认徽章初始化失败:', err));
});

// 初始化 Socket.IO
setupSocketIO(io);

export default app;
