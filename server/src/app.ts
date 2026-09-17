import express from 'express';
import cors from 'cors';
import { createServer } from 'http';
import { Server } from 'socket.io';
import rateLimit from 'express-rate-limit';
import { config } from './config';
import { errorHandler } from './middlewares/errorHandler';
import { setupSocketIO } from './socket';

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

// 错误处理
app.use(errorHandler);

// 启动服务
httpServer.listen(config.port, () => {
  console.log(`服务器运行在 http://localhost:${config.port}`);
});

// 初始化 Socket.IO
setupSocketIO(io);

export default app;
