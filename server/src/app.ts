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
import crisisService from './services/crisis.service';
import privacyService from './services/privacy.service';

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
import adminRoutes from './routes/admin.routes';
import privacyRoutes from './routes/privacy.routes';

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
// 管理端专用（仅 ADMIN）：/api/admin/consultants 及其审核动作
app.use('/api/admin', adminRoutes);
// 数据控制中心：授权 / 导出 / 删除 / 保留期限（登录用户对自己生效）
app.use('/api/privacy', privacyRoutes);

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

/**
 * 危机超时兜底看门狗。
 *
 * 「检测到了」和「有人接住了」之间必须有东西负责催办 —— 否则一条 OPEN 的危机
 * 记录可以永远躺着，管理端列表看起来「已记录」，实际上没有任何人知道。
 *
 * 每 5 分钟扫一次：超过 15 分钟仍为 OPEN 的记录，再次提醒管理员（幂等，
 * 靠 metadata.overdueNotifiedAt 去重，重复执行不会重复打扰）。
 */
const CRISIS_WATCHDOG_INTERVAL_MS = 5 * 60 * 1000;
function startCrisisWatchdog() {
  const tick = async () => {
    try {
      const escalated = await crisisService.escalateOverdue();
      if (escalated > 0) {
        console.warn(`[CrisisWatchdog] ${escalated} 条危机记录超时未接收，已再次提醒管理员`);
      }
    } catch (err) {
      console.error('[CrisisWatchdog] 执行失败:', err);
    }
  };
  const timer = setInterval(tick, CRISIS_WATCHDOG_INTERVAL_MS);
  // 不阻塞进程退出
  if (typeof timer.unref === 'function') timer.unref();
  // 启动 30 秒后先跑一次，覆盖「服务刚重启、历史记录已超时」的情况
  const kick = setTimeout(tick, 30 * 1000);
  if (typeof kick.unref === 'function') kick.unref();
}

// 启动服务
httpServer.listen(config.port, () => {
  console.log(`服务器运行在 http://localhost:${config.port}`);
  // 徽章表由本方法幂等补齐（prisma/seed.ts 不覆盖 achievements）
  new AchievementService().initDefaultAchievements()
    .then((n) => console.log(`默认徽章已就绪：${n} 枚`))
    .catch((err) => console.error('默认徽章初始化失败:', err));
  startCrisisWatchdog();
  startRetentionPurge();
});

/**
 * 保留期限清理。
 *
 * 前端「隐私与数据控制」里写着「超过此期限的数据将自动删除」。在本次改动之前，
 * 那句话背后没有任何执行体——用户设置了 30 天，数据依然永久留存。
 *
 * 每天跑一次，按每个用户的 `dataRetentionDays` 硬删除过期记录
 * （范围见 privacy.service.ts 的 purgeExpired：不含量表、危机记录与授权记录）。
 */
const RETENTION_PURGE_INTERVAL_MS = 24 * 60 * 60 * 1000;
function startRetentionPurge() {
  const tick = async () => {
    try {
      const result = await privacyService.purgeExpired();
      if (result.usersTouched > 0) {
        console.log(`[Retention] 已按用户保留期限清理 ${result.usersTouched} 个账号的过期数据`, result.deleted);
      }
    } catch (err) {
      console.error('[Retention] 清理失败:', err);
    }
  };
  const timer = setInterval(tick, RETENTION_PURGE_INTERVAL_MS);
  if (typeof timer.unref === 'function') timer.unref();
  // 启动 90 秒后先跑一次，验证链路可用（避免只有重启一天后才生效）
  const kick = setTimeout(tick, 90 * 1000);
  if (typeof kick.unref === 'function') kick.unref();
}

// 初始化 Socket.IO
setupSocketIO(io);

export default app;
