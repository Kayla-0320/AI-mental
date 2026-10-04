import { Router, Response, NextFunction } from 'express';
import { AchievementService } from '../services/achievement.service';
import { SleepService, PaymentService, HealthReportService, FeedbackService } from '../services/extra.service';
import { authenticate, AuthRequest } from '../middlewares/auth';
import prisma from '../config/database';

const router = Router();
const achievementService = new AchievementService();
const sleepService = new SleepService();
const paymentService = new PaymentService();
const healthReportService = new HealthReportService();
const feedbackService = new FeedbackService();

// ===== 成就系统 =====
router.get('/achievements', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const progress = await achievementService.getAchievementProgress(req.userId!);
    res.json({ code: 0, data: progress });
  } catch (e) { next(e); }
});

router.get('/achievements/all', async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const achievements = await achievementService.getAllAchievements();
    res.json({ code: 0, data: achievements });
  } catch (e) { next(e); }
});

// ===== 睡眠追踪 =====
router.post('/sleep', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const record = await sleepService.record(req.userId!, req.body);
    res.json({ code: 0, data: record });
  } catch (e) { next(e); }
});

router.get('/sleep', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const days = parseInt(req.query.days as string) || 30;
    const records = await sleepService.getRecords(req.userId!, days);
    res.json({ code: 0, data: records });
  } catch (e) { next(e); }
});

router.get('/sleep/stats', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const days = parseInt(req.query.days as string) || 30;
    const stats = await sleepService.getStats(req.userId!, days);
    res.json({ code: 0, data: stats });
  } catch (e) { next(e); }
});

// ===== 支付 =====
router.post('/payment', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const payment = await paymentService.createPayment(req.userId!, req.body);
    res.json({ code: 0, data: payment });
  } catch (e) { next(e); }
});

router.get('/payments', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const payments = await paymentService.getUserPayments(req.userId!);
    res.json({ code: 0, data: payments });
  } catch (e) { next(e); }
});

// ===== 健康报告 =====
router.post('/reports/weekly', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const report = await healthReportService.generateWeeklyReport(req.userId!);
    res.json({ code: 0, data: report });
  } catch (e) { next(e); }
});

router.get('/reports', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const type = req.query.type as string;
    const reports = await healthReportService.getUserReports(req.userId!, type);
    res.json({ code: 0, data: reports });
  } catch (e) { next(e); }
});

// ===== 用户反馈 =====
router.post('/feedback', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const feedback = await feedbackService.create(req.userId!, req.body);
    res.json({ code: 0, data: feedback });
  } catch (e) { next(e); }
});

router.get('/feedback', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const feedbacks = await feedbackService.getUserFeedbacks(req.userId!);
    res.json({ code: 0, data: feedbacks });
  } catch (e) { next(e); }
});

// 管理员：获取所有反馈
router.get('/feedback/all', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    if (req.userRole! !== 'ADMIN') return res.status(403).json({ code: 403, message: '无权限' });
    const status = req.query.status as string;
    const feedbacks = await feedbackService.getAllFeedbacks(status);
    res.json({ code: 0, data: feedbacks });
  } catch (e) { next(e); }
});

// 管理员：回复反馈
router.put('/feedback/:id/reply', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    if (req.userRole! !== 'ADMIN') return res.status(403).json({ code: 403, message: '无权限' });
    const feedback = await feedbackService.replyFeedback(req.params.id as string, req.body.reply);
    res.json({ code: 0, data: feedback });
  } catch (e) { next(e); }
});

// ===== 管理员：仪表盘统计数据 =====
router.get('/admin/stats', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    if (req.userRole! !== 'ADMIN') return res.status(403).json({ code: 403, message: '无权限' });

    const [userStats, conversationStats, assessmentStats, consultantStats, moodStats, recentActivities] = await Promise.all([
      // 用户统计
      prisma.$queryRawUnsafe(`
        SELECT
          COUNT(*)::int as total_users,
          COUNT(*) FILTER (WHERE "lastLoginAt" >= CURRENT_DATE)::int as active_users,
          COUNT(*) FILTER (WHERE "createdAt" >= CURRENT_DATE)::int as new_today
        FROM users WHERE role = 'PATIENT'
      `),
      // 对话统计
      prisma.$queryRawUnsafe(`SELECT COUNT(*)::int as total FROM conversations`),
      // 测评统计
      prisma.$queryRawUnsafe(`SELECT COUNT(*)::int as total FROM assessments`),
      // 咨询师统计
      prisma.$queryRawUnsafe(`
        SELECT COUNT(*)::int as total, COUNT(*) FILTER (WHERE "isAvailable" = true)::int as online
        FROM consultant_profiles
      `),
      // 心情打卡统计
      prisma.$queryRawUnsafe(`SELECT COUNT(*)::int as total FROM mood_checkins WHERE "checkInDate" >= CURRENT_DATE`),
      // 近期动态
      prisma.$queryRawUnsafe(`
        SELECT 'user_register' as type, u.nickname as name, u."createdAt" as time FROM users u
        WHERE u."createdAt" >= NOW() - INTERVAL '24 hours'
        ORDER BY u."createdAt" DESC LIMIT 5
      `),
    ]);

    const users = (userStats as any)[0];
    const consultants = (consultantStats as any)[0];

    res.json({
      code: 0,
      data: {
        totalUsers: users.total_users || 0,
        activeUsers: users.active_users || 0,
        newToday: users.new_today || 0,
        totalConversations: (conversationStats as any)[0].total || 0,
        totalAssessments: (assessmentStats as any)[0].total || 0,
        totalConsultants: consultants.total || 0,
        onlineConsultants: consultants.online || 0,
        todayMoodCheckIns: (moodStats as any)[0].total || 0,
        recentActivities: recentActivities || [],
      },
    });
  } catch (e) { next(e); }
});

// ===== 管理员：获取用户列表 =====
router.get('/admin/users', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    if (req.userRole! !== 'ADMIN') return res.status(403).json({ code: 403, message: '无权限' });
    const VALID_ROLES = ['ADMIN', 'CONSULTANT', 'PATIENT'];
    const search = req.query.search as string;
    const role = req.query.role as string;
    const page = parseInt(req.query.page as string) || 1;
    const pageSize = parseInt(req.query.pageSize as string) || 20;

    const conditions: string[] = [];
    if (search) {
      conditions.push(`(u.nickname ILIKE '%${search.replace(/'/g, "''")}%' OR u.email ILIKE '%${search.replace(/'/g, "''")}%')`);
    }
    if (role) {
      if (!VALID_ROLES.includes(role.toUpperCase())) {
        return res.json({ code: 0, data: { users: [], total: 0, page, pageSize } });
      }
      conditions.push(`u.role = '${role.toUpperCase().replace(/'/g, "''")}'`);
    }
    const whereClause = conditions.length > 0 ? `WHERE ${conditions.join(' AND ')}` : '';

    const [users, total] = await Promise.all([
      prisma.$queryRawUnsafe(`
        SELECT u.id, u.nickname, u.email, u.role, u."isActive" as "isActive", u."createdAt"
        FROM users u
        ${whereClause}
        ORDER BY u."createdAt" DESC
        LIMIT ${pageSize} OFFSET ${(page - 1) * pageSize}
      `),
      prisma.$queryRawUnsafe(`SELECT COUNT(*)::int as count FROM users u ${whereClause}`),
    ]);

    res.json({ code: 0, data: { users, total: (total as any)[0].count, page, pageSize } });
  } catch (e) { next(e); }
});

export default router;
