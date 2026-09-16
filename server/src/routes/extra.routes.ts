import { Router, Response, NextFunction } from 'express';
import { AchievementService } from '../services/achievement.service';
import { SleepService, PaymentService, HealthReportService, FeedbackService } from '../services/extra.service';
import { authenticate, AuthRequest } from '../middlewares/auth';

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
    const feedback = await feedbackService.replyFeedback(req.params.id, req.body.reply);
    res.json({ code: 0, data: feedback });
  } catch (e) { next(e); }
});

export default router;
