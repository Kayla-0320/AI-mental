import { Router, Response, NextFunction } from 'express';
import { MoodService } from '../services/mood.service';
import { authenticate, AuthRequest } from '../middlewares/auth';

const router = Router();
const moodService = new MoodService();

// 每日心情打卡
router.post('/checkin', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const record = await moodService.checkIn(req.userId!, req.body);
    res.json({ code: 0, data: record });
  } catch (e) { next(e); }
});

// 获取打卡历史
router.get('/checkin/history', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const days = parseInt(req.query.days as string) || 30;
    const records = await moodService.getCheckInHistory(req.userId!, days);
    res.json({ code: 0, data: records });
  } catch (e) { next(e); }
});

// 获取情绪趋势
router.get('/trend', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const days = parseInt(req.query.days as string) || 30;
    const records = await moodService.getMoodTrend(req.userId!, days);
    res.json({ code: 0, data: records });
  } catch (e) { next(e); }
});

// 获取打卡日历
router.get('/checkin/calendar', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const year = parseInt(req.query.year as string) || new Date().getFullYear();
    const month = parseInt(req.query.month as string) || new Date().getMonth() + 1;
    const records = await moodService.getCheckInCalendar(req.userId!, year, month);
    res.json({ code: 0, data: records });
  } catch (e) { next(e); }
});

// 检查今日是否已打卡
router.get('/checkin/today', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);
    const record = await (await import('../config/database')).default.moodCheckIn.findFirst({
      where: { userId: req.userId!, checkInDate: { gte: today, lt: tomorrow } },
    });
    res.json({ code: 0, data: { checkedIn: !!record, record } });
  } catch (e) { next(e); }
});

export default router;
