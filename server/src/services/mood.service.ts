import prisma from '../config/database';
import { AppError } from '../middlewares/errorHandler';
import achievementService from './achievement.service';

export class MoodService {
  // 每日心情打卡
  async checkIn(userId: string, data: { mood: string; score: number; energy?: number; sleepHours?: number; note?: string; tags?: string[] }) {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);

    // 检查今天是否已打卡
    const existing = await prisma.moodCheckIn.findFirst({
      where: { userId, checkInDate: { gte: today, lt: tomorrow } },
    });
    if (existing) throw new AppError('今天已经打卡过了', 400);

    const record = await prisma.moodCheckIn.create({
      data: {
        userId,
        mood: data.mood,
        score: data.score,
        energy: data.energy,
        sleepHours: data.sleepHours,
        note: data.note,
        tags: data.tags ? JSON.stringify(data.tags) : null,
      },
    });

    // 同时创建 MoodRecord 用于趋势分析
    await prisma.moodRecord.create({
      data: {
        userId,
        mood: data.mood,
        score: data.score,
        note: data.note,
        tags: data.tags ? JSON.stringify(data.tags) : null,
      },
    });

    // 检查成就解锁（统一走徽章引擎：连续天数用真实连击，不再用累计数冒充）
    const newlyUnlocked = await achievementService.evaluateAchievements(userId);

    return { ...record, newlyUnlocked };
  }

  /**
   * 更正今天的打卡（点错表情的补救）。
   *
   * 语义仍是「每天一条」：今天已有记录就原地更新，没有就走首次打卡。
   * 这样既不会让手滑的用户被锁一整天，也不会多出第二条记录影响连续天数统计。
   */
  async updateTodayCheckIn(userId: string, data: { mood: string; score: number; energy?: number; sleepHours?: number; note?: string; tags?: string[] }) {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);

    const existing = await prisma.moodCheckIn.findFirst({
      where: { userId, checkInDate: { gte: today, lt: tomorrow } },
    });
    if (!existing) return this.checkIn(userId, data);

    const record = await prisma.moodCheckIn.update({
      where: { id: existing.id },
      data: {
        mood: data.mood,
        score: data.score,
        energy: data.energy,
        sleepHours: data.sleepHours,
        note: data.note,
        tags: data.tags ? JSON.stringify(data.tags) : undefined,
      },
    });

    // 趋势表里今天那条也一起改，避免「打卡记录改了、趋势还是旧分数」
    const todayRecord = await prisma.moodRecord.findFirst({
      where: { userId, recordedAt: { gte: today, lt: tomorrow } },
      orderBy: { recordedAt: 'desc' },
    });
    if (todayRecord) {
      await prisma.moodRecord.update({
        where: { id: todayRecord.id },
        data: { mood: data.mood, score: data.score, note: data.note },
      });
    }

    const newlyUnlocked = await achievementService.evaluateAchievements(userId);

    return { ...record, revised: true, newlyUnlocked };
  }

  // 获取打卡历史
  async getCheckInHistory(userId: string, days: number = 30) {
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);
    return prisma.moodCheckIn.findMany({
      where: { userId, checkInDate: { gte: startDate } },
      orderBy: { checkInDate: 'desc' },
    });
  }

  // 获取情绪趋势数据
  async getMoodTrend(userId: string, days: number = 30) {
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);
    const records = await prisma.moodRecord.findMany({
      where: { userId, createdAt: { gte: startDate } },
      orderBy: { createdAt: 'asc' },
    });
    return records.map((r: any) => ({
      ...r,
      tags: r.tags ? JSON.parse(r.tags) : [],
    }));
  }

  // 获取打卡日历数据
  async getCheckInCalendar(userId: string, year: number, month: number) {
    const start = new Date(year, month - 1, 1);
    const end = new Date(year, month, 0, 23, 59, 59);
    const records = await prisma.moodCheckIn.findMany({
      where: { userId, checkInDate: { gte: start, lte: end } },
    });
    return records;
  }

  // 检查并解锁成就：已由 achievement.service 统一实现
  // （旧实现在这里只查 category:'checkin'，导致疗愈/测评徽章永不触发；
  //   且把 streak 写成 totalCheckIns >= days —— 详见 docs/growth_metrics_spec.md G1/G2）
}
