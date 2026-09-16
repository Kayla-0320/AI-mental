import prisma from '../config/database';
import { AppError } from '../middlewares/errorHandler';

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

    // 检查成就解锁
    await this.checkAchievements(userId);

    return record;
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

  // 检查并解锁成就
  private async checkAchievements(userId: string) {
    const totalCheckIns = await prisma.moodCheckIn.count({ where: { userId } });
    const achievements = await prisma.achievement.findMany({
      where: { category: 'checkin' },
    });

    for (const achievement of achievements) {
      const condition = JSON.parse(achievement.condition);
      let unlocked = false;
      if (condition.type === 'total_checkins' && totalCheckIns >= condition.count) unlocked = true;
      if (condition.type === 'streak' && totalCheckIns >= condition.days) unlocked = true;

      if (unlocked) {
        await prisma.userAchievement.upsert({
          where: { userId_achievementId: { userId, achievementId: achievement.id } },
          update: {},
          create: { userId, achievementId: achievement.id },
        });
      }
    }
  }
}
