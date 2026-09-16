import prisma from '../config/database';
import { AppError } from '../middlewares/errorHandler';

export class AchievementService {
  // 获取所有成就
  async getAllAchievements() {
    return prisma.achievement.findMany({ orderBy: { category: 'asc' } });
  }

  // 获取用户已解锁成就
  async getUserAchievements(userId: string) {
    return prisma.userAchievement.findMany({
      where: { userId },
      include: { achievement: true },
      orderBy: { unlockedAt: 'desc' },
    });
  }

  // 获取用户成就进度
  async getAchievementProgress(userId: string) {
    const all = await prisma.achievement.count();
    const unlocked = await prisma.userAchievement.count({ where: { userId } });
    const userAchievements = await this.getUserAchievements(userId);

    // 统计各项数据
    const [totalCheckIns, totalHealingSessions, totalAssessments] = await Promise.all([
      prisma.moodCheckIn.count({ where: { userId } }),
      prisma.healingSession.count({ where: { userId, completed: true } }),
      prisma.assessment.count({ where: { userId } }),
    ]);

    return {
      total: all,
      unlocked,
      progress: Math.round((unlocked / all) * 100),
      stats: { totalCheckIns, totalHealingSessions, totalAssessments },
      achievements: userAchievements,
    };
  }

  // 初始化默认成就
  async initDefaultAchievements() {
    const defaults = [
      { name: '初次打卡', description: '完成第一次心情打卡', icon: '🌱', category: 'checkin', condition: JSON.stringify({ type: 'total_checkins', count: 1 }) },
      { name: '坚持一周', description: '连续打卡7天', icon: '🔥', category: 'checkin', condition: JSON.stringify({ type: 'streak', days: 7 }) },
      { name: '月度达人', description: '连续打卡30天', icon: '', category: 'checkin', condition: JSON.stringify({ type: 'streak', days: 30 }) },
      { name: '疗愈新手', description: '完成第一次疗愈练习', icon: '🧘', category: 'healing', condition: JSON.stringify({ type: 'total_sessions', count: 1 }) },
      { name: '疗愈大师', description: '完成10次疗愈练习', icon: '🌟', category: 'healing', condition: JSON.stringify({ type: 'total_sessions', count: 10 }) },
      { name: '测评先锋', description: '完成第一次心理测评', icon: '📋', category: 'assessment', condition: JSON.stringify({ type: 'total_assessments', count: 1 }) },
      { name: '自我探索者', description: '完成5次心理测评', icon: '', category: 'assessment', condition: JSON.stringify({ type: 'total_assessments', count: 5 }) },
    ];

    for (const item of defaults) {
      await prisma.achievement.upsert({
        where: { id: item.name }, // Use name as unique identifier for upsert
        update: item,
        create: { id: `ach_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`, ...item },
      }).catch(() => {
        // Ignore duplicate errors
      });
    }
  }
}
