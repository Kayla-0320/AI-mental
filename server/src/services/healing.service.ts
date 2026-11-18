import prisma from '../config/database';
import achievementService from './achievement.service';
import { AppError } from '../middlewares/errorHandler';

class HealingService {
  // 创建疗愈会话
  async createSession(userId: string, type: string, title: string, duration?: number) {
    if (type === 'support_agent_state') throw new AppError('该记录类型由持续陪伴服务管理', 400);
    return prisma.healingSession.create({
      data: {
        userId,
        type,
        title,
        duration: duration || 0,
      },
    });
  }

  // 完成疗愈会话
  async completeSession(sessionId: string, userId: string, data: {
    duration?: number;
    content?: string;
    moodBefore?: string;
    moodAfter?: string;
    metadata?: any;
  }) {
    const existing = await prisma.healingSession.findFirst({ where: { id: sessionId, userId } });
    if (!existing) throw new AppError('练习不存在', 404);
    let managedByAgent = existing.type === 'support_agent_state';
    try { managedByAgent ||= JSON.parse(existing.metadata || '{}').schema === 'support-action.v1'; } catch { /* 普通练习允许非 JSON 历史备注 */ }
    if (managedByAgent) throw new AppError('请通过持续陪伴记录活动回执', 409);
    const session = await prisma.healingSession.update({
      where: { id: sessionId, userId },
      data: {
        completed: true,
        duration: data.duration,
        content: data.content,
        moodBefore: data.moodBefore,
        moodAfter: data.moodAfter,
        metadata: data.metadata ? JSON.stringify(data.metadata) : undefined,
      },
    });

    // 疗愈徽章在此解锁：旧实现只在打卡时判定，疗愈类徽章永远不会亮（spec G1）
    const newlyUnlocked = await achievementService.evaluateAchievements(userId);

    return { ...session, newlyUnlocked };
  }

  // 获取疗愈历史
  async getSessions(userId: string, type?: string, page: number = 1, pageSize: number = 20) {
    if (type === 'support_agent_state') throw new AppError('设置记录不属于练习列表', 400);
    const where: any = { userId, type: { not: 'support_agent_state' } };
    if (type) where.type = type;

    const [sessions, total] = await Promise.all([
      prisma.healingSession.findMany({
        where,
        orderBy: { createdAt: 'desc' },
        skip: (page - 1) * pageSize,
        take: pageSize,
      }),
      prisma.healingSession.count({ where }),
    ]);

    return { sessions, total, page, pageSize };
  }

  // 获取疗愈统计
  async getStats(userId: string) {
    const sessions = await prisma.healingSession.findMany({
      where: { userId, completed: true, type: { not: 'support_agent_state' } },
    });

    const totalDuration = sessions.reduce((sum, s) => sum + s.duration, 0);
    const byType: Record<string, number> = {};
    sessions.forEach(s => {
      byType[s.type] = (byType[s.type] || 0) + 1;
    });

    return {
      totalSessions: sessions.length,
      totalDuration,
      byType,
      streak: await this.calculateStreak(userId),
    };
  }

  // 计算连续练习天数
  private async calculateStreak(userId: string) {
    const sessions = await prisma.healingSession.findMany({
      where: { userId, completed: true, type: { not: 'support_agent_state' } },
      orderBy: { createdAt: 'desc' },
      take: 60,
    });

    if (sessions.length === 0) return 0;

    let streak = 1;
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    for (let i = 1; i < sessions.length; i++) {
      const prev = new Date(sessions[i - 1].createdAt);
      const curr = new Date(sessions[i].createdAt);
      prev.setHours(0, 0, 0, 0);
      curr.setHours(0, 0, 0, 0);

      const diffDays = (prev.getTime() - curr.getTime()) / (1000 * 60 * 60 * 24);
      if (diffDays <= 1) {
        streak++;
      } else {
        break;
      }
    }

    return streak;
  }
}

export default new HealingService();
