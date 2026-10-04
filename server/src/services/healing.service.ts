import prisma from '../config/database';

class HealingService {
  // 创建疗愈会话
  async createSession(userId: string, type: string, title: string, duration?: number) {
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
    return prisma.healingSession.update({
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
  }

  // 获取疗愈历史
  async getSessions(userId: string, type?: string, page: number = 1, pageSize: number = 20) {
    const where: any = { userId };
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
      where: { userId, completed: true },
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
      where: { userId, completed: true },
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
