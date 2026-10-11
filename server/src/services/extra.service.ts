import prisma from '../config/database';
import { AppError } from '../middlewares/errorHandler';

export class SleepService {
  /**
   * 记录睡眠。
   *
   * 修掉的历史问题（`sleep_records` 全表 0 行就是这么来的）：
   *  - 首页睡眠步骤传的是 `bedTime: ''` / `wakeTime: ''`，`new Date('')` 是 Invalid Date，
   *    Prisma 直接抛错 → 接口 500 → 前端 `.catch` 静默吞掉，用户点完没有任何反应；
   *  - 首页还按 0-100 传 `quality`（80/60/40），而字段与消费方（`×10` 换算）都按 1-10，
   *    真写进去会得到「睡眠质量 800 分」这种脏数据。
   * 现在：允许只给 `hours`（「昨晚睡了 8 小时」），并把 quality 一律收敛到 1-10。
   */
  async record(userId: string, data: {
    bedTime?: string;
    wakeTime?: string;
    /** 只报时长时的简写入口（首页睡眠步骤用） */
    hours?: number;
    quality: number;
    dreamRecall?: string;
    notes?: string;
  }) {
    const bed = data.bedTime ? new Date(data.bedTime) : null;
    const wake = data.wakeTime ? new Date(data.wakeTime) : null;
    const bedValid = !!bed && !Number.isNaN(bed.getTime());
    const wakeValid = !!wake && !Number.isNaN(wake.getTime());

    let durationHours: number;
    if (bedValid && wakeValid) {
      durationHours = (wake!.getTime() - bed!.getTime()) / (1000 * 60 * 60);
      if (durationHours <= 0) throw new AppError('起床时间需要晚于入睡时间', 400);
    } else if (typeof data.hours === 'number' && data.hours > 0 && data.hours <= 24) {
      durationHours = data.hours;
    } else {
      throw new AppError('请提供有效的入睡/起床时间，或直接给 hours（睡眠小时数）', 400);
    }

    // 质量统一按 1-10 存；>10 视为旧调用方传来的 0-100 分制，收敛回来
    const rawQuality = Number(data.quality);
    const quality = Number.isFinite(rawQuality)
      ? Math.max(1, Math.min(10, Math.round(rawQuality > 10 ? rawQuality / 10 : rawQuality)))
      : 5;

    const wakeTime = wakeValid ? wake! : new Date();
    const bedTime = bedValid ? bed! : new Date(wakeTime.getTime() - durationHours * 60 * 60 * 1000);

    return prisma.sleepRecord.create({
      data: {
        userId,
        bedTime,
        wakeTime,
        duration: Math.round(durationHours * 10) / 10,
        quality,
        dreamRecall: data.dreamRecall,
        notes: data.notes,
        recordDate: bedTime,
      },
    });
  }

  async getRecords(userId: string, days: number = 30) {
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);
    return prisma.sleepRecord.findMany({
      where: { userId, recordDate: { gte: startDate } },
      orderBy: { recordDate: 'desc' },
    });
  }

  async getStats(userId: string, days: number = 30) {
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);
    const records = await prisma.sleepRecord.findMany({
      where: { userId, recordDate: { gte: startDate } },
    });
    if (records.length === 0) return { avgDuration: 0, avgQuality: 0, total: 0 };
    const avgDuration = records.reduce((s: number, r: any) => s + r.duration, 0) / records.length;
    const avgQuality = records.reduce((s: number, r: any) => s + r.quality, 0) / records.length;
    return { avgDuration: Math.round(avgDuration * 10) / 10, avgQuality: Math.round(avgQuality), total: records.length };
  }
}

export class PaymentService {
  async createPayment(userId: string, data: { bookingId: string; amount: number; method: string; description?: string }) {
    const payment = await prisma.payment.create({
      data: {
        userId,
        bookingId: data.bookingId,
        amount: data.amount,
        method: data.method,
        description: data.description,
        status: 'SUCCESS', // 模拟支付成功
        paidAt: new Date(),
        transactionId: `TXN_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      },
    });

    // 更新预约状态
    if (data.bookingId) {
      await prisma.expertBooking.update({
        where: { id: data.bookingId },
        data: { isPaid: true, paidAmount: data.amount, paymentId: payment.id },
      });
    }

    return payment;
  }

  async getUserPayments(userId: string) {
    return prisma.payment.findMany({
      where: { userId },
      include: { booking: { include: { consultant: { select: { id: true, nickname: true } } } } },
      orderBy: { createdAt: 'desc' },
    });
  }
}

export class HealthReportService {
  async generateWeeklyReport(userId: string) {
    const end = new Date();
    const start = new Date();
    start.setDate(start.getDate() - 7);

    const [checkIns, assessments, healingSessions, sleepRecords] = await Promise.all([
      prisma.moodCheckIn.findMany({ where: { userId, checkInDate: { gte: start, lte: end } } }),
      prisma.assessment.findMany({ where: { userId, completedAt: { gte: start, lte: end } } }),
      prisma.healingSession.findMany({ where: { userId, createdAt: { gte: start, lte: end } } }),
      prisma.sleepRecord.findMany({ where: { userId, recordDate: { gte: start, lte: end } } }),
    ]);

    const avgMood = checkIns.length > 0 ? Math.round(checkIns.reduce((s: number, c: any) => s + c.score, 0) / checkIns.length) : 0;
    const avgSleep = sleepRecords.length > 0 ? Math.round(sleepRecords.reduce((s: number, r: any) => s + r.duration, 0) / sleepRecords.length * 10) / 10 : 0;

    const content = JSON.stringify({
      period: `${start.toLocaleDateString()} - ${end.toLocaleDateString()}`,
      moodTrend: checkIns.map((c: any) => ({ date: c.checkInDate, mood: c.mood, score: c.score })),
      avgMood,
      totalCheckIns: checkIns.length,
      totalAssessments: assessments.length,
      totalHealingSessions: healingSessions.length,
      avgSleepDuration: avgSleep,
      suggestions: this.generateSuggestions(avgMood, avgSleep, checkIns.length),
    });

    const report = await prisma.healthReport.create({
      data: {
        userId,
        type: 'WEEKLY',
        title: `周报 ${start.toLocaleDateString()} ~ ${end.toLocaleDateString()}`,
        content,
        periodStart: start,
        periodEnd: end,
        summary: `本周平均心情 ${avgMood}/10，平均睡眠 ${avgSleep} 小时，打卡 ${checkIns.length} 天`,
      },
    });

    // 创建通知
    await prisma.notification.create({
      data: { userId, type: 'system', title: '周报已生成', content: report.summary || '查看你的本周心理健康报告', link: '/profile' },
    });

    return report;
  }

  private generateSuggestions(avgMood: number, avgSleep: number, checkInDays: number): string[] {
    const suggestions: string[] = [];
    if (avgMood < 5) suggestions.push('心情偏低，建议增加户外活动和社交互动');
    if (avgSleep < 7) suggestions.push('睡眠不足，建议保持规律作息');
    if (checkInDays < 5) suggestions.push('打卡次数较少，坚持每日记录有助于自我觉察');
    if (suggestions.length === 0) suggestions.push('状态良好，继续保持！');
    return suggestions;
  }

  async getUserReports(userId: string, type?: string) {
    const where: any = { userId };
    if (type) where.type = type;
    return prisma.healthReport.findMany({ where, orderBy: { createdAt: 'desc' } });
  }
}

export class FeedbackService {
  async create(userId: string, data: { type: string; title: string; content: string; contact?: string }) {
    return prisma.userFeedback.create({ data: { userId, ...data } });
  }

  async getUserFeedbacks(userId: string) {
    return prisma.userFeedback.findMany({ where: { userId }, orderBy: { createdAt: 'desc' } });
  }

  async getAllFeedbacks(status?: string) {
    const where: any = {};
    if (status) where.status = status;
    return prisma.userFeedback.findMany({
      where,
      include: { user: { select: { id: true, nickname: true, email: true } } },
      orderBy: { createdAt: 'desc' },
    });
  }

  async replyFeedback(feedbackId: string, reply: string) {
    return prisma.userFeedback.update({
      where: { id: feedbackId },
      data: { adminReply: reply, status: 'RESOLVED' },
    });
  }
}
