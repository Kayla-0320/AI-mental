import prisma from '../config/database';
import { AppError } from '../middlewares/errorHandler';

class LearningService {
  // 获取学习内容列表
  async getContents(filters?: { category?: string; type?: string; level?: string; page?: number; pageSize?: number }) {
    const page = filters?.page || 1;
    const pageSize = filters?.pageSize || 12;

    const where: any = { isPublished: true };
    if (filters?.category) where.category = filters.category;
    if (filters?.type) where.type = filters.type;
    if (filters?.level) where.level = filters.level;

    const [contents, total] = await Promise.all([
      prisma.learningContent.findMany({
        where,
        orderBy: { viewCount: 'desc' },
        skip: (page - 1) * pageSize,
        take: pageSize,
      }),
      prisma.learningContent.count({ where }),
    ]);

    return { contents, total, page, pageSize };
  }

  // 获取内容详情
  async getContentDetail(contentId: string) {
    const content = await prisma.learningContent.findUnique({
      where: { id: contentId, isPublished: true },
    });

    if (!content) {
      throw new AppError('内容不存在', 404);
    }

    // 增加浏览量
    await prisma.learningContent.update({
      where: { id: contentId },
      data: { viewCount: { increment: 1 } },
    });

    return content;
  }

  // 获取用户学习记录
  async getUserRecords(userId: string) {
    return prisma.learningRecord.findMany({
      where: { userId },
      orderBy: { updatedAt: 'desc' },
    });
  }

  // 开始/更新学习进度
  async updateProgress(userId: string, contentId: string, progress: number, notes?: string) {
    const content = await prisma.learningContent.findUnique({
      where: { id: contentId },
    });

    if (!content) {
      throw new AppError('内容不存在', 404);
    }

    const existing = await prisma.learningRecord.findFirst({
      where: { userId, contentId },
    });

    if (existing) {
      return prisma.learningRecord.update({
        where: { id: existing.id },
        data: {
          progress,
          completed: progress >= 100,
          lastStudiedAt: new Date(),
          notes: notes || existing.notes,
        },
      });
    }

    return prisma.learningRecord.create({
      data: {
        userId,
        contentId,
        contentType: content.type,
        title: content.title,
        progress,
        completed: progress >= 100,
        lastStudiedAt: new Date(),
        notes,
      },
    });
  }

  // 获取推荐内容
  async getRecommendations(userId: string, limit: number = 6) {
    // 获取用户最近的学习记录和心情
    const recentRecords = await prisma.learningRecord.findMany({
      where: { userId },
      orderBy: { updatedAt: 'desc' },
      take: 5,
    });

    const completedCategories = new Set(recentRecords.map(r => r.contentType));

    // 基于学习历史推荐不同类别的内容
    const contents = await prisma.learningContent.findMany({
      where: {
        isPublished: true,
        id: { notIn: recentRecords.map(r => r.contentId) },
      },
      orderBy: { viewCount: 'desc' },
      take: limit,
    });

    return contents;
  }

  // 获取学习统计
  async getStats(userId: string) {
    const records = await prisma.learningRecord.findMany({
      where: { userId },
    });

    const totalCompleted = records.filter(r => r.completed).length;
    const totalInProgress = records.filter(r => !r.completed && r.progress > 0).length;
    const avgProgress = records.length > 0
      ? Math.round(records.reduce((sum, r) => sum + r.progress, 0) / records.length)
      : 0;

    return {
      totalLearned: records.length,
      totalCompleted,
      totalInProgress,
      avgProgress,
      recentRecords: records.slice(0, 5),
    };
  }
}

export default new LearningService();
