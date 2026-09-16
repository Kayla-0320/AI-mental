import prisma from '../config/database';
import { AppError } from '../middlewares/errorHandler';

export class ReviewService {
  // 创建咨询评价
  async createReview(userId: string, data: { bookingId: string; rating: number; comment?: string; tags?: string[]; isAnonymous?: boolean }) {
    const booking = await prisma.expertBooking.findUnique({
      where: { id: data.bookingId },
      include: { consultant: true },
    });
    if (!booking) throw new AppError('预约不存在', 404);
    if (booking.patientId !== userId) throw new AppError('无权评价此咨询', 403);
    if (booking.status !== 'COMPLETED') throw new AppError('只能评价已完成的咨询', 400);

    // 检查是否已评价
    const existing = await prisma.consultantReview.findUnique({ where: { bookingId: data.bookingId } });
    if (existing) throw new AppError('已评价过此咨询', 400);

    const consultantProfile = await prisma.consultantProfile.findUnique({
      where: { userId: booking.consultantId },
    });

    const review = await prisma.consultantReview.create({
      data: {
        bookingId: data.bookingId,
        patientId: userId,
        consultantId: booking.consultantId,
        consultantProfileId: consultantProfile?.id,
        rating: data.rating,
        comment: data.comment,
        tags: data.tags ? JSON.stringify(data.tags) : null,
        isAnonymous: data.isAnonymous ?? true,
      },
    });

    // 更新咨询师评分
    if (consultantProfile) {
      const allReviews = await prisma.consultantReview.findMany({
        where: { consultantId: booking.consultantId },
      });
      const avgRating = allReviews.reduce((sum: number, r: any) => sum + r.rating, 0) / allReviews.length;
      await prisma.consultantProfile.update({
        where: { id: consultantProfile.id },
        data: {
          rating: avgRating,
          totalSessions: { increment: 1 },
        },
      });
    }

    return review;
  }

  // 获取咨询师的评价
  async getConsultantReviews(consultantId: string, page: number = 1, limit: number = 10) {
    const [reviews, total] = await Promise.all([
      prisma.consultantReview.findMany({
        where: { consultantId },
        include: {
          patient: { select: { id: true, nickname: true, avatar: true } },
        },
        orderBy: { createdAt: 'desc' },
        skip: (page - 1) * limit,
        take: limit,
      }),
      prisma.consultantReview.count({ where: { consultantId } }),
    ]);
    return { reviews, total, page, limit };
  }
}
