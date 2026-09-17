import prisma from '../config/database';
import { AppError } from '../middlewares/errorHandler';

class ExpertService {
  // 获取咨询师列表
  async getConsultants(filters?: { specialty?: string; minRating?: number; page?: number; pageSize?: number }) {
    const page = filters?.page || 1;
    const pageSize = filters?.pageSize || 10;

    const where: any = { isAvailable: true, user: { isActive: true } };
    if (filters?.specialty) {
      where.specialties = { contains: filters.specialty };
    }
    if (filters?.minRating) {
      where.rating = { gte: filters.minRating };
    }

    const [consultants, total] = await Promise.all([
      prisma.consultantProfile.findMany({
        where,
        include: {
          user: {
            select: { id: true, nickname: true, avatar: true },
          },
        },
        orderBy: { rating: 'desc' },
        skip: (page - 1) * pageSize,
        take: pageSize,
      }),
      prisma.consultantProfile.count({ where }),
    ]);

    return { consultants, total, page, pageSize };
  }

  // 获取咨询师详情
  async getConsultantDetail(consultantId: string) {
    const consultant = await prisma.consultantProfile.findUnique({
      where: { id: consultantId },
      include: {
        user: {
          select: { id: true, nickname: true, avatar: true },
        },
      },
    });

    if (!consultant) {
      throw new AppError('咨询师不存在', 404);
    }

    return consultant;
  }

  // 创建预约
  async createBooking(patientId: string, consultantId: string, scheduledAt: Date, notes?: string, type?: string) {
    // consultantId 可能是 User ID 或 ConsultantProfile ID，需要兼容两种情况
    let consultant = await prisma.consultantProfile.findUnique({
      where: { id: consultantId },
    });

    // 如果没找到，尝试通过 userId 查找
    if (!consultant) {
      consultant = await prisma.consultantProfile.findUnique({
        where: { userId: consultantId },
      });
    }

    if (!consultant || !consultant.isAvailable) {
      throw new AppError('咨询师不可用', 400);
    }

    // 使用 User ID 作为外键
    const bookingConsultantId = consultant.userId;

    // 检查时间冲突
    const existingBooking = await prisma.expertBooking.findFirst({
      where: {
        consultantId: bookingConsultantId,
        scheduledAt,
        status: { in: ['PENDING', 'CONFIRMED'] },
      },
    });

    if (existingBooking) {
      throw new AppError('该时间段已被预约', 400);
    }

    return prisma.expertBooking.create({
      data: {
        patientId,
        consultantId: bookingConsultantId,
        scheduledAt,
        notes,
        type: type || 'TEXT',
        status: 'PENDING',
      },
      include: {
        patient: { select: { id: true, nickname: true, avatar: true } },
        consultant: { select: { id: true, nickname: true, avatar: true } },
      },
    });
  }

  // 获取用户预约列表
  async getBookings(userId: string, role: string) {
    const where = role === 'CONSULTANT'
      ? { consultantId: userId }
      : { patientId: userId };

    return prisma.expertBooking.findMany({
      where,
      include: {
        patient: { select: { id: true, nickname: true, avatar: true } },
        consultant: { select: { id: true, nickname: true, avatar: true } },
      },
      orderBy: { scheduledAt: 'desc' },
    });
  }

  async getBookingById(bookingId: string) {
    const booking = await prisma.expertBooking.findUnique({
      where: { id: bookingId },
      include: {
        patient: { select: { id: true, nickname: true, email: true, avatar: true } },
        consultant: { select: { id: true, nickname: true, email: true, avatar: true } },
      },
    });
    if (!booking) throw new AppError('预约不存在', 404);
    return booking;
  }

  // 更新预约状态
  async updateBookingStatus(bookingId: string, status: string, userId: string) {
    const booking = await prisma.expertBooking.findUnique({
      where: { id: bookingId },
    });

    if (!booking) {
      throw new AppError('预约不存在', 404);
    }

    const updateData: any = { status };
    if (status === 'IN_PROGRESS') updateData.startedAt = new Date();
    if (status === 'COMPLETED' || status === 'CANCELLED') updateData.endedAt = new Date();

    return prisma.expertBooking.update({
      where: { id: bookingId },
      data: updateData,
    });
  }

  // 提交评价
  async submitFeedback(bookingId: string, userId: string, rating: number, feedback?: string) {
    const booking = await prisma.expertBooking.findFirst({
      where: { id: bookingId, patientId: userId, status: 'COMPLETED' },
    });

    if (!booking) {
      throw new AppError('预约不存在或未完成', 404);
    }

    await prisma.expertBooking.update({
      where: { id: bookingId },
      data: { rating, feedback },
    });

    // 更新咨询师评分
    const allRatings = await prisma.expertBooking.findMany({
      where: { consultantId: booking.consultantId, rating: { not: null } },
      select: { rating: true },
    });

    const avgRating = allRatings.reduce((sum, b) => sum + (b.rating || 0), 0) / allRatings.length;

    await prisma.consultantProfile.update({
      where: { id: booking.consultantId },
      data: {
        rating: avgRating,
        totalSessions: { increment: 1 },
      },
    });

    return { rating, feedback };
  }

  // 发送专家咨询消息
  async sendMessage(bookingId: string, senderId: string, content: string, contentType: string = 'text') {
    return prisma.expertMessage.create({
      data: {
        bookingId,
        senderId,
        content,
        contentType,
      },
    });
  }

  // 获取咨询师自己的档案
  async getMyProfile(userId: string) {
    const profile = await prisma.consultantProfile.findUnique({
      where: { userId },
      include: { user: { select: { id: true, nickname: true, avatar: true, email: true } } },
    });
    if (!profile) throw new AppError('咨询师档案不存在', 404);
    return profile;
  }

  // 更新咨询师自己的档案
  async updateMyProfile(userId: string, data: { title?: string; specialties?: string; introduction?: string; pricePerSession?: number }) {
    const profile = await prisma.consultantProfile.findUnique({ where: { userId } });
    if (!profile) throw new AppError('咨询师档案不存在', 404);

    return prisma.consultantProfile.update({
      where: { userId },
      data: {
        ...(data.title !== undefined && { title: data.title }),
        ...(data.specialties !== undefined && { specialties: data.specialties }),
        ...(data.introduction !== undefined && { introduction: data.introduction }),
        ...(data.pricePerSession !== undefined && { pricePerSession: data.pricePerSession }),
      },
    });
  }

  // 获取专家咨询消息
  async getMessages(bookingId: string, userId: string, page: number = 1, pageSize: number = 50) {
    const booking = await prisma.expertBooking.findFirst({
      where: {
        id: bookingId,
        OR: [{ patientId: userId }, { consultantId: userId }],
      },
    });

    if (!booking) {
      throw new AppError('无权访问该咨询', 403);
    }

    const messages = await prisma.expertMessage.findMany({
      where: { bookingId },
      orderBy: { createdAt: 'asc' },
      skip: (page - 1) * pageSize,
      take: pageSize,
      include: {
        sender: { select: { id: true, nickname: true, avatar: true } },
      },
    });

    // 标记已读
    await prisma.expertMessage.updateMany({
      where: { bookingId, senderId: { not: userId }, isRead: false },
      data: { isRead: true },
    });

    return messages;
  }
}

export default new ExpertService();
