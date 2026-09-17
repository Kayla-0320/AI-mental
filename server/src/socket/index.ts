import { Server, Socket } from 'socket.io';
import jwt from 'jsonwebtoken';
import { config } from '../config';
import prisma from '../config/database';

interface AuthSocket extends Socket {
  userId?: string;
}

export const setupSocketIO = (io: Server) => {
  // 认证中间件
  io.use((socket: AuthSocket, next) => {
    const token = socket.handshake.auth.token;
    if (!token) {
      return next(new Error('认证失败：未提供令牌'));
    }

    try {
      const decoded = jwt.verify(token, config.jwt.secret) as { userId: string };
      socket.userId = decoded.userId;
      next();
    } catch {
      next(new Error('认证失败：令牌无效'));
    }
  });

  io.on('connection', (socket: AuthSocket) => {
    const userId = socket.userId!;
    console.log(`用户连接: ${userId}`);

    // 加入用户专属房间
    socket.join(`user:${userId}`);

    // 专家咨询消息
    socket.on('expert:join', async (bookingId: string) => {
      // 验证用户是否是此预约的参与者
      try {
        const booking = await prisma.expertBooking.findUnique({
          where: { id: bookingId },
          select: { patientId: true, consultantId: true, status: true },
        });

        if (!booking) {
          socket.emit('error', { message: '预约不存在' });
          return;
        }

        // 只有患者或咨询师才能加入
        if (booking.patientId !== userId && booking.consultantId !== userId) {
          socket.emit('error', { message: '无权访问此咨询' });
          return;
        }

        // 检查预约状态（只有进行中的咨询才能加入）
        if (booking.status !== 'CONFIRMED' && booking.status !== 'IN_PROGRESS') {
          socket.emit('error', { message: '咨询未开始或已结束' });
          return;
        }

        socket.join(`booking:${bookingId}`);
        console.log(`用户 ${userId} 加入咨询房间 ${bookingId}`);
      } catch (error) {
        socket.emit('error', { message: '加入房间失败' });
      }
    });

    socket.on('expert:leave', (bookingId: string) => {
      socket.leave(`booking:${bookingId}`);
      console.log(`用户 ${userId} 离开咨询房间 ${bookingId}`);
    });

    // 咨询结束：清理实时数据
    socket.on('consultation:end', async (bookingId: string) => {
      try {
        const booking = await prisma.expertBooking.findUnique({
          where: { id: bookingId },
          select: { patientId: true, consultantId: true },
        });

        if (!booking) return;

        // 只有参与者才能结束咨询
        if (booking.patientId !== userId && booking.consultantId !== userId) return;

        // 更新预约状态为已完成
        await prisma.expertBooking.update({
          where: { id: bookingId },
          data: { status: 'COMPLETED', endedAt: new Date() },
        });

        // 通知房间内所有人清理实时数据
        io.to(`booking:${bookingId}`).emit('consultation:ended', { bookingId });

        // 离开房间
        socket.leave(`booking:${bookingId}`);
        console.log(`咨询 ${bookingId} 已结束，实时数据已清理`);
      } catch (error) {
        console.error('结束咨询失败:', error);
      }
    });

    socket.on('expert:message', async (data: { bookingId: string; content: string; contentType?: string }) => {
      try {
        const message = await prisma.expertMessage.create({
          data: {
            bookingId: data.bookingId,
            senderId: userId,
            content: data.content,
            contentType: data.contentType || 'text',
          },
          include: {
            sender: { select: { id: true, nickname: true, avatar: true } },
          },
        });

        // 广播给咨询双方
        io.to(`booking:${data.bookingId}`).emit('expert:message', message);
      } catch (error) {
        socket.emit('error', { message: '消息发送失败' });
      }
    });

    // AI 咨询消息状态
    socket.on('typing:start', (conversationId: string) => {
      socket.to(`user:${userId}`).emit('typing:update', { conversationId, isTyping: true });
    });

    socket.on('typing:stop', (conversationId: string) => {
      socket.to(`user:${userId}`).emit('typing:update', { conversationId, isTyping: false });
    });

    // 多模态心理数据同步（患者 → 咨询师）
    socket.on('multimodal:update', async (data: { bookingId: string; multimodalData: any }) => {
      try {
        // 验证预约关系
        const booking = await prisma.expertBooking.findUnique({
          where: { id: data.bookingId },
          select: { patientId: true, consultantId: true, status: true },
        });

        if (!booking) return;

        // 只有患者才能发送多模态数据
        if (booking.patientId !== userId) return;

        // 只有进行中的咨询才同步数据
        if (booking.status !== 'CONFIRMED' && booking.status !== 'IN_PROGRESS') return;

        // 转发给咨询房间内的咨询师
        socket.to(`booking:${data.bookingId}`).emit('patient:multimodal', data.multimodalData);
      } catch (error) {
        console.error('多模态数据同步失败:', error);
      }
    });

    // ===== WebRTC 双向视频信令 =====
    // 加入视频房间
    socket.on('webrtc:join', async (data: { bookingId: string }) => {
      try {
        const booking = await prisma.expertBooking.findUnique({
          where: { id: data.bookingId },
          select: { patientId: true, consultantId: true, status: true },
        });
        if (!booking) return;
        if (booking.patientId !== userId && booking.consultantId !== userId) return;

        socket.join(`webrtc:${data.bookingId}`);

        // 通知房间内其他人有新成员加入
        socket.to(`webrtc:${data.bookingId}`).emit('webrtc:user-joined', {
          userId,
          role: booking.patientId === userId ? 'patient' : 'consultant',
        });

        console.log(`[WebRTC] 用户 ${userId} 加入视频房间 ${data.bookingId}`);
      } catch (error) {
        console.error('[WebRTC] 加入视频房间失败:', error);
      }
    });

    // 转发 SDP Offer
    socket.on('webrtc:offer', (data: { bookingId: string; offer: any }) => {
      socket.to(`webrtc:${data.bookingId}`).emit('webrtc:offer', {
        userId,
        offer: data.offer,
      });
    });

    // 转发 SDP Answer
    socket.on('webrtc:answer', (data: { bookingId: string; answer: any }) => {
      socket.to(`webrtc:${data.bookingId}`).emit('webrtc:answer', {
        userId,
        answer: data.answer,
      });
    });

    // 转发 ICE Candidate
    socket.on('webrtc:ice-candidate', (data: { bookingId: string; candidate: any }) => {
      socket.to(`webrtc:${data.bookingId}`).emit('webrtc:ice-candidate', {
        userId,
        candidate: data.candidate,
      });
    });

    // 离开视频房间
    socket.on('webrtc:leave', (bookingId: string) => {
      socket.leave(`webrtc:${bookingId}`);
      socket.to(`webrtc:${bookingId}`).emit('webrtc:user-left', { userId });
      console.log(`[WebRTC] 用户 ${userId} 离开视频房间 ${bookingId}`);
    });

    // 通知
    socket.on('notification:read', async (notificationIds: string[]) => {
      await prisma.notification.updateMany({
        where: { id: { in: notificationIds }, userId },
        data: { isRead: true },
      });
    });

    socket.on('disconnect', () => {
      console.log(`用户断开: ${userId}`);
    });
  });
};
