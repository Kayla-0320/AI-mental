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
    socket.on('expert:join', (bookingId: string) => {
      socket.join(`booking:${bookingId}`);
    });

    socket.on('expert:leave', (bookingId: string) => {
      socket.leave(`booking:${bookingId}`);
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
