import { Router } from 'express';
import prisma from '../config/database';
import { authenticate, AuthRequest } from '../middlewares/auth';

const router = Router();

// 获取通知列表
router.get('/', authenticate, async (req: AuthRequest, res) => {
  try {
    const userId = req.userId!;
    const notifications = await prisma.notification.findMany({
      where: { userId },
      orderBy: { createdAt: 'desc' },
      take: 50,
    });

    const unreadCount = await prisma.notification.count({
      where: { userId, isRead: false },
    });

    res.json({ code: 0, data: { notifications, unreadCount } });
  } catch (error: any) {
    res.status(500).json({ code: 500, message: error.message });
  }
});

// 标记为已读
router.put('/:id/read', authenticate, async (req: AuthRequest, res) => {
  try {
    const userId = req.userId!;
    const { id } = req.params;

    await prisma.notification.updateMany({
      where: { id: id as string, userId },
      data: { isRead: true },
    });

    res.json({ code: 0, message: '已标记为已读' });
  } catch (error: any) {
    res.status(500).json({ code: 500, message: error.message });
  }
});

// 全部标记为已读
router.put('/read-all', authenticate, async (req: AuthRequest, res) => {
  try {
    const userId = req.userId!;

    await prisma.notification.updateMany({
      where: { userId, isRead: false },
      data: { isRead: true },
    });

    res.json({ code: 0, message: '全部已标记为已读' });
  } catch (error: any) {
    res.status(500).json({ code: 500, message: error.message });
  }
});

// 删除通知
router.delete('/:id', authenticate, async (req: AuthRequest, res) => {
  try {
    const userId = req.userId!;
    const { id } = req.params;

    await prisma.notification.deleteMany({
      where: { id: id as string, userId },
    });

    res.json({ code: 0, message: '已删除' });
  } catch (error: any) {
    res.status(500).json({ code: 500, message: error.message });
  }
});

export default router;
