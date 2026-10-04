import { Router, Response, NextFunction } from 'express';
import { CrisisService } from '../services/crisis.service';
import { authenticate, AuthRequest } from '../middlewares/auth';
import prisma from '../config/database';

const router = Router();
const crisisService = new CrisisService();

// 获取心理援助热线
router.get('/hotlines', (req, res) => {
  res.json({ code: 0, data: crisisService.getHotlines() });
});

// 检测文本危机（用于 AI 对话中）
router.post('/detect', authenticate, (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const { text } = req.body;
    const result = crisisService.detectCrisis(text);
    res.json({ code: 0, data: result });
  } catch (e) { next(e); }
});

// ===== 管理员：获取危机记录列表 =====
router.get('/records', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    if (req.userRole !== 'ADMIN' && req.userRole !== 'CONSULTANT') {
      return res.status(403).json({ code: 403, message: '无权限' });
    }
    const status = req.query.status as string;
    const severity = req.query.severity as string;
    const page = parseInt(req.query.page as string) || 1;
    const pageSize = parseInt(req.query.pageSize as string) || 20;

    const where: any = {};
    if (status) where.status = status;
    if (severity) where.severity = severity;

    const [records, total] = await Promise.all([
      prisma.$queryRawUnsafe(`
        SELECT cr.*, u.nickname as "userName", u.email as "userEmail"
        FROM crisis_records cr
        JOIN users u ON cr."userId" = u.id
        ${status ? `WHERE cr.status = '${status}'` : ''}
        ${severity && !status ? `WHERE cr.severity = '${severity}'` : ''}
        ORDER BY cr."detectedAt" DESC
        LIMIT ${pageSize} OFFSET ${(page - 1) * pageSize}
      `),
      prisma.$queryRawUnsafe(`
        SELECT COUNT(*)::int as count FROM crisis_records
        ${status ? `WHERE status = '${status}'` : ''}
        ${severity && !status ? `WHERE severity = '${severity}'` : ''}
      `),
    ]);

    res.json({ code: 0, data: { records, total: (total as any)[0].count, page, pageSize } });
  } catch (e) { next(e); }
});

// ===== 管理员：获取危机统计 =====
router.get('/stats', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    if (req.userRole !== 'ADMIN' && req.userRole !== 'CONSULTANT') {
      return res.status(403).json({ code: 403, message: '无权限' });
    }

    const [openCount, criticalCount, weekCount, resolvedCount] = await Promise.all([
      prisma.$queryRawUnsafe(`SELECT COUNT(*)::int as count FROM crisis_records WHERE status = 'OPEN'`),
      prisma.$queryRawUnsafe(`SELECT COUNT(*)::int as count FROM crisis_records WHERE severity = 'CRITICAL' AND status != 'RESOLVED' AND status != 'CLOSED'`),
      prisma.$queryRawUnsafe(`SELECT COUNT(*)::int as count FROM crisis_records WHERE "detectedAt" >= NOW() - INTERVAL '7 days'`),
      prisma.$queryRawUnsafe(`SELECT COUNT(*)::int as count FROM crisis_records WHERE status = 'RESOLVED'`),
    ]);

    res.json({
      code: 0,
      data: {
        open: (openCount as any)[0].count,
        critical: (criticalCount as any)[0].count,
        thisWeek: (weekCount as any)[0].count,
        resolved: (resolvedCount as any)[0].count,
      },
    });
  } catch (e) { next(e); }
});

// ===== 管理员：更新危机记录状态 =====
router.put('/records/:id', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    if (req.userRole !== 'ADMIN' && req.userRole !== 'CONSULTANT') {
      return res.status(403).json({ code: 403, message: '无权限' });
    }
    const { status, resolution } = req.body;
    const resolvedAt = status === 'RESOLVED' || status === 'CLOSED' ? 'NOW()' : 'NULL';

    await prisma.$executeRawUnsafe(`
      UPDATE crisis_records
      SET status = $1, resolution = $2, "resolvedAt" = ${resolvedAt}, "updatedAt" = NOW()
      WHERE id = $3
    `, status, resolution || null, req.params.id);

    res.json({ code: 0, message: '更新成功' });
  } catch (e) { next(e); }
});

// 获取紧急联系人
router.get('/contacts', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const contacts = await crisisService.getEmergencyContacts(req.userId!);
    res.json({ code: 0, data: contacts });
  } catch (e) { next(e); }
});

// 添加紧急联系人
router.post('/contacts', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const contact = await crisisService.addContact(req.userId!, req.body);
    res.json({ code: 0, data: contact });
  } catch (e) { next(e); }
});

// 更新紧急联系人
router.put('/contacts/:id', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const contact = await crisisService.updateContact(req.params.id as string, req.userId!, req.body);
    res.json({ code: 0, data: contact });
  } catch (e) { next(e); }
});

// 删除紧急联系人
router.delete('/contacts/:id', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    await crisisService.deleteContact(req.params.id as string, req.userId!);
    res.json({ code: 0, message: '删除成功' });
  } catch (e) { next(e); }
});

export default router;
