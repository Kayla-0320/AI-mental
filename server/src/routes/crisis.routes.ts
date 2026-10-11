import { Router, Response, NextFunction } from 'express';
import { CrisisService } from '../services/crisis.service';
import { authenticate, AuthRequest } from '../middlewares/auth';
import prisma from '../config/database';

const router = Router();
const crisisService = new CrisisService();

/** 危机记录状态机：只允许这些流转，禁止任意写 status */
const ALLOWED_TRANSITIONS: Record<string, string[]> = {
  OPEN: ['IN_PROGRESS', 'RESOLVED', 'CLOSED'],
  IN_PROGRESS: ['RESOLVED', 'CLOSED'],
  RESOLVED: ['CLOSED'],
  CLOSED: [],
};

const VALID_STATUS = Object.keys(ALLOWED_TRANSITIONS);
const VALID_SEVERITY = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'];

function assertStaff(req: AuthRequest, res: Response): boolean {
  if (req.userRole !== 'ADMIN' && req.userRole !== 'CONSULTANT') {
    res.status(403).json({ code: 403, message: '无权限' });
    return false;
  }
  return true;
}

/** 危机记录对外形状：保持 userName/userEmail 口径，前端无需改动 */
type CrisisRow = {
  id: string;
  userId: string;
  severity: string;
  status: string;
  trigger: string;
  triggerType: string;
  sourceText: string | null;
  description: string | null;
  detectedAt: Date;
  resolvedAt: Date | null;
  resolvedBy: string | null;
  resolution: string | null;
  user?: { nickname: string | null; email: string } | null;
};

function toApiShape(row: CrisisRow) {
  const { user, ...rest } = row;
  return {
    ...rest,
    userName: user?.nickname || '未知',
    userEmail: user?.email || '',
  };
}

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

// ===== 管理端/咨询师端：危机待办（未确认 + 处理中） =====
// 注意：必须声明在 `/records` 之前，避免被静态匹配规则吞掉。
router.get('/records/todo', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    if (!assertStaff(req, res)) return;
    const rows = await prisma.crisisRecord.findMany({
      where: { status: { in: ['OPEN', 'IN_PROGRESS'] } },
      include: { user: { select: { nickname: true, email: true } } },
      orderBy: { detectedAt: 'desc' },
      take: 100,
    });
    // severity 是字符串枚举，数据库排序会把 MEDIUM 排到 CRITICAL 前面，因此在内存里按临床严重度排。
    const severityRank: Record<string, number> = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3 };
    const sorted = rows
      .map(toApiShape)
      .sort((a, b) => {
        const bySeverity = (severityRank[a.severity] ?? 9) - (severityRank[b.severity] ?? 9);
        if (bySeverity !== 0) return bySeverity;
        // 同级别里，还没被任何人接收的排前面
        const byAck = (a.status === 'OPEN' ? 0 : 1) - (b.status === 'OPEN' ? 0 : 1);
        if (byAck !== 0) return byAck;
        return new Date(b.detectedAt).getTime() - new Date(a.detectedAt).getTime();
      });
    res.json({ code: 0, data: sorted });
  } catch (e) { next(e); }
});

// ===== 管理端：获取危机记录列表 =====
router.get('/records', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    if (!assertStaff(req, res)) return;
    const status = req.query.status as string | undefined;
    const severity = req.query.severity as string | undefined;
    const page = parseInt(req.query.page as string) || 1;
    const pageSize = parseInt(req.query.pageSize as string) || 20;

    const where: Record<string, unknown> = {};
    if (status) {
      // 白名单校验：非法状态直接返回空集，而不是拼进 SQL
      if (!VALID_STATUS.includes(status)) {
        return res.json({ code: 0, data: { records: [], total: 0, page, pageSize } });
      }
      where.status = status;
    }
    if (severity) {
      if (!VALID_SEVERITY.includes(severity)) {
        return res.json({ code: 0, data: { records: [], total: 0, page, pageSize } });
      }
      where.severity = severity;
    }

    const [rows, total] = await Promise.all([
      prisma.crisisRecord.findMany({
        where,
        include: { user: { select: { nickname: true, email: true } } },
        orderBy: { detectedAt: 'desc' },
        skip: (page - 1) * pageSize,
        take: pageSize,
      }),
      prisma.crisisRecord.count({ where }),
    ]);

    res.json({ code: 0, data: { records: rows.map(toApiShape), total, page, pageSize } });
  } catch (e) { next(e); }
});

// ===== 管理端：获取危机统计 =====
router.get('/stats', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    if (!assertStaff(req, res)) return;

    const weekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
    const ackDeadline = new Date(Date.now() - crisisService.ACK_TIMEOUT_MS);
    const [openCount, criticalCount, weekCount, resolvedCount, overdueCount] = await Promise.all([
      prisma.crisisRecord.count({ where: { status: { in: ['OPEN', 'IN_PROGRESS'] } } }),
      prisma.crisisRecord.count({ where: { severity: 'CRITICAL', status: { in: ['OPEN', 'IN_PROGRESS'] } } }),
      prisma.crisisRecord.count({ where: { detectedAt: { gte: weekAgo } } }),
      prisma.crisisRecord.count({ where: { status: { in: ['RESOLVED', 'CLOSED'] } } }),
      // 超时未确认：OPEN 超过 ACK_TIMEOUT_MS 仍未被任何人接收
      prisma.crisisRecord.count({ where: { status: 'OPEN', detectedAt: { lt: ackDeadline } } }),
    ]);

    res.json({
      code: 0,
      data: {
        open: openCount,
        critical: criticalCount,
        thisWeek: weekCount,
        resolved: resolvedCount,
        overdue: overdueCount,
      },
    });
  } catch (e) { next(e); }
});

// ===== 管理端/咨询师端：确认接收（ACK）=====
// 与 `PUT /records/:id` 的区别：这里只做一件事——把 OPEN 变成 IN_PROGRESS 并记下接收人。
// 独立出来是为了让「谁接的、什么时候接的」可追踪，而不是靠前端拼一个 PUT。
router.post('/records/:id/ack', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    if (!assertStaff(req, res)) return;
    const id = req.params.id as string;
    const existing = await prisma.crisisRecord.findUnique({ where: { id } });
    if (!existing) return res.status(404).json({ code: 404, message: '危机记录不存在' });

    if (existing.status === 'IN_PROGRESS') {
      return res.json({ code: 0, data: existing, message: '已接收' });
    }
    if (existing.status !== 'OPEN') {
      return res.status(409).json({ code: 409, message: `记录已处于「${existing.status}」，无法重复接收` });
    }

    const updated = await prisma.crisisRecord.update({
      where: { id },
      data: { status: 'IN_PROGRESS', resolvedBy: req.userId!, updatedAt: new Date() },
    });
    res.json({ code: 0, data: updated });
  } catch (e) { next(e); }
});

// ===== 管理端/咨询师端：更新危机记录状态 =====
router.put('/records/:id', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    if (!assertStaff(req, res)) return;
    const id = req.params.id as string;
    const { status, resolution } = req.body as { status?: string; resolution?: string };

    if (!status || !VALID_STATUS.includes(status)) {
      return res.status(400).json({ code: 400, message: '无效的状态' });
    }

    const existing = await prisma.crisisRecord.findUnique({ where: { id } });
    if (!existing) return res.status(404).json({ code: 404, message: '危机记录不存在' });

    const allowed = ALLOWED_TRANSITIONS[existing.status] || [];
    if (!allowed.includes(status)) {
      return res.status(409).json({
        code: 409,
        message: `不允许从「${existing.status}」流转到「${status}」`,
      });
    }

    // 结案必须有处置结论：否则「已解决」三个字后面什么都没有，事后无法复盘。
    if ((status === 'RESOLVED' || status === 'CLOSED') && !(resolution || existing.resolution)) {
      return res.status(400).json({ code: 400, message: '结案前必须填写处置结论' });
    }

    const updated = await prisma.crisisRecord.update({
      where: { id },
      data: {
        status,
        ...(resolution ? { resolution } : {}),
        // 接收人/处置人：首次进入处理态或结案时记录当前操作者
        ...(existing.resolvedBy ? {} : { resolvedBy: req.userId! }),
        ...(status === 'RESOLVED' || status === 'CLOSED'
          ? { resolvedAt: new Date() }
          : { resolvedAt: null }),
        updatedAt: new Date(),
      },
    });

    res.json({ code: 0, data: updated });
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
