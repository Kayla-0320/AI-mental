import { Router, Response, NextFunction } from 'express';
import { Prisma } from '@prisma/client';
import prisma from '../config/database';
import { authenticate, authorize, AuthRequest } from '../middlewares/auth';

/** 只读取本人最近一轮元数据；打开页面不触发模型生成或支持活动。 */
export async function readLatestInnovation(userId: string, db: Prisma.TransactionClient = prisma): Promise<Record<string, unknown>> {
  const conversation = await db.conversation.findFirst({ where: { userId, isActive: true },
    orderBy: [{ updatedAt: 'desc' }, { id: 'desc' }], select: { id: true, title: true } });
  if (!conversation) return { status: 'no_conversation', conversation: null, reference: null };
  const latest = await db.message.findFirst({ where: { conversationId: conversation.id },
    orderBy: [{ createdAt: 'desc' }, { id: 'desc' }], select: { id: true, role: true, createdAt: true, sentiment: true } });
  if (!latest || latest.role !== 'assistant') return { status: latest ? 'awaiting_reply' : 'no_messages', conversation, reference: null };
  let metadata: Record<string, unknown> = {};
  try {
    if (latest.sentiment && Buffer.byteLength(latest.sentiment, 'utf8') <= 65536) {
      const parsed: unknown = JSON.parse(latest.sentiment);
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) metadata = parsed as Record<string, unknown>;
    }
  } catch { /* 历史元数据损坏时展示缺少证据，不推造观测。 */ }
  return { status: metadata.innovation ? 'recorded' : 'no_reference', conversation,
    messageId: latest.id, recordedAt: latest.createdAt, reference: metadata.innovation ?? null };
}

/** 区分可连接、已配置与实际单轮结果，不能用文件存在冒充推理成功。 */
async function readRuntime(): Promise<Record<string, unknown>> {
  try {
    const base = process.env.ALGORITHM_API_URL || 'http://127.0.0.1:8001';
    const response = await fetch(`${base}/api/v1/innovation/capabilities`, { signal: AbortSignal.timeout(4000) });
    if (!response.ok) return { connected: false };
    const raw = await response.text();
    if (Buffer.byteLength(raw, 'utf8') > 16384) return { connected: false };
    const value = JSON.parse(raw) as Record<string, unknown>;
    return { connected: true, enabled: value.enabled === true, artifactPresent: value.fusion_artifact_present === true };
  } catch { return { connected: false }; }
}

const router = Router();
router.use(authenticate, authorize('PATIENT'));
router.get('/', async (req: AuthRequest, res: Response, next: NextFunction): Promise<void> => {
  try {
    const [latest, runtime] = await Promise.all([readLatestInnovation(req.userId!), readRuntime()]);
    res.json({ code: 0, data: { latest, runtime } });
  } catch (error) { next(error); }
});
export default router;
