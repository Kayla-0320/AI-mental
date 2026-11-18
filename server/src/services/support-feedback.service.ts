import { z } from 'zod';
import { isDeepStrictEqual } from 'node:util';
import prisma from '../config/database';
import { AppError } from '../middlewares/errorHandler';
import { invalidateSupportAgentOffers } from './support-agent.service';

export interface SupportStatePayload {
  schema_version: 'support.v1';
  session_id: string;
  revision: number;
  goal: string;
  declined: boolean;
  items: Array<Record<string, unknown>>;
  feedback_log: Array<Record<string, unknown>>;
  [key: string]: unknown;
}

const feedbackSchema = z.object({
  feedback_id: z.string().min(1).max(100),
  expected_revision: z.number().int().nonnegative(),
  action: z.enum(['confirm', 'correct', 'reject', 'set_goal', 'decline', 'resume']),
  item_id: z.string().max(80).default(''),
  text: z.string().max(240).default(''),
  goal: z.enum(['', 'unspecified', 'listen', 'understand', 'act']).default(''),
}).strict().superRefine((value, ctx) => {
  if (['confirm', 'correct', 'reject'].includes(value.action) && !value.item_id.trim()) {
    ctx.addIssue({ code: z.ZodIssueCode.custom, message: '请指定要反馈的理解', path: ['item_id'] });
  }
  if (value.action === 'correct' && !value.text.trim()) {
    ctx.addIssue({ code: z.ZodIssueCode.custom, message: '修正内容不能为空', path: ['text'] });
  }
  if (value.action === 'set_goal' && !value.goal) {
    ctx.addIssue({ code: z.ZodIssueCode.custom, message: '请选择支持目标', path: ['goal'] });
  }
});

export function storedSentiment(raw: string | null | undefined): Record<string, unknown> {
  try {
    const value: unknown = JSON.parse(raw || '{}');
    return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {};
  } catch { return {}; }
}

export function readSupportState(rows: Array<{ role: string; sentiment?: string | null }>, sessionId: string): SupportStatePayload | null {
  // 只读最近一条助手消息；缺失时不能复活更早已失效的理解。
  const latest = [...rows].reverse().find(row => row.role === 'assistant');
  const state = storedSentiment(latest?.sentiment).support_state as SupportStatePayload | undefined;
  return state?.schema_version === 'support.v1' && state.session_id === sessionId ? state : null;
}

export async function computeSupportFeedback(payload: Record<string, unknown>): Promise<SupportStatePayload> {
  const base = process.env.ALGORITHM_API_URL || 'http://127.0.0.1:8001';
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 10000);
  try {
    const response = await fetch(`${base}/api/v1/intervention/support-feedback`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload), signal: controller.signal,
    });
    if (!response.ok) {
      throw new AppError(response.status === 409 ? '理解已更新或反馈无效，请刷新后重试' : '理解反馈服务暂不可用', response.status === 409 ? 409 : 503);
    }
    const result = await response.json() as { support_state?: SupportStatePayload };
    const state = result.support_state;
    if (!state || state.schema_version !== 'support.v1' || state.session_id !== payload.session_id || !Number.isInteger(state.revision)) {
      throw new AppError('理解反馈服务返回无效状态', 502);
    }
    return state;
  } catch (error) {
    if (error instanceof AppError) throw error;
    throw new AppError('理解反馈未保存，请稍后重试', 503);
  } finally { clearTimeout(timer); }
}

/** 所有权、最新消息与版本均在写入前验证；反馈失败不会假装保存成功。 */
export class SupportFeedbackService {
  constructor(private db = prisma, private compute = computeSupportFeedback) {}

  async submit(conversationId: string, userId: string, messageId: string, input: unknown) {
    const parsed = feedbackSchema.safeParse(input);
    if (!parsed.success) throw new AppError(parsed.error.issues[0]?.message || '反馈内容无效', 400);
    const conversation = await this.db.conversation.findFirst({ where: { id: conversationId, userId, isActive: true }, select: { id: true } });
    if (!conversation) throw new AppError('对话不存在', 404);
    const message = await this.db.message.findFirst({ where: { id: messageId, conversationId, role: 'assistant' } });
    if (!message) throw new AppError('消息不存在', 404);
    const latest = await this.db.message.findFirst({ where: { conversationId }, orderBy: [{ createdAt: 'desc' }, { id: 'desc' }], select: { id: true } });
    if (latest?.id !== messageId) throw new AppError('请等待当前回复完成，或反馈最新一条理解', 409);
    const state = readSupportState([message], conversationId);
    if (!state) throw new AppError('这条消息没有可反馈的理解', 409);
    const updated = await this.compute({ session_id: conversationId, state, ...parsed.data });
    // 短事务只包含复验和写入；外部 HTTP 不在锁内。
    // PostgreSQL 的父行 FOR UPDATE 与 Message 外键插入的 KEY SHARE 冲突，
    // 关闭“检查最新消息后、写入前恰好又插入新轮次”的窗口。
    return this.db.$transaction(async tx => {
      // 与账号删除、后台任务保持 User→Conversation 顺序；外部计算仍在事务之外。
      const owner = await tx.$queryRaw<Array<{ id: string }>>`SELECT id FROM users WHERE id = ${userId} AND "isActive" = true FOR UPDATE`;
      if (owner.length !== 1) throw new AppError('账号不可用', 401);
      const locked = await tx.$queryRaw<Array<{ id: string }>>`
        SELECT id FROM conversations
        WHERE id = ${conversationId} AND "userId" = ${userId} AND "isActive" = true FOR UPDATE`;
      if (locked.length !== 1) throw new AppError('对话不存在', 404);
      const stillLatest = await tx.message.findFirst({ where: { conversationId }, orderBy: [{ createdAt: 'desc' }, { id: 'desc' }], select: { id: true } });
      if (stillLatest?.id !== messageId) throw new AppError('对话已继续，请反馈最新一条理解', 409);
      // 情绪分析可在外部计算期间完成；只对理解状态判冲突，并在行锁内合并最新元数据。
      await tx.$queryRaw`SELECT id FROM messages WHERE id = ${messageId} AND "conversationId" = ${conversationId} FOR UPDATE`;
      const current = await tx.message.findFirst({ where: { id: messageId, conversationId, role: 'assistant' } });
      if (!current || !isDeepStrictEqual(readSupportState([current], conversationId), state)) {
        throw new AppError('理解已被其他操作更新，请刷新后重试', 409);
      }
      const result = await tx.message.updateMany({
        where: { id: message.id, conversationId, sentiment: current.sentiment },
        data: { sentiment: JSON.stringify({ ...storedSentiment(current.sentiment), support_state: updated }) },
      });
      if (result.count !== 1) throw new AppError('理解已被其他操作更新，请刷新后重试', 409);
      await invalidateSupportAgentOffers(tx, userId, true);
      return { messageId, supportState: updated };
    });
  }
}

export default new SupportFeedbackService();
