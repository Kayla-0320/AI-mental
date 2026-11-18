import { createHash, randomUUID } from 'node:crypto';
import { Prisma } from '@prisma/client';
import { z } from 'zod';
import { forecastSelfReports, PersonalForecast } from './personal-forecast';
import prisma from '../config/database';
import { AppError } from '../middlewares/errorHandler';
import { fetchPlatformModelReference, ModelReference, ModelReferenceProvider, modelReferenceSchema, PersonalEvidence,
  PersonalSupportContext, readPersonalSupportContext, validateModelReference } from './support-agent-context';

/** 保留状态记录不代表一次实际练习，必须从练习列表和成绩统计中排除。 */
export const SUPPORT_STATE_TYPE = 'support_agent_state';
export const SUPPORT_TOOLS = {
  breathing: { title: '两轮气球呼吸', durationSeconds: 28, resourceId: 'balloon_4_4_6_v1' },
  music: { title: '听一会儿雨声', durationSeconds: 30, resourceId: 'rain_mixkit_2394' },
} as const;
const toolSchema = z.enum(['breathing', 'music']);
const statusSchema = z.enum(['offered', 'accepted', 'running', 'completed', 'failed', 'cancelled', 'declined', 'expired']);
const actionSchema = z.object({
  id: z.string(), tool: toolSchema, title: z.string(), resourceId: z.string(), durationSeconds: z.number(),
  reason: z.string(), status: statusSchema, offeredAt: z.string(), expiresAt: z.string(),
  acceptedAt: z.string().optional(), startedAt: z.string().optional(), endedAt: z.string().optional(),
  sessionId: z.string().optional(), executionToken: z.string().optional(), failure: z.string().optional(),
  feedback: z.enum(['helpful', 'neutral', 'unhelpful']).optional(),
  origin: z.enum(['page', 'background']).default('page'),
  contextRevision: z.number().int().nonnegative().optional(), contextMessageId: z.string().optional(),
  notificationId: z.string().optional(),
}).strict();
const backgroundSchema = z.object({
  enabled: z.boolean().default(false), intervalHours: z.union([z.literal(6), z.literal(12), z.literal(24)]).default(24),
  notifyInApp: z.boolean().default(false),
  quietStart: z.number().int().min(0).max(23).default(22), quietEnd: z.number().int().min(0).max(23).default(8),
  nextRunAt: z.string().datetime().nullable().default(null), lastEvidenceKey: z.string().max(300).default(''),
  lease: z.object({ owner: z.string().max(100), token: z.string().uuid(), until: z.string().datetime(), scheduledAt: z.string().datetime() }).strict().nullable().default(null),
  events: z.array(z.object({ at: z.string().datetime(), result: z.string().max(80) }).strict()).max(32).default([]),
}).strict();
const stateSchema = z.object({
  schema: z.literal('support-agent.v1'), revision: z.number().int().nonnegative(), enabled: z.boolean(),
  preferredTool: z.enum(['none', 'breathing', 'music']), updatedAt: z.string(),
  actions: z.array(actionSchema).max(20),
  requests: z.array(z.object({ id: z.string(), digest: z.string() }).strict()).max(64),
  background: backgroundSchema.default({}), modelReference: modelReferenceSchema.nullable().default(null),
}).strict();
export type SupportAgentState = z.infer<typeof stateSchema>;
export type SupportAgentAction = z.infer<typeof actionSchema>;
const common = { requestId: z.string().uuid(), expectedRevision: z.number().int().nonnegative() };
const target = { ...common, actionId: z.string().uuid() };
const execution = { ...target, executionToken: z.string().uuid() };
const commandSchema = z.discriminatedUnion('op', [
  z.object({ ...common, op: z.literal('configure'), enabled: z.boolean(), preferredTool: z.enum(['none', 'breathing', 'music']) }).strict(),
  z.object({ ...common, op: z.literal('propose') }).strict(),
  z.object({ ...common, op: z.literal('background_settings'), enabled: z.boolean(),
    notifyInApp: z.boolean().default(false),
    intervalHours: z.union([z.literal(6), z.literal(12), z.literal(24)]), quietStart: z.number().int().min(0).max(23), quietEnd: z.number().int().min(0).max(23) }).strict(),
  z.object({ ...target, op: z.literal('accept') }).strict(),
  z.object({ ...target, op: z.literal('decline') }).strict(),
  z.object({ ...execution, op: z.literal('started') }).strict(),
  z.object({ ...execution, op: z.literal('complete') }).strict(),
  z.object({ ...execution, op: z.literal('fail'), reason: z.enum(['playback_failed', 'tool_failed', 'interrupted']) }).strict(),
  z.object({ ...target, op: z.literal('cancel') }).strict(),
  z.object({ ...target, op: z.literal('feedback'), value: z.enum(['helpful', 'neutral', 'unhelpful']) }).strict(),
]);
export type SupportAgentCommand = z.infer<typeof commandSchema>;
type MoodPoint = { id: string; mood: string; score: number; recordedAt: Date };
export interface BackgroundSupportLease { userId: string; owner: string; token: string; evidence: PersonalEvidence | null }
export interface SupportAgentOverview {
  state: SupportAgentState;
  trend: { source: 'self_report'; observations: number; points: Array<{ id: string; mood: string; score: number; at: string }>;
    latestScore: number | null; mean: number | null; change: number | null; stale: boolean; label: string; forecast: PersonalForecast };
  suggestionEligible: boolean;
  context: PersonalSupportContext;
  modelReference: ModelReference | null;
  replayed?: boolean;
}
const active = (action: SupportAgentAction): boolean => ['offered', 'accepted', 'running'].includes(action.status);
const stateId = (userId: string): string => `support-agent:${createHash('sha256').update(userId).digest('hex').slice(0, 40)}`;

/** 将北京时间安静时段内的候选时间顺延到安静结束，避免24小时间隔永久卡在夜间。 */
export function nextAllowedSupportTime(candidate: Date, quietStart: number, quietEnd: number): Date {
  const local = new Date(candidate.getTime() + 480 * 60000); const hour = local.getUTCHours();
  const quiet = quietStart > quietEnd ? hour >= quietStart || hour < quietEnd : hour >= quietStart && hour < quietEnd;
  if (!quiet) return new Date(candidate);
  const end = new Date(local); end.setUTCHours(quietEnd, 0, 0, 0);
  if (end <= local) end.setUTCDate(end.getUTCDate() + 1);
  return new Date(end.getTime() - 480 * 60000);
}

/** 撤回本人证据或删除会话时，同一事务取消旧建议并令站内提醒失效。锁顺序固定为用户→会话。 */
export async function invalidateSupportAgentOffers(tx: Prisma.TransactionClient, userId: string, userAlreadyLocked = false): Promise<void> {
  if (!userAlreadyLocked) {
    const users = await tx.$queryRaw<Array<{ id: string }>>`SELECT id FROM users WHERE id = ${userId} AND "isActive" = true FOR UPDATE`;
    if (!users.length) throw new AppError('账号不可用', 401);
  }
  const row = await tx.healingSession.findFirst({ where: { id: stateId(userId), userId, type: SUPPORT_STATE_TYPE }, select: { metadata: true } });
  if (!row) return;
  await tx.notification.updateMany({ where: { userId, type: 'support_agent', isRead: false }, data: { isRead: true } });
  let state: SupportAgentState;
  try { state = stateSchema.parse(JSON.parse(row.metadata || '{}')); } catch { return; }
  state.modelReference = null; state.background.lease = null;
  for (const action of state.actions) if (action.status === 'offered') { action.status = 'cancelled'; action.endedAt = new Date().toISOString(); }
  state.revision++; state.updatedAt = new Date().toISOString();
  await tx.healingSession.updateMany({ where: { id: stateId(userId), userId, type: SUPPORT_STATE_TYPE }, data: { metadata: JSON.stringify(state) } });
}

/** 只整理本人已发生的自评，不把分类器概率当本人情绪或未来预测。 */
export function summarizeSelfReports(rows: MoodPoint[], now: Date): SupportAgentOverview['trend'] {
  const points = rows.filter(row => Number.isInteger(row.score) && row.score >= 1 && row.score <= 10
    && Number.isFinite(row.recordedAt.getTime()) && row.recordedAt <= now)
    .sort((a, b) => a.recordedAt.getTime() - b.recordedAt.getTime()).slice(-14);
  const latest = points[points.length - 1];
  const stale = !latest || now.getTime() - latest.recordedAt.getTime() > 48 * 60 * 60 * 1000;
  return {
    source: 'self_report', observations: points.length,
    forecast: forecastSelfReports(rows, now),
    points: points.map(row => ({ id: row.id, mood: row.mood, score: row.score, at: row.recordedAt.toISOString() })),
    latestScore: latest?.score ?? null,
    mean: points.length >= 3 ? Number((points.reduce((sum, row) => sum + row.score, 0) / points.length).toFixed(2)) : null,
    change: points.length >= 3 ? latest!.score - points[0].score : null,
    stale,
    label: !latest ? '还没有本人心情记录' : points.length < 3 ? '记录较少，只展示本人打卡' : stale ? '记录较早，暂不推断当前心情' : '最近自评记录，仅描述过去变化',
  };
}

/** 规则式支持 Agent；写操作在本人用户行锁内串行，状态与实际执行分别持久化。 */
export class SupportAgentService {
  constructor(private db = prisma, private clock: () => Date = () => new Date(), private modelProvider: ModelReferenceProvider = fetchPlatformModelReference) {}

  private readState(raw: string | null | undefined, now: Date): SupportAgentState {
    if (!raw) return stateSchema.parse({ schema: 'support-agent.v1', revision: 0, enabled: false, preferredTool: 'none', updatedAt: now.toISOString(), actions: [], requests: [] });
    if (Buffer.byteLength(raw, 'utf8') > 65536) throw new AppError('陪伴记录超出上限，请联系维护人员', 503);
    try { return stateSchema.parse(JSON.parse(raw)); }
    catch { throw new AppError('陪伴记录暂时无法读取，未覆盖已有记录', 503); }
  }

  private async overviewFor(db: Prisma.TransactionClient, userId: string, state: SupportAgentState, now: Date): Promise<SupportAgentOverview> {
    const rows = state.enabled ? await db.moodRecord.findMany({
      where: { userId, recordedAt: { gte: new Date(now.getTime() - 14 * 86400000), lte: now } },
      select: { id: true, mood: true, score: true, recordedAt: true }, orderBy: [{ recordedAt: 'desc' }, { id: 'desc' }], take: 512,
    }) : [];
    const trend = summarizeSelfReports(rows, now);
    const context = state.enabled ? await readPersonalSupportContext(db, userId, now) : { goal: 'unspecified', sourceMessageId: null, revision: null, evidence: [], revoked: 0, status: 'unavailable' } as PersonalSupportContext;
    const latestOffer = state.actions[state.actions.length - 1];
    const coolingDown = latestOffer && now.getTime() - Date.parse(latestOffer.offeredAt) < 6 * 3600000;
    const hasActive = state.actions.some(action => active(action) && Date.parse(action.expiresAt) > now.getTime());
    let matchedReference: ModelReference | null = null;
    const referenceEvidence = state.modelReference && context.evidence.find(item => item.sourceId === state.modelReference!.evidence_source.source_message_id
      && createHash('sha256').update(item.quote).digest('hex') === state.modelReference!.evidence_source.source_sha256);
    if (referenceEvidence && state.modelReference) {
      try { matchedReference = validateModelReference(state.modelReference, referenceEvidence, now); } catch { /* 来源、版本或时间失效时不再使用历史模型参考 */ }
    }
    const respectsGoal = !['listen', 'understand'].includes(context.goal) && context.status !== 'declined';
    return { state: { ...state, modelReference: matchedReference }, trend, context, modelReference: matchedReference, suggestionEligible: state.enabled && !trend.stale && !hasActive && !coolingDown && respectsGoal };
  }

  /** 关闭状态不读取本人历史；刷新只返回状态，不自动执行任何活动。 */
  async overview(userId: string): Promise<SupportAgentOverview> {
    const now = this.clock();
    const row = await this.db.healingSession.findFirst({ where: { id: stateId(userId), userId, type: SUPPORT_STATE_TYPE }, select: { metadata: true } });
    return this.overviewFor(this.db, userId, this.readState(row?.metadata, now), now);
  }

  private async save(tx: Prisma.TransactionClient, userId: string, state: SupportAgentState, now: Date): Promise<void> {
    state.revision += 1; state.updatedAt = now.toISOString(); state.requests = state.requests.slice(-64); state.actions = state.actions.slice(-20);
    state.background.events = state.background.events.slice(-32);
    const metadata = JSON.stringify(state);
    if (Buffer.byteLength(metadata, 'utf8') > 65536) throw new AppError('陪伴记录超出上限，本次操作未保存', 409);
    await tx.healingSession.upsert({ where: { id: stateId(userId) },
      create: { id: stateId(userId), userId, type: SUPPORT_STATE_TYPE, title: '持续陪伴设置与执行记录', duration: 0, metadata }, update: { metadata } });
  }

  /** 研究模型只接收已有本人确认的片段，结果须经校验且来源仍有效才落库。 */
  async refreshModelReference(userId: string): Promise<SupportAgentOverview> {
    const before = await this.overview(userId);
    const evidence = before.state.enabled ? before.context.evidence[0] : null;
    if (!evidence) throw new AppError('当前没有本人确认且来源有效的片段', 409);
    const reference = await this.modelProvider(evidence).catch(() => null);
    return this.db.$transaction(async tx => {
      const locked = await tx.$queryRaw<Array<{ id: string }>>`SELECT id FROM users WHERE id = ${userId} AND "isActive" = true FOR UPDATE`;
      if (locked.length !== 1) throw new AppError('账号不可用', 401);
      const now = this.clock();
      const row = await tx.healingSession.findFirst({ where: { id: stateId(userId), userId, type: SUPPORT_STATE_TYPE }, select: { metadata: true } });
      const state = this.readState(row?.metadata, now);
      const context = await readPersonalSupportContext(tx, userId, now);
      if (!state.enabled || !context.evidence.some(item => item.sourceId === evidence.sourceId && item.quote === evidence.quote && item.revision === evidence.revision)) throw new AppError('本人证据已经变更，未保存旧模型解释', 409);
      state.modelReference = reference ? validateModelReference(reference, evidence, now) : null;
      await this.save(tx, userId, state, now);
      return this.overviewFor(tx, userId, state, now);
    });
  }

  /** 后台扫描只返回保留状态的所有者；游标仅用于公平扫描，任务进度和租约都在数据库。 */
  async scanBackgroundUsers(afterId?: string): Promise<Array<{ id: string; userId: string }>> {
    return this.db.healingSession.findMany({ where: { type: SUPPORT_STATE_TYPE }, select: { id: true, userId: true },
      orderBy: { id: 'asc' }, take: 25, ...(afterId ? { cursor: { id: afterId }, skip: 1 } : {}) });
  }

  /** 多实例抢占到期任务；过期租约可恢复，错过时段不补发积压建议。 */
  async claimBackground(userId: string, owner: string): Promise<BackgroundSupportLease | null> {
    return this.db.$transaction(async tx => {
      const users = await tx.$queryRaw<Array<{ id: string }>>`SELECT id FROM users WHERE id = ${userId} AND "isActive" = true FOR UPDATE`;
      if (!users.length) return null;
      const now = this.clock();
      const row = await tx.healingSession.findFirst({ where: { id: stateId(userId), userId, type: SUPPORT_STATE_TYPE }, select: { metadata: true } });
      const state = this.readState(row?.metadata, now); const job = state.background;
      if (!state.enabled || !job.enabled || !job.nextRunAt || Date.parse(job.nextRunAt) > now.getTime()
        || job.lease && Date.parse(job.lease.until) > now.getTime()) return null;
      const localHour = new Date(now.getTime() + 480 * 60000).getUTCHours();
      const quiet = job.quietStart > job.quietEnd ? localHour >= job.quietStart || localHour < job.quietEnd : localHour >= job.quietStart && localHour < job.quietEnd;
      const missed = now.getTime() - Date.parse(job.nextRunAt) > 30 * 60000;
      if (quiet || missed) {
        job.lease = null; job.nextRunAt = nextAllowedSupportTime(quiet ? now : new Date(now.getTime() + job.intervalHours * 3600000), job.quietStart, job.quietEnd).toISOString();
        job.events.push({ at: now.toISOString(), result: quiet ? 'quiet_hours_skipped' : 'missed_window_skipped' });
        await this.save(tx, userId, state, now); return null;
      }
      const token = randomUUID(); job.lease = { owner, token, until: new Date(now.getTime() + 120000).toISOString(), scheduledAt: job.nextRunAt };
      const context = await readPersonalSupportContext(tx, userId, now);
      await this.save(tx, userId, state, now);
      return { userId, owner, token, evidence: context.goal === 'listen' || context.status === 'declined' ? null : context.evidence[0] || null };
    });
  }

  /** HTTP 运算在事务锁外；取消或租约失效后结果不能回写，模型永远不直接启动工具。 */
  async executeBackground(lease: BackgroundSupportLease): Promise<string> {
    let reference: ModelReference | null = null;
    const preflight = await this.overview(lease.userId);
    const currentEvidence = lease.evidence && preflight.context.evidence.some(item => item.sourceId === lease.evidence!.sourceId && item.quote === lease.evidence!.quote && item.revision === lease.evidence!.revision);
    if (lease.evidence && currentEvidence && preflight.state.enabled && preflight.state.background.enabled
      && preflight.state.background.lease?.token === lease.token && preflight.context.goal !== 'listen') {
      reference = await this.modelProvider(lease.evidence).then(value => value ? validateModelReference(value, lease.evidence!, this.clock()) : null).catch(() => null);
    }
    return this.db.$transaction(async tx => {
      const users = await tx.$queryRaw<Array<{ id: string }>>`SELECT id FROM users WHERE id = ${lease.userId} AND "isActive" = true FOR UPDATE`;
      if (!users.length) return 'cancelled';
      const now = this.clock();
      const row = await tx.healingSession.findFirst({ where: { id: stateId(lease.userId), userId: lease.userId, type: SUPPORT_STATE_TYPE }, select: { metadata: true } });
      const state = this.readState(row?.metadata, now); const job = state.background;
      if (!state.enabled || !job.enabled || !job.lease || job.lease.token !== lease.token || job.lease.owner !== lease.owner || Date.parse(job.lease.until) <= now.getTime()) return 'lease_lost';
      const overview = await this.overviewFor(tx, lease.userId, state, now);
      const latestPoint = overview.trend.points[overview.trend.points.length - 1];
      const evidenceKey = `${latestPoint?.id || ''}:${overview.context.sourceMessageId || ''}:${overview.context.revision ?? ''}:${state.preferredTool}`;
      let result = !overview.suggestionEligible ? ['listen', 'understand'].includes(overview.context.goal) ? 'explicit_goal_respected' : 'no_eligible_observation'
        : evidenceKey === job.lastEvidenceKey ? 'evidence_unchanged' : 'offered';
      if (reference && lease.evidence && overview.context.evidence.some(item => item.sourceId === lease.evidence!.sourceId && item.quote === lease.evidence!.quote && item.revision === lease.evidence!.revision)) state.modelReference = reference;
      if (result === 'offered') {
        await this.apply(tx, lease.userId, state, { op: 'propose', requestId: randomUUID(), expectedRevision: state.revision }, now);
        const action = state.actions[state.actions.length - 1]; action.origin = 'background';
        action.expiresAt = new Date(now.getTime() + job.intervalHours * 3600000).toISOString();
        if (job.notifyInApp) {
          await tx.notification.updateMany({ where: { userId: lease.userId, type: 'support_agent', isRead: false }, data: { isRead: true } });
          action.notificationId = `support-agent-offer:${action.id}`;
          await tx.notification.upsert({ where: { id: action.notificationId }, update: {}, create: {
            id: action.notificationId, userId: lease.userId, type: 'support_agent', title: '有一条新的陪伴建议',
            content: '有一个你可以选择或跳过的小活动，回来后可以在疗愈空间查看。', link: '/healing',
          } });
        }
        job.lastEvidenceKey = evidenceKey;
      }
      job.lease = null; job.nextRunAt = nextAllowedSupportTime(new Date(now.getTime() + job.intervalHours * 3600000), job.quietStart, job.quietEnd).toISOString();
      job.events.push({ at: now.toISOString(), result });
      await this.save(tx, lease.userId, state, now);
      return result;
    });
  }

  /** 严格解析命令、校验版本及幂等键，客户端不能写入自造状态或自选资源 URL。 */
  async command(userId: string, input: unknown): Promise<SupportAgentOverview> {
    const parsed = commandSchema.safeParse(input);
    if (!parsed.success) throw new AppError(parsed.error.issues[0]?.message || '请求无效', 400);
    const command = parsed.data;
    const digest = createHash('sha256').update(JSON.stringify(command)).digest('hex');
    return this.db.$transaction(async (tx: Prisma.TransactionClient): Promise<SupportAgentOverview> => {
      const locked = await tx.$queryRaw<Array<{ id: string }>>`SELECT id FROM users WHERE id = ${userId} AND "isActive" = true FOR UPDATE`;
      if (locked.length !== 1) throw new AppError('账号不存在或已停用', 401);
      const now = this.clock();
      const row = await tx.healingSession.findFirst({ where: { id: stateId(userId), userId, type: SUPPORT_STATE_TYPE }, select: { metadata: true } });
      const state = this.readState(row?.metadata, now);
      const replay = state.requests.find(item => item.id === command.requestId);
      if (replay) {
        if (replay.digest !== digest) throw new AppError('请求编号已用于其他操作', 409);
        return { ...await this.overviewFor(tx, userId, state, now), replayed: true };
      }
      if (state.revision !== command.expectedRevision) throw new AppError('记录已更新，请刷新后再操作', 409);
      if (command.op !== 'configure' && !state.enabled) throw new AppError('请先选择开启持续陪伴', 409);
      await this.apply(tx, userId, state, command, now);
      state.requests.push({ id: command.requestId, digest });
      await this.save(tx, userId, state, now);
      return this.overviewFor(tx, userId, state, now);
    });
  }

  private async closeSession(tx: Prisma.TransactionClient, userId: string, action: SupportAgentAction, now: Date): Promise<void> {
    action.endedAt = action.endedAt || now.toISOString();
    if (action.notificationId) await tx.notification.updateMany({ where: { id: action.notificationId, userId, type: 'support_agent' }, data: { isRead: true } });
    if (action.sessionId) {
      const duration = action.startedAt ? Math.max(0, Math.min(action.durationSeconds, Math.floor((now.getTime() - Date.parse(action.startedAt)) / 1000))) : 0;
      await tx.healingSession.updateMany({ where: { id: action.sessionId, userId }, data: {
        duration, completed: action.status === 'completed', metadata: JSON.stringify({ schema: 'support-action.v1', actionId: action.id,
          status: action.status, resourceId: action.resourceId, feedback: action.feedback, receiptSource: 'browser_report', failure: action.failure }),
      } });
    }
  }

  private async apply(tx: Prisma.TransactionClient, userId: string, state: SupportAgentState, command: SupportAgentCommand, now: Date): Promise<void> {
    if (command.op === 'configure') {
      state.enabled = command.enabled;
      state.preferredTool = command.preferredTool;
      if (!state.enabled) {
        state.background.enabled = false; state.background.nextRunAt = null; state.background.lease = null; state.modelReference = null;
        await tx.notification.updateMany({ where: { userId, type: 'support_agent', isRead: false }, data: { isRead: true } });
      }
      if (!state.enabled) for (const action of state.actions.filter(active)) {
        action.status = 'cancelled';
        await this.closeSession(tx, userId, action, now);
      }
      return;
    }
    if (command.op === 'background_settings') {
      if (command.quietStart === command.quietEnd) throw new AppError('安静时段的开始和结束不能相同', 400);
      state.background = { ...state.background, enabled: command.enabled, intervalHours: command.intervalHours,
        notifyInApp: command.notifyInApp, quietStart: command.quietStart, quietEnd: command.quietEnd, nextRunAt: command.enabled ? now.toISOString() : null, lease: null };
      if (!command.enabled || !command.notifyInApp) await tx.notification.updateMany({ where: { userId, type: 'support_agent', isRead: false }, data: { isRead: true } });
      if (!command.enabled) for (const item of state.actions.filter(item => item.origin === 'background' && item.status === 'offered')) { item.status = 'cancelled'; await this.closeSession(tx, userId, item, now); }
      return;
    }
    if (command.op === 'propose') {
      for (const action of state.actions.filter(item => active(item) && Date.parse(item.expiresAt) <= now.getTime())) {
        action.status = 'expired';
        await this.closeSession(tx, userId, action, now);
      }
      const overview = await this.overviewFor(tx, userId, state, now);
      if (!overview.suggestionEligible) throw new AppError('请先更新心情打卡，或稍后再查看新的建议', 409);
      const latest = overview.trend.points[overview.trend.points.length - 1]!;
      const lastUnhelpful = [...state.actions].reverse().find(action => action.feedback === 'unhelpful');
      const feedbackPreference = lastUnhelpful ? lastUnhelpful.tool === 'music' ? 'breathing' : 'music' : null;
      const tool = state.preferredTool !== 'none' ? state.preferredTool : feedbackPreference || (['tired', 'sad'].includes(latest.mood) ? 'music' : 'breathing');
      const spec = SUPPORT_TOOLS[tool];
      state.actions.push({ id: randomUUID(), tool, ...spec, origin: 'page',
        ...(overview.context.sourceMessageId ? { contextMessageId: overview.context.sourceMessageId, contextRevision: overview.context.revision! } : {}), reason: state.preferredTool !== 'none'
        ? `你选择了${tool === 'music' ? '自然音' : '呼吸练习'}偏好；最近一次本人打卡为 ${latest.score}/10。`
        : feedbackPreference ? `上次你反馈${lastUnhelpful!.tool === 'music' ? '自然音' : '呼吸练习'}没有帮助，这次换一个可选活动；你也可以修改偏好。`
        : `最近一次本人打卡为 ${latest.score}/10，提供一个可以尝试或跳过的小活动。`,
        status: 'offered', offeredAt: now.toISOString(), expiresAt: new Date(now.getTime() + 15 * 60000).toISOString() });
      return;
    }
    const action = state.actions.find(item => item.id === command.actionId);
    if (!action) throw new AppError('活动不存在或已不在保留记录内', 404);
    if (command.op === 'cancel') {
      if (!active(action)) throw new AppError('活动已经结束', 409);
      action.status = 'cancelled';
      await this.closeSession(tx, userId, action, now);
      return;
    }
    if (command.op === 'feedback') {
      if (action.status !== 'completed') throw new AppError('请在活动完成后反馈', 409);
      action.feedback = command.value;
      await this.closeSession(tx, userId, action, now);
      return;
    }
    if (Date.parse(action.expiresAt) <= now.getTime()) throw new AppError('活动已过期，请刷新后查看新的建议', 409);
    if (command.op === 'decline') {
      if (action.status !== 'offered') throw new AppError('这条建议已经处理', 409);
      action.status = 'declined'; await this.closeSession(tx, userId, action, now); return;
    }
    if (command.op === 'accept') {
      if (action.status !== 'offered') throw new AppError('活动已接收，请勿重复启动', 409);
      const context = await readPersonalSupportContext(tx, userId, now);
      if (['listen', 'understand'].includes(context.goal) || context.status === 'declined' || (action.contextMessageId && (action.contextMessageId !== context.sourceMessageId || action.contextRevision !== context.revision))) throw new AppError('本人目标或证据已更新，请重新查看建议', 409);
      const session = await tx.healingSession.create({ data: { userId, type: action.tool, title: action.title, duration: 0,
        metadata: JSON.stringify({ schema: 'support-action.v1', actionId: action.id, status: 'accepted' }) } });
      action.sessionId = session.id; action.status = 'accepted'; action.acceptedAt = now.toISOString(); action.executionToken = randomUUID();
      if (action.notificationId) await tx.notification.updateMany({ where: { id: action.notificationId, userId, type: 'support_agent' }, data: { isRead: true } });
      return;
    }
    if (command.executionToken !== action.executionToken) throw new AppError('执行回执与活动不匹配', 409);
    if (command.op === 'started') {
      if (action.status !== 'accepted') throw new AppError('活动已启动或已结束', 409);
      action.status = 'running'; action.startedAt = now.toISOString();
      return;
    }
    if (command.op === 'fail') {
      if (!['accepted', 'running'].includes(action.status)) throw new AppError('活动已经结束', 409);
      action.status = 'failed'; action.failure = command.reason;
      await this.closeSession(tx, userId, action, now); return;
    }
    if (command.op === 'complete') {
      if (action.status !== 'running' || !action.startedAt || now.getTime() - Date.parse(action.startedAt) < action.durationSeconds * 1000) {
        throw new AppError('活动尚未完成，未记录为完成', 409);
      }
      action.status = 'completed'; await this.closeSession(tx, userId, action, now);
    }
  }
}

export default new SupportAgentService();
