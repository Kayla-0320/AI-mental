import { createHash } from 'node:crypto';
import { Prisma } from '@prisma/client';
import { z } from 'zod';

export interface PersonalEvidence {
  id: string; quote: string; sourceId: string; sourceSha256: string; origin: 'user_message' | 'user_correction';
  conversationId: string; stateMessageId: string; revision: number;
  start: number; end: number;
  observedAt: string;
}
export interface PersonalSupportContext {
  goal: 'unspecified' | 'listen' | 'understand' | 'act'; sourceMessageId: string | null; revision: number | null;
  evidence: PersonalEvidence[]; revoked: number; status: 'unavailable' | 'current' | 'declined' | 'awaiting_reply' | 'existing_escalation';
}
export const modelReferenceSchema = z.object({
  available: z.boolean(), task_id: z.literal('sentiment_polarity_3'),
  label_schema: z.tuple([z.literal('negative'), z.literal('neutral'), z.literal('positive')]),
  model_version: z.string().max(160), artifact_sha256: z.string().max(64),
  calibration: z.object({ status: z.enum(['calibrated', 'uncalibrated']), method: z.string().max(120) }).strict(),
  evidence_source: z.object({ source_message_id: z.string().max(160), source_sha256: z.string().regex(/^[a-f0-9]{64}$/) }).strict(),
  probabilities: z.array(z.number().finite().min(0).max(1)).max(3), observed_at: z.string().datetime({ offset: true }),
  limitations: z.array(z.string().max(500)).max(20), reason: z.string().max(500).optional(),
}).strict();
export type ModelReference = z.infer<typeof modelReferenceSchema>;
export type ModelReferenceProvider = (evidence: PersonalEvidence) => Promise<ModelReference | null>;
const sha = (text: string): string => createHash('sha256').update(text).digest('hex');
const object = (value: unknown): Record<string, unknown> | null => value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null;

/** 从最近活动会话的最新助手状态重算本人证据；撤销、删除和新话题不会复活旧内容。 */
export async function readPersonalSupportContext(db: Prisma.TransactionClient, userId: string, now: Date): Promise<PersonalSupportContext> {
  const empty: PersonalSupportContext = { goal: 'unspecified', sourceMessageId: null, revision: null, evidence: [], revoked: 0, status: 'unavailable' };
  const conversation = await db.conversation.findFirst({ where: { userId, isActive: true, updatedAt: { lte: now } }, orderBy: [{ updatedAt: 'desc' }, { id: 'desc' }], select: { id: true } });
  if (!conversation) return empty;
  const rows = await db.message.findMany({ where: { conversationId: conversation.id, createdAt: { lte: now } }, orderBy: [{ createdAt: 'desc' }, { id: 'desc' }], take: 32,
    select: { id: true, role: true, userId: true, content: true, sentiment: true, createdAt: true } });
  const latest = rows[0];
  if (!latest || latest.role !== 'assistant') return { ...empty, status: 'awaiting_reply' };
  if (!latest.sentiment || Buffer.byteLength(latest.sentiment, 'utf8') > 65536) return empty;
  let state: Record<string, unknown> | null = null;
  try {
    const metadata = object(JSON.parse(latest.sentiment));
    // ⚠️ 比赛 Demo 为模拟实现，生产环境需接入真实危机干预流程。
    // 仅尊重既有高风险/危机标记，不修改分级阈值、不替代原有升级链路。
    if (metadata?.risk_level === 'high' || metadata?.risk_level === 'crisis') return { ...empty, sourceMessageId: latest.id, goal: 'listen', status: 'existing_escalation' };
    state = object(metadata?.support_state);
  } catch { return empty; }
  if (!state || state.schema_version !== 'support.v1' || state.session_id !== conversation.id || !Number.isInteger(state.revision) || (state.revision as number) < 0
    || !Array.isArray(state.items) || state.items.length > 4 || !Array.isArray(state.feedback_log) || state.feedback_log.length > 24) return empty;
  const revision = state.revision as number;
  if (state.declined === true) return { ...empty, sourceMessageId: latest.id, revision, goal: 'listen', status: 'declined' };
  const evidence: PersonalEvidence[] = [];
  let revoked = 0;
  for (const candidate of state.items) {
    const item = object(candidate);
    if (!item) continue;
    if (item.status === 'rejected') { revoked++; continue; }
    if (!['confirmed', 'corrected'].includes(String(item.status)) || typeof item.quote !== 'string' || !item.quote.trim() || [...item.quote].length > 240
      || typeof item.id !== 'string' || item.id.length > 80 || typeof item.source_id !== 'string' || item.source_id.length > 160
      || item.source_sha256 !== sha(item.quote) && item.origin === 'user_correction') continue;
    let valid = false;
    if (item.origin === 'user_message') {
      const source = rows.find(row => row.id === item.source_id && row.role === 'user' && row.userId === userId);
      valid = !!source && sha(source.content) === item.source_sha256 && Number.isInteger(item.start) && Number.isInteger(item.end)
        && (item.start as number) >= 0 && (item.end as number) <= [...source.content].length
        && (item.end as number) - (item.start as number) === [...item.quote].length
        && [...source.content].slice(item.start as number, item.end as number).join('') === item.quote;
    } else if (item.origin === 'user_correction') {
      const log = state.feedback_log.map(object).find(entry => entry?.feedback_id === (item.source_id as string).replace(/^feedback:/, '')
        && entry?.action === 'correct' && entry?.item_id === item.id && entry?.text === item.quote);
      valid = item.source_id.startsWith('feedback:') && !!log && item.start === 0 && item.end === [...item.quote].length;
    }
    if (valid) evidence.push({ id: item.id, quote: item.quote, sourceId: item.source_id, sourceSha256: String(item.source_sha256),
      origin: item.origin as PersonalEvidence['origin'], conversationId: conversation.id, stateMessageId: latest.id, revision,
      start: item.start as number, end: item.end as number,
      observedAt: (item.origin === 'user_message' ? rows.find(row => row.id === item.source_id)!.createdAt : latest.createdAt).toISOString() });
  }
  const goal = ['listen', 'understand', 'act'].includes(String(state.goal)) && ['user_feedback', 'explicit_message'].includes(String(state.goal_source))
    ? state.goal as PersonalSupportContext['goal'] : 'unspecified';
  return { goal, sourceMessageId: latest.id, revision, evidence, revoked, status: 'current' };
}

/** 严格核对任务、顺序、工件摘要、校准及来源；不可用结果不伪装为现有五类模型。 */
export function validateModelReference(payload: unknown, evidence: PersonalEvidence, now: Date): ModelReference {
  const value = modelReferenceSchema.parse(payload);
  if (value.evidence_source.source_message_id !== evidence.sourceId || value.evidence_source.source_sha256 !== sha(evidence.quote)) throw new Error('model_source_mismatch');
  if (Date.parse(value.observed_at) !== Date.parse(evidence.observedAt)) throw new Error('model_observation_time_mismatch');
  if (value.available && (Date.parse(value.observed_at) > now.getTime() + 60000 || now.getTime() - Date.parse(value.observed_at) > 48 * 3600000)) throw new Error('model_observation_stale');
  if (value.available && (!value.model_version || !/^[a-f0-9]{64}$/.test(value.artifact_sha256) || value.probabilities.length !== 3
    || Math.abs(value.probabilities.reduce((sum, probability) => sum + probability, 0) - 1) > 1e-5)) throw new Error('model_artifact_or_probabilities_invalid');
  if (value.available && process.env.SUPPORT_AGENT_MODEL_SHA256 && value.artifact_sha256 !== process.env.SUPPORT_AGENT_MODEL_SHA256) throw new Error('model_artifact_not_deployed');
  if (value.available && process.env.SUPPORT_AGENT_MODEL_VERSION && value.model_version !== process.env.SUPPORT_AGENT_MODEL_VERSION) throw new Error('model_version_not_deployed');
  if (value.available && value.calibration.status !== 'calibrated') throw new Error('model_uncalibrated_reference');
  if (!value.available && value.probabilities.length !== 0) throw new Error('model_unavailable_with_probabilities');
  return value;
}

/** 用户开启支持助手后发送已确认原话，正式入口不依赖研究令牌。 */
export async function fetchPlatformModelReference(evidence: PersonalEvidence): Promise<ModelReference | null> {
  if (process.env.SUPPORT_AGENT_MODEL_REFERENCE === '0') return null;
  const base = process.env.ALGORITHM_API_URL || 'http://127.0.0.1:8001';
  const controller = new AbortController(); const timer = setTimeout(() => controller.abort(), 6000);
  try {
    const response = await fetch(`${base}/api/v1/innovation/observation`, { method: 'POST', signal: controller.signal,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: evidence.quote, source_message_id: evidence.sourceId,
        source_sha256: sha(evidence.quote), observed_at: evidence.observedAt }) });
    if (!response.ok || Number(response.headers.get('content-length') || 0) > 16384) throw new Error('research_model_unavailable');
    const raw = await response.text(); if (Buffer.byteLength(raw, 'utf8') > 16384) throw new Error('research_model_response_too_large');
    return validateModelReference(JSON.parse(raw), evidence, new Date());
  } finally { clearTimeout(timer); }
}
