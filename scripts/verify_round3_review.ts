/** 独立业务复核：有界内存事务夹具，不冒充PostgreSQL锁或真人效果验证。 */
import assert from 'node:assert/strict';
import { createHash, randomUUID } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import prisma from '../server/src/config/database';
import { createAgentFixture } from './verify_support_agent';
import { invalidateSupportAgentOffers, nextAllowedSupportTime, SupportAgentService } from '../server/src/services/support-agent.service';
import { SupportFeedbackService } from '../server/src/services/support-feedback.service';

async function main(): Promise<void> {
  const passed: string[] = [];
  const test = async (label: string, work: () => Promise<void> | void): Promise<void> => { await work(); passed.push(label); };
  const run = async (service: SupportAgentService, op: string, extra: Record<string, unknown> = {}): Promise<void> => {
    const before = await service.overview('patient-a');
    await service.command('patient-a', { op, requestId: randomUUID(), expectedRevision: before.state.revision, ...extra });
  };
  await test('24h quiet排程从北京时间23点顺延到次日8点，此后不永久困在夜间', () => {
    const night = new Date('2026-11-15T15:00:00Z');
    const initial = nextAllowedSupportTime(night, 22, 8);
    assert.equal(initial.toISOString(), '2026-11-16T00:00:00.000Z');
    for (let day = 1; day <= 4; day++) {
      const next = nextAllowedSupportTime(new Date(initial.getTime() + day * 86400000), 22, 8);
      assert.equal(new Date(next.getTime() + 480 * 60000).getUTCHours(), 8);
    }
  });
  await test('白天quiet区间及跨午夜区间均不会提前或漏过边界', () => {
    assert.equal(nextAllowedSupportTime(new Date('2026-11-15T04:30:00Z'), 12, 14).toISOString(), '2026-11-15T06:00:00.000Z');
    assert.equal(nextAllowedSupportTime(new Date('2026-11-15T00:00:00Z'), 22, 8).toISOString(), '2026-11-15T00:00:00.000Z');
    assert.equal(nextAllowedSupportTime(new Date('2026-11-15T14:00:00Z'), 22, 8).toISOString(), '2026-11-16T00:00:00.000Z');
  });
  await test('无旧context标识的offer也必须尊重新增understand目标', async () => {
    const f = createAgentFixture();
    await run(f.service, 'configure', { enabled: true, preferredTool: 'none' });
    await run(f.service, 'propose');
    const action = (await f.service.overview('patient-a')).state.actions[0];
    assert.equal(action.contextMessageId, undefined);
    f.db.conversation.findFirst = async () => ({ id: 'new-conversation' });
    f.db.message = { findMany: async () => [{ id: 'assistant-new', role: 'assistant', userId: null, content: '理解', createdAt: new Date('2026-11-15T03:00:00Z'),
      sentiment: JSON.stringify({ support_state: { schema_version: 'support.v1', session_id: 'new-conversation', revision: 1, goal: 'understand', goal_source: 'user_feedback', declined: false, items: [], feedback_log: [] } }) }] };
    assert.equal((await f.service.overview('patient-a')).context.goal, 'understand');
    await assert.rejects(run(f.service, 'accept', { actionId: action.id }));
    assert.equal([...f.sessions.values()].filter(row => row.type === 'breathing' || row.type === 'music').length, 0);
  });
  await test('同一后台租约重放只留一份通知；证据撤回同步取消offer并令提醒失效', async () => {
    const f = createAgentFixture();
    const notifications = new Map<string, any>();
    f.db.notification = {
      updateMany: async ({ where, data }: any) => { let count = 0; for (const [id, row] of notifications) {
        if (Object.entries(where).every(([key, value]) => row[key] === value)) { notifications.set(id, { ...row, ...data }); count++; }
      } return { count }; },
      upsert: async ({ where, create, update }: any) => { const row = notifications.has(where.id) ? { ...notifications.get(where.id), ...update } : { isRead: false, ...create }; notifications.set(where.id, row); return row; },
    };
    await run(f.service, 'configure', { enabled: true, preferredTool: 'none' });
    await run(f.service, 'background_settings', { enabled: true, notifyInApp: true, intervalHours: 6, quietStart: 22, quietEnd: 8 });
    const lease = await f.service.claimBackground('patient-a', 'review-worker');
    assert.ok(lease);
    assert.equal(await f.service.executeBackground(lease), 'offered');
    assert.equal(await f.service.executeBackground(lease), 'lease_lost');
    assert.equal(notifications.size, 1);
    assert.equal([...notifications.values()].filter(row => !row.isRead).length, 1);
    await f.db.$transaction(async (tx: any) => invalidateSupportAgentOffers(tx, 'patient-a'));
    const after = await f.service.overview('patient-a');
    assert.equal(after.state.actions[0].status, 'cancelled');
    assert.equal(after.state.background.lease, null);
    assert.equal([...notifications.values()].filter(row => !row.isRead).length, 0);
  });
  await test('重复本人反馈返回同一revision时不会取消随后产生的新合法offer', async () => {
    const f = createAgentFixture();
    await run(f.service, 'configure', { enabled: true, preferredTool: 'none' });
    const support = { schema_version: 'support.v1', session_id: 'replay-conversation', revision: 1, goal: 'act', goal_source: 'user_feedback',
      declined: false, items: [], feedback_log: [{ feedback_id: 'already-applied-feedback', action: 'confirm' }] };
    const message = { id: 'replay-assistant', conversationId: 'replay-conversation', role: 'assistant', userId: null, content: '理解',
      createdAt: new Date('2026-11-15T03:00:00Z'), sentiment: JSON.stringify({ support_state: support }) };
    f.db.conversation.findFirst = async () => ({ id: 'replay-conversation' });
    f.db.message = { findFirst: async () => message, findMany: async () => [message], updateMany: async ({ data }: any) => { message.sentiment = data.sentiment; return { count: 1 }; } };
    await run(f.service, 'propose');
    const before = await f.service.overview('patient-a');
    const feedback = new SupportFeedbackService(f.db, async () => structuredClone(support)); // 代表Python已识别同feedback_id的幂等响应。
    await feedback.submit('replay-conversation', 'patient-a', 'replay-assistant', { feedback_id: 'already-applied-feedback', expected_revision: 0, action: 'confirm', item_id: 'previous-item' });
    const after = await f.service.overview('patient-a');
    assert.equal(after.state.revision, before.state.revision);
    assert.equal(after.state.actions[0].status, 'offered');
  });
  await test('模型provider在事务外运行；其间撤回租约后旧结果不能产生offer', async () => {
    const f = createAgentFixture();
    const quote = '我担心考试';
    const source = { id: 'source-user', role: 'user', userId: 'patient-a', content: quote, sentiment: null, createdAt: new Date('2026-11-15T02:59:59Z') };
    const support = { schema_version: 'support.v1', session_id: 'provider-conversation', revision: 1, goal: 'act', goal_source: 'user_feedback', declined: false,
      items: [{ id: 'thought1', quote, source_id: source.id, source_sha256: createHash('sha256').update(quote).digest('hex'), start: 0, end: [...quote].length, status: 'confirmed', origin: 'user_message' }], feedback_log: [] };
    f.db.conversation.findFirst = async () => ({ id: 'provider-conversation' });
    f.db.message = { findMany: async () => [{ id: 'provider-assistant', role: 'assistant', userId: null, content: '理解', createdAt: new Date('2026-11-15T03:00:00Z'), sentiment: JSON.stringify({ support_state: support }) }, source] };
    let depth = 0, calls = 0;
    const originalTransaction = f.db.$transaction;
    f.db.$transaction = async (operation: (tx: any) => Promise<unknown>) => originalTransaction(async (tx: any) => {
      depth++; try { return await operation(tx); } finally { depth--; }
    });
    const service = new SupportAgentService(f.db, () => new Date('2026-11-15T03:00:00Z'), async () => {
      assert.equal(depth, 0); calls++;
      await f.db.$transaction(async (tx: any) => invalidateSupportAgentOffers(tx, 'patient-a'));
      return null;
    });
    await run(service, 'configure', { enabled: true, preferredTool: 'none' });
    await run(service, 'background_settings', { enabled: true, intervalHours: 6, quietStart: 22, quietEnd: 8 });
    const lease = await service.claimBackground('patient-a', 'provider-review-worker');
    assert.ok(lease?.evidence);
    assert.equal(await service.executeBackground(lease), 'lease_lost');
    assert.equal(calls, 1);
    assert.equal((await service.overview('patient-a')).state.actions.length, 0);
  });
  const output = process.env.ROUND3_REVIEW_OUTPUT || 'D:/Develop/AIC/_国赛优化/round3/data';
  fs.mkdirSync(output, { recursive: true });
  const result = { status: 'passed', cases: passed.length, passed, database: 'independent in-memory business fixture; no PostgreSQL concurrency claim',
    human_participants: 0, clinical_effectiveness_validated: false };
  fs.writeFileSync(path.join(output, 'node_independent_review.json'), JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result, null, 2));
}

main().catch((failure: unknown) => { console.error(failure); process.exitCode = 1; }).finally(async () => { await prisma.$disconnect(); });
