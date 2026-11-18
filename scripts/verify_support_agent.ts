/** 持续陪伴的业务与 HTTP 契约验证；数据库为带回滚和串行队列的内存替身。 */
import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import express from '../server/node_modules/express';
import jwt from '../server/node_modules/jsonwebtoken';
import prisma from '../server/src/config/database';
import { config } from '../server/src/config';
import { errorHandler } from '../server/src/middlewares/errorHandler';
import { createSupportAgentRouter } from '../server/src/routes/support-agent.routes';
import { SupportAgentService, SUPPORT_STATE_TYPE, summarizeSelfReports } from '../server/src/services/support-agent.service';

/** 构造明确不冒充 PostgreSQL 的隔离测试存储。 */
export function createAgentFixture(): { db: any; service: SupportAgentService; advance: (seconds: number) => void; sessions: Map<string, any>; moodReads: () => number } {
  let now = new Date('2026-11-15T03:00:00Z');
  const sessions = new Map<string, any>();
  const moods = [1, 2, 3].map((n) => ({ id: `mood-${n}`, userId: 'patient-a', mood: 'anxious', score: 4 + n, recordedAt: new Date(now.getTime() - (4 - n) * 3600000) }));
  let moodReads = 0;
  let queue = Promise.resolve();
  const db: any = {
    conversation: { findFirst: async () => null },
    notification: { updateMany: async () => ({ count: 0 }) },
    $queryRaw: async (_strings: TemplateStringsArray, userId: string) => ['patient-a', 'patient-b'].includes(userId) ? [{ id: userId }] : [],
    moodRecord: { findMany: async ({ where }: any) => { moodReads++; return moods.filter(row => row.userId === where.userId && row.recordedAt >= where.recordedAt.gte && row.recordedAt <= where.recordedAt.lte); } },
    healingSession: {
      findFirst: async ({ where }: any) => {
        const row = sessions.get(where.id);
        return row && row.userId === where.userId && (!where.type || row.type === where.type) ? structuredClone(row) : null;
      },
      create: async ({ data }: any) => { const value = { id: randomUUID(), completed: false, ...data }; sessions.set(value.id, structuredClone(value)); return value; },
      upsert: async ({ where, create, update }: any) => {
        const value = sessions.has(where.id) ? { ...sessions.get(where.id), ...update } : create;
        sessions.set(where.id, structuredClone(value)); return value;
      },
      updateMany: async ({ where, data }: any) => {
        const row = sessions.get(where.id);
        if (!row || row.userId !== where.userId) return { count: 0 };
        sessions.set(where.id, { ...row, ...data }); return { count: 1 };
      },
    },
    $transaction: async (operation: (tx: any) => Promise<unknown>) => {
      const previous = queue;
      let release!: () => void;
      queue = new Promise<void>(resolve => { release = resolve; });
      await previous;
      const snapshot = structuredClone([...sessions]);
      try { return await operation(db); }
      catch (error) { sessions.clear(); snapshot.forEach(([key, value]: [string, any]) => sessions.set(key, value)); throw error; }
      finally { release(); }
    },
  };
  return { db, sessions, service: new SupportAgentService(db, () => new Date(now)),
    advance: (seconds: number): void => { now = new Date(now.getTime() + seconds * 1000); }, moodReads: () => moodReads };
}

async function main(): Promise<void> {
  const labels: string[] = [];
  const test = async (label: string, operation: () => void | Promise<void>): Promise<void> => { await operation(); labels.push(label); };
  const f = createAgentFixture();
  let state = await f.service.overview('patient-a');
  const body = (op: string, extra: Record<string, unknown> = {}): Record<string, unknown> => ({ requestId: randomUUID(), expectedRevision: state.state.revision, op, ...extra });
  const update = async (op: string, extra: Record<string, unknown> = {}): Promise<void> => { state = await f.service.command('patient-a', body(op, extra)); };
  await test('默认关闭且关闭时不读取自评历史', () => { assert.equal(state.state.enabled, false); assert.equal(f.moodReads(), 0); });
  await test('未加入不能提议', async () => { await assert.rejects(f.service.command('patient-a', body('propose'))); });
  await test('客户端不能提交userId或自造状态', async () => {
    await assert.rejects(f.service.command('patient-a', body('configure', { enabled: true, preferredTool: 'none', userId: 'patient-b' })));
    await assert.rejects(f.service.command('patient-a', body('configure', { enabled: true, preferredTool: 'none', state: {} })));
  });
  const join = body('configure', { enabled: true, preferredTool: 'breathing' });
  state = await f.service.command('patient-a', join);
  await test('加入持久化且只读本人自评', () => { assert.equal(state.trend.observations, 3); assert.equal(state.trend.mean, 6); assert.equal(state.trend.change, 2); });
  await test('相同请求幂等重放', async () => { const replay = await f.service.command('patient-a', join); assert.equal(replay.replayed, true); assert.equal(replay.state.revision, 1); });
  await test('同幂等键不同载荷拒绝', async () => { await assert.rejects(f.service.command('patient-a', { ...join, preferredTool: 'music' })); });
  await test('过期版本拒绝', async () => { await assert.rejects(f.service.command('patient-a', { ...join, requestId: randomUUID() })); });
  await test('刷新重建Service后仍读持久状态', async () => { const fresh = new SupportAgentService(f.db); assert.equal((await fresh.overview('patient-a')).state.enabled, true); });
  await test('其他本人账号看不到A的状态与历史', async () => { const other = await f.service.overview('patient-b'); assert.equal(other.state.actions.length, 0); assert.equal(other.trend.observations, 0); });
  await test('主动提议来自白名单且没有启动真实活动', async () => { await update('propose'); assert.equal(state.state.actions[0].tool, 'breathing'); assert.equal(f.sessions.size, 1); });
  const actionId = state.state.actions[0].id;
  await test('他人不能接受A的活动', async () => { const other = await f.service.command('patient-b', { op: 'configure', requestId: randomUUID(), expectedRevision: 0, enabled: true, preferredTool: 'none' });
    await assert.rejects(f.service.command('patient-b', { op: 'accept', requestId: randomUUID(), expectedRevision: other.state.revision, actionId })); });
  await test('并发接受只生成一个真实活动', async () => {
    const a = body('accept', { actionId }); const b = body('accept', { actionId });
    const results = await Promise.allSettled([f.service.command('patient-a', a), f.service.command('patient-a', b)]);
    assert.equal(results.filter(item => item.status === 'fulfilled').length, 1);
    state = await f.service.overview('patient-a');
    assert.equal([...f.sessions.values()].filter(row => row.type === 'breathing').length, 1);
  });
  const token = state.state.actions[0].executionToken;
  await test('回执不能用其他执行令牌', async () => { await assert.rejects(f.service.command('patient-a', body('started', { actionId, executionToken: randomUUID() }))); });
  await test('没有启动不能宣告完成', async () => { await assert.rejects(f.service.command('patient-a', body('complete', { actionId, executionToken: token }))); });
  await update('started', { actionId, executionToken: token });
  await test('时长不足不能宣告完成', async () => { await assert.rejects(f.service.command('patient-a', body('complete', { actionId, executionToken: token }))); });
  f.advance(29);
  await test('运行达到时长后持久化真实会话完成', async () => { await update('complete', { actionId, executionToken: token }); const row = [...f.sessions.values()].find(item => item.type === 'breathing'); assert.equal(row.completed, true); assert.equal(row.duration, 28); });
  await test('本人反馈独立保存且不改变活动结束时刻', async () => { const endedAt = state.state.actions[0].endedAt; f.advance(30); await update('feedback', { actionId, value: 'unhelpful' }); assert.equal(state.state.actions[0].feedback, 'unhelpful'); assert.equal(state.state.actions[0].endedAt, endedAt); });
  await test('终态不能再次启动', async () => { await assert.rejects(f.service.command('patient-a', body('started', { actionId, executionToken: token }))); });
  await test('冷却期内不重复提议', async () => { await assert.rejects(f.service.command('patient-a', body('propose'))); });
  f.advance(6 * 3600);
  await update('configure', { enabled: true, preferredTool: 'music' });
  await update('propose');
  const musicId = state.state.actions[1].id;
  await update('accept', { actionId: musicId });
  await test('播放失败保持未完成', async () => { await update('fail', { actionId: musicId, executionToken: state.state.actions[1].executionToken, reason: 'playback_failed' }); const row = [...f.sessions.values()].find(item => item.type === 'music'); assert.equal(row.completed, false); assert.equal(JSON.parse(row.metadata).status, 'failed'); });
  f.advance(6 * 3600);
  await update('propose');
  const cancelId = state.state.actions[2].id;
  await update('accept', { actionId: cancelId });
  await test('退出立即取消活动且关闭历史读取', async () => { await update('configure', { enabled: false, preferredTool: 'music' }); assert.equal(state.state.actions[2].status, 'cancelled'); const before = f.moodReads(); await f.service.overview('patient-a'); assert.equal(f.moodReads(), before); });
  await test('少于3条不输出趋势', () => { const value = summarizeSelfReports([{ id: 'x', mood: 'sad', score: 3, recordedAt: new Date('2026-11-15') }], new Date('2026-11-16')); assert.equal(value.mean, null); assert.equal(value.change, null); });
  await test('无效分数与未来记录不进入追踪', () => { const value = summarizeSelfReports([{ id: 'x', mood: 'sad', score: 30, recordedAt: new Date('2026-11-15') }, { id: 'future', mood: 'happy', score: 8, recordedAt: new Date('2027-01-01') }], new Date('2026-11-16')); assert.equal(value.observations, 0); });
  await test('旧自评不触发新建议', async () => { f.advance(72 * 3600); await update('configure', { enabled: true, preferredTool: 'none' }); assert.equal(state.suggestionEligible, false); });
  await test('保留记录有界', async () => { for (let n = 0; n < 70; n++) await update('configure', { enabled: false, preferredTool: 'none' }); assert.equal(state.state.requests.length, 64); assert.ok(Buffer.byteLength(JSON.stringify(state.state)) < 65536); });
  await test('损坏状态不被静默重置覆盖', async () => { const row = [...f.sessions.values()].find(item => item.type === SUPPORT_STATE_TYPE && item.userId === 'patient-b'); row.metadata = 'bad'; await assert.rejects(f.service.overview('patient-b')); });
  await test('用户已删除时事务不写入', async () => { await assert.rejects(f.service.command('deleted', { op: 'configure', requestId: randomUUID(), expectedRevision: 0, enabled: true, preferredTool: 'none' })); });

  const originalFind = prisma.user.findUnique;
  (prisma.user.findUnique as any) = async () => ({ id: 'patient-a', role: 'PATIENT', isActive: true });
  const app = express(); app.use(express.json()); app.use('/api/support-agent', createSupportAgentRouter(f.service)); app.use(errorHandler);
  const server = app.listen(0, '127.0.0.1');
  await new Promise<void>(resolve => server.on('listening', resolve));
  const address = server.address() as { port: number };
  const endpoint = `http://127.0.0.1:${address.port}/api/support-agent`;
  try {
    await test('HTTP无认证返回401', async () => { assert.equal((await fetch(endpoint)).status, 401); });
    await test('HTTP非法令牌返回401', async () => { assert.equal((await fetch(endpoint, { headers: { Authorization: 'Bearer invalid' } })).status, 401); });
    const token = jwt.sign({ userId: 'patient-a', role: 'PATIENT' }, config.jwt.secret, { expiresIn: '1m' });
    await test('HTTP有效认证读取本人状态', async () => { const response = await fetch(endpoint, { headers: { Authorization: `Bearer ${token}` } }); assert.equal(response.status, 200); assert.equal((await response.json() as any).data.state.enabled, false); });
    await test('HTTP注入用户字段返回400', async () => { const response = await fetch(`${endpoint}/commands`, { method: 'POST', headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' }, body: JSON.stringify({ ...body('configure', { enabled: true, preferredTool: 'none' }), userId: 'patient-b' }) }); assert.equal(response.status, 400); });
    await test('HTTP顾问角色不能进入患者Agent', async () => { (prisma.user.findUnique as any) = async () => ({ id: 'consultant', role: 'CONSULTANT', isActive: true }); const response = await fetch(endpoint, { headers: { Authorization: `Bearer ${token}` } }); assert.equal(response.status, 403); });
  } finally { prisma.user.findUnique = originalFind; await new Promise<void>(resolve => server.close(() => resolve())); }
  const report = { status: 'passed', cases: labels.length, database: 'in-memory transactional fixture; see separate PostgreSQL evidence', labels };
  const output = process.env.ROUND2_AGENT_OUTPUT || 'D:/Develop/AIC/_国赛优化/round2/agent';
  fs.mkdirSync(output, { recursive: true }); fs.writeFileSync(path.join(output, 'business-verification.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
}
if (require.main === module) main().catch((error: unknown) => { console.error(error); process.exitCode = 1; });
