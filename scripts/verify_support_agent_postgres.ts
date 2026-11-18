/** 仅对明确命名的隔离库执行真实 PostgreSQL 事务验证，不触碰用户原有数据库。 */
import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import prisma from '../server/src/config/database';
import { SupportAgentService, SUPPORT_STATE_TYPE } from '../server/src/services/support-agent.service';
import healingService from '../server/src/services/healing.service';
import { HealthReportService } from '../server/src/services/extra.service';

/** 在专用隔离库中创建带 round2 前缀的测试用户和自评，供业务及浏览器验证复用。 */
export async function createPostgresAgentFixture(): Promise<{ userId: string; otherId: string; service: SupportAgentService; advance: (seconds: number) => void; cleanup: () => Promise<void> }> {
  const url = new URL(process.env.DATABASE_URL || '');
  assert.equal(url.hostname, '127.0.0.1'); assert.equal(url.port, '55432'); assert.equal(url.pathname, '/aic_round2');
  const stamp = randomUUID();
  const user = await prisma.user.create({ data: { email: `round2-agent-${stamp}@example.invalid`, password: 'test-account-no-login', nickname: '第二轮隔离验证', role: 'PATIENT' } });
  const other = await prisma.user.create({ data: { email: `round2-agent-other-${stamp}@example.invalid`, password: 'test-account-no-login', nickname: '第二轮隔离验证B', role: 'PATIENT' } });
  const now = new Date();
  let offsetSeconds = 0;
  await prisma.moodRecord.createMany({ data: [5, 4, 6].map((score, n) => ({ userId: user.id, mood: 'anxious', score, recordedAt: new Date(now.getTime() - (3 - n) * 3600000) })) });
  return { userId: user.id, otherId: other.id, service: new SupportAgentService(prisma, () => new Date(Date.now() + offsetSeconds * 1000)),
    advance: (seconds: number): void => { offsetSeconds += seconds; },
    cleanup: async (): Promise<void> => { await prisma.user.deleteMany({ where: { id: { in: [user.id, other.id] } } }); await prisma.$disconnect(); } };
}

async function main(): Promise<void> {
  const f = await createPostgresAgentFixture();
  const labels: string[] = [];
  const test = async (name: string, operation: () => Promise<void>): Promise<void> => { await operation(); labels.push(name); };
  let state = await f.service.overview(f.userId);
  const body = (op: string, extra: Record<string, unknown> = {}): Record<string, unknown> => ({ op, requestId: randomUUID(), expectedRevision: state.state.revision, ...extra });
  const run = async (op: string, extra: Record<string, unknown> = {}): Promise<void> => { state = await f.service.command(f.userId, body(op, extra)); };
  try {
    await test('真实PG写入设置并跨Service重读', async () => { await run('configure', { enabled: true, preferredTool: 'breathing' }); assert.equal((await new SupportAgentService(prisma).overview(f.userId)).state.enabled, true); });
    await test('真实PG仅读取本人自评', async () => { assert.equal(state.trend.observations, 3); assert.equal((await f.service.overview(f.otherId)).trend.observations, 0); });
    await run('propose');
    const actionId = state.state.actions[0].id;
    await test('真实PG12路相同接受请求只创建一次并有11次幂等重放', async () => {
      const request = body('accept', { actionId });
      const results = await Promise.all(Array.from({ length: 12 }, () => f.service.command(f.userId, request)));
      assert.equal(results.filter(result => result.replayed).length, 11);
      assert.equal(await prisma.healingSession.count({ where: { userId: f.userId, type: 'breathing' } }), 1);
      state = await f.service.overview(f.userId);
    });
    await test('真实PG另一用户不能操作该活动', async () => { const other = await f.service.command(f.otherId, { op: 'configure', requestId: randomUUID(), expectedRevision: 0, enabled: true, preferredTool: 'none' });
      await assert.rejects(f.service.command(f.otherId, { op: 'accept', requestId: randomUUID(), expectedRevision: other.state.revision, actionId })); });
    await test('真实PG两个不同请求争用同版本仅一个成功', async () => {
      const results = await Promise.allSettled([f.service.command(f.userId, body('configure', { enabled: true, preferredTool: 'music' })), f.service.command(f.userId, body('configure', { enabled: true, preferredTool: 'breathing' }))]);
      assert.equal(results.filter(result => result.status === 'fulfilled').length, 1); state = await f.service.overview(f.userId);
    });
    const executionToken = state.state.actions[0].executionToken;
    await run('started', { actionId, executionToken }); f.advance(29);
    await test('真实PG完成回执更新实际活动但不完成状态行', async () => { await run('complete', { actionId, executionToken }); assert.equal(await prisma.healingSession.count({ where: { userId: f.userId, type: 'breathing', completed: true } }), 1); assert.equal(await prisma.healingSession.count({ where: { userId: f.userId, type: SUPPORT_STATE_TYPE, completed: true } }), 0); });
    f.advance(6 * 3600); await run('propose'); const rollbackId = state.state.actions[1].id;
    await test('真实PG写状态失败时回滚已经创建的实际活动', async () => {
      const before = await prisma.healingSession.count({ where: { userId: f.userId } });
      const rollbackDb: any = new Proxy(prisma, { get(target, key) {
        if (key !== '$transaction') return Reflect.get(target, key);
        return async (operation: (tx: any) => Promise<unknown>): Promise<unknown> => prisma.$transaction(async tx => {
          const txProxy = new Proxy(tx, { get(txTarget, txKey) {
            if (txKey !== 'healingSession') return Reflect.get(txTarget, txKey);
            return new Proxy(tx.healingSession, { get(delegate, method) {
              if (method === 'upsert') return async (): Promise<never> => { throw new Error('intentional rollback test'); };
              return Reflect.get(delegate, method);
            } });
          } });
          return operation(txProxy);
        });
      } });
      await assert.rejects(new SupportAgentService(rollbackDb).command(f.userId, body('accept', { actionId: rollbackId })));
      assert.equal(await prisma.healingSession.count({ where: { userId: f.userId } }), before);
      assert.equal((await f.service.overview(f.userId)).state.actions[1].status, 'offered');
    });
    await test('真实PG关闭后保留操作审计并取消待办', async () => { await run('configure', { enabled: false, preferredTool: 'none' }); assert.equal(state.state.actions[1].status, 'cancelled'); assert.equal(state.trend.observations, 0); });
    await test('真实PG普通练习接口不能覆盖Agent状态或伪造完成', async () => {
      const managed = await prisma.healingSession.findFirstOrThrow({ where: { userId: f.userId, type: SUPPORT_STATE_TYPE } });
      await assert.rejects(healingService.completeSession(managed.id, f.userId, { metadata: { enabled: true } }));
      await assert.rejects(healingService.createSession(f.userId, SUPPORT_STATE_TYPE, '伪造状态'));
      await assert.rejects(healingService.getSessions(f.userId, SUPPORT_STATE_TYPE));
      assert.equal((await f.service.overview(f.userId)).state.enabled, false);
    });
    await test('真实PG练习列表统计与周报不把设置行算为活动', async () => {
      const listed = await healingService.getSessions(f.userId);
      assert.equal(listed.sessions.filter(row => row.type === SUPPORT_STATE_TYPE).length, 0);
      assert.equal((await healingService.getStats(f.userId)).totalSessions, 1);
      const report = await new HealthReportService().generateWeeklyReport(f.userId);
      assert.equal(JSON.parse(report.content).totalHealingSessions, listed.total);
    });
    const result = { status: 'passed', database: 'isolated PostgreSQL 16, 127.0.0.1:55432/aic_round2', cases: labels.length, labels, clinical_efficacy_tested: false };
    const output = process.env.ROUND2_AGENT_OUTPUT || 'D:/Develop/AIC/_国赛优化/round2/agent';
    fs.mkdirSync(output, { recursive: true }); fs.writeFileSync(path.join(output, 'postgres-verification.json'), JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result, null, 2));
  } finally { await f.cleanup(); }
}
if (require.main === module) main().catch((error: unknown) => { console.error(error); process.exitCode = 1; });
