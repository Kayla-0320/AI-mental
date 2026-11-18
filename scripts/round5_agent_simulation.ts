/** 合成用户驱动实际Agent业务服务；内存事务替身，不冒充真实数据库或真人研究。 */
import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { createAgentFixture } from './verify_support_agent';
import { nextAllowedSupportTime } from '../server/src/services/support-agent.service';

async function main(): Promise<void> {
  const fixture = createAgentFixture();
  const ids = Array.from({ length: 100 }, (_, i) => `SYNTHETIC-agent-${String(i).padStart(3, '0')}`);
  const allowed = new Set(ids);
  fixture.db.$queryRaw = async (_strings: TemplateStringsArray, userId: string): Promise<unknown[]> => allowed.has(userId) ? [{ id: userId }] : [];
  fixture.db.moodRecord.findMany = async ({ where }: any): Promise<any[]> => [4, 5, 6].map((score, i) => ({
    id: `synthetic-mood-${where.userId}-${i}`, userId: where.userId, mood: 'neutral', score,
    recordedAt: new Date(where.recordedAt.lte.getTime() - (3 - i) * 3600000),
  }));
  let checks = 0; const totals = { completed: 0, failed: 0, cancelled: 0, replayed: 0 };
  for (const [index, userId] of ids.entries()) {
    let state = await fixture.service.overview(userId);
    const run = async (op: string, extra: Record<string, unknown> = {}): Promise<void> => {
      state = await fixture.service.command(userId, { op, requestId: randomUUID(), expectedRevision: state.state.revision, ...extra });
    };
    assert.equal(state.state.enabled, false); checks++;
    await run('configure', { enabled: true, preferredTool: index % 2 ? 'music' : 'breathing' });
    await run('propose');
    const action = state.state.actions.at(-1)!;
    assert.equal(action.status, 'offered'); checks++;
    const request = { op: 'accept', requestId: randomUUID(), expectedRevision: state.state.revision, actionId: action.id };
    const accepts = await Promise.all(Array.from({ length: 5 }, () => fixture.service.command(userId, request)));
    assert.equal(accepts.filter(row => row.replayed).length, 4); checks++; totals.replayed += 4;
    state = await fixture.service.overview(userId);
    const token = state.state.actions.at(-1)!.executionToken;
    await assert.rejects(fixture.service.command(userId, { op: 'started', requestId: randomUUID(), expectedRevision: state.state.revision, actionId: action.id, executionToken: randomUUID() })); checks++;
    await run('started', { actionId: action.id, executionToken: token });
    await assert.rejects(fixture.service.command(userId, { op: 'complete', requestId: randomUUID(), expectedRevision: state.state.revision, actionId: action.id, executionToken: token })); checks++;
    fixture.advance(31);
    if (index % 4 === 0) { await run('fail', { actionId: action.id, executionToken: token, reason: 'playback_failed' }); totals.failed++; }
    else { await run('complete', { actionId: action.id, executionToken: token }); await run('feedback', { actionId: action.id, value: index % 3 === 0 ? 'unhelpful' : 'neutral' }); totals.completed++; }
    assert.equal(state.state.actions.at(-1)!.status, index % 4 === 0 ? 'failed' : 'completed'); checks++;
    fixture.advance(6 * 3600); await run('propose');
    const offered = state.state.actions.at(-1)!;
    await run('configure', { enabled: false, preferredTool: 'none' });
    assert.equal(state.state.actions.at(-1)!.status, 'cancelled'); checks++; totals.cancelled++;
    await assert.rejects(fixture.service.command(ids[(index + 1) % ids.length], { op: 'accept', requestId: randomUUID(), expectedRevision: 0, actionId: offered.id })); checks++;
    const otherState = await fixture.service.overview(ids[(index + 1) % ids.length]);
    assert.ok(!otherState.state.actions.some(row => row.id === action.id || row.id === offered.id)); checks++;
  }
  for (let hour = 0; hour < 72; hour++) {
    const candidate = new Date(Date.UTC(2026, 10, 15, hour));
    const next = nextAllowedSupportTime(candidate, 22, 8);
    const localHour = new Date(next.getTime() + 8 * 3600000).getUTCHours();
    assert.ok(localHour >= 8 && localHour < 22); assert.ok(next >= candidate); checks += 2;
  }
  const output = process.env.ROUND5_SIMULATION_OUTPUT || 'D:/Develop/AIC/_国赛优化/round5/simulation';
  fs.mkdirSync(output, { recursive: true });
  const result = { status: 'passed', synthetic: true, human_participants: 0, users: ids.length, storage: 'memory transaction fixture; not PostgreSQL or browser playback',
    checks, totals, actual_service: 'SupportAgentService', clinical_efficacy_tested: false, failures_injected: ['wrong token', 'early completion', 'playback failure', 'cross-user action', 'concurrent duplicate acceptance'] };
  fs.writeFileSync(path.join(output, 'agent_service_simulation.json'), JSON.stringify(result, null, 2)); console.log(JSON.stringify(result));
}
main().catch((error: unknown) => { console.error(error); process.exitCode = 1; });
