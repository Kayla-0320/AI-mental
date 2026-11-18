/** 第三轮真实 PostgreSQL 后台租约、本人证据撤销与研究模型边界验证。 */
import assert from 'node:assert/strict';
import { createHash, randomUUID } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import prisma from '../server/src/config/database';
import { SupportAgentService } from '../server/src/services/support-agent.service';
import { PersonalEvidence, ModelReference, validateModelReference } from '../server/src/services/support-agent-context';
import { SupportAgentWorker } from '../server/src/services/support-agent-worker';
import { SupportFeedbackService } from '../server/src/services/support-feedback.service';
import { createPostgresAgentFixture } from './verify_support_agent_postgres';

async function main(): Promise<void> {
  const fixture = await createPostgresAgentFixture();
  let now = Date.now() + 1000;
  const service = new SupportAgentService(prisma, () => new Date(now));
  const tests: string[] = [];
  const test = async (label: string, action: () => Promise<void>): Promise<void> => { await action(); tests.push(label); };
  const run = async (op: string, extra: Record<string, unknown> = {}): Promise<void> => {
    const current = await service.overview(fixture.userId);
    await service.command(fixture.userId, { op, requestId: randomUUID(), expectedRevision: current.state.revision, ...extra });
  };
  const setupBackground = async (): Promise<void> => {
    const hour = new Date(now + 480 * 60000).getUTCHours();
    await run('background_settings', { enabled: true, notifyInApp: true, intervalHours: 6, quietStart: (hour + 10) % 24, quietEnd: (hour + 11) % 24 });
  };
  let conversationId = ''; let assistantId = ''; let userMessageId = '';
  const quote = '我担心被忽视，不是觉得对方讨厌我';
  const correctionId = randomUUID();
  const support = (goal = 'act', revision = 1, declined = false): Record<string, unknown> => ({ schema_version: 'support.v1', session_id: conversationId, revision,
    goal, goal_source: 'user_feedback', declined, items: [{ id: 'e1', slot: 'thought', quote,
      source_id: `feedback:${correctionId}`, source_sha256: createHash('sha256').update(quote).digest('hex'), start: 0, end: [...quote].length,
      status: 'corrected', origin: 'user_correction' }], feedback_log: [{ feedback_id: correctionId, action: 'correct', item_id: 'e1', text: quote }] });
  const replaceSupport = async (value: Record<string, unknown>, risk = 'low'): Promise<void> => {
    await prisma.message.update({ where: { id: assistantId }, data: { sentiment: JSON.stringify({ support_state: value, risk_level: risk }) } });
  };
  const model = (evidence: PersonalEvidence): ModelReference => ({ available: true, task_id: 'sentiment_polarity_3', label_schema: ['negative', 'neutral', 'positive'],
    model_version: 'test-model-v1', artifact_sha256: 'a'.repeat(64), calibration: { status: 'calibrated', method: 'validation_temperature' },
    evidence_source: { source_message_id: evidence.sourceId, source_sha256: createHash('sha256').update(evidence.quote).digest('hex') },
    probabilities: [0.5, 0.2, 0.3], observed_at: evidence.observedAt, limitations: ['synthetic model response fixture; no clinical effect evidence'] });
  try {
    await run('configure', { enabled: true, preferredTool: 'none' });
    await test('后台默认独立关闭且未同意不能认领', async () => { assert.equal((await service.overview(fixture.userId)).state.background.enabled, false); assert.equal(await service.claimBackground(fixture.userId, 'worker-a'), null); });
    await setupBackground();
    let lease: Awaited<ReturnType<typeof service.claimBackground>> = null;
    await test('真实PG十二实例抢占同一到期任务只有一份租约', async () => {
      const values = await Promise.all(Array.from({ length: 12 }, (_, n) => service.claimBackground(fixture.userId, `worker-${n}`)));
      assert.equal(values.filter(Boolean).length, 1); lease = values.find(Boolean)!;
    });
    await test('退出后台后旧租约不能回写建议', async () => {
      const current = (await service.overview(fixture.userId)).state.background;
      await run('background_settings', { enabled: false, intervalHours: 6, quietStart: current.quietStart, quietEnd: current.quietEnd });
      assert.equal(await service.executeBackground(lease!), 'lease_lost'); assert.equal((await service.overview(fixture.userId)).state.actions.length, 0);
    });
    await setupBackground(); const oldLease = await service.claimBackground(fixture.userId, 'crashed-worker');
    await test('重启实例等待租约过期后恢复且旧实例不能提交', async () => {
      assert.equal(await service.claimBackground(fixture.userId, 'restart-worker'), null); now += 121000;
      const recovered = await new SupportAgentService(prisma, () => new Date(now)).claimBackground(fixture.userId, 'restart-worker'); assert.ok(recovered);
      assert.equal(await service.executeBackground(oldLease!), 'lease_lost');
      assert.equal(await service.executeBackground(recovered!), 'offered');
      const current = await service.overview(fixture.userId); assert.equal(current.state.actions.length, 1); assert.equal(current.state.actions[0].origin, 'background');
      assert.equal(await prisma.healingSession.count({ where: { userId: fixture.userId, type: { in: ['breathing', 'music'] } } }), 0);
    });
    await test('后台建议只入站内不会启动真实工具', async () => { assert.equal(await prisma.healingSession.count({ where: { userId: fixture.userId, completed: true } }), 0); });
    await test('站内提醒与offer同事务防重且不含原话', async () => {
      const notifications = await prisma.notification.findMany({ where: { userId: fixture.userId, type: 'support_agent' } });
      assert.equal(notifications.length, 1); assert.equal(notifications[0].isRead, false); assert.equal(notifications[0].link, '/healing');
      assert.equal(notifications[0].content.includes(quote), false); assert.equal(await service.executeBackground(oldLease!), 'lease_lost');
      assert.equal(await prisma.notification.count({ where: { userId: fixture.userId, type: 'support_agent' } }), 1);
    });
    await test('进程错过窗口跳过积压而非补发', async () => {
      await setupBackground(); now += 31 * 60000; assert.equal(await service.claimBackground(fixture.userId, 'late'), null);
      assert.equal((await service.overview(fixture.userId)).state.background.events.slice(-1)[0].result, 'missed_window_skipped');
    });
    await test('安静时段不认领并持久记录跳过', async () => {
      const hour = new Date(now + 480 * 60000).getUTCHours();
      await run('background_settings', { enabled: true, intervalHours: 6, quietStart: hour, quietEnd: (hour + 1) % 24 });
      assert.equal(await service.claimBackground(fixture.userId, 'quiet'), null);
      assert.equal((await service.overview(fixture.userId)).state.background.events.slice(-1)[0].result, 'quiet_hours_skipped');
    });
    const conversation = await prisma.conversation.create({ data: { userId: fixture.userId, title: '第三轮本人修正夹具' } }); conversationId = conversation.id;
    const original = await prisma.message.create({ data: { userId: fixture.userId, conversationId, role: 'user', content: '我以为自己被忽视了', createdAt: new Date(now - 2000) } }); userMessageId = original.id;
    const assistant = await prisma.message.create({ data: { conversationId, role: 'assistant', content: '我记下了你的修正。', sentiment: JSON.stringify({ support_state: support() }), createdAt: new Date(now - 1000) } }); assistantId = assistant.id;
    await test('已保存本人修正按反馈日志和SHA核验而非相信标签', async () => { const value = await service.overview(fixture.userId); assert.equal(value.context.evidence[0].quote, quote); assert.equal(value.context.evidence[0].sourceId, `feedback:${correctionId}`); });
    await test('第一轮本人撤回在同一真实事务取消旧offer及站内提醒', async () => {
      const feedback = new SupportFeedbackService(prisma, async (args: any) => ({ ...args.state, revision: args.state.revision + 1, declined: true }));
      await feedback.submit(conversationId, fixture.userId, assistantId, { feedback_id: randomUUID(), expected_revision: 1, action: 'decline' });
      assert.equal(await prisma.notification.count({ where: { userId: fixture.userId, type: 'support_agent', isRead: false } }), 0);
      assert.equal((await service.overview(fixture.userId)).state.actions.filter(item => item.status === 'offered').length, 0);
      await replaceSupport(support('act', 1));
    });
    await test('拒绝或撤销证据实时失效', async () => { const changed = support('act', 2); (changed.items as any[])[0].status = 'rejected'; await replaceSupport(changed); assert.equal((await service.overview(fixture.userId)).context.evidence.length, 0); await replaceSupport(support('act', 3)); });
    await test('最新目标listen覆盖后台活动且租约提交时重新检查', async () => {
      await setupBackground(); const claimed = await service.claimBackground(fixture.userId, 'goal-change'); assert.ok(claimed);
      await replaceSupport(support('listen', 4)); let modelCalls = 0;
      const provider = new SupportAgentService(prisma, () => new Date(now), async evidence => { modelCalls++; return model(evidence); });
      assert.equal(await provider.executeBackground(claimed!), 'explicit_goal_respected'); assert.equal(modelCalls, 0);
      assert.equal((await service.overview(fixture.userId)).suggestionEligible, false); await replaceSupport(support('act', 5));
    });
    await test('既有high/crisis只读守卫不被普通活动替代', async () => { await replaceSupport(support('act', 6), 'high'); assert.equal((await service.overview(fixture.userId)).context.status, 'existing_escalation'); assert.equal((await service.overview(fixture.userId)).suggestionEligible, false); await replaceSupport(support('act', 7)); });
    await test('研究模型校验后与本人有效来源共同展示', async () => {
      const provider = new SupportAgentService(prisma, () => new Date(now), async evidence => model(evidence));
      const value = await provider.refreshModelReference(fixture.userId); assert.equal(value.modelReference?.available, true);
      await replaceSupport(support('act', 8, true)); assert.equal((await service.overview(fixture.userId)).modelReference, null); await replaceSupport(support('act', 9));
    });
    await test('错误任务类别/未校准/来源/概率/工件拒绝并保留原状态', async () => {
      const evidence = (await service.overview(fixture.userId)).context.evidence[0];
      for (const patch of [{ task_id: 'emotion_5' }, { label_schema: ['happy', 'sad', 'neutral'] }, { calibration: { status: 'uncalibrated', method: 'none' } }, { probabilities: [0.9, 0.9, 0.9] }, { artifact_sha256: 'invalid' }, { evidence_source: { source_message_id: 'another', source_sha256: 'a'.repeat(64) } }]) assert.throws(() => validateModelReference({ ...model(evidence), ...patch }, evidence, new Date(now)));
      process.env.SUPPORT_AGENT_MODEL_SHA256 = 'b'.repeat(64); assert.throws(() => validateModelReference(model(evidence), evidence, new Date(now))); delete process.env.SUPPORT_AGENT_MODEL_SHA256;
    });
    await test('异步模型期间本人撤回后旧结果不能回写', async () => {
      const provider = new SupportAgentService(prisma, () => new Date(now), async evidence => { await replaceSupport(support('act', 10, true)); return model(evidence); });
      await assert.rejects(provider.refreshModelReference(fixture.userId)); await replaceSupport(support('act', 11));
    });
    await test('伪造修正日志或丢失原消息时不继续使用出处', async () => {
      const changed = support('act', 12); changed.feedback_log = []; await replaceSupport(changed); assert.equal((await service.overview(fixture.userId)).context.evidence.length, 0);
      const originalState = support('act', 13); originalState.items = [{ id: 'e1', slot: 'thought', quote: '我以为自己被忽视了', source_id: userMessageId,
        source_sha256: createHash('sha256').update('我以为自己被忽视了').digest('hex'), start: 0, end: [...'我以为自己被忽视了'].length, status: 'confirmed', origin: 'user_message' }];
      await replaceSupport(originalState); assert.equal((await service.overview(fixture.userId)).context.evidence.length, 1);
      const invalidRange = structuredClone(originalState); (invalidRange.items as any[])[0].end = 999999;
      await replaceSupport(invalidRange); assert.equal((await service.overview(fixture.userId)).context.evidence.length, 0);
      await replaceSupport(originalState);
      await prisma.message.delete({ where: { id: userMessageId } }); assert.equal((await service.overview(fixture.userId)).context.evidence.length, 0); await replaceSupport(support('act', 14));
    });
    await test('本人负反馈改变默认选项而显式偏好仍优先', async () => {
      await run('configure', { enabled: false, preferredTool: 'none' }); now += 7 * 3600000;
      await run('configure', { enabled: true, preferredTool: 'none' }); await run('propose');
      let state = await service.overview(fixture.userId); const action = state.state.actions.slice(-1)[0]; assert.equal(action.tool, 'breathing');
      await run('accept', { actionId: action.id }); state = await service.overview(fixture.userId);
      const token = state.state.actions.slice(-1)[0].executionToken; await run('started', { actionId: action.id, executionToken: token }); now += 29000;
      await run('complete', { actionId: action.id, executionToken: token }); await run('feedback', { actionId: action.id, value: 'unhelpful' });
      now += 7 * 3600000; await run('propose'); state = await service.overview(fixture.userId); assert.equal(state.state.actions.slice(-1)[0].tool, 'music');
      await run('decline', { actionId: state.state.actions.slice(-1)[0].id }); await run('configure', { enabled: true, preferredTool: 'breathing' }); now += 7 * 3600000;
      await run('propose'); assert.equal((await service.overview(fixture.userId)).state.actions.slice(-1)[0].tool, 'breathing');
    });
    await test('worker无数据库退避且同实例tick不会重叠', async () => {
      const unavailable = new SupportAgentWorker({ scanBackgroundUsers: async () => { throw new Error('private DSN must not be logged'); } } as any, 10);
      const result = await unavailable.tick(); assert.equal(result.error, true); assert.equal(unavailable.status().backoffMs, 20);
      await unavailable.tick(); assert.equal(unavailable.status().backoffMs, 40); await unavailable.stop();
      const worker = new SupportAgentWorker(service, 10); const results = await Promise.all([worker.tick(), worker.tick()]);
      assert.ok(results.some(value => value.scanned === 0)); await worker.stop();
    });
    await test('单个损坏状态不阻断其他用户后台任务', async () => {
      const other = await service.overview(fixture.otherId);
      await service.command(fixture.otherId, { op: 'configure', requestId: randomUUID(), expectedRevision: other.state.revision, enabled: true, preferredTool: 'none' });
      const row = await prisma.healingSession.findFirstOrThrow({ where: { userId: fixture.otherId, type: 'support_agent_state' } });
      await prisma.healingSession.update({ where: { id: row.id }, data: { metadata: 'bad-json' } });
      await setupBackground(); const worker = new SupportAgentWorker(service);
      const result = await worker.tick(); assert.equal(result.error, false); assert.ok(result.completed >= 1);
      assert.equal((await prisma.healingSession.findUniqueOrThrow({ where: { id: row.id } })).metadata, 'bad-json'); await worker.stop();
    });
    await test('反馈和账号删除并发保持User到Conversation锁序且级联清理', async () => {
      const feedback = new SupportFeedbackService(prisma, async (args: any) => ({ ...args.state, revision: args.state.revision + 1, goal: 'listen' }));
      const results = await Promise.allSettled([
        feedback.submit(conversationId, fixture.userId, assistantId, { feedback_id: randomUUID(), expected_revision: 14, action: 'set_goal', goal: 'listen' }),
        prisma.user.delete({ where: { id: fixture.userId } }),
      ]);
      assert.equal(results[1].status, 'fulfilled');
      for (const item of results) if (item.status === 'rejected') assert.notEqual((item.reason as any)?.code, 'P2034', 'deadlock or transaction conflict');
      assert.equal(await service.claimBackground(fixture.userId, 'deleted'), null);
      assert.equal(await prisma.healingSession.count({ where: { userId: fixture.userId } }), 0);
      assert.equal(await prisma.message.count({ where: { conversationId } }), 0);
    });
    const result = { status: 'passed', database: 'isolated PostgreSQL16', cases: tests.length, tests, human_participants: 0, clinical_efficacy_tested: false, research_model_response: 'injected contract fixture, not actual model inference' };
    const output = process.env.ROUND3_AGENT_OUTPUT || 'D:/Develop/AIC/_国赛优化/round3/agent'; fs.mkdirSync(output, { recursive: true });
    fs.writeFileSync(path.join(output, 'worker-postgres-verification.json'), JSON.stringify(result, null, 2)); console.log(JSON.stringify(result, null, 2));
  } finally { await fixture.cleanup(); }
}
main().catch((failure: unknown) => { console.error(failure); process.exitCode = 1; });
