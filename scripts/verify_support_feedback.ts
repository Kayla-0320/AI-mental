/** 无真实数据库/付费模型依赖的业务层契约回归。 */
import assert from 'node:assert/strict';
import { SupportFeedbackService, readSupportState } from '../server/src/services/support-feedback.service';

const baseState = { schema_version: 'support.v1', session_id: 'conv', revision: 1, goal: 'unspecified', declined: false, items: [], feedback_log: [] };
const request = { feedback_id: 'request-1', expected_revision: 1, action: 'set_goal', goal: 'listen' };

function fixture(options: { owner?: boolean; latest?: string; count?: number; race?: boolean; computeFailure?: boolean; deletedDuringCompute?: boolean; emotionDuringCompute?: boolean; feedbackDuringCompute?: boolean } = {}) {
  let saved: any;
  let calls = 0;
  let latestReads = 0;
  const record = { id: 'msg', role: 'assistant', conversationId: 'conv', sentiment: JSON.stringify({ support_state: baseState, audit_passed: true }) };
  const db: any = {
    healingSession: { findFirst: async () => null },
    conversation: { findFirst: async () => options.owner === false ? null : { id: 'conv' } },
    message: {
      findFirst: async (arg: any) => arg.where.id ? record : { id: options.latest || (options.race && ++latestReads > 1 ? 'new' : 'msg') },
      updateMany: async (arg: any) => { calls++; saved = arg; return { count: options.count ?? 1 }; },
    },
  };
  db.$queryRaw = async () => options.deletedDuringCompute ? [] : [{ id: 'conv' }];
  db.$transaction = async (run: any) => run(db);
  const compute = async (arg: any) => {
    if (options.computeFailure) throw new Error('test service unavailable');
    if (options.emotionDuringCompute) {
      // PostgreSQL jsonb 合并也会重排对象键，不能按 JSON 字符串判定理解冲突。
      const { schema_version, ...rest } = baseState;
      record.sentiment = JSON.stringify({ primaryEmotion: '难过', audit_passed: true, support_state: { ...rest, schema_version } });
    }
    if (options.feedbackDuringCompute) {
      record.sentiment = JSON.stringify({ support_state: { ...baseState, revision: 2, declined: true } });
    }
    return { ...arg.state, revision: 2, goal: arg.goal };
  };
  const service = new SupportFeedbackService(db, compute);
  return { service, saved: () => saved, calls: () => calls };
}

async function main() {
  const normal = fixture();
  const result = await normal.service.submit('conv', 'owner', 'msg', request);
  assert.equal(result.supportState.goal, 'listen');
  assert.equal(normal.calls(), 1);
  assert.equal(JSON.parse(normal.saved().data.sentiment).audit_passed, true);
  assert.equal(normal.saved().where.sentiment, JSON.stringify({ support_state: baseState, audit_passed: true }));
  for (const options of [{ owner: false }, { latest: 'new-message' }, { count: 0 }]) {
    const f = fixture(options);
    await assert.rejects(f.service.submit('conv', 'stranger', 'msg', request));
    if (options.owner === false || options.latest) assert.equal(f.calls(), 0);
  }
  const invalid = fixture();
  await assert.rejects(invalid.service.submit('conv', 'owner', 'msg', { ...request, state: baseState }));
  await assert.rejects(invalid.service.submit('conv', 'owner', 'msg', { ...request, action: 'correct', item_id: 'item', text: '' }));
  await assert.rejects(invalid.service.submit('conv', 'owner', 'msg', { ...request, expected_revision: '1' }));
  assert.equal(invalid.calls(), 0);
  const concurrentEmotion = fixture({ emotionDuringCompute: true });
  const concurrentResult = await concurrentEmotion.service.submit('conv', 'owner', 'msg', request);
  assert.equal(concurrentResult.supportState.goal, 'listen');
  const merged = JSON.parse(concurrentEmotion.saved().data.sentiment);
  assert.equal(merged.primaryEmotion, '难过');
  assert.equal(merged.audit_passed, true);
  assert.equal(merged.support_state.revision, 2);
  for (const options of [{ race: true }, { computeFailure: true }, { deletedDuringCompute: true }, { feedbackDuringCompute: true }]) {
    const failing = fixture(options);
    await assert.rejects(failing.service.submit('conv', 'owner', 'msg', request));
    assert.equal(failing.calls(), 0);
  }
  assert.equal(readSupportState([{ role: 'assistant', sentiment: JSON.stringify({ support_state: baseState }) }], 'other'), null);
  assert.equal(readSupportState([{ role: 'assistant', sentiment: JSON.stringify({ support_state: baseState }) }, { role: 'assistant', sentiment: '{}' }], 'conv'), null);
  console.log('support-feedback business contract: all cases passed (mock database)');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
