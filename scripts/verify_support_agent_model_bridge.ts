/** 在真实PG本人证据基础上调用实际Python研究模型；不将单条运行探针当准确率或疗效。 */
import assert from 'node:assert/strict';
import { createHash, randomUUID } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import prisma from '../server/src/config/database';
import { SupportAgentService } from '../server/src/services/support-agent.service';
import { createPostgresAgentFixture } from './verify_support_agent_postgres';

async function main(): Promise<void> {
  assert.equal(process.env.SUPPORT_AGENT_MODEL_REFERENCE, '1'); assert.ok(process.env.ROUND2_RESEARCH_TOKEN);
  const fixture = await createPostgresAgentFixture();
  const service = new SupportAgentService(prisma);
  const quote = '我担心自己没有被理解，想把刚才的意思说清楚。'; const feedbackId = randomUUID();
  const conversation = await prisma.conversation.create({ data: { userId: fixture.userId, title: '第三轮研究模型实际HTTP探针' } });
  const assistant = await prisma.message.create({ data: { conversationId: conversation.id, role: 'assistant', content: '这条为自编工程测试内容。', sentiment: JSON.stringify({ support_state: {
    schema_version: 'support.v1', session_id: conversation.id, revision: 1, goal: 'understand', goal_source: 'user_feedback', declined: false,
    items: [{ id: 'e1', quote, source_id: `feedback:${feedbackId}`, source_sha256: createHash('sha256').update(quote).digest('hex'),
      start: 0, end: [...quote].length, status: 'corrected', origin: 'user_correction' }],
    feedback_log: [{ feedback_id: feedbackId, action: 'correct', item_id: 'e1', text: quote }],
  } }) } });
  const originalPin = process.env.SUPPORT_AGENT_MODEL_SHA256;
  try {
    await service.command(fixture.userId, { op: 'configure', requestId: randomUUID(), expectedRevision: 0, enabled: true, preferredTool: 'none' });
    const result = await service.refreshModelReference(fixture.userId);
    assert.equal(result.modelReference?.available, true, 'actual Python model did not return a usable calibrated matching observation');
    assert.equal(result.modelReference!.task_id, 'sentiment_polarity_3');
    assert.deepEqual(result.modelReference!.label_schema, ['negative', 'neutral', 'positive']);
    assert.equal(result.suggestionEligible, false, 'explicit understand goal must not become automatic tool selection');
    assert.equal(Date.parse(result.modelReference!.observed_at), assistant.createdAt.getTime());
    assert.equal(result.modelReference!.evidence_source.source_sha256, createHash('sha256').update(quote).digest('hex'));
    process.env.SUPPORT_AGENT_MODEL_SHA256 = '0'.repeat(64);
    const rejected = await service.refreshModelReference(fixture.userId); assert.equal(rejected.modelReference, null);
    if (originalPin) process.env.SUPPORT_AGENT_MODEL_SHA256 = originalPin; else delete process.env.SUPPORT_AGENT_MODEL_SHA256;
    process.env.SUPPORT_AGENT_MODEL_REFERENCE = '0';
    assert.equal((await service.refreshModelReference(fixture.userId)).modelReference, null);
    process.env.SUPPORT_AGENT_MODEL_REFERENCE = '1';
    const report = { status: 'passed', scope: 'real PostgreSQL personal confirmed correction -> Node provider -> actual Python HTTP model -> source/task/calibration/deployment checks',
      available: result.modelReference!.available, task_id: result.modelReference!.task_id, labels: result.modelReference!.label_schema,
      model_version: result.modelReference!.model_version, artifact_sha256: result.modelReference!.artifact_sha256,
      calibration: result.modelReference!.calibration, probabilities: result.modelReference!.probabilities,
      exact_source_hash_and_time_matched: true, wrong_deployment_pin_rejected: true, provider_disabled_has_no_fake_prediction: true,
      explicit_goal_understand_did_not_trigger_tools: true, synthetic_text_probe_only: true, human_participants: 0, clinical_efficacy_tested: false };
    const output = process.env.ROUND3_AGENT_OUTPUT || 'D:/Develop/AIC/_国赛优化/round3/agent'; fs.mkdirSync(output, { recursive: true });
    fs.writeFileSync(path.join(output, 'http-model-verification.json'), JSON.stringify(report, null, 2)); console.log(JSON.stringify(report, null, 2));
  } finally { if (originalPin) process.env.SUPPORT_AGENT_MODEL_SHA256 = originalPin; else delete process.env.SUPPORT_AGENT_MODEL_SHA256; await fixture.cleanup(); }
}
main().catch((failure: unknown) => { console.error(failure); process.exitCode = 1; });
