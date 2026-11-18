import assert from 'node:assert/strict';
import { readLatestInnovation } from '../server/src/routes/innovation.routes';
import { parseInnovation, innovationFromSentiment } from '../client/src/services/innovationPresentation';

async function main(): Promise<void> {
  let conversation: unknown = null;
  let latest: unknown = null;
  let reads = 0;
  const db: any = {
    conversation: { findFirst: async (query: any) => { assert.deepEqual(query.where, { userId: 'owner', isActive: true }); return conversation; } },
    message: { findFirst: async (query: any) => { reads++; assert.equal(query.where.conversationId, 'owned-conversation'); assert.equal(query.orderBy[0].createdAt, 'desc'); return latest; } },
  };
  assert.equal((await readLatestInnovation('owner', db)).status, 'no_conversation'); assert.equal(reads, 0);
  conversation = { id: 'owned-conversation', title: '本人对话' };
  assert.equal((await readLatestInnovation('owner', db)).status, 'no_messages');
  latest = { role: 'user', sentiment: null };
  assert.equal((await readLatestInnovation('owner', db)).status, 'awaiting_reply');
  const reference = { available: true, observed_modalities: ['text'], expert_names: ['text'], expert_weights: [1], quality: {}, warnings: [], knowledge_matches: [] };
  latest = { id: 'answer', role: 'assistant', createdAt: new Date(), sentiment: JSON.stringify({ innovation: reference, support_state: { private: true } }), content: '不应从该接口返回原始聊天内容' };
  const result = await readLatestInnovation('owner', db);
  assert.equal(result.status, 'recorded'); assert.deepEqual(result.reference, reference);
  assert.ok(!('content' in result) && !('support_state' in result));
  for (const sentiment of ['{broken', 'a'.repeat(65537)]) {
    latest = { role: 'assistant', sentiment };
    assert.equal((await readLatestInnovation('owner', db)).status, 'no_reference');
  }
  assert.equal(innovationFromSentiment('broken'), null);
  assert.equal(parseInnovation({ available: true, expert_names: ['text'], expert_weights: [NaN] })?.expert_weights.length, 0);
  assert.equal(parseInnovation({ available: true, quality: { visual: { quality_score: Infinity } } })?.quality.visual, undefined);
  assert.equal(parseInnovation({ available: false })?.available, false);
  assert.equal(parseInnovation(null), null);
  console.log('innovation overview: owner scope, newest-message selection, pending reply, corrupt data and presentation validation passed');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
