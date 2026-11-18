/** 仅用当前连接的临时表验证 PostgreSQL 原子合并，不读写真实对话记录。 */
import assert from 'node:assert/strict';
import path from 'node:path';
import dotenv from '../server/node_modules/dotenv';
import prisma from '../server/src/config/database';
import { saveMessageEmotion } from '../server/src/services/message-metadata.service';

async function main(): Promise<void> {
  dotenv.config({ path: path.resolve(__dirname, '../server/.env') });
  dotenv.config({ path: path.resolve(__dirname, '../.env') });
  try {
    await prisma.$transaction(async tx => {
      // pg_temp 在未限定表名的查询中优先；事务完成时自动删除此测试表。
      await tx.$executeRaw`CREATE TEMP TABLE messages (id TEXT PRIMARY KEY, sentiment TEXT) ON COMMIT DROP`;
      const support = { schema_version: 'support.v1', session_id: 'synthetic-conv', revision: 2, goal: 'act',
        declined: false, items: [{ id: 'item', quote: '我自己修正的重点', status: 'corrected' }], feedback_log: [{ feedback_id: 'saved' }] };
      const initial = { support_state: support, audit_passed: true, risk_level: 'low', dialog_state: 'EXPLORE',
        tamer_result: { rqi: 0.5 }, innovation: { status: 'observed' }, future_metadata: { preserved: true } };
      await tx.$executeRaw`INSERT INTO messages (id, sentiment) VALUES ('synthetic-message', ${JSON.stringify(initial)}), ('empty', NULL)`;
      await saveMessageEmotion('synthetic-message', { primaryEmotion: '难过', intensity: 45, support_state: null,
        audit_passed: false, risk_level: 'crisis' }, tx);
      const read = async (id: string): Promise<Record<string, any>> => {
        const rows = await tx.$queryRaw<Array<{ sentiment: string }>>`SELECT sentiment FROM messages WHERE id = ${id}`;
        assert.equal(rows.length, 1);
        return JSON.parse(rows[0].sentiment);
      };
      assert.deepEqual(await read('synthetic-message'), { ...initial, primaryEmotion: '难过', intensity: 45 });
      // 模拟先保存本人反馈，再收到迟到的第二次情绪分析。
      const updated = { ...support, revision: 3, declined: true };
      await tx.$executeRaw`UPDATE messages SET sentiment = jsonb_set(sentiment::jsonb, '{support_state}', ${JSON.stringify(updated)}::jsonb)::text WHERE id = 'synthetic-message'`;
      await saveMessageEmotion('synthetic-message', { primaryEmotion: '平静', intensity: 10 }, tx);
      const after = await read('synthetic-message');
      assert.deepEqual(after.support_state, updated);
      assert.deepEqual(after.future_metadata, initial.future_metadata);
      assert.equal(after.primaryEmotion, '平静');
      await saveMessageEmotion('empty', { primaryEmotion: '平静' }, tx);
      assert.deepEqual(await read('empty'), { primaryEmotion: '平静' });
      await saveMessageEmotion('already-deleted', { primaryEmotion: '平静' }, tx);
      const count = await tx.$queryRaw<Array<{ count: number }>>`SELECT COUNT(*)::int AS count FROM messages`;
      assert.equal(count[0].count, 2);
    });
    console.log('message metadata: PostgreSQL temporary-table checks passed; real conversations untouched');
  } finally { await prisma.$disconnect(); }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
