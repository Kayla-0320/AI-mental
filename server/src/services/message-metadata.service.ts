import type { PrismaClient } from '@prisma/client';
import prisma from '../config/database';

const EMOTION_FIELDS = [
  'primaryEmotion', 'secondaryEmotions', 'intensity', 'tone', 'urgency',
  'keywords', 'sentimentScore', 'summary',
] as const;

/** 原子合并情绪字段，保留对话状态及用户同时保存的确认、修正和目标。 */
export async function saveMessageEmotion(
  messageId: string, emotion: Record<string, unknown>, db: Pick<PrismaClient, '$executeRaw'> = prisma,
): Promise<void> {
  const patch = Object.fromEntries(EMOTION_FIELDS.filter(key => Object.prototype.hasOwnProperty.call(emotion, key)).map(key => [key, emotion[key]]));
  await db.$executeRaw`
    UPDATE messages
    SET sentiment = (COALESCE(sentiment::jsonb, '{}'::jsonb) || ${JSON.stringify(patch)}::jsonb)::text
    WHERE id = ${messageId}`;
}
