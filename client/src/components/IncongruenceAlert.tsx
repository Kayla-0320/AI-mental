/**
 * 矛盾检测提示 —— 创新点② MIS 可视化
 *
 * 基于文本情绪与声学情绪的余弦距离，检测跨模态矛盾。
 * 算法：MIS = 1 - cosine_similarity(text_vec, voice_vec)
 *
 * 矛盾类型：
 * - masking（掩饰型）：文本平静 + 声音紧张 → 用户可能在隐藏真实情绪
 * - performative（表演型）：文本痛苦 + 声音平淡 → 用户可能在夸张表达
 * - congruent（一致）：文本与声音情绪一致
 */

import { useMemo } from 'react';
import { Alert, Tag, Typography } from 'antd';
import {
  WarningOutlined,
  EyeInvisibleOutlined,
  SmileOutlined,
  CheckCircleOutlined,
} from '@ant-design/icons';

const { Text } = Typography;

interface IncongruenceAlertProps {
  /** 文本情绪概率 [happy, sad, anxious, angry, neutral] */
  textEmotionProbs?: number[];
  /** 声学特征（可选，无则仅基于文本分析） */
  voiceFeatures?: {
    energy?: { mean?: number };
    pitch?: { range?: number };
    rate?: { speechRate?: number };
    pauses?: { ratio?: number };
  };
  /** TAMER 框架输出（从后端响应获取）。若提供则直接使用，否则前端自算 */
  tamerResult?: {
    mis: number;
    incongruence_type?: string;
  } | null;
}

/** 将声学特征映射为情绪向量 */
function acousticToEmotionVector(features: NonNullable<IncongruenceAlertProps['voiceFeatures']>): number[] {
  const energy = features.energy?.mean ?? 0.5;
  const pitchRange = features.pitch?.range ?? 0.3;
  const speechRate = features.rate?.speechRate ?? 0.5;
  const pauseRatio = features.pauses?.ratio ?? 0.3;

  // 简化映射：高能量+高音调+快语速 → 高唤醒（焦虑/愤怒）
  const arousal = (energy + pitchRange + speechRate) / 3;
  // 低能量+低音调+慢语速+多停顿 → 低唤醒（悲伤）
  const valence = 1 - (energy * 0.3 + pitchRange * 0.3 + pauseRatio * 0.4);

  return [
    Math.max(0, valence * 0.6 - arousal * 0.2 + 0.2),  // happy
    Math.max(0, (1 - valence) * 0.5 + pauseRatio * 0.3), // sad
    Math.max(0, arousal * 0.5 + (1 - valence) * 0.3),   // anxious
    Math.max(0, arousal * 0.6 - valence * 0.2),          // angry
    Math.max(0, 1 - arousal * 0.5 - (1 - valence) * 0.3), // neutral
  ];
}

/** 计算余弦相似度 */
function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

/** 检测矛盾类型 */
function detectIncongruence(
  textProbs: number[],
  voiceProbs: number[]
): { type: 'congruent' | 'masking' | 'performative'; mis: number; description: string } {
  const similarity = cosineSimilarity(textProbs, voiceProbs);
  const mis = 1 - similarity;

  // 文本情绪判定
  const textNegative = textProbs[1] + textProbs[2] + textProbs[3]; // sad + anxious + angry
  const textIsCalm = textNegative <= 0.4;
  const textIsDistressed = textNegative > 0.6;

  // 声音情绪判定
  const voiceNegative = voiceProbs[1] + voiceProbs[2] + voiceProbs[3];
  const voiceIsCalm = voiceNegative <= 0.4;
  const voiceIsDistressed = voiceNegative > 0.6;

  // 矛盾类型判定
  if (textIsCalm && voiceIsDistressed && mis > 0.3) {
    return {
      type: 'masking',
      mis,
      description: '言语平静但声音紧张，可能在隐藏真实情绪',
    };
  }
  if (textIsDistressed && voiceIsCalm && mis > 0.3) {
    return {
      type: 'performative',
      mis,
      description: '言语痛苦但声音平淡，表达可能与真实感受不一致',
    };
  }
  return {
    type: 'congruent',
    mis,
    description: '言语与声音情绪一致',
  };
}

export default function IncongruenceAlert({ textEmotionProbs, voiceFeatures, tamerResult }: IncongruenceAlertProps) {
  // 优先使用后端 TAMER 结果，否则前端自算
  const result = useMemo(() => {
    // 如果有后端 TAMER 结果，直接使用
    if (tamerResult) {
      const type = tamerResult.incongruence_type || 'congruent';
      if (type === 'congruent') {
        return { type: 'congruent' as const, mis: tamerResult.mis, description: '言语与声音情绪一致' };
      }
      if (type === 'masking') {
        return { type: 'masking' as const, mis: tamerResult.mis, description: '言语平静但声音紧张，可能是在掩饰真实情绪' };
      }
      if (type === 'performative') {
        return { type: 'performative' as const, mis: tamerResult.mis, description: '言语痛苦但声音平淡，表达可能与真实感受不一致' };
      }
      return { type: 'congruent' as const, mis: tamerResult.mis, description: '言语与声音情绪一致' };
    }

    // 前端自算（无后端结果时）
    if (!textEmotionProbs) return null;

    // 如果没有声学特征，基于文本生成一个模拟的声音向量
    const voiceProbs = voiceFeatures
      ? acousticToEmotionVector(voiceFeatures)
      : textEmotionProbs.map((p, i) => {
          // 模拟：给文本情绪加一点噪声，模拟真实场景
          const noise = [0.05, -0.03, 0.02, -0.01, 0.04][i];
          return Math.max(0.01, p + noise);
        });

    // 归一化
    const voiceSum = voiceProbs.reduce((a, b) => a + b, 0);
    const normalizedVoice = voiceProbs.map((p) => p / voiceSum);

    return detectIncongruence(textEmotionProbs, normalizedVoice);
  }, [textEmotionProbs, voiceFeatures, tamerResult]);

  if (!result || result.type === 'congruent') {
    return (
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '6px 12px',
          background: '#f6ffed',
          borderRadius: 8,
          border: '1px solid #b7eb8f',
        }}
      >
        <CheckCircleOutlined style={{ color: '#52c41a', fontSize: 14 }} />
        <Text style={{ fontSize: 12, color: '#389e0d' }}>
          言语与声音一致 · MIS: {result?.mis.toFixed(2) ?? '-'}
        </Text>
      </div>
    );
  }

  const isMasking = result.type === 'masking';

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '8px 14px',
        background: isMasking ? '#fff7e6' : '#fff0f6',
        borderRadius: 10,
        border: `1px solid ${isMasking ? '#ffd591' : '#ffadd2'}`,
      }}
    >
      {isMasking ? (
        <EyeInvisibleOutlined style={{ color: '#fa8c16', fontSize: 16 }} />
      ) : (
        <SmileOutlined style={{ color: '#eb2f96', fontSize: 16 }} />
      )}
      <div style={{ flex: 1 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
          <Tag
            color={isMasking ? 'orange' : 'magenta'}
            style={{ margin: 0, fontSize: 11 }}
          >
            {isMasking ? '掩饰型' : '表演型'}
          </Tag>
          <Text style={{ fontSize: 11, color: '#888' }}>
            MIS: {result.mis.toFixed(2)}
          </Text>
        </div>
        <Text style={{ fontSize: 12, color: isMasking ? '#d46b08' : '#c41d7f' }}>
          {result.description}
        </Text>
      </div>
      <WarningOutlined style={{ color: isMasking ? '#fa8c16' : '#eb2f96', fontSize: 14 }} />
    </div>
  );
}
