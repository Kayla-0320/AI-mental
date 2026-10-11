/**
 * 矛盾检测提示 —— 创新点② MIS 可视化
 *
 * ⚠️ 后端 (`assessment/multimodal_incongruence.py`) 是**唯一权威源**：
 * 掩饰型/表演型的判定规则、阈值与量纲归一化都在 Python 侧，
 * 这里只做展示。前端不再自己另算一套（否则两边必然随时间漂移）。
 *
 * MIS = 1 - cosine_similarity(text_vec, voice_vec)，作为**幅度指标**展示；
 * 是否矛盾由通道方向性对比决定（言语平静 + 声音紧张 → 掩饰型；
 * 言语痛苦 + 声音平静 → 表演型）。
 */

import { useMemo } from 'react';
import { Tag, Typography } from 'antd';
import {
  WarningOutlined,
  EyeInvisibleOutlined,
  SmileOutlined,
  CheckCircleOutlined,
  AudioOutlined,
} from '@ant-design/icons';

const { Text } = Typography;

/** 检测结果：三类判定 + 一种"没做检测"的中性态 */
type IncongruenceState =
  | { type: 'congruent' | 'masking' | 'performative'; mis: number; description: string }
  | { type: 'no_voice'; mis: number; description: string };

interface IncongruenceAlertProps {
  /** 文本情绪概率 [happy, sad, anxious, angry, neutral] */
  textEmotionProbs?: number[];
  /** 声学特征（后端结果缺失时用于前端兜底自算；单位见下方注释） */
  voiceFeatures?: {
    energy?: { mean?: number };
    pitch?: { range?: number };
    rate?: { speechRate?: number };
    pauses?: { ratio?: number };
  };
  /** TAMER 框架输出（后端响应）。提供时直接使用 */
  tamerResult?: {
    mis: number;
    incongruence_type?: string;
    /** 本轮是否真的有语音输入；false 时 MIS 不构成跨模态证据 */
    has_voice_input?: boolean;
  } | null;
}

/**
 * 将声学特征映射为情绪向量（后端缺失时的兜底路径）。
 *
 * ⚠️ 客户端上报的原始量纲**不是 0~1**：
 *   energy.mean      dBFS（约 -60 ~ 0）
 *   pitch.range      Hz（约 25 ~ 120）
 *   rate.speechRate  原始刻度（约 0 ~ 8）
 *   pauses.ratio     0 ~ 1
 * 必须先归一化，否则向量会整体塌掉（与后端
 * `tamer_framework.normalize_*` 同一口径）。
 */
function acousticToEmotionVector(features: NonNullable<IncongruenceAlertProps['voiceFeatures']>): number[] {
  const rawEnergy = features.energy?.mean ?? 0.5;
  const rawPitch = features.pitch?.range ?? 0.3;
  const rawRate = features.rate?.speechRate ?? 0.5;
  const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
  const energy = rawEnergy < 0 ? clamp01((rawEnergy + 60) / 60) : clamp01(rawEnergy);
  const pitchRange = rawPitch > 1.5 ? clamp01(rawPitch / 150) : clamp01(rawPitch);
  const speechRate = rawRate > 1.5 ? clamp01(rawRate / 8) : clamp01(rawRate);
  const pauseRatio = clamp01(features.pauses?.ratio ?? 0.3);

  // 与后端 acoustic_to_emotion_vector 同源（改动必须两边同步）
  const happy = 0.3 * energy + 0.3 * pitchRange + 0.2 * (1 - Math.abs(speechRate - 0.5));
  const sad = 0.3 * (1 - energy) + 0.3 * (1 - pitchRange) + 0.4 * pauseRatio;
  const anxious = 0.4 * speechRate + 0.3 * energy + 0.3 * (1 - Math.abs(pitchRange - 0.5));
  const angry = 0.4 * energy + 0.3 * pitchRange + 0.3 * (1 - pauseRatio);
  const neutral =
    0.2 * (1 - Math.abs(energy - 0.4)) +
    0.2 * (1 - Math.abs(pitchRange - 0.3)) +
    0.2 * (1 - Math.abs(speechRate - 0.4)) +
    0.4 * (1 - Math.abs(pauseRatio - 0.3));

  const total = happy + sad + anxious + angry + neutral;
  if (total < 1e-10) return [0.2, 0.2, 0.2, 0.2, 0.2];
  return [happy, sad, anxious, angry, neutral].map((x) => Math.max(0, x / total));
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
/**
 * 后端缺失时的兜底判定。
 *
 * ⚠️ 判据必须与 `assessment/multimodal_incongruence.py` 的
 * `classify_incongruence` 保持一致（同一套阈值、同一套通道方向规则）：
 * 主判据是**通道方向性对比**，不是 MIS 大小 —— 实测 MIS 在本项目的特征空间里
 * 与"是否矛盾"反相关（一致样本 0.59 > 掩饰样本 0.27）。改动请同步两边。
 */
const TEXT_CALM_MAX = 0.45;
const TEXT_DISTRESS_MIN = 0.55;
const VOICE_AROUSAL_HIGH = 0.62;
const VOICE_AROUSAL_LOW = 0.46;

function detectIncongruence(
  textProbs: number[],
  voiceProbs: number[]
): { type: 'congruent' | 'masking' | 'performative'; mis: number; description: string } {
  const mis = 1 - cosineSimilarity(textProbs, voiceProbs);

  const textNegative = textProbs[1] + textProbs[2] + textProbs[3]; // 悲伤 + 焦虑 + 愤怒
  const voiceArousal = voiceProbs[0] + voiceProbs[2] + voiceProbs[3]; // 快乐 + 焦虑 + 愤怒

  // 掩饰型：言语平静 ∧ 声音紧张
  if (textNegative <= TEXT_CALM_MAX && voiceArousal >= VOICE_AROUSAL_HIGH) {
    return {
      type: 'masking',
      mis,
      description: '言语平静但声音紧张，可能在隐藏真实情绪',
    };
  }
  // 表演型：言语痛苦 ∧ 声音平静
  if (textNegative >= TEXT_DISTRESS_MIN && voiceArousal <= VOICE_AROUSAL_LOW) {
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
  const result = useMemo<IncongruenceState | null>(() => {
    // ① 后端 TAMER 结果优先（唯一权威源）
    if (tamerResult) {
      const mis = tamerResult.mis;
      // ⚠️ 本轮没有真实声学输入（纯文字聊天）时，后端的 MIS 是用缺省声学值算出来的，
      // 不构成跨模态证据 —— 不能渲染成"言语与声音一致"。
      if (tamerResult.has_voice_input === false) {
        return {
          type: 'no_voice',
          mis,
          description: '本轮没有语音输入，未做跨模态矛盾检测',
        };
      }
      const type = tamerResult.incongruence_type;
      if (type === 'masking') {
        return { type: 'masking', mis, description: '言语平静但声音紧张，可能是在掩饰真实情绪' };
      }
      if (type === 'performative') {
        return { type: 'performative', mis, description: '言语痛苦但声音平淡，表达可能与真实感受不一致' };
      }
      return { type: 'congruent', mis, description: '言语与声音情绪一致' };
    }

    // ② 无后端结果：只有拿到**真实**声学特征才自算。
    //    绝不"用文本加噪声伪造一条声音向量"—— 那是把凭空造的数据呈现成测量结果。
    if (!textEmotionProbs || !voiceFeatures) {
      return {
        type: 'no_voice',
        mis: 0,
        description: '暂未取到算法结果或语音输入，未做跨模态矛盾检测',
      };
    }

    const voiceProbs = acousticToEmotionVector(voiceFeatures);
    return detectIncongruence(textEmotionProbs, voiceProbs);
  }, [textEmotionProbs, voiceFeatures, tamerResult]);

  if (!result) return null;

  // 没做检测：中性提示，不给任何"MIS 数值"以免被当成测量结果
  if (result.type === 'no_voice') {
    return (
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '6px 12px',
          background: '#fafafa',
          borderRadius: 8,
          border: '1px solid #e8e8e8',
        }}
      >
        <AudioOutlined style={{ color: '#8c8c8c', fontSize: 14 }} />
        <Text style={{ fontSize: 12, color: '#8c8c8c' }}>{result.description}</Text>
      </div>
    );
  }

  if (result.type === 'congruent') {
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
          言语与声音一致 · MIS: {result.mis.toFixed(2)}
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
