/**
 * 关系阶段指示器 —— 创新点① RQI 可视化
 *
 * 基于对话历史计算关系质量指数（RQI），展示治疗联盟阶段。
 * 算法：RQI(t) = α × RQI(t-1) + (1-α) × Σ(wᵢ × signalᵢ)
 *
 * 阶段划分：
 * - RQI < 0.3 → 陌生期（Stranger）
 * - 0.3 ≤ RQI < 0.6 → 熟悉期（Acquainted）
 * - RQI ≥ 0.6 → 信任期（Trusted）
 */

import { useMemo } from 'react';
import { Progress, Tag, Tooltip, Typography } from 'antd';
import {
  TeamOutlined,
  HeartOutlined,
  SafetyCertificateOutlined,
} from '@ant-design/icons';

const { Text } = Typography;

interface Message {
  id: string;
  role: string;
  content: string;
  createdAt: string;
}

interface RelationshipStageIndicatorProps {
  messages: Message[];
  /** TAMER 框架输出（从后端响应获取）。若提供则直接使用，否则前端自算 */
  tamerResult?: {
    rqi: number;
    relationship_depth: string;
  } | null;
}

/** 计算 RQI */
function computeRQI(messages: Message[]): { rqi: number; signals: Record<string, number> } {
  if (messages.length === 0) {
    return { rqi: 0, signals: {} };
  }

  const userMessages = messages.filter((m) => m.role === 'user');
  const sessionCount = userMessages.length;

  // Signal 1: 互动频率（对数归一化，10次对话达到满分）
  const interactionFreq = Math.min(1, Math.log(1 + sessionCount) / Math.log(11));

  // Signal 2: 自我暴露深度（平均消息长度 + 情感词汇密度）
  const avgLength =
    userMessages.reduce((sum, m) => sum + m.content.length, 0) / Math.max(1, sessionCount);
  const emotionalWords = ['难受', '痛苦', '烦', '累', '悲伤', '焦虑', '害怕', '孤独', '绝望', '崩溃'];
  const emotionalCount = userMessages.reduce((sum, m) => {
    return sum + emotionalWords.filter((w) => m.content.includes(w)).length;
  }, 0);
  const selfDisclosure = Math.min(1, avgLength / 100 + emotionalCount / Math.max(1, sessionCount) * 0.3);

  // Signal 3: 话题多样性（按句号分句，统计不同话题）
  const allSentences = userMessages.flatMap((m) => m.content.split(/[。！？]/).filter((s) => s.trim().length > 3));
  const topicDiversity = Math.min(1, allSentences.length / 20);

  // Signal 4: 信任信号（包含信任关键词的消息比例）
  const trustKeywords = ['秘密', '只有你知道', '你懂', '朋友', '信任', '相信'];
  const trustSignals = userMessages.filter((m) =>
    trustKeywords.some((kw) => m.content.includes(kw))
  ).length;
  const trustScore = Math.min(1, trustSignals / Math.max(1, sessionCount) * 3);

  // Signal 5: 情感共鸣（简化：假设 AI 能匹配用户情绪，给默认值）
  const emotionMatch = 0.7;

  // 加权融合
  const weights = {
    interactionFreq: 0.20,
    selfDisclosure: 0.25,
    topicDiversity: 0.15,
    trustScore: 0.25,
    emotionMatch: 0.15,
  };

  const rqi =
    interactionFreq * weights.interactionFreq +
    selfDisclosure * weights.selfDisclosure +
    topicDiversity * weights.topicDiversity +
    trustScore * weights.trustScore +
    emotionMatch * weights.emotionMatch;

  return {
    rqi: Math.min(1, Math.max(0, rqi)),
    signals: {
      '互动频率': interactionFreq,
      '自我暴露': selfDisclosure,
      '话题多样': topicDiversity,
      '信任信号': trustScore,
      '情感共鸣': emotionMatch,
    },
  };
}

function getStage(rqi: number): { label: string; color: string; icon: React.ReactNode; desc: string } {
  if (rqi >= 0.6) {
    return {
      label: '信任期',
      color: '#52c41a',
      icon: <SafetyCertificateOutlined />,
      desc: '用户已建立深度信任，AI 可使用更深入的干预策略',
    };
  }
  if (rqi >= 0.3) {
    return {
      label: '熟悉期',
      color: '#1890ff',
      icon: <HeartOutlined />,
      desc: '关系正在建立，AI 以共情倾听为主',
    };
  }
  return {
    label: '陌生期',
    color: '#faad14',
    icon: <TeamOutlined />,
    desc: '初次接触，AI 以温和引导为主',
  };
}

export default function RelationshipStageIndicator({ messages, tamerResult }: RelationshipStageIndicatorProps) {
  // 前端自算的兜底 RQI（后端没给 tamerResult 时用）
  const localRqi = useMemo(() => computeRQI(messages).rqi, [messages]);

  // 优先使用后端 TAMER 结果，否则前端自算。
  //
  // ⚠️ Hook 必须**无条件**调用：原写法 `tamerResult?.rqi ?? useMemo(...)`
  // 里 `??` 右侧是懒求值的，一旦 tamerResult 有值这一帧就少调一个 Hook ——
  // React 会抛 "Rendered fewer hooks than expected"。此前后端恒不返回
  // tamerResult（Node 层没透传），所以"因为没数据才没炸"，数据一接通必炸。
  const rqi = tamerResult?.rqi ?? localRqi;
  const stage = getStage(rqi);
  const percent = Math.round(rqi * 100);

  return (
    <Tooltip title={stage.desc}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          padding: '8px 16px',
          background: 'linear-gradient(135deg, #f0f5ff 0%, #e6f7ff 100%)',
          borderRadius: 12,
          border: `1px solid ${stage.color}33`,
        }}
      >
        <div style={{ fontSize: 20, color: stage.color }}>{stage.icon}</div>
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <Tag color={stage.color} style={{ margin: 0, fontSize: 11 }}>
              {stage.label}
            </Tag>
            <Text type="secondary" style={{ fontSize: 11 }}>
              RQI: {rqi.toFixed(2)}
            </Text>
          </div>
          <Progress
            percent={percent}
            size="small"
            strokeColor={stage.color}
            showInfo={false}
            style={{ margin: 0 }}
          />
        </div>
        <div style={{ textAlign: 'right', minWidth: 60 }}>
          <Text style={{ fontSize: 18, fontWeight: 'bold', color: stage.color }}>{percent}%</Text>
        </div>
      </div>
    </Tooltip>
  );
}
