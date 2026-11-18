/**
 * CUSUM 时序轨迹图 —— 创新点③ 可视化
 *
 * 基于 CUSUM 变点检测算法追踪情绪时序变化。
 * 算法：S_t = max(0, S_{t-1} + (x_t - μ₀ - δ))
 *
 * 检测方向：
 * - S_t > h → 恶化（deteriorating）
 * - S_t < -h → 好转（improving）
 * - 其他 → 稳定（stable）
 */

import { useMemo } from 'react';
import { Typography, Tag } from 'antd';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  AreaChart,
} from 'recharts';
import {
  UpOutlined,
  DownOutlined,
  SwapOutlined,
} from '@ant-design/icons';

const { Text } = Typography;

interface Message {
  id: string;
  role: string;
  content: string;
  createdAt: string;
  sentiment?: string;
}

interface CUSUMTrajectoryChartProps {
  messages: Message[];
  /** TAMER 框架输出（从后端响应获取）。若提供则直接使用，否则前端自算 */
  tamerResult?: {
    trajectory: string;
  } | null;
}

/** 从消息中提取情绪分数（负性情绪强度） */
function extractEmotionScore(message: Message): number {
  if (!message.sentiment) return 0.3; // 默认中性
  try {
    const parsed = JSON.parse(message.sentiment);
    const emotionProbs = parsed.emotion_probs || [0.2, 0.2, 0.2, 0.2, 0.2];
    // 负性情绪 = 悲伤 + 焦虑 + 愤怒
    return emotionProbs[1] + emotionProbs[2] + emotionProbs[3];
  } catch {
    return 0.3;
  }
}

/** CUSUM 变点检测 */
function cusumDetect(
  scores: number[],
  baselineMean: number = 0.3,
  slack: number = 0.05,
  thresholdMultiplier: number = 4
): { cusumPos: number[]; cusumNeg: number[]; direction: string; changePoints: number[] } {
  const n = scores.length;
  if (n === 0) return { cusumPos: [], cusumNeg: [], direction: 'stable', changePoints: [] };

  // 计算基线标准差
  const mean = scores.reduce((a, b) => a + b, 0) / n;
  const variance = scores.reduce((sum, x) => sum + (x - mean) ** 2, 0) / Math.max(1, n);
  const std = Math.sqrt(variance) || 0.1;
  const threshold = thresholdMultiplier * std;

  const cusumPos: number[] = [0];
  const cusumNeg: number[] = [0];
  const changePoints: number[] = [];

  for (let t = 1; t < n; t++) {
    const x = scores[t];
    const pos = Math.max(0, cusumPos[t - 1] + (x - baselineMean - slack));
    const neg = Math.max(0, cusumNeg[t - 1] - (x - baselineMean + slack));
    cusumPos.push(pos);
    cusumNeg.push(neg);

    if (pos > threshold || neg > threshold) {
      changePoints.push(t);
    }
  }

  // 判定方向
  const lastPos = cusumPos[n - 1];
  const lastNeg = cusumNeg[n - 1];
  let direction = 'stable';
  if (lastPos > threshold) direction = 'deteriorating';
  else if (lastNeg > threshold) direction = 'improving';

  return { cusumPos, cusumNeg, direction, changePoints };
}

/**
 * 把后端 CUSUM 的方向取值归一化到本组件使用的三档。
 *
 * ⚠️ 两边词表不同源：`assessment/temporal_trajectory.py` 产出的是
 * `worsening / improving / stable / unknown / none`，而这里的
 * `getDirectionInfo` 只认 `deteriorating / improving / stable`。
 * 不归一化的话 `worsening` 会落进 default 分支、把"恶化"显示成"稳定"。
 */
function normalizeDirection(
  trajectory?: string | null,
): 'deteriorating' | 'improving' | 'stable' | null {
  if (!trajectory) return null;
  switch (trajectory) {
    case 'worsening':
    case 'deteriorating':
      return 'deteriorating';
    case 'improving':
      return 'improving';
    default:
      return 'stable';
  }
}

function getDirectionInfo(direction: string) {
  switch (direction) {
    case 'deteriorating':
      return {
        label: '恶化趋势',
        color: '#ff4d4f',
        icon: <UpOutlined />,
        desc: 'CUSUM 检测到持续恶化，建议关注',
      };
    case 'improving':
      return {
        label: '好转趋势',
        color: '#52c41a',
        icon: <DownOutlined />,
        desc: 'CUSUM 检测到持续好转',
      };
    default:
      return {
        label: '稳定',
        color: '#1890ff',
        icon: <SwapOutlined />,
        desc: '情绪波动在正常范围内',
      };
  }
}

export default function CUSUMTrajectoryChart({ messages, tamerResult }: CUSUMTrajectoryChartProps) {
  const chartData = useMemo(() => {
    const userMessages = messages.filter((m) => m.role === 'user');
    if (userMessages.length < 2) return [];

    const scores = userMessages.map(extractEmotionScore);
    const { cusumPos, cusumNeg, direction, changePoints } = cusumDetect(scores);

    return userMessages.map((msg, i) => ({
      turn: i + 1,
      score: scores[i],
      cusumPos: cusumPos[i],
      cusumNeg: cusumNeg[i],
      isChangePoint: changePoints.includes(i),
      date: msg.createdAt?.slice(5, 10) || `T${i + 1}`,
    }));
  }, [messages]);

  // 前端自算的兜底方向（后端没给 tamerResult 时用）
  const localDirection = useMemo(() => {
    if (chartData.length === 0) return 'stable';
    const lastPos = chartData[chartData.length - 1]?.cusumPos ?? 0;
    const lastNeg = chartData[chartData.length - 1]?.cusumNeg ?? 0;
    const threshold = 0.4; // 简化阈值
    if (lastPos > threshold) return 'deteriorating';
    if (lastNeg > threshold) return 'improving';
    return 'stable';
  }, [chartData]);

  // 优先使用后端 TAMER 结果，否则前端自算。
  //
  // ⚠️ Hook 必须**无条件**调用：原写法 `tamerResult?.trajectory ?? useMemo(...)`
  // 里 `??` 右侧是懒求值的，一旦 tamerResult 有值这一帧就少调一个 Hook ——
  // React 会抛 "Rendered fewer hooks than expected"。此前后端恒不返回
  // tamerResult（Node 层没透传），所以"因为没数据才没炸"，数据一接通必炸。
  // ⚠️ 后端取值需归一化，见 `normalizeDirection`。
  const direction = normalizeDirection(tamerResult?.trajectory) ?? localDirection;

  const dirInfo = getDirectionInfo(direction);

  if (chartData.length < 2) {
    return (
      <div style={{ padding: 20, textAlign: 'center' }}>
        <Text type="secondary">至少需要 2 轮对话才能生成轨迹图</Text>
      </div>
    );
  }

  return (
    <div>
      {/* 方向指示器 */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          padding: '8px 14px',
          background: `${dirInfo.color}11`,
          borderRadius: 10,
          border: `1px solid ${dirInfo.color}33`,
          marginBottom: 16,
        }}
      >
        <div style={{ fontSize: 18, color: dirInfo.color }}>{dirInfo.icon}</div>
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Tag color={dirInfo.color} style={{ margin: 0, fontSize: 11 }}>
              {dirInfo.label}
            </Tag>
            <Text style={{ fontSize: 11, color: '#888' }}>CUSUM 变点检测</Text>
          </div>
          <Text style={{ fontSize: 12, color: '#666' }}>{dirInfo.desc}</Text>
        </div>
      </div>

      {/* 情绪轨迹图 */}
      <ResponsiveContainer width="100%" height={200}>
        <AreaChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
          <defs>
            <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis
            dataKey="turn"
            tick={{ fontSize: 11 }}
            label={{ value: '对话轮次', position: 'bottom', offset: -5, fontSize: 11 }}
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fontSize: 11 }}
            label={{ value: '负性情绪', angle: -90, position: 'left', offset: 10, fontSize: 11 }}
          />
          <Tooltip
            contentStyle={{ borderRadius: 8, fontSize: 12 }}
            formatter={(value: number, name: string) => {
              if (name === 'score') return [value.toFixed(2), '负性情绪强度'];
              if (name === 'cusumPos') return [value.toFixed(2), 'CUSUM+'];
              return [value.toFixed(2), name];
            }}
          />
          <ReferenceLine y={0.5} stroke="#faad14" strokeDasharray="3 3" label={{ value: '警戒线', fontSize: 10 }} />
          <Area
            type="monotone"
            dataKey="score"
            stroke="#6366f1"
            strokeWidth={2}
            fill="url(#scoreGradient)"
            dot={(props: any) => {
              const { cx, cy, payload } = props;
              if (payload.isChangePoint) {
                return (
                  <circle
                    key={`cp-${payload.turn}`}
                    cx={cx}
                    cy={cy}
                    r={5}
                    fill="#ff4d4f"
                    stroke="#fff"
                    strokeWidth={2}
                  />
                );
              }
              return (
                <circle
                  key={`dot-${payload.turn}`}
                  cx={cx}
                  cy={cy}
                  r={3}
                  fill="#6366f1"
                  stroke="#fff"
                  strokeWidth={1}
                />
              );
            }}
          />
        </AreaChart>
      </ResponsiveContainer>

      {/* 变点标记说明 */}
      {chartData.some((d) => d.isChangePoint) && (
        <div style={{ marginTop: 8, fontSize: 11, color: '#888', textAlign: 'center' }}>
          <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: '#ff4d4f', marginRight: 4 }} />
          红点 = CUSUM 检测到的变点
        </div>
      )}
    </div>
  );
}
