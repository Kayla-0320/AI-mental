/**
 * 对话关系参考 —— 把研究性的 RQI 重新放回聊天顶部。
 *
 * RQI 只看交流轮次、表达长度、话题与信任词等可观察信号，用来提示“我们聊天的节奏”，
 * 不是心理测量，也不代表信任程度、疗效或诊断。界面上保留数值与进度，同时用一句话
 * 和 tooltip 说清它的边界，避免把研究指标包装成“信任度”。
 */
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Progress, Tag, Tooltip, Typography } from 'antd';
import { TeamOutlined, HeartOutlined, SafetyCertificateOutlined } from '@ant-design/icons';

const { Text } = Typography;

interface Message {
  id: string;
  role: string;
  content: string;
  createdAt: string;
  sentiment?: string;
}

interface Props {
  messages: Message[];
}

/** 只读当前会话已落库的 EMA 结果；等待回复时沿用上一轮，首轮从 0 开始。 */
function latestConfirmedRQI(messages: Message[]): number {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role !== 'assistant' || !message.sentiment) continue;
    try {
      const rqi: unknown = JSON.parse(message.sentiment)?.tamer_result?.rqi;
      if (typeof rqi === 'number' && Number.isFinite(rqi) && rqi >= 0 && rqi <= 1) return rqi;
    } catch { /* 旧记录缺少或损坏的元数据不覆盖上一轮有效读数。 */ }
  }
  return 0;
}

function getStage(rqi: number): { label: string; color: string; icon: ReactNode; desc: string } {
  if (rqi >= 0.6) return {
    label: '更自在一些', color: '#7bb89a', icon: <SafetyCertificateOutlined />,
    desc: '你可以决定聊多深，我跟着你。',
  };
  if (rqi >= 0.3) return {
    label: '在熟悉', color: '#9b81c6', icon: <HeartOutlined />,
    desc: '我们慢慢找到说话的节奏。',
  };
  return {
    label: '刚认识', color: '#bfa3dd', icon: <TeamOutlined />,
    desc: '先从轻松的话题开始，不着急。',
  };
}

export default function RelationshipStageIndicator({ messages }: Props) {
  const targetRqi = useMemo(() => latestConfirmedRQI(messages), [messages]);
  const [rqi, setRqi] = useState(targetRqi);
  const displayedRqi = useRef(targetRqi);
  useEffect(() => {
    const startValue = displayedRqi.current;
    if (startValue === targetRqi) return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      displayedRqi.current = targetRqi;
      setRqi(targetRqi);
      return;
    }
    // 数字和进度共用同一条 900ms 渐变，避免数值先跳、进度条后追。
    const startedAt = performance.now();
    let frame: number;
    const animate = (now: number): void => {
      const progress = Math.max(0, Math.min(1, (now - startedAt) / 900));
      const value = startValue + (targetRqi - startValue) * (1 - (1 - progress) ** 2);
      displayedRqi.current = value;
      setRqi(value);
      if (progress < 1) frame = requestAnimationFrame(animate);
    };
    frame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frame);
  }, [targetRqi]);
  const count = messages.filter(message => message.role === 'user').length;
  if (!count) return null;
  const stage = getStage(rqi);
  const percent = Math.round(rqi * 100);

  return <Tooltip title={<div>
    <div>{stage.desc}</div>
    <div style={{ marginTop: 4 }}>RQI 是根据交流轮次、表达长度等算出的研究性参考，只用来看看聊天的节奏；不是心理测量，也不代表信任程度或治疗效果。</div>
  </div>}>
    <div data-testid="relationship-stage" style={{
      display: 'flex', flexDirection: 'column', gap: 5,
      padding: '8px 12px', borderRadius: 12,
      background: 'linear-gradient(135deg, #faf7ff 0%, #fff9fc 100%)',
      border: '1px solid #efe6f7',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
        <span style={{ color: stage.color, fontSize: 16, display: 'inline-flex' }}>{stage.icon}</span>
        <Tag color={stage.color} style={{ margin: 0, fontSize: 11, borderRadius: 999 }}>{stage.label}</Tag>
        <Text type="secondary" style={{ fontSize: 11 }}>RQI {rqi.toFixed(2)}</Text>
        <div style={{ flex: 1, minWidth: 90 }}><Progress percent={percent} size="small" strokeColor={stage.color} showInfo={false} style={{ margin: 0 }} /></div>
        <Text style={{ fontSize: 16, fontWeight: 600, color: stage.color }}>{percent}%</Text>
      </div>
      <div style={{ fontSize: 11, color: '#9a8aae' }}>
        我们已经聊了 {count} 轮 · 想换个话题也没关系 · RQI 是对话节奏的参考
      </div>
    </div>
  </Tooltip>;
}
