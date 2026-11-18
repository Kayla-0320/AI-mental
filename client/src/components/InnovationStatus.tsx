import { useState } from 'react';
import { Button, Drawer, Space, Tag, Typography } from 'antd';
import { innovationFromSentiment } from '../services/innovationPresentation';
import { FusionEvidence, QualityEvidence, CompanionLink } from './InnovationEvidence';

/** 说明实际采用的证据及质量降级，不展示内部分类或心理断言。 */
export default function InnovationStatus({ sentiment }: { sentiment?: string }) {
  const [open, setOpen] = useState(false);
  const reference = innovationFromSentiment(sentiment);
  if (!reference) return null;
  const visual = reference.observed_modalities?.includes('visual');
  const fallback = reference.warnings?.some(item => item.startsWith('visual_') && item.includes('fallback'));
  const listening = !reference.available ? '这次我听得有点不确定'
    : visual ? '我也留意了你的画面'
      : fallback ? '这次主要听你说'
        : '我在听你说';
  return <><div className="ie-status" data-testid="innovation-turn-status">
    <Tag color={fallback ? 'orange' : reference.available ? 'purple' : 'default'}>{listening}</Tag>
    <Typography.Text type="secondary" style={{ fontSize: 12 }}>你说的是准的</Typography.Text>
    <Button type="link" size="small" onClick={() => setOpen(true)}>为什么这样理解</Button>
  </div><Drawer title="我为什么这样理解你" open={open} onClose={() => setOpen(false)} width={Math.min(560, window.innerWidth)}>
    <Typography.Title level={5}>01 · 我是怎么理解你的</Typography.Title><FusionEvidence reference={reference} />
    <Typography.Title level={5} style={{ marginTop: 28 }}>02 · 我听得清不清楚</Typography.Title><QualityEvidence reference={reference} />
    <Space style={{ marginTop: 24 }}><CompanionLink onClick={() => setOpen(false)} /></Space>
  </Drawer></>;
}
