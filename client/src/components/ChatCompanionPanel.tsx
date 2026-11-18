import { Tabs, Typography } from 'antd';
import { FusionEvidence, QualityEvidence } from './InnovationEvidence';
import PersonalSupportAgent from './PersonalSupportAgent';
import type { InnovationReference } from '../services/innovationPresentation';
import './chat-companion.css';
import type { LiveQualityState } from './LiveQualityPreview';

export type CompanionTab = 'fusion' | 'quality' | 'tracking';

/** 陪伴属于当前倾诉工作区；切换能力时保持聊天与支持活动挂载。 */
export default function ChatCompanionPanel({ reference, recordedAt, waiting, activeTab, onTabChange, refreshKey, liveQuality }: {
  reference: InnovationReference | null; recordedAt?: string; waiting: boolean;
  activeTab: CompanionTab; onTabChange: (tab: CompanionTab) => void; refreshKey: string;
  liveQuality?: LiveQualityState;
}) {
  return <div className="chat-companion-panel" data-testid="chat-companion-panel">
    <Typography.Paragraph type="secondary" className="chat-companion-intro">
      我会留意你说的话，也会把我是怎么理解的放在这里。理解得不对，你随时改。
    </Typography.Paragraph>
    <Tabs activeKey={activeTab} onChange={key => onTabChange(key as CompanionTab)} size="small" items={[
      { key: 'fusion', label: '图文理解', children: <>
        <Typography.Title level={5}>我是怎么理解你的</Typography.Title>
        <Typography.Paragraph type="secondary" className="chat-companion-source">{waiting ? '等你这一轮说完，我再看。' : recordedAt ? `这一轮 · ${new Date(recordedAt).toLocaleString('zh-CN')}` : '等我们聊起来，这里会慢慢有内容。'}</Typography.Paragraph>
        <FusionEvidence reference={reference} />
      </> },
      { key: 'quality', label: '质量纠偏', children: <>
        <Typography.Title level={5}>画面清不清楚，一眼就知道</Typography.Title>
        <Typography.Paragraph type="secondary" className="chat-companion-source">{recordedAt && !waiting ? `这一轮 · ${new Date(recordedAt).toLocaleString('zh-CN')}` : '实时预览随摄像头更新；下方是每轮对话的质量检查记录。'}</Typography.Paragraph>
        <QualityEvidence reference={reference} waiting={waiting} live={liveQuality} active={activeTab === 'quality'} />
      </> },
      { key: 'tracking', label: '持续陪伴', forceRender: true, children: <>
        <Typography.Title level={5}>我会记得你</Typography.Title>
        <PersonalSupportAgent embedded refreshKey={refreshKey} />
      </> },
    ]} />
  </div>;
}
