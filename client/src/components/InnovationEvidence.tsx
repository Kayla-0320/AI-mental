import { Empty, Progress, Space, Tag, Typography } from 'antd';
import { Link, useParams } from 'react-router-dom';
import { expertLabel, type InnovationReference } from '../services/innovationPresentation';
export { default as QualityEvidence } from './QualityEvidence';
import './innovation-evidence.css';

const { Text, Paragraph } = Typography;
const conceptLabel = (value: string): string => ({ emotion_positive: '积极感受', emotion_sadness: '悲伤表达', emotion_fear: '担心与恐惧', emotion_anger: '愤怒表达',
  interpersonal_event: '人际事件', future_uncertainty: '未来不确定性', event: '情境事件', subjective_interpretation: '本人解读', subjective_feeling: '本人感受',
  others_true_intent: '他人的真实意图', observed_event: '已发生的事实', possible_reaction_to: '可能是对其的反应', possibly_related_to: '可能关联', has_possible_reaction: '可能引起', not_equivalent_to: '不等同于' }[value] || value);

export function FusionEvidence({ reference }: { reference: InnovationReference | null }) {
  if (!reference?.available) return <div className="ie-empty"><Empty image={Empty.PRESENTED_IMAGE_SIMPLE}
    description={reference ? '这一轮我还没有形成可展示的理解。你继续说说，我下一轮再留意。' : '等我们聊起来，这里会慢慢有内容。'} />
    <Paragraph type="secondary">等我们聊起来，这里会慢慢出现我是怎么听、怎么想的。</Paragraph></div>;
  const active = reference.expert_names.map((name, i) => ({ name, weight: reference.expert_weights[i] })).filter(item => item.weight > 0);
  return <div data-testid="fusion-evidence">
    <Space wrap className="ie-tags">
      <Tag color="blue">文字{reference.observed_modalities.includes('text') ? '已采用' : '未采集'}</Tag>
      <Tag color={reference.observed_modalities.includes('visual') ? 'green' : 'default'}>画面{reference.observed_modalities.includes('visual') ? '已采用' : '未采用'}</Tag>
      <Tag>声音{reference.observed_modalities.includes('audio') ? '已采用' : '未作为专家输入'}</Tag>
    </Space>
    <p className="ie-label">这一轮，我是怎么听的</p>
    {active.length ? active.map(item => <div className="ie-expert" key={item.name}>
      <Text>{expertLabel(item.name)}</Text><Progress percent={Number((item.weight * 100).toFixed(1))} size="small" strokeColor="#8b7fc6" />
    </div>) : <Text type="secondary">这一轮没有留下可读的参考比例。</Text>}
    <Paragraph type="secondary" className="ie-note">这些比例只说明我参考了哪些线索，不代表给你下判断，也不是诊断。</Paragraph>
    <p className="ie-label">我参考了哪些知识</p>
    {reference.knowledge_matches.length ? reference.knowledge_matches.map(entry => <div className="ie-knowledge" key={entry.id}>
      <Space size={6} wrap><Tag>{entry.review_status === 'expert_reviewed' ? '已记录专家审核' : entry.review_status === 'user_attested' ? '用户背书 · 未独立审核' : '待审核候选'}</Tag>
        {/^https:\/\//i.test(entry.source_url) && <a href={entry.source_url} target="_blank" rel="noreferrer">查看概念来源</a>}</Space>
      <Paragraph style={{ margin: '8px 0 0' }}>{entry.text}</Paragraph>
      {entry.relations.length > 0 && <div className="ie-relations" aria-label="命中概念的关系依据">{entry.relations.map((relation, i) => <div key={i}>
        <span>{conceptLabel(entry.id)}</span><small>— {conceptLabel(relation.predicate)} →</small><span>{conceptLabel(relation.target)}</span>
      </div>)}</div>}
    </div>) : <Paragraph type="secondary">这一轮我没有套用现成的知识概念，也不会为了显得懂你而硬编一个。</Paragraph>}
    <details className="ie-details"><summary>看看技术细节</summary>
      <Paragraph type="secondary" style={{ wordBreak: 'break-all', marginTop: 8 }}>权重摘要：{reference.model_id || '未记录'}</Paragraph>
      <Text type="secondary">这些技术细节只是辅助我理解你；它还没有在和你一样的人身上验证过，所以你的感受永远更算数。</Text>
    </details>
  </div>;
}

export function CompanionLink({ onClick }: { onClick?: () => void } = {}) {
  const { conversationId } = useParams();
  return <Link to={`${conversationId ? `/chat/${encodeURIComponent(conversationId)}` : '/chat'}?panel=tracking`} onClick={onClick}>看看我最近记得你什么 →</Link>;
}
