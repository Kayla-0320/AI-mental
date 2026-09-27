import { useEffect, useState } from 'react';
import { Card, Row, Col, Typography, Tag, Statistic, Spin, Empty, Button, Input, Slider, Space, Alert, Divider } from 'antd';
import { ThunderboltOutlined, ExperimentOutlined, HistoryOutlined, QuestionCircleOutlined } from '@ant-design/icons';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer } from 'recharts';
import { PERCEPTION_API } from '../../config';
import api from '../../services/api';

const { Title, Text, Paragraph } = Typography;

const DIM_LABELS: Record<string, string> = {
  text_emotion: '文本情绪', voice_acoustic: '语音声学', facial: '面部表情',
  circadian: '昼夜节律', cognitive: '认知扭曲', behavior: '行为模式',
  hrv: '心率变异性', breathing: '呼吸模式', behavioral_act: '行为激活',
  eye: '眨眼/头姿', voice_semantics: '语音语义',
};

export default function MyDigitalTwin() {
  const [userId, setUserId] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [landscape, setLandscape] = useState<any>(null);
  const [futures, setFutures] = useState<any>(null);
  const [counterfactual, setCounterfactual] = useState<any>(null);
  const [cfQuery, setCfQuery] = useState<{ dim: string; value: number }>({ dim: 'circadian', value: 0.7 });

  useEffect(() => {
    loadUserId();
  }, []);

  const loadUserId = async () => {
    try {
      const res: any = await api.get('/auth/profile');
      // API 返回结构：{ code: 0, data: { id: '...', ... } }
      setUserId(res.data?.id || res.data?.user?.id || '');
    } catch {}
  };

  const loadData = async () => {
    if (!userId) return;
    setLoading(true);
    try {
      const post = (path: string, body: any) =>
        fetch(`/algorithm/api/v1/digital-twin/${path}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_id: userId, ...body }),
        }).then(r => (r.ok ? r.json() : null)).catch(() => null);
      const [land, fut] = await Promise.all([
        post('landscape', { horizon: 12 }),
        post('futures', { n_particles: 200, horizon: 20, seed: 0 }),
      ]);
      if (land) setLandscape(land);
      if (fut) setFutures(fut);
    } catch {}
    finally { setLoading(false); }
  };

  const runCounterfactual = async () => {
    if (!userId) return;
    setLoading(true);
    try {
      const res = await fetch('/algorithm/api/v1/digital-twin/counterfactual', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          do_spec: { [cfQuery.dim]: cfQuery.value },
          horizon: 8,
          clamp: true,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setCounterfactual(data);
      }
    } catch {}
    finally { setLoading(false); }
  };

  useEffect(() => {
    if (userId) loadData();
  }, [userId]);

  if (!userId) {
    return <div style={{ padding: 40, textAlign: 'center' }}><Spin tip="加载用户信息..." /></div>;
  }

  return (
    <div style={{ padding: '24px 32px', maxWidth: 1200, margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <div>
          <Title level={3} style={{ margin: 0 }}>
            <ThunderboltOutlined style={{ color: '#722ed1', marginRight: 8 }} />
            我的数字孪生
          </Title>
          <Text type="secondary">AI 在虚拟世界里为你模拟未来，帮你预见风险、探索改变</Text>
        </div>
        <Button icon={<HistoryOutlined />} loading={loading} onClick={loadData}>刷新</Button>
      </div>

      {loading && !landscape && !futures ? (
        <div style={{ textAlign: 'center', padding: 60 }}><Spin size="large" tip="数字孪生推演中..." /></div>
      ) : (
        <>
          {/* 危机风险概览 */}
          <Card style={{ borderRadius: 12, marginBottom: 20, borderLeft: '4px solid #722ed1' }}>
            <Title level={4} style={{ marginTop: 0 }}>
              <ExperimentOutlined style={{ color: '#722ed1', marginRight: 8 }} />
              未来 12 小时风险预测
            </Title>
            {landscape && !landscape.fallback ? (
              <Row gutter={[16, 16]}>
                <Col xs={12} md={6}>
                  <Statistic
                    title="崩溃风险概率"
                    value={Math.round((landscape.crisis_probability || 0) * 100)}
                    suffix="%"
                    valueStyle={{
                      color: (landscape.crisis_probability || 0) > 0.3 ? '#ff4d4f' : (landscape.crisis_probability || 0) > 0.15 ? '#faad14' : '#52c41a',
                      fontSize: 28,
                    }}
                  />
                </Col>
                <Col xs={12} md={6}>
                  <Statistic title="心理韧性" value={landscape.barrier_height?.toFixed(2) || '—'} valueStyle={{ fontSize: 24, color: '#722ed1' }} />
                </Col>
                <Col xs={12} md={6}>
                  <Statistic title="潜空间维度" value={landscape.n_dim || 0} suffix={`解释${Math.round((landscape.latent_var_explained || 0) * 100)}%`} valueStyle={{ fontSize: 24 }} />
                </Col>
                <Col xs={12} md={6}>
                  <Statistic title="梯度/通量比" value={landscape.gradient_flux_ratio?.toFixed(2) || '—'} valueStyle={{ fontSize: 24 }} />
                </Col>
              </Row>
            ) : (
              <Alert
                type="info"
                message="数据积累中"
                description="需要更多历史数据才能激活多维风险预测。当前使用基础模型。"
                showIcon
              />
            )}
          </Card>

          {/* 崩溃 domino 链 */}
          {landscape?.collapse_order?.length > 0 && (
            <Card style={{ borderRadius: 12, marginBottom: 20, background: '#faf7ff' }}>
              <Title level={5} style={{ marginTop: 0, color: '#531dab' }}>
                ⚠️ 崩溃 domino 链
              </Title>
              <Text type="secondary" style={{ display: 'block', marginBottom: 12 }}>
                如果风险持续恶化，以下方面可能按顺序受到影响：
              </Text>
              <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 8 }}>
                {landscape.collapse_order.map((m: string, i: number) => (
                  <span key={i} style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                    {i > 0 && <span style={{ color: '#bbb', fontSize: 16 }}>→</span>}
                    <Tag color={i === 0 ? 'red' : i === 1 ? 'volcano' : 'purple'} style={{ fontSize: 13, padding: '4px 12px' }}>
                      {i + 1}. {DIM_LABELS[m] || m}
                    </Tag>
                  </span>
                ))}
              </div>
              <Text type="secondary" style={{ fontSize: 12, display: 'block', marginTop: 12 }}>
                💡 提前干预排在前面的方面，可以有效阻断后续连锁反应
              </Text>
            </Card>
          )}

          {/* 平行未来 */}
          {futures && !futures.fallback && (
            <Card style={{ borderRadius: 12, marginBottom: 20, background: '#f5faff' }}>
              <Title level={5} style={{ marginTop: 0, color: '#0958d9' }}>
                🌀 平行未来推演
              </Title>
              <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
                <Col xs={8}>
                  <Statistic title="模拟轨迹数" value={futures.n_particles || 0} valueStyle={{ fontSize: 20 }} />
                </Col>
                <Col xs={8}>
                  <Statistic
                    title="危机率"
                    value={Math.round((futures.crisis_rate || 0) * 100)}
                    suffix="%"
                    valueStyle={{ fontSize: 20, color: (futures.crisis_rate || 0) > 0.3 ? '#ff4d4f' : '#faad14' }}
                  />
                </Col>
                <Col xs={8}>
                  <Statistic title="平均首崩时间" value={futures.time_to_crisis_mean?.toFixed(1) || '—'} suffix="步" valueStyle={{ fontSize: 20 }} />
                </Col>
              </Row>
              <Text type="secondary" style={{ fontSize: 12 }}>
                AI 模拟了 {futures.n_particles} 条可能的未来轨迹，其中 {Math.round((futures.crisis_rate || 0) * 100)}% 会进入危机状态
              </Text>
            </Card>
          )}

          {/* 反事实查询 */}
          <Card style={{ borderRadius: 12, marginBottom: 20, background: '#f6ffed' }}>
            <Title level={5} style={{ marginTop: 0, color: '#389e0d' }}>
              <QuestionCircleOutlined style={{ marginRight: 8 }} />
              如果...会怎样？
            </Title>
            <Paragraph type="secondary">
              探索改变某个方面后，你的心理状态会如何变化。例如："如果我每天多睡 2 小时"或"如果我减少反刍思维"
            </Paragraph>

            <div style={{ marginBottom: 16 }}>
              <Text strong style={{ display: 'block', marginBottom: 8 }}>选择想改变的方面：</Text>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {Object.entries(DIM_LABELS).map(([key, label]) => (
                  <Tag
                    key={key}
                    color={cfQuery.dim === key ? 'green' : 'default'}
                    style={{ cursor: 'pointer', fontSize: 13 }}
                    onClick={() => setCfQuery({ ...cfQuery, dim: key })}
                  >
                    {label}
                  </Tag>
                ))}
              </div>
            </div>

            <div style={{ marginBottom: 16 }}>
              <Text strong style={{ display: 'block', marginBottom: 8 }}>
                改变程度：{Math.round(cfQuery.value * 100)}%
              </Text>
              <Slider
                min={0}
                max={1}
                step={0.05}
                value={cfQuery.value}
                onChange={(v) => setCfQuery({ ...cfQuery, value: v })}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#999' }}>
                <span>维持现状</span>
                <span>显著改善</span>
              </div>
            </div>

            <Button type="primary" icon={<ExperimentOutlined />} loading={loading} onClick={runCounterfactual}>
              模拟这个改变
            </Button>

            {counterfactual && (
              <div style={{ marginTop: 16, padding: 12, background: '#fff', borderRadius: 8, border: '1px solid #b7eb8f' }}>
                <Text strong style={{ color: '#389e0d' }}>模拟结果：</Text>
                <div style={{ marginTop: 8, fontSize: 13, lineHeight: 1.6 }}>
                  {counterfactual.fallback ? (
                    <Text type="secondary">数据不足，无法模拟。需要更多历史数据。</Text>
                  ) : (
                    <>
                      <Text>
                        如果将「{DIM_LABELS[cfQuery.dim] || cfQuery.dim}」改善到 {Math.round(cfQuery.value * 100)}%，
                        未来 8 步的心理状态会有所改善。
                      </Text>
                      {counterfactual.effect && (
                        <div style={{ marginTop: 8 }}>
                          <Text type="secondary">预期效果：</Text>
                          <div style={{ marginTop: 4 }}>
                            {Object.entries(counterfactual.effect).map(([dim, val]: [string, any]) => (
                              <Tag key={dim} color={(val as number) > 0 ? 'green' : 'red'} style={{ marginBottom: 4 }}>
                                {DIM_LABELS[dim] || dim}: {(val as number) > 0 ? '+' : ''}{(val as number).toFixed(2)}
                              </Tag>
                            ))}
                          </div>
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>
            )}
          </Card>

          {/* 模态耦合关系 */}
          {landscape?.coupling?.fitted && (
            <Card style={{ borderRadius: 12, marginBottom: 20 }}>
              <Title level={5} style={{ marginTop: 0 }}>
                🔗 你的心理模态耦合关系
              </Title>
              <Text type="secondary" style={{ display: 'block', marginBottom: 12 }}>
                基于你的历史数据，以下方面之间存在显著的相互影响：
              </Text>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {landscape.coupling.top_couplings?.slice(0, 6).map((c: any, i: number) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13 }}>
                    <Tag color="blue">{DIM_LABELS[c.cause] || c.cause}</Tag>
                    <span style={{ color: '#999' }}>→</span>
                    <Tag color="purple">{DIM_LABELS[c.effect] || c.effect}</Tag>
                    <Text type="secondary" style={{ fontSize: 11 }}>
                      (强度：{Math.abs(c.strength).toFixed(2)})
                    </Text>
                  </div>
                ))}
              </div>
              <Text type="secondary" style={{ fontSize: 12, display: 'block', marginTop: 12 }}>
                💡 理解这些耦合关系，可以帮助你找到杠杆点——改变一个方面，可能带动多个方面改善
              </Text>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
