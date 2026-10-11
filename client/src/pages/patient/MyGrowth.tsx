import { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Alert, Button, Card, Col, Divider, Empty, Progress, Row, Space, Spin, Tag, Tooltip, Typography,
} from 'antd';
import {
  InfoCircleOutlined, TrophyOutlined, RiseOutlined, ThunderboltOutlined, ReloadOutlined,
} from '@ant-design/icons';
import {
  Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip as ChartTooltip, XAxis, YAxis,
} from 'recharts';
import dayjs from 'dayjs';
import { growthApi } from '../../services';

/**
 * 我的成长 —— 指标板块。
 *
 * 数据全部来自 `GET /api/extra/growth`（见 docs/growth_metrics_spec.md）：
 * 后端算好指标值、口径说明、空态与下一步动作，本页只负责展示。
 * 旧版把 4 个接口在前端拼装、失败静默显示 0、等级靠「徽章数 / 3」封顶，
 * 那些问题在这里被一次性消掉。
 */

const { Title, Text, Paragraph } = Typography;

const GROUPS: Array<{ key: string; label: string; icon: string; color: string; desc: string }> = [
  { key: 'persistence', label: '坚持', icon: '🔥', color: '#fa8c16', desc: '你来了多少次' },
  { key: 'mood', label: '心情', icon: '💗', color: '#ff8fab', desc: '你自己打的分' },
  { key: 'investment', label: '投入', icon: '🌿', color: '#52c41a', desc: '你为自己做了什么' },
  { key: 'exploration', label: '探索', icon: '🔍', color: '#6366f1', desc: '你了解自己多少' },
];

const categoryLabels: Record<string, string> = {
  checkin: '打卡', healing: '疗愈', assessment: '探索', social: '社交', milestone: '里程碑',
};

const STATUS_TAG: Record<string, { color: string; text: string }> = {
  ok: { color: 'green', text: '已记录' },
  empty: { color: 'default', text: '待采集' },
  building: { color: 'gold', text: '样本积累中' },
};

/** 指标卡：数值 + 目标进度 + 口径说明。空态不是 0，而是「待采集 + 怎么采集」。 */
function MetricCard({ metric, onAction }: { metric: any; onAction: (key: string) => void }) {
  const group = GROUPS.find((g) => g.key === metric.group);
  const status = STATUS_TAG[metric.status] || STATUS_TAG.ok;
  const isEmpty = metric.status !== 'ok';

  return (
    <Card
      className="cloud-card"
      style={{ height: '100%', borderRadius: 16 }}
      styles={{ body: { padding: 16 } }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <Text style={{ fontSize: 12, color: '#8a7a9a' }}>{metric.label}</Text>
        <Tooltip
          title={(
            <div style={{ fontSize: 12, lineHeight: 1.7 }}>
              <div>口径：{metric.hint}</div>
              <div style={{ opacity: 0.75 }}>来源：{metric.source}</div>
            </div>
          )}
        >
          <InfoCircleOutlined style={{ color: '#c8bcd6', fontSize: 12 }} />
        </Tooltip>
      </div>

      <div style={{
        fontSize: metric.display.length > 12 ? 15 : 24,
        fontWeight: 700,
        marginTop: 6,
        lineHeight: 1.35,
        wordBreak: 'break-word',
        color: isEmpty ? '#b8aec6' : group?.color,
      }}>
        {metric.display}
      </div>

      {metric.percent !== null && metric.percent !== undefined && !isEmpty ? (
        <Progress
          percent={metric.percent}
          showInfo={false}
          strokeColor={group?.color}
          trailColor="#f2eef7"
          size="small"
          style={{ marginTop: 8, marginBottom: 0 }}
        />
      ) : null}

      <div style={{ marginTop: 8 }}>
        {isEmpty ? (
          <Button type="link" size="small" style={{ padding: 0, fontSize: 12 }} onClick={() => onAction(metric.key)}>
            去补齐这条数据 →
          </Button>
        ) : (
          <Tag color={status.color} style={{ fontSize: 10, borderRadius: 10, marginInlineEnd: 0 }}>
            {status.text}
          </Tag>
        )}
      </div>
    </Card>
  );
}

export default function MyGrowth() {
  const navigate = useNavigate();
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /** silent=true 时不显示整页 loading（用于回来时静默刷新，避免页面闪烁） */
  const load = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    try {
      const res: any = await growthApi.getOverview(14);
      if (res?.code === 0 && res.data) {
        setData(res.data);
        setError(null);
      } else {
        // 不再静默降级成 0：说清楚是没拿到数据，而不是「你什么都没做」
        setError(res?.message || '成长数据暂时取不到');
      }
    } catch (e: any) {
      setError(e?.message || '网络异常，成长数据加载失败');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    load();
    // 打卡发生在首页、甚至另一个标签页 —— 回到本页时不该还停在旧数字上
    // （「打卡了但成长页还是 0」十有八九就是没有重新取数）
    const onFocus = () => load(true);
    window.addEventListener('focus', onFocus);
    document.addEventListener('visibilitychange', onFocus);
    return () => {
      window.removeEventListener('focus', onFocus);
      document.removeEventListener('visibilitychange', onFocus);
    };
  }, [load]);

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}><Spin /></div>;

  if (error) {
    return (
      <div>
        <Title level={4} style={{ color: '#5a4a6a' }}>🌱 我的成长</Title>
        <Alert
          type="warning"
          showIcon
          message="成长指标没有加载出来"
          description={`${error}。这不是「你没有数据」，请稍后重试。`}
          action={<Button size="small" onClick={() => window.location.reload()}>重试</Button>}
        />
      </div>
    );
  }

  const { index, level, summary, counters, metrics, keywords, trend, heatmap, badges, newlyUnlocked, nextActions } = data;
  const trendHasData = trend.some((t: any) => t.score !== null);
  const trendAvg = trendHasData
    ? Math.round((trend.filter((t: any) => t.score !== null).reduce((s: number, t: any) => s + t.score, 0)
        / trend.filter((t: any) => t.score !== null).length) * 10) / 10
    : null;

  const actionPath = (key: string) => {
    const found = nextActions.find((a: any) => a.key === key);
    if (found) return found.path;
    // 指标卡上的「去补齐」按指标归到对应页面
    if (key === 'assessments') return '/profile';
    if (key === 'healingSessions') return '/healing';
    return '/';
  };

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 12 }}>
        <div>
          <Title level={4} style={{ color: '#5a4a6a', marginBottom: 8 }}>🌱 我的成长</Title>
          <Text type="secondary" style={{ display: 'block', marginBottom: 20 }}>
            每一步都算数，下面每一个数字都能指到你的某一次记录
            {counters.streakDays === 0 && '（今天还没打卡，打了「连续打卡」就会动）'}
          </Text>
        </div>
        <Button
          size="small"
          icon={<ReloadOutlined />}
          loading={refreshing}
          onClick={() => { setRefreshing(true); load(true); }}
          style={{ borderRadius: 12, marginTop: 4 }}
        >
          刷新
        </Button>
      </div>

      {newlyUnlocked?.length > 0 && (
        <Alert
          type="success"
          showIcon
          style={{ marginBottom: 16, borderRadius: 12 }}
          message={`解锁新徽章：${newlyUnlocked.map((b: any) => `${b.icon} ${b.name}`).join('、')}`}
        />
      )}

      {/* ① 成长指数 + 等级
          渐变底必须画在 .ant-card-body 上：全局主题 healing-theme.css:324 用
          `.ant-card { background: ... !important }` 覆盖了 Card 的行内背景，
          画在根节点上会被吃掉 —— 结果就是白字白底、整块内容看不见。 */}
      <Card
        style={{ borderRadius: 20, border: 'none', marginBottom: 24, overflow: 'hidden' }}
        styles={{
          body: {
            padding: 24,
            background: 'linear-gradient(135deg, #ffd9b8 0%, #fcaa8c 55%, #f79a86 100%)',
            borderRadius: 20,
          },
        }}
      >
        <Row gutter={[24, 16]} align="middle">
          <Col xs={24} md={7} style={{ textAlign: 'center' }}>
            <Progress
              type="dashboard"
              percent={index.score}
              strokeColor="#fff"
              trailColor="rgba(255,255,255,0.35)"
              format={(p) => (
                <span style={{ color: '#fff' }}>
                  <span style={{ fontSize: 30, fontWeight: 700 }}>{p}</span>
                  <span style={{ fontSize: 12, display: 'block' }}>成长指数</span>
                </span>
              )}
            />
            <Text style={{ color: 'rgba(255,255,255,0.95)', display: 'block', marginTop: 4 }}>
              {index.grade}
            </Text>
          </Col>

          <Col xs={24} md={17}>
            <div style={{ fontSize: 40 }}>{level.icon}</div>
            <Title level={3} style={{ color: '#fff', margin: '4px 0' }}>
              Lv.{level.level} {level.name}
            </Title>
            <Text style={{ color: 'rgba(255,255,255,0.95)', fontSize: 14 }}>{summary}</Text>

            <Progress
              percent={level.progress}
              strokeColor="#fff"
              trailColor="rgba(255,255,255,0.35)"
              style={{ marginTop: 12, marginBottom: 4 }}
              format={() => (
                <span style={{ color: '#fff', fontSize: 12 }}>
                  {level.isMax ? '已满级' : `${level.xp} / ${level.nextLevelXp} 成长值`}
                </span>
              )}
            />

            {/* 各维度分：让「指数怎么来的」一眼可见 */}
            <Row gutter={[12, 12]} style={{ marginTop: 12 }}>
              {index.dimensions.map((d: any) => (
                <Col xs={12} sm={6} key={d.key}>
                  <Tooltip title={`${d.hint}（权重 ${Math.round(d.weight * 100)}%）`}>
                    <div style={{
                      background: 'rgba(255,255,255,0.22)',
                      borderRadius: 12,
                      padding: '8px 10px',
                      cursor: 'help',
                    }}>
                      <Text style={{ color: '#fff', fontSize: 12 }}>{d.label}</Text>
                      <div style={{ color: '#fff', fontSize: 18, fontWeight: 700 }}>
                        {d.score === null ? '—' : d.score}
                      </div>
                    </div>
                  </Tooltip>
                </Col>
              ))}
            </Row>
          </Col>
        </Row>
      </Card>

      {/* ② 指标板块 */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
        <ThunderboltOutlined style={{ color: '#6366f1' }} />
        <Title level={5} style={{ color: '#5a4a6a', margin: 0 }}>指标板块</Title>
        <Text type="secondary" style={{ fontSize: 12 }}>点卡片右上角 ⓘ 可以看到每个数字怎么算的</Text>
      </div>

      {GROUPS.map((g) => {
        const groupMetrics = metrics.filter((m: any) => m.group === g.key);
        if (groupMetrics.length === 0) return null;
        return (
          <div key={g.key} style={{ marginBottom: 20 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
              <span style={{ fontSize: 16 }}>{g.icon}</span>
              <Text strong style={{ color: g.color }}>{g.label}</Text>
              <Text type="secondary" style={{ fontSize: 12 }}>{g.desc}</Text>
            </div>
            <Row gutter={[12, 12]}>
              {groupMetrics.map((m: any) => (
                <Col xs={12} sm={12} md={8} lg={6} key={m.key}>
                  <MetricCard metric={m} onAction={(k) => navigate(actionPath(k))} />
                </Col>
              ))}
            </Row>
          </div>
        );
      })}

      {/* ③ 心情轨迹 + 打卡热力图 */}
      <Card className="cloud-card" style={{ marginBottom: 20, borderRadius: 16 }}>
        <Title level={5} style={{ color: '#5a4a6a', marginBottom: 4 }}>
          <RiseOutlined /> 近 14 天心情轨迹
        </Title>
        <Text type="secondary" style={{ fontSize: 12 }}>
          纵轴 1–10 分，就是你打卡时自己选的那颗表情；虚线是你这 14 天的平均
        </Text>

        {!trendHasData ? (
          <Empty
            style={{ marginTop: 16 }}
            description="还没有心情记录，去首页点一个表情就有了"
          >
            <Button type="primary" onClick={() => navigate('/')} style={{ borderRadius: 16 }}>去打卡</Button>
          </Empty>
        ) : (
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={trend} margin={{ top: 16, right: 12, left: -18, bottom: 0 }}>
              <XAxis dataKey="label" tick={{ fontSize: 11 }} />
              <YAxis domain={[1, 10]} ticks={[1, 3, 5, 7, 10]} tick={{ fontSize: 11 }} />
              <ChartTooltip
                formatter={(v: any, _n: any, payload: any) => [
                  v === null ? '未打卡' : `${v} 分${payload?.payload?.mood ? `（${payload.payload.mood}）` : ''}`,
                  '心情',
                ]}
                labelFormatter={(_l: any, payload: any) =>
                  payload?.[0]?.payload?.dayKey || ''}
              />
              {trendAvg !== null && (
                <ReferenceLine y={trendAvg} stroke="#c8bcd6" strokeDasharray="4 4"
                  label={{ value: `平均 ${trendAvg}`, fontSize: 11, fill: '#8a7a9a', position: 'right' }} />
              )}
              <Line
                type="monotone"
                dataKey="score"
                stroke="#ff8fab"
                strokeWidth={2}
                dot={{ fill: '#ff8fab', r: 4 }}
                connectNulls={false}
                name="心情"
              />
            </LineChart>
          </ResponsiveContainer>
        )}

        <Divider style={{ margin: '16px 0 12px' }} />
        <Text strong style={{ color: '#5a4a6a', fontSize: 13 }}>近 30 天打卡</Text>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 10 }}>
          {heatmap.map((d: any) => (
            <Tooltip key={d.dayKey} title={`${d.dayKey} ${d.score === null ? '未打卡' : `${d.score} 分`}`}>
              <div style={{
                width: 14, height: 14, borderRadius: 4,
                background: d.level === 0 ? '#f2eef7' : d.level === 1 ? '#ffd6e7' : '#ff9ec4',
              }} />
            </Tooltip>
          ))}
        </div>

        <Divider style={{ margin: '16px 0 12px' }} />
        <Text strong style={{ color: '#5a4a6a', fontSize: 13 }}>你的情绪关键词</Text>
        <div style={{ marginTop: 10 }}>
          {keywords.length === 0 ? (
            <Text type="secondary" style={{ fontSize: 13 }}>
              记录几天后，这里会出现你自己写下的词（不会用固定词补位）
            </Text>
          ) : (
            <Space wrap size={[8, 8]}>
              {keywords.map((word: string, i: number) => (
                <Tag
                  key={word + i}
                  style={{
                    fontSize: 13, padding: '4px 14px', borderRadius: 16, border: 'none',
                    background: ['#fff0f5', '#f0f5ff', '#fff7e6', '#f6ffed', '#f9f0ff', '#e6fffb'][i % 6],
                    color: ['#ff8fab', '#6366f1', '#fa8c16', '#52c41a', '#8b5cf6', '#13c2c2'][i % 6],
                  }}
                >
                  {word}
                </Tag>
              ))}
            </Space>
          )}
          <Paragraph type="secondary" style={{ fontSize: 12, marginTop: 10, marginBottom: 0 }}>
            💡 来自你近 14 天的打卡记录（心情标签 / 备注 / 自定义标签），只做整理，不做任何判断。
          </Paragraph>
        </div>
      </Card>

      {/* ④ 徽章墙 */}
      <Card className="cloud-card" style={{ marginBottom: 20, borderRadius: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
          <TrophyOutlined style={{ color: '#fa8c16' }} />
          <Title level={5} style={{ color: '#5a4a6a', margin: 0 }}>徽章墙</Title>
          <Tag color="pink">{badges.unlocked} / {badges.total} 已解锁</Tag>
        </div>

        {badges.total === 0 ? (
          <Empty description="徽章数据还没初始化，重启一次后端服务即可" />
        ) : (
          Object.keys(categoryLabels).map((cat) => {
            const group = badges.badges.filter((b: any) => b.category === cat);
            if (group.length === 0) return null;
            return (
              <div key={cat} style={{ marginBottom: 14 }}>
                <Text type="secondary" style={{ fontSize: 12 }}>{categoryLabels[cat]}</Text>
                <Row gutter={[12, 12]} style={{ marginTop: 8 }}>
                  {group.map((b: any) => (
                    <Col xs={12} sm={8} lg={6} key={b.id}>
                      <Card
                        size="small"
                        style={{
                          borderRadius: 12,
                          height: '100%',
                          opacity: b.unlocked ? 1 : 0.72,
                          background: b.unlocked ? '#fffaf5' : '#fbfaff',
                          border: b.unlocked ? '1px solid #ffe0c2' : '1px solid #f0eef5',
                        }}
                      >
                        <div style={{ textAlign: 'center' }}>
                          <div style={{ fontSize: 30, filter: b.unlocked ? 'none' : 'grayscale(100%)' }}>
                            {b.icon}
                          </div>
                          <Text strong style={{ fontSize: 13, color: b.unlocked ? '#5a4a6a' : '#9a90a8', display: 'block' }}>
                            {b.name}
                          </Text>
                          <Text type="secondary" style={{ fontSize: 11, display: 'block', marginTop: 2 }}>
                            {b.description}
                          </Text>
                        </div>
                        {b.unlocked ? (
                          <div style={{ textAlign: 'center', marginTop: 6 }}>
                            <Tag color="green" style={{ fontSize: 10, borderRadius: 10 }}>
                              {b.unlockedAt ? `${dayjs(b.unlockedAt).format('MM-DD')} 解锁` : '已解锁'}
                            </Tag>
                          </div>
                        ) : (
                          <div style={{ marginTop: 6 }}>
                            <Progress
                              percent={b.progress.percent}
                              showInfo={false}
                              size="small"
                              strokeColor="#ffb6c1"
                              trailColor="#f2eef7"
                              style={{ marginBottom: 2 }}
                            />
                            <Text type="secondary" style={{ fontSize: 11 }}>
                              还差 {(b.progress.target - b.progress.current) > 0
                                ? b.progress.target - b.progress.current
                                : 0}
                              （{b.progress.current}/{b.progress.target}）
                            </Text>
                          </div>
                        )}
                      </Card>
                    </Col>
                  ))}
                </Row>
              </div>
            );
          })
        )}
      </Card>

      {/* ⑤ 下一步 */}
      <Card className="cloud-card" style={{ borderRadius: 16 }}>
        <Title level={5} style={{ color: '#5a4a6a', marginBottom: 12 }}>下一步可以做</Title>
        <Row gutter={[12, 12]}>
          {nextActions.map((a: any) => (
            <Col xs={24} sm={12} md={8} key={a.key}>
              <Button
                block
                onClick={() => navigate(a.path)}
                style={{
                  height: 'auto', textAlign: 'left', borderRadius: 12, padding: '12px 14px',
                  whiteSpace: 'normal', border: '1px solid #efe4ff', background: '#faf7ff',
                }}
              >
                <div style={{ fontSize: 20 }}>{a.icon}</div>
                <div style={{ fontWeight: 600, color: '#5a4a6a' }}>{a.label}</div>
                <div style={{ fontSize: 12, color: '#8a7a9a' }}>{a.desc}</div>
              </Button>
            </Col>
          ))}
        </Row>
        <Divider style={{ margin: '16px 0 10px' }} />
        <Text type="secondary" style={{ fontSize: 12 }}>
          累计：打卡 {counters.totalCheckIns} 天 · 活跃 {counters.activeDays30}/30 天 ·
          练习 {counters.healingSessions} 次 · 测评 {counters.assessments} 次
        </Text>
      </Card>
    </div>
  );
}
