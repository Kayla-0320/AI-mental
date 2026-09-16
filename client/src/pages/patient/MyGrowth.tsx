import { useState, useEffect } from 'react';
import { Card, Row, Col, Typography, Progress, Tag, Space, Empty, Spin, Divider, Tabs } from 'antd';
import {
  TrophyOutlined, StarFilled, FireOutlined, HeartOutlined,
  SmileOutlined, LockOutlined, RiseOutlined,
} from '@ant-design/icons';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { extraApi, profileApi } from '../../services';

const { Title, Text, Paragraph } = Typography;

const categoryLabels: Record<string, string> = {
  checkin: '打卡', healing: '疗愈', assessment: '探索', social: '社交', milestone: '里程碑',
};

const categoryIcons: Record<string, string> = {
  checkin: '📅', healing: '🌿', assessment: '🔍', social: '🤝', milestone: '🎯',
};

export default function MyGrowth() {
  const [allAchievements, setAllAchievements] = useState<any[]>([]);
  const [unlockedIds, setUnlockedIds] = useState<Set<string>>(new Set());
  const [stats, setStats] = useState<any>({ totalCheckIns: 0, totalHealingSessions: 0, totalAssessments: 0 });
  const [moodTrend, setMoodTrend] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      extraApi.getAllAchievements(),
      extraApi.getAchievements(),
      profileApi.getMoodTrend().catch(() => ({ data: [] })),
    ]).then(([allRes, progressRes, moodRes]: any[]) => {
      if (allRes.code === 0) setAllAchievements(allRes.data);
      if (progressRes.code === 0) {
        const ids = new Set<string>(progressRes.data.achievements.map((ua: any) => ua.achievementId));
        setUnlockedIds(ids);
        setStats(progressRes.data.stats);
      }
      if (moodRes.data) setMoodTrend(moodRes.data.slice(-14));
      setLoading(false);
    });
  }, []);

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}><Spin /></div>;

  const total = allAchievements.length;
  const unlocked = unlockedIds.size;
  const percent = total > 0 ? Math.round((unlocked / total) * 100) : 0;

  // 计算等级
  const level = Math.floor(unlocked / 3) + 1;
  const levelNames = ['', '萌芽', '新芽', '花苞', '初绽', '盛放', '向阳', '绽放', '璀璨', '星光', '彩虹'];
  const levelName = levelNames[Math.min(level, 10)] || '彩虹';

  return (
    <div>
      <Title level={4} style={{ color: '#5a4a6a', marginBottom: 8 }}>
        🌱 我的成长
      </Title>
      <Text type="secondary" style={{ display: 'block', marginBottom: 24 }}>
        每一步都算数，看看你走了多远
      </Text>

      <Tabs items={[
        {
          key: 'journey',
          label: <span><TrophyOutlined /> 成长旅程</span>,
          children: (
            <div>
              {/* 等级卡片 */}
              <Card style={{
                borderRadius: 20,
                background: 'linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)',
                border: 'none',
                marginBottom: 24,
                textAlign: 'center',
              }}>
                <div style={{ fontSize: 56, marginBottom: 8 }}>
                  {level <= 3 ? '🌱' : level <= 6 ? '🌸' : '🌈'}
                </div>
                <Title level={2} style={{ color: '#fff', margin: 0, textShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
                  Lv.{level} {levelName}
                </Title>
                <Text style={{ color: 'rgba(255,255,255,0.9)', fontSize: 15 }}>
                  已收集 {unlocked}/{total} 枚徽章
                </Text>
                <Progress
                  percent={percent}
                  strokeColor="#fff"
                  trailColor="rgba(255,255,255,0.3)"
                  style={{ maxWidth: 300, margin: '16px auto 0' }}
                  format={(p) => <span style={{ color: '#fff', fontWeight: 600 }}>{p}%</span>}
                />
              </Card>

              {/* 数据概览 */}
              <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
                {[
                  { icon: '📅', label: '心情打卡', value: stats.totalCheckIns || 0, color: '#ff8fab' },
                  { icon: '🌿', label: '放松练习', value: stats.totalHealingSessions || 0, color: '#52c41a' },
                  { icon: '🔍', label: '心情检测', value: stats.totalAssessments || 0, color: '#6366f1' },
                  { icon: '🔥', label: '连续天数', value: 0, color: '#fa8c16' },
                ].map(({ icon, label, value, color }) => (
                  <Col xs={12} sm={6} key={label}>
                    <Card className="cloud-card" style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: 32 }}>{icon}</div>
                      <div style={{ fontSize: 28, fontWeight: 'bold', color, marginTop: 4 }}>{value}</div>
                      <Text style={{ fontSize: 12, color: '#8a7a9a' }}>{label}</Text>
                    </Card>
                  </Col>
                ))}
              </Row>

              {/* 徽章墙 */}
              <Title level={5} style={{ color: '#5a4a6a', marginBottom: 12 }}>
                徽章墙 <Tag color="pink">{unlocked} 已解锁</Tag>
              </Title>
              {allAchievements.length === 0 ? (
                <Empty description="还没有徽章，快去探索吧~" />
              ) : (
                <Row gutter={[12, 12]}>
                  {allAchievements.map((ach: any) => {
                    const isUnlocked = unlockedIds.has(ach.id);
                    return (
                      <Col xs={8} sm={6} lg={4} key={ach.id}>
                        <Card
                          size="small"
                          style={{
                            textAlign: 'center',
                            borderRadius: 12,
                            opacity: isUnlocked ? 1 : 0.4,
                            filter: isUnlocked ? 'none' : 'grayscale(100%)',
                            position: 'relative',
                            minHeight: 100,
                          }}
                        >
                          {!isUnlocked && (
                            <LockOutlined style={{
                              position: 'absolute', top: 6, right: 6,
                              fontSize: 10, color: '#bbb',
                            }} />
                          )}
                          <div style={{ fontSize: 32 }}>{ach.icon || '🔒'}</div>
                          <Text strong style={{ fontSize: 12, color: isUnlocked ? '#5a4a6a' : '#999', display: 'block' }}>
                            {ach.name}
                          </Text>
                          <Tag
                            color={isUnlocked ? 'pink' : 'default'}
                            style={{ fontSize: 10, borderRadius: 10, marginTop: 4 }}
                          >
                            {categoryLabels[ach.category] || ach.category}
                          </Tag>
                        </Card>
                      </Col>
                    );
                  })}
                </Row>
              )}
            </div>
          ),
        },
        {
          key: 'mood',
          label: <span><HeartOutlined /> 心情轨迹</span>,
          children: (
            <div>
              <Card className="cloud-card" style={{ marginBottom: 24 }}>
                <Title level={5} style={{ color: '#5a4a6a', marginBottom: 16 }}>
                  <RiseOutlined /> 最近 14 天心情变化
                </Title>
                {moodTrend.length === 0 ? (
                  <Empty description="还没有心情记录，去首页打卡记录今天的心情吧~" />
                ) : (
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={moodTrend}>
                      <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                      <YAxis domain={[0, 10]} tick={{ fontSize: 11 }} />
                      <Tooltip />
                      <Line
                        type="monotone"
                        dataKey="mood"
                        stroke="#ff8fab"
                        strokeWidth={2}
                        dot={{ fill: '#ff8fab', r: 4 }}
                        name="心情"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </Card>

              <Card className="cloud-card">
                <Title level={5} style={{ color: '#5a4a6a', marginBottom: 12 }}>
                  <SmileOutlined /> 你的情绪关键词
                </Title>
                <Space wrap size={[8, 8]}>
                  {['平静', '专注', '温暖', '好奇', '放松', '勇敢'].map((word, i) => (
                    <Tag
                      key={word}
                      style={{
                        fontSize: 14,
                        padding: '6px 16px',
                        borderRadius: 20,
                        background: ['#fff0f5', '#f0f5ff', '#fff7e6', '#f6ffed', '#f9f0ff', '#e6fffb'][i],
                        color: ['#ff8fab', '#6366f1', '#fa8c16', '#52c41a', '#8b5cf6', '#13c2c2'][i],
                        border: 'none',
                      }}
                    >
                      {word}
                    </Tag>
                  ))}
                </Space>
                <Divider />
                <Text type="secondary" style={{ fontSize: 13 }}>
                  💡 这些词是根据你的心情记录和聊天内容提取的，它们代表着正在成长中的你。
                </Text>
              </Card>
            </div>
          ),
        },
      ]} />
    </div>
  );
}
