import { useState, useEffect } from 'react';
import { Card, Row, Col, Typography, Progress, Tag, Divider } from 'antd';
import { LockOutlined } from '@ant-design/icons';
import { extraApi } from '../../services';

const { Title, Text } = Typography;

const categoryLabels: Record<string, string> = {
  checkin: '打卡',
  healing: '疗愈',
  assessment: '测评',
  social: '社交',
  milestone: '里程碑',
};

export default function Achievements() {
  const [allAchievements, setAllAchievements] = useState<any[]>([]);
  const [unlockedIds, setUnlockedIds] = useState<Set<string>>(new Set());
  const [stats, setStats] = useState<any>({ totalCheckIns: 0, totalHealingSessions: 0, totalAssessments: 0 });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      extraApi.getAllAchievements(),
      extraApi.getAchievements(),
    ]).then(([allRes, progressRes]: any[]) => {
      if (allRes.code === 0) setAllAchievements(allRes.data);
      if (progressRes.code === 0) {
        const ids = new Set<string>(progressRes.data.achievements.map((ua: any) => ua.achievementId));
        setUnlockedIds(ids);
        setStats(progressRes.data.stats);
      }
      setLoading(false);
    });
  }, []);

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}><Text>加载中...</Text></div>;

  const total = allAchievements.length;
  const unlocked = unlockedIds.size;
  const percent = total > 0 ? Math.round((unlocked / total) * 100) : 0;

  return (
    <div>
      <Title level={4} style={{ color: '#5a4a6a', marginBottom: 24 }}>成就徽章</Title>

      {/* 总进度 */}
      <Card className="cloud-card" style={{ marginBottom: 24, textAlign: 'center' }}>
        <div style={{ fontSize: 48, marginBottom: 8 }}>🏆</div>
        <Title level={3} style={{ color: '#ff8fab', margin: '8px 0' }}>{unlocked} / {total}</Title>
        <Text style={{ color: '#8a7a9a' }}>已解锁徽章</Text>
        <Progress
          percent={percent}
          strokeColor={{ from: '#ffb6c1', to: '#ff8fab' }}
          style={{ maxWidth: 300, margin: '16px auto 0' }}
        />
        <Divider />
        <Row gutter={16} style={{ textAlign: 'center' }}>
          <Col span={8}>
            <div style={{ fontSize: 24, fontWeight: 'bold', color: '#ff8fab' }}>{stats.totalCheckIns}</div>
            <Text style={{ fontSize: 12, color: '#8a7a9a' }}>心情打卡</Text>
          </Col>
          <Col span={8}>
            <div style={{ fontSize: 24, fontWeight: 'bold', color: '#ff8fab' }}>{stats.totalHealingSessions}</div>
            <Text style={{ fontSize: 12, color: '#8a7a9a' }}>疗愈练习</Text>
          </Col>
          <Col span={8}>
            <div style={{ fontSize: 24, fontWeight: 'bold', color: '#ff8fab' }}>{stats.totalAssessments}</div>
            <Text style={{ fontSize: 12, color: '#8a7a9a' }}>心理测评</Text>
          </Col>
        </Row>
      </Card>

      {/* 全部徽章 */}
      <Title level={5} style={{ color: '#5a4a6a', marginBottom: 12 }}>全部徽章</Title>
      <Row gutter={[16, 16]}>
        {allAchievements.map((ach: any) => {
          const isUnlocked = unlockedIds.has(ach.id);
          return (
            <Col xs={12} sm={8} lg={6} key={ach.id}>
              <Card
                className="cloud-card"
                style={{
                  textAlign: 'center',
                  opacity: isUnlocked ? 1 : 0.45,
                  filter: isUnlocked ? 'none' : 'grayscale(100%)',
                  position: 'relative',
                }}
              >
                {/* 未解锁锁头标记 */}
                {!isUnlocked && (
                  <div style={{
                    position: 'absolute',
                    top: 8,
                    right: 8,
                    width: 24,
                    height: 24,
                    borderRadius: '50%',
                    background: 'rgba(0,0,0,0.08)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}>
                    <LockOutlined style={{ fontSize: 12, color: '#999' }} />
                  </div>
                )}
                <div style={{ fontSize: 40, marginBottom: 8 }}>{ach.icon || '🔒'}</div>
                <Text strong style={{ fontSize: 14, color: isUnlocked ? '#5a4a6a' : '#999' }}>
                  {ach.name}
                </Text>
                <div style={{ marginTop: 4 }}>
                  <Tag
                    color={isUnlocked ? 'pink' : 'default'}
                    style={{ fontSize: 11, borderRadius: 20 }}
                  >
                    {categoryLabels[ach.category] || ach.category}
                  </Tag>
                </div>
                <Text style={{ fontSize: 11, color: isUnlocked ? '#b0a0c0' : '#ccc', display: 'block', marginTop: 4 }}>
                  {ach.description}
                </Text>
                {isUnlocked && (
                  <div style={{ marginTop: 6 }}>
                    <Tag color="green" style={{ fontSize: 10, borderRadius: 20 }}>已解锁</Tag>
                  </div>
                )}
              </Card>
            </Col>
          );
        })}
      </Row>

      {allAchievements.length === 0 && (
        <Card className="cloud-card" style={{ textAlign: 'center', padding: 40 }}>
          <Text style={{ color: '#8a7a9a' }}>暂无徽章数据</Text>
        </Card>
      )}
    </div>
  );
}
