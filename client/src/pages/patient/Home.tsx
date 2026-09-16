import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Row, Col, Card, Typography, Button, Space, Tag, Progress, Spin, Empty, message, Modal } from 'antd';
import {
  MessageOutlined, HeartOutlined, TeamOutlined,
  ArrowRightOutlined, PhoneOutlined,
  ThunderboltOutlined, BulbOutlined, SmileOutlined,
} from '@ant-design/icons';
import { profileApi, healingApi, moodApi, consultationApi } from '../../services';

const { Title, Text, Paragraph } = Typography;

const moodEmojis = [
  { emoji: '😄', label: '超好', color: '#52c41a', mood: 9 },
  { emoji: '😊', label: '不错', color: '#73d13d', mood: 7 },
  { emoji: '😐', label: '一般', color: '#faad14', mood: 5 },
  { emoji: '😔', label: '低落', color: '#fa8c16', mood: 3 },
  { emoji: '😢', label: '很难过', color: '#ff4d4f', mood: 1 },
];

export default function Home() {
  const navigate = useNavigate();
  const [greeting, setGreeting] = useState('');
  const [checkedIn, setCheckedIn] = useState(false);
  const [selectedMood, setSelectedMood] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [crisisOpen, setCrisisOpen] = useState(false);
  const [streak, setStreak] = useState(0);

  useEffect(() => {
    const hour = new Date().getHours();
    if (hour < 6) setGreeting('夜深了，还没睡呀');
    else if (hour < 12) setGreeting('早上好');
    else if (hour < 14) setGreeting('中午好');
    else if (hour < 18) setGreeting('下午好');
    else setGreeting('晚上好');

    // 检查今天是否已打卡
    checkTodayCheckIn();
    loadStreak();
  }, []);

  const checkTodayCheckIn = async () => {
    try {
      const res: any = await moodApi.getCheckInHistory(7);
      const checkIns = res.data?.checkIns || [];
      const today = new Date().toISOString().split('T')[0];
      const todayCheckIn = checkIns.find((c: any) => c.checkInDate?.startsWith(today));
      if (todayCheckIn) {
        setCheckedIn(true);
        setSelectedMood(todayCheckIn.mood);
      }
    } catch {}
  };

  const loadStreak = async () => {
    try {
      const res: any = await moodApi.getCheckInHistory(365);
      const checkIns = res.data?.checkIns || [];
      let s = 0;
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      for (let i = 0; i < checkIns.length; i++) {
        const d = new Date(checkIns[i].checkInDate);
        d.setHours(0, 0, 0, 0);
        const diff = (today.getTime() - d.getTime()) / 86400000;
        if (diff === i) s++;
        else break;
      }
      setStreak(s);
    } catch {}
  };

  const handleMoodCheckIn = async (moodIdx: number) => {
    const m = moodEmojis[moodIdx];
    setLoading(true);
    try {
      await moodApi.checkIn({
        mood: m.label,
        score: m.mood,
        note: `${m.emoji} ${m.label}`,
      });
      setCheckedIn(true);
      setSelectedMood(moodIdx);
      message.success({ content: `已记录：${m.emoji} ${m.label}`, duration: 2 });
      loadStreak();
    } catch {
      message.error('记录失败，再试试？');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      {/* 心情打卡卡片 */}
      <Card style={{
        borderRadius: 20,
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        border: 'none',
        marginBottom: 24,
        overflow: 'hidden',
      }}>
        <Row align="middle" gutter={24}>
          <Col flex="auto">
            <Title level={3} style={{ color: '#fff', marginBottom: 4 }}>
              {greeting} <SmileOutlined />
            </Title>
            <Text style={{ color: 'rgba(255,255,255,0.85)', fontSize: 15 }}>
              {checkedIn ? '今天已经打过卡啦~' : '今天心情怎么样？点一点记录一下'}
            </Text>
            {!checkedIn && (
              <div style={{ display: 'flex', gap: 12, marginTop: 16 }}>
                {moodEmojis.map((m, idx) => (
                  <Button
                    key={idx}
                    loading={loading && selectedMood === idx}
                    onClick={() => handleMoodCheckIn(idx)}
                    style={{
                      width: 56,
                      height: 56,
                      borderRadius: '50%',
                      fontSize: 28,
                      border: '2px solid rgba(255,255,255,0.3)',
                      background: selectedMood === idx ? 'rgba(255,255,255,0.25)' : 'rgba(255,255,255,0.1)',
                      cursor: 'pointer',
                      transition: 'all 0.2s',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                    onMouseEnter={(e) => { e.currentTarget.style.transform = 'scale(1.15)'; }}
                    onMouseLeave={(e) => { e.currentTarget.style.transform = 'scale(1)'; }}
                  >
                    {m.emoji}
                  </Button>
                ))}
              </div>
            )}
            {checkedIn && selectedMood !== null && (
              <div style={{ marginTop: 12 }}>
                <Tag style={{
                  background: 'rgba(255,255,255,0.2)',
                  border: 'none',
                  color: '#fff',
                  fontSize: 16,
                  padding: '4px 16px',
                  borderRadius: 20,
                }}>
                  {moodEmojis[selectedMood].emoji} {moodEmojis[selectedMood].label}
                  {streak > 0 && <span style={{ marginLeft: 8 }}>🔥 连续 {streak} 天</span>}
                </Tag>
              </div>
            )}
          </Col>
          <Col>
            <Button
              type="primary"
              ghost
              size="large"
              onClick={() => navigate('/chat')}
              style={{ borderColor: '#fff', color: '#fff', borderRadius: 24, height: 48 }}
            >
              想聊聊 <ArrowRightOutlined />
            </Button>
          </Col>
        </Row>
      </Card>

      {/* 紧急求助按钮 */}
      <Card
        style={{
          borderRadius: 16,
          background: 'linear-gradient(135deg, #fff5f5 0%, #fff0f6 100%)',
          border: '1px solid #ffccc7',
          marginBottom: 24,
        }}
        bodyStyle={{ padding: '16px 24px' }}
      >
        <Row align="middle" justify="space-between">
          <Col>
            <Space>
              <PhoneOutlined style={{ fontSize: 20, color: '#ff4d4f' }} />
              <div>
                <Text strong style={{ color: '#5a4a6a' }}>需要紧急帮助？</Text>
                <Text type="secondary" style={{ fontSize: 12, display: 'block' }}>
                  24小时心理援助热线 · 随时有人愿意听你说
                </Text>
              </div>
            </Space>
          </Col>
          <Col>
            <Button
              danger
              type="primary"
              onClick={() => setCrisisOpen(true)}
              style={{ borderRadius: 20 }}
            >
              立即求助
            </Button>
          </Col>
        </Row>
      </Card>

      {/* 快捷入口 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        {[
          { icon: '💬', title: '聊聊', desc: 'AI 陪你说话', color: '#6366f1', path: '/chat' },
          { icon: '🫂', title: '同伴', desc: '和懂你的人在一起', color: '#ec4899', path: '/companions' },
          { icon: '🌿', title: '放松空间', desc: '冥想、呼吸、放松', color: '#52c41a', path: '/healing' },
          { icon: '☕', title: '找帮手', desc: '预约专业咨询师', color: '#f59e0b', path: '/experts' },
        ].map(({ icon, title, desc, color, path }) => (
          <Col xs={12} sm={6} key={title}>
            <Card
              hoverable
              onClick={() => navigate(path)}
              style={{ borderRadius: 16, height: '100%', border: 'none', boxShadow: '0 2px 8px rgba(0,0,0,0.06)' }}
              bodyStyle={{ padding: 20, textAlign: 'center' }}
            >
              <div style={{ fontSize: 36, marginBottom: 8 }}>{icon}</div>
              <Title level={5} style={{ marginBottom: 4, color }}>{title}</Title>
              <Text type="secondary" style={{ fontSize: 12 }}>{desc}</Text>
            </Card>
          </Col>
        ))}
      </Row>

      {/* 今日情绪洞察 */}
      <Card
        title={<Space><ThunderboltOutlined style={{ color: '#6366f1' }} /> 今日心情洞察</Space>}
        style={{ borderRadius: 16, marginBottom: 24 }}
      >
        <Spin spinning={false}>
          <Empty
            description={
              <span style={{ color: '#8a7a9a' }}>
                和 AI 聊聊天，就能获得心情洞察哦~
              </span>
            }
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          >
            <Button type="primary" onClick={() => navigate('/chat')} style={{ borderRadius: 20 }}>
              去聊聊
            </Button>
          </Empty>
        </Spin>
      </Card>

      {/* 连续打卡提示 */}
      {streak > 0 && (
        <Card
          style={{
            borderRadius: 16,
            background: 'linear-gradient(135deg, #fff7e6 0%, #fffbe6 100%)',
            border: '1px solid #ffe58f',
            textAlign: 'center',
          }}
        >
          <div style={{ fontSize: 36 }}>🔥</div>
          <Title level={4} style={{ color: '#fa8c16', margin: '8px 0' }}>已连续打卡 {streak} 天</Title>
          <Text type="secondary">
            {streak < 7 ? '再坚持几天就能解锁「一周达人」徽章啦！' :
             streak < 30 ? '太厉害了！你已经是打卡达人了~' :
             '你是传说中的打卡之王！'}
          </Text>
        </Card>
      )}

      {/* 紧急求助弹窗 */}
      <Modal
        open={crisisOpen}
        onCancel={() => setCrisisOpen(false)}
        footer={null}
        title={<Space><PhoneOutlined style={{ color: '#ff4d4f' }} /> 紧急求助</Space>}
      >
        <div style={{ padding: '16px 0' }}>
          <Paragraph style={{ fontSize: 15, color: '#5a4a6a' }}>
            如果你现在很痛苦，请记住，有人愿意帮助你：
          </Paragraph>
          <Card style={{ borderRadius: 12, marginBottom: 16, background: '#fff5f5', border: '1px solid #ffccc7' }}>
            <div style={{ marginBottom: 12 }}>
              <Text strong>🆘 全国24小时心理援助热线</Text>
              <Title level={2} style={{ color: '#ff4d4f', margin: '4px 0' }}>400-161-9995</Title>
            </div>
            <div style={{ marginBottom: 12 }}>
              <Text strong>📞 北京心理危机研究与干预中心</Text>
              <Title level={3} style={{ color: '#fa8c16', margin: '4px 0' }}>010-82951332</Title>
            </div>
            <div>
              <Text strong>💬 生命热线</Text>
              <Title level={3} style={{ color: '#6366f1', margin: '4px 0' }}>400-821-1215</Title>
            </div>
          </Card>
          <Paragraph type="secondary" style={{ fontSize: 13 }}>
            你也可以直接去「聊聊」页面，AI 会一直陪着你。如果情况紧急，请拨打上面的电话，会有专业的人帮助你。
          </Paragraph>
          <Button
            type="primary"
            block
            size="large"
            onClick={() => { setCrisisOpen(false); navigate('/chat'); }}
            style={{ borderRadius: 12, marginTop: 8 }}
          >
            去找 AI 聊聊
          </Button>
        </div>
      </Modal>
    </div>
  );
}
