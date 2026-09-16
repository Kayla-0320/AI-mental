import { useState, useEffect } from 'react';
import { Card, Row, Col, Typography, Button, Progress, Space, Modal, InputNumber, Tag, Empty, Slider, message } from 'antd';
import { HeartOutlined, CloudOutlined, SoundOutlined, EditOutlined, SmileOutlined, MoonOutlined } from '@ant-design/icons';
import { healingApi } from '../../services';

const { Title, Text, Paragraph } = Typography;

const healingModules = [
  { type: 'meditation', icon: <CloudOutlined style={{ fontSize: 36 }} />, title: '冥想引导', desc: 'AI 引导冥想，放松身心', color: '#8b5cf6', duration: 15 },
  { type: 'breathing', icon: <HeartOutlined style={{ fontSize: 36 }} />, title: '呼吸练习', desc: '4-7-8 呼吸法，缓解焦虑', color: '#ec4899', duration: 5 },
  { type: 'music', icon: <SoundOutlined style={{ fontSize: 36 }} />, title: '疗愈音乐', desc: '白噪音与自然声音', color: '#06b6d4', duration: 30 },
  { type: 'journal', icon: <EditOutlined style={{ fontSize: 36 }} />, title: '情绪日记', desc: '记录心情，AI 情感标注', color: '#f59e0b', duration: 10 },
  { type: 'mindfulness', icon: <SmileOutlined style={{ fontSize: 36 }} />, title: '正念练习', desc: '身体扫描与觉察练习', color: '#10b981', duration: 20 },
  { type: 'sleep', icon: <MoonOutlined style={{ fontSize: 36 }} />, title: '睡眠记录', desc: '记录昨晚睡了多久', color: '#6366f1', duration: 0 },
];

const moodOptions = [
  { value: 'happy', label: '开心', emoji: '😊' },
  { value: 'calm', label: '平静', emoji: '😌' },
  { value: 'neutral', label: '一般', emoji: '😐' },
  { value: 'sad', label: '难过', emoji: '😢' },
  { value: 'anxious', label: '焦虑', emoji: '😰' },
  { value: 'angry', label: '烦躁', emoji: '😠' },
];

export default function Healing() {
  const [activeSession, setActiveSession] = useState<any>(null);
  const [stats, setStats] = useState<any>({ totalSessions: 0, totalDuration: 0, streak: 0 });
  const [moodBefore, setMoodBefore] = useState('');
  const [moodAfter, setMoodAfter] = useState('');
  const [journalContent, setJournalContent] = useState('');
  const [sessionActive, setSessionActive] = useState(false);
  const [sleepModal, setSleepModal] = useState(false);
  const [sleepHours, setSleepHours] = useState(7);
  const [sleepQuality, setSleepQuality] = useState(5);

  const startSession = async (module: typeof healingModules[0]) => {
    if (module.type === 'sleep') {
      setSleepModal(true);
      return;
    }
    try {
      const res = await healingApi.createSession({ type: module.type, title: module.title, duration: module.duration * 60 }) as any;
      setActiveSession(res.data);
      setSessionActive(true);
    } catch {}
  };

  const completeSession = async () => {
    if (!activeSession) return;
    try {
      await healingApi.completeSession(activeSession.id, {
        moodBefore, moodAfter,
        content: journalContent,
        completed: true,
      });
      setSessionActive(false);
      setActiveSession(null);
      setJournalContent('');
    } catch {}
  };

  return (
    <div>
      {/* 统计 */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={8}>
          <Card className="cloud-card" style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 36, fontWeight: 'bold', color: '#ff8fab', textShadow: '0 2px 4px rgba(255,182,193,0.3)' }}>{stats.totalSessions}</div>
            <Text style={{ color: '#8a7a9a' }}>累计练习</Text>
          </Card>
        </Col>
        <Col span={8}>
          <Card className="cloud-card" style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 36, fontWeight: 'bold', color: '#ff8fab', textShadow: '0 2px 4px rgba(255,182,193,0.3)' }}>{Math.round(stats.totalDuration / 60)}</div>
            <Text style={{ color: '#8a7a9a' }}>练习时长(分钟)</Text>
          </Card>
        </Col>
        <Col span={8}>
          <Card className="cloud-card" style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 36, fontWeight: 'bold', color: '#ff8fab', textShadow: '0 2px 4px rgba(255,182,193,0.3)' }}>{stats.streak}</div>
            <Text style={{ color: '#8a7a9a' }}>连续天数</Text>
          </Card>
        </Col>
      </Row>

      {/* 疗愈模块 */}
      <Title level={4} style={{ marginBottom: 16 }}>选择练习</Title>
      <Row gutter={[16, 16]}>
        {healingModules.map((module) => (
          <Col xs={24} sm={12} lg={8} key={module.type}>
            <Card className="cloud-card" hoverable onClick={() => startSession(module)} bodyStyle={{ padding: 24 }}>
              <div style={{ color: module.color, marginBottom: 12 }}>{module.icon}</div>
              <Title level={5} style={{ marginBottom: 4 }}>{module.title}</Title>
              <Text type="secondary">{module.desc}</Text>
              <div style={{ marginTop: 12 }}>
                <Tag>{module.duration} 分钟</Tag>
              </div>
            </Card>
          </Col>
        ))}
      </Row>

      {/* 练习弹窗 */}
      <Modal open={sessionActive} onCancel={() => setSessionActive(false)} title={activeSession?.title}
        footer={null} width={600}>
        <div style={{ padding: '20px 0' }}>
          <Paragraph style={{ fontSize: 16, lineHeight: 2, color: '#555' }}>
            找一个安静的地方，舒适地坐下。闭上眼睛，慢慢深呼吸...
            吸气4秒，屏息7秒，呼气8秒。感受空气进入你的身体，
            感受每一次呼吸带来的平静与放松。
          </Paragraph>
          <Space direction="vertical" style={{ width: '100%' }} size="middle">
            <div>
              <Text>练习前心情：</Text>
              <Space wrap style={{ marginTop: 8 }}>
                {moodOptions.map(m => (
                  <Tag key={m.value} color={moodBefore === m.value ? '#6366f1' : 'default'}
                    onClick={() => setMoodBefore(m.value)} style={{ cursor: 'pointer', padding: '4px 12px' }}>
                    {m.emoji} {m.label}
                  </Tag>
                ))}
              </Space>
            </div>
            <div>
              <Text>练习后心情：</Text>
              <Space wrap style={{ marginTop: 8 }}>
                {moodOptions.map(m => (
                  <Tag key={m.value} color={moodAfter === m.value ? '#6366f1' : 'default'}
                    onClick={() => setMoodAfter(m.value)} style={{ cursor: 'pointer', padding: '4px 12px' }}>
                    {m.emoji} {m.label}
                  </Tag>
                ))}
              </Space>
            </div>
            <Button type="primary" block onClick={completeSession} className="cloud-btn" style={{ borderRadius: 50, height: 44, fontSize: 15 }}>完成练习</Button>
          </Space>
        </div>
      </Modal>

      {/* 睡眠记录弹窗 */}
      <Modal
        open={sleepModal}
        onCancel={() => setSleepModal(false)}
        onOk={() => {
          message.success(`已记录：昨晚睡了 ${sleepHours} 小时，质量 ${sleepQuality}/10`);
          setSleepModal(false);
        }}
        title={<Space><MoonOutlined style={{ color: '#6366f1' }} /> 昨晚睡得怎么样？</Space>}
        okText="记录"
        cancelText="取消"
      >
        <div style={{ padding: '16px 0' }}>
          <div style={{ marginBottom: 24 }}>
            <Text strong style={{ display: 'block', marginBottom: 12 }}>🕐 睡了多久？</Text>
            <Slider
              min={1}
              max={12}
              value={sleepHours}
              onChange={setSleepHours}
              marks={{ 1: '1h', 4: '4h', 7: '7h', 10: '10h', 12: '12h' }}
              tooltip={{ formatter: (v) => `${v} 小时` }}
            />
            <div style={{ textAlign: 'center', marginTop: 8 }}>
              <Tag color="#6366f1" style={{ fontSize: 18, padding: '4px 16px', borderRadius: 20 }}>
                {sleepHours} 小时
              </Tag>
            </div>
          </div>
          <div>
            <Text strong style={{ display: 'block', marginBottom: 12 }}>💤 睡眠质量如何？</Text>
            <Slider
              min={1}
              max={10}
              value={sleepQuality}
              onChange={setSleepQuality}
              marks={{ 1: '很差', 5: '一般', 10: '超棒' }}
              tooltip={{ formatter: (v) => `${v}/10` }}
            />
          </div>
          <div style={{ marginTop: 16, padding: '12px 16px', background: '#f0f5ff', borderRadius: 12 }}>
            <Text type="secondary" style={{ fontSize: 13 }}>
              💡 青少年建议睡 8-10 小时。如果经常失眠，可以试试「冥想引导」或「呼吸练习」。
            </Text>
          </div>
        </div>
      </Modal>
    </div>
  );
}
