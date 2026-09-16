import { Card, Typography, Select, Space, Empty, Spin, Row, Col, Progress, Tag, Avatar } from 'antd';
import { FileTextOutlined, ThunderboltOutlined, UserOutlined } from '@ant-design/icons';
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer } from 'recharts';
import { useEffect, useState } from 'react';
import api from '../../services/api';

const { Title, Text, Paragraph } = Typography;

const riskColors: Record<string, string> = { LOW: '#52c41a', MEDIUM: '#faad14', HIGH: '#ff4d4f', CRISIS: '#cf1322' };
const riskLabels: Record<string, string> = { LOW: '低风险', MEDIUM: '中风险', HIGH: '高风险', CRISIS: '危机' };
const anxietyLevelColors: Record<string, string> = { low: '#52c41a', medium: '#faad14', high: '#ff4d4f' };
const anxietyLevelLabels: Record<string, string> = { low: '良好', medium: '轻度', high: '偏高' };

export default function ConsultantProfiles() {
  const [patients, setPatients] = useState<any[]>([]);
  const [selectedId, setSelectedId] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [profileData, setProfileData] = useState<Record<string, any>>({});
  const [anxietyData, setAnxietyData] = useState<Record<string, any>>({});

  useEffect(() => {
    loadPatients();
  }, []);

  useEffect(() => {
    if (patients.length > 0 && !selectedId) {
      setSelectedId(patients[0].userId);
    }
  }, [patients]);

  const loadPatients = async () => {
    setLoading(true);
    try {
      // 从预约记录中获取来访者列表
      const res = await api.get('/expert/bookings') as any;
      const bookings = res.data?.bookings || [];
      // 去重，按患者分组
      const patientMap = new Map<string, any>();
      bookings.forEach((b: any) => {
        if (b.patient?.id && !patientMap.has(b.patient.id)) {
          patientMap.set(b.patient.id, {
            userId: b.patient.id,
            name: b.patient.nickname || '未知',
            email: b.patient.email || '',
          });
        }
      });
      const patientList = Array.from(patientMap.values());
      setPatients(patientList);

      // 获取每个患者的画像和焦虑数据
      const profilePromises = patientList.map(p =>
        api.get(`/profile/profile/consultation-data?patientId=${p.userId}`).catch(() => null)
      );
      const results = await Promise.all(profilePromises);
      const profiles: Record<string, any> = {};
      const anxieties: Record<string, any> = {};
      patientList.forEach((p, i) => {
        if (results[i]?.data) {
          profiles[p.userId] = results[i].data;
          if (results[i].data.anxiety) {
            anxieties[p.userId] = results[i].data.anxiety;
          }
        }
      });
      setProfileData(profiles);
      setAnxietyData(anxieties);
    } catch (err) {
      console.error('获取来访者列表失败:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 100 }}>
        <Spin size="large" />
        <div style={{ marginTop: 16 }}><Text type="secondary">加载中...</Text></div>
      </div>
    );
  }

  const selected = patients.find(p => p.userId === selectedId);
  const profile = selected ? profileData[selected.userId] : null;
  const anxiety = selected ? anxietyData[selected.userId] : null;

  const radarData = profile?.profile ? [
    { subject: '焦虑', value: profile.profile.anxiety },
    { subject: '抑郁', value: profile.profile.depression },
    { subject: '压力', value: profile.profile.stress },
    { subject: '睡眠', value: profile.profile.sleepQuality },
    { subject: '社交', value: profile.profile.socialActivity },
    { subject: '情绪稳定', value: profile.profile.emotionalStability },
  ] : [];

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
        <Title level={4}><FileTextOutlined /> 来访者心理画像</Title>
        <Select
          placeholder="选择来访者"
          style={{ width: 200 }}
          value={selectedId || undefined}
          onChange={setSelectedId}
          options={patients.map(p => ({ value: p.userId, label: p.name }))}
        />
      </div>

      {patients.length === 0 ? (
        <Card style={{ borderRadius: 12, textAlign: 'center', padding: '60px 40px' }}>
          <div style={{ fontSize: 48, marginBottom: 16 }}>👥</div>
          <Title level={5} style={{ marginBottom: 8 }}>暂无来访者数据</Title>
          <Text type="secondary" style={{ display: 'block', marginBottom: 16 }}>
            当有患者预约并完成咨询后，其心理画像将自动同步到这里
          </Text>
          <Text type="secondary" style={{ fontSize: 12 }}>
            💡 提示：先通过「预约管理」确认患者预约，开始咨询后即可看到画像数据
          </Text>
        </Card>
      ) : !profile ? (
        <Card style={{ borderRadius: 12, textAlign: 'center', padding: '60px 40px' }}>
          <div style={{ fontSize: 48, marginBottom: 16 }}>📊</div>
          <Title level={5} style={{ marginBottom: 8 }}>该来访者暂无画像数据</Title>
          <Text type="secondary">
            患者需要完成心理测评或与 AI 对话后，系统才会生成心理画像
          </Text>
        </Card>
      ) : (
        <Card
          title={
            <Space>
              <Avatar size={32} style={{ backgroundColor: '#6366f1' }} icon={<UserOutlined />} />
              <span>{selected?.name} 的心理画像</span>
            </Space>
          }
          extra={
            <Space>
              {anxiety && (
                <Tag color={anxietyLevelColors[anxiety.level] || '#d9d9d9'}>
                  <ThunderboltOutlined /> 焦虑 {anxietyLevelLabels[anxiety.level] || '未知'} ({anxiety.anxietyIndex})
                </Tag>
              )}
              <Tag color={riskColors[profile.patient?.riskLevel] || '#52c41a'} style={{ fontSize: 13 }}>
                {riskLabels[profile.patient?.riskLevel] || '低风险'}
              </Tag>
            </Space>
          }
          style={{ borderRadius: 12, marginBottom: 16 }}
        >
          <Row gutter={24}>
            <Col xs={24} md={10}>
              <div style={{ textAlign: 'center', marginBottom: 16 }}>
                <Text type="secondary">综合评分</Text>
                <Progress type="dashboard" percent={profile.profile?.overallScore || 0} size={120}
                  strokeColor={(profile.profile?.overallScore || 0) > 60 ? '#52c41a' : (profile.profile?.overallScore || 0) > 30 ? '#faad14' : '#ff4d4f'} />
              </div>
            </Col>
            <Col xs={24} md={14}>
              <ResponsiveContainer width="100%" height={220}>
                <RadarChart data={radarData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="subject" />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} />
                  <Radar name="评分" dataKey="value" stroke="#1890ff" fill="#1890ff" fillOpacity={0.3} />
                </RadarChart>
              </ResponsiveContainer>
            </Col>
          </Row>
          {profile.profile?.report && (
            <Card type="inner" title="AI 分析报告" style={{ marginTop: 16, borderRadius: 8 }}>
              <Paragraph style={{ lineHeight: 1.8 }}>{profile.profile.report}</Paragraph>
            </Card>
          )}
        </Card>
      )}
    </div>
  );
}
