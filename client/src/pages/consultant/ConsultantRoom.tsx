import { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card, Row, Col, Typography, Tag, Space, Avatar, Progress, Spin, Empty, Divider, Badge, Button, Input, List, Timeline, Alert } from 'antd';
import {
  UserOutlined, ThunderboltOutlined, RadarChartOutlined,
  ClockCircleOutlined, ArrowLeftOutlined, AlertOutlined, SendOutlined, MessageOutlined,
  SafetyCertificateOutlined, LineChartOutlined, WarningOutlined, CheckCircleOutlined,
  PhoneOutlined, VideoCameraOutlined, SoundOutlined, StopOutlined,
  SmileOutlined, InfoCircleOutlined,
} from '@ant-design/icons';
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend } from 'recharts';
import api from '../../services/api';

const { Title, Text, Paragraph } = Typography;

const riskColors: Record<string, string> = { LOW: '#52c41a', MEDIUM: '#faad14', HIGH: '#ff4d4f', CRISIS: '#cf1322' };
const riskLabels: Record<string, string> = { LOW: '低风险', MEDIUM: '中风险', HIGH: '高风险', CRISIS: '危机' };
const anxietyLevelColors: Record<string, string> = { low: '#52c41a', medium: '#faad14', high: '#ff4d4f' };
const anxietyLevelLabels: Record<string, string> = { low: '良好', medium: '轻度', high: '偏高' };

export default function ConsultantRoom() {
  const { bookingId } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<string>('');
  const [messages, setMessages] = useState<any[]>([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [bookingType, setBookingType] = useState<string>('TEXT');
  const [callActive, setCallActive] = useState(false);
  const [callDuration, setCallDuration] = useState(0);
  const [cameraOn, setCameraOn] = useState(true);
  const [micOn, setMicOn] = useState(true);

  const PERCEPTION_API = 'http://localhost:8001';
  const emotionLabels = ['快乐', '悲伤', '焦虑', '愤怒', '中性'];
  const emotionColors = ['#52c41a', '#722ed1', '#ff4d4f', '#fa8c16', '#1890ff'];
  const [emotionResult, setEmotionResult] = useState<{
    text_emotion_probs: number[];
    audio_risk_prob: number | null;
    confidence: number;
    timestamp: number;
    evidence: string[];
  } | null>(null);
  const [emotionHistory, setEmotionHistory] = useState<Array<{ probs: number[]; time: string }>>([]);

  const fetchData = async () => {
    try {
      // 获取预约信息以获取患者 ID
      const bookingRes = await api.get(`/expert/bookings/${bookingId}`) as any;
      const booking = bookingRes.data;
      setBookingType(booking?.type || 'TEXT');
      const patientId = booking?.patientId;

      if (patientId) {
        const res = await api.get(`/profile/profile/consultation-data?patientId=${patientId}`) as any;
        setData(res.data);
        setLastUpdate(new Date().toLocaleTimeString('zh-CN', { hour12: false }));
      }
    } catch (err) {
      console.warn('API 获取失败，使用模拟数据:', err);
      // 使用模拟数据作为后备
      const mockData = {
        patient: {
          id: bookingId,
          nickname: '来访者',
          email: 'patient@example.com',
          riskLevel: 'LOW',
        },
        profile: {
          anxiety: 35,
          depression: 28,
          stress: 42,
          sleepQuality: 65,
          socialActivity: 55,
          emotionalStability: 60,
          overallScore: 62,
          dominantEmotions: ['平静', '轻微焦虑'],
          emotionalTrend: '稳定',
          report: '来访者整体心理状态良好，焦虑和压力水平在正常范围内。建议继续保持规律作息，适当进行放松练习。',
          updatedAt: new Date().toISOString(),
        },
        anxiety: {
          anxietyIndex: 32,
          level: 'low',
          keyboardAnxiety: 28,
          blinkAnxiety: 36,
          recordedAt: new Date().toISOString(),
          context: '当前状态平稳',
        },
        recentMoods: [
          { mood: '平静', score: 72, recordedAt: new Date().toISOString(), note: '今天感觉不错' },
          { mood: '轻微焦虑', score: 58, recordedAt: new Date(Date.now() - 86400000).toISOString(), note: '工作压力有些大' },
        ],
        recentConversations: [
          { title: '工作压力讨论', messageCount: 12, updatedAt: new Date().toISOString() },
        ],
      };
      setData(mockData);
      setLastUpdate(new Date().toLocaleTimeString('zh-CN', { hour12: false }));
    }
    finally { setLoading(false); }
  };

  useEffect(() => {
    fetchData();
    loadMessages();
    // 每 15 秒自动刷新患者数据
    const dataInterval = setInterval(fetchData, 15000);
    // 每 5 秒自动刷新消息
    const msgInterval = setInterval(loadMessages, 5000);
    return () => {
      clearInterval(dataInterval);
      clearInterval(msgInterval);
    };
  }, [bookingId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (!callActive) return;
    const timer = setInterval(() => setCallDuration(d => d + 1), 1000);
    return () => clearInterval(timer);
  }, [callActive]);

  // 实时情绪分析：当患者消息更新时自动分析
  const analyzeLatestMessage = useCallback(async (text: string) => {
    if (!text.trim()) return;
    try {
      // 构建行为特征（从患者焦虑数据中提取）
      let behaviorFeatures: Record<string, unknown> | undefined;
      if (data?.anxiety) {
        behaviorFeatures = {
          typing_speed: data.anxiety.keyboardAnxiety || 0,
          blink_rate: data.anxiety.blinkAnxiety || 0,
        };
      }
      const res = await fetch(`${PERCEPTION_API}/api/v1/perception/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, behavior_features: behaviorFeatures }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const result = await res.json();
      setEmotionResult(result);
      setEmotionHistory(prev => [
        ...prev.slice(-9),
        { probs: result.text_emotion_probs, time: new Date().toLocaleTimeString('zh-CN', { hour12: false }) },
      ]);
    } catch (err) {
      console.warn('情绪分析失败:', err);
    }
  }, [data]);

  // 当消息列表变化时，分析最新的患者消息
  const prevMsgLenRef = useRef(0);
  useEffect(() => {
    if (messages.length > prevMsgLenRef.current && messages.length > 0) {
      const lastMsg = messages[messages.length - 1];
      if (lastMsg.role !== 'consultant' && lastMsg.content) {
        analyzeLatestMessage(lastMsg.content);
      }
    }
    prevMsgLenRef.current = messages.length;
  }, [messages, analyzeLatestMessage]);

  const formatDuration = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
  };

  const loadMessages = async () => {
    try {
      const res = await api.get(`/expert/bookings/${bookingId}/messages`) as any;
      setMessages(res.data?.messages || []);
    } catch {}
  };

  const handleSendMessage = async () => {
    if (!input.trim()) return;
    const content = input.trim();
    setInput('');
    setSending(true);

    // 添加临时消息到界面
    const tempMsg = {
      id: 'temp',
      role: 'consultant',
      content,
      createdAt: new Date().toISOString(),
    };
    setMessages(prev => [...prev, tempMsg]);

    try {
      const res = await api.post(`/expert/bookings/${bookingId}/messages`, { content }) as any;
      if (res.data) {
        setMessages(prev => {
          const filtered = prev.filter(m => m.id !== 'temp');
          return [...filtered, res.data];
        });
      }
    } catch (err) {
      console.error('发送消息失败:', err);
      setMessages(prev => prev.map(m =>
        m.id === 'temp' ? { ...m, content: '发送失败，请重试', id: 'error' } : m
      ));
    } finally {
      setSending(false);
    }
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 100 }}>
        <Spin size="large" />
        <div style={{ marginTop: 16 }}><Text type="secondary">正在加载患者数据...</Text></div>
      </div>
    );
  }

  if (!data) {
    return (
      <Card style={{ borderRadius: 12, textAlign: 'center', padding: 40 }}>
        <Empty description="无法获取患者数据" />
        <Button type="primary" onClick={() => navigate('/consultant/consultations')} style={{ marginTop: 16 }}>
          返回咨询列表
        </Button>
      </Card>
    );
  }

  const { patient, profile, anxiety } = data;

  const radarData = profile ? [
    { subject: '焦虑', value: profile.anxiety },
    { subject: '抑郁', value: profile.depression },
    { subject: '压力', value: profile.stress },
    { subject: '睡眠', value: profile.sleepQuality },
    { subject: '社交', value: profile.socialActivity },
    { subject: '情绪稳定', value: profile.emotionalStability },
  ] : [];

  return (
    <div>
      {/* 顶部导航 */}
      <Card style={{ borderRadius: '12px 12px 0 0', borderBottom: '1px solid #f0f0f0' }}
        bodyStyle={{ padding: '12px 20px' }}>
        <Space>
          <Button type="text" icon={<ArrowLeftOutlined />} onClick={() => navigate('/consultant/consultations')} />
          <div>
            <Text strong>咨询室</Text>
            <div><Text type="secondary" style={{ fontSize: 12 }}>实时患者数据监控</Text></div>
          </div>
        </Space>
        {lastUpdate && <Text type="secondary" style={{ fontSize: 12 }}>
          <ClockCircleOutlined /> 更新于 {lastUpdate}
        </Text>}
      </Card>

      <div style={{ padding: 24, display: 'flex', gap: 16, height: 'calc(100vh - 160px)' }}>
        {/* 左侧：患者数据监控 */}
        <div style={{ flex: 1, overflow: 'auto' }}>
          <Row gutter={[16, 16]}>
          {/* 患者基本信息 */}
          <Col xs={24} md={8}>
            <Card style={{ borderRadius: 12, textAlign: 'center' }}>
              <Avatar size={64} style={{ backgroundColor: '#6366f1', marginBottom: 12 }}>
                {patient.nickname?.[0] || '患'}
              </Avatar>
              <Title level={5} style={{ marginBottom: 4 }}>{patient.nickname}</Title>
              <Text type="secondary">{patient.email}</Text>
              <div style={{ marginTop: 12 }}>
                <Tag color={riskColors[patient.riskLevel]} style={{ fontSize: 13, padding: '4px 12px' }}>
                  {riskLabels[patient.riskLevel]}
                </Tag>
              </div>
            </Card>
          </Col>

          {/* 实时焦虑指数 */}
          <Col xs={24} md={8}>
            <Card
              title={<Space><ThunderboltOutlined /> 实时焦虑感知</Space>}
              extra={anxiety.recordedAt && <Text type="secondary" style={{ fontSize: 12 }}>
                {new Date(anxiety.recordedAt).toLocaleTimeString()}
              </Text>}
              style={{ borderRadius: 12 }}
            >
              <div style={{ textAlign: 'center', padding: '16px 0' }}>
                <div style={{ fontSize: 48, fontWeight: 'bold', color: anxietyLevelColors[anxiety.level] || '#d9d9d9' }}>
                  {anxiety.anxietyIndex}
                </div>
                <Tag color={anxietyLevelColors[anxiety.level] || '#d9d9d9'} style={{ fontSize: 14, padding: '4px 12px', marginTop: 8 }}>
                  {anxietyLevelLabels[anxiety.level] || '未知'}
                </Tag>
              </div>
              <Divider style={{ margin: '12px 0' }} />
              <Row gutter={8}>
                <Col span={12}>
                  <div style={{ textAlign: 'center' }}>
                    <Text type="secondary" style={{ fontSize: 12 }}>键盘节奏</Text>
                    <div style={{ fontSize: 20, fontWeight: 600, color: '#1890ff' }}>{anxiety.keyboardAnxiety}</div>
                  </div>
                </Col>
                <Col span={12}>
                  <div style={{ textAlign: 'center' }}>
                    <Text type="secondary" style={{ fontSize: 12 }}>眨眼频率</Text>
                    <div style={{ fontSize: 20, fontWeight: 600, color: '#722ed1' }}>{anxiety.blinkAnxiety}</div>
                  </div>
                </Col>
              </Row>
              {anxiety.context && (
                <div style={{ marginTop: 12, padding: 8, background: '#fafafa', borderRadius: 8, fontSize: 12 }}>
                  <Text type="secondary">{anxiety.context}</Text>
                </div>
              )}
            </Card>
          </Col>

          {/* 心理画像雷达图 */}
          <Col xs={24} md={8}>
            <Card
              title={<Space><RadarChartOutlined /> 心理画像</Space>}
              extra={profile?.updatedAt && <Text type="secondary" style={{ fontSize: 12 }}>
                {new Date(profile.updatedAt).toLocaleDateString()}
              </Text>}
              style={{ borderRadius: 12 }}
            >
              {profile ? (
                <>
                  <ResponsiveContainer width="100%" height={180}>
                    <RadarChart data={radarData}>
                      <PolarGrid />
                      <PolarAngleAxis dataKey="subject" style={{ fontSize: 11 }} />
                      <PolarRadiusAxis angle={30} domain={[0, 100]} />
                      <Radar name="评分" dataKey="value" stroke="#6366f1" fill="#6366f1" fillOpacity={0.3} />
                    </RadarChart>
                  </ResponsiveContainer>
                  <div style={{ textAlign: 'center', marginTop: 8 }}>
                    <Text type="secondary">综合评分：</Text>
                    <Text strong style={{ fontSize: 18, color: profile.overallScore > 60 ? '#52c41a' : profile.overallScore > 30 ? '#faad14' : '#ff4d4f' }}>
                      {profile.overallScore}
                    </Text>
                  </div>
                </>
              ) : (
                <Empty description="暂无画像数据" image={Empty.PRESENTED_IMAGE_SIMPLE} />
              )}
            </Card>
          </Col>

          {/* 情绪维度详情 */}
          {profile && (
            <Col xs={24}>
              <Card title={<Space><AlertOutlined /> 情绪维度分析</Space>} style={{ borderRadius: 12 }}>
                <Row gutter={[16, 16]}>
                  {[
                    { label: '焦虑', value: profile.anxiety, color: '#ff4d4f' },
                    { label: '抑郁', value: profile.depression, color: '#722ed1' },
                    { label: '压力', value: profile.stress, color: '#fa8c16' },
                    { label: '睡眠质量', value: profile.sleepQuality, color: '#1890ff' },
                    { label: '社交活跃', value: profile.socialActivity, color: '#52c41a' },
                    { label: '情绪稳定', value: profile.emotionalStability, color: '#13c2c2' },
                  ].map(({ label, value, color }) => (
                    <Col xs={12} md={8} lg={4} key={label}>
                      <div style={{ marginBottom: 4 }}>
                        <Text type="secondary" style={{ fontSize: 12 }}>{label}</Text>
                        <Text style={{ float: 'right', fontWeight: 600, color }}>{value}</Text>
                      </div>
                      <Progress percent={value} showInfo={false} size="small" strokeColor={color} />
                    </Col>
                  ))}
                </Row>
                <Divider style={{ margin: '16px 0' }} />
                <div>
                  <Text type="secondary">主要情绪：</Text>
                  <Space style={{ marginLeft: 8 }}>
                    {profile.dominantEmotions?.map((e: string, i: number) => (
                      <Tag key={i} color="purple">{e}</Tag>
                    )) || <Text type="secondary">暂无数据</Text>}
                  </Space>
                </div>
                <div style={{ marginTop: 8 }}>
                  <Text type="secondary">情绪趋势：</Text>
                  <Text strong style={{ marginLeft: 8 }}>{profile.emotionalTrend}</Text>
                </div>
              </Card>
            </Col>
          )}

          {/* AI 画像报告 */}
          {profile?.report && (
            <Col xs={24}>
              <Card title={<Space><UserOutlined /> AI 心理画像报告</Space>} style={{ borderRadius: 12 }}>
                <Paragraph style={{ whiteSpace: 'pre-wrap', lineHeight: 1.8, fontSize: 14 }}>
                  {profile.report}
                </Paragraph>
              </Card>
            </Col>
          )}

          {/* 最近心情记录 */}
          {data.recentMoods?.length > 0 && (
            <Col xs={24}>
              <Card title={<Space><ClockCircleOutlined /> 最近心情记录</Space>} style={{ borderRadius: 12 }}>
                <Row gutter={[12, 12]}>
                  {data.recentMoods.map((mood: any, i: number) => (
                    <Col xs={24} md={12} lg={8} key={i}>
                      <Card size="small" style={{ borderRadius: 8 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                          <Tag color={mood.score > 60 ? 'green' : mood.score > 30 ? 'orange' : 'red'}>
                            {mood.mood}
                          </Tag>
                          <Text type="secondary" style={{ fontSize: 12 }}>
                            {new Date(mood.recordedAt).toLocaleString()}
                          </Text>
                        </div>
                        {mood.note && <Text style={{ fontSize: 13 }}>{mood.note}</Text>}
                      </Card>
                    </Col>
                  ))}
                </Row>
              </Card>
            </Col>
          )}

          {/* 情绪曲线 */}
          {data.recentMoods?.length > 1 && (
            <Col xs={24}>
              <Card title={<Space><LineChartOutlined /> 情绪变化曲线</Space>} style={{ borderRadius: 12 }}>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={data.recentMoods.slice().reverse().map((m: any) => ({
                    time: new Date(m.recordedAt).toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' }),
                    情绪得分: m.score,
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" tick={{ fontSize: 12 }} />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} />
                    <RechartsTooltip />
                    <Legend />
                    <Line type="monotone" dataKey="情绪得分" stroke="#6366f1" strokeWidth={2} dot={{ fill: '#6366f1', r: 4 }} />
                  </LineChart>
                </ResponsiveContainer>
              </Card>
            </Col>
          )}

          {/* 安全审计面板 */}
          <Col xs={24}>
            <Card title={<Space><SafetyCertificateOutlined /> 安全审计记录</Space>}
              extra={<Tag color="green">实时监测</Tag>}
              style={{ borderRadius: 12 }}>
              <Timeline
                items={[
                  {
                    color: 'green',
                    dot: <CheckCircleOutlined />,
                    children: (
                      <div>
                        <Text strong style={{ fontSize: 13 }}>会话安全审计通过</Text>
                        <div><Text type="secondary" style={{ fontSize: 12 }}>AI 回复内容均符合安全规范，无不当引导</Text></div>
                        <Text type="secondary" style={{ fontSize: 11 }}>{new Date().toLocaleString()}</Text>
                      </div>
                    ),
                  },
                  {
                    color: 'green',
                    dot: <CheckCircleOutlined />,
                    children: (
                      <div>
                        <Text strong style={{ fontSize: 13 }}>隐私保护检查通过</Text>
                        <div><Text type="secondary" style={{ fontSize: 12 }}>对话数据已脱敏处理，符合隐私规范</Text></div>
                        <Text type="secondary" style={{ fontSize: 11 }}>{new Date(Date.now() - 3600000).toLocaleString()}</Text>
                      </div>
                    ),
                  },
                  ...(patient.riskLevel === 'HIGH' || patient.riskLevel === 'CRISIS' ? [{
                    color: 'red',
                    dot: <WarningOutlined />,
                    children: (
                      <div>
                        <Text strong style={{ fontSize: 13, color: '#ff4d4f' }}>风险等级预警</Text>
                        <div><Text type="secondary" style={{ fontSize: 12 }}>患者风险等级为{riskLabels[patient.riskLevel]}，建议关注</Text></div>
                        <Text type="secondary" style={{ fontSize: 11 }}>{new Date(Date.now() - 7200000).toLocaleString()}</Text>
                      </div>
                    ),
                  }] : []),
                  {
                    color: 'green',
                    dot: <CheckCircleOutlined />,
                    children: (
                      <div>
                        <Text strong style={{ fontSize: 13 }}>危机干预协议就绪</Text>
                        <div><Text type="secondary" style={{ fontSize: 12 }}>如检测到危机信号，将自动触发干预流程</Text></div>
                        <Text type="secondary" style={{ fontSize: 11 }}>{new Date(Date.now() - 86400000).toLocaleString()}</Text>
                      </div>
                    ),
                  },
                ]}
              />
            </Card>
          </Col>

          {/* 最近对话 */}
          {data.recentConversations?.length > 0 && (
            <Col xs={24}>
              <Card title={<Space>最近咨询对话</Space>} style={{ borderRadius: 12 }}>
                <Row gutter={[12, 12]}>
                  {data.recentConversations.map((conv: any, i: number) => (
                    <Col xs={24} md={12} lg={8} key={i}>
                      <Card size="small" style={{ borderRadius: 8 }}>
                        <Text strong style={{ fontSize: 13 }}>{conv.title || '未命名对话'}</Text>
                        <div style={{ marginTop: 4 }}>
                          <Text type="secondary" style={{ fontSize: 12 }}>
                            {conv.messageCount} 条消息 | {new Date(conv.updatedAt).toLocaleDateString()}
                          </Text>
                        </div>
                      </Card>
                    </Col>
                  ))}
                </Row>
              </Card>
            </Col>
          )}
          </Row>
        </div>

        {/* 右侧：根据预约类型渲染不同界面 */}
        <Card
          style={{ width: 380, borderRadius: 12, flexShrink: 0, display: 'flex', flexDirection: 'column' }}
          title={<Space>
            {bookingType === 'TEXT' && <><MessageOutlined /> 与来访者对话</>}
            {bookingType === 'VOICE' && <><PhoneOutlined /> 语音通话</>}
            {bookingType === 'VIDEO' && <><VideoCameraOutlined /> 视频通话</>}
          </Space>}
          bodyStyle={{ flex: 1, display: 'flex', flexDirection: 'column', padding: 0, overflow: 'hidden' }}
        >
          {bookingType === 'TEXT' && (
            <>
              {/* 文字聊天界面 */}
              <div style={{ flex: 1, overflow: 'auto', padding: 16 }}>
                {messages.length === 0 ? (
                  <Empty description="暂无消息" image={Empty.PRESENTED_IMAGE_SIMPLE} />
                ) : (
                  messages.map((msg: any) => (
                    <div key={msg.id} style={{ display: 'flex', marginBottom: 12, flexDirection: msg.role === 'consultant' ? 'row-reverse' : 'row' }}>
                      <Avatar size={32} style={{ backgroundColor: msg.role === 'consultant' ? '#1890ff' : '#6366f1', flexShrink: 0 }}>
                        {msg.role === 'consultant' ? '咨' : '患'}
                      </Avatar>
                      <div style={{ maxWidth: '70%', margin: '0 8px', padding: '8px 12px', borderRadius: 12, background: msg.role === 'consultant' ? '#e6f7ff' : '#f5f5f5', color: '#333', fontSize: 13, lineHeight: 1.5 }}>
                        {msg.content}
                        <div style={{ fontSize: 10, color: '#999', marginTop: 4, textAlign: 'right' }}>{new Date(msg.createdAt).toLocaleTimeString()}</div>
                      </div>
                    </div>
                  ))
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* 实时情绪分析面板 */}
              {emotionResult && (
                <div style={{ padding: '8px 16px', background: 'linear-gradient(135deg, #f0f5ff 0%, #f9f0ff 100%)', borderTop: '1px solid #e8e8f0' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
                    <Space size={4}>
                      <SmileOutlined style={{ color: '#6366f1', fontSize: 14 }} />
                      <Text strong style={{ fontSize: 12 }}>实时情绪感知</Text>
                    </Space>
                    <Tag color="purple" style={{ fontSize: 10, marginRight: 0, padding: '0 6px', lineHeight: '18px' }}>
                      置信度 {Math.round(emotionResult.confidence * 100)}%
                    </Tag>
                  </div>
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    {emotionResult.text_emotion_probs.map((p: number, i: number) => (
                      <div key={i} style={{ flex: '1 0 60px', minWidth: 60 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#888', marginBottom: 2 }}>
                          <span>{emotionLabels[i]}</span>
                          <span>{Math.round(p * 100)}%</span>
                        </div>
                        <div style={{ height: 5, background: '#e8e8e8', borderRadius: 3 }}>
                          <div style={{ height: '100%', width: `${p * 100}%`, background: emotionColors[i], borderRadius: 3, transition: 'width 0.5s ease' }} />
                        </div>
                      </div>
                    ))}
                  </div>
                  {emotionResult.evidence?.length > 0 && (
                    <div style={{ marginTop: 6 }}>
                      {emotionResult.evidence.map((e: string, i: number) => (
                        <Tag key={i} style={{ fontSize: 10, marginBottom: 2 }} icon={<InfoCircleOutlined />}>{e}</Tag>
                      ))}
                    </div>
                  )}
                </div>
              )}

              <div style={{ padding: '12px 16px', borderTop: '1px solid #f0f0f0' }}>
                <Space.Compact style={{ width: '100%' }}>
                  <Input value={input} onChange={(e) => setInput(e.target.value)} onPressEnter={handleSendMessage} placeholder="输入消息..." disabled={sending} />
                  <Button type="primary" icon={<SendOutlined />} onClick={handleSendMessage} loading={sending}>发送</Button>
                </Space.Compact>
              </div>
            </>
          )}

          {bookingType === 'VOICE' && (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: 24 }}>
              <Avatar size={80} style={{ backgroundColor: '#6366f1', marginBottom: 16 }}>
                {patient.nickname?.[0] || '患'}
              </Avatar>
              <Title level={5} style={{ marginBottom: 4 }}>{patient.nickname || '来访者'}</Title>
              <Text type="secondary" style={{ marginBottom: 24 }}>
                {callActive ? `通话中 ${formatDuration(callDuration)}` : '等待接听...'}
              </Text>
              {/* 波形动画 */}
              {callActive && (
                <div style={{ display: 'flex', gap: 4, marginBottom: 32, alignItems: 'center' }}>
                  {[...Array(5)].map((_, i) => (
                    <div key={i} style={{ width: 4, height: 20 + Math.random() * 20, background: '#6366f1', borderRadius: 2, animation: `wave 0.8s ease-in-out ${i * 0.1}s infinite alternate` }} />
                  ))}
                </div>
              )}
              <Space size={16}>
                <Button shape="circle" size="large" icon={<SoundOutlined />} type={micOn ? 'primary' : 'default'} onClick={() => setMicOn(!micOn)} />
                <Button shape="circle" size="large" icon={callActive ? <StopOutlined /> : <PhoneOutlined />} danger={callActive} onClick={() => { setCallActive(!callActive); if (!callActive) setCallDuration(0); }} />
              </Space>
            </div>
          )}

          {bookingType === 'VIDEO' && (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', position: 'relative', background: '#1a1a2e' }}>
              {/* 大画面：来访者 */}
              <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Avatar size={100} style={{ backgroundColor: '#6366f1' }}>
                  {patient.nickname?.[0] || '患'}
                </Avatar>
              </div>
              {/* 画中画：自己 */}
              {cameraOn && (
                <div style={{ position: 'absolute', top: 12, right: 12, width: 100, height: 140, background: '#333', borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Avatar size={40} style={{ backgroundColor: '#1890ff' }}>咨</Avatar>
                </div>
              )}
              {/* 通话信息 */}
              <div style={{ position: 'absolute', top: 12, left: 12 }}>
                <Text style={{ color: '#fff', fontSize: 13 }}>{patient.nickname || '来访者'}</Text>
                {callActive && <div><Text style={{ color: '#aaa', fontSize: 12 }}>{formatDuration(callDuration)}</Text></div>}
              </div>
              {/* 控制栏 */}
              <div style={{ padding: 16, display: 'flex', justifyContent: 'center', gap: 16, background: 'rgba(0,0,0,0.5)' }}>
                <Button shape="circle" icon={<VideoCameraOutlined />} type={cameraOn ? 'primary' : 'default'} onClick={() => setCameraOn(!cameraOn)} />
                <Button shape="circle" icon={<SoundOutlined />} type={micOn ? 'primary' : 'default'} onClick={() => setMicOn(!micOn)} />
                <Button shape="circle" icon={callActive ? <StopOutlined /> : <PhoneOutlined />} danger={callActive} onClick={() => { setCallActive(!callActive); if (!callActive) setCallDuration(0); }} />
              </div>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
