import { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card, Row, Col, Typography, Tag, Space, Avatar, Progress, Spin, Divider, Button, Input, Empty, Collapse, Modal, Switch, message } from 'antd';
import {
  ArrowLeftOutlined, ThunderboltOutlined, ClockCircleOutlined,
  SendOutlined, MessageOutlined, UserOutlined,
  PhoneOutlined, VideoCameraOutlined, SoundOutlined, StopOutlined,
  ExperimentOutlined, DashboardOutlined, SafetyCertificateOutlined,
} from '@ant-design/icons';
import { useAnxiety } from '../../context/AnxietyContext';
import api from '../../services/api';
import { getSocket } from '../../services/socket';
import { useWebRTC } from '../../hooks/useWebRTC';

const { Title, Text, Paragraph } = Typography;

const EMOTION_LABELS = ['快乐', '悲伤', '焦虑', '愤怒', '中性'];
const EMOTION_COLORS = ['#52c41a', '#722ed1', '#ff4d4f', '#fa8c16', '#1890ff'];

export default function PatientRoom() {
  const { bookingId } = useParams();
  const navigate = useNavigate();
  const { comprehensiveState: anxietyState } = useAnxiety();
  const [myProfile, setMyProfile] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [messages, setMessages] = useState<any[]>([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [bookingType, setBookingType] = useState<string>('TEXT');
  const [callActive, setCallActive] = useState(false);
  const [callDuration, setCallDuration] = useState(0);
  const [cameraOn, setCameraOn] = useState(true);
  const [micOn, setMicOn] = useState(true);

  // 知情同意状态
  const [consentModalOpen, setConsentModalOpen] = useState(true);
  const [consentGiven, setConsentGiven] = useState(false);
  const [shareAnxiety, setShareAnxiety] = useState(true);
  const [shareProfile, setShareProfile] = useState(true);
  const [shareBehavior, setShareBehavior] = useState(true);

  // 多模态数据采集
  const inputRef = useRef<any>(null);
  const keystrokeTimes = useRef<number[]>([]);
  const lastKeyTime = useRef(0);
  const deletionCount = useRef(0);
  const [behaviorData, setBehaviorData] = useState({ typingSpeed: 0, pauseFrequency: 0, deletionRate: 0, sessionMinutes: 0 });
  const [audioLevel, setAudioLevel] = useState(0);
  const [audioActive, setAudioActive] = useState(false);
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioAnimRef = useRef(0);
  const videoRef = useRef<HTMLVideoElement>(null);
  const remoteVideoRef = useRef<HTMLVideoElement>(null);
  const [videoActive, setVideoActive] = useState(false);
  const [videoStream, setVideoStream] = useState<MediaStream | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sessionStart = useRef(Date.now());
  const socketRef = useRef<ReturnType<typeof getSocket> | null>(null);

  // WebRTC 双向视频
  const webrtc = useWebRTC(socketRef.current, bookingId || null);

  // 按键行为采集
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setInput(val);
    const now = Date.now();
    if (lastKeyTime.current > 0) {
      keystrokeTimes.current.push(now - lastKeyTime.current);
      if (keystrokeTimes.current.length > 50) keystrokeTimes.current.shift();
    }
    lastKeyTime.current = now;
    if (val.length < (input.length || 0)) deletionCount.current++;
  };

  // 音频采集分析（语音通话模式）
  const startAudioCapture = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const ctx = new AudioContext();
      audioContextRef.current = ctx;
      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      const tick = () => {
        analyser.getByteFrequencyData(dataArray);
        const avg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
        setAudioLevel(Math.round(avg));
        audioAnimRef.current = requestAnimationFrame(tick);
      };
      tick();
      setAudioActive(true);
    } catch (err) {
      console.warn('麦克风访问失败:', err);
    }
  };

  // 视频采集（视频通话模式）—— 使用 WebRTC
  const startVideoCapture = async () => {
    try {
      const stream = await webrtc.startLocalStream(true, true);
      if (stream) {
        setVideoStream(stream);
        setVideoActive(true);
        // 发起 WebRTC 通话
        await webrtc.call();
      }
    } catch (err) {
      console.warn('摄像头访问失败:', err);
    }
  };

  // 绑定本地视频流
  useEffect(() => {
    if (videoStream && videoRef.current) {
      videoRef.current.srcObject = videoStream;
      videoRef.current.play().catch(() => {});
    }
  }, [videoStream]);

  // 绑定远端视频流
  useEffect(() => {
    if (webrtc.remoteStream && remoteVideoRef.current) {
      remoteVideoRef.current.srcObject = webrtc.remoteStream;
      remoteVideoRef.current.play().catch(() => {});
    }
  }, [webrtc.remoteStream]);

  // 停止所有采集
  const stopAllCapture = () => {
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    if (audioAnimRef.current) cancelAnimationFrame(audioAnimRef.current);
    audioContextRef.current?.close();
    audioContextRef.current = null;
    setAudioActive(false);
    setVideoActive(false);
    setAudioLevel(0);
  };

  // 多模态采集生命周期
  useEffect(() => {
    sessionStart.current = Date.now();
    const el = inputRef.current?.input?.querySelector('input');
    if (el) el.addEventListener('keydown', () => {});
    const behaviorTimer = setInterval(() => {
      const times = keystrokeTimes.current;
      const avgInterval = times.length > 0 ? times.reduce((a, b) => a + b, 0) / times.length : 0;
      const pauses = times.filter(t => t > 1500).length;
      setBehaviorData({
        typingSpeed: Math.round(avgInterval > 0 ? 60000 / avgInterval : 0),
        pauseFrequency: Math.min(100, Math.round((pauses / Math.max(1, times.length)) * 100)),
        deletionRate: Math.min(100, Math.round((deletionCount.current / Math.max(1, times.length)) * 100)),
        sessionMinutes: Math.round((Date.now() - sessionStart.current) / 60000),
      });
    }, 3000);
    if (bookingType === 'VOICE') startAudioCapture();
    if (bookingType === 'VIDEO') startVideoCapture();
    return () => { clearInterval(behaviorTimer); stopAllCapture(); };
  }, [bookingType]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (!callActive) return;
    const timer = setInterval(() => setCallDuration(d => d + 1), 1000);
    return () => clearInterval(timer);
  }, [callActive]);

  const formatDuration = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
  };

  useEffect(() => {
    const fetchBookingType = async () => {
      try {
        const res = await api.get(`/expert/bookings/${bookingId}`) as any;
        setBookingType(res.data?.type || 'TEXT');
      } catch {}
    };
    fetchBookingType();

    const fetchData = async () => {
      try {
        const res = await api.get('/profile/psychological') as any;
        setMyProfile(res.data);
      } catch {}
      finally { setLoading(false); }
    };
    fetchData();
    const dataInterval = setInterval(fetchData, 15000);

    // 加载消息
    const loadMessages = async () => {
      if (!bookingId) return;
      try {
        const res = await api.get(`/expert/bookings/${bookingId}/messages`) as any;
        if (res.data?.messages) setMessages(res.data.messages);
      } catch {}
    };
    loadMessages();
    const msgInterval = setInterval(loadMessages, 5000);

    return () => { clearInterval(dataInterval); clearInterval(msgInterval); };
  }, [bookingId]);

  // Socket 连接 + 多模态数据同步
  useEffect(() => {
    if (!bookingId) return;
    const socket = getSocket();
    socketRef.current = socket;
    socket.emit('expert:join', bookingId);
    socket.emit('webrtc:join', { bookingId });

    // 每 5 秒发送一次多模态数据给咨询师（需知情同意）
    const multimodalInterval = setInterval(() => {
      // 检查知情同意
      if (!consentGiven) return;
      if (anxietyState.activeModalities.length === 0) return;

      // 根据 consent 过滤数据
      const filteredData: any = {
        timestamp: Date.now(),
      };

      if (shareBehavior) {
        filteredData.comprehensiveState = {
          activeModalities: anxietyState.activeModalities,
          emotionProbs: anxietyState.emotionProbs,
          dominantEmotion: anxietyState.dominantEmotion,
          riskLevel: anxietyState.riskLevel,
          confidence: anxietyState.confidence,
        };
      }

      if (shareAnxiety) {
        // 从情绪概率中提取焦虑指数
        const anxietyProb = anxietyState.emotionProbs?.[2] || 0; // 索引2是焦虑
        filteredData.anxietyState = {
          anxietyProbability: Math.round(anxietyProb * 100),
          dominantEmotion: anxietyState.dominantEmotion,
        };
      }

      if (shareProfile && myProfile) {
        filteredData.profile = {
          anxiety: myProfile.anxiety,
          depression: myProfile.depression,
          stress: myProfile.stress,
          sleepQuality: myProfile.sleepQuality,
          socialActivity: myProfile.socialActivity,
          emotionalStability: myProfile.emotionalStability,
        };
      }

      // 只有至少共享了一项数据才发送
      if (filteredData.comprehensiveState || filteredData.anxietyState || filteredData.profile) {
        socket.emit('multimodal:update', {
          bookingId,
          multimodalData: filteredData,
        });
      }
    }, 5000);

    return () => {
      clearInterval(multimodalInterval);
      socket.emit('expert:leave', bookingId);
      socket.emit('webrtc:leave', bookingId);
    };
  }, [bookingId, consentGiven, shareAnxiety, shareProfile, shareBehavior, myProfile]);

  const handleSend = async () => {
    if (!input.trim() || !bookingId) return;
    const content = input.trim();
    setInput('');
    setSending(true);
    try {
      await api.post(`/expert/bookings/${bookingId}/messages`, { content });
      const res = await api.get(`/expert/bookings/${bookingId}/messages`) as any;
      if (res.data?.messages) setMessages(res.data.messages);
    } catch {}
    finally { setSending(false); }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 60px)' }}>
      {/* 知情同意弹窗 */}
      <Modal
        open={consentModalOpen && !consentGiven}
        footer={null}
        closable={false}
        width={520}
        centered
      >
        <div style={{ padding: '16px 0' }}>
          <div style={{ textAlign: 'center', marginBottom: 16 }}>
            <SafetyCertificateOutlined style={{ fontSize: 36, color: '#6366f1' }} />
            <Title level={4} style={{ marginTop: 8, marginBottom: 4 }}>进入咨询室前</Title>
            <Text type="secondary">你需要知道咨询师能看到什么</Text>
          </div>
          <div style={{ background: '#f9f0ff', padding: '16px 20px', borderRadius: 12, marginBottom: 16 }}>
            <Text style={{ fontSize: 14, color: '#5a4a6a' }}>
              💡 你的隐私很重要。以下数据会在咨询过程中同步给咨询师，你可以选择关闭不想分享的：
            </Text>
          </div>
          <div style={{ marginBottom: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 0', borderBottom: '1px solid #f0f0f0' }}>
              <Space>
                <ThunderboltOutlined style={{ color: '#6366f1' }} />
                <div>
                  <Text strong style={{ fontSize: 14 }}>焦虑状态</Text>
                  <div><Text type="secondary" style={{ fontSize: 12 }}>键盘打字节奏、眨眼频率计算的焦虑指数</Text></div>
                </div>
              </Space>
              <Switch checked={shareAnxiety} onChange={setShareAnxiety} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 0', borderBottom: '1px solid #f0f0f0' }}>
              <Space>
                <DashboardOutlined style={{ color: '#1890ff' }} />
                <div>
                  <Text strong style={{ fontSize: 14 }}>心理画像</Text>
                  <div><Text type="secondary" style={{ fontSize: 12 }}>焦虑、抑郁、压力等维度评分</Text></div>
                </div>
              </Space>
              <Switch checked={shareProfile} onChange={setShareProfile} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 0' }}>
              <Space>
                <ExperimentOutlined style={{ color: '#52c41a' }} />
                <div>
                  <Text strong style={{ fontSize: 14 }}>行为数据</Text>
                  <div><Text type="secondary" style={{ fontSize: 12 }}>打字速度、停顿频率、删除率</Text></div>
                </div>
              </Space>
              <Switch checked={shareBehavior} onChange={setShareBehavior} />
            </div>
          </div>
          <div style={{ background: '#f6ffed', padding: '12px 16px', borderRadius: 8, marginBottom: 16 }}>
            <Text style={{ fontSize: 13, color: '#52c41a' }}>
              🔒 所有数据仅用于辅助咨询师了解你的状态，不会泄露给任何第三方。
            </Text>
          </div>
          <Button
            type="primary"
            block
            size="large"
            onClick={() => { setConsentGiven(true); setConsentModalOpen(false); }}
            style={{ borderRadius: 12 }}
          >
            我了解了，进入咨询室
          </Button>
        </div>
      </Modal>
      {/* 顶部导航 */}
      <Card style={{ borderRadius: '12px 12px 0 0', borderBottom: '1px solid #f0f0f0', flexShrink: 0 }}
        bodyStyle={{ padding: '12px 20px' }}>
        <Space>
          <Button type="text" icon={<ArrowLeftOutlined />} onClick={() => navigate('/experts')} />
          <div>
            <Text strong>咨询室</Text>
            <div><Text type="secondary" style={{ fontSize: 12 }}>与咨询师实时连线中</Text></div>
          </div>
        </Space>
      </Card>

      {/* 主体：左侧数据 + 右侧对话 */}
      <div style={{ flex: 1, display: 'flex', gap: 16, padding: 24, overflow: 'hidden' }}>
        {/* 左侧：数据面板 */}
        <div style={{ flex: 1, overflow: 'auto' }}>
          <Row gutter={[16, 16]}>
            <Col xs={24} md={12}>
              <Card title={<Space><ThunderboltOutlined /> 多模态情绪感知</Space>} style={{ borderRadius: 12 }}>
                {anxietyState.activeModalities.length > 0 ? (
                  <>
                    <div style={{ textAlign: 'center', padding: '8px 0' }}>
                      <div style={{ fontSize: 36, fontWeight: 'bold', color: EMOTION_COLORS[EMOTION_LABELS.indexOf(anxietyState.dominantEmotion)] || '#666' }}>
                        {anxietyState.dominantEmotion}
                      </div>
                      <Tag color="purple" style={{ fontSize: 13, padding: '4px 12px', marginTop: 8 }}>
                        置信度 {Math.round(anxietyState.confidence * 100)}%
                      </Tag>
                    </div>
                    <Divider style={{ margin: '12px 0' }} />
                    {anxietyState.emotionProbs.map((prob, i) => (
                      <div key={EMOTION_LABELS[i]} style={{ marginBottom: 6 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 2 }}>
                          <span style={{ color: '#666' }}>{EMOTION_LABELS[i]}</span>
                          <span style={{ color: EMOTION_COLORS[i], fontWeight: 500 }}>{Math.round(prob * 100)}%</span>
                        </div>
                        <Progress percent={Math.round(prob * 100)} showInfo={false} size="small" strokeColor={EMOTION_COLORS[i]} />
                      </div>
                    ))}
                    <div style={{ marginTop: 8, textAlign: 'center' }}>
                      <Text type="secondary" style={{ fontSize: 12 }}>
                        <ClockCircleOutlined /> 数据实时同步给咨询师
                      </Text>
                    </div>
                  </>
                ) : (
                  <div style={{ textAlign: 'center', padding: '20px 0' }}>
                    <Text type="secondary">开启浮动窗口的实时感知后显示</Text>
                  </div>
                )}
              </Card>
            </Col>

            <Col xs={24} md={12}>
              <Card title="我的心理画像" style={{ borderRadius: 12 }}>
                {loading ? (
                  <div style={{ textAlign: 'center', padding: 40 }}><Spin /></div>
                ) : myProfile ? (
                  <Row gutter={[8, 8]}>
                    {[
                      { label: '焦虑', value: myProfile.anxiety, color: '#ff4d4f' },
                      { label: '抑郁', value: myProfile.depression, color: '#722ed1' },
                      { label: '压力', value: myProfile.stress, color: '#fa8c16' },
                      { label: '睡眠', value: myProfile.sleepQuality, color: '#1890ff' },
                      { label: '社交', value: myProfile.socialActivity, color: '#52c41a' },
                      { label: '情绪稳定', value: myProfile.emotionalStability, color: '#13c2c2' },
                    ].map(({ label, value, color }) => (
                      <Col span={12} key={label}>
                        <div style={{ marginBottom: 4 }}>
                          <Text type="secondary" style={{ fontSize: 12 }}>{label}</Text>
                          <Text style={{ float: 'right', fontWeight: 600, color }}>{value}</Text>
                        </div>
                        <Progress percent={value} showInfo={false} size="small" strokeColor={color} />
                      </Col>
                    ))}
                  </Row>
                ) : (
                  <div style={{ textAlign: 'center', padding: 40 }}>
                    <Text type="secondary">暂无心理画像数据</Text>
                  </div>
                )}
              </Card>
            </Col>

            <Col xs={24}>
              <Card style={{ borderRadius: 12, background: 'linear-gradient(135deg, #f0f9ff 0%, #f5f3ff 100%)' }}>
                <div style={{ textAlign: 'center' }}>
                  <Text style={{ fontSize: 15, color: '#6366f1' }}>
                    {consentGiven ? (
                      <>
                        {shareAnxiety && '✅ 焦虑状态 '}
                        {shareProfile && '✅ 心理画像 '}
                        {shareBehavior && '✅ 行为数据 '}
                        已同步给咨询师
                      </>
                    ) : (
                      '你已关闭所有数据共享，咨询师将无法看到你的状态数据'
                    )}
                  </Text>
                </div>
              </Card>
            </Col>

            {/* 多模态数据采集面板 */}
            <Col xs={24}>
              <Collapse
                ghost
                defaultActiveKey={['behavior']}
                items={[{
                  key: 'behavior',
                  label: <Space><ExperimentOutlined style={{ color: '#6366f1' }} /> <Text strong style={{ fontSize: 13 }}>多模态数据采集</Text></Space>,
                  children: (
                    <Row gutter={[12, 12]}>
                      <Col xs={12} md={6}>
                        <div style={{ textAlign: 'center', padding: 8, background: '#f6ffed', borderRadius: 8 }}>
                          <DashboardOutlined style={{ fontSize: 18, color: '#52c41a' }} />
                          <div style={{ fontSize: 11, color: '#888', marginTop: 4 }}>打字速度</div>
                          <div style={{ fontSize: 18, fontWeight: 600, color: '#52c41a' }}>{behaviorData.typingSpeed}<span style={{ fontSize: 11, fontWeight: 400 }}> 字/分</span></div>
                        </div>
                      </Col>
                      <Col xs={12} md={6}>
                        <div style={{ textAlign: 'center', padding: 8, background: '#fff7e6', borderRadius: 8 }}>
                          <ClockCircleOutlined style={{ fontSize: 18, color: '#fa8c16' }} />
                          <div style={{ fontSize: 11, color: '#888', marginTop: 4 }}>停顿频率</div>
                          <div style={{ fontSize: 18, fontWeight: 600, color: '#fa8c16' }}>{behaviorData.pauseFrequency}<span style={{ fontSize: 11, fontWeight: 400 }}>%</span></div>
                        </div>
                      </Col>
                      <Col xs={12} md={6}>
                        <div style={{ textAlign: 'center', padding: 8, background: '#fff1f0', borderRadius: 8 }}>
                          <SoundOutlined style={{ fontSize: 18, color: '#ff4d4f' }} />
                          <div style={{ fontSize: 11, color: '#888', marginTop: 4 }}>删除率</div>
                          <div style={{ fontSize: 18, fontWeight: 600, color: '#ff4d4f' }}>{behaviorData.deletionRate}<span style={{ fontSize: 11, fontWeight: 400 }}>%</span></div>
                        </div>
                      </Col>
                      <Col xs={12} md={6}>
                        <div style={{ textAlign: 'center', padding: 8, background: '#e6f7ff', borderRadius: 8 }}>
                          <ClockCircleOutlined style={{ fontSize: 18, color: '#1890ff' }} />
                          <div style={{ fontSize: 11, color: '#888', marginTop: 4 }}>会话时长</div>
                          <div style={{ fontSize: 18, fontWeight: 600, color: '#1890ff' }}>{behaviorData.sessionMinutes}<span style={{ fontSize: 11, fontWeight: 400 }}> 分钟</span></div>
                        </div>
                      </Col>
                      {bookingType === 'VOICE' && (
                        <Col xs={24}>
                          <div style={{ padding: '8px 12px', background: audioActive ? '#f6ffed' : '#f5f5f5', borderRadius: 8, display: 'flex', alignItems: 'center', gap: 8 }}>
                            <SoundOutlined style={{ color: audioActive ? '#52c41a' : '#bbb' }} />
                            <Text style={{ fontSize: 12, flex: 1 }}>{audioActive ? '音频采集中' : '音频未采集'}</Text>
                            {audioActive && <div style={{ width: 60, height: 6, background: '#e8e8e8', borderRadius: 3 }}><div style={{ width: `${Math.min(100, audioLevel * 2)}%`, height: '100%', background: '#52c41a', borderRadius: 3, transition: 'width 0.1s' }} /></div>}
                            <Tag color={audioActive ? 'green' : 'default'} style={{ fontSize: 10, margin: 0 }}>{audioLevel}</Tag>
                          </div>
                        </Col>
                      )}
                      {bookingType === 'VIDEO' && (
                        <Col xs={24}>
                          <div style={{ padding: '8px 12px', background: videoActive ? '#f6ffed' : '#f5f5f5', borderRadius: 8, display: 'flex', alignItems: 'center', gap: 8 }}>
                            <VideoCameraOutlined style={{ color: videoActive ? '#52c41a' : '#bbb' }} />
                            <Text style={{ fontSize: 12, flex: 1 }}>{videoActive ? '视频采集中' : '视频未采集'}</Text>
                            <Tag color={videoActive ? 'green' : 'default'} style={{ fontSize: 10, margin: 0 }}>{videoActive ? 'ON' : 'OFF'}</Tag>
                          </div>
                        </Col>
                      )}
                      <Col xs={24}>
                        <div style={{ textAlign: 'center', padding: '6px 0' }}>
                          <Text type="secondary" style={{ fontSize: 11 }}>数据仅用于辅助咨询师分析，不会泄露您的隐私</Text>
                        </div>
                      </Col>
                    </Row>
                  ),
                }]}
              />
            </Col>
          </Row>
        </div>

        {/* 右侧：根据预约类型渲染 */}
        <Card
          title={<Space>
            {bookingType === 'TEXT' && <><MessageOutlined /> 与咨询师对话</>}
            {bookingType === 'VOICE' && <><PhoneOutlined /> 语音通话</>}
            {bookingType === 'VIDEO' && <><VideoCameraOutlined /> 视频通话</>}
          </Space>}
          extra={
            <Button
              size="small"
              danger
              onClick={() => {
                Modal.confirm({
                  title: '结束咨询',
                  content: '确定要结束本次咨询吗？',
                  okText: '确定结束',
                  cancelText: '取消',
                  okButtonProps: { danger: true },
                  onOk: () => {
                    const socket = getSocket();
                    socket.emit('consultation:end', bookingId);
                    message.success('咨询已结束');
                    navigate('/patient/bookings');
                  },
                });
              }}
            >
              结束咨询
            </Button>
          }
          style={{ width: 380, borderRadius: 12, flexShrink: 0, display: 'flex', flexDirection: 'column' }}
          bodyStyle={{ flex: 1, display: 'flex', flexDirection: 'column', padding: 0, overflow: 'hidden' }}
        >
          {bookingType === 'TEXT' && (
            <>
              <div style={{ flex: 1, overflow: 'auto', padding: 16, marginBottom: 0 }}>
                {messages.length === 0 ? (
                  <Empty description="等待咨询师发起对话..." image={Empty.PRESENTED_IMAGE_SIMPLE} />
                ) : (
                  messages.map((msg) => (
                    <div key={msg.id} style={{ display: 'flex', marginBottom: 12, flexDirection: msg.senderId === 'me' ? 'row-reverse' : 'row' }}>
                      <Avatar icon={<UserOutlined />} style={{ backgroundColor: msg.senderId === 'me' ? '#6366f1' : '#ffb6c1', flexShrink: 0 }} />
                      <div style={{ maxWidth: '70%', margin: '0 8px', padding: '8px 12px', borderRadius: 12, background: msg.senderId === 'me' ? '#6366f1' : '#f5f5f5', color: msg.senderId === 'me' ? '#fff' : '#333', fontSize: 14, lineHeight: 1.6 }}>
                        {msg.content}
                      </div>
                    </div>
                  ))
                )}
                <div ref={messagesEndRef} />
              </div>
              <div style={{ padding: '12px 16px', borderTop: '1px solid #f0f0f0', display: 'flex', gap: 8 }}>
                <Input ref={inputRef} value={input} onChange={handleInputChange} onPressEnter={handleSend} placeholder="输入消息..." disabled={sending} style={{ borderRadius: 8 }} />
                <Button type="primary" icon={<SendOutlined />} onClick={handleSend} loading={sending} style={{ background: '#6366f1', borderColor: '#6366f1', borderRadius: 8 }} />
              </div>
            </>
          )}

          {bookingType === 'VOICE' && (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: 24 }}>
              <Avatar size={80} style={{ backgroundColor: '#ffb6c1', marginBottom: 16 }} icon={<UserOutlined />} />
              <Title level={5} style={{ marginBottom: 4 }}>咨询师</Title>
              <Text type="secondary" style={{ marginBottom: 24 }}>
                {callActive ? `通话中 ${formatDuration(callDuration)}` : '等待咨询师接听...'}
              </Text>
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
              {/* 远端视频（咨询师）—— 主画面 */}
              <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
                {webrtc.remoteStream ? (
                  <video
                    ref={remoteVideoRef}
                    autoPlay
                    playsInline
                    style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                  />
                ) : (
                  <div style={{ textAlign: 'center' }}>
                    <Avatar size={100} style={{ backgroundColor: '#ffb6c1' }} icon={<UserOutlined />} />
                    <div style={{ marginTop: 12 }}>
                      <Text style={{ color: '#aaa', fontSize: 13 }}>
                        {webrtc.isConnected ? '已连接' : videoActive ? '等待咨询师接入...' : '点击下方按钮开始视频通话'}
                      </Text>
                    </div>
                  </div>
                )}
                {webrtc.error && (
                  <Tag color="red" style={{ position: 'absolute', top: 12, left: '50%', transform: 'translateX(-50%)' }}>{webrtc.error}</Tag>
                )}
              </div>
              {/* 本地视频（自己）—— 画中画 */}
              {cameraOn && videoStream && (
                <div style={{ position: 'absolute', top: 12, right: 12, width: 120, height: 160, background: '#333', borderRadius: 8, overflow: 'hidden', border: '2px solid rgba(255,255,255,0.2)' }}>
                  <video
                    ref={videoRef}
                    autoPlay
                    muted
                    playsInline
                    style={{ width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)' }}
                  />
                </div>
              )}
              <div style={{ position: 'absolute', top: 12, left: 12 }}>
                <Text style={{ color: '#fff', fontSize: 13 }}>咨询师</Text>
                {callActive && <div><Text style={{ color: '#aaa', fontSize: 12 }}>{formatDuration(callDuration)}</Text></div>}
                {webrtc.isConnected && <Tag color="green" style={{ marginTop: 4, fontSize: 10 }}>已连接</Tag>}
              </div>
              <div style={{ padding: 16, display: 'flex', justifyContent: 'center', gap: 16, background: 'rgba(0,0,0,0.5)' }}>
                <Button shape="circle" icon={<VideoCameraOutlined />} type={cameraOn ? 'primary' : 'default'} onClick={() => { setCameraOn(!cameraOn); webrtc.toggleCamera(!cameraOn); }} />
                <Button shape="circle" icon={<SoundOutlined />} type={micOn ? 'primary' : 'default'} onClick={() => { setMicOn(!micOn); webrtc.toggleMic(!micOn); }} />
                <Button shape="circle" icon={callActive ? <StopOutlined /> : <PhoneOutlined />} danger={callActive} onClick={() => {
                  if (callActive) { webrtc.hangUp(); setCallActive(false); setCallDuration(0); }
                  else { startVideoCapture(); setCallActive(true); setCallDuration(0); }
                }} />
              </div>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
