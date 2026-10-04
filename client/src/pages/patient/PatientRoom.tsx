import { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card, Typography, Tag, Space, Avatar, Button, Input, Empty, Modal, Switch, message, Tooltip, Badge } from 'antd';
import {
  ArrowLeftOutlined, ThunderboltOutlined, ClockCircleOutlined,
  SendOutlined, MessageOutlined, UserOutlined,
  PhoneOutlined, VideoCameraOutlined, SoundOutlined, StopOutlined,
  ExperimentOutlined, SafetyCertificateOutlined,
  EyeOutlined, EyeInvisibleOutlined, CaretLeftOutlined, CaretRightOutlined,
} from '@ant-design/icons';
import { useAnxiety } from '../../context/AnxietyContext';
import api from '../../services/api';
import { getSocket } from '../../services/socket';
import { useWebRTC } from '../../hooks/useWebRTC';

const { Title, Text } = Typography;

const EMOTION_LABELS = ['快乐', '悲伤', '焦虑', '愤怒', '中性'];
const EMOTION_COLORS = ['#52c41a', '#722ed1', '#ff4d4f', '#fa8c16', '#1890ff'];

export default function PatientRoom() {
  const { bookingId } = useParams();
  const navigate = useNavigate();
  const { comprehensiveState: anxietyState, baseline, baselineDeviation, moodTrajectory, interventionHistory } = useAnxiety();
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

  // 侧边栏折叠
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true);

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
  const webrtc = useWebRTC(bookingId || null);

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

  // 视频采集（视频通话模式）
  const startVideoCapture = async () => {
    try {
      const stream = await webrtc.startLocalStream(true, true);
      if (stream) {
        setVideoStream(stream);
        setVideoActive(true);
        await webrtc.call();
      }
    } catch (err) {
      console.warn('摄像头访问失败:', err);
    }
  };

  useEffect(() => {
    if (videoStream && videoRef.current) {
      videoRef.current.srcObject = videoStream;
      videoRef.current.play().catch(() => {});
    }
    // 自动接听场景：hook 中获取了本地流但 videoStream 未更新
    if (webrtc.localStream && !videoStream) {
      setVideoStream(webrtc.localStream);
      setVideoActive(true);
    }
  }, [videoStream, webrtc.localStream]);

  useEffect(() => {
    if (webrtc.remoteStream && remoteVideoRef.current) {
      remoteVideoRef.current.srcObject = webrtc.remoteStream;
      remoteVideoRef.current.play().catch(() => {});
    }
  }, [webrtc.remoteStream]);

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

  useEffect(() => {
    sessionStart.current = Date.now();
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

    const multimodalInterval = setInterval(() => {
      if (!consentGiven) return;
      if (anxietyState.activeModalities.length === 0) return;

      const filteredData: any = { timestamp: Date.now() };

      if (shareBehavior) {
        filteredData.comprehensiveState = {
          activeModalities: anxietyState.activeModalities,
          emotionProbs: anxietyState.emotionProbs,
          dominantEmotion: anxietyState.dominantEmotion,
          riskLevel: anxietyState.riskLevel,
          confidence: anxietyState.confidence,
          narrativeAnalysis: anxietyState.narrativeAnalysis,
          caringMessage: anxietyState.caringMessage,
          evidence: anxietyState.evidence,
          fusionWeights: anxietyState.fusionWeights,
          text: anxietyState.text,
          voice: anxietyState.voice,
          facial: anxietyState.facial,
          keyboard: anxietyState.keyboard,
          circadian: anxietyState.circadian,
          cognitiveDistortion: anxietyState.cognitiveDistortion,
          hrv: anxietyState.hrv,
          breathing: anxietyState.breathing,
          behavioralActivation: anxietyState.behavioralActivation,
          eyeMovement: anxietyState.eyeMovement,
          voiceSemantics: anxietyState.voiceSemantics,
          baseline: baselineDeviation.calibrated ? {
            heartRate: baseline.heartRate,
            breathingRate: baseline.breathingRate,
            typingSpeed: baseline.typingSpeed,
            blinkRate: baseline.blinkRate,
            voiceF0: baseline.voiceF0,
            speechRate: baseline.speechRate,
            sampleCount: baseline.sampleCount,
            calibrated: baselineDeviation.calibrated,
          } : null,
          trajectorySummary: moodTrajectory.points.length >= 5 ? {
            trend: moodTrajectory.trend,
            trendStrength: moodTrajectory.trendStrength,
            weeklyChange: moodTrajectory.weeklyChange,
            recentPoints: moodTrajectory.points.slice(-14).map(p => ({
              t: p.timestamp,
              a: Math.round(p.anxietyProb * 100),
              e: p.dominantEmotion,
            })),
          } : null,
          interventionHistory: interventionHistory.records.length > 0 ? interventionHistory : null,
        };
      }

      if (shareAnxiety) {
        const anxietyProb = anxietyState.emotionProbs?.[2] || 0;
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

      if (filteredData.comprehensiveState || filteredData.anxietyState || filteredData.profile) {
        socket.emit('multimodal:update', { bookingId, multimodalData: filteredData });
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

  const handleEndConsultation = () => {
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
  };

  // ===== 渲染 =====
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 60px)', background: '#f8f6ff' }}>
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
            <Text type="secondary">你需要知道公益咨询师能看到什么</Text>
          </div>
          <div style={{ background: '#f9f0ff', padding: '16px 20px', borderRadius: 12, marginBottom: 16 }}>
            <Text style={{ fontSize: 14, color: '#5a4a6a' }}>
               你的隐私很重要。以下数据会在咨询过程中同步给公益咨询师，你可以选择关闭不想分享的：
            </Text>
          </div>
          <div style={{ marginBottom: 16 }}>
            {[
              { icon: <ThunderboltOutlined style={{ color: '#6366f1' }} />, label: '焦虑状态', desc: '键盘打字节奏、眨眼频率计算的焦虑指数', value: shareAnxiety, onChange: setShareAnxiety },
              { icon: <ExperimentOutlined style={{ color: '#1890ff' }} />, label: '心理画像', desc: '焦虑、抑郁、压力等维度评分', value: shareProfile, onChange: setShareProfile },
              { icon: <ExperimentOutlined style={{ color: '#52c41a' }} />, label: '行为数据', desc: '打字速度、停顿频率、删除率', value: shareBehavior, onChange: setShareBehavior },
            ].map((item, i) => (
              <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 0', borderBottom: i < 2 ? '1px solid #f0f0f0' : 'none' }}>
                <Space>
                  {item.icon}
                  <div>
                    <Text strong style={{ fontSize: 14 }}>{item.label}</Text>
                    <div><Text type="secondary" style={{ fontSize: 12 }}>{item.desc}</Text></div>
                  </div>
                </Space>
                <Switch checked={item.value} onChange={item.onChange} />
              </div>
            ))}
          </div>
          <div style={{ background: '#f6ffed', padding: '12px 16px', borderRadius: 8, marginBottom: 16 }}>
            <Text style={{ fontSize: 13, color: '#52c41a' }}>
              🔒 所有数据仅用于辅助公益咨询师了解你的状态，不会泄露给任何第三方。
            </Text>
          </div>
          <Button type="primary" block size="large" onClick={() => { setConsentGiven(true); setConsentModalOpen(false); }} style={{ borderRadius: 12 }}>
            我了解了，进入咨询室
          </Button>
        </div>
      </Modal>

      {/* 顶部导航栏 */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '10px 20px', background: '#fff', borderBottom: '1px solid #f0eef5', flexShrink: 0,
      }}>
        <Space>
          <Button type="text" icon={<ArrowLeftOutlined />} onClick={() => navigate('/experts')} />
          <div>
            <Text strong style={{ fontSize: 15 }}>咨询室</Text>
            <div><Text type="secondary" style={{ fontSize: 12 }}>与公益咨询师实时连线中</Text></div>
          </div>
        </Space>
        <Space size={12}>
          {/* 数据同步状态指示器 */}
          {consentGiven && (
            <Tooltip title={`${[shareAnxiety && '焦虑状态', shareProfile && '心理画像', shareBehavior && '行为数据'].filter(Boolean).join('、')} 已同步`}>
              <Tag color="green" style={{ borderRadius: 20, padding: '2px 12px', cursor: 'default' }}>
                <Badge status="success" style={{ marginRight: 4 }} />
                数据同步中
              </Tag>
            </Tooltip>
          )}
          <Button size="small" danger onClick={handleEndConsultation}>结束咨询</Button>
        </Space>
      </div>

      {/* 主体区域 */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* 左侧数据侧边栏（可折叠） */}
        <div style={{
          width: sidebarCollapsed ? 0 : 300, flexShrink: 0, overflow: 'hidden',
          background: '#fff', borderRight: sidebarCollapsed ? 'none' : '1px solid #f0eef5',
          transition: 'width 0.3s ease', display: 'flex', flexDirection: 'column',
        }}>
          {(!sidebarCollapsed) && (
            <div style={{ flex: 1, overflow: 'auto', padding: 16 }}>
              {/* 主导情绪 */}
              {anxietyState.activeModalities.length > 0 && (
                <div style={{ textAlign: 'center', marginBottom: 20 }}>
                  <div style={{ fontSize: 28, fontWeight: 'bold', color: EMOTION_COLORS[EMOTION_LABELS.indexOf(anxietyState.dominantEmotion)] || '#666' }}>
                    {anxietyState.dominantEmotion}
                  </div>
                  <Tag color="purple" style={{ marginTop: 6 }}>置信度 {Math.round(anxietyState.confidence * 100)}%</Tag>
                </div>
              )}

              {/* 情绪分布 */}
              {anxietyState.activeModalities.length > 0 && anxietyState.emotionProbs?.length > 0 && (
                <div style={{ marginBottom: 20 }}>
                  <Text strong style={{ fontSize: 13, display: 'block', marginBottom: 10, color: '#5a4a6a' }}>情绪分布</Text>
                  {anxietyState.emotionProbs.map((prob: number, i: number) => (
                    <div key={EMOTION_LABELS[i]} style={{ marginBottom: 8 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 2 }}>
                        <span style={{ color: '#888' }}>{EMOTION_LABELS[i]}</span>
                        <span style={{ color: EMOTION_COLORS[i], fontWeight: 500 }}>{Math.round(prob * 100)}%</span>
                      </div>
                      <div style={{ height: 4, background: '#f0f0f0', borderRadius: 2 }}>
                        <div style={{ height: '100%', width: `${prob * 100}%`, background: EMOTION_COLORS[i], borderRadius: 2, transition: 'width 0.5s' }} />
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* 心理画像 */}
              {!loading && myProfile && (
                <div style={{ marginBottom: 20 }}>
                  <Text strong style={{ fontSize: 13, display: 'block', marginBottom: 10, color: '#5a4a6a' }}>心理画像</Text>
                  {[
                    { label: '焦虑', value: myProfile.anxiety, color: '#ff4d4f' },
                    { label: '抑郁', value: myProfile.depression, color: '#722ed1' },
                    { label: '压力', value: myProfile.stress, color: '#fa8c16' },
                    { label: '社交', value: myProfile.socialActivity, color: '#52c41a' },
                    { label: '情绪稳定', value: myProfile.emotionalStability, color: '#13c2c2' },
                  ].map(({ label, value, color }) => (
                    <div key={label} style={{ marginBottom: 8 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 2 }}>
                        <Text type="secondary">{label}</Text>
                        <Text style={{ fontWeight: 600, color, fontSize: 12 }}>{value}</Text>
                      </div>
                      <div style={{ height: 4, background: '#f0f0f0', borderRadius: 2 }}>
                        <div style={{ height: '100%', width: `${value}%`, background: color, borderRadius: 2 }} />
                      </div>
                    </div>
                  ))}
                  {/* 睡眠无法自动检测：只有用户手动记录过才显示分数 */}
                  <div style={{ marginBottom: 8 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 2 }}>
                      <Text type="secondary">睡眠</Text>
                      {(myProfile.sleepQuality || 0) > 0
                        ? <Text style={{ fontWeight: 600, color: '#1890ff', fontSize: 12 }}>{myProfile.sleepQuality}</Text>
                        : <Text type="secondary" style={{ fontSize: 12 }}>未记录</Text>}
                    </div>
                    <div style={{ height: 4, background: '#f0f0f0', borderRadius: 2 }}>
                      <div style={{ height: '100%', width: `${(myProfile.sleepQuality || 0)}%`, background: '#1890ff', borderRadius: 2 }} />
                    </div>
                  </div>
                </div>
              )}

              {/* 行为数据 */}
              <div style={{ marginBottom: 16 }}>
                <Text strong style={{ fontSize: 13, display: 'block', marginBottom: 10, color: '#5a4a6a' }}>行为数据</Text>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                  {[
                    { label: '打字速度', value: `${behaviorData.typingSpeed} 字/分`, color: '#52c41a', bg: '#f6ffed' },
                    { label: '停顿频率', value: `${behaviorData.pauseFrequency}%`, color: '#fa8c16', bg: '#fff7e6' },
                    { label: '删除率', value: `${behaviorData.deletionRate}%`, color: '#ff4d4f', bg: '#fff1f0' },
                    { label: '会话时长', value: `${behaviorData.sessionMinutes} 分钟`, color: '#1890ff', bg: '#e6f7ff' },
                  ].map((item, i) => (
                    <div key={i} style={{ padding: 8, background: item.bg, borderRadius: 8, textAlign: 'center' }}>
                      <div style={{ fontSize: 10, color: '#888' }}>{item.label}</div>
                      <div style={{ fontSize: 14, fontWeight: 600, color: item.color }}>{item.value}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* 共享控制 */}
              <div style={{ borderTop: '1px solid #f0f0f0', paddingTop: 12 }}>
                <Text strong style={{ fontSize: 13, display: 'block', marginBottom: 10, color: '#5a4a6a' }}>
                  <EyeOutlined style={{ marginRight: 4 }} />数据共享控制
                </Text>
                {[
                  { label: '焦虑状态', value: shareAnxiety, onChange: setShareAnxiety },
                  { label: '心理画像', value: shareProfile, onChange: setShareProfile },
                  { label: '行为数据', value: shareBehavior, onChange: setShareBehavior },
                ].map((item, i) => (
                  <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 0' }}>
                    <Text style={{ fontSize: 13 }}>{item.label}</Text>
                    <Switch size="small" checked={item.value} onChange={item.onChange} />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* 折叠/展开按钮 */}
        <div
          onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
          style={{
            width: 20, display: 'flex', alignItems: 'center', justifyContent: 'center',
            cursor: 'pointer', background: '#fff', borderRight: '1px solid #f0eef5',
            flexShrink: 0, zIndex: 10,
          }}
        >
          {sidebarCollapsed ? <CaretRightOutlined style={{ fontSize: 10, color: '#bbb' }} /> : <CaretLeftOutlined style={{ fontSize: 10, color: '#bbb' }} />}
        </div>

        {/* 右侧主区域：对话/通话 */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {bookingType === 'TEXT' && (
            <>
              <div style={{ flex: 1, overflow: 'auto', padding: 20 }}>
                {messages.length === 0 ? (
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#bbb' }}>
                    <MessageOutlined style={{ fontSize: 48, marginBottom: 16, color: '#ddd' }} />
                    <Text type="secondary">等待公益咨询师发起对话...</Text>
                  </div>
                ) : (
                  <div>
                    {messages.map((msg) => (
                      <div key={msg.id} style={{ display: 'flex', marginBottom: 12, flexDirection: msg.senderId === 'me' ? 'row-reverse' : 'row' }}>
                        <Avatar icon={<UserOutlined />} style={{ backgroundColor: msg.senderId === 'me' ? '#6366f1' : '#ffb6c1', flexShrink: 0 }} />
                        <div style={{ maxWidth: '65%', margin: '0 10px', padding: '10px 14px', borderRadius: 16, background: msg.senderId === 'me' ? '#6366f1' : '#fff', color: msg.senderId === 'me' ? '#fff' : '#333', fontSize: 14, lineHeight: 1.6, boxShadow: msg.senderId === 'me' ? 'none' : '0 1px 4px rgba(0,0,0,0.06)' }}>
                          {msg.content}
                        </div>
                      </div>
                    ))}
                    <div ref={messagesEndRef} />
                  </div>
                )}
              </div>
              <div style={{ padding: '12px 20px', borderTop: '1px solid #f0eef5', background: '#fff', display: 'flex', gap: 10 }}>
                <Input ref={inputRef} value={input} onChange={handleInputChange} onPressEnter={handleSend} placeholder="输入消息..." disabled={sending} size="large" style={{ borderRadius: 24 }} />
                <Button type="primary" icon={<SendOutlined />} onClick={handleSend} loading={sending} size="large" style={{ background: '#6366f1', borderColor: '#6366f1', borderRadius: 24, width: 48 }} />
              </div>
            </>
          )}

          {bookingType === 'VOICE' && (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: 24, background: 'linear-gradient(180deg, #f8f6ff 0%, #fff 100%)' }}>
              <Avatar size={100} style={{ backgroundColor: '#ffb6c1', marginBottom: 20, boxShadow: '0 4px 20px rgba(255,182,193,0.4)' }} icon={<UserOutlined />} />
              <Title level={4} style={{ marginBottom: 4, color: '#5a4a6a' }}>公益咨询师</Title>
              <Text type="secondary" style={{ marginBottom: 32, fontSize: 14 }}>
                {callActive ? `通话中 ${formatDuration(callDuration)}` : '等待公益咨询师接听...'}
              </Text>
              {callActive && (
                <div style={{ display: 'flex', gap: 4, marginBottom: 40, alignItems: 'center' }}>
                  {[...Array(5)].map((_, i) => (
                    <div key={i} style={{ width: 4, height: 20 + Math.random() * 20, background: '#6366f1', borderRadius: 2, animation: `wave 0.8s ease-in-out ${i * 0.1}s infinite alternate` }} />
                  ))}
                </div>
              )}
              <Space size={20}>
                <Button shape="circle" size="large" icon={<SoundOutlined />} type={micOn ? 'primary' : 'default'} onClick={() => setMicOn(!micOn)} style={{ width: 56, height: 56 }} />
                <Button shape="circle" size="large" icon={callActive ? <StopOutlined /> : <PhoneOutlined />} danger={callActive} onClick={() => { setCallActive(!callActive); if (!callActive) setCallDuration(0); }} style={{ width: 56, height: 56 }} />
              </Space>
            </div>
          )}

          {bookingType === 'VIDEO' && (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', position: 'relative', background: '#1a1a2e' }}>
              {/* 远端视频 */}
              <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
                {webrtc.remoteStream ? (
                  <video ref={remoteVideoRef} autoPlay playsInline style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                ) : (
                  <div style={{ textAlign: 'center' }}>
                    <Avatar size={100} style={{ backgroundColor: '#ffb6c1' }} icon={<UserOutlined />} />
                    <div style={{ marginTop: 12 }}>
                      <Text style={{ color: '#aaa', fontSize: 13 }}>
                        {webrtc.isConnected ? '已连接' : videoActive ? '等待公益咨询师接入...' : '点击下方按钮开始视频通话'}
                      </Text>
                    </div>
                  </div>
                )}
                {webrtc.error && (
                  <Tag color="red" style={{ position: 'absolute', top: 12, left: '50%', transform: 'translateX(-50%)' }}>{webrtc.error}</Tag>
                )}
              </div>
              {/* 本地视频画中画 */}
              {cameraOn && videoStream && (
                <div style={{ position: 'absolute', top: 12, right: 12, width: 120, height: 160, background: '#333', borderRadius: 8, overflow: 'hidden', border: '2px solid rgba(255,255,255,0.2)' }}>
                  <video ref={videoRef} autoPlay muted playsInline style={{ width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)' }} />
                </div>
              )}
              {/* 状态信息 */}
              <div style={{ position: 'absolute', top: 12, left: 12 }}>
                <Text style={{ color: '#fff', fontSize: 13 }}>公益咨询师</Text>
                {callActive && <div><Text style={{ color: '#aaa', fontSize: 12 }}>{formatDuration(callDuration)}</Text></div>}
                {webrtc.isConnected && <Tag color="green" style={{ marginTop: 4, fontSize: 10 }}>已连接</Tag>}
              </div>
              {/* 控制栏 */}
              <div style={{ padding: 16, display: 'flex', justifyContent: 'center', gap: 16, background: 'rgba(0,0,0,0.5)' }}>
                <Button shape="circle" size="large" icon={<VideoCameraOutlined />} type={cameraOn ? 'primary' : 'default'} onClick={() => { setCameraOn(!cameraOn); webrtc.toggleCamera(!cameraOn); }} style={{ width: 48, height: 48 }} />
                <Button shape="circle" size="large" icon={<SoundOutlined />} type={micOn ? 'primary' : 'default'} onClick={() => { setMicOn(!micOn); webrtc.toggleMic(!micOn); }} style={{ width: 48, height: 48 }} />
                <Button shape="circle" size="large" icon={callActive ? <StopOutlined /> : <PhoneOutlined />} danger={callActive} onClick={() => {
                  if (callActive) { webrtc.hangUp(); setCallActive(false); setCallDuration(0); }
                  else { startVideoCapture(); setCallActive(true); setCallDuration(0); }
                }} style={{ width: 48, height: 48 }} />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
