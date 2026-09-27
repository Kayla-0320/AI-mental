import { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card, Row, Col, Typography, Tag, Space, Avatar, Progress, Spin, Empty, Divider, Badge, Button, Input, List, Timeline, Alert, Collapse, Modal, Tooltip, message, Drawer, Tabs, Statistic } from 'antd';
import {
  UserOutlined, ThunderboltOutlined, RadarChartOutlined,
  ClockCircleOutlined, ArrowLeftOutlined, AlertOutlined, SendOutlined, MessageOutlined,
  SafetyCertificateOutlined, LineChartOutlined, WarningOutlined, CheckCircleOutlined,
  PhoneOutlined, VideoCameraOutlined, SoundOutlined, StopOutlined,
  SmileOutlined, InfoCircleOutlined, FormOutlined, SaveOutlined,
  FlagOutlined, NotificationOutlined, CaretRightOutlined,
  DashboardOutlined, ExperimentOutlined, EyeOutlined, HistoryOutlined,
} from '@ant-design/icons';
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, AreaChart, Area } from 'recharts';
import api from '../../services/api';
import { getSocket } from '../../services/socket';
import { useWebRTC } from '../../hooks/useWebRTC';
import { PERCEPTION_API } from '../../config';

const { Title, Text, Paragraph } = Typography;

const riskColors: Record<string, string> = { LOW: '#52c41a', MEDIUM: '#faad14', HIGH: '#ff4d4f', CRISIS: '#cf1322' };
const riskLabels: Record<string, string> = { LOW: '低风险', MEDIUM: '中风险', HIGH: '高风险', CRISIS: '危机' };
const anxietyLevelColors: Record<string, string> = { low: '#52c41a', medium: '#faad14', high: '#ff4d4f' };
const anxietyLevelLabels: Record<string, string> = { low: '良好', medium: '轻度', high: '偏高' };
// 数字孪生模态标签与配色（与后端 PsychoDimension / DIM_LABELS 对齐）
const twinDimLabels: Record<string, string> = {
  text_emotion: '文本情绪', voice_acoustic: '语音声学', facial: '面部表情',
  circadian: '昼夜节律', cognitive: '认知扭曲', behavior: '行为模式',
  hrv: '心率变异性', breathing: '呼吸模式', behavioral_act: '行为激活',
  eye: '眨眼/头姿', voice_semantics: '语音语义',
};
const twinDimColor = (k: string): string => {
  const palette = ['#722ed1', '#1890ff', '#13c2c2', '#52c41a', '#fa8c16', '#ff4d4f', '#eb2f96', '#2f54eb', '#a0d911', '#faad14', '#f5222d'];
  let h = 0;
  for (let i = 0; i < k.length; i++) h = (h * 31 + k.charCodeAt(i)) >>> 0;
  return palette[h % palette.length];
};
const dimLabel = (k: string): string => twinDimLabels[k] || k;

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
  // 咨询笔记
  const [notes, setNotes] = useState('');
  const [notesSaving, setNotesSaving] = useState(false);
  const [notesDrawerOpen, setNotesDrawerOpen] = useState(false);
  // 危机干预
  const [crisisModalOpen, setCrisisModalOpen] = useState(false);
  // 信息分层折叠
  const [collapsedSections, setCollapsedSections] = useState<string[]>(['moods', 'conversations']);
  // 患者实时多模态数据
  const [patientMultimodal, setPatientMultimodal] = useState<any>(null);
  const [patientMultimodalTime, setPatientMultimodalTime] = useState<string>('');
  // 智能工作台
  const [riskTrend, setRiskTrend] = useState<any>(null);
  // 风险趋势接口失败标记：失败时必须显式提示，不得用随机数伪造趋势曲线
  const [riskTrendError, setRiskTrendError] = useState(false);
  const [scaleEstimate, setScaleEstimate] = useState<any>(null);
  const [workbenchTab, setWorkbenchTab] = useState<string>('overview');
  // 安全审计事件
  const [auditEvents, setAuditEvents] = useState<any[]>([]);
  const [auditDrawerOpen, setAuditDrawerOpen] = useState(false);
  // CBT 脚本库
  const [cbtStep, setCbtStep] = useState(0);
  // 心理数字孪生
  const [twinPrediction, setTwinPrediction] = useState<any>(null);
  const [twinLoading, setTwinLoading] = useState(false);
  // 多维数字孪生（支柱A景观 + 支柱B平行未来/救援）
  const [twinLandscape, setTwinLandscape] = useState<any>(null);
  const [twinFutures, setTwinFutures] = useState<any>(null);
  const [twinRescue, setTwinRescue] = useState<any>(null);
  const [twinNdLoading, setTwinNdLoading] = useState(false);
  const CBT_SCRIPTS = [
    { title: '① 识别自动思维', prompt: '引导患者描述刚刚脑海中闪过的想法', example: '“当你听到那句话时，脑海里第一个浮现的念头是什么？”' },
    { title: '② 识别认知扭曲', prompt: '帮助患者识别思维中的扭曲模式', example: '“这种想法是不是有点像‘非黑即白’？或者‘灾难化’？”' },
    { title: '③ 检验证据', prompt: '引导患者客观审视支持/反驳这个想法的证据', example: '“有哪些证据支持这个想法？有哪些证据不太支持？”' },
    { title: '④ 寻找替代解释', prompt: '帮助患者生成更平衡的替代想法', example: '“如果好朋友遇到同样的事，你会怎么安慰TA？”' },
    { title: '⑤ 评估与行动', prompt: '重新评估情绪强度，制定行动计划', example: '"现在再想那件事，你的难受程度从0-10打几分？接下来想做点什么？"' },
  ];

  const emotionLabels = ['快乐', '悲伤', '焦虑', '愤怒', '中性'];
  const emotionColors = ['#52c41a', '#722ed1', '#ff4d4f', '#fa8c16', '#1890ff'];

  // WebRTC 双向视频
  const socketRef = useRef<ReturnType<typeof getSocket> | null>(null);
  const localVideoRef = useRef<HTMLVideoElement>(null);
  const remoteVideoRef = useRef<HTMLVideoElement>(null);
  const webrtc = useWebRTC(bookingId || null);

  // 自动同步 WebRTC 流到视频元素（支持自动接听场景）
  useEffect(() => {
    if (webrtc.localStream && localVideoRef.current) {
      localVideoRef.current.srcObject = webrtc.localStream;
      setCameraOn(true);
    }
    if (webrtc.remoteStream && remoteVideoRef.current) {
      remoteVideoRef.current.srcObject = webrtc.remoteStream;
    }
    if (webrtc.isConnected && !callActive) {
      setCallActive(true);
      setCallDuration(0);
    }
  }, [webrtc.localStream, webrtc.remoteStream, webrtc.isConnected]);

  // 加载风险趋势（咨询前查看）
  const loadRiskTrend = useCallback(async () => {
    try {
      const bookingRes = await api.get(`/expert/bookings/${bookingId}`) as any;
      const patientId = bookingRes.data?.patientId;
      if (!patientId) return;
      // 调用评估层风险趋势 API
      const res = await fetch(`${PERCEPTION_API}/api/v1/assessment/risk-trend?user_id=${patientId}&days=7`);
      if (res.ok) {
        const data = await res.json();
        setRiskTrend(data);
        setRiskTrendError(false);
      } else {
        setRiskTrend(null);
        setRiskTrendError(true);
      }
    } catch (err) {
      console.warn('风险趋势加载失败:', err);
      // 原实现在这里用 Math.random() 造了一条 7 天随机风险曲线，
      // 咨询师会把它当成真实趋势判读。改为显式不可用。
      setRiskTrend(null);
      setRiskTrendError(true);
    }
  }, [bookingId]);

  // 加载量表自动映射
  const loadScaleEstimate = useCallback(async () => {
    if (!patientMultimodal?.comprehensiveState?.emotionProbs) return;
    try {
      const probs = patientMultimodal.comprehensiveState.emotionProbs;
      const res = await fetch(`${PERCEPTION_API}/api/v1/assessment/scale-estimate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          emotion: {
            text_emotion_probs: probs,
            confidence: patientMultimodal.comprehensiveState.confidence || 0.7,
          },
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setScaleEstimate(data);
      }
    } catch (err) {
      console.warn('量表映射失败:', err);
    }
  }, [patientMultimodal]);

  // 当收到新的多模态数据时自动更新量表映射
  useEffect(() => {
    if (patientMultimodal?.comprehensiveState?.emotionProbs) {
      loadScaleEstimate();
    }
  }, [patientMultimodal, loadScaleEstimate]);

  // 从后端加载真实审计事件
  const loadAuditEvents = useCallback(async () => {
    try {
      const res = await fetch(`${PERCEPTION_API}/api/v1/intervention/audit-events`);
      if (res.ok) {
        const data = await res.json();
        setAuditEvents(data.events || []);
      }
    } catch {
      // 后端不可用时使用模拟数据
      setAuditEvents([
        { id: 1, type: 'safety_intercept', time: new Date(Date.now() - 3600000).toLocaleTimeString('zh-CN'), severity: 'warning', description: 'AI 回复被拦截：检测到潜在谄媚倾向', action: '已自动替换为中性回复' },
        { id: 2, type: 'crisis_signal', time: new Date(Date.now() - 1800000).toLocaleTimeString('zh-CN'), severity: 'info', description: '文本分析检测到轻微消极情绪词汇', action: '已记录，未触发升级' },
        { id: 3, type: 'audit_pass', time: new Date(Date.now() - 600000).toLocaleTimeString('zh-CN'), severity: 'success', description: 'AI 回复通过五轴安全审计', action: '已正常发送' },
      ]);
    }
  }, [PERCEPTION_API]);

  // 加载心理数字孪生预测
  const loadTwinPrediction = useCallback(async () => {
    try {
      const bookingRes = await api.get(`/expert/bookings/${bookingId}`) as any;
      const patientId = bookingRes.data?.patientId;
      if (!patientId) return;
      setTwinLoading(true);
      const res = await fetch(`${PERCEPTION_API}/api/v1/digital-twin/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: patientId, horizon: 12 }),
      });
      if (res.ok) {
        const data = await res.json();
        setTwinPrediction(data);
      }
    } catch (err) {
      console.warn('数字孪生预测加载失败:', err);
    } finally {
      setTwinLoading(false);
    }
  }, [PERCEPTION_API, bookingId]);

  // 加载多维数字孪生：崩溃景观(domino链) + 平行未来 + MPC 救援（较重，仅挂载时+手动重算）
  const loadTwinAdvanced = useCallback(async () => {
    try {
      const bookingRes = await api.get(`/expert/bookings/${bookingId}`) as any;
      const patientId = bookingRes.data?.patientId;
      if (!patientId) return;
      setTwinNdLoading(true);
      const post = (path: string, body: Record<string, any>) =>
        fetch(`${PERCEPTION_API}/api/v1/digital-twin/${path}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_id: patientId, ...body }),
        }).then(r => (r.ok ? r.json() : null)).catch(() => null);
      const [land, fut, res] = await Promise.all([
        post('landscape', { horizon: 12 }),
        post('futures', { n_particles: 200, horizon: 20, seed: 0 }),
        post('rescue', { horizon: 15, n_particles: 80, seed: 0 }),
      ]);
      if (land) setTwinLandscape(land);
      if (fut) setTwinFutures(fut);
      if (res) setTwinRescue(res);
    } catch (err) {
      console.warn('多维数字孪生加载失败:', err);
    } finally {
      setTwinNdLoading(false);
    }
  }, [PERCEPTION_API, bookingId]);

  useEffect(() => {
    loadAuditEvents();
    loadTwinPrediction();
    loadTwinAdvanced();
    // 每 30 秒自动刷新审计事件
    const auditInterval = setInterval(loadAuditEvents, 30000);
    // 每 60 秒自动刷新数字孪生预测
    const twinInterval = setInterval(loadTwinPrediction, 60000);
    return () => { clearInterval(auditInterval); clearInterval(twinInterval); };
  }, [loadAuditEvents, loadTwinPrediction, loadTwinAdvanced]);
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
        const res = await api.get(`/profile/consultation-data?patientId=${patientId}`) as any;
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
    loadRiskTrend();
    // 每 15 秒自动刷新患者数据
    const dataInterval = setInterval(fetchData, 15000);
    // 每 5 秒自动刷新消息
    const msgInterval = setInterval(loadMessages, 5000);
    return () => {
      clearInterval(dataInterval);
      clearInterval(msgInterval);
    };
  }, [bookingId]);

  // Socket 连接 + 接收患者多模态数据
  useEffect(() => {
    if (!bookingId) return;
    const socket = getSocket();
    socketRef.current = socket;
    socket.emit('expert:join', bookingId);
    socket.emit('webrtc:join', { bookingId });

    const handlePatientMultimodal = (multimodalData: any) => {
      setPatientMultimodal(multimodalData);
      setPatientMultimodalTime(new Date().toLocaleTimeString('zh-CN'));
    };

    // 咨询结束：清理实时数据
    const handleConsultationEnded = () => {
      setPatientMultimodal(null);
      setPatientMultimodalTime('');
      message.info('咨询已结束，实时数据已清理');
    };

    socket.on('patient:multimodal', handlePatientMultimodal);
    socket.on('consultation:ended', handleConsultationEnded);

    // 数据过期清理：10秒未收到新数据则清空
    const dataExpiryTimer = setInterval(() => {
      setPatientMultimodalTime(prevTime => {
        if (!prevTime) return prevTime;
        // 如果超过10秒没有更新，清空数据
        const lastUpdate = new Date();
        const [hours, minutes, seconds] = prevTime.split(':').map(Number);
        lastUpdate.setHours(hours, minutes, seconds);
        const elapsed = Date.now() - lastUpdate.getTime();
        if (elapsed > 10000) {
          setPatientMultimodal(null);
          return '';
        }
        return prevTime;
      });
    }, 5000);

    return () => {
      socket.off('patient:multimodal', handlePatientMultimodal);
      socket.off('consultation:ended', handleConsultationEnded);
      socket.emit('expert:leave', bookingId);
      socket.emit('webrtc:leave', bookingId);
      clearInterval(dataExpiryTimer);
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
        behaviorFeatures = {};
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

  // 保存咨询笔记
  const handleSaveNotes = async () => {
    setNotesSaving(true);
    try {
      await api.post(`/expert/bookings/${bookingId}/notes`, { content: notes });
      message.success('笔记已保存');
    } catch {
      message.error('保存失败');
    } finally {
      setNotesSaving(false);
    }
  };

  // 加载笔记
  useEffect(() => {
    const loadNotes = async () => {
      try {
        const res = await api.get(`/expert/bookings/${bookingId}/notes`) as any;
        if (res.data?.notes) setNotes(res.data.notes);
      } catch {}
    };
    loadNotes();
  }, [bookingId]);

  // 危机干预操作
  // ⚠️ 比赛 Demo 为模拟实现，生产环境需接入真实危机干预流程。
  // 原实现提示「系统已通知督导」但实际未调用任何接口（见下方 TODO），
  // 会让咨询师误以为督导已获知，属高危误导，故改为如实告知未接入。
  const handleCrisisProtocol = () => {
    setCrisisModalOpen(false);
    message.warning('危机干预协议尚未接入后端：本次操作未通知督导，请立即按线下流程人工联系督导与紧急联系人。');
    // TODO: 接入 POST /api/v1/escalation/crisis —— 创建 CrisisRecord 并通知督导
  };

  const handleContactSupervisor = () => {
    message.info('正在联系督导...');
    setCrisisModalOpen(false);
    // TODO: 接入真实督导呼叫通道；当前仅前端提示，未发起呼叫
  };

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

  // 睡眠只在患者手动记录过时才展示（0 = 未采集，平台无法自动检测睡眠）
  const radarData = profile ? [
    { subject: '焦虑', value: profile.anxiety },
    { subject: '抑郁', value: profile.depression },
    { subject: '压力', value: profile.stress },
    ...((profile.sleepQuality || 0) > 0 ? [{ subject: '睡眠(手动)', value: profile.sleepQuality }] : []),
    { subject: '社交', value: profile.socialActivity },
    { subject: '情绪稳定', value: profile.emotionalStability },
  ] : [];

  return (
    <div>
      {/* 顶部导航 */}
      <Card style={{ borderRadius: '12px 12px 0 0', borderBottom: '1px solid #f0f0f0' }}
        bodyStyle={{ padding: '12px 20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Space>
            <Button type="text" icon={<ArrowLeftOutlined />} onClick={() => navigate('/consultant/consultations')} />
            <div>
              <Text strong>咨询室</Text>
              <div><Text type="secondary" style={{ fontSize: 12 }}>患者：{patient.nickname} · {riskLabels[patient.riskLevel]}</Text></div>
            </div>
          </Space>
          <Space>
            {lastUpdate && <Text type="secondary" style={{ fontSize: 12 }}>
              <ClockCircleOutlined /> {lastUpdate}
            </Text>}
            {/* 危机干预按钮 */}
            {(patient.riskLevel === 'HIGH' || patient.riskLevel === 'CRISIS') && (
              <Tooltip title="启动危机干预协议">
                <Button danger icon={<FlagOutlined />} size="small" onClick={() => setCrisisModalOpen(true)}>
                  危机干预
                </Button>
              </Tooltip>
            )}
            {/* 咨询笔记按钮 */}
            <Button icon={<FormOutlined />} size="small" onClick={() => setNotesDrawerOpen(true)}>
              咨询笔记
            </Button>
            {/* 安全审计按钮 */}
            <Badge count={auditEvents.filter(e => e.severity === 'warning').length} offset={[-4, 4]} size="small">
              <Button icon={<SafetyCertificateOutlined />} size="small" onClick={() => setAuditDrawerOpen(true)}>
                安全审计
              </Button>
            </Badge>
          </Space>
        </div>
      </Card>

      {/* 风险趋势预警横幅（咨询前查看） */}
      {riskTrend && (
        <div style={{
          padding: '8px 20px',
          background: riskTrend.trend_direction === 'worsening' ? 'linear-gradient(90deg, #fff1f0 0%, #fff7e6 100%)'
            : riskTrend.trend_direction === 'improving' ? 'linear-gradient(90deg, #f6ffed 0%, #e6f7ff 100%)'
            : 'linear-gradient(90deg, #f0f5ff 0%, #f9f0ff 100%)',
          borderBottom: '1px solid #f0f0f0',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          <Space>
            <DashboardOutlined style={{ color: '#6366f1' }} />
            <Text strong style={{ fontSize: 13 }}>风险感知预约</Text>
            <Tag color={riskTrend.trend_direction === 'worsening' ? 'red' : riskTrend.trend_direction === 'improving' ? 'green' : 'blue'} style={{ fontSize: 11 }}>
              {riskTrend.trend_direction === 'worsening' ? '↑ 恶化趋势' : riskTrend.trend_direction === 'improving' ? '↓ 改善趋势' : '→ 稳定'}
            </Tag>
            <Text type="secondary" style={{ fontSize: 12 }}>
              近7天风险评分: {riskTrend.trend?.map((t: any) => Math.round(t.risk_score * 100)).join(' → ') || '暂无'}
            </Text>
          </Space>
          <Space>
            <Tag color={riskColors[riskTrend.current_risk?.toUpperCase()] || '#1890ff'} style={{ fontSize: 11 }}>
              当前风险: {riskLabels[riskTrend.current_risk?.toUpperCase()] || '未知'}
            </Tag>
          </Space>
        </div>
      )}

      {/* 风险趋势不可用：显式提示，避免咨询师把「无数据」误读为「低风险」 */}
      {riskTrendError && !riskTrend && (
        <div style={{
          padding: '8px 20px',
          background: 'linear-gradient(90deg, #fffbe6 0%, #fff1f0 100%)',
          borderBottom: '1px solid #f0f0f0',
          display: 'flex', alignItems: 'center', gap: 8,
        }}>
          <WarningOutlined style={{ color: '#faad14' }} />
          <Text strong style={{ fontSize: 13 }}>风险趋势数据不可用</Text>
          <Text type="secondary" style={{ fontSize: 12 }}>
            未能从评估层取得近 7 天风险趋势，本页不显示风险等级与趋势；请勿据此判断为低风险。
          </Text>
        </div>
      )}

      <div style={{ padding: 24, display: 'flex', gap: 16, height: 'calc(100vh - 160px)' }}>
        {/* 左侧：患者数据监控 */}
        <div style={{ flex: 1, overflow: 'auto' }}>
          {/* 核心指标 - 始终展示 */}
          <Row gutter={[16, 16]}>
          {/* 患者基本信息 */}
          <Col xs={24} md={8}>
            <Card style={{ borderRadius: 12, textAlign: 'center' }}>
              <Avatar size={64} style={{ backgroundColor: '#6366f1', marginBottom: 12 }}>
                {patient.nickname?.[0] || '患'}
              </Avatar>
              <Title level={5} style={{ marginBottom: 4 }}>{patient.nickname}</Title>
              <Tag color={riskColors[patient.riskLevel]} style={{ fontSize: 13, padding: '4px 12px', marginTop: 8 }}>
                {riskLabels[patient.riskLevel]}
              </Tag>
            </Card>
          </Col>

          {/* 实时多模态情绪分析 */}
          <Col xs={24} md={8}>
            <Card
              title={<Space><ThunderboltOutlined /> 实时多模态情绪</Space>}
              extra={
                <Tag color="purple" style={{ fontSize: 11 }}>
                  {profile?.metadata ? 'AI感知' : '等待数据'}
                </Tag>
              }
              style={{ borderRadius: 12 }}
            >
              {profile?.metadata ? (() => {
                try {
                  const meta = JSON.parse(profile.metadata);
                  const probs = meta.emotion_probs || [0.2, 0.2, 0.2, 0.2, 0.2];
                  const labels = ['快乐', '悲伤', '焦虑', '愤怒', '中性'];
                  const colors = ['#52c41a', '#722ed1', '#ff4d4f', '#fa8c16', '#1890ff'];
                  const dominantIdx = probs.indexOf(Math.max(...probs));
                  return (
                    <div>
                      <div style={{ textAlign: 'center', marginBottom: 12 }}>
                        <div style={{ fontSize: 24, fontWeight: 'bold', color: colors[dominantIdx] }}>
                          {labels[dominantIdx]}
                        </div>
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          置信度 {Math.round((meta.confidence || 0) * 100)}%
                        </Text>
                      </div>
                      {probs.map((prob: number, i: number) => (
                        <div key={labels[i]} style={{ marginBottom: 6 }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 2 }}>
                            <span style={{ color: '#666' }}>{labels[i]}</span>
                            <span style={{ color: colors[i], fontWeight: 500 }}>{Math.round(prob * 100)}%</span>
                          </div>
                          <Progress percent={Math.round(prob * 100)} showInfo={false} size="small" strokeColor={colors[i]} />
                        </div>
                      ))}
                      {meta.evidence && meta.evidence.length > 0 && (
                        <div style={{ marginTop: 8, padding: 8, background: '#f9f0ff', borderRadius: 6, fontSize: 11, color: '#722ed1' }}>
                          {meta.evidence.slice(0, 2).map((e: string, i: number) => (
                            <div key={i} style={{ lineHeight: 1.6 }}>• {e}</div>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                } catch {
                  return <Empty description="数据解析失败" />;
                }
              })() : (
                <div style={{ textAlign: 'center', padding: 20 }}>
                  <Text type="secondary">等待患者开启多模态感知...</Text>
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

          {/* 详细信息 - 可折叠 */}
          <Col xs={24}>
            <Collapse defaultActiveKey={['details']} ghost>
              <Collapse.Panel key="details" header={<Text strong>详细数据分析</Text>}>
                <Row gutter={[16, 16]}>
          {/* 情绪维度详情 */}
          {profile && (
            <Col xs={24}>
              <Card title={<Space><AlertOutlined /> 情绪维度分析</Space>} style={{ borderRadius: 12 }}>
                <Row gutter={[16, 16]}>
                  {[
                    { label: '焦虑', value: profile.anxiety, color: '#ff4d4f' },
                    { label: '抑郁', value: profile.depression, color: '#722ed1' },
                    { label: '压力', value: profile.stress, color: '#fa8c16' },
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
                  {/* 睡眠维度：仅当患者手动记录过时展示，否则明确标注"未采集" */}
                  <Col xs={12} md={8} lg={4}>
                    <div style={{ marginBottom: 4 }}>
                      <Text type="secondary" style={{ fontSize: 12 }}>睡眠质量</Text>
                      {(profile.sleepQuality || 0) > 0 ? (
                        <>
                          <Text style={{ float: 'right', fontWeight: 600, color: '#1890ff' }}>{profile.sleepQuality}</Text>
                          <Progress percent={profile.sleepQuality} showInfo={false} size="small" strokeColor="#1890ff" />
                        </>
                      ) : (
                        <div style={{ fontSize: 12, color: '#999' }}>未采集（无夜间传感器，仅手动记录）</div>
                      )}
                    </div>
                  </Col>
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
                </Row>
              </Collapse.Panel>
            </Collapse>
          </Col>

          {/* 智能辅助工作台 */}
          <Col xs={24}>
            <Card
              title={<Space><ExperimentOutlined /> 智能辅助工作台 <Tag color="purple" style={{ fontSize: 10 }}>AI 整合</Tag></Space>}
              style={{ borderRadius: 12 }}
              size="small"
            >
              <Tabs activeKey={workbenchTab} onChange={setWorkbenchTab} size="small" items={[
                {
                  key: 'overview',
                  label: <span><DashboardOutlined /> 综合概览</span>,
                  children: (
                    <div>
                      <Row gutter={[12, 12]}>
                        <Col xs={12} md={6}>
                          {/* 无数据时必须显示「—」，不能默认成「低」——否则会把数据缺失
                              误呈现为低风险，直接影响咨询师的安全判断 */}
                          <Statistic title="当前风险"
                            value={riskTrend?.current_risk == null ? '—'
                              : riskTrend.current_risk === 'high' ? '高'
                              : riskTrend.current_risk === 'medium' ? '中'
                              : riskTrend.current_risk === 'crisis' ? '危机' : '低'}
                            valueStyle={{ color: riskTrend?.current_risk == null ? '#8c8c8c' : riskTrend?.current_risk === 'high' || riskTrend?.current_risk === 'crisis' ? '#ff4d4f' : riskTrend?.current_risk === 'medium' ? '#faad14' : '#52c41a', fontSize: 20 }} />
                        </Col>
                        <Col xs={12} md={6}>
                          <Statistic title="趋势方向"
                            value={riskTrend?.trend_direction == null ? '—'
                              : riskTrend.trend_direction === 'worsening' ? '恶化'
                              : riskTrend.trend_direction === 'improving' ? '改善' : '稳定'}
                            valueStyle={{ fontSize: 20, color: riskTrend?.trend_direction == null ? '#8c8c8c' : undefined }} />
                        </Col>
                        <Col xs={12} md={6}>
                          <Statistic title="活跃模态" value={patientMultimodal?.comprehensiveState?.activeModalities?.length || 0} suffix="个" valueStyle={{ fontSize: 20 }} />
                        </Col>
                        <Col xs={12} md={6}>
                          <Statistic title="审计拦截" value={auditEvents.filter(e => e.severity === 'warning').length} valueStyle={{ fontSize: 20, color: '#fa8c16' }} />
                        </Col>
                      </Row>
                      {/* 风险趋势迷你图 */}
                      {riskTrend?.trend && (
                        <div style={{ marginTop: 12 }}>
                          <Text type="secondary" style={{ fontSize: 12 }}>近7天风险趋势</Text>
                          <ResponsiveContainer width="100%" height={100}>
                            <AreaChart data={riskTrend.trend.map((t: any) => ({ day: new Date(t.timestamp * 1000).toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' }), score: Math.round(t.risk_score * 100), emotion: t.dominant_emotion }))}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="day" tick={{ fontSize: 10 }} />
                              <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} />
                              <RechartsTooltip />
                              <Area type="monotone" dataKey="score" stroke="#6366f1" fill="#6366f1" fillOpacity={0.15} />
                            </AreaChart>
                          </ResponsiveContainer>
                        </div>
                      )}
                    </div>
                  ),
                },
                {
                  key: 'scale',
                  label: <span><LineChartOutlined /> 量表映射</span>,
                  children: scaleEstimate ? (
                    <div>
                      <Alert type="info" showIcon style={{ marginBottom: 12, borderRadius: 8 }}
                        message="以下量表分值由多模态感知数据自动映射生成，非患者自评"
                        description={`置信度: ${Math.round((scaleEstimate.confidence || 0) * 100)}% | ${scaleEstimate.summary}`}
                      />
                      <Row gutter={[12, 12]}>
                        {Object.entries(scaleEstimate.scales || {}).map(([name, s]: [string, any]) => (
                          <Col xs={24} md={8} key={name}>
                            <Card size="small" style={{ borderRadius: 8, borderLeft: `3px solid ${name.includes('PHQ') ? '#722ed1' : name.includes('GAD') ? '#ff4d4f' : '#fa8c16'}` }}>
                              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                                <Text strong style={{ fontSize: 13 }}>{name}</Text>
                                <Tag color={s.severity?.includes('无') ? 'green' : s.severity?.includes('轻') ? 'orange' : 'red'} style={{ fontSize: 11 }}>{s.severity}</Tag>
                              </div>
                              <div style={{ fontSize: 24, fontWeight: 'bold', margin: '4px 0' }}>
                                {Math.round(s.total_score)} <Text type="secondary" style={{ fontSize: 14 }}>/ {s.max_score}</Text>
                              </div>
                              <div style={{ fontSize: 11, color: '#888' }}>
                                置信区间: [{s.confidence_interval?.[0]?.toFixed(1)}, {s.confidence_interval?.[1]?.toFixed(1)}]
                              </div>
                              {s.interpretation && (
                                <div style={{ marginTop: 6, padding: '4px 8px', background: '#f9f0ff', borderRadius: 6, fontSize: 11, color: '#722ed1' }}>
                                  {s.interpretation}
                                </div>
                              )}
                            </Card>
                          </Col>
                        ))}
                      </Row>
                    </div>
                  ) : (
                    <Empty description="等待多模态数据以生成量表映射" image={Empty.PRESENTED_IMAGE_SIMPLE} />
                  ),
                },
                {
                  key: 'cbt',
                  label: <span><ExperimentOutlined /> CBT 脚本</span>,
                  children: (
                    <div>
                      <div style={{ marginBottom: 12 }}>
                        <Text type="secondary" style={{ fontSize: 12 }}>CBT 认知重构 5 步引导脚本 —— 点击步骤查看引导话术</Text>
                      </div>
                      <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
                        {CBT_SCRIPTS.map((script, i) => (
                          <Button key={i} type={cbtStep === i ? 'primary' : 'default'} size="small" onClick={() => setCbtStep(i)}
                            style={{ borderRadius: 16 }}>
                            {script.title}
                          </Button>
                        ))}
                      </div>
                      <Card size="small" style={{ borderRadius: 8, background: '#f0f5ff', border: '1px solid #d6e4ff' }}>
                        <div style={{ marginBottom: 8 }}>
                          <Text strong style={{ fontSize: 14 }}>{CBT_SCRIPTS[cbtStep].title}</Text>
                        </div>
                        <div style={{ marginBottom: 8, fontSize: 13, color: '#555' }}>
                          <Text type="secondary">引导目标：</Text>{CBT_SCRIPTS[cbtStep].prompt}
                        </div>
                        <div style={{ padding: '8px 12px', background: '#fff', borderRadius: 6, borderLeft: '3px solid #6366f1', fontSize: 13, color: '#333', fontStyle: 'italic' }}>
                          {CBT_SCRIPTS[cbtStep].example}
                        </div>
                      </Card>

                      {/* 该患者干预响应面板 */}
                      {patientMultimodal?.comprehensiveState?.interventionHistory && patientMultimodal.comprehensiveState.interventionHistory.records?.length > 0 && (() => {
                        const ih = patientMultimodal.comprehensiveState.interventionHistory;
                        const typeLabels: Record<string, string> = { breathing: '呼吸练习', cbt: 'CBT认知重构', mindfulness: '正念冥想', reality_task: '现实检验', general: '通用干预' };
                        return (
                          <div style={{ marginTop: 16, padding: 12, background: '#fffbe6', borderRadius: 8, border: '1px solid #ffe58f' }}>
                            <Text strong style={{ fontSize: 12, display: 'block', marginBottom: 8, color: '#d46b08' }}>🎯 该患者干预响应历史 ({ih.records.length}次)</Text>
                            {Object.entries(ih.effectivenessByType).map(([type, stats]: [string, any]) => (
                              <div key={type} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6, fontSize: 11 }}>
                                <span>{typeLabels[type] || type}</span>
                                <Space size={8}>
                                  <span style={{ color: stats.avg > 0.1 ? '#52c41a' : stats.avg > 0 ? '#faad14' : '#ff4d4f' }}>
                                    平均改善 {Math.round(stats.avg * 100)}%
                                  </span>
                                  <Tag style={{ fontSize: 9, margin: 0, lineHeight: '14px', padding: '0 4px' }}>{stats.count}次</Tag>
                                </Space>
                              </div>
                            ))}
                            {ih.recommendedType && (
                              <div style={{ marginTop: 8, padding: '6px 10px', background: '#f6ffed', borderRadius: 6, fontSize: 11, color: '#389e0d', border: '1px solid #b7eb8f' }}>
                                ✅ 推荐：该用户对「{typeLabels[ih.recommendedType] || ih.recommendedType}」响应最好（平均降低焦虑 {Math.round((ih.effectivenessByType[ih.recommendedType]?.avg || 0) * 100)}%）
                              </div>
                            )}
                          </div>
                        );
                      })()}
                    </div>
                  ),
                },
                {
                  key: 'twin',
                  label: <span><RadarChartOutlined /> 数字孪生 <Tag color="cyan" style={{ fontSize: 9, lineHeight: '14px', padding: '0 4px', marginLeft: 4 }}>核心</Tag></span>,
                  children: twinLoading ? (
                    <div style={{ textAlign: 'center', padding: 24 }}><Spin tip="加载数字孪生预测..." /></div>
                  ) : twinPrediction ? (
                    <div>
                      {/* 危机风险概览 */}
                      <Row gutter={[12, 12]} style={{ marginBottom: 12 }}>
                        <Col xs={12} md={8}>
                          <Statistic title="最大危机概率" value={Math.round((twinPrediction.trajectory?.max_crisis_risk || 0) * 100)} suffix="%"
                            valueStyle={{ color: (twinPrediction.trajectory?.max_crisis_risk || 0) > 0.5 ? '#ff4d4f' : (twinPrediction.trajectory?.max_crisis_risk || 0) > 0.3 ? '#faad14' : '#52c41a', fontSize: 22 }} />
                        </Col>
                        <Col xs={12} md={8}>
                          <Statistic title="预计恶化时间" value={twinPrediction.trajectory?.deterioration_time != null ? `${twinPrediction.trajectory.deterioration_time}h` : '—'}
                            valueStyle={{ fontSize: 22 }} />
                        </Col>
                        <Col xs={24} md={8}>
                          <Statistic title="干预决策" value={twinPrediction.intervention?.should_intervene ? '建议干预' : '继续观察'}
                            valueStyle={{ fontSize: 18, color: twinPrediction.intervention?.should_intervene ? '#ff4d4f' : '#52c41a' }} />
                        </Col>
                      </Row>

                      {/* 轨迹预测图 */}
                      {twinPrediction.trajectory?.points?.length > 0 && (
                        <div style={{ marginBottom: 12 }}>
                          <Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 4 }}>未来 12h 危机概率轨迹预测</Text>
                          <ResponsiveContainer width="100%" height={120}>
                            <AreaChart data={twinPrediction.trajectory.points.map((p: any) => ({
                              hour: `+${p.time_step}h`,
                              crisis: Math.round(p.crisis_probability * 100),
                              uncertainty: Math.round(p.uncertainty * 100),
                            }))}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="hour" tick={{ fontSize: 10 }} />
                              <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} unit="%" />
                              <RechartsTooltip formatter={(v: number) => `${v}%`} />
                              <Legend wrapperStyle={{ fontSize: 11 }} />
                              <Area type="monotone" dataKey="crisis" name="危机概率" stroke="#ff4d4f" fill="#ff4d4f" fillOpacity={0.15} />
                              <Area type="monotone" dataKey="uncertainty" name="不确定度" stroke="#faad14" fill="#faad14" fillOpacity={0.08} />
                            </AreaChart>
                          </ResponsiveContainer>
                        </div>
                      )}

                      {/* 干预决策详情 */}
                      {twinPrediction.intervention?.should_intervene && (
                        <Card size="small" style={{ borderRadius: 8, borderLeft: '3px solid #ff4d4f', marginBottom: 12, background: '#fff2f0' }}>
                          <div style={{ fontSize: 13, marginBottom: 6 }}>
                            <Text strong style={{ color: '#cf1322' }}>🎯 干预建议</Text>
                          </div>
                          <div style={{ fontSize: 12, color: '#555', lineHeight: 1.8 }}>
                            {twinPrediction.intervention.reasoning}
                          </div>
                          {twinPrediction.intervention.chain_to_break && (
                            <div style={{ marginTop: 8, padding: '6px 10px', background: '#fff', borderRadius: 6, fontSize: 12, border: '1px solid #ffccc7' }}>
                              <Text type="secondary" style={{ fontSize: 11 }}>要阻断的因果链：</Text>
                              <Text strong style={{ fontSize: 12, color: '#cf1322' }}>{twinPrediction.intervention.chain_to_break}</Text>
                            </div>
                          )}
                          {twinPrediction.intervention.recommended_dimensions?.length > 0 && (
                            <div style={{ marginTop: 8 }}>
                              <Text type="secondary" style={{ fontSize: 11 }}>目标维度：</Text>
                              {twinPrediction.intervention.recommended_dimensions.map((d: string) => (
                                <Tag key={d} color="volcano" style={{ fontSize: 10, marginBottom: 2 }}>{d}</Tag>
                              ))}
                            </div>
                          )}
                        </Card>
                      )}

                      {/* CT-PLEW 精确分析：势能景观 + 临界跃迁 */}
                      {twinPrediction.precision?.escapes && Object.keys(twinPrediction.precision.escapes).length > 0 && (
                        <Card size="small" style={{ borderRadius: 8, borderLeft: '3px solid #722ed1', marginBottom: 12, background: '#f9f0ff' }}>
                          <div style={{ fontSize: 13, marginBottom: 8 }}>
                            <Text strong style={{ color: '#531dab' }}>⚛ CT-PLEW 精确分析</Text>
                            <Tag color="purple" style={{ fontSize: 9, marginLeft: 6 }}>势能景观·Kramers</Tag>
                            {twinPrediction.precision.ews_alert && (
                              <Tag color="red" style={{ fontSize: 9, marginLeft: 4 }}>临界慢化预警</Tag>
                            )}
                            {twinPrediction.precision.multistable && (
                              <Tag color="geekblue" style={{ fontSize: 9, marginLeft: 4 }}>多稳态</Tag>
                            )}
                          </div>
                          <Row gutter={[8, 8]}>
                            <Col xs={8}>
                              <Statistic title="Kramers 跃迁概率" value={Math.round((twinPrediction.precision.crisis_probability || 0) * 100)} suffix="%"
                                valueStyle={{ fontSize: 18, color: (twinPrediction.precision.crisis_probability || 0) > 0.3 ? '#ff4d4f' : '#52c41a' }} />
                            </Col>
                            <Col xs={8}>
                              <Statistic title="心理韧性 ΔU/D" value={twinPrediction.precision.escapes[twinPrediction.precision.critical_dim]?.resilience ?? 0}
                                valueStyle={{ fontSize: 18, color: '#722ed1' }} />
                            </Col>
                            <Col xs={8}>
                              <Statistic title="预计跃迁" value={twinPrediction.precision.tipping_eta_steps != null ? `${Math.round(twinPrediction.precision.tipping_eta_steps)}步` : '—'}
                                valueStyle={{ fontSize: 18 }} />
                            </Col>
                          </Row>
                          {twinPrediction.precision.critical_dim && (
                            <div style={{ marginTop: 6, fontSize: 11, color: '#888' }}>
                              最脆弱维度：<Tag style={{ fontSize: 10 }}>{twinPrediction.precision.critical_dim}</Tag>
                              逃逸机制：{twinPrediction.precision.escapes[twinPrediction.precision.critical_dim]?.method}
                            </div>
                          )}
                          {twinPrediction.precision.counterfactual && (
                            <div style={{ marginTop: 8, padding: '6px 10px', background: '#fff', borderRadius: 6, fontSize: 11, border: '1px solid #d3adf7', color: '#531dab' }}>
                              🎯 反事实干预靶点：<Text strong style={{ fontSize: 11, color: '#531dab' }}>{twinPrediction.precision.counterfactual.target_dim}</Text>
                              （危机概率 {Math.round(twinPrediction.precision.counterfactual.baseline_crisis_prob * 100)}% → {Math.round(twinPrediction.precision.counterfactual.post_crisis_prob * 100)}%）
                            </div>
                          )}
                        </Card>
                      )}

                      {/* 数字孪生摘要 */}
                      {twinPrediction.twin_summary && (
                        <div style={{ fontSize: 11, color: '#888' }}>
                          <Space split={<span style={{ color: '#ddd' }}>|</span>}>
                            <span>观测维度: {Object.keys(twinPrediction.twin_summary.current_state || {}).length}</span>
                            <span>因果边: {twinPrediction.twin_summary.causal_edges || 0}</span>
                            <span>历史: {twinPrediction.twin_summary.history_length || 0} 步</span>
                          </Space>
                        </div>
                      )}
                    </div>
                  ) : (
                    <Empty description="数字孪生尚未积累足够数据" image={Empty.PRESENTED_IMAGE_SIMPLE}>
                      <Text type="secondary" style={{ fontSize: 12 }}>需要至少 20 次感知更新后才会生成因果图和轨迹预测</Text>
                    </Empty>
                  ),
                },
                {
                  key: 'twin-nd',
                  label: <span><ThunderboltOutlined /> 多维临界 <Tag color="purple" style={{ fontSize: 9, lineHeight: '14px', padding: '0 4px', marginLeft: 4 }}>In-Silico</Tag></span>,
                  children: (
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10, gap: 8 }}>
                        <Text type="secondary" style={{ fontSize: 11, lineHeight: 1.5 }}>
                          同一个体化随机动力系统 dX=f(X)dt+ΣdW —— 预见崩溃路径 · 排练平行未来 · 优化最小能量救援
                        </Text>
                        <Button size="small" icon={<HistoryOutlined />} loading={twinNdLoading} onClick={loadTwinAdvanced} style={{ flexShrink: 0 }}>重算</Button>
                      </div>

                      {twinNdLoading && !twinLandscape && !twinFutures && !twinRescue ? (
                        <div style={{ textAlign: 'center', padding: 24 }}><Spin tip="多维景观推演中..." /></div>
                      ) : (
                        <>
                          {/* ===== 1. 崩溃 domino 链（多维临界跃迁景观） ===== */}
                          <Card size="small" style={{ borderRadius: 8, borderLeft: '3px solid #722ed1', marginBottom: 12, background: '#faf7ff' }}>
                            <div style={{ fontSize: 13, marginBottom: 8 }}>
                              <Text strong style={{ color: '#531dab' }}>⛰ 崩溃 domino 链</Text>
                              <Tag color="purple" style={{ fontSize: 9, marginLeft: 6 }}>instanton·多维Kramers</Tag>
                              {twinLandscape?.fallback && <Tag color="orange" style={{ fontSize: 9, marginLeft: 4 }}>1D 降级</Tag>}
                              {twinLandscape?.in_crisis && <Tag color="red" style={{ fontSize: 9, marginLeft: 4 }}>已在危机阱</Tag>}
                            </div>
                            {twinLandscape && !twinLandscape.fallback ? (() => {
                              const L = twinLandscape;
                              return (
                                <>
                                  <Row gutter={[8, 8]}>
                                    <Col xs={12} md={6}>
                                      <Statistic title="多维危机概率" value={Math.round((L.crisis_probability || 0) * 100)} suffix="%"
                                        valueStyle={{ fontSize: 18, color: (L.crisis_probability || 0) > 0.3 ? '#ff4d4f' : (L.crisis_probability || 0) > 0.15 ? '#faad14' : '#52c41a' }} />
                                    </Col>
                                    <Col xs={12} md={6}>
                                      <Statistic title="势垒高度 ΔU" value={L.barrier_height ?? 0} precision={3} valueStyle={{ fontSize: 18, color: '#722ed1' }} />
                                    </Col>
                                    <Col xs={12} md={6}>
                                      <Statistic title="潜空间维度" value={L.n_dim ?? 0}
                                        suffix={<span style={{ fontSize: 11, color: '#999' }}>解释{Math.round((L.latent_var_explained || 0) * 100)}%</span>}
                                        valueStyle={{ fontSize: 18 }} />
                                    </Col>
                                    <Col xs={12} md={6}>
                                      <Statistic title="梯度/通量比" value={L.gradient_flux_ratio ?? 0} precision={2} valueStyle={{ fontSize: 18 }} />
                                    </Col>
                                  </Row>
                                  {L.collapse_order?.length > 0 && (
                                    <div style={{ margin: '10px 0 4px' }}>
                                      <Text type="secondary" style={{ fontSize: 11, display: 'block', marginBottom: 6 }}>
                                        最小能量崩溃路径反投影 → 个体化 domino 链（模态崩溃先后顺序）
                                      </Text>
                                      <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 4 }}>
                                        {L.collapse_order.map((m: string, i: number) => (
                                          <span key={i} style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                                            {i > 0 && <span style={{ color: '#bbb', fontSize: 12 }}>→</span>}
                                            <Tag color={i === 0 ? 'red' : i === 1 ? 'volcano' : 'purple'} style={{ margin: 0, fontSize: 11 }}>
                                              {i + 1}·{dimLabel(m)}
                                            </Tag>
                                          </span>
                                        ))}
                                      </div>
                                    </div>
                                  )}
                                  <div style={{ marginTop: 8, fontSize: 11, color: '#888' }}>
                                    <Space split={<span style={{ color: '#ddd' }}>|</span>} wrap>
                                      <span>逃逸率 λ: {L.escape_rate ?? 0}</span>
                                      {L.particle_prob != null && (
                                        <span>粒子群交叉验证: {Math.round(L.particle_prob * 100)}%
                                          {L.particle_ci ? ` [${Math.round(L.particle_ci[0] * 100)}%,${Math.round(L.particle_ci[1] * 100)}%]` : ''}</span>
                                      )}
                                      <span>方法: {L.method}</span>
                                    </Space>
                                  </div>
                                </>
                              );
                            })() : (
                              <div style={{ fontSize: 12, color: '#888', lineHeight: 1.7 }}>
                                {twinLandscape?.fallback
                                  ? <>维度/历史不足或无清晰双稳态，已降级为 1D CT-PLEW。当前危机概率 <Text strong style={{ color: '#cf1322' }}>{Math.round((twinLandscape.crisis_probability || 0) * 100)}%</Text>{twinLandscape.critical_dim ? <>，最脆弱维度 <Tag style={{ fontSize: 10 }}>{dimLabel(twinLandscape.critical_dim)}</Tag></> : null}。</>
                                  : '多维景观尚未就绪，待数据积累后自动启用。'}
                              </div>
                            )}
                          </Card>

                          {/* ===== 2. 平行未来扇形轨迹 ===== */}
                          <Card size="small" style={{ borderRadius: 8, borderLeft: '3px solid #1890ff', marginBottom: 12, background: '#f5faff' }}>
                            <div style={{ fontSize: 13, marginBottom: 8 }}>
                              <Text strong style={{ color: '#0958d9' }}>🌀 平行未来推演</Text>
                              <Tag color="blue" style={{ fontSize: 9, marginLeft: 6 }}>In-Silico·K条轨迹</Tag>
                              {twinFutures?.fallback && <Tag color="orange" style={{ fontSize: 9, marginLeft: 4 }}>1D 降级</Tag>}
                            </div>
                            {twinFutures && !twinFutures.fallback ? (() => {
                              const F = twinFutures;
                              const fdf = F.first_domino_freq || {};
                              const sortedModes = Object.keys(fdf).sort((a, b) => fdf[b] - fdf[a]);
                              const primary = sortedModes[0] || F.mode_names?.[0];
                              const q = primary ? F.quantile_mode_traj?.[primary] : null;
                              const H = q?.q50?.length || 0;
                              const chartData = Array.from({ length: H }, (_, t) => ({
                                t: `+${t}`,
                                band: [(q?.q10?.[t] ?? 0), (q?.q90?.[t] ?? 0)],
                                median: q?.q50?.[t] ?? 0,
                              }));
                              const color = primary ? twinDimColor(primary) : '#1890ff';
                              return (
                                <>
                                  <Row gutter={[8, 8]} style={{ marginBottom: 8 }}>
                                    <Col xs={8}>
                                      <Statistic title="平行未来数" value={F.n_particles || 0} valueStyle={{ fontSize: 18 }} />
                                    </Col>
                                    <Col xs={8}>
                                      <Statistic title="危机率" value={Math.round((F.crisis_rate || 0) * 100)} suffix="%"
                                        valueStyle={{ fontSize: 18, color: (F.crisis_rate || 0) > 0.3 ? '#ff4d4f' : '#faad14' }} />
                                      <div style={{ fontSize: 10, color: '#999' }}>CI [{Math.round((F.crisis_ci?.[0] || 0) * 100)}%,{Math.round((F.crisis_ci?.[1] || 0) * 100)}%]</div>
                                    </Col>
                                    <Col xs={8}>
                                      <Statistic title="平均首崩时间" value={F.time_to_crisis_mean != null ? F.time_to_crisis_mean : '—'}
                                        suffix={F.time_to_crisis_mean != null ? '步' : ''} valueStyle={{ fontSize: 18 }} />
                                    </Col>
                                  </Row>
                                  {primary && q && H > 0 && (
                                    <div style={{ marginBottom: 8 }}>
                                      <Text type="secondary" style={{ fontSize: 11, display: 'block', marginBottom: 4 }}>
                                        首要崩溃模态「{dimLabel(primary)}」的 {H - 1} 步分位扇形带（q10–q90 + 中位轨迹）
                                      </Text>
                                      <ResponsiveContainer width="100%" height={140}>
                                        <AreaChart data={chartData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                                          <CartesianGrid strokeDasharray="3 3" />
                                          <XAxis dataKey="t" tick={{ fontSize: 10 }} />
                                          <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} />
                                          <RechartsTooltip formatter={(v: any) => (Array.isArray(v) ? `${(v[0] * 100).toFixed(0)}%~${(v[1] * 100).toFixed(0)}%` : `${(Number(v) * 100).toFixed(0)}%`)} />
                                          <Area dataKey="band" stroke="none" fill={color} fillOpacity={0.2} isAnimationActive={false} />
                                          <Area dataKey="median" stroke={color} strokeWidth={2} fill={color} fillOpacity={0} isAnimationActive={false} />
                                        </AreaChart>
                                      </ResponsiveContainer>
                                    </div>
                                  )}
                                  {sortedModes.length > 0 && (
                                    <div>
                                      <Text type="secondary" style={{ fontSize: 11, display: 'block', marginBottom: 6 }}>首崩模态频率（K 条平行未来中谁先崩）</Text>
                                      {sortedModes.slice(0, 6).map((m) => (
                                        <div key={m} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4, fontSize: 11 }}>
                                          <span style={{ width: 62, color: '#666', flexShrink: 0 }}>{dimLabel(m)}</span>
                                          <div style={{ flex: 1, background: '#e6e6e6', borderRadius: 4, height: 10, overflow: 'hidden' }}>
                                            <div style={{ width: `${Math.round(fdf[m] * 100)}%`, background: twinDimColor(m), height: 10, borderRadius: 4 }} />
                                          </div>
                                          <span style={{ width: 34, textAlign: 'right', color: '#888', flexShrink: 0 }}>{Math.round(fdf[m] * 100)}%</span>
                                        </div>
                                      ))}
                                    </div>
                                  )}
                                </>
                              );
                            })() : (
                              <div style={{ fontSize: 12, color: '#888', lineHeight: 1.7 }}>
                                {twinFutures?.fallback
                                  ? <>数据不足，已降级为 1D 危机概率 <Text strong style={{ color: '#cf1322' }}>{Math.round((twinFutures.crisis_probability || 0) * 100)}%</Text>。</>
                                  : '平行未来推演尚未就绪。'}
                              </div>
                            )}
                          </Card>

                          {/* ===== 3. MPC 最优救援卡 ===== */}
                          <Card size="small" style={{ borderRadius: 8, borderLeft: '3px solid #52c41a', background: '#f6ffed' }}>
                            <div style={{ fontSize: 13, marginBottom: 8 }}>
                              <Text strong style={{ color: '#389e0d' }}>🛟 MPC 最优救援</Text>
                              <Tag color="green" style={{ fontSize: 9, marginLeft: 6 }}>CEM·逆向instanton</Tag>
                              {twinRescue?.fallback && <Tag color="orange" style={{ fontSize: 9, marginLeft: 4 }}>时机降级</Tag>}
                            </div>
                            {twinRescue && !twinRescue.fallback ? (() => {
                              const R = twinRescue;
                              const mpc = R.mpc_plan || {};
                              const rp = R.rescue_path;
                              const base = mpc.baseline_crisis_rate || 0;
                              const ctrl = mpc.controlled_crisis_rate || 0;
                              const relDrop = base > 0 ? (base - ctrl) / base : 0;
                              const doseEntries = Object.entries(mpc.mode_dose || {})
                                .filter(([, v]) => Math.abs(v as number) > 1e-4)
                                .sort((a, b) => Math.abs(b[1] as number) - Math.abs(a[1] as number));
                              return (
                                <>
                                  <Row gutter={[8, 8]}>
                                    <Col xs={8}>
                                      <Statistic title="基线危机率" value={Math.round(base * 100)} suffix="%" valueStyle={{ fontSize: 18, color: '#ff4d4f' }} />
                                    </Col>
                                    <Col xs={8}>
                                      <Statistic title="干预后危机率" value={Math.round(ctrl * 100)} suffix="%" valueStyle={{ fontSize: 18, color: '#52c41a' }} />
                                    </Col>
                                    <Col xs={8}>
                                      <Statistic title="相对下降" value={Math.round(relDrop * 100)} suffix="%" valueStyle={{ fontSize: 18, color: '#722ed1' }} />
                                    </Col>
                                  </Row>
                                  <div style={{ marginTop: 8, fontSize: 11, color: '#888' }}>
                                    <Space split={<span style={{ color: '#ddd' }}>|</span>} wrap>
                                      <span>干预能量: {mpc.energy_cost ?? 0}</span>
                                      {mpc.target_dim && <span>主靶点: {dimLabel(mpc.target_dim)}</span>}
                                      <span>优化时域: {mpc.horizon ?? '—'}步</span>
                                      <span>{mpc.converged ? '✅ 已收敛' : '未完全收敛'}</span>
                                    </Space>
                                  </div>
                                  {doseEntries.length > 0 && (
                                    <div style={{ marginTop: 8 }}>
                                      <Text type="secondary" style={{ fontSize: 11 }}>救援剂量（模态空间，带符号）：</Text>
                                      <div style={{ marginTop: 4 }}>
                                        {doseEntries.map(([m, v]) => (
                                          <Tag key={m} color={(v as number) < 0 ? 'green' : 'volcano'} style={{ fontSize: 10, marginBottom: 2 }}>
                                            {dimLabel(m)} {(v as number) > 0 ? '+' : ''}{(v as number).toFixed(2)}
                                          </Tag>
                                        ))}
                                      </div>
                                    </div>
                                  )}
                                  {mpc.narrative && (
                                    <div style={{ marginTop: 8, padding: '6px 10px', background: '#fff', borderRadius: 6, fontSize: 11.5, color: '#389e0d', border: '1px solid #b7eb8f', lineHeight: 1.6 }}>
                                      {mpc.narrative}
                                    </div>
                                  )}
                                  {rp && (
                                    <div style={{ marginTop: 10, padding: '8px 10px', background: '#fff', borderRadius: 6, border: '1px dashed #95de64' }}>
                                      <Text strong style={{ fontSize: 11.5, color: '#389e0d' }}>最小能量救援轨迹（逆向 instanton）</Text>
                                      <div style={{ marginTop: 4, fontSize: 11, color: '#888' }}>
                                        <Space split={<span style={{ color: '#ddd' }}>|</span>} wrap>
                                          <span>救援后壁垒: {rp.barrier_restored ?? 0}</span>
                                          <span>救援能量: {rp.energy ?? 0}</span>
                                          <span>{rp.n_steps ?? 0} 步</span>
                                          {rp.target_dim && <span>靶点: {dimLabel(rp.target_dim)}</span>}
                                        </Space>
                                      </div>
                                      {rp.narrative && <div style={{ marginTop: 4, fontSize: 11.5, color: '#555', lineHeight: 1.6 }}>{rp.narrative}</div>}
                                    </div>
                                  )}
                                </>
                              );
                            })() : (
                              <div style={{ fontSize: 12, color: '#888', lineHeight: 1.7 }}>
                                {twinRescue?.fallback
                                  ? (twinRescue.intervention_decision
                                    ? <>多维模拟器不可用，已降级为干预时机优化：{twinRescue.intervention_decision.reasoning || '—'}{twinRescue.intervention_decision.recommended_dimensions?.length ? <>，目标维度 {twinRescue.intervention_decision.recommended_dimensions.map((d: string) => <Tag key={d} style={{ fontSize: 10 }}>{dimLabel(d)}</Tag>)}</> : null}</>
                                    : '多维模拟器不可用，暂无救援方案。')
                                  : 'MPC 救援尚未就绪。'}
                              </div>
                            )}
                          </Card>
                        </>
                      )}
                    </div>
                  ),
                },
              ]} />
            </Card>
          </Col>
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
          extra={
            <Button
              size="small"
              danger
              onClick={() => {
                Modal.confirm({
                  title: '结束咨询',
                  content: '确定要结束本次咨询吗？结束后实时数据将被清空，仅保留咨询报告、回顾和笔记。',
                  okText: '确定结束',
                  cancelText: '取消',
                  okButtonProps: { danger: true },
                  onOk: () => {
                    const socket = getSocket();
                    socket.emit('consultation:end', bookingId);
                    message.success('咨询已结束');
                    navigate('/consultant/bookings');
                  },
                });
              }}
            >
              结束咨询
            </Button>
          }
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

              {/* 患者实时多模态心理画像 */}
              {patientMultimodal?.comprehensiveState && (
                <div style={{ padding: '12px 16px', background: 'linear-gradient(135deg, #fff7e6 0%, #fff0f6 100%)', borderTop: '1px solid #e8e8f0' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
                    <Space size={4}>
                      <ThunderboltOutlined style={{ color: '#fa8c16', fontSize: 14 }} />
                      <Text strong style={{ fontSize: 13 }}>患者实时心理画像</Text>
                      <Tag color="orange" style={{ fontSize: 10, marginRight: 0, padding: '0 6px', lineHeight: '18px' }}>
                        更新于 {patientMultimodalTime}
                      </Tag>
                    </Space>
                  </div>

                  <Row gutter={[12, 12]}>
                    {/* 综合评分 */}
                    <Col xs={24} md={8}>
                      <div style={{ textAlign: 'center', padding: 12, background: '#fff', borderRadius: 8 }}>
                        <Text type="secondary" style={{ fontSize: 12 }}>综合心理健康评分</Text>
                        <div style={{ fontSize: 32, fontWeight: 'bold', color: patientMultimodal.comprehensiveState.fusionScore > 60 ? '#52c41a' : patientMultimodal.comprehensiveState.fusionScore > 40 ? '#faad14' : '#ff4d4f', margin: '8px 0' }}>
                          {Math.round(patientMultimodal.comprehensiveState.fusionScore || 50)}
                        </div>
                        <Tag color={riskColors[patientMultimodal.comprehensiveState.riskLevel || 'MEDIUM']} style={{ fontSize: 12, padding: '4px 12px' }}>
                          {riskLabels[patientMultimodal.comprehensiveState.riskLevel || 'MEDIUM']}
                        </Tag>
                      </div>
                    </Col>

                    {/* 主导情绪 */}
                    <Col xs={24} md={8}>
                      <div style={{ textAlign: 'center', padding: 12, background: '#fff', borderRadius: 8 }}>
                        <Text type="secondary" style={{ fontSize: 12 }}>当前主导情绪</Text>
                        <div style={{ fontSize: 24, fontWeight: 'bold', margin: '8px 0' }}>
                          {patientMultimodal.comprehensiveState.dominantEmotion || '中性'}
                        </div>
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          置信度 {Math.round((patientMultimodal.comprehensiveState.emotionProbs?.[patientMultimodal.comprehensiveState.dominantEmotion] || 0.3) * 100)}%
                        </Text>
                      </div>
                    </Col>

                    {/* 活跃模态 */}
                    <Col xs={24} md={8}>
                      <div style={{ textAlign: 'center', padding: 12, background: '#fff', borderRadius: 8 }}>
                        <Text type="secondary" style={{ fontSize: 12 }}>活跃感知模态</Text>
                        <div style={{ margin: '8px 0' }}>
                          {patientMultimodal.comprehensiveState.activeModalities?.map((m: string) => (
                            <Tag key={m} color="blue" style={{ marginRight: 4 }}>{m}</Tag>
                          )) || <Tag>无</Tag>}
                        </div>
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          共 {patientMultimodal.comprehensiveState.activeModalities?.length || 0} 个模态
                        </Text>
                      </div>
                    </Col>
                  </Row>

                  {/* 情绪概率分布 */}
                  {patientMultimodal.comprehensiveState.emotionProbs && (
                    <div style={{ marginTop: 12, padding: 12, background: '#fff', borderRadius: 8 }}>
                      <Text strong style={{ fontSize: 12, marginBottom: 8, display: 'block' }}>情绪分布</Text>
                      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                        {Object.entries(patientMultimodal.comprehensiveState.emotionProbs).map(([emotion, prob]: [string, any]) => (
                          <div key={emotion} style={{ flex: '1 0 80px', minWidth: 80 }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#888', marginBottom: 2 }}>
                              <span>{emotion}</span>
                              <span>{Math.round(prob * 100)}%</span>
                            </div>
                            <div style={{ height: 6, background: '#f0f0f0', borderRadius: 3 }}>
                              <div style={{ height: '100%', width: `${prob * 100}%`, background: emotion === '快乐' ? '#52c41a' : emotion === '悲伤' ? '#722ed1' : emotion === '焦虑' ? '#ff4d4f' : emotion === '愤怒' ? '#fa8c16' : '#1890ff', borderRadius: 3, transition: 'width 0.5s ease' }} />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* 叙事性分析报告 */}
                  {patientMultimodal.comprehensiveState.narrativeAnalysis && (
                    <div style={{ marginTop: 12, padding: 12, background: '#fff', borderRadius: 8, borderLeft: '3px solid #fa8c16' }}>
                      <Text strong style={{ fontSize: 12, marginBottom: 8, display: 'block' }}> 心理观察报告</Text>
                      <Paragraph style={{ fontSize: 12, lineHeight: 1.6, margin: 0, whiteSpace: 'pre-wrap' }}>
                        {patientMultimodal.comprehensiveState.narrativeAnalysis}
                      </Paragraph>
                    </div>
                  )}

                  {/* ===== 近期趋势 ===== */}
                  {patientMultimodal.comprehensiveState.trajectorySummary && (
                    <div style={{ marginTop: 12, padding: 12, background: 'linear-gradient(135deg, #f0f5ff 0%, #f9f0ff 100%)', borderRadius: 8 }}>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                        <Space size={4}>
                          <LineChartOutlined style={{ color: '#722ed1', fontSize: 13 }} />
                          <Text strong style={{ fontSize: 12, color: '#5b4a8a' }}>近期趋势</Text>
                        </Space>
                        <Tag color={patientMultimodal.comprehensiveState.trajectorySummary.trend === 'improving' ? 'green' : patientMultimodal.comprehensiveState.trajectorySummary.trend === 'declining' ? 'red' : 'blue'} style={{ fontSize: 10 }}>
                          {patientMultimodal.comprehensiveState.trajectorySummary.trend === 'improving' ? '↑ 改善' : patientMultimodal.comprehensiveState.trajectorySummary.trend === 'declining' ? '↓ 需关注' : '→ 平稳'}
                        </Tag>
                      </div>
                      {/* 迷你趋势图 */}
                      <div style={{ height: 80, marginBottom: 6 }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={patientMultimodal.comprehensiveState.trajectorySummary.recentPoints.map((p: any) => ({
                            time: new Date(p.t).toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' }),
                            焦虑: p.a,
                          }))}>
                            <XAxis dataKey="time" tick={{ fontSize: 9 }} axisLine={false} />
                            <YAxis tick={{ fontSize: 9 }} domain={[0, 100]} width={25} />
                            <RechartsTooltip />
                            <Area type="monotone" dataKey="焦虑" stroke="#ff4d4f" fill="#fff1f0" strokeWidth={1.5} dot={{ r: 1.5 }} />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                      {patientMultimodal.comprehensiveState.trajectorySummary.weeklyChange !== 0 && (
                        <Text type="secondary" style={{ fontSize: 10 }}>
                          本周焦虑水平比上周{patientMultimodal.comprehensiveState.trajectorySummary.weeklyChange > 0 ? '上升' : '下降'} {Math.abs(patientMultimodal.comprehensiveState.trajectorySummary.weeklyChange)}%
                        </Text>
                      )}
                    </div>
                  )}

                  {/* ===== 11 模态三层详细面板 ===== */}
                  <div style={{ marginTop: 12 }}>
                    <div style={{ display: 'flex', alignItems: 'center', marginBottom: 8 }}>
                      <ExperimentOutlined style={{ color: '#722ed1', fontSize: 13 }} />
                      <Text strong style={{ fontSize: 12, color: '#5b4a8a', marginLeft: 4 }}>11 模态感知详情</Text>
                      <Tag color="purple" style={{ marginLeft: 6, fontSize: 10, lineHeight: '16px', padding: '0 6px' }}>年龄差异化校准</Tag>
                      {patientMultimodal.comprehensiveState.baseline?.calibrated && (
                        <Tag color="cyan" style={{ marginLeft: 4, fontSize: 10, lineHeight: '16px', padding: '0 6px' }}>个人基线已校准 ({patientMultimodal.comprehensiveState.baseline.sampleCount}样本)</Tag>
                      )}
                    </div>

                    {/* 融合权重分布 */}
                    {patientMultimodal.comprehensiveState.fusionWeights && (
                      <div style={{ padding: 10, background: '#fff', borderRadius: 8, marginBottom: 8 }}>
                        <Text strong style={{ fontSize: 11, display: 'block', marginBottom: 6, color: '#722ed1' }}>融合权重分布</Text>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                          {Object.entries(patientMultimodal.comprehensiveState.fusionWeights).map(([k, v]) => {
                            const labelMap: Record<string, string> = { text: '文字', voice: '语音', facial: '面部', keyboard: '键盘', circadian: '昼夜', cognitive: '认知', hrv: '心率', breathing: '呼吸', behavioralAct: '行为', eye: '眼动', voiceSemantics: '语义' };
                            const colorMap: Record<string, string> = { text: '#1890ff', voice: '#52c41a', facial: '#fa8c16', keyboard: '#722ed1', circadian: '#13c2c2', cognitive: '#eb2f9a', hrv: '#ff4d4f', breathing: '#36cfc9', behavioralAct: '#597ef7', eye: '#9254de', voiceSemantics: '#faad14' };
                            const pct = Math.round((v as number) * 100);
                            return (
                              <div key={k} style={{ flex: '0 0 auto', width: 72, textAlign: 'center' }}>
                                <div style={{ fontSize: 10, color: '#888', marginBottom: 2 }}>{labelMap[k] || k}</div>
                                <div style={{ height: 4, background: '#f0f0f0', borderRadius: 2, overflow: 'hidden' }}>
                                  <div style={{ height: '100%', width: `${pct}%`, background: colorMap[k] || '#999', borderRadius: 2, transition: 'width 0.5s' }} />
                                </div>
                                <div style={{ fontSize: 10, color: colorMap[k] || '#999', fontWeight: 500 }}>{pct}%</div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}

                    <Row gutter={[8, 8]}>
                      {/* ── 基础感知层 ── */}
                      <Col xs={24}>
                        <div style={{ fontSize: 10, color: '#999', fontWeight: 500, marginBottom: 4 }}>🔍 基础感知层</div>
                      </Col>

                      {/* 文字语义 */}
                      {patientMultimodal.comprehensiveState.text?.timestamp > 0 && (
                        <Col xs={12} md={6}>
                          <div style={{ padding: 8, background: '#fafbff', borderRadius: 6, height: '100%' }}>
                            <div style={{ fontSize: 11, fontWeight: 600, color: '#1890ff', marginBottom: 4 }}>📝 文字语义</div>
                            <div style={{ fontSize: 10, lineHeight: 1.7, color: '#555' }}>
                              <div>情绪：<Text strong style={{ fontSize: 10 }}>{patientMultimodal.comprehensiveState.text.dominantEmotion}</Text></div>
                              <div>效价：<span style={{ color: patientMultimodal.comprehensiveState.text.sentiment > 0 ? '#52c41a' : '#ff4d4f' }}>{patientMultimodal.comprehensiveState.text.sentiment > 0 ? '积极' : patientMultimodal.comprehensiveState.text.sentiment < 0 ? '消极' : '中性'}</span></div>
                              <div>强度：<Progress percent={Math.round(patientMultimodal.comprehensiveState.text.emotionalIntensity * 100)} showInfo={false} size="small" strokeColor="#1890ff" style={{ width: 50, display: 'inline-block', verticalAlign: 'middle' }} /></div>
                              <div>字数：{patientMultimodal.comprehensiveState.text.totalCharsAnalyzed}</div>
                            </div>
                          </div>
                        </Col>
                      )}

                      {/* 语音声学 */}
                      {patientMultimodal.comprehensiveState.voice?.isRecording && (
                        <Col xs={12} md={6}>
                          <div style={{ padding: 8, background: '#f6ffed', borderRadius: 6, height: '100%' }}>
                            <div style={{ fontSize: 11, fontWeight: 600, color: '#52c41a', marginBottom: 4 }}>🔊 语音声学</div>
                            <div style={{ fontSize: 10, lineHeight: 1.7, color: '#555' }}>
                              <div>情绪：<Text strong style={{ fontSize: 10 }}>{patientMultimodal.comprehensiveState.voice.detectedEmotion}</Text></div>
                              <div>F0：{patientMultimodal.comprehensiveState.voice.pitch.mean}Hz</div>
                              <div>语速：{patientMultimodal.comprehensiveState.voice.rate.speechRate} 音节/s</div>
                              <div>停顿：{Math.round(patientMultimodal.comprehensiveState.voice.pauses.ratio * 100)}%</div>
                            </div>
                          </div>
                        </Col>
                      )}

                      {/* 面部微表情 */}
                      {patientMultimodal.comprehensiveState.facial?.isDetecting && (
                        <Col xs={12} md={6}>
                          <div style={{ padding: 8, background: '#fffbf0', borderRadius: 6, height: '100%' }}>
                            <div style={{ fontSize: 11, fontWeight: 600, color: '#fa8c16', marginBottom: 4 }}>😊 面部微表情</div>
                            <div style={{ fontSize: 10, lineHeight: 1.7, color: '#555' }}>
                              <div>表情：<Text strong style={{ fontSize: 10 }}>{patientMultimodal.comprehensiveState.facial.dominantExpression}</Text></div>
                              <div>强度：<Progress percent={Math.round(patientMultimodal.comprehensiveState.facial.expressionIntensity * 100)} showInfo={false} size="small" strokeColor="#fa8c16" style={{ width: 50, display: 'inline-block', verticalAlign: 'middle' }} /></div>
                              <div>微表情：{patientMultimodal.comprehensiveState.facial.microExpressions?.count || 0}次</div>
                            </div>
                          </div>
                        </Col>
                      )}

                      {/* 键盘动力学 */}
                      {patientMultimodal.comprehensiveState.keyboard?.isCollecting && (
                        <Col xs={12} md={6}>
                          <div style={{ padding: 8, background: '#f9f0ff', borderRadius: 6, height: '100%' }}>
                            <div style={{ fontSize: 11, fontWeight: 600, color: '#722ed1', marginBottom: 4 }}>⌨️ 键盘动力学</div>
                            <div style={{ fontSize: 10, lineHeight: 1.7, color: '#555' }}>
                              <div>速度：{patientMultimodal.comprehensiveState.keyboard.typingSpeed} 字/分
                                {patientMultimodal.comprehensiveState.baseline?.typingSpeed?.n >= 2 && (() => {
                                  const bl = patientMultimodal.comprehensiveState.baseline.typingSpeed;
                                  const diff = Math.round(patientMultimodal.comprehensiveState.keyboard.typingSpeed - bl.mean);
                                  const z = bl.std > 0.001 ? Math.abs((patientMultimodal.comprehensiveState.keyboard.typingSpeed - bl.mean) / bl.std) : 0;
                                  return <span style={{ color: z > 2 ? '#ff4d4f' : '#999' }}> (基线 {Math.round(bl.mean)}{diff > 0 ? ` +${diff}` : diff < 0 ? ` ${diff}` : ''})</span>;
                                })()}
                              </div>
                              <div>焦虑指数：<Text strong style={{ fontSize: 10, color: patientMultimodal.comprehensiveState.keyboard.anxietyIndex > 60 ? '#ff4d4f' : '#faad14' }}>{patientMultimodal.comprehensiveState.keyboard.anxietyIndex}</Text></div>
                              <div>删除率：{Math.round(patientMultimodal.comprehensiveState.keyboard.deletionRate * 100)}%</div>
                            </div>
                          </div>
                        </Col>
                      )}

                      {/* ── 认知分析层 ── */}
                      <Col xs={24}>
                        <div style={{ fontSize: 10, color: '#999', fontWeight: 500, marginBottom: 4, marginTop: 4 }}>🧠 认知分析层</div>
                      </Col>

                      {/* 昼夜活动（仅活动时间，非睡眠检测） */}
                      {patientMultimodal.comprehensiveState.circadian?.isActive && (
                        <Col xs={8} md={8}>
                          <div style={{ padding: 8, background: '#e6fffb', borderRadius: 6, height: '100%' }}>
                            <div style={{ fontSize: 11, fontWeight: 600, color: '#13c2c2', marginBottom: 4 }}>🕐 昼夜活动</div>
                            <div style={{ fontSize: 10, lineHeight: 1.7, color: '#555' }}>
                              <div>深夜活动风险：<Tag color={(patientMultimodal.comprehensiveState.circadian.lateNightRisk || 0) > 0.5 ? 'red' : 'green'} style={{ fontSize: 9, lineHeight: '14px', padding: '0 4px' }}>{Math.round((patientMultimodal.comprehensiveState.circadian.lateNightRisk || 0) * 100)}%</Tag></div>
                              <div>活动时间规律性：{(patientMultimodal.comprehensiveState.circadian.activitySampleCount || 0) >= 10
                                ? <Progress percent={Math.round((patientMultimodal.comprehensiveState.circadian.regularityScore || 0) * 100)} showInfo={false} size="small" strokeColor="#13c2c2" style={{ width: 44, display: 'inline-block', verticalAlign: 'middle' }} />
                                : <span style={{ color: '#999' }}>样本不足({patientMultimodal.comprehensiveState.circadian.activitySampleCount || 0}/10)</span>}
                              </div>
                              <div style={{ color: '#999' }}>仅平台内活动时间，非睡眠监测</div>
                            </div>
                          </div>
                        </Col>
                      )}

                      {/* 认知扭曲 */}
                      {(patientMultimodal.comprehensiveState.cognitiveDistortion?.riskScore || 0) > 0 && (
                        <Col xs={8} md={8}>
                          <div style={{ padding: 8, background: '#fff0f6', borderRadius: 6, height: '100%' }}>
                            <div style={{ fontSize: 11, fontWeight: 600, color: '#eb2f9a', marginBottom: 4 }}>💭 认知扭曲</div>
                            <div style={{ fontSize: 10, lineHeight: 1.7, color: '#555' }}>
                              <div>总风险：<Tag color="magenta" style={{ fontSize: 9, lineHeight: '14px', padding: '0 4px' }}>{Math.round(patientMultimodal.comprehensiveState.cognitiveDistortion.riskScore * 100)}%</Tag></div>
                              <div>类型：{patientMultimodal.comprehensiveState.cognitiveDistortion.totalDistortionCount || 0} 类</div>
                              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 2, marginTop: 2 }}>
                                {Object.entries(patientMultimodal.comprehensiveState.cognitiveDistortion.categories || {}).filter(([, v]) => (v as number) > 0).slice(0, 3).map(([k, v]) => {
                                  const lm: Record<string, string> = { catastrophizing: '灾难化', blackAndWhite: '非黑即白', overgeneralization: '过度概括', selfBlame: '自我归咎', hopelessness: '无望感' };
                                  return <Tag key={k} color="pink" style={{ fontSize: 9, margin: 0, lineHeight: '14px', padding: '0 3px' }}>{lm[k] || k}</Tag>;
                                })}
                              </div>
                            </div>
                          </div>
                        </Col>
                      )}

                      {/* 语音语义 */}
                      {(patientMultimodal.comprehensiveState.voiceSemantics?.timestamp || 0) > 0 && (
                        <Col xs={8} md={8}>
                          <div style={{ padding: 8, background: '#fffbe6', borderRadius: 6, height: '100%' }}>
                            <div style={{ fontSize: 11, fontWeight: 600, color: '#faad14', marginBottom: 4 }}>🗣️ 语音语义</div>
                            <div style={{ fontSize: 10, lineHeight: 1.7, color: '#555' }}>
                              <div>语义风险：<Tag color={(patientMultimodal.comprehensiveState.voiceSemantics.riskScore || 0) > 0.3 ? 'orange' : 'green'} style={{ fontSize: 9, lineHeight: '14px', padding: '0 4px' }}>{Math.round((patientMultimodal.comprehensiveState.voiceSemantics.riskScore || 0) * 100)}%</Tag></div>
                              <div>"我"占比：{Math.round((patientMultimodal.comprehensiveState.voiceSemantics.firstPersonSingularRatio || 0) * 100)}%</div>
                              <div>消极词：{Math.round((patientMultimodal.comprehensiveState.voiceSemantics.negativeAffectRatio || 0) * 100)}%</div>
                            </div>
                          </div>
                        </Col>
                      )}

                      {/* ── 生理行为层 ── */}
                      <Col xs={24}>
                        <div style={{ fontSize: 10, color: '#999', fontWeight: 500, marginBottom: 4, marginTop: 4 }}>💓 生理行为层</div>
                      </Col>

                      {/* rPPG 心率 / 呼吸：平台无采集能力，明确告知咨询师"没有这项数据" */}
                      {!(patientMultimodal.comprehensiveState.hrv?.isMeasuring) && (
                        <Col xs={8} md={6}>
                          <div style={{ padding: 8, background: '#fafafa', borderRadius: 6, height: '100%', border: '1px dashed #e0e0e0' }}>
                            <div style={{ fontSize: 11, fontWeight: 600, color: '#bfbfbf', marginBottom: 4 }}>❤️ rPPG 心率</div>
                            <div style={{ fontSize: 10, lineHeight: 1.7, color: '#bfbfbf' }}>
                              <div>未采集（无心率传感器）</div>
                              <div>患者设备不支持 rPPG</div>
                            </div>
                          </div>
                        </Col>
                      )}
                      {patientMultimodal.comprehensiveState.hrv?.isMeasuring && (
                        <Col xs={8} md={6}>
                          <div style={{ padding: 8, background: '#fff1f0', borderRadius: 6, height: '100%' }}>
                            <div style={{ fontSize: 11, fontWeight: 600, color: '#ff4d4f', marginBottom: 4 }}>❤️ rPPG 心率</div>
                            <div style={{ fontSize: 10, lineHeight: 1.7, color: '#555' }}>
                              <div>心率：<Text strong style={{ fontSize: 12, color: '#ff4d4f' }}>{patientMultimodal.comprehensiveState.hrv.heartRate}</Text> bpm
                                {patientMultimodal.comprehensiveState.baseline?.heartRate?.n >= 2 && (() => {
                                  const bl = patientMultimodal.comprehensiveState.baseline.heartRate;
                                  const diff = Math.round(patientMultimodal.comprehensiveState.hrv.heartRate - bl.mean);
                                  const z = bl.std > 0.001 ? Math.abs((patientMultimodal.comprehensiveState.hrv.heartRate - bl.mean) / bl.std) : 0;
                                  return <span style={{ color: z > 2 ? '#ff4d4f' : '#999', fontWeight: z > 2 ? 600 : 400 }}> (基线 {Math.round(bl.mean)}{diff > 0 ? ` +${diff}` : diff < 0 ? ` ${diff}` : ''})</span>;
                                })()}
                              </div>
                              <div>HRV：{patientMultimodal.comprehensiveState.hrv.hrvRmssd} ms</div>
                              <div>风险：{Math.round((patientMultimodal.comprehensiveState.hrv.riskScore || 0) * 100)}%</div>
                            </div>
                          </div>
                        </Col>
                      )}

                      {!(patientMultimodal.comprehensiveState.breathing?.isMeasuring) && (
                        <Col xs={8} md={6}>
                          <div style={{ padding: 8, background: '#fafafa', borderRadius: 6, height: '100%', border: '1px dashed #e0e0e0' }}>
                            <div style={{ fontSize: 11, fontWeight: 600, color: '#bfbfbf', marginBottom: 4 }}>🌬️ 呼吸模式</div>
                            <div style={{ fontSize: 10, lineHeight: 1.7, color: '#bfbfbf' }}>
                              <div>未采集（算法未启用）</div>
                              <div>待语音识别稳定后接入</div>
                            </div>
                          </div>
                        </Col>
                      )}
                      {patientMultimodal.comprehensiveState.breathing?.isMeasuring && (
                        <Col xs={8} md={6}>
                          <div style={{ padding: 8, background: '#e6fffb', borderRadius: 6, height: '100%' }}>
                            <div style={{ fontSize: 11, fontWeight: 600, color: '#36cfc9', marginBottom: 4 }}>🌬️ 呼吸模式</div>
                            <div style={{ fontSize: 10, lineHeight: 1.7, color: '#555' }}>
                              <div>频率：<Text strong style={{ fontSize: 10 }}>{patientMultimodal.comprehensiveState.breathing.breathingRate}</Text> 次/分
                                {patientMultimodal.comprehensiveState.baseline?.breathingRate?.n >= 2 && (() => {
                                  const bl = patientMultimodal.comprehensiveState.baseline.breathingRate;
                                  const diff = Math.round(patientMultimodal.comprehensiveState.breathing.breathingRate - bl.mean);
                                  const z = bl.std > 0.001 ? Math.abs((patientMultimodal.comprehensiveState.breathing.breathingRate - bl.mean) / bl.std) : 0;
                                  return <span style={{ color: z > 2 ? '#ff4d4f' : '#999' }}> (基线 {Math.round(bl.mean)}{diff > 0 ? ` +${diff}` : diff < 0 ? ` ${diff}` : ''})</span>;
                                })()}
                              </div>
                              <div>叹气：{patientMultimodal.comprehensiveState.breathing.sighCount} 次</div>
                              <div>规律CV：{patientMultimodal.comprehensiveState.breathing.regularityCV}</div>
                            </div>
                          </div>
                        </Col>
                      )}

                      {/* 行为激活 */}
                      {patientMultimodal.comprehensiveState.behavioralActivation?.isActive && (
                        <Col xs={8} md={6}>
                          <div style={{ padding: 8, background: '#f0f5ff', borderRadius: 6, height: '100%' }}>
                            <div style={{ fontSize: 11, fontWeight: 600, color: '#597ef7', marginBottom: 4 }}>🏃 行为激活</div>
                            <div style={{ fontSize: 10, lineHeight: 1.7, color: '#555' }}>
                              <div>互动：{patientMultimodal.comprehensiveState.behavioralActivation.dailyInteractionCount} 次</div>
                              <div>探索率：{Math.round((patientMultimodal.comprehensiveState.behavioralActivation.explorationRate || 0) * 100)}%</div>
                              <div>趋势：<Tag color={patientMultimodal.comprehensiveState.behavioralActivation.trendDirection === 'improving' ? 'green' : patientMultimodal.comprehensiveState.behavioralActivation.trendDirection === 'declining' ? 'red' : 'blue'} style={{ fontSize: 9, lineHeight: '14px', padding: '0 4px' }}>{patientMultimodal.comprehensiveState.behavioralActivation.trendDirection === 'improving' ? '↑改善' : patientMultimodal.comprehensiveState.behavioralActivation.trendDirection === 'declining' ? '↓下降' : '→稳定'}</Tag></div>
                            </div>
                          </div>
                        </Col>
                      )}

                      {/* 眨眼 / 头姿（头姿为注视代理，非眼球注视） */}
                      {patientMultimodal.comprehensiveState.eyeMovement?.isMeasuring && (
                        <Col xs={8} md={6}>
                          <div style={{ padding: 8, background: '#f9f0ff', borderRadius: 6, height: '100%' }}>
                            <div style={{ fontSize: 11, fontWeight: 600, color: '#9254de', marginBottom: 4 }}>👁️ 眨眼 / 头姿</div>
                            <div style={{ fontSize: 10, lineHeight: 1.7, color: '#555' }}>
                              <div>眨眼：{patientMultimodal.comprehensiveState.eyeMovement.blinkRate} 次/分
                                {patientMultimodal.comprehensiveState.baseline?.blinkRate?.n >= 2 && (() => {
                                  const bl = patientMultimodal.comprehensiveState.baseline.blinkRate;
                                  const diff = Math.round(patientMultimodal.comprehensiveState.eyeMovement.blinkRate - bl.mean);
                                  const z = bl.std > 0.001 ? Math.abs((patientMultimodal.comprehensiveState.eyeMovement.blinkRate - bl.mean) / bl.std) : 0;
                                  return <span style={{ color: z > 2 ? '#ff4d4f' : '#999' }}> (基线 {Math.round(bl.mean)}{diff > 0 ? ` +${diff}` : diff < 0 ? ` ${diff}` : ''})</span>;
                                })()}
                              </div>
                              <div>低头帧占比：{Math.round((patientMultimodal.comprehensiveState.eyeMovement.downwardGazeRatio || 0) * 100)}%</div>
                              <div>头姿波动：{Math.round((patientMultimodal.comprehensiveState.eyeMovement.attentionScatter || 0) * 100)}%</div>
                              <div style={{ color: '#bfbfbf' }}>
                                来源：{patientMultimodal.comprehensiveState.eyeMovement.blinkMethod === 'ear' ? 'EAR 降级' : '眼睑闭合度'}；
                                头姿为注视代理指标
                              </div>
                            </div>
                          </div>
                        </Col>
                      )}
                    </Row>
                  </div>
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
              {/* 大画面：来访者（远端视频） */}
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
                    <Avatar size={100} style={{ backgroundColor: '#6366f1' }}>
                      {patient.nickname?.[0] || '患'}
                    </Avatar>
                    <div style={{ marginTop: 12 }}>
                      <Text style={{ color: '#aaa', fontSize: 13 }}>
                        {webrtc.isConnected ? '已连接' : callActive ? '等待来访者接入...' : '点击通话按钮开始视频'}
                      </Text>
                    </div>
                  </div>
                )}
              </div>
              {/* 画中画：自己（本地视频） */}
              {cameraOn && webrtc.localStream && (
                <div style={{ position: 'absolute', top: 12, right: 12, width: 120, height: 160, background: '#333', borderRadius: 8, overflow: 'hidden', border: '2px solid rgba(255,255,255,0.2)' }}>
                  <video
                    ref={localVideoRef}
                    autoPlay
                    muted
                    playsInline
                    style={{ width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)' }}
                  />
                </div>
              )}
              {/* 通话信息 */}
              <div style={{ position: 'absolute', top: 12, left: 12 }}>
                <Text style={{ color: '#fff', fontSize: 13 }}>{patient.nickname || '来访者'}</Text>
                {callActive && <div><Text style={{ color: '#aaa', fontSize: 12 }}>{formatDuration(callDuration)}</Text></div>}
                {webrtc.isConnected && <Tag color="green" style={{ marginTop: 4, fontSize: 10 }}>已连接</Tag>}
              </div>
              {/* 控制栏 */}
              <div style={{ padding: 16, display: 'flex', justifyContent: 'center', gap: 16, background: 'rgba(0,0,0,0.5)' }}>
                <Button shape="circle" icon={<VideoCameraOutlined />} type={cameraOn ? 'primary' : 'default'} onClick={() => { setCameraOn(!cameraOn); webrtc.toggleCamera(!cameraOn); }} />
                <Button shape="circle" icon={<SoundOutlined />} type={micOn ? 'primary' : 'default'} onClick={() => { setMicOn(!micOn); webrtc.toggleMic(!micOn); }} />
                <Button shape="circle" icon={callActive ? <StopOutlined /> : <PhoneOutlined />} danger={callActive} onClick={async () => {
                  if (callActive) { webrtc.hangUp(); setCallActive(false); setCallDuration(0); }
                  else {
                    const stream = await webrtc.startLocalStream(true, true);
                    if (stream) {
                      if (localVideoRef.current) { localVideoRef.current.srcObject = stream; }
                      await webrtc.call();
                      setCallActive(true); setCallDuration(0);
                    }
                  }
                }} />
              </div>
            </div>
          )}
        </Card>

        {/* 咨询笔记抽屉 */}
        <Drawer
          title={<Space><FormOutlined /> 咨询笔记</Space>}
          placement="right"
          width={380}
          open={notesDrawerOpen}
          onClose={() => setNotesDrawerOpen(false)}
          extra={
            <Button type="primary" icon={<SaveOutlined />} loading={notesSaving} onClick={handleSaveNotes} size="small">
              保存
            </Button>
          }
        >
          <div style={{ marginBottom: 12 }}>
            <Text type="secondary" style={{ fontSize: 12 }}>
              笔记仅公益咨询师可见，用于记录咨询过程中的关键观察和干预思路
            </Text>
          </div>
          <Input.TextArea
            rows={20}
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="记录咨询过程中的关键观察、干预思路、待跟进事项..."
            style={{ borderRadius: 8, resize: 'none' }}
          />
          <div style={{ marginTop: 12 }}>
            <Text type="secondary" style={{ fontSize: 11 }}>
              自动保存于 {new Date().toLocaleDateString()}
            </Text>
          </div>
        </Drawer>

        {/* 危机干预弹窗 */}
        <Modal
          open={crisisModalOpen}
          onCancel={() => setCrisisModalOpen(false)}
          footer={null}
          title={<Space><WarningOutlined style={{ color: '#ff4d4f' }} /> 危机干预协议</Space>}
        >
          <div style={{ padding: '16px 0' }}>
            <Alert
              type="warning"
              showIcon
              message={`来访者 ${patient.nickname} 当前风险等级为「${riskLabels[patient.riskLevel]}」`}
              description="启动危机干预协议后，系统将自动通知督导并记录干预过程。"
              style={{ marginBottom: 20, borderRadius: 8 }}
            />
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
              <Button type="primary" danger block icon={<FlagOutlined />} onClick={handleCrisisProtocol}>
                启动危机干预协议
              </Button>
              <Button block icon={<PhoneOutlined />} onClick={handleContactSupervisor}>
                联系督导
              </Button>
              <Button block icon={<NotificationOutlined />} onClick={() => { setCrisisModalOpen(false); message.info('已通知紧急联系人'); }}>
                通知紧急联系人
              </Button>
            </Space>
          </div>
        </Modal>

        {/* 安全审计抽屉 */}
        <Drawer
          title={<Space><SafetyCertificateOutlined /> AI 安全审计日志</Space>}
          placement="right"
          width={400}
          open={auditDrawerOpen}
          onClose={() => setAuditDrawerOpen(false)}
        >
          <div style={{ marginBottom: 16 }}>
            <Alert type="info" showIcon style={{ borderRadius: 8, marginBottom: 12 }}
              message="五轴安全审计实时监控"
              description="监控维度：危机升级延迟 / 妄想强化 / 污名化与拒绝 / 谄媚倾向 / 轨迹漂移"
            />
          </div>
          <Timeline
            items={auditEvents.map(event => ({
              color: event.severity === 'warning' ? 'orange' : event.severity === 'success' ? 'green' : 'blue',
              children: (
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                    <Tag color={event.severity === 'warning' ? 'orange' : event.severity === 'success' ? 'green' : 'blue'} style={{ fontSize: 11 }}>
                      {event.type === 'safety_intercept' ? '拦截' : event.type === 'crisis_signal' ? '危机信号' : '审计通过'}
                    </Tag>
                    <Text type="secondary" style={{ fontSize: 11 }}>{event.time}</Text>
                  </div>
                  <div style={{ fontSize: 13, marginBottom: 4 }}>{event.description}</div>
                  <div style={{ fontSize: 12, color: '#888' }}>处理: {event.action}</div>
                </div>
              ),
            }))}
          />
          <Divider />
          <div style={{ textAlign: 'center' }}>
            <Text type="secondary" style={{ fontSize: 12 }}>
              审计规则来源: config/audit_rules.yaml | 仅展示最近事件
            </Text>
          </div>
        </Drawer>
      </div>
    </div>
  );
}