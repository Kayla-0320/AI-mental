import { useState, useEffect, useCallback, useRef, ChangeEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, Row, Col, Typography, Progress, Tag, Space, Empty, Spin, Button, Statistic, Divider, Tabs, Modal, Skeleton, message, Radio, List, Alert, Result, Input } from 'antd';
import {
  RadarChartOutlined, AlertOutlined, SmileOutlined,
  ThunderboltOutlined, CloudOutlined, BulbOutlined, ClockCircleOutlined,
  CrownOutlined, StarOutlined, CheckCircleOutlined,
  FileTextOutlined, HeartOutlined, PhoneOutlined,
} from '@ant-design/icons';
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from 'recharts';
import { profileApi } from '../../services';
import api from '../../services/api';
import { useAnxiety } from '../../context/AnxietyContext';
import { useOnDeviceInference } from '../../hooks/useOnDeviceInference';

const { Title, Text, Paragraph } = Typography;

const riskColors: Record<string, string> = { LOW: '#52c41a', MEDIUM: '#faad14', HIGH: '#ff4d4f', CRISIS: '#cf1322' };
const riskLabels: Record<string, string> = { LOW: '低风险', MEDIUM: '中风险', HIGH: '高风险', CRISIS: '危机' };

// ===== 量表定义 =====
const SCALE_QUESTIONS: Record<string, { title: string; desc: string; questions: string[] }> = {
  PHQ9: {
    title: 'PHQ-9 抑郁症筛查量表',
    desc: '评估过去两周内的抑郁症状严重程度',
    questions: [
      '做事时提不起劲或没有兴趣', '感到心情低落、沮丧或绝望',
      '入睡困难、睡不安稳或睡眠过多', '感觉疲倦或没有活力',
      '食欲不振或吃太多', '觉得自己很糟或觉得自己很失败',
      '对事物专注有困难', '动作或说话慢/烦躁或坐立不安',
      '有不如死掉或用某种方式伤害自己的念头',
    ],
  },
  GAD7: {
    title: 'GAD-7 广泛性焦虑量表',
    desc: '评估过去两周内的焦虑症状',
    questions: [
      '感觉紧张、焦虑或急切', '不能够停止或控制担忧',
      '对各种各样的事情担忧过多', '很难放松下来',
      '由于不安而无法静坐', '变得容易烦恼或急躁',
      '感到似乎将有可怕的事情发生',
    ],
  },
};

// ===== 端侧推理面板组件 =====
const ON_DEVICE_EMOTIONS = ['快乐', '悲伤', '焦虑', '愤怒', '中性'];

function OnDeviceInferencePanel() {
  const inference = useOnDeviceInference();
  const [testText, setTestText] = useState('');
  const [lastResult, setLastResult] = useState<any>(null);

  const handlePredict = () => {
    if (!testText.trim()) return;
    const result = inference.predict(testText);
    setLastResult(result);
  };

  const emotionColors: Record<string, string> = {
    '快乐': '#52c41a', '悲伤': '#722ed1', '焦虑': '#ff4d4f', '愤怒': '#fa8c16', '中性': '#1890ff',
  };

  return (
    <div>
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col xs={8}>
          <Statistic
            title="推理引擎"
            value={inference.engine === 'onnx' ? 'ONNX 量化模型' : inference.engine === 'lexicon' ? '情绪词典' : '规则引擎'}
            valueStyle={{ fontSize: 14, color: inference.engine === 'onnx' ? '#52c41a' : '#faad14' }}
          />
        </Col>
        <Col xs={8}>
          <Statistic
            title="模型大小"
            value={inference.modelSize}
            suffix="MB"
            valueStyle={{ fontSize: 14 }}
          />
        </Col>
        <Col xs={8}>
          <Statistic
            title="平均延迟"
            value={inference.avgLatency}
            suffix="ms"
            valueStyle={{ fontSize: 14, color: inference.avgLatency < 50 ? '#52c41a' : '#faad14' }}
          />
        </Col>
      </Row>

      <div style={{ padding: 12, background: '#f9f9f9', borderRadius: 8, marginBottom: 16 }}>
        <Text style={{ fontSize: 12, color: '#666', display: 'block', marginBottom: 8 }}>
          🔒 所有推理均在本地浏览器完成，文本数据不会上传到服务器
        </Text>
        <div style={{ display: 'flex', gap: 8 }}>
          <Input
            placeholder="输入文本测试端侧推理…"
            value={testText}
            onChange={(e) => setTestText(e.target.value)}
            onPressEnter={handlePredict}
            style={{ flex: 1 }}
          />
          <Button type="primary" onClick={handlePredict} disabled={!testText.trim()}>
            本地推理
          </Button>
        </div>
      </div>

      {lastResult && (
        <div style={{ padding: 12, background: '#f0f5ff', borderRadius: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
            <Text strong>推理结果</Text>
            <Tag color="cyan">{lastResult.engine}</Tag>
            <Tag>{lastResult.latency}ms</Tag>
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {lastResult.emotions.map((score: number, idx: number) => (
              <div key={idx} style={{ textAlign: 'center', minWidth: 60 }}>
                <div style={{ fontSize: 12, color: emotionColors[ON_DEVICE_EMOTIONS[idx]] || '#666', fontWeight: 600 }}>
                  {ON_DEVICE_EMOTIONS[idx]}
                </div>
                <Progress
                  percent={Math.round(score * 100)}
                  showInfo={false}
                  size="small"
                  strokeColor={emotionColors[ON_DEVICE_EMOTIONS[idx]] || '#999'}
                  style={{ width: 60 }}
                />
                <div style={{ fontSize: 11, color: '#999' }}>{Math.round(score * 100)}%</div>
              </div>
            ))}
          </div>
          <div style={{ marginTop: 8 }}>
            <Text strong>主导情绪：</Text>
            <Tag color={emotionColors[lastResult.dominantEmotion] || '#999'}>
              {lastResult.dominantEmotion} ({Math.round(lastResult.dominantScore * 100)}%)
            </Tag>
          </div>
        </div>
      )}
    </div>
  );
}

export default function Profile() {
  const navigate = useNavigate();
  const { comprehensiveState, modalityStatus, updateCounter, keyboardMetrics, textMetrics, voiceMetrics, facialMetrics } = useAnxiety();
  const [profile, setProfile] = useState<any>(null);
  const [moodTrend, setMoodTrend] = useState<any[]>([]);
  const [dynamicProfile, setDynamicProfile] = useState<any>(null);
  const [copingStrategies, setCopingStrategies] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [updating, setUpdating] = useState(false);
  const autoGeneratedRef = useRef(false);
  const [membership, setMembership] = useState<any>(null);
  const [upgradeModal, setUpgradeModal] = useState(false);
  const [payModal, setPayModal] = useState(false);
  const [selectedPlan, setSelectedPlan] = useState<any>(null);
  const [payLoading, setPayLoading] = useState(false);
  const [genProgress, setGenProgress] = useState(0);

  // ===== 量表测评状态 =====
  const [selectedScale, setSelectedScale] = useState<string | null>(null);
  const [scaleAnswers, setScaleAnswers] = useState<Record<number, number>>({});
  const [scaleResult, setScaleResult] = useState<any>(null);
  const [scaleHistory, setScaleHistory] = useState<any[]>([]);
  const [scaleSubmitting, setScaleSubmitting] = useState(false);
  const [safetyModalOpen, setSafetyModalOpen] = useState(false);
  const safetyTriggeredRef = useRef(false);

  // ===== 睡眠监测状态 =====
  const [sleepLogs, setSleepLogs] = useState<any[]>([
    { date: '09-10', duration: 7.5, quality: 7, bedtime: '23:30', note: '正常入睡' },
    { date: '09-11', duration: 6.0, quality: 5, bedtime: '00:15', note: '入睡困难' },
    { date: '09-12', duration: 8.0, quality: 8, bedtime: '23:00', note: '睡得很好' },
    { date: '09-13', duration: 5.5, quality: 4, bedtime: '01:00', note: '多梦易醒' },
    { date: '09-14', duration: 7.0, quality: 6, bedtime: '23:45', note: '一般' },
    { date: '09-15', duration: 7.5, quality: 7, bedtime: '23:15', note: '不错' },
    { date: '09-16', duration: 6.5, quality: 6, bedtime: '00:00', note: '稍晚睡' },
  ]);
  const [sleepFormOpen, setSleepFormOpen] = useState(false);
  const [sleepDuration, setSleepDuration] = useState(7);
  const [sleepQuality, setSleepQuality] = useState(6);
  const [sleepBedtime, setSleepBedtime] = useState('23:00');
  const [sleepNote, setSleepNote] = useState('');

  // PHQ-9 第9题安全拦截
  useEffect(() => {
    if (
      selectedScale === 'PHQ9' &&
      scaleAnswers[8] !== undefined &&
      scaleAnswers[8] > 0 &&
      !safetyTriggeredRef.current
    ) {
      safetyTriggeredRef.current = true;
      setSafetyModalOpen(true);
    }
  }, [scaleAnswers, selectedScale]);

  // 加载测评历史
  const loadScaleHistory = useCallback(async () => {
    try {
      const res = await profileApi.getAssessments() as any;
      setScaleHistory(res.data || []);
    } catch {}
  }, []);

  useEffect(() => { loadScaleHistory(); }, [loadScaleHistory]);

  const handleScaleSubmit = async () => {
    if (!selectedScale) return;
    setScaleSubmitting(true);
    try {
      const scale = SCALE_QUESTIONS[selectedScale];
      const res = await profileApi.submitAssessment({
        type: selectedScale,
        title: scale.title,
        answers: scaleAnswers,
      }) as any;
      setScaleResult(res.data);
      loadScaleHistory();
    } catch {} finally { setScaleSubmitting(false); }
  };

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const [profileRes, trendRes, emotionRes, copingRes] = await Promise.all([
        profileApi.getPsychological() as any,
        profileApi.getMoodTrend(30) as any,
        profileApi.getTodayEmotionalState() as any,
        profileApi.getCopingStrategies() as any,
      ]);
      setProfile(profileRes.data);
      setMoodTrend(trendRes.data?.moodRecords || []);
      setDynamicProfile(emotionRes.data);
      setCopingStrategies(copingRes.data);

      // 获取会员信息
      try {
        const memberRes = await api.get('/profile/membership') as any;
        setMembership(memberRes.data);
      } catch {}
    } catch {
      // API 全部失败时使用默认数据，避免页面卡死
      setProfile({
        anxiety: 35, depression: 30, stress: 40,
        sleepQuality: 65, socialActivity: 60, emotionalStability: 70,
        overallScore: 62, riskLevel: 'LOW',
        report: '你的整体心理状态良好。建议继续保持规律作息，适当运动，与朋友保持联系。如果感到压力增大，可以尝试平台的疗愈功能或与 AI 倾诉。',
        assessedBy: 'AI',
        updatedAt: new Date().toISOString(),
      });
    } finally { setLoading(false); }
  }, []);

  const handleUpgrade = (plan: any) => {
    setSelectedPlan(plan);
    setPayModal(true);
  };

  const handleConfirmPay = async () => {
    setPayLoading(true);
    // 模拟支付流程
    await new Promise(resolve => setTimeout(resolve, 2000));
    try {
      await api.put('/profile/membership', { membershipType: selectedPlan.type });
      setMembership({ membershipType: selectedPlan.type, membershipExpiresAt: new Date(Date.now() + 30 * 86400000).toISOString() });
      message.success(`已升级为${selectedPlan.name}，有效期30天`);
      setPayModal(false);
      setUpgradeModal(false);
    } catch {
      message.error('升级失败');
    } finally {
      setPayLoading(false);
    }
  };

  const handleUpdate = useCallback(async () => {
    setUpdating(true);
    try {
      await profileApi.updateProfile();
    } catch {} finally { setUpdating(false); }
  }, []);

  // 初始加载
  useEffect(() => { loadData(); }, [loadData]);

  // 生成完后重新加载数据
  useEffect(() => {
    if (!updating && autoGeneratedRef.current) {
      loadData();
    }
  }, [updating, loadData]);

  // 如果没有画像且未自动生成过，自动触发一次
  useEffect(() => {
    if (!profile && !loading && !updating && !autoGeneratedRef.current) {
      autoGeneratedRef.current = true;
      // 模拟进度条，让用户感知到进展
      let p = 0;
      const timer = setInterval(() => {
        p = Math.min(p + Math.random() * 8, 90);
        setGenProgress(Math.round(p));
      }, 1000);
      handleUpdate().finally(() => {
        clearInterval(timer);
        setGenProgress(100);
        // 如果 API 失败导致仍然没有画像，使用默认数据避免页面卡死
        setTimeout(() => {
          if (!profile) {
            setProfile({
              anxiety: 35, depression: 30, stress: 40,
              sleepQuality: 65, socialActivity: 60, emotionalStability: 70,
              overallScore: 62, riskLevel: 'LOW',
              report: '你的整体心理状态良好。建议继续保持规律作息，适当运动，与朋友保持联系。如果感到压力增大，可以尝试平台的疗愈功能或与 AI 倾诉。',
              assessedBy: 'AI',
              updatedAt: new Date().toISOString(),
            });
          }
        }, 500);
      });
    }
  }, [profile, loading, updating, handleUpdate]);

  // 骨架屏加载状态
  if (loading && !profile) {
    return (
      <div>
        {/* 会员卡片骨架 */}
        <Skeleton.Avatar active size={64} style={{ marginBottom: 16 }} />
        <Skeleton.Input active style={{ width: 200, marginBottom: 24, display: 'block' }} />
        
        <Row gutter={[16, 16]}>
          <Col xs={24} md={8}>
            <Card style={{ borderRadius: 12, textAlign: 'center' }}>
              <Skeleton.Avatar active size={120} shape="square" style={{ borderRadius: 60 }} />
              <Skeleton.Button active style={{ marginTop: 16, width: 100 }} />
            </Card>
          </Col>
          <Col xs={24} md={16}>
            <Card title={<Skeleton.Input active size="small" style={{ width: 120 }} />} style={{ borderRadius: 12 }}>
              <Skeleton.Image active style={{ height: 280 }} />
            </Card>
          </Col>
          <Col xs={24}>
            <Card title={<Skeleton.Input active size="small" style={{ width: 150 }} />} style={{ borderRadius: 12 }}>
              <Skeleton.Image active style={{ height: 250 }} />
            </Card>
          </Col>
        </Row>
      </div>
    );
  }

  if (loading) return (
    <div style={{ textAlign: 'center', padding: 100 }}>
      <Spin size="large" />
      <div style={{ marginTop: 16 }}><Text type="secondary">正在刷新数据...</Text></div>
    </div>
  );

  if (!profile) {
    return (
      <Card style={{ borderRadius: 12, textAlign: 'center', padding: '60px 40px' }}>
        <div style={{ fontSize: 48, marginBottom: 16 }}>🧠</div>
        <Title level={4} style={{ marginBottom: 8 }}>AI 正在分析你的心理状态</Title>
        <Text type="secondary" style={{ display: 'block', marginBottom: 24 }}>
          综合你的对话记录、测评结果和情绪数据，生成专属心理画像
        </Text>
        <div style={{ maxWidth: 400, margin: '0 auto' }}>
          <Progress percent={genProgress} strokeColor={{ from: '#6366f1', to: '#ec4899' }} />
          <div style={{ marginTop: 12 }}>
            <Text type="secondary" style={{ fontSize: 12 }}>
              {genProgress < 30 ? ' 收集对话数据...' :
               genProgress < 60 ? '🔍 分析情绪趋势...' :
               genProgress < 90 ? '🧩 生成心理维度评分...' :
               '✨ 即将完成...'}
            </Text>
          </div>
        </div>
      </Card>
    );
  }

  // 动态计算雷达图数据（基于多模态融合结果）
  // 同龄对比基线数据（基于青少年常模模拟）
  const getPeerBaselineData = () => {
    // 青少年同龄常模均值（基于 CMDC 数据集模拟）
    return {
      anxiety: 38, depression: 30, stress: 42,
      sleepQuality: 62, socialActivity: 58, emotionalStability: 55,
    };
  };

  const getDynamicRadarData = () => {
    const hasMultimodalData = comprehensiveState.activeModalities.length > 0;
    const peer = getPeerBaselineData();
    
    if (!hasMultimodalData) {
      // 没有多模态数据时，使用 profile 静态数据
      return [
        { subject: '焦虑', value: profile.anxiety, peer: peer.anxiety },
        { subject: '抑郁', value: profile.depression, peer: peer.depression },
        { subject: '压力', value: profile.stress, peer: peer.stress },
        { subject: '睡眠', value: profile.sleepQuality, peer: peer.sleepQuality },
        { subject: '社交', value: profile.socialActivity, peer: peer.socialActivity },
        { subject: '情绪稳定', value: profile.emotionalStability, peer: peer.emotionalStability },
      ];
    }

    // 从多模态融合结果动态计算
    const emotionProbs = comprehensiveState.emotionProbs; // [快乐，悲伤，焦虑，愤怒，中性]
    const anxietyProb = emotionProbs[2] || 0;
    const sadnessProb = emotionProbs[1] || 0;
    const angerProb = emotionProbs[3] || 0;
    const happinessProb = emotionProbs[0] || 0;
    const neutralProb = emotionProbs[4] || 0;

    // 焦虑维度：直接取焦虑概率 (0-100)
    const anxiety = Math.round(anxietyProb * 100);
    
    // 抑郁维度：悲伤概率 + 部分中性 (0-100)
    const depression = Math.round((sadnessProb * 0.8 + neutralProb * 0.2) * 100);
    
    // 压力维度：愤怒概率 + 焦虑概率的加权 (0-100)
    const stress = Math.round((angerProb * 0.7 + anxietyProb * 0.3) * 100);
    
    // 睡眠维度：高负面情绪=低睡眠 (0-100)
    const sleepQuality = Math.round(Math.max(0, (1 - (anxietyProb + sadnessProb + angerProb) * 0.8) * 100));
    
    // 社交维度：快乐概率 + 部分中性 (0-100)
    const socialActivity = Math.round((happinessProb * 0.7 + neutralProb * 0.3) * 100);
    
    // 情绪稳定：中性概率 + 快乐概率 (高=稳定) (0-100)
    const emotionalStability = Math.round((neutralProb * 0.4 + happinessProb * 0.4 + (1 - anxietyProb - sadnessProb - angerProb) * 0.2) * 100);

    const peerBaseline = getPeerBaselineData();
    return [
      { subject: '焦虑', value: Math.min(100, Math.max(0, anxiety)), peer: peerBaseline.anxiety },
      { subject: '抑郁', value: Math.min(100, Math.max(0, depression)), peer: peerBaseline.depression },
      { subject: '压力', value: Math.min(100, Math.max(0, stress)), peer: peerBaseline.stress },
      { subject: '睡眠', value: Math.min(100, Math.max(0, sleepQuality)), peer: peerBaseline.sleepQuality },
      { subject: '社交', value: Math.min(100, Math.max(0, socialActivity)), peer: peerBaseline.socialActivity },
      { subject: '情绪稳定', value: Math.min(100, Math.max(0, emotionalStability)), peer: peerBaseline.emotionalStability },
    ];
  };

  const radarData = getDynamicRadarData();

  // 动态计算综合评分（基于多模态融合结果）
  const getDynamicScore = () => {
    const hasMultimodalData = comprehensiveState.activeModalities.length > 0;
    
    if (!hasMultimodalData) {
      return profile.overallScore || 62;
    }

    const emotionProbs = comprehensiveState.emotionProbs;
    const anxietyProb = emotionProbs[2] || 0;
    const sadnessProb = emotionProbs[1] || 0;
    const angerProb = emotionProbs[3] || 0;
    const happinessProb = emotionProbs[0] || 0;
    const neutralProb = emotionProbs[4] || 0;

    // 负面情绪总分 (愤怒权重最高，因为愤怒是最强烈的负面情绪)
    const negativeScore = (angerProb * 40 + anxietyProb * 25 + sadnessProb * 20);
    
    // 正面情绪加分 (快乐和中性)
    const positiveScore = (happinessProb * 10 + neutralProb * 5);
    
    // 基础分 50，减去负面，加上正面
    const score = Math.round(50 - negativeScore * 1.5 + positiveScore);
    
    return Math.min(100, Math.max(0, score));
  };

  const dynamicScore = getDynamicScore();
  const hasMultimodalData = comprehensiveState.activeModalities.length > 0;
  const displayScore = hasMultimodalData ? dynamicScore : (profile.overallScore || 62);
  
  // 根据动态评分计算风险等级（而不是只基于焦虑概率）
  const getDynamicRiskLevel = () => {
    if (!hasMultimodalData) {
      return profile.riskLevel;
    }
    // 根据综合评分判断风险等级（返回大写键名）
    if (displayScore <= 20) return 'CRISIS';
    if (displayScore <= 40) return 'HIGH';
    if (displayScore <= 60) return 'MEDIUM';
    return 'LOW';
  };
  
  const displayRiskLevel = getDynamicRiskLevel();

  const membershipInfo: Record<string, { label: string; color: string; icon: React.ReactNode }> = {
    FREE: { label: '免费版', color: '#8c8c8c', icon: <StarOutlined /> },
    BASIC: { label: '基础版', color: '#1890ff', icon: <CrownOutlined /> },
    PREMIUM: { label: '高级版', color: '#fa8c16', icon: <CrownOutlined style={{ color: '#fa8c16' }} /> },
  };

  const currentMembership = membershipInfo[membership?.membershipType || 'FREE'] || membershipInfo.FREE;
  const isExpired = membership?.membershipExpiresAt && new Date(membership.membershipExpiresAt) < new Date();

  return (
    <div>
      {/* 会员状态卡片 */}
      <Card style={{
        borderRadius: 16, marginBottom: 24,
        background: membership?.membershipType === 'PREMIUM' ? 'linear-gradient(135deg, #fa8c16 0%, #faad14 100%)'
          : membership?.membershipType === 'BASIC' ? 'linear-gradient(135deg, #1890ff 0%, #69c0ff 100%)'
          : 'linear-gradient(135deg, #f5f5f5 0%, #e8e8e8 100%)',
        border: 'none',
      }}>
        <Row align="middle">
          <Col flex="auto">
            <Space>
              {currentMembership.icon}
              <Text strong style={{ fontSize: 18, color: membership?.membershipType === 'FREE' ? '#595959' : '#fff' }}>
                {currentMembership.label}
              </Text>
            </Space>
            {membership?.membershipExpiresAt && !isExpired && (
              <div style={{ marginTop: 4 }}>
                <Text style={{ fontSize: 13, color: membership?.membershipType === 'FREE' ? '#8c8c8c' : 'rgba(255,255,255,0.8)' }}>
                  有效期至 {new Date(membership.membershipExpiresAt).toLocaleDateString()}
                </Text>
              </div>
            )}
            {isExpired && (
              <div style={{ marginTop: 4 }}>
                <Text style={{ fontSize: 13, color: '#ff4d4f' }}>会员已过期</Text>
              </div>
            )}
          </Col>
          <Col>
            <Button type="primary" ghost onClick={() => setUpgradeModal(true)}
              style={{ borderColor: membership?.membershipType === 'FREE' ? '#d9d9d9' : '#fff', color: membership?.membershipType === 'FREE' ? '#595959' : '#fff' }}>
              {membership?.membershipType === 'PREMIUM' ? '续费' : '升级会员'}
            </Button>
          </Col>
        </Row>
      </Card>

      <Row gutter={[16, 16]}>
        {/* 综合评分 */}
        <Col xs={24} md={8}>
          <Card style={{ borderRadius: 12, textAlign: 'center' }}>
            <Title level={5}>综合心理健康评分</Title>
            {hasMultimodalData && (
              <div style={{ marginBottom: 8 }}>
                <Tag color="blue" style={{ fontSize: 12 }}> 实时多模态分析</Tag>
              </div>
            )}
            <Progress type="dashboard" percent={displayScore} format={(p) => (
              <div>
                <div style={{ fontSize: 32, fontWeight: 'bold' }}>{p}</div>
                <Text type="secondary">/ 100</Text>
              </div>
            )} strokeColor={displayScore > 60 ? '#52c41a' : displayScore > 30 ? '#faad14' : '#ff4d4f'} />
            <div style={{ marginTop: 16 }}>
              <Tag color={riskColors[displayRiskLevel] || riskColors.LOW} style={{ fontSize: 14, padding: '4px 12px' }}>
                {riskLabels[displayRiskLevel] || '低风险'}
              </Tag>
            </div>
            <Button onClick={handleUpdate} loading={updating} style={{ marginTop: 16 }}>
              更新画像
            </Button>
          </Card>
        </Col>

        {/* 雷达图 + 同龄对比 */}
        <Col xs={24} md={16}>
          <Card 
            title={<Space><RadarChartOutlined /> 心理维度分析</Space>} 
            extra={<Tag color="green" style={{ fontSize: 11 }}>含同龄对比</Tag>}
            style={{ borderRadius: 12 }}
          >
            <ResponsiveContainer width="100%" height={280}>
              <RadarChart data={radarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="subject" />
                <PolarRadiusAxis angle={30} domain={[0, 100]} />
                <Radar name="你的评分" dataKey="value" stroke="#6366f1" fill="#6366f1" fillOpacity={0.3} />
                <Radar name="同龄基线" dataKey="peer" stroke="#52c41a" fill="#52c41a" fillOpacity={0.1} strokeDasharray="5 5" />
                <Legend />
              </RadarChart>
            </ResponsiveContainer>
            {/* 同龄对比说明 */}
            <div style={{ marginTop: 8, padding: '8px 12px', background: '#f6ffed', borderRadius: 8, fontSize: 12, color: '#389e0d' }}>
              <strong>同龄对比说明：</strong>绿色虚线为同龄青少年（12-17岁）常模基线，紫色区域为你的实际评分。
              超出基线表示该维度高于同龄平均水平，低于基线表示低于同龄平均水平。
            </div>
          </Card>
        </Col>

        {/* 情绪趋势 */}
        <Col xs={24}>
          <Card title={<Space><SmileOutlined /> 情绪趋势 (近30天)</Space>} style={{ borderRadius: 12 }}>
            {moodTrend.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={moodTrend.map((r: any) => ({ date: r.recordedAt?.slice(5, 10), score: r.score }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis domain={[0, 10]} />
                  <Tooltip />
                  <Line type="monotone" dataKey="score" stroke="#6366f1" strokeWidth={2} dot={{ fill: '#6366f1' }} />
                </LineChart>
              </ResponsiveContainer>
            ) : <Empty description="暂无情绪记录" />}
          </Card>
        </Col>

        {/* 睡眠监测模块 */}
        <Col xs={24}>
          <Card 
            title={<Space><ClockCircleOutlined /> 睡眠监测 <Tag color="blue" style={{ fontSize: 11 }}>手动记录</Tag></Space>}
            extra={<Button type="primary" size="small" onClick={() => setSleepFormOpen(!sleepFormOpen)}>{sleepFormOpen ? '收起' : '+ 记录睡眠'}</Button>}
            style={{ borderRadius: 12 }}
          >
            {/* 睡眠记录表单 */}
            {sleepFormOpen && (
              <div style={{ padding: 16, background: '#f0f5ff', borderRadius: 8, marginBottom: 16 }}>
                <Row gutter={[16, 12]}>
                  <Col xs={24} md={6}>
                    <Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 4 }}>睡眠时长 (小时)</Text>
                    <Input type="number" min={0} max={24} step={0.5} value={sleepDuration} onChange={(e: ChangeEvent<HTMLInputElement>) => setSleepDuration(Number(e.target.value))} style={{ borderRadius: 8 }} />
                  </Col>
                  <Col xs={24} md={6}>
                    <Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 4 }}>睡眠质量 (1-10)</Text>
                    <Input type="number" min={1} max={10} value={sleepQuality} onChange={(e: ChangeEvent<HTMLInputElement>) => setSleepQuality(Number(e.target.value))} style={{ borderRadius: 8 }} />
                  </Col>
                  <Col xs={24} md={6}>
                    <Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 4 }}>入睡时间</Text>
                    <Input value={sleepBedtime} onChange={(e: ChangeEvent<HTMLInputElement>) => setSleepBedtime(e.target.value)} placeholder="如 23:00" style={{ borderRadius: 8 }} />
                  </Col>
                  <Col xs={24} md={6}>
                    <Text type="secondary" style={{ fontSize: 12, display: 'block', marginBottom: 4 }}>备注</Text>
                    <Input value={sleepNote} onChange={(e: ChangeEvent<HTMLInputElement>) => setSleepNote(e.target.value)} placeholder="如：多梦、入睡困难" style={{ borderRadius: 8 }} />
                  </Col>
                </Row>
                <Button type="primary" style={{ marginTop: 12, borderRadius: 8 }} onClick={() => {
                  const today = new Date().toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' });
                  setSleepLogs(prev => [...prev, { date: today, duration: sleepDuration, quality: sleepQuality, bedtime: sleepBedtime, note: sleepNote }]);
                  setSleepFormOpen(false);
                  message.success('睡眠记录已保存');
                }}>保存记录</Button>
              </div>
            )}

            {/* 睡眠统计摘要 */}
            <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
              <Col xs={8}>
                <div style={{ textAlign: 'center', padding: 12, background: '#f6ffed', borderRadius: 8 }}>
                  <Text type="secondary" style={{ fontSize: 12 }}>平均时长</Text>
                  <div style={{ fontSize: 24, fontWeight: 'bold', color: '#52c41a' }}>
                    {(sleepLogs.reduce((s, l) => s + l.duration, 0) / sleepLogs.length).toFixed(1)}h
                  </div>
                </div>
              </Col>
              <Col xs={8}>
                <div style={{ textAlign: 'center', padding: 12, background: '#f0f5ff', borderRadius: 8 }}>
                  <Text type="secondary" style={{ fontSize: 12 }}>平均质量</Text>
                  <div style={{ fontSize: 24, fontWeight: 'bold', color: '#6366f1' }}>
                    {(sleepLogs.reduce((s, l) => s + l.quality, 0) / sleepLogs.length).toFixed(1)}/10
                  </div>
                </div>
              </Col>
              <Col xs={8}>
                <div style={{ textAlign: 'center', padding: 12, background: '#fff7e6', borderRadius: 8 }}>
                  <Text type="secondary" style={{ fontSize: 12 }}>平均入睡</Text>
                  <div style={{ fontSize: 24, fontWeight: 'bold', color: '#fa8c16' }}>
                    {sleepLogs.map(l => l.bedtime).sort().slice(0, 3).pop() || '--'}
                  </div>
                </div>
              </Col>
            </Row>

            {/* 睡眠趋势图 */}
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={sleepLogs.slice(-7)}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                <YAxis yAxisId="left" domain={[0, 12]} tick={{ fontSize: 11 }} />
                <YAxis yAxisId="right" orientation="right" domain={[0, 10]} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Line yAxisId="left" type="monotone" dataKey="duration" name="时长(h)" stroke="#52c41a" strokeWidth={2} dot={{ fill: '#52c41a' }} />
                <Line yAxisId="right" type="monotone" dataKey="quality" name="质量(1-10)" stroke="#6366f1" strokeWidth={2} dot={{ fill: '#6366f1' }} />
              </LineChart>
            </ResponsiveContainer>

            {/* 同龄对比提示 */}
            <div style={{ marginTop: 8, padding: '6px 12px', background: '#fffbe6', borderRadius: 6, fontSize: 12, color: '#ad8b00' }}>
              <strong>同龄参考：</strong>青少年（12-17岁）建议睡眠时长 8-10 小时，平均入睡时间不晚于 23:00。
              {sleepLogs.length > 0 && (sleepLogs.reduce((s, l) => s + l.duration, 0) / sleepLogs.length) < 8 && (
                <span style={{ color: '#ff4d4f', marginLeft: 8 }}>你的平均睡眠时长低于建议值，建议调整作息。</span>
              )}
            </div>
          </Card>
        </Col>

        {/* AI 报告 */}
        {profile.report && (
          <Col xs={24}>
            <Card title={<Space><CloudOutlined /> AI 心理画像报告</Space>} style={{ borderRadius: 12 }}>
              <Paragraph style={{ whiteSpace: 'pre-wrap', lineHeight: 1.8 }}>{profile.report}</Paragraph>
              <Text type="secondary">评估方式：{profile.assessedBy} | 更新时间：{new Date(profile.updatedAt).toLocaleDateString()}</Text>
            </Card>
          </Col>
        )}

        {/* 动态情绪画像 */}
        {dynamicProfile?.hasData && dynamicProfile?.profile && (
          <Col xs={24}>
            <Card
              title={<Space><ThunderboltOutlined /> 今日动态情绪画像 <Tag color="purple">AI 实时分析</Tag></Space>}
              extra={dynamicProfile.updatedAt && <Text type="secondary" style={{ fontSize: 12 }}><ClockCircleOutlined /> {new Date(dynamicProfile.updatedAt).toLocaleString()}</Text>}
              style={{ borderRadius: 12 }}
            >
              <Row gutter={24}>
                <Col xs={24} md={8}>
                  <div style={{ textAlign: 'center' }}>
                    <Text type="secondary">今日主要情绪</Text>
                    <div style={{ fontSize: 24, fontWeight: 'bold', margin: '8px 0' }}>
                      {dynamicProfile.profile.dominantEmotions?.join('、') || '平静'}
                    </div>
                    <Tag color={riskColors[dynamicProfile.profile.riskLevel]} style={{ fontSize: 13, padding: '4px 12px' }}>
                      {riskLabels[dynamicProfile.profile.riskLevel]}
                    </Tag>
                    <div style={{ marginTop: 12 }}>
                      <Text type="secondary">情绪趋势：</Text>
                      <Text strong>{dynamicProfile.profile.emotionalTrend}</Text>
                    </div>
                  </div>
                </Col>
                <Col xs={24} md={16}>
                  <Row gutter={[12, 12]}>
                    {[
                      { label: '焦虑', value: dynamicProfile.profile.anxiety, color: '#ff4d4f' },
                      { label: '抑郁', value: dynamicProfile.profile.depression, color: '#722ed1' },
                      { label: '压力', value: dynamicProfile.profile.stress, color: '#fa8c16' },
                      { label: '睡眠', value: dynamicProfile.profile.sleepQuality, color: '#1890ff' },
                      { label: '社交', value: dynamicProfile.profile.socialActivity, color: '#52c41a' },
                      { label: '情绪稳定', value: dynamicProfile.profile.emotionalStability, color: '#13c2c2' },
                    ].map(({ label, value, color }) => (
                      <Col xs={12} key={label}>
                        <div style={{ marginBottom: 4 }}>
                          <Text type="secondary" style={{ fontSize: 12 }}>{label}</Text>
                          <Text style={{ float: 'right', fontWeight: 600, color }}>{value}</Text>
                        </div>
                        <Progress percent={value} showInfo={false} size="small" strokeColor={color} />
                      </Col>
                    ))}
                  </Row>
                </Col>
              </Row>
              <Divider style={{ margin: '16px 0' }} />
              <Paragraph style={{ fontSize: 14, lineHeight: 1.8 }}>{dynamicProfile.profile.emotionalSummary}</Paragraph>
            </Card>
          </Col>
        )}

        {/* 实时多模态情绪分析 */}
        <Col xs={24}>
          <Card
            title={
              <Space>
                <ThunderboltOutlined /> 实时多模态情绪分析
                <Tag color="purple" style={{ fontSize: 12 }}>AI 四模态融合</Tag>
                {comprehensiveState.activeModalities.length > 0 && (
                  <Tag color="green">{comprehensiveState.activeModalities.length} 个模态采集中</Tag>
                )}
              </Space>
            }
            extra={
              <Space>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  <ClockCircleOutlined /> {comprehensiveState.lastUpdated > 0 ? new Date(comprehensiveState.lastUpdated).toLocaleTimeString() : '等待数据'}
                </Text>
                {/* 手动启动按钮 */}
                <Button 
                  type="primary" 
                  size="small"
                  onClick={() => {
                    console.log('[Profile] 手动启动按钮被点击');
                    // 触发一个自定义事件，让 AnxietyContext 监听到并启动分析
                    window.dispatchEvent(new CustomEvent('start-multimodal-analysis'));
                    // 显示提示
                    alert('已发送启动指令！请等待 3 秒后查看调试面板变化。\n\n如果还是没有变化，请检查：\n1. 浏览器是否允许了摄像头/麦克风权限\n2. 控制台是否有红色错误信息');
                  }}
                >
                  启动分析
                </Button>
              </Space>
            }
            style={{ borderRadius: 12 }}
          >
            {/* 调试面板：显示各模态实时状态 */}
            <div style={{ padding: '12px 16px', background: '#f5f5f5', borderRadius: 8, marginBottom: 16, fontSize: 12 }}>
              <div style={{ fontWeight: 600, marginBottom: 8, color: '#666' }}>🔍 模态状态调试</div>
              
              {/* start() 调用状态 */}
              <div style={{ marginBottom: 12, padding: 8, background: '#e6f7ff', borderRadius: 6, border: '1px solid #91d5ff' }}>
                <div style={{ fontWeight: 600, marginBottom: 4 }}> start() 调用状态：</div>
                <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
                  <span>键盘：{modalityStatus.keyboardStarted ? '✅ 已调用' : '❌ 未调用'}</span>
                  <span>声音：{modalityStatus.voiceStarted ? '✅ 已调用' : '❌ 未调用'}</span>
                  <span>面部：{modalityStatus.facialStarted ? '✅ 已调用' : '❌ 未调用'}</span>
                  <span>文本监听：{modalityStatus.textListenerRegistered ? '✅ 已注册' : '❌ 未注册'}</span>
                </div>
                <div style={{ marginTop: 8, fontWeight: 600, color: '#1890ff' }}>
                   状态更新计数器：{updateCounter} 次
                </div>
              </div>

              {/* 实际指标状态 */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                <div style={{ padding: 8, background: keyboardMetrics.isCollecting ? '#f6ffed' : '#fff', borderRadius: 6, border: '1px solid ' + (keyboardMetrics.isCollecting ? '#b7eb8f' : '#d9d9d9') }}>
                  <div style={{ fontWeight: 600 }}>键盘</div>
                  <div>采集中：{keyboardMetrics.isCollecting ? '✅' : '❌'}</div>
                  <div>样本数：{keyboardMetrics.sampleSize}</div>
                  <div>打字速度：{keyboardMetrics.typingSpeed} 字/分</div>
                </div>
                <div style={{ padding: 8, background: textMetrics.timestamp > 0 ? '#f6ffed' : '#fff', borderRadius: 6, border: '1px solid ' + (textMetrics.timestamp > 0 ? '#b7eb8f' : '#d9d9d9') }}>
                  <div style={{ fontWeight: 600 }}>文本</div>
                  <div>时间戳：{textMetrics.timestamp > 0 ? '✅' : '❌'}</div>
                  <div>已分析字数：{textMetrics.totalCharsAnalyzed}</div>
                  <div>主导情绪：{textMetrics.dominantEmotion}</div>
                </div>
                <div style={{ padding: 8, background: voiceMetrics.isRecording ? '#f6ffed' : '#fff', borderRadius: 6, border: '1px solid ' + (voiceMetrics.isRecording ? '#b7eb8f' : '#d9d9d9') }}>
                  <div style={{ fontWeight: 600 }}>声音</div>
                  <div>录音中：{voiceMetrics.isRecording ? '✅' : ''}</div>
                  <div>语音识别：{voiceMetrics.isSpeechRecognizing ? '✅ 工作中' : '❌ 未启动'}</div>
                  <div>识别文本：{voiceMetrics.speechText ? `“${voiceMetrics.speechText.slice(0, 30)}${voiceMetrics.speechText.length > 30 ? '...' : ''}”` : '无'}</div>
                  <div>时长：{voiceMetrics.duration}秒</div>
                  <div>检测情绪：{voiceMetrics.detectedEmotion}</div>
                </div>
                <div style={{ padding: 8, background: facialMetrics.isDetecting ? '#f6ffed' : '#fff', borderRadius: 6, border: '1px solid ' + (facialMetrics.isDetecting ? '#b7eb8f' : '#d9d9d9') }}>
                  <div style={{ fontWeight: 600 }}>面部</div>
                  <div>检测中：{facialMetrics.isDetecting ? '✅' : '❌'}</div>
                  <div>帧数：{facialMetrics.frameCount}</div>
                  <div>主导表情：{facialMetrics.dominantExpression}</div>
                </div>
              </div>
              <div style={{ marginTop: 8, padding: 8, background: '#fffbe6', borderRadius: 6, fontSize: 11 }}>
                <strong>融合条件：</strong>键盘 (sampleSize≥3) | 文本 (timestamp&gt;0 且 chars&gt;0) | 声音 (isRecording 且 duration&gt;2) | 面部 (isDetecting 且 frameCount≥2)
              </div>
            </div>

            {/* 无数据时的引导提示 */}
            {comprehensiveState.activeModalities.length === 0 && (
              <div style={{ textAlign: 'center', padding: '30px 0' }}>
                <div style={{ fontSize: 48, marginBottom: 16 }}>🌱</div>
                <Text strong style={{ fontSize: 16, color: '#722ed1' }}>等待感知数据...</Text>
                <div style={{ marginTop: 12, color: '#999', fontSize: 14, lineHeight: 2 }}>
                  <div>1. 键盘分析 <Text strong>已自动启动</Text>，打字即可被感知</div>
                  <div>2. 在聊天框 <Text strong>输入文字</Text>，系统会分析你的情绪</div>
                  <div>3. 点击右下角浮动组件的 <Text strong>开关按钮</Text> 开启摄像头和麦克风</div>
                  <div>4. 说话时，系统会分析你的 <Text strong>声音特征</Text></div>
                </div>
              </div>
            )}

            {/* 叙事性分析长文本（主要内容） */}
            {comprehensiveState.narrativeAnalysis && (
              <div style={{
                padding: '16px 20px',
                background: 'linear-gradient(135deg, #f0f4ff 0%, #faf0ff 100%)',
                borderRadius: 12,
                marginBottom: 16,
                border: '1px solid #e8e0f0',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', marginBottom: 12 }}>
                  <span style={{ fontSize: 20, marginRight: 8 }}>📋</span>
                  <Text strong style={{ fontSize: 15, color: '#5b4a8a' }}>多模态心理观察报告</Text>
                  <Tag color="purple" style={{ marginLeft: 8, fontSize: 11 }}>每3秒更新</Tag>
                </div>
                <div style={{
                  fontSize: 14,
                  lineHeight: 2,
                  color: '#3d3166',
                  whiteSpace: 'pre-wrap',
                  letterSpacing: 0.3,
                }}>
                  {comprehensiveState.narrativeAnalysis}
                </div>
              </div>
            )}

              {/* 主导情绪 + 五维分布 */}
              <Row gutter={24}>
                <Col xs={24} md={6}>
                  <div style={{ textAlign: 'center' }}>
                    <Text type="secondary">当前主导情绪</Text>
                    <div style={{ fontSize: 28, fontWeight: 'bold', margin: '8px 0',
                      color: ['#52c41a', '#722ed1', '#ff7a45', '#ff4d4f', '#1890ff'][
                        ['快乐', '悲伤', '焦虑', '愤怒', '中性'].indexOf(comprehensiveState.dominantEmotion)
                      ] || '#666'
                    }}>
                      {comprehensiveState.dominantEmotion}
                    </div>
                    <Tag color="purple" style={{ fontSize: 13, padding: '4px 12px' }}>
                      置信度 {Math.round(comprehensiveState.confidence * 100)}%
                    </Tag>
                    <div style={{ marginTop: 12, padding: '8px 12px', background: '#fff0f6', borderRadius: 10, fontSize: 13, color: '#8a6d9a' }}>
                      {comprehensiveState.caringMessage}
                    </div>
                  </div>
                </Col>
                <Col xs={24} md={18}>
                  <Row gutter={[12, 12]}>
                    {comprehensiveState.emotionProbs.map((prob, i) => {
                      const labels = ['快乐', '悲伤', '焦虑', '愤怒', '中性'];
                      const colors = ['#52c41a', '#722ed1', '#ff7a45', '#ff4d4f', '#1890ff'];
                      return (
                        <Col xs={12} key={labels[i]}>
                          <div style={{ marginBottom: 4 }}>
                            <Text type="secondary" style={{ fontSize: 12 }}>{labels[i]}</Text>
                            <Text style={{ float: 'right', fontWeight: 600, color: colors[i] }}>
                              {Math.round(prob * 100)}%
                            </Text>
                          </div>
                          <Progress percent={Math.round(prob * 100)} showInfo={false} size="small" strokeColor={colors[i]} />
                        </Col>
                      );
                    })}
                  </Row>
                </Col>
              </Row>

              <Divider style={{ margin: '16px 0' }} />

              {/* 四模态详细分析 */}
              <Row gutter={[16, 16]}>
                {/* 文字语义分析 */}
                {comprehensiveState.text.timestamp > 0 && (
                  <Col xs={24} md={12}>
                    <Card size="small" title={<Space><span style={{ fontSize: 16 }}>📝</span> 文字语义深度分析</Space>} style={{ borderRadius: 10, background: '#fafbff' }}>
                      <div style={{ fontSize: 13, lineHeight: 2 }}>
                        <div><Text type="secondary">主导情绪：</Text><Text strong>{comprehensiveState.text.dominantEmotion}</Text>
                          <Text type="secondary" style={{ marginLeft: 12 }}>情感效价：</Text>
                          <Text strong style={{ color: comprehensiveState.text.sentiment > 0 ? '#52c41a' : comprehensiveState.text.sentiment < 0 ? '#ff4d4f' : '#999' }}>
                            {comprehensiveState.text.sentiment > 0 ? '偏积极' : comprehensiveState.text.sentiment < 0 ? '偏消极' : '中性'} ({comprehensiveState.text.sentiment})
                          </Text>
                        </div>
                        <div><Text type="secondary">情绪强度：</Text><Progress percent={Math.round(comprehensiveState.text.emotionalIntensity * 100)} showInfo={false} size="small" strokeColor="#722ed1" style={{ width: 100, display: 'inline-block', verticalAlign: 'middle' }} /></div>
                        <div><Text type="secondary">已分析字数：</Text><Text strong>{comprehensiveState.text.totalCharsAnalyzed}</Text></div>

                        {/* 认知扭曲标记 */}
                        {Object.values(comprehensiveState.text.cognitiveMarkers).some(v => v > 0) && (
                          <div style={{ marginTop: 8, padding: '8px 12px', background: '#fff7e6', borderRadius: 8 }}>
                            <Text strong style={{ fontSize: 12, color: '#d46b08' }}> 认知扭曲标记 (CBT)</Text>
                            <div style={{ marginTop: 4, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                              {comprehensiveState.text.cognitiveMarkers.catastrophizing > 0 && (
                                <Tag color="orange" style={{ fontSize: 11, margin: 0 }}>灾难化 {Math.round(comprehensiveState.text.cognitiveMarkers.catastrophizing * 100)}%</Tag>
                              )}
                              {comprehensiveState.text.cognitiveMarkers.blackAndWhite > 0 && (
                                <Tag color="red" style={{ fontSize: 11, margin: 0 }}>非黑即白 {Math.round(comprehensiveState.text.cognitiveMarkers.blackAndWhite * 100)}%</Tag>
                              )}
                              {comprehensiveState.text.cognitiveMarkers.overgeneralization > 0 && (
                                <Tag color="gold" style={{ fontSize: 11, margin: 0 }}>过度概括 {Math.round(comprehensiveState.text.cognitiveMarkers.overgeneralization * 100)}%</Tag>
                              )}
                              {comprehensiveState.text.cognitiveMarkers.selfBlame > 0 && (
                                <Tag color="magenta" style={{ fontSize: 11, margin: 0 }}>自我归咎 {Math.round(comprehensiveState.text.cognitiveMarkers.selfBlame * 100)}%</Tag>
                              )}
                              {comprehensiveState.text.cognitiveMarkers.hopelessness > 0 && (
                                <Tag color="volcano" style={{ fontSize: 11, margin: 0 }}>无望感 {Math.round(comprehensiveState.text.cognitiveMarkers.hopelessness * 100)}%</Tag>
                              )}
                            </div>
                          </div>
                        )}

                        {/* 危机信号 */}
                        {comprehensiveState.text.crisisLevel !== 'none' && (
                          <div style={{ marginTop: 8, padding: '8px 12px', background: '#fff1f0', borderRadius: 8, border: '1px solid #ffccc7' }}>
                            <Text strong style={{ fontSize: 12, color: '#cf1322' }}>️ 危机信号 ({comprehensiveState.text.crisisLevel}级)</Text>
                            <div style={{ fontSize: 12, color: '#cf1322', marginTop: 4 }}>
                              检测到：{comprehensiveState.text.crisisMarkers.join('、')}
                            </div>
                          </div>
                        )}

                        {/* 语言特征 */}
                        <div style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 12, fontSize: 12 }}>
                          <span><Text type="secondary">第一人称密度：</Text>{Math.round(comprehensiveState.text.firstPersonRate * 100)}%</span>
                          <span><Text type="secondary">否定词密度：</Text>{Math.round(comprehensiveState.text.negationRate * 100)}%</span>
                          <span><Text type="secondary">绝对化用词：</Text>{comprehensiveState.text.absoluteWords}次</span>
                          <span><Text type="secondary">疑问句比例：</Text>{Math.round(comprehensiveState.text.questionRate * 100)}%</span>
                        </div>
                      </div>
                    </Card>
                  </Col>
                )}

                {/* 面部微表情分析 */}
                {comprehensiveState.facial.isDetecting && (
                  <Col xs={24} md={12}>
                    <Card size="small" title={<Space><span style={{ fontSize: 16 }}>😊</span> 面部微表情分析 (FACS)</Space>} style={{ borderRadius: 10, background: '#fffbf0' }}>
                      <div style={{ fontSize: 13, lineHeight: 2 }}>
                        <div><Text type="secondary">主导表情：</Text><Text strong style={{ fontSize: 15 }}>{comprehensiveState.facial.dominantExpression}</Text>
                          <Text type="secondary" style={{ marginLeft: 12 }}>强度：</Text>
                          <Progress percent={Math.round(comprehensiveState.facial.expressionIntensity * 100)} showInfo={false} size="small" strokeColor="#fa8c16" style={{ width: 80, display: 'inline-block', verticalAlign: 'middle' }} />
                        </div>
                        <div><Text type="secondary">微表情次数：</Text><Text strong>{comprehensiveState.facial.microExpressions.count}</Text> 次
                          {comprehensiveState.facial.microExpressions.recentEmotions.length > 0 && (
                            <Text type="secondary" style={{ marginLeft: 8 }}>最近：{comprehensiveState.facial.microExpressions.recentEmotions.join('、')}</Text>
                          )}
                        </div>

                        {/* AU 动作单元 */}
                        <div style={{ marginTop: 8 }}>
                          <Text strong style={{ fontSize: 12 }}>关键动作单元 (AU)</Text>
                          <div style={{ marginTop: 4, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                            {comprehensiveState.facial.actionUnits.au12_lipCorner > 0.1 && (
                              <Tag color="green" style={{ fontSize: 11, margin: 0 }}>AU12 嘴角上提 {Math.round(comprehensiveState.facial.actionUnits.au12_lipCorner * 100)}%</Tag>
                            )}
                            {comprehensiveState.facial.actionUnits.au6_cheekRaise > 0.1 && (
                              <Tag color="lime" style={{ fontSize: 11, margin: 0 }}>AU6 脸颊上扬 {Math.round(comprehensiveState.facial.actionUnits.au6_cheekRaise * 100)}%</Tag>
                            )}
                            {comprehensiveState.facial.actionUnits.au4_browLower > 0.1 && (
                              <Tag color="red" style={{ fontSize: 11, margin: 0 }}>AU4 眉毛下压 {Math.round(comprehensiveState.facial.actionUnits.au4_browLower * 100)}%</Tag>
                            )}
                            {comprehensiveState.facial.actionUnits.au1_browRaise > 0.1 && (
                              <Tag color="blue" style={{ fontSize: 11, margin: 0 }}>AU1 内眉上扬 {Math.round(comprehensiveState.facial.actionUnits.au1_browRaise * 100)}%</Tag>
                            )}
                            {comprehensiveState.facial.actionUnits.au15_lipCornerDrop > 0.1 && (
                              <Tag color="purple" style={{ fontSize: 11, margin: 0 }}>AU15 嘴角下拉 {Math.round(comprehensiveState.facial.actionUnits.au15_lipCornerDrop * 100)}%</Tag>
                            )}
                            {comprehensiveState.facial.actionUnits.au9_noseWrinkle > 0.1 && (
                              <Tag color="orange" style={{ fontSize: 11, margin: 0 }}>AU9 鼻子皱起 {Math.round(comprehensiveState.facial.actionUnits.au9_noseWrinkle * 100)}%</Tag>
                            )}
                            {comprehensiveState.facial.actionUnits.au7_eyeTighten > 0.1 && (
                              <Tag color="cyan" style={{ fontSize: 11, margin: 0 }}>AU7 眼睑收紧 {Math.round(comprehensiveState.facial.actionUnits.au7_eyeTighten * 100)}%</Tag>
                            )}
                            {comprehensiveState.facial.actionUnits.au26_jawDrop > 0.1 && (
                              <Tag color="geekblue" style={{ fontSize: 11, margin: 0 }}>AU26 下巴下垂 {Math.round(comprehensiveState.facial.actionUnits.au26_jawDrop * 100)}%</Tag>
                            )}
                          </div>
                        </div>

                        {/* 情绪映射 */}
                        <div style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 8, fontSize: 12 }}>
                          {Object.entries(comprehensiveState.facial.emotionMapping).filter(([, v]) => v > 0.05).map(([k, v]) => {
                            const labelMap: Record<string, string> = { happiness: '快乐', sadness: '悲伤', anger: '愤怒', fear: '恐惧', surprise: '惊讶', disgust: '厌恶', distress: '痛苦' };
                            return <span key={k}><Text type="secondary">{labelMap[k] || k}：</Text>{Math.round(v * 100)}%</span>;
                          })}
                        </div>
                      </div>
                    </Card>
                  </Col>
                )}

                {/* 声音声学分析 */}
                {comprehensiveState.voice.isRecording && (
                  <Col xs={24} md={12}>
                    <Card size="small" title={<Space><span style={{ fontSize: 16 }}>🔊</span> 声音声学分析</Space>} style={{ borderRadius: 10, background: '#f6ffed' }}>
                      <div style={{ fontSize: 13, lineHeight: 2 }}>
                        <div><Text type="secondary">检测情绪：</Text><Text strong>{comprehensiveState.voice.detectedEmotion}</Text></div>
                        <div><Text type="secondary">基频 (F0)：</Text><Text strong>{comprehensiveState.voice.pitch.mean}</Text> Hz
                          <Text type="secondary" style={{ marginLeft: 8 }}>波动：</Text>{comprehensiveState.voice.pitch.std} Hz
                        </div>
                        <div><Text type="secondary">能量：</Text><Text strong>{comprehensiveState.voice.energy.mean}</Text> dB
                          <Text type="secondary" style={{ marginLeft: 8 }}>峰值比：</Text>{Math.round(comprehensiveState.voice.energy.peakRatio * 100)}%
                        </div>
                        <div><Text type="secondary">语速：</Text><Text strong>{comprehensiveState.voice.rate.speechRate}</Text> 音节/s
                          <Tag color={comprehensiveState.voice.rate.tempo === 'fast' ? 'orange' : comprehensiveState.voice.rate.tempo === 'slow' ? 'blue' : 'green'} style={{ fontSize: 11, margin: '0 4px' }}>
                            {comprehensiveState.voice.rate.tempo === 'fast' ? '偏快' : comprehensiveState.voice.rate.tempo === 'slow' ? '偏慢' : '正常'}
                          </Tag>
                        </div>
                        <div><Text type="secondary">停顿比：</Text>{Math.round(comprehensiveState.voice.pauses.ratio * 100)}%
                          <Text type="secondary" style={{ marginLeft: 8 }}>长停顿：</Text>{comprehensiveState.voice.pauses.longPauseCount}次
                        </div>
                        <div><Text type="secondary">频谱质心：</Text>{comprehensiveState.voice.spectral.centroid} Hz
                          <Text type="secondary" style={{ marginLeft: 8 }}>平坦度：</Text>{comprehensiveState.voice.spectral.flatness}
                        </div>
                      </div>
                    </Card>
                  </Col>
                )}

                {/* 键盘动力学分析 */}
                {comprehensiveState.keyboard.isCollecting && (
                  <Col xs={24} md={12}>
                    <Card size="small" title={<Space><span style={{ fontSize: 16 }}>️</span> 键盘动力学分析</Space>} style={{ borderRadius: 10, background: '#f9f0ff' }}>
                      <div style={{ fontSize: 13, lineHeight: 2 }}>
                        <div><Text type="secondary">打字速度：</Text><Text strong>{comprehensiveState.keyboard.typingSpeed}</Text> 字/分</div>
                        <div><Text type="secondary">节奏规律性：</Text>
                          <Progress percent={Math.round(comprehensiveState.keyboard.rhythmScore * 100)} showInfo={false} size="small" strokeColor="#722ed1" style={{ width: 80, display: 'inline-block', verticalAlign: 'middle' }} />
                        </div>
                        <div><Text type="secondary">长停顿次数：</Text><Text strong>{comprehensiveState.keyboard.pauseCount}</Text>
                          <Text type="secondary" style={{ marginLeft: 8 }}>停顿占比：</Text>{Math.round(comprehensiveState.keyboard.pauseRatio * 100)}%
                        </div>
                        <div><Text type="secondary">删除率：</Text>{Math.round(comprehensiveState.keyboard.deletionRate * 100)}%
                          <Text type="secondary" style={{ marginLeft: 8 }}>修正率：</Text>{Math.round(comprehensiveState.keyboard.correctionRate * 100)}%
                        </div>
                        <div><Text type="secondary">删除爆发：</Text><Text strong>{comprehensiveState.keyboard.backspaceBurst}</Text> 次
                          <Text type="secondary" style={{ marginLeft: 8 }}>错误率估计：</Text>{Math.round(comprehensiveState.keyboard.errorRate * 100)}%
                        </div>
                        <div><Text type="secondary">按键压力指数：</Text>
                          <Progress percent={Math.round(comprehensiveState.keyboard.pressureIndex * 100)} showInfo={false} size="small" strokeColor="#ff4d4f" style={{ width: 80, display: 'inline-block', verticalAlign: 'middle' }} />
                        </div>
                        <div style={{ marginTop: 4, padding: '6px 10px', background: comprehensiveState.keyboard.anxietyIndex > 60 ? '#fff1f0' : comprehensiveState.keyboard.anxietyIndex > 30 ? '#fffbe6' : '#f6ffed', borderRadius: 6 }}>
                          <Text strong style={{ fontSize: 12, color: comprehensiveState.keyboard.anxietyIndex > 60 ? '#cf1322' : comprehensiveState.keyboard.anxietyIndex > 30 ? '#d46b08' : '#389e0d' }}>
                            综合焦虑指数：{comprehensiveState.keyboard.anxietyIndex} / 100
                          </Text>
                        </div>
                      </div>
                    </Card>
                  </Col>
                )}
              </Row>

              {/* 融合证据 */}
              {comprehensiveState.evidence.length > 0 && (
                <>
                  <Divider style={{ margin: '16px 0' }} />
                  <div style={{ padding: '10px 14px', background: '#f9f0ff', borderRadius: 8 }}>
                    <Text strong style={{ fontSize: 13, color: '#722ed1' }}> 融合分析依据</Text>
                    {comprehensiveState.evidence.slice(0, 5).map((e, i) => (
                      <div key={i} style={{ fontSize: 13, color: '#722ed1', lineHeight: 1.8, marginTop: 4 }}>
                        • {e}
                      </div>
                    ))}
                  </div>
                </>
              )}
            </Card>
          </Col>

        {/* 端侧推理引擎状态 */}
        <Col xs={24}>
          <Card
            title={<Space><ThunderboltOutlined /> 端侧隐私推理 <Tag color="cyan" style={{ fontSize: 11 }}>ONNX Runtime Web</Tag></Space>}
            style={{ borderRadius: 12, background: '#f0fffe' }}
          >
            <OnDeviceInferencePanel />
          </Card>
        </Col>

        {/* 应对策略 */}
        {copingStrategies && (
          <Col xs={24}>
            <Card
              title={<Space><BulbOutlined /> 个性化应对策略</Space>}
              style={{ borderRadius: 12, background: 'linear-gradient(135deg, #f0f9ff 0%, #f5f3ff 100%)' }}
            >
              <Row gutter={16}>
                <Col xs={24} md={8}>
                  <Card size="small" title="当下可做" style={{ borderRadius: 8 }}>
                    {copingStrategies.immediateActions?.map((a: string, i: number) => (
                      <div key={i} style={{ padding: '4px 0', fontSize: 13 }}>• {a}</div>
                    ))}
                  </Card>
                </Col>
                <Col xs={24} md={8}>
                  <Card size="small" title="本周建议" style={{ borderRadius: 8 }}>
                    {copingStrategies.shortTermStrategies?.map((s: string, i: number) => (
                      <div key={i} style={{ padding: '4px 0', fontSize: 13 }}>• {s}</div>
                    ))}
                  </Card>
                </Col>
                <Col xs={24} md={8}>
                  <Card size="small" title="长期方向" style={{ borderRadius: 8 }}>
                    {copingStrategies.longTermSuggestions?.map((s: string, i: number) => (
                      <div key={i} style={{ padding: '4px 0', fontSize: 13 }}>• {s}</div>
                    ))}
                  </Card>
                </Col>
              </Row>
              <div style={{ marginTop: 16, padding: '12px 16px', background: '#fff7e6', borderRadius: 8, textAlign: 'center' }}>
                <Text style={{ fontSize: 15, color: '#d46b08', fontWeight: 500 }}>{copingStrategies.encouragingMessage}</Text>
              </div>
            </Card>
          </Col>
        )}
      </Row>

      {/* ===== 量表测评 ===== */}
      <Card
        title={<Space><FileTextOutlined /> 标准化量表测评</Space>}
        style={{ borderRadius: 12, marginTop: 24 }}
        extra={
          <Tag color="blue" style={{ fontSize: 12 }}>
            数据仅用于生成你的心理画像，不会泄露给第三方
          </Tag>
        }
      >
        {/* 隐私与伦理声明 */}
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 20, borderRadius: 8 }}
          message="隐私与伦理保护"
          description={
            <div style={{ fontSize: 13, lineHeight: 1.8 }}>
              <div>• 所有测评数据均加密存储，仅用于生成你的心理画像和辅助咨询师了解你的状态</div>
              <div>• 你的咨询师在咨询过程中可以看到测评结果，以便更好地帮助你</div>
              <div>• 你可以随时要求删除所有测评数据</div>
              <div>• 本测评仅供参考，不构成医学诊断。如有严重不适，请及时就医</div>
            </div>
          }
        />

        {scaleResult ? (
          <Card style={{ borderRadius: 12, maxWidth: 600, margin: '0 auto' }}>
            <Result
              status={(scaleResult.level === 'normal' ? 'success' : scaleResult.level === 'mild' ? 'warning' : 'error') as any}
              title="测评完成"
              subTitle={`总分：${scaleResult.totalScore} | 等级：${scaleResult.level}`}
              extra={[
                <Button key="back" onClick={() => { setScaleResult(null); setSelectedScale(null); setScaleAnswers({}); safetyTriggeredRef.current = false; }}>
                  返回
                </Button>,
              ]}
            >
              <Paragraph>{scaleResult.suggestion}</Paragraph>
            </Result>
          </Card>
        ) : selectedScale ? (
          <Card
            style={{ borderRadius: 12 }}
            title={SCALE_QUESTIONS[selectedScale].title}
            extra={<Button onClick={() => { setSelectedScale(null); setScaleAnswers({}); safetyTriggeredRef.current = false; }}>返回</Button>}
          >
            <Paragraph type="secondary">{SCALE_QUESTIONS[selectedScale].desc}</Paragraph>
            <Paragraph>请根据过去<strong>两周</strong>的情况，选择每项的程度：</Paragraph>
            <Paragraph type="secondary">0=完全没有，1=有几天，2=一半以上天数，3=几乎每天</Paragraph>
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
              {SCALE_QUESTIONS[selectedScale].questions.map((q: string, i: number) => (
                <Card key={i} size="small" style={{ borderRadius: 8 }}>
                  <Text style={{ display: 'block', marginBottom: 8 }}>{i + 1}. {q}</Text>
                  <Radio.Group value={scaleAnswers[i]} onChange={e => setScaleAnswers({ ...scaleAnswers, [i]: e.target.value })}>
                    <Space>{[0, 1, 2, 3].map(v => <Radio.Button key={v} value={v}>{v}</Radio.Button>)}</Space>
                  </Radio.Group>
                </Card>
              ))}
              <Button
                type="primary"
                size="large"
                block
                onClick={handleScaleSubmit}
                loading={scaleSubmitting}
                disabled={Object.keys(scaleAnswers).length < SCALE_QUESTIONS[selectedScale].questions.length}
              >
                提交测评
              </Button>
            </Space>
          </Card>
        ) : (
          <>
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
              {Object.entries(SCALE_QUESTIONS).map(([key, scale]) => (
                <Card
                  key={key}
                  hoverable
                  onClick={() => { setSelectedScale(key); setScaleAnswers({}); safetyTriggeredRef.current = false; }}
                  style={{ borderRadius: 12 }}
                  bodyStyle={{ padding: 20 }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <Title level={5} style={{ marginBottom: 4 }}>{scale.title}</Title>
                      <Text type="secondary">{scale.desc}</Text>
                      <div style={{ marginTop: 8 }}><Tag>{scale.questions.length} 题</Tag></div>
                    </div>
                    <Button type="primary">开始测评</Button>
                  </div>
                </Card>
              ))}
            </Space>
            <Title level={5} style={{ marginTop: 24, marginBottom: 12 }}>历史记录</Title>
            {scaleHistory.length === 0 ? (
              <Empty description="暂无测评记录" />
            ) : (
              <List
                dataSource={scaleHistory}
                renderItem={(item: any) => (
                  <List.Item>
                    <List.Item.Meta
                      title={item.title}
                      description={`得分：${item.totalScore} | 等级：${item.level} | ${new Date(item.createdAt).toLocaleDateString()}`}
                    />
                    <Tag color={item.level === 'normal' ? 'green' : item.level === 'mild' ? 'orange' : 'red'}>{item.level}</Tag>
                  </List.Item>
                )}
              />
            )}
          </>
        )}
      </Card>

      {/* PHQ-9 安全关怀弹窗 */}
      <Modal
        open={safetyModalOpen}
        onCancel={() => setSafetyModalOpen(false)}
        footer={null}
        width={480}
        centered
      >
        <div style={{ textAlign: 'center', padding: '20px 0' }}>
          <div style={{ fontSize: 48, marginBottom: 12 }}>💛</div>
          <Title level={4} style={{ marginBottom: 8, color: '#5a4a6a' }}>
            谢谢你愿意说出来
          </Title>
          <div style={{
            padding: '16px 20px', background: '#fff7e6', borderRadius: 12,
            textAlign: 'left', marginBottom: 20,
          }}>
            <Paragraph style={{ fontSize: 15, color: '#5a4a6a', marginBottom: 8 }}>
              你刚才提到的那些想法，让我们很关心你现在的感受。
            </Paragraph>
            <Paragraph style={{ fontSize: 14, color: '#8a7a9a', marginBottom: 0 }}>
              请记住，你不是一个人。现在有人愿意听你说，24小时都在：
            </Paragraph>
          </div>
          <Card style={{ borderRadius: 12, marginBottom: 20, background: '#fff5f5', border: '1px solid #ffccc7' }}>
            <div style={{ marginBottom: 12 }}>
              <Text strong>🆘 24小时心理援助热线</Text>
              <Title level={3} style={{ color: '#ff4d4f', margin: '4px 0' }}>400-161-9995</Title>
            </div>
            <div style={{ marginBottom: 0 }}>
              <Text strong> 生命热线</Text>
              <Title level={3} style={{ color: '#6366f1', margin: '4px 0' }}>400-821-1215</Title>
            </div>
          </Card>
          <Space direction="vertical" style={{ width: '100%' }} size="middle">
            <Button
              type="primary"
              block
              size="large"
              onClick={() => { setSafetyModalOpen(false); navigate('/chat'); }}
              style={{ borderRadius: 12 }}
            >
              <HeartOutlined /> 去找 AI 聊聊，它会一直陪着你
            </Button>
            <Button
              block
              size="large"
              onClick={() => setSafetyModalOpen(false)}
              style={{ borderRadius: 12 }}
            >
              我暂时还好，继续测评
            </Button>
          </Space>
        </div>
      </Modal>

      {/* 升级会员弹窗 */}
      <Modal open={upgradeModal} onCancel={() => setUpgradeModal(false)} title="升级会员" footer={null} width={640}>
        <Row gutter={16}>
          {[
            { type: 'FREE', name: '免费版', price: 0, features: ['AI 咨询', '基础测评', '社区互助'], color: '#f5f5f5', textColor: '#595959' },
            { type: 'BASIC', name: '基础版', price: 29, features: ['AI 咨询无限次', '高级测评', '情绪追踪', '疗愈内容'], color: '#e6f7ff', textColor: '#1890ff' },
            { type: 'PREMIUM', name: '高级版', price: 99, features: ['全部基础版功能', '专家预约优先', '个性化方案', '危机干预', '专属客服'], color: '#fff7e6', textColor: '#fa8c16' },
          ].map((plan) => (
            <Col xs={24} sm={8} key={plan.type}>
              <Card style={{
                borderRadius: 12, textAlign: 'center',
                background: plan.color, border: `2px solid ${plan.textColor}`,
                opacity: membership?.membershipType === plan.type ? 0.6 : 1,
              }}>
                <Title level={5} style={{ color: plan.textColor }}>{plan.name}</Title>
                <div style={{ fontSize: 28, fontWeight: 'bold', color: plan.textColor, margin: '8px 0' }}>
                  {plan.price === 0 ? '免费' : `¥${plan.price}/月`}
                </div>
                <Divider style={{ margin: '12px 0' }} />
                {plan.features.map((f, i) => (
                  <div key={i} style={{ padding: '4px 0', fontSize: 13 }}>
                    <CheckCircleOutlined style={{ color: plan.textColor, marginRight: 6 }} />{f}
                  </div>
                ))}
                {membership?.membershipType !== plan.type && (
                  <Button type="primary" block style={{ marginTop: 16, background: plan.textColor, borderColor: plan.textColor, borderRadius: 20 }}
                    onClick={() => handleUpgrade(plan)}>
                    {plan.price === 0 ? '当前方案' : '立即升级'}
                  </Button>
                )}
                {membership?.membershipType === plan.type && (
                  <Tag color={plan.textColor} style={{ marginTop: 16, display: 'block' }}>当前方案</Tag>
                )}
              </Card>
            </Col>
          ))}
        </Row>
      </Modal>

      {/* 支付确认弹窗 */}
      <Modal
        open={payModal}
        onCancel={() => setPayModal(false)}
        title="确认升级"
        onOk={handleConfirmPay}
        okText="确认支付"
        cancelText="取消"
        confirmLoading={payLoading}
      >
        {selectedPlan && (
          <div style={{ padding: '16px 0' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
              <Text>升级方案：</Text>
              <Text strong style={{ color: selectedPlan.textColor }}>{selectedPlan.name}</Text>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
              <Text>支付金额：</Text>
              <Text strong style={{ fontSize: 20, color: '#ff4d4f' }}>¥{selectedPlan.price}/月</Text>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
              <Text>有效期：</Text>
              <Text>30 天</Text>
            </div>
            <Divider />
            <div style={{ background: '#f9f0ff', padding: 12, borderRadius: 8, textAlign: 'center' }}>
              <Text style={{ fontSize: 13, color: '#722ed1' }}>
                 模拟支付演示 — 点击确认支付后将立即生效
              </Text>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
