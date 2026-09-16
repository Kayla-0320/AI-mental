import { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Card, Row, Col, Typography, Tag, Space, Avatar, Progress, Spin, Divider, Button, Input, Empty, Collapse } from 'antd';
import {
  ArrowLeftOutlined, ThunderboltOutlined, ClockCircleOutlined,
  SendOutlined, MessageOutlined, UserOutlined,
  PhoneOutlined, VideoCameraOutlined, SoundOutlined, StopOutlined,
  ExperimentOutlined, DashboardOutlined,
} from '@ant-design/icons';
import { useAnxiety } from '../../context/AnxietyContext';
import api from '../../services/api';

const { Title, Text, Paragraph } = Typography;

const anxietyLevelColors: Record<string, string> = { low: '#52c41a', medium: '#faad14', high: '#ff4d4f' };
const anxietyLevelLabels: Record<string, string> = { low: '良好', medium: '轻度', high: '偏高' };

export default function PatientRoom() {
  const { bookingId } = useParams();
  const navigate = useNavigate();
  const { state: anxietyState } = useAnxiety();
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
  const [videoActive, setVideoActive] = useState(false);
  const streamRef = useRef<MediaStream | null>(null);
  const sessionStart = useRef(Date.now());

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
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setVideoActive(true);
    } catch (err) {
      console.warn('摄像头访问失败:', err);
    }
  };

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
              <Card title={<Space><ThunderboltOutlined /> 我的焦虑感知</Space>} style={{ borderRadius: 12 }}>
                <div style={{ textAlign: 'center', padding: '16px 0' }}>
                  <div style={{ fontSize: 48, fontWeight: 'bold', color: anxietyLevelColors[anxietyState.level] }}>
                    {anxietyState.combinedIndex}
                  </div>
                  <Tag color={anxietyLevelColors[anxietyState.level]} style={{ fontSize: 14, padding: '4px 12px', marginTop: 8 }}>
                    {anxietyLevelLabels[anxietyState.level]}
                  </Tag>
                </div>
                <Divider style={{ margin: '12px 0' }} />
                <Row gutter={8}>
                  <Col span={12}>
                    <div style={{ textAlign: 'center' }}>
                      <Text type="secondary" style={{ fontSize: 12 }}>键盘节奏</Text>
                      <div style={{ fontSize: 20, fontWeight: 600, color: '#1890ff' }}>
                        {anxietyState.keyboard.anxietyIndex}
                      </div>
                    </div>
                  </Col>
                  <Col span={12}>
                    <div style={{ textAlign: 'center' }}>
                      <Text type="secondary" style={{ fontSize: 12 }}>眨眼频率</Text>
                      <div style={{ fontSize: 20, fontWeight: 600, color: '#722ed1' }}>
                        {anxietyState.blink.anxietyIndex}
                      </div>
                    </div>
                  </Col>
                </Row>
                <div style={{ marginTop: 12, textAlign: 'center' }}>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    <ClockCircleOutlined /> 数据实时同步给咨询师
                  </Text>
                </div>
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
                    您的焦虑数据和心理画像已实时同步给咨询师，咨询师将根据您的状态提供专业帮助
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
                          <video ref={videoRef} style={{ display: 'none' }} autoPlay muted playsInline />
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
              <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Avatar size={100} style={{ backgroundColor: '#ffb6c1' }} icon={<UserOutlined />} />
              </div>
              {cameraOn && (
                <div style={{ position: 'absolute', top: 12, right: 12, width: 100, height: 140, background: '#333', borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Avatar size={40} style={{ backgroundColor: '#6366f1' }} icon={<UserOutlined />} />
                </div>
              )}
              <div style={{ position: 'absolute', top: 12, left: 12 }}>
                <Text style={{ color: '#fff', fontSize: 13 }}>咨询师</Text>
                {callActive && <div><Text style={{ color: '#aaa', fontSize: 12 }}>{formatDuration(callDuration)}</Text></div>}
              </div>
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
