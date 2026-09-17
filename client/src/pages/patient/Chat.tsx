import { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Input, Button, List, Avatar, Typography, Spin, Empty, Card, Space, Tag, Tooltip, Popconfirm, Modal } from 'antd';
import { SendOutlined, PlusOutlined, RobotOutlined, UserOutlined, ThunderboltOutlined, DeleteOutlined, HeartOutlined, PhoneOutlined, SmileOutlined } from '@ant-design/icons';
import { consultationApi } from '../../services';
import { useAnxiety } from '../../context/AnxietyContext';

const PERCEPTION_API = 'http://localhost:8001';

const { Text } = Typography;

interface Message {
  id: string;
  role: string;
  content: string;
  createdAt: string;
  sentiment?: string;
}

// 危机关键词检测
const crisisKeywords = [
  '自杀', '自残', '自伤', '不想活', '想死', '去死', '活着没意思',
  '活着没有意义', '不如死了', '伤害自己', '结束生命', '跳楼',
  '割腕', '没有活下去的理由', '世界没有我会更好',
];

function detectCrisis(text: string): boolean {
  return crisisKeywords.some(kw => text.includes(kw));
}

// AI 聊天子组件
function AIChatTab() {
  const { reportToServer } = useAnxiety();
  const { conversationId } = useParams();
  const navigate = useNavigate();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sending, setSending] = useState(false);
  const [conversations, setConversations] = useState<any[]>([]);
  const [crisisModalOpen, setCrisisModalOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // 实时情绪分析结果（最近一条用户消息的情绪）
  const [latestEmotion, setLatestEmotion] = useState<{
    text_emotion_probs: number[];
    confidence: number;
    evidence: string[];
  } | null>(null);
  const emotionLabels = ['快乐', '悲伤', '焦虑', '愤怒', '中性'];
  const emotionBarColors = ['#52c41a', '#722ed1', '#ff4d4f', '#fa8c16', '#1890ff'];

  // 键盘监测由浮动窗口的统一开关控制，不再自动启动

  useEffect(() => {
    loadConversations();
    if (conversationId) loadMessages(conversationId);
  }, [conversationId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const loadConversations = async () => {
    try {
      const res = await consultationApi.getConversations() as any;
      setConversations(res.data?.conversations || []);
    } catch {}
  };

  const loadMessages = async (convId: string) => {
    setLoading(true);
    try {
      const res = await consultationApi.getMessages(convId) as any;
      setMessages(res.data?.messages || []);
    } catch {} finally { setLoading(false); }
  };

  const handleDeleteConversation = async (e: React.MouseEvent, convId: string) => {
    e.stopPropagation();
    try {
      await consultationApi.deleteConversation(convId);
      setConversations(prev => prev.filter(c => c.id !== convId));
      if (convId === conversationId) {
        navigate('/chat');
        setMessages([]);
      }
    } catch {}
  };

  const handleNewChat = async () => {
    try {
      const res = await consultationApi.createConversation() as any;
      navigate(`/chat/${res.data.id}`);
      setMessages([]);
      loadConversations();
    } catch {}
  };

  // 调用感知 API 分析情绪
  const analyzeEmotion = useCallback(async (text: string) => {
    if (!text.trim()) return;
    try {
      const res = await fetch(`${PERCEPTION_API}/api/v1/perception/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const result = await res.json();
      setLatestEmotion(result);
    } catch {
      // 感知服务不可用时静默降级，不影响对话
    }
  }, []);

  const handleSend = async () => {
    if (!input.trim() || !conversationId) return;
    const content = input.trim();
    setInput('');

    // 危机信号检测
    if (detectCrisis(content)) {
      setCrisisModalOpen(true);
    }

    // 实时情绪分析（无感调用，不阻塞对话）
    analyzeEmotion(content);

    const userMsg: Message = { id: 'temp', role: 'user', content, createdAt: new Date().toISOString() };
    setMessages(prev => [...prev, userMsg]);
    setSending(true);
    try {
      const res = await consultationApi.sendMessage(conversationId, content) as any;
      reportToServer('AI咨询对话');
      if (res.data) {
        setMessages(prev => {
          const filtered = prev.filter(m => m.id !== 'temp');
          return [...filtered, res.data.userMessage, res.data.aiMessage];
        });
      }
    } catch {
      setMessages(prev => [...prev, { id: 'error', role: 'assistant', content: '网络异常，请稍后重试', createdAt: new Date().toISOString() }]);
    } finally { setSending(false); }
  };

  const emotionColors: Record<string, string> = {
    '焦虑': 'orange', '悲伤': 'blue', '愤怒': 'red', '恐惧': 'purple',
    '喜悦': 'green', '平静': 'cyan', '困惑': 'gold', '孤独': 'geekblue',
    '压力': 'volcano', '无助': 'magenta',
  };

  const renderEmotionTag = (msg: Message) => {
    if (msg.role !== 'user' || !msg.sentiment) return null;
    try {
      const emotion = JSON.parse(msg.sentiment);
      if (!emotion.primaryEmotion) return null;
      return (
        <Tooltip title={`强度: ${emotion.intensity}% | 语气: ${emotion.tone || '未知'}`}>
          <Tag color={emotionColors[emotion.primaryEmotion] || 'default'} style={{ marginTop: 6, fontSize: 11, cursor: 'pointer' }}>
            <ThunderboltOutlined /> {emotion.primaryEmotion} {emotion.intensity > 0 ? `${emotion.intensity}%` : ''}
          </Tag>
        </Tooltip>
      );
    } catch { return null; }
  };

  return (
    <div style={{ display: 'flex', gap: 16, height: 'calc(100vh - 200px)' }}>
      <CrisisModal
        open={crisisModalOpen}
        onClose={() => setCrisisModalOpen(false)}
        onGoChat={() => setCrisisModalOpen(false)}
      />
      <Card style={{ width: 280, height: '100%', borderRadius: 12 }}
        title={<Text strong>对话历史</Text>}
        extra={<Button type="text" icon={<PlusOutlined />} onClick={handleNewChat} />}
        bodyStyle={{ padding: '8px 0', height: 'calc(100% - 57px)', overflowY: 'auto' }}>
        <List
          dataSource={conversations}
          locale={{ emptyText: <Empty description="还没有对话，点击上方 + 开始" image={Empty.PRESENTED_IMAGE_SIMPLE} /> }}
          renderItem={(item: any) => (
            <List.Item
              onClick={() => navigate(`/chat/${item.id}`)}
              style={{ cursor: 'pointer', padding: '10px 8px', borderRadius: 8, background: item.id === conversationId ? '#f5f3ff' : 'transparent' }}
              actions={[
                <Popconfirm key="del" title="确定删除？" onConfirm={(e) => handleDeleteConversation(e as any, item.id)} onCancel={(e) => e?.stopPropagation()} okText="删除" cancelText="取消" okButtonProps={{ danger: true }}>
                  <DeleteOutlined onClick={(e) => e.stopPropagation()} style={{ color: '#ccc', fontSize: 14, cursor: 'pointer' }}
                    onMouseEnter={(e) => (e.currentTarget.style.color = '#ff4d4f')} onMouseLeave={(e) => (e.currentTarget.style.color = '#ccc')} />
                </Popconfirm>,
              ]}
            >
              <Text ellipsis style={{ fontSize: 13 }}>{item.title || '新对话'}</Text>
            </List.Item>
          )}
        />
      </Card>

      <Card style={{ flex: 1, height: '100%', borderRadius: 12, display: 'flex', flexDirection: 'column' }}
        bodyStyle={{ flex: 1, display: 'flex', flexDirection: 'column', padding: 0, overflow: 'hidden' }}>
        {!conversationId ? (
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: 40 }}>
            <div style={{ fontSize: 64, marginBottom: 16 }}>💬</div>
            <Empty description={
              <span style={{ color: '#8a7a9a', fontSize: 15 }}>
                创建新对话，AI 会陪你聊聊
              </span>
            } image={null} />
            <div style={{ marginTop: 24, maxWidth: 400, textAlign: 'center' }}>
              <Text type="secondary" style={{ fontSize: 13, display: 'block', marginBottom: 12 }}>
                不知道说什么？试试这些话题：
              </Text>
              <Space wrap size={[8, 8]} style={{ justifyContent: 'center' }}>
                {['学业压力好大', '和朋友闹别扭了', '最近睡不好', '就是心情不好', '对未来很迷茫', '想找人说说'].map(topic => (
                  <Tag
                    key={topic}
                    style={{
                      cursor: 'pointer', padding: '6px 14px', borderRadius: 20,
                      background: '#f5f3ff', color: '#6366f1', border: 'none', fontSize: 13,
                    }}
                    onClick={handleNewChat}
                  >
                    {topic}
                  </Tag>
                ))}
              </Space>
            </div>
            <Button type="primary" size="large" icon={<PlusOutlined />} onClick={handleNewChat}
              style={{ marginTop: 24, borderRadius: 24, padding: '0 32px' }}>
              开始新对话
            </Button>
          </div>
        ) : (
          <>
            <div style={{ flex: 1, overflow: 'auto', padding: 24 }}>
              {loading ? <div style={{ textAlign: 'center' }}><Spin /></div> : (
                messages.map((msg) => (
                  <div key={msg.id} style={{ display: 'flex', marginBottom: 16, flexDirection: msg.role === 'user' ? 'row-reverse' : 'row' }}>
                    <Avatar icon={msg.role === 'user' ? <UserOutlined /> : <RobotOutlined />}
                      style={{ backgroundColor: msg.role === 'user' ? '#6366f1' : '#f0f0f0', color: msg.role === 'user' ? '#fff' : '#6366f1', flexShrink: 0 }} />
                    <div style={{ maxWidth: '70%', margin: '0 12px', padding: '12px 16px', borderRadius: 12,
                      background: msg.role === 'user' ? '#6366f1' : '#f5f5f5', color: msg.role === 'user' ? '#fff' : '#333', whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
                      {msg.content}
                    </div>
                    {renderEmotionTag(msg)}
                  </div>
                ))
              )}
              {sending && (
                <div style={{ display: 'flex', marginBottom: 16 }}>
                  <Avatar icon={<RobotOutlined />} style={{ backgroundColor: '#f0f0f0', color: '#6366f1' }} />
                  <div style={{ marginLeft: 12, padding: '12px 16px', background: '#f5f5f5', borderRadius: 12 }}>
                    <Spin size="small" /> <Text type="secondary" style={{ marginLeft: 8 }}>正在思考中...</Text>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
            <div style={{ padding: '16px 24px', borderTop: '1px solid #f0f0f0' }}>
              {/* 实时情绪感知条 */}
              {latestEmotion && (
                <div style={{ padding: '8px 0 12px', display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
                  <Space size={4}>
                    <SmileOutlined style={{ color: '#6366f1', fontSize: 14 }} />
                    <Text strong style={{ fontSize: 12 }}>AI 实时感知</Text>
                  </Space>
                  {latestEmotion.text_emotion_probs.map((p: number, i: number) => (
                    <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                      <span style={{ fontSize: 11, color: emotionBarColors[i], minWidth: 28 }}>{emotionLabels[i]}</span>
                      <div style={{ width: 40, height: 4, background: '#f0f0f0', borderRadius: 2 }}>
                        <div style={{ height: '100%', width: `${Math.round(p * 100)}%`, background: emotionBarColors[i], borderRadius: 2, transition: 'width 0.5s' }} />
                      </div>
                      <span style={{ fontSize: 10, color: '#999', minWidth: 28 }}>{Math.round(p * 100)}%</span>
                    </div>
                  ))}
                  <Tag color="purple" style={{ fontSize: 10, margin: 0 }}>置信度 {Math.round(latestEmotion.confidence * 100)}%</Tag>
                </div>
              )}
              <Space.Compact style={{ width: '100%' }}>
                <Input value={input} onChange={(e) => setInput(e.target.value)} onPressEnter={handleSend}
                  placeholder="说说你的想法..." disabled={sending} size="large" style={{ borderRadius: '8px 0 0 8px' }} />
                <Button type="primary" icon={<SendOutlined />} onClick={handleSend} loading={sending} size="large" style={{ borderRadius: '0 8px 8px 0' }}>发送</Button>
              </Space.Compact>
            </div>
          </>
        )}
      </Card>
    </div>
  );
}

// 主组件
export default function Chat() {
  return <AIChatTab />;
}

// 危机干预弹窗（独立组件，可在需要时复用）
function CrisisModal({ open, onClose, onGoChat }: { open: boolean; onClose: () => void; onGoChat: () => void }) {
  return (
    <Modal open={open} onCancel={onClose} footer={null} width={460} centered>
      <div style={{ textAlign: 'center', padding: '20px 0' }}>
        <div style={{ fontSize: 48, marginBottom: 12 }}>💛</div>
        <Typography.Title level={4} style={{ marginBottom: 8, color: '#5a4a6a' }}>
          我们很关心你
        </Typography.Title>
        <div style={{
          padding: '16px 20px', background: '#fff7e6', borderRadius: 12,
          textAlign: 'left', marginBottom: 20,
        }}>
          <Typography.Paragraph style={{ fontSize: 15, color: '#5a4a6a', marginBottom: 8 }}>
            你说的话让我们有些担心。不管发生了什么，你都不是一个人。
          </Typography.Paragraph>
          <Typography.Paragraph style={{ fontSize: 14, color: '#8a7a9a', marginBottom: 0 }}>
            现在有人愿意听你说，24小时都在：
          </Typography.Paragraph>
        </div>
        <Card style={{ borderRadius: 12, marginBottom: 20, background: '#fff5f5', border: '1px solid #ffccc7' }}>
          <div style={{ marginBottom: 12 }}>
            <Typography.Text strong>🆘 24小时心理援助热线</Typography.Text>
            <Typography.Title level={3} style={{ color: '#ff4d4f', margin: '4px 0' }}>400-161-9995</Typography.Title>
          </div>
          <div>
            <Typography.Text strong>💬 生命热线</Typography.Text>
            <Typography.Title level={3} style={{ color: '#6366f1', margin: '4px 0' }}>400-821-1215</Typography.Title>
          </div>
        </Card>
        <Space direction="vertical" style={{ width: '100%' }} size="middle">
          <Button type="primary" block size="large" onClick={onGoChat}
            style={{ borderRadius: 12 }}>
            <HeartOutlined /> 继续和 AI 聊聊
          </Button>
          <Button block size="large" onClick={onClose} style={{ borderRadius: 12 }}>
            我知道了
          </Button>
        </Space>
      </div>
    </Modal>
  );
}
