import { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Input, Button, List, Avatar, Typography, Spin, Empty, Card, Space, Tag, Tooltip, Popconfirm } from 'antd';
import { SendOutlined, PlusOutlined, RobotOutlined, UserOutlined, ThunderboltOutlined, DeleteOutlined } from '@ant-design/icons';
import { consultationApi } from '../../services';
import { useAnxiety } from '../../context/AnxietyContext';

const { Text } = Typography;

interface Message {
  id: string;
  role: string;
  content: string;
  createdAt: string;
  sentiment?: string;
}

// AI 咨询子组件
function AIChatTab() {
  const { reportToServer, startKeyboard } = useAnxiety();
  const { conversationId } = useParams();
  const navigate = useNavigate();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sending, setSending] = useState(false);
  const [conversations, setConversations] = useState<any[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    startKeyboard();
  }, []);

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

  const handleSend = async () => {
    if (!input.trim() || !conversationId) return;
    const content = input.trim();
    setInput('');
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
      <Card style={{ width: 280, height: '100%', borderRadius: 12 }}
        title={<Text strong>对话历史</Text>}
        extra={<Button type="text" icon={<PlusOutlined />} onClick={handleNewChat} />}
        bodyStyle={{ padding: '8px 0', height: 'calc(100% - 57px)', overflowY: 'auto' }}>
        <List
          dataSource={conversations}
          locale={{ emptyText: <Empty description="暂无对话" image={Empty.PRESENTED_IMAGE_SIMPLE} /> }}
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
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Empty description="选择一个对话或创建新对话开始咨询" />
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
