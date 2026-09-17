import { useState, useEffect, useRef } from 'react';
import { Input, Button, Avatar, Typography, Spin, Empty, Card, Space, Tag, Tooltip } from 'antd';
import { SendOutlined, PlusOutlined, MessageOutlined, ArrowLeftOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { consultationApi } from '../../services';
import { useAnxiety } from '../../context/AnxietyContext';

const { Text } = Typography;

interface Message {
  id: string;
  role: string;
  content: string;
  createdAt: string;
}

const welcomeMessages = [
  '今天有什么想吐槽的吗？',
  '心里堵得慌？说出来会好一些。',
  '不用组织语言，想到什么说什么。',
];

export default function SocraticChat() {
  const { reportToServer } = useAnxiety();
  const navigate = useNavigate();
  const [conversationId, setConversationId] = useState<string>('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [welcomeMsg] = useState(() => welcomeMessages[Math.floor(Math.random() * welcomeMessages.length)]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // 键盘监测由浮动窗口的统一开关控制，不再自动启动

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleNewChat = async () => {
    try {
      const res = await consultationApi.createConversation('树洞对话') as any;
      setConversationId(res.data.id);
      setMessages([]);
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
      const res = await consultationApi.sendSocraticMessage(conversationId, content) as any;
      reportToServer('话痨树洞对话');
      if (res.data) {
        setMessages(prev => {
          const filtered = prev.filter(m => m.id !== 'temp');
          return [...filtered, res.data.userMessage, res.data.aiMessage];
        });
      }
    } catch {
      setMessages(prev => [...prev, {
        id: 'error', role: 'assistant',
        content: '嗯...我还在听，能再说一遍吗？',
        createdAt: new Date().toISOString(),
      }]);
    } finally { setSending(false); }
  };

  if (!conversationId) {
    return (
      <div style={{ maxWidth: 600, margin: '0 auto', padding: '40px 20px' }}>
        <Card style={{ borderRadius: 16, textAlign: 'center' }}>
          <div style={{ fontSize: 64, marginBottom: 16 }}>🕳️</div>
          <Typography.Title level={3} style={{ marginBottom: 8 }}>话痨树洞</Typography.Title>
          <Text type="secondary" style={{ fontSize: 15, display: 'block', marginBottom: 24 }}>
            不给建议，只陪你聊。用反问帮你看见自己没注意到的东西。
          </Text>
          <div style={{
            background: '#f9f0ff', padding: '16px 20px', borderRadius: 12,
            marginBottom: 24, textAlign: 'left',
          }}>
            <Text style={{ fontSize: 13, color: '#722ed1' }}>
              💡 这里不会告诉你"应该怎么做"，只会问你"你是怎么想的"。
              <br />把这里当作一个安全的出口，说什么都可以。
            </Text>
          </div>
          <Button type="primary" size="large" onClick={handleNewChat}
            style={{ borderRadius: 24, padding: '0 40px', height: 48, fontSize: 16 }}>
            <MessageOutlined /> 开始倾诉
          </Button>
        </Card>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 140px)' }}>
      {/* 顶部栏 */}
      <Card style={{ borderRadius: '12px 12px 0 0', borderBottom: '1px solid #f0f0f0' }}
        bodyStyle={{ padding: '12px 20px' }}>
        <Space>
          <Button type="text" icon={<ArrowLeftOutlined />} onClick={() => { setConversationId(''); setMessages([]); }} />
          <div>
            <Text strong>🕳️ 话痨树洞</Text>
            <div><Text type="secondary" style={{ fontSize: 12 }}>苏格拉底式反问，不给建议只引导</Text></div>
          </div>
        </Space>
      </Card>

      {/* 消息区域 */}
      <div style={{ flex: 1, overflow: 'auto', padding: 24, background: '#fafafa' }}>
        {messages.length === 0 && (
          <div style={{ textAlign: 'center', padding: '40px 0' }}>
            <Text type="secondary" style={{ fontSize: 15 }}>{welcomeMsg}</Text>
          </div>
        )}
        {messages.map((msg) => (
          <div key={msg.id} style={{
            display: 'flex', marginBottom: 16,
            flexDirection: msg.role === 'user' ? 'row-reverse' : 'row',
          }}>
            <Avatar
              icon={msg.role === 'user' ? <MessageOutlined /> : <span style={{ fontSize: 14 }}>🕳️</span>}
              style={{
                backgroundColor: msg.role === 'user' ? '#722ed1' : '#f0f0f0',
                color: msg.role === 'user' ? '#fff' : '#333',
                flexShrink: 0,
              }}
            />
            <div style={{
              maxWidth: '75%', margin: '0 12px',
              padding: '12px 16px', borderRadius: 16,
              background: msg.role === 'user' ? '#722ed1' : '#fff',
              color: msg.role === 'user' ? '#fff' : '#333',
              whiteSpace: 'pre-wrap', lineHeight: 1.7,
              boxShadow: msg.role === 'assistant' ? '0 1px 4px rgba(0,0,0,0.06)' : 'none',
            }}>
              {msg.content}
            </div>
          </div>
        ))}
        {sending && (
          <div style={{ display: 'flex', marginBottom: 16 }}>
            <Avatar style={{ backgroundColor: '#f0f0f0' }}><span style={{ fontSize: 14 }}>🕳️</span></Avatar>
            <div style={{ marginLeft: 12, padding: '12px 16px', background: '#fff', borderRadius: 16, boxShadow: '0 1px 4px rgba(0,0,0,0.06)' }}>
              <Spin size="small" /> <Text type="secondary" style={{ marginLeft: 8 }}>正在思考怎么问你...</Text>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* 输入区域 */}
      <div style={{ padding: '16px 24px', borderTop: '1px solid #f0f0f0', background: '#fff' }}>
        <Space.Compact style={{ width: '100%' }}>
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onPressEnter={handleSend}
            placeholder="说什么都行，不用顾虑..."
            disabled={sending}
            size="large"
            style={{ borderRadius: '8px 0 0 8px' }}
          />
          <Button type="primary" icon={<SendOutlined />} onClick={handleSend}
            loading={sending} size="large"
            style={{ borderRadius: '0 8px 8px 0', background: '#722ed1', borderColor: '#722ed1' }}>
            发送
          </Button>
        </Space.Compact>
      </div>
    </div>
  );
}
