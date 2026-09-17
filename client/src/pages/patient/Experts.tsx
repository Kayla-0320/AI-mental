import { useState, useEffect, useRef } from 'react';
import { Card, Row, Col, Typography, Button, Rate, Tag, Space, Avatar, Empty, Spin, Modal, DatePicker, Divider, List, message, Input, Tabs, Radio } from 'antd';
import { TeamOutlined, StarFilled, MessageOutlined, VideoCameraOutlined, ClockCircleOutlined, SendOutlined, UserOutlined, PhoneOutlined } from '@ant-design/icons';
import { expertApi } from '../../services';
import { useNavigate } from 'react-router-dom';

const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;

// 专家对话子组件
function ExpertChatTab({ bookings }: { bookings: any[] }) {
  const [selectedBooking, setSelectedBooking] = useState<any>(null);
  const [messages, setMessages] = useState<any[]>([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (selectedBooking) loadMessages(selectedBooking.id);
  }, [selectedBooking]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (!selectedBooking) return;
    const interval = setInterval(() => loadMessages(selectedBooking.id), 5000);
    return () => clearInterval(interval);
  }, [selectedBooking]);

  const loadMessages = async (bookingId: string) => {
    try {
      const res = await expertApi.getMessages(bookingId) as any;
      setMessages(res.data?.messages || []);
    } catch {}
  };

  const handleSend = async () => {
    if (!input.trim() || !selectedBooking) return;
    const content = input.trim();
    setInput('');
    setSending(true);

    const tempMsg = { id: 'temp', role: 'patient', content, createdAt: new Date().toISOString() };
    setMessages(prev => [...prev, tempMsg]);

    try {
      const res = await expertApi.sendMessage(selectedBooking.id, content) as any;
      if (res.data) {
        setMessages(prev => {
          const filtered = prev.filter(m => m.id !== 'temp');
          return [...filtered, res.data];
        });
      }
    } catch {
      setMessages(prev => prev.map(m => m.id === 'temp' ? { ...m, content: '发送失败，请重试', id: 'error' } : m));
    } finally { setSending(false); }
  };

  return (
    <div style={{ display: 'flex', gap: 16, height: 'calc(100vh - 260px)' }}>
      <Card style={{ width: 280, height: '100%', borderRadius: 12 }}
        title={<Text strong><TeamOutlined /> 我的咨询师</Text>}
        bodyStyle={{ padding: '8px 0', height: 'calc(100% - 57px)', overflowY: 'auto' }}>
        <List
          dataSource={bookings}
          locale={{ emptyText: <Empty description="暂无预约记录" image={Empty.PRESENTED_IMAGE_SIMPLE} /> }}
          renderItem={(item: any) => (
            <List.Item
              onClick={() => setSelectedBooking(item)}
              style={{
                cursor: 'pointer', padding: '10px 8px', borderRadius: 8,
                background: selectedBooking?.id === item.id ? '#f5f3ff' : 'transparent',
              }}
            >
              <div>
                <Text strong style={{ fontSize: 13 }}>{item.consultant?.user?.nickname || '咨询师'}</Text>
                <div><Text type="secondary" style={{ fontSize: 11 }}>{new Date(item.scheduledAt).toLocaleString()}</Text></div>
                <Tag color={item.status === 'CONFIRMED' ? 'green' : item.status === 'IN_PROGRESS' ? 'blue' : 'default'} style={{ fontSize: 10, marginTop: 4 }}>
                  {item.status === 'CONFIRMED' ? '已确认' : item.status === 'IN_PROGRESS' ? '进行中' : item.status}
                </Tag>
              </div>
            </List.Item>
          )}
        />
      </Card>

      <Card style={{ flex: 1, height: '100%', borderRadius: 12, display: 'flex', flexDirection: 'column' }}
        bodyStyle={{ flex: 1, display: 'flex', flexDirection: 'column', padding: 0, overflow: 'hidden' }}>
        {!selectedBooking ? (
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Empty description="选择一个咨询师开始对话" />
          </div>
        ) : (
          <>
            <div style={{ flex: 1, overflow: 'auto', padding: 24 }}>
              {messages.map((msg: any) => (
                <div key={msg.id} style={{ display: 'flex', marginBottom: 16, flexDirection: msg.role === 'patient' ? 'row-reverse' : 'row' }}>
                  <Avatar
                    icon={<UserOutlined />}
                    style={{ backgroundColor: msg.role === 'patient' ? '#6366f1' : '#1890ff', flexShrink: 0 }}
                  >
                    {msg.role === 'patient' ? '我' : '咨'}
                  </Avatar>
                  <div style={{
                    maxWidth: '70%', margin: '0 12px', padding: '12px 16px', borderRadius: 12,
                    background: msg.role === 'patient' ? '#6366f1' : '#e6f7ff',
                    color: msg.role === 'patient' ? '#fff' : '#333', whiteSpace: 'pre-wrap', lineHeight: 1.6,
                  }}>
                    {msg.content}
                    <div style={{ fontSize: 10, color: msg.role === 'patient' ? 'rgba(255,255,255,0.6)' : '#999', marginTop: 4, textAlign: 'right' }}>
                      {new Date(msg.createdAt).toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
            <div style={{ padding: '16px 24px', borderTop: '1px solid #f0f0f0' }}>
              <Space.Compact style={{ width: '100%' }}>
                <Input value={input} onChange={(e) => setInput(e.target.value)} onPressEnter={handleSend}
                  placeholder="给咨询师发消息..." disabled={sending} size="large" style={{ borderRadius: '8px 0 0 8px' }} />
                <Button type="primary" icon={<SendOutlined />} onClick={handleSend} loading={sending} size="large" style={{ borderRadius: '0 8px 8px 0' }}>发送</Button>
              </Space.Compact>
            </div>
          </>
        )}
      </Card>
    </div>
  );
}

export default function Experts() {
  const navigate = useNavigate();
  const [consultants, setConsultants] = useState<any[]>([]);
  const [bookings, setBookings] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [bookingModal, setBookingModal] = useState(false);
  const [selectedConsultant, setSelectedConsultant] = useState<any>(null);
  const [bookingDate, setBookingDate] = useState<any>(null);
  const [bookingNotes, setBookingNotes] = useState('');
  const [bookingType, setBookingType] = useState<string>('TEXT');
  const [bookingLoading, setBookingLoading] = useState(false);

  useEffect(() => {
    loadConsultants();
    loadBookings();
  }, []);

  const loadConsultants = async () => {
    setLoading(true);
    try {
      const res = await expertApi.getConsultants() as any;
      const list = res.data?.consultants || [];
      if (list.length > 0) {
        setConsultants(list);
      } else {
        // API 无数据时使用模拟数据
        setConsultants([
          { id: '1', user: { id: 'mock-1', nickname: '张明华' }, title: '资深心理咨询师', rating: 4.8, specialties: '["焦虑","抑郁","人际关系"]', pricePerSession: 300, introduction: '国家二级心理咨询师，从业10年，擅长认知行为疗法和正念疗法。帮助过500+来访者走出焦虑和抑郁困扰。' },
          { id: '2', user: { id: 'mock-2', nickname: '李思雨' }, title: '心理治疗师', rating: 4.6, specialties: '["压力管理","睡眠障碍","职场心理"]', pricePerSession: 250, introduction: '临床心理学硕士，专注压力管理和睡眠改善。结合CBT和ACT疗法，帮助来访者建立健康的生活方式。' },
          { id: '3', user: { id: 'mock-3', nickname: '王建国' }, title: '精神科医师', rating: 4.9, specialties: '["创伤后应激","情绪障碍","青少年心理"]', pricePerSession: 500, introduction: '三甲医院精神科副主任医师，20年临床经验。擅长复杂心理问题的诊断与治疗，注重药物与心理治疗结合。' },
        ]);
      }
    } catch {
      setConsultants([
        { id: '1', user: { id: 'mock-1', nickname: '张明华' }, title: '资深心理咨询师', rating: 4.8, specialties: '["焦虑","抑郁","人际关系"]', pricePerSession: 300, introduction: '国家二级心理咨询师，从业10年，擅长认知行为疗法和正念疗法。' },
        { id: '2', user: { id: 'mock-2', nickname: '李思雨' }, title: '心理治疗师', rating: 4.6, specialties: '["压力管理","睡眠障碍","职场心理"]', pricePerSession: 250, introduction: '临床心理学硕士，专注压力管理和睡眠改善。' },
        { id: '3', user: { id: 'mock-3', nickname: '王建国' }, title: '精神科医师', rating: 4.9, specialties: '["创伤后应激","情绪障碍","青少年心理"]', pricePerSession: 500, introduction: '三甲医院精神科副主任医师，20年临床经验。' },
      ]);
    } finally { setLoading(false); }
  };

  const loadBookings = async () => {
    try {
      const res = await expertApi.getBookings() as any;
      setBookings(res.data?.bookings || []);
    } catch {}
  };

  const handleBooking = (consultant: any) => {
    setSelectedConsultant(consultant);
    setBookingModal(true);
  };

  const confirmBooking = async () => {
    if (!bookingDate) {
      message.warning('请选择预约时间');
      return;
    }
    setBookingLoading(true);
    try {
      await expertApi.createBooking({
        consultantId: selectedConsultant.user?.id || selectedConsultant.id,
        scheduledAt: bookingDate.toISOString(),
        notes: bookingNotes,
        type: bookingType,
      });
      message.success('预约成功，等待咨询师确认');
      setBookingModal(false);
      setBookingDate(null);
      setBookingNotes('');
      setBookingType('TEXT');
      loadBookings();
    } catch (err: any) {
      message.error(err.message || '预约失败');
    } finally {
      setBookingLoading(false);
    }
  };

  const statusColors: Record<string, string> = {
    PENDING: 'orange',
    CONFIRMED: 'blue',
    IN_PROGRESS: 'green',
    COMPLETED: 'default',
    CANCELLED: 'red',
  };
  const statusLabels: Record<string, string> = {
    PENDING: '待确认',
    CONFIRMED: '已确认',
    IN_PROGRESS: '进行中',
    COMPLETED: '已完成',
    CANCELLED: '已取消',
  };

  if (loading) return <div style={{ textAlign: 'center', padding: 100 }}><Spin size="large" /></div>;

  return (
    <div>
      <Tabs
        defaultActiveKey="list"
        items={[
          {
            key: 'list',
            label: <Space><TeamOutlined /> 咨询师列表</Space>,
            children: (
              <div>
                {/* 我的预约 */}
                {bookings.length > 0 && (
                  <Card title={<Space><ClockCircleOutlined /> 我的预约</Space>} style={{ borderRadius: 12, marginBottom: 24 }}>
                    <List
                      dataSource={bookings}
                      renderItem={(item: any) => (
                        <List.Item
                          actions={[
                            item.status !== 'COMPLETED' && item.status !== 'CANCELLED' ? (
                              <Button
                                key="enter"
                                type="primary"
                                size="small"
                                icon={
                                  item.type === 'VIDEO' ? <VideoCameraOutlined /> :
                                  item.type === 'VOICE' ? <PhoneOutlined /> :
                                  <MessageOutlined />
                                }
                                onClick={() => navigate(`/experts/room/${item.id}`)}
                              >
                                {item.type === 'VIDEO' ? '进入视频咨询' :
                                 item.type === 'VOICE' ? '进入语音咨询' :
                                 '进入文字咨询'}
                              </Button>
                            ) : null,
                          ]}
                        >
                          <List.Item.Meta
                            avatar={<Avatar style={{ backgroundColor: '#6366f1' }}>{item.consultant?.user?.nickname?.[0] || '咨'}</Avatar>}
                            title={
                              <Space>
                                <Text strong>{item.consultant?.user?.nickname}</Text>
                                <Tag color={statusColors[item.status]}>{statusLabels[item.status]}</Tag>
                              </Space>
                            }
                            description={
                              <Space size={4}>
                                <Text type="secondary" style={{ fontSize: 12 }}>
                                  {item.scheduledAt ? new Date(item.scheduledAt).toLocaleString() : '时间待定'}
                                </Text>
                                <Tag color={item.type === 'VIDEO' ? 'purple' : item.type === 'VOICE' ? 'blue' : 'green'} style={{ fontSize: 10 }}>
                                  {item.type === 'VIDEO' ? '视频' : item.type === 'VOICE' ? '语音' : '文字'}
                                </Tag>
                              </Space>
                            }
                          />
                        </List.Item>
                      )}
                    />
                    <Divider />
                  </Card>
                )}

                <Title level={4} style={{ marginBottom: 16 }}>
                  <TeamOutlined /> 专业咨询师团队
                </Title>

                {consultants.length === 0 ? (
                  <Empty description="暂无可用咨询师" />
                ) : (
                  <Row gutter={[16, 16]}>
                    {consultants.map((c: any) => (
                      <Col xs={24} md={12} lg={8} key={c.id}>
                        <Card style={{ borderRadius: 12 }} bodyStyle={{ padding: 24 }}>
                          <div style={{ display: 'flex', gap: 16 }}>
                            <Avatar size={64} style={{ backgroundColor: '#6366f1', flexShrink: 0 }}>
                              {c.user?.nickname?.[0] || '咨'}
                            </Avatar>
                            <div style={{ flex: 1 }}>
                              <Title level={5} style={{ marginBottom: 4 }}>{c.user?.nickname}</Title>
                              <Text type="secondary">{c.title}</Text>
                              <div style={{ marginTop: 8 }}>
                                <Rate disabled value={Number(c.rating)} style={{ fontSize: 14 }} />
                                <Text style={{ marginLeft: 8 }}>{String(c.rating)}</Text>
                              </div>
                            </div>
                          </div>
                          <Paragraph type="secondary" ellipsis={{ rows: 2 }} style={{ marginTop: 12 }}>
                            {c.introduction}
                          </Paragraph>
                          <div style={{ marginBottom: 12 }}>
                            {JSON.parse(c.specialties || '[]').map((s: string, i: number) => (
                              <Tag key={i} color="purple">{s}</Tag>
                            ))}
                          </div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <Text strong style={{ color: '#f59e0b' }}>¥{String(c.pricePerSession)}/次</Text>
                            <Button type="primary" onClick={() => handleBooking(c)}>预约咨询</Button>
                          </div>
                        </Card>
                      </Col>
                    ))}
                  </Row>
                )}
              </div>
            ),
          },
          {
            key: 'chat',
            label: <Space><MessageOutlined /> 专家对话</Space>,
            children: <ExpertChatTab bookings={bookings} />,
          },
        ]}
      />

      <Modal open={bookingModal} onCancel={() => { setBookingModal(false); setBookingDate(null); setBookingNotes(''); setBookingType('TEXT'); }}
        title={`预约 - ${selectedConsultant?.user?.nickname || '咨询师'}`}
        onOk={confirmBooking} okText="确认预约" cancelText="取消"
        confirmLoading={bookingLoading}>
        <div style={{ padding: '20px 0' }}>
          <Text>咨询方式：</Text>
          <Radio.Group value={bookingType} onChange={(e) => setBookingType(e.target.value)} style={{ marginTop: 8, marginBottom: 16 }}>
            <Radio.Button value="TEXT"><Space><MessageOutlined /> 文字咨询</Space></Radio.Button>
            <Radio.Button value="VOICE"><Space><PhoneOutlined /> 语音咨询</Space></Radio.Button>
            <Radio.Button value="VIDEO"><Space><VideoCameraOutlined /> 视频咨询</Space></Radio.Button>
          </Radio.Group>
          <div style={{ marginTop: 16 }}>
            <Text>选择预约时间：</Text>
            <DatePicker showTime format="YYYY-MM-DD HH:mm"
              value={bookingDate} onChange={(date) => setBookingDate(date)}
              style={{ width: '100%', marginTop: 8 }} placeholder="请选择日期和时间" />
          </div>
          <div style={{ marginTop: 16 }}>
            <Text>备注（可选）：</Text>
            <TextArea rows={3} placeholder="描述你的情况或需求..."
              value={bookingNotes} onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setBookingNotes(e.target.value)}
              style={{ marginTop: 8, borderRadius: 8 }} />
          </div>
        </div>
      </Modal>
    </div>
  );
}
