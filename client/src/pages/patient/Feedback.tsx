import { useState, useEffect } from 'react';
import { Card, Typography, Button, Modal, Input, Select, message, Empty, Tag, Timeline } from 'antd';
import { SoundOutlined, PlusOutlined } from '@ant-design/icons';
import { extraApi } from '../../services';

const { Title, Text } = Typography;
const { TextArea } = Input;

export default function Feedback() {
  const [feedbacks, setFeedbacks] = useState<any[]>([]);
  const [modal, setModal] = useState(false);
  const [form, setForm] = useState({ type: 'suggestion', title: '', content: '', contact: '' });

  useEffect(() => {
    extraApi.getFeedbacks().then((res: any) => { if (res.code === 0) setFeedbacks(res.data); });
  }, []);

  const handleSubmit = async () => {
    if (!form.title || !form.content) return message.warning('请填写标题和内容');
    try {
      await extraApi.createFeedback(form);
      message.success('提交成功，感谢您的反馈！');
      setModal(false);
      setForm({ type: 'suggestion', title: '', content: '', contact: '' });
      extraApi.getFeedbacks().then((res: any) => { if (res.code === 0) setFeedbacks(res.data); });
    } catch (e: any) { message.error(e.message); }
  };

  const statusColors: Record<string, string> = { PENDING: 'orange', PROCESSING: 'blue', RESOLVED: 'green' };
  const statusLabels: Record<string, string> = { PENDING: '待处理', PROCESSING: '处理中', RESOLVED: '已解决' };

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <Title level={4} style={{ margin: 0, color: '#5a4a6a' }}>意见反馈</Title>
        <Button className="cloud-btn" icon={<PlusOutlined />} onClick={() => setModal(true)}>
          提交反馈
        </Button>
      </div>

      {feedbacks.length === 0 ? (
        <Card className="cloud-card" style={{ textAlign: 'center', padding: 40 }}>
          <SoundOutlined style={{ fontSize: 48, color: '#ffb6c1', marginBottom: 16 }} />
          <Text style={{ color: '#8a7a9a' }}>还没有反馈记录</Text>
          <div style={{ marginTop: 8 }}>
            <Text style={{ fontSize: 12, color: '#b0a0c0' }}>你的每一条建议都让我们变得更好</Text>
          </div>
        </Card>
      ) : (
        <Timeline>
          {feedbacks.map((f: any) => (
            <Timeline.Item key={f.id} color={f.status === 'RESOLVED' ? 'green' : 'pink'}>
              <Card className="cloud-card" size="small">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <Text strong style={{ color: '#5a4a6a' }}>{f.title}</Text>
                  <Tag color={statusColors[f.status]} style={{ borderRadius: 20 }}>{statusLabels[f.status]}</Tag>
                </div>
                <Text style={{ color: '#8a7a9a', fontSize: 13 }}>{f.content}</Text>
                {f.adminReply && (
                  <div style={{ marginTop: 8, padding: '8px 12px', background: '#fff5f7', borderRadius: 8 }}>
                    <Text style={{ fontSize: 12, color: '#ff8fab' }}>管理员回复：{f.adminReply}</Text>
                  </div>
                )}
                <Text style={{ fontSize: 11, color: '#b0a0c0', display: 'block', marginTop: 4 }}>
                  {new Date(f.createdAt).toLocaleString()}
                </Text>
              </Card>
            </Timeline.Item>
          ))}
        </Timeline>
      )}

      <Modal open={modal} onCancel={() => setModal(false)} onOk={handleSubmit} title="提交反馈" okText="提交">
        <Select
          value={form.type}
          onChange={v => setForm({ ...form, type: v })}
          style={{ width: '100%', marginBottom: 12, borderRadius: 12 }}
          options={[
            { value: 'bug', label: 'Bug 反馈' },
            { value: 'feature', label: '功能建议' },
            { value: 'suggestion', label: '其他建议' },
            { value: 'complaint', label: '投诉' },
          ]}
        />
        <Input
          placeholder="标题"
          value={form.title}
          onChange={e => setForm({ ...form, title: e.target.value })}
          style={{ marginBottom: 12, borderRadius: 12 }}
        />
        <TextArea
          rows={4}
          placeholder="详细描述..."
          value={form.content}
          onChange={e => setForm({ ...form, content: e.target.value })}
          style={{ marginBottom: 12, borderRadius: 12 }}
        />
        <Input
          placeholder="联系方式（可选，方便我们回复你）"
          value={form.contact}
          onChange={e => setForm({ ...form, contact: e.target.value })}
          style={{ borderRadius: 12 }}
        />
      </Modal>
    </div>
  );
}
