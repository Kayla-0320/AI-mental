import { useState, useEffect } from 'react';
import { Card, Typography, Table, Tag, Space, Button, Modal, Input, message, Tabs, Badge, Rate, Empty } from 'antd';
import { SoundOutlined, MessageOutlined, CheckOutlined, ClockCircleOutlined, StarOutlined } from '@ant-design/icons';
import { extraApi } from '../../services';

const { Title, Text, Paragraph } = Typography;

const typeLabels: Record<string, string> = { suggestion: '💡 建议', bug: '🐛 问题', feature: '✨ 需求', other: '💬 其他' };
const statusColors: Record<string, string> = { PENDING: 'orange', REPLIED: 'green', CLOSED: 'default' };
const statusLabels: Record<string, string> = { PENDING: '待处理', REPLIED: '已回复', CLOSED: '已关闭' };

export default function AdminFeedback() {
  const [feedbacks, setFeedbacks] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [replyModal, setReplyModal] = useState(false);
  const [selectedFeedback, setSelectedFeedback] = useState<any>(null);
  const [replyContent, setReplyContent] = useState('');
  const [replyLoading, setReplyLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('all');

  useEffect(() => {
    loadFeedbacks();
  }, []);

  const loadFeedbacks = async () => {
    setLoading(true);
    try {
      const res = await extraApi.getAllFeedbacks() as any;
      const list = res.data?.feedbacks || res.data || [];
      setFeedbacks(Array.isArray(list) ? list.map((f: any) => ({ ...f, key: f.id })) : []);
    } catch {
      setFeedbacks([
        { key: '1', id: '1', type: 'suggestion', title: '希望增加夜间模式', content: '晚上使用的时候屏幕太亮了，希望能有夜间模式保护眼睛。', rating: 4, status: 'PENDING', user: { nickname: '小明' }, createdAt: new Date(Date.now() - 86400000).toISOString() },
        { key: '2', id: '2', type: 'bug', title: '心情打卡闪退', content: '选择心情后点确认，APP会闪退。用的是iPhone 13。', rating: 2, status: 'PENDING', user: { nickname: '小红' }, createdAt: new Date(Date.now() - 172800000).toISOString() },
        { key: '3', id: '3', type: 'feature', title: '想要冥想提醒功能', content: '每天下午5点左右提醒我做冥想练习，这样就不会忘记了。', rating: 5, status: 'REPLIED', reply: '感谢建议！我们已在开发计划中加入了定时提醒功能，预计下个版本更新。', user: { nickname: '小刚' }, createdAt: new Date(Date.now() - 259200000).toISOString() },
        { key: '4', id: '4', type: 'suggestion', title: '社区内容太少了', content: '希望社区能多一些内容，有时候不知道看什么。可以增加一些热门话题推荐吗？', rating: 3, status: 'PENDING', user: { nickname: '小丽' }, createdAt: new Date(Date.now() - 345600000).toISOString() },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleReply = async () => {
    if (!replyContent.trim() || !selectedFeedback) return;
    setReplyLoading(true);
    try {
      await extraApi.replyFeedback(selectedFeedback.id, replyContent);
      setFeedbacks(prev => prev.map(f =>
        f.id === selectedFeedback.id ? { ...f, status: 'REPLIED', reply: replyContent } : f
      ));
      message.success('已回复');
      setReplyModal(false);
      setReplyContent('');
    } catch {
      message.error('回复失败');
    } finally {
      setReplyLoading(false);
    }
  };

  const openReply = (feedback: any) => {
    setSelectedFeedback(feedback);
    setReplyContent(feedback.reply || '');
    setReplyModal(true);
  };

  const pendingCount = feedbacks.filter(f => f.status === 'PENDING').length;

  const filteredFeedbacks = activeTab === 'all' ? feedbacks
    : activeTab === 'pending' ? feedbacks.filter(f => f.status === 'PENDING')
    : feedbacks.filter(f => f.status !== 'PENDING');

  const columns = [
    {
      title: '类型', dataIndex: 'type', key: 'type', width: 100,
      render: (v: string) => <Tag>{typeLabels[v] || v}</Tag>,
    },
    { title: '标题', dataIndex: 'title', key: 'title', ellipsis: true },
    {
      title: '评分', dataIndex: 'rating', key: 'rating', width: 140,
      render: (v: number) => v > 0 ? <Rate disabled value={v} style={{ fontSize: 14 }} /> : <Text type="secondary">未评</Text>,
    },
    {
      title: '用户', key: 'user', width: 100,
      render: (_: any, r: any) => r.user?.nickname || '匿名',
    },
    {
      title: '状态', dataIndex: 'status', key: 'status', width: 100,
      render: (v: string) => <Tag color={statusColors[v] || 'default'}>{statusLabels[v] || v}</Tag>,
    },
    {
      title: '时间', dataIndex: 'createdAt', key: 'createdAt', width: 120,
      render: (d: string) => new Date(d).toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' }),
    },
    {
      title: '操作', key: 'action', width: 100,
      render: (_: any, record: any) => (
        <Button type="link" size="small" icon={<MessageOutlined />} onClick={() => openReply(record)}>
          {record.status === 'REPLIED' ? '查看' : '回复'}
        </Button>
      ),
    },
  ];

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <Title level={4}><SoundOutlined /> 反馈管理</Title>
        {pendingCount > 0 && (
          <Badge count={pendingCount} style={{ backgroundColor: '#faad14' }}>
            <Text style={{ fontSize: 13, padding: '4px 12px' }}>待处理反馈</Text>
          </Badge>
        )}
      </div>

      <Card style={{ borderRadius: 12 }}>
        <Tabs
          activeKey={activeTab}
          onChange={setActiveTab}
          items={[
            { key: 'all', label: <span>全部 <Badge count={feedbacks.length} style={{ backgroundColor: '#eee', color: '#666', marginLeft: 4 }} /></span> },
            { key: 'pending', label: <span>待处理 <Badge count={pendingCount} style={{ backgroundColor: '#faad14', marginLeft: 4 }} /></span> },
            { key: 'replied', label: '已处理' },
          ]}
        />
        <Table
          columns={columns}
          dataSource={filteredFeedbacks}
          loading={loading}
          pagination={{ pageSize: 10 }}
          locale={{ emptyText: <Empty description="暂无反馈" /> }}
        />
      </Card>

      {/* 回复弹窗 */}
      <Modal
        open={replyModal}
        onCancel={() => setReplyModal(false)}
        title={
          <Space>
            <MessageOutlined />
            回复反馈
            {selectedFeedback && <Tag>{typeLabels[selectedFeedback.type] || selectedFeedback.type}</Tag>}
          </Space>
        }
        footer={
          <Space>
            <Button onClick={() => setReplyModal(false)}>取消</Button>
            <Button type="primary" loading={replyLoading} onClick={handleReply} disabled={!replyContent.trim()}>
              发送回复
            </Button>
          </Space>
        }
      >
        {selectedFeedback && (
          <div>
            <div style={{ marginBottom: 16 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                <Text strong>{selectedFeedback.title}</Text>
                {selectedFeedback.rating > 0 && <Rate disabled value={selectedFeedback.rating} style={{ fontSize: 14 }} />}
              </div>
              <Card size="small" style={{ borderRadius: 8, background: '#fafafa' }}>
                <Paragraph style={{ marginBottom: 0, whiteSpace: 'pre-wrap' }}>{selectedFeedback.content}</Paragraph>
              </Card>
              <div style={{ marginTop: 8 }}>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  <ClockCircleOutlined /> {new Date(selectedFeedback.createdAt).toLocaleString()} · {selectedFeedback.user?.nickname || '匿名'}
                </Text>
              </div>
            </div>
            <Input.TextArea
              rows={4}
              value={replyContent}
              onChange={(e) => setReplyContent(e.target.value)}
              placeholder="输入回复内容..."
              style={{ borderRadius: 8 }}
            />
          </div>
        )}
      </Modal>
    </div>
  );
}
