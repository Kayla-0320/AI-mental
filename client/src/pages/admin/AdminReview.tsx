import { useState, useEffect } from 'react';
import { Card, Typography, Table, Tag, Space, Button, message, Popconfirm, Modal, Input, Badge, Tabs } from 'antd';
import { AuditOutlined, CheckOutlined, CloseOutlined, EyeOutlined, WarningOutlined } from '@ant-design/icons';
import { communityApi } from '../../services';

const { Title, Text, Paragraph } = Typography;

const riskColors: Record<string, string> = { low: 'green', medium: 'orange', high: 'red', crisis: 'magenta' };
const riskLabels: Record<string, string> = { low: '安全', medium: '注意', high: '高风险', crisis: '危机' };

export default function AdminReview() {
  const [pendingPosts, setPendingPosts] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [detailModal, setDetailModal] = useState(false);
  const [selectedPost, setSelectedPost] = useState<any>(null);
  const [activeTab, setActiveTab] = useState('pending');

  useEffect(() => {
    loadPendingPosts();
  }, []);

  const loadPendingPosts = async () => {
    setLoading(true);
    try {
      const res = await communityApi.getPending() as any;
      const posts = res.data?.posts || res.data || [];
      setPendingPosts(Array.isArray(posts) ? posts.map((p: any) => ({ ...p, key: p.id })) : []);
    } catch {
      // 后备模拟数据
      setPendingPosts([
        { key: '1', id: '1', title: '分享我的抗抑郁经历', content: '去年被确诊为轻度抑郁，经过半年的心理咨询和药物治疗，现在好多了。想和大家分享我的经历...', category: 'treehole', author: { nickname: '匿名' }, isAnonymous: true, createdAt: new Date(Date.now() - 3600000).toISOString(), riskLevel: 'low' },
        { key: '2', id: '2', title: '我觉得生活没有意义', content: '每天都觉得很累，不想上学，不想见人。有没有人和我一样？', category: 'treehole', author: { nickname: '用户B' }, isAnonymous: false, createdAt: new Date(Date.now() - 7200000).toISOString(), riskLevel: 'high' },
        { key: '3', id: '3', title: '推荐《被讨厌的勇气》', content: '这本书改变了我对人际关系的看法，推荐给每一个在意别人目光的人。', category: 'sharing', author: { nickname: '爱读书的人' }, isAnonymous: false, createdAt: new Date(Date.now() - 1800000).toISOString(), riskLevel: 'low' },
        { key: '4', id: '4', title: '有人想一起...', content: '活着太累了，有没有人想一起...', category: 'treehole', author: { nickname: '匿名' }, isAnonymous: true, createdAt: new Date(Date.now() - 900000).toISOString(), riskLevel: 'crisis' },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleApprove = async (id: string) => {
    try {
      await communityApi.approvePost(id, true);
      setPendingPosts(prev => prev.filter(p => p.id !== id));
      message.success('已通过');
    } catch {
      message.error('操作失败');
    }
  };

  const handleReject = async (id: string) => {
    try {
      await communityApi.approvePost(id, false);
      setPendingPosts(prev => prev.filter(p => p.id !== id));
      message.info('已拒绝');
    } catch {
      message.error('操作失败');
    }
  };

  const handleViewDetail = (post: any) => {
    setSelectedPost(post);
    setDetailModal(true);
  };

  const crisisCount = pendingPosts.filter(p => p.riskLevel === 'crisis' || p.riskLevel === 'high').length;

  const columns = [
    {
      title: '风险', dataIndex: 'riskLevel', key: 'riskLevel', width: 80,
      render: (v: string) => (
        <Tag color={riskColors[v] || 'default'}>
          {v === 'crisis' ? '🚨 ' : ''}{riskLabels[v] || '未知'}
        </Tag>
      ),
      sorter: (a: any, b: any) => {
        const order: Record<string, number> = { crisis: 0, high: 1, medium: 2, low: 3 };
        return (order[a.riskLevel] ?? 4) - (order[b.riskLevel] ?? 4);
      },
      defaultSortOrder: 'ascend' as const,
    },
    { title: '标题', dataIndex: 'title', key: 'title', ellipsis: true },
    {
      title: '分类', dataIndex: 'category', key: 'category', width: 100,
      render: (v: string) => <Tag>{v === 'treehole' ? '树洞' : v === 'sharing' ? '分享' : v}</Tag>,
    },
    {
      title: '作者', key: 'author', width: 100,
      render: (_: any, r: any) => r.isAnonymous ? '匿名' : (r.author?.nickname || '未知'),
    },
    {
      title: '时间', dataIndex: 'createdAt', key: 'createdAt', width: 120,
      render: (d: string) => new Date(d).toLocaleString('zh-CN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }),
    },
    {
      title: '操作', key: 'action', width: 200,
      render: (_: any, record: any) => (
        <Space>
          <Button type="link" size="small" icon={<EyeOutlined />} onClick={() => handleViewDetail(record)}>查看</Button>
          <Popconfirm title="确定通过这条内容？" onConfirm={() => handleApprove(record.id)}>
            <Button type="link" size="small" icon={<CheckOutlined />} style={{ color: '#52c41a' }}>通过</Button>
          </Popconfirm>
          <Popconfirm title="确定拒绝这条内容？" onConfirm={() => handleReject(record.id)}>
            <Button type="link" size="small" icon={<CloseOutlined />} danger>拒绝</Button>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <Title level={4}><AuditOutlined /> 社区内容审核</Title>
        {crisisCount > 0 && (
          <Badge count={crisisCount} style={{ backgroundColor: '#ff4d4f' }}>
            <Tag color="red" style={{ fontSize: 13, padding: '4px 12px' }}>
              <WarningOutlined /> 危机内容待处理
            </Tag>
          </Badge>
        )}
      </div>

      <Card style={{ borderRadius: 12 }}>
        <Tabs
          activeKey={activeTab}
          onChange={setActiveTab}
          items={[
            {
              key: 'pending',
              label: <span>待审核 <Badge count={pendingPosts.length} style={{ backgroundColor: '#faad14', marginLeft: 4 }} /></span>,
              children: (
                <Table
                  columns={columns}
                  dataSource={pendingPosts}
                  loading={loading}
                  pagination={{ pageSize: 10 }}
                  locale={{ emptyText: '暂无待审核内容 🎉' }}
                  rowClassName={(record) => record.riskLevel === 'crisis' ? 'crisis-row' : ''}
                />
              ),
            },
            {
              key: 'guidelines',
              label: '审核标准',
              children: (
                <div style={{ padding: '20px 0' }}>
                  <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                    <Card size="small" style={{ borderRadius: 8, borderLeft: '3px solid #52c41a' }}>
                      <Text strong>✅ 通过标准</Text>
                      <ul style={{ marginTop: 8, marginBottom: 0, paddingLeft: 20 }}>
                        <li>内容积极向上或真实分享个人经历</li>
                        <li>不包含攻击性、歧视性语言</li>
                        <li>不涉及自伤/自杀的具体方法描述</li>
                        <li>不包含广告、引流等商业内容</li>
                      </ul>
                    </Card>
                    <Card size="small" style={{ borderRadius: 8, borderLeft: '3px solid #ff4d4f' }}>
                      <Text strong>❌ 拒绝标准</Text>
                      <ul style={{ marginTop: 8, marginBottom: 0, paddingLeft: 20 }}>
                        <li>包含自伤/自杀的具体方法或鼓励</li>
                        <li>攻击、辱骂、歧视其他用户</li>
                        <li>涉及违法、违规内容</li>
                        <li>广告、引流、商业推广</li>
                      </ul>
                    </Card>
                    <Card size="small" style={{ borderRadius: 8, borderLeft: '3px solid #cf1322' }}>
                      <Text strong>🚨 危机内容处理流程</Text>
                      <ol style={{ marginTop: 8, marginBottom: 0, paddingLeft: 20 }}>
                        <li>立即标记并通知管理员</li>
                        <li>自动触发危机干预协议</li>
                        <li>保留内容作为证据</li>
                        <li>尝试联系用户确认安全</li>
                      </ol>
                    </Card>
                  </Space>
                </div>
              ),
            },
          ]}
        />
      </Card>

      {/* 帖子详情弹窗 */}
      <Modal
        open={detailModal}
        onCancel={() => setDetailModal(false)}
        title={selectedPost?.title || '帖子详情'}
        width={600}
        footer={
          <Space>
            <Button onClick={() => { handleReject(selectedPost?.id); setDetailModal(false); }} danger>拒绝</Button>
            <Button type="primary" onClick={() => { handleApprove(selectedPost?.id); setDetailModal(false); }}>通过</Button>
          </Space>
        }
      >
        {selectedPost && (
          <div>
            <div style={{ marginBottom: 12 }}>
              <Space>
                <Tag color={riskColors[selectedPost.riskLevel] || 'default'}>
                  {riskLabels[selectedPost.riskLevel] || '未知'}
                </Tag>
                <Tag>{selectedPost.category === 'treehole' ? '树洞' : selectedPost.category}</Tag>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  {new Date(selectedPost.createdAt).toLocaleString()}
                </Text>
              </Space>
            </div>
            <Paragraph style={{ whiteSpace: 'pre-wrap', background: '#fafafa', padding: 16, borderRadius: 8 }}>
              {selectedPost.content}
            </Paragraph>
            {selectedPost.riskLevel === 'crisis' && (
              <Card size="small" style={{ marginTop: 12, background: '#fff1f0', border: '1px solid #ffccc7', borderRadius: 8 }}>
                <Text type="danger"><WarningOutlined /> 该帖包含可能的危机信号，建议立即启动危机干预流程</Text>
              </Card>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
}
