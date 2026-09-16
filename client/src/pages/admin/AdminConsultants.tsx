import { useState, useEffect } from 'react';
import { Card, Typography, Table, Tag, Space, Button, Rate, message, Popconfirm, Modal, Input, Select } from 'antd';
import { TeamOutlined, CheckOutlined, CloseOutlined, StarOutlined } from '@ant-design/icons';
import api from '../../services/api';

const { Title, Text } = Typography;

const statusColors: Record<string, string> = { PENDING: 'orange', APPROVED: 'green', REJECTED: 'red', SUSPENDED: 'default' };
const statusLabels: Record<string, string> = { PENDING: '待审核', APPROVED: '已通过', REJECTED: '已拒绝', SUSPENDED: '已停用' };

export default function AdminConsultants() {
  const [consultants, setConsultants] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [detailModal, setDetailModal] = useState(false);
  const [selected, setSelected] = useState<any>(null);

  useEffect(() => { loadConsultants(); }, []);

  const loadConsultants = async () => {
    setLoading(true);
    try {
      const res = await api.get('/expert/consultants') as any;
      const list = res.data?.consultants || [];
      setConsultants(list.map((c: any) => ({ ...c, key: c.id })));
    } catch {
      setConsultants([
        { key: '1', id: '1', user: { nickname: '张咨询师' }, title: '心理咨询师', rating: 4.8, status: 'APPROVED', specialties: '["焦虑","抑郁"]', pricePerSession: 200 },
        { key: '2', id: '2', user: { nickname: '李老师' }, title: '心理治疗师', rating: 4.5, status: 'APPROVED', specialties: '["压力","睡眠"]', pricePerSession: 300 },
        { key: '3', id: '3', user: { nickname: '王医生' }, title: '精神科医师', rating: 0, status: 'PENDING', specialties: '["创伤","PTSD"]', pricePerSession: 500 },
      ]);
    } finally { setLoading(false); }
  };

  const handleApprove = async (id: string) => {
    try {
      await api.put(`/expert/consultants/${id}/status`, { status: 'APPROVED' });
      message.success('已通过审核');
      loadConsultants();
    } catch { message.error('操作失败'); }
  };

  const handleReject = async (id: string) => {
    try {
      await api.put(`/expert/consultants/${id}/status`, { status: 'REJECTED' });
      message.info('已拒绝');
      loadConsultants();
    } catch { message.error('操作失败'); }
  };

  const columns = [
    { title: '姓名', key: 'name', render: (_: any, r: any) => r.user?.nickname || '未知' },
    { title: '职称', dataIndex: 'title', key: 'title' },
    { title: '评分', dataIndex: 'rating', key: 'rating', render: (v: number) => v > 0 ? <Rate disabled value={v} style={{ fontSize: 14 }} /> : <Text type="secondary">暂无</Text> },
    { title: '单价', dataIndex: 'pricePerSession', key: 'pricePerSession', render: (v: number) => `¥${v}` },
    { title: '状态', dataIndex: 'status', key: 'status', render: (s: string) => <Tag color={statusColors[s]}>{statusLabels[s]}</Tag> },
    {
      title: '操作', key: 'action',
      render: (_: any, record: any) => {
        if (record.status === 'PENDING') {
          return (
            <Space>
              <Popconfirm title="确定通过审核？" onConfirm={() => handleApprove(record.id)}>
                <Button type="link" size="small" icon={<CheckOutlined />} style={{ color: '#52c41a' }}>通过</Button>
              </Popconfirm>
              <Popconfirm title="确定拒绝？" onConfirm={() => handleReject(record.id)}>
                <Button type="link" size="small" icon={<CloseOutlined />} danger>拒绝</Button>
              </Popconfirm>
            </Space>
          );
        }
        return (
          <Button type="link" size="small" onClick={() => { setSelected(record); setDetailModal(true); }}>查看详情</Button>
        );
      },
    },
  ];

  return (
    <div>
      <Title level={4} style={{ marginBottom: 16 }}><TeamOutlined /> 咨询师管理</Title>
      <Card style={{ borderRadius: 12 }}>
        <Table columns={columns} dataSource={consultants} loading={loading} pagination={{ pageSize: 10 }} />
      </Card>

      <Modal open={detailModal} onCancel={() => setDetailModal(false)} title="咨询师详情" footer={null} width={600}>
        {selected && (
          <div>
            <div style={{ marginBottom: 16 }}>
              <Text strong>姓名：</Text><Text>{selected.user?.nickname}</Text>
            </div>
            <div style={{ marginBottom: 16 }}>
              <Text strong>职称：</Text><Text>{selected.title}</Text>
            </div>
            <div style={{ marginBottom: 16 }}>
              <Text strong>评分：</Text><Rate disabled value={Number(selected.rating)} style={{ fontSize: 14 }} />
            </div>
            <div style={{ marginBottom: 16 }}>
              <Text strong>擅长领域：</Text>
              {JSON.parse(selected.specialties || '[]').map((s: string, i: number) => (
                <Tag key={i} color="purple">{s}</Tag>
              ))}
            </div>
            <div style={{ marginBottom: 16 }}>
              <Text strong>单价：</Text><Text>¥{selected.pricePerSession}/次</Text>
            </div>
            <div>
              <Text strong>简介：</Text>
              <div style={{ marginTop: 8, padding: 12, background: '#f9f9f9', borderRadius: 8 }}>
                {selected.introduction || '暂无简介'}
              </div>
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}
