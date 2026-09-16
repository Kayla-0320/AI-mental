import { useState, useEffect } from 'react';
import { Card, Typography, Table, Tag, Space, Button, message, Popconfirm, Spin, Empty } from 'antd';
import { CalendarOutlined, ClockCircleOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import api from '../../services/api';

const { Title, Text } = Typography;

const statusColors: Record<string, string> = {
  PENDING: 'orange', CONFIRMED: 'blue', IN_PROGRESS: 'processing',
  COMPLETED: 'green', CANCELLED: 'default',
};
const statusLabels: Record<string, string> = {
  PENDING: '待确认', CONFIRMED: '已确认', IN_PROGRESS: '进行中',
  COMPLETED: '已完成', CANCELLED: '已取消',
};

export default function ConsultantAppointments() {
  const navigate = useNavigate();
  const [appointments, setAppointments] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadAppointments();
  }, []);

  const loadAppointments = async () => {
    setLoading(true);
    try {
      const res = await api.get('/expert/bookings') as any;
      const bookings = res.data || [];
      // 转换为表格数据格式
      const data = bookings.map((b: any, i: number) => ({
        key: b.id,
        index: i + 1,
        patient: b.patient?.nickname || '未知',
        date: new Date(b.scheduledAt).toLocaleDateString(),
        time: new Date(b.scheduledAt).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' }),
        duration: 50,
        type: ({ TEXT: '文字咨询', VOICE: '语音咨询', VIDEO: '视频咨询' } as Record<string, string>)[b.type] || b.type || '文字咨询',
        status: b.status,
        raw: b,
      }));
      setAppointments(data);
    } catch (err) {
      console.error('获取预约列表失败:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleConfirm = async (key: string) => {
    try {
      await api.put(`/expert/bookings/${key}/status`, { status: 'CONFIRMED' });
      setAppointments(prev => prev.map(apt =>
        apt.key === key ? { ...apt, status: 'CONFIRMED' } : apt
      ));
      message.success('已确认预约');
    } catch {
      message.error('操作失败');
    }
  };

  const handleReject = async (key: string) => {
    try {
      await api.put(`/expert/bookings/${key}/status`, { status: 'CANCELLED' });
      setAppointments(prev => prev.map(apt =>
        apt.key === key ? { ...apt, status: 'CANCELLED' } : apt
      ));
      message.info('已拒绝预约');
    } catch {
      message.error('操作失败');
    }
  };

  const handleStart = (key: string) => {
    setAppointments(prev => prev.map(apt =>
      apt.key === key ? { ...apt, status: 'IN_PROGRESS' } : apt
    ));
    message.success('开始咨询');
    navigate(`/consultant/consultations/${key}`);
  };

  const handleComplete = async (key: string) => {
    try {
      await api.put(`/expert/bookings/${key}/status`, { status: 'COMPLETED' });
      setAppointments(prev => prev.map(apt =>
        apt.key === key ? { ...apt, status: 'COMPLETED' } : apt
      ));
      message.success('咨询已完成');
    } catch {
      message.error('操作失败');
    }
  };

  const columns = [
    {
      title: '序号',
      dataIndex: 'index',
      key: 'index',
      width: 60,
    },
    {
      title: '来访者',
      dataIndex: 'patient',
      key: 'patient',
    },
    {
      title: '日期时间',
      key: 'datetime',
      render: (_: any, record: any) => (
        <Space>
          <CalendarOutlined /> {record.date}
          <ClockCircleOutlined /> {record.time}
        </Space>
      ),
    },
    {
      title: '时长',
      dataIndex: 'duration',
      key: 'duration',
      render: (d: number) => `${d} 分钟`,
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (t: string) => <Tag>{t}</Tag>,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (s: string) => <Tag color={statusColors[s]}>{statusLabels[s]}</Tag>,
    },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: any) => {
        if (record.status === 'PENDING') {
          return (
            <Space>
              <Button type="link" size="small" onClick={() => handleConfirm(record.key)}>确认</Button>
              <Popconfirm
                title="确定拒绝这个预约吗？"
                onConfirm={() => handleReject(record.key)}
                okText="确定"
                cancelText="取消"
              >
                <Button type="link" size="small" danger>拒绝</Button>
              </Popconfirm>
            </Space>
          );
        }
        if (record.status === 'CONFIRMED') {
          return <Button type="link" size="small" onClick={() => handleStart(record.key)}>开始咨询</Button>;
        }
        if (record.status === 'IN_PROGRESS') {
          return (
            <Space>
              <Button type="link" size="small" onClick={() => navigate(`/consultant/consultations/${record.key}`)}>进入咨询室</Button>
              <Popconfirm title="确定结束咨询？" onConfirm={() => handleComplete(record.key)} okText="确定" cancelText="取消">
                <Button type="link" size="small">结束咨询</Button>
              </Popconfirm>
            </Space>
          );
        }
        if (record.status === 'COMPLETED') {
          return <Text type="secondary">已结束</Text>;
        }
        if (record.status === 'CANCELLED') {
          return <Text type="secondary">已拒绝</Text>;
        }
        return null;
      },
    },
  ];

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 100 }}>
        <Spin size="large" />
        <div style={{ marginTop: 16 }}><Text type="secondary">加载中...</Text></div>
      </div>
    );
  }

  return (
    <div>
      <Title level={4} style={{ marginBottom: 16 }}>预约管理</Title>
      {appointments.length === 0 ? (
        <Card style={{ borderRadius: 12, textAlign: 'center', padding: '60px 40px' }}>
          <div style={{ fontSize: 48, marginBottom: 16 }}>📅</div>
          <Title level={5} style={{ marginBottom: 8 }}>暂无预约记录</Title>
          <Text type="secondary" style={{ display: 'block', marginBottom: 16 }}>
            当患者通过专家连线页面预约您的咨询后，预约信息将显示在这里
          </Text>
          <Text type="secondary" style={{ fontSize: 12 }}>
            💡 提示：请确保您的咨询师资料已完善，以便患者能够找到并预约您
          </Text>
        </Card>
      ) : (
        <Card style={{ borderRadius: 12 }} bodyStyle={{ maxHeight: 'calc(100vh - 200px)', overflow: 'auto' }}>
          <Table columns={columns} dataSource={appointments} pagination={{ pageSize: 10 }} />
        </Card>
      )}
    </div>
  );
}
