import { useState, useEffect } from 'react';
import { Card, Typography, List, Tag, Space, Avatar, Empty, Button, Spin, Tabs, Badge } from 'antd';
import { MessageOutlined, VideoCameraOutlined, CalendarOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import api from '../../services/api';

const { Title, Text } = Typography;

export default function ConsultantConsultations() {
  const navigate = useNavigate();
  const [bookings, setBookings] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('all');

  useEffect(() => {
    const fetchBookings = async () => {
      try {
        const res = await api.get('/expert/bookings') as any;
        const data = res.data?.bookings || [];
        setBookings(data);
      } catch (err) {
        console.error('获取预约列表失败:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchBookings();
  }, []);

  const statusLabels: Record<string, string> = {
    PENDING: '待确认', CONFIRMED: '已确认', IN_PROGRESS: '进行中',
    COMPLETED: '已完成', CANCELLED: '已取消',
  };

  const filteredBookings = activeTab === 'all' ? bookings : bookings.filter(b => b.status === activeTab);
  const counts = {
    all: bookings.length,
    IN_PROGRESS: bookings.filter(b => b.status === 'IN_PROGRESS').length,
    CONFIRMED: bookings.filter(b => b.status === 'CONFIRMED').length,
    PENDING: bookings.filter(b => b.status === 'PENDING').length,
    COMPLETED: bookings.filter(b => b.status === 'COMPLETED').length,
  };

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
      <Title level={4} style={{ marginBottom: 16 }}>咨询记录</Title>
      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        style={{ marginBottom: 16 }}
        items={[
          { key: 'all', label: <span>全部 <Badge count={counts.all} size="small" style={{ marginLeft: 4 }} /></span> },
          { key: 'IN_PROGRESS', label: <span>进行中 <Badge count={counts.IN_PROGRESS} size="small" style={{ marginLeft: 4, backgroundColor: '#52c41a' }} /></span> },
          { key: 'CONFIRMED', label: <span>已确认 <Badge count={counts.CONFIRMED} size="small" style={{ marginLeft: 4, backgroundColor: '#1890ff' }} /></span> },
          { key: 'PENDING', label: <span>待确认 <Badge count={counts.PENDING} size="small" style={{ marginLeft: 4, backgroundColor: '#faad14' }} /></span> },
          { key: 'COMPLETED', label: <span>已完成 <Badge count={counts.COMPLETED} size="small" style={{ marginLeft: 4, backgroundColor: '#d9d9d9' }} /></span> },
        ]}
      />
      {filteredBookings.length === 0 ? (
        <Empty description="暂无咨询记录" />
      ) : (
        <List
          dataSource={filteredBookings}
          renderItem={(item: any) => (
            <Card key={item.id} hoverable style={{ borderRadius: 12, marginBottom: 12 }}>
              <div style={{ display: 'flex', gap: 16 }}>
                <Avatar size={48} style={{ backgroundColor: '#1890ff', flexShrink: 0 }}>
                  {item.patient?.nickname?.[0] || '访'}
                </Avatar>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                    <Space>
                      <Text strong style={{ fontSize: 15 }}>{item.patient?.nickname || '未知来访者'}</Text>
                      <Tag color="purple">{item.type || '文字咨询'}</Tag>
                      <Tag color={
                        item.status === 'IN_PROGRESS' ? 'green' :
                        item.status === 'CONFIRMED' ? 'blue' :
                        item.status === 'PENDING' ? 'orange' : 'default'
                      }>
                        {statusLabels[item.status] || item.status}
                      </Tag>
                    </Space>
                    <Text type="secondary">{new Date(item.scheduledAt).toLocaleDateString()}</Text>
                  </div>
                  <Text type="secondary" style={{ fontSize: 13 }}>
                    <CalendarOutlined /> {new Date(item.scheduledAt).toLocaleString()}
                  </Text>
                  <div style={{ marginTop: 8, display: 'flex', gap: 8 }}>
                    {(item.status === 'IN_PROGRESS' || item.status === 'CONFIRMED') && (
                      <Button
                        type="primary"
                        size="small"
                        icon={<VideoCameraOutlined />}
                        onClick={() => navigate(`/consultant/consultations/${item.id}`)}
                      >
                        进入咨询室
                      </Button>
                    )}
                    {item.status === 'COMPLETED' && (
                      <Button
                        size="small"
                        onClick={() => navigate(`/consultant/consultations/${item.id}`)}
                      >
                        查看记录
                      </Button>
                    )}
                  </div>
                </div>
              </div>
            </Card>
          )}
        />
      )}
    </div>
  );
}
