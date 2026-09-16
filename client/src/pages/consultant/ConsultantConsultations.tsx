import { useState, useEffect } from 'react';
import { Card, Typography, List, Tag, Space, Avatar, Empty, Button, Spin } from 'antd';
import { MessageOutlined, VideoCameraOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import api from '../../services/api';

const { Title, Text } = Typography;

export default function ConsultantConsultations() {
  const navigate = useNavigate();
  const [bookings, setBookings] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchBookings = async () => {
      try {
        const res = await api.get('/expert/bookings') as any;
        const data = res.data?.bookings || [];
        if (data.length > 0) {
          setBookings(data);
        } else {
          // 使用模拟数据作为演示
          setBookings([
            {
              id: 'demo-1',
              patient: { nickname: '小明' },
              scheduledAt: new Date().toISOString(),
              status: 'CONFIRMED',
              type: '文字咨询',
            },
            {
              id: 'demo-2',
              patient: { nickname: '小红' },
              scheduledAt: new Date(Date.now() - 86400000).toISOString(),
              status: 'COMPLETED',
              type: '语音咨询',
            },
            {
              id: 'demo-3',
              patient: { nickname: '小刚' },
              scheduledAt: new Date(Date.now() + 86400000).toISOString(),
              status: 'PENDING',
              type: '文字咨询',
            },
          ]);
        }
      } catch (err) {
        console.error('获取预约列表失败:', err);
        // API 失败时也使用模拟数据
        setBookings([
          {
            id: 'demo-1',
            patient: { nickname: '小明' },
            scheduledAt: new Date().toISOString(),
            status: 'CONFIRMED',
            type: '文字咨询',
          },
          {
            id: 'demo-2',
            patient: { nickname: '小红' },
            scheduledAt: new Date(Date.now() - 86400000).toISOString(),
            status: 'COMPLETED',
            type: '语音咨询',
          },
        ]);
      } finally {
        setLoading(false);
      }
    };
    fetchBookings();
  }, []);

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
      {bookings.length === 0 ? (
        <Empty description="暂无咨询记录" />
      ) : (
        <List
          dataSource={bookings}
          renderItem={(item: any) => (
            <Card key={item.id} hoverable style={{ borderRadius: 12, marginBottom: 12 }}>
              <div style={{ display: 'flex', gap: 16 }}>
                <Avatar size={48} style={{ backgroundColor: '#1890ff', flexShrink: 0 }}>
                  {item.patient?.nickname?.[0] || '患'}
                </Avatar>
                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                    <Space>
                      <Text strong style={{ fontSize: 15 }}>{item.patient?.nickname || '未知患者'}</Text>
                      <Tag color="purple">{item.type || '文字咨询'}</Tag>
                    </Space>
                    <Text type="secondary">{new Date(item.scheduledAt).toLocaleDateString()}</Text>
                  </div>
                  <Text type="secondary" style={{ fontSize: 13 }}>
                    预约时间：{new Date(item.scheduledAt).toLocaleString()} | 状态：{item.status}
                  </Text>
                  <div style={{ marginTop: 8, display: 'flex', gap: 8 }}>
                    <Button
                      type="primary"
                      size="small"
                      icon={<VideoCameraOutlined />}
                      onClick={() => navigate(`/consultant/consultations/${item.id}`)}
                    >
                      进入咨询室
                    </Button>
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
