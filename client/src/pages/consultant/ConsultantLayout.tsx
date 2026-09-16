import { useState, useEffect } from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { Layout, Menu, Avatar, Dropdown, Badge, Space, Typography, Tag, Popover, List, Button, Empty } from 'antd';
import {
  DashboardOutlined, CalendarOutlined, MessageOutlined,
  UserOutlined, BellOutlined, LogoutOutlined, SettingOutlined,
  HeartOutlined,
} from '@ant-design/icons';
import { useAuthStore } from '../../store/authStore';
import api from '../../services/api';

const { Header, Sider, Content } = Layout;
const { Text } = Typography;

const menuItems = [
  { key: '/consultant', icon: <DashboardOutlined />, label: '工作台' },
  { key: '/consultant/appointments', icon: <CalendarOutlined />, label: '预约管理' },
  { key: '/consultant/consultations', icon: <MessageOutlined />, label: '咨询记录' },
  { key: '/consultant/profiles', icon: <UserOutlined />, label: '来访者画像' },
];

export default function ConsultantLayout() {
  const navigate = useNavigate();
  const location = useLocation();
  const { user, logout } = useAuthStore();
  const [notificationCount, setNotificationCount] = useState(0);
  const [notifications, setNotifications] = useState<any[]>([]);

  useEffect(() => {
    const loadNotifications = async () => {
      try {
        const res = await api.get('/notifications') as any;
        if (res.data) {
          setNotificationCount(res.data.unreadCount || 0);
          setNotifications(res.data.notifications || []);
        }
      } catch {}
    };
    loadNotifications();
    const interval = setInterval(loadNotifications, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const userMenu = {
    items: [
      { key: 'settings', icon: <SettingOutlined />, label: '个人设置' },
      { type: 'divider' as const },
      { key: 'logout', icon: <LogoutOutlined />, label: '退出登录', onClick: handleLogout },
    ],
  };

  return (
    <Layout style={{ minHeight: '100vh', background: 'transparent' }}>
      <Sider
        width={220}
        style={{
          background: 'rgba(255,255,255,0.7)',
          backdropFilter: 'blur(20px)',
          borderRight: '1px solid rgba(255,182,193,0.15)',
        }}
      >
        <div style={{
          padding: '24px 16px',
          textAlign: 'center',
          borderBottom: '1px solid rgba(255,182,193,0.15)',
        }}>
          <div style={{ fontSize: 36, marginBottom: 4 }}>🌸</div>
          <Text strong style={{ fontSize: 16, color: '#ff8fab' }}>咨询师工作台</Text>
          <div style={{ marginTop: 4 }}>
            <Tag color="pink" style={{ fontSize: 11, borderRadius: 20 }}>PRO</Tag>
          </div>
        </div>
        <Menu
          mode="inline"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={({ key }) => navigate(key)}
          style={{ border: 'none', marginTop: 8, background: 'transparent' }}
        />
      </Sider>

      <Layout>
        <Header style={{
          background: 'rgba(255,255,255,0.5)',
          backdropFilter: 'blur(20px)',
          padding: '0 24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          borderBottom: '1px solid rgba(255,182,193,0.1)',
          height: 60,
        }}>
          <Text style={{ fontSize: 16, fontWeight: 500, color: '#5a4a6a' }}>
            {menuItems.find(m => m.key === location.pathname)?.label || '工作台'}
          </Text>
          <Space size="middle">
            <Popover
              title={
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Text strong>通知</Text>
                  {notifications.length > 0 && (
                    <Button type="link" size="small" onClick={async () => {
                      try { await api.put('/notifications/read-all'); } catch {}
                      setNotificationCount(0);
                    }}>全部已读</Button>
                  )}
                </div>
              }
              content={
                notifications.length === 0 ? (
                  <div style={{ padding: '20px 0', textAlign: 'center' }}>
                    <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无通知" />
                  </div>
                ) : (
                  <List
                    size="small"
                    style={{ maxHeight: 300, overflow: 'auto' }}
                    dataSource={notifications}
                    renderItem={(item: any) => (
                      <List.Item
                        style={{
                          padding: '10px 0',
                          background: item.isRead ? 'transparent' : '#fffbe6',
                          borderRadius: 6,
                          marginBottom: 4,
                          cursor: 'pointer',
                        }}
                        onClick={async () => {
                          try { await api.put(`/notifications/${item.id}/read`); } catch {}
                          setNotificationCount(prev => Math.max(0, prev - 1));
                          setNotifications(prev => prev.map(n => n.id === item.id ? { ...n, isRead: true } : n));
                        }}
                      >
                        <div style={{ width: '100%' }}>
                          <div style={{ fontSize: 13, fontWeight: item.isRead ? 400 : 600 }}>{item.title}</div>
                          <div style={{ fontSize: 12, color: '#999', marginTop: 2 }}>{item.content}</div>
                          <div style={{ fontSize: 11, color: '#bbb', marginTop: 4 }}>
                            {new Date(item.createdAt).toLocaleString('zh-CN')}
                          </div>
                        </div>
                      </List.Item>
                    )}
                  />
                )
              }
              trigger="click"
              onOpenChange={(open) => { if (open) setNotificationCount(0); }}
            >
              <Badge count={notificationCount}>
                <BellOutlined style={{ fontSize: 18, cursor: 'pointer', color: '#ff8fab' }} />
              </Badge>
            </Popover>
            <Dropdown menu={userMenu}>
              <Space style={{ cursor: 'pointer' }}>
                <Avatar style={{ backgroundColor: '#ffb6c1', boxShadow: '0 2px 8px rgba(255,182,193,0.3)' }} icon={<UserOutlined />} />
                <Text style={{ color: '#5a4a6a' }}>{user?.nickname}</Text>
                <Tag color="pink" style={{ fontSize: 11, margin: 0, borderRadius: 20 }}>咨询师</Tag>
              </Space>
            </Dropdown>
          </Space>
        </Header>

        <Content style={{ padding: 24, overflow: 'auto' }}>
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  );
}
