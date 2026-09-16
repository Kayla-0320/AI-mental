import { useState, useEffect } from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { Layout, Menu, Avatar, Dropdown, Badge, Space, Typography, Popover, List, Button, Empty } from 'antd';
import {
  HomeOutlined, MessageOutlined, UserOutlined, HeartOutlined,
  TeamOutlined, BellOutlined, LogoutOutlined, SettingOutlined,
  CoffeeOutlined, AimOutlined, TrophyOutlined,
} from '@ant-design/icons';
import { useAuthStore } from '../../store/authStore';
import api from '../../services/api';
import { AnxietyProvider } from '../../context/AnxietyContext';
import AnxietyFloatingWidget from '../../components/AnxietyFloatingWidget';
import RealityTaskModal from '../../components/RealityTaskModal';
import CrisisInterventionWidget from '../../components/CrisisInterventionWidget';

const { Header, Sider, Content } = Layout;
const { Text } = Typography;

const menuItems = [
  { key: '/', icon: <HomeOutlined />, label: '首页' },
  { key: '/chat', icon: <MessageOutlined />, label: '聊聊' },
  { key: '/companions', icon: <TeamOutlined />, label: '同伴' },
  { key: '/healing', icon: <HeartOutlined />, label: '放松空间' },
  { key: '/assessment', icon: <AimOutlined />, label: '心情检测' },
  { key: '/growth', icon: <TrophyOutlined />, label: '我的成长' },
  { key: '/experts', icon: <CoffeeOutlined />, label: '找帮手' },
  { key: '/settings', icon: <SettingOutlined />, label: '设置' },
];

export default function PatientLayout() {
  const navigate = useNavigate();
  const location = useLocation();
  const { user, logout } = useAuthStore();
  const [notificationCount, setNotificationCount] = useState(0);
  const [notifications, setNotifications] = useState<any[]>([]);

  // 加载通知
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
      { key: 'settings', icon: <SettingOutlined />, label: '个人设置', onClick: () => navigate('/settings') },
      { type: 'divider' as const },
      { key: 'logout', icon: <LogoutOutlined />, label: '退出登录', onClick: handleLogout },
    ],
  };

  return (
    <AnxietyProvider>
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
          <Text strong style={{ fontSize: 16, color: '#ff8fab' }}>心灵花园</Text>
          <div style={{ fontSize: 11, color: '#b0a0c0', marginTop: 2 }}>温暖陪伴每一天</div>
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
            {menuItems.find(m => m.key === location.pathname)?.label || '首页'}
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
              </Space>
            </Dropdown>
          </Space>
        </Header>

        <Content style={{ padding: 24, overflow: 'auto' }}>
          <Outlet />
        </Content>
      </Layout>

      {/* 全局焦虑感知：浮动按钮 + 现实任务弹窗 + 危机干预 */}
      <AnxietyFloatingWidget />
      <RealityTaskModal />
      <CrisisInterventionWidget />
    </Layout>
    </AnxietyProvider>
  );
}
