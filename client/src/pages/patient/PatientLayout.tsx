import { useState, useEffect } from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { Layout, Menu, Avatar, Dropdown, Badge, Space, Typography, Popover, List, Button, Empty, Modal, Input, Rate, message, Drawer } from 'antd';
import {
  HomeOutlined, MessageOutlined, UserOutlined, HeartOutlined,
  TeamOutlined, BellOutlined, LogoutOutlined, SettingOutlined,
  CoffeeOutlined, TrophyOutlined, BulbOutlined,
  SoundOutlined, MenuOutlined, CloseOutlined,
} from '@ant-design/icons';
import { useAuthStore } from '../../store/authStore';
import api from '../../services/api';
import { extraApi } from '../../services';
import { AnxietyProvider } from '../../context/AnxietyContext';
import AnxietyFloatingWidget from '../../components/AnxietyFloatingWidget';
import RealityTaskModal from '../../components/RealityTaskModal';
import CrisisInterventionWidget from '../../components/CrisisInterventionWidget';

const { Header, Sider, Content } = Layout;
const { Text } = Typography;

const menuItems = [
  { key: '/', icon: <HomeOutlined />, label: '首页' },
  { key: '/chat', icon: <MessageOutlined />, label: 'AI 倾诉' },
  { key: '/socratic', icon: <BulbOutlined />, label: '话痨树洞' },
  { key: '/companions', icon: <TeamOutlined />, label: '同伴社区' },
  { key: '/healing', icon: <HeartOutlined />, label: '疗愈空间' },
  { key: '/growth', icon: <TrophyOutlined />, label: '我的成长' },
  { key: '/profile', icon: <UserOutlined />, label: '心理画像' },
  { key: '/experts', icon: <CoffeeOutlined />, label: '我的咨询' },
  { key: '/settings', icon: <SettingOutlined />, label: '设置' },
];

export default function PatientLayout() {
  const navigate = useNavigate();
  const location = useLocation();
  const { user, logout } = useAuthStore();
  const [notificationCount, setNotificationCount] = useState(0);
  const [notifications, setNotifications] = useState<any[]>([]);
  // 反馈弹窗
  const [feedbackOpen, setFeedbackOpen] = useState(false);
  const [feedbackType, setFeedbackType] = useState('suggestion');
  const [feedbackContent, setFeedbackContent] = useState('');
  const [feedbackRating, setFeedbackRating] = useState(0);
  const [feedbackLoading, setFeedbackLoading] = useState(false);
  // 移动端菜单
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);

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

  // 响应式监听
  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 768);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // 提交反馈
  const handleSubmitFeedback = async () => {
    if (!feedbackContent.trim()) return;
    setFeedbackLoading(true);
    try {
      await extraApi.createFeedback({
        type: feedbackType,
        title: feedbackContent.slice(0, 20),
        content: feedbackContent,
      });
      message.success('感谢你的反馈！我们会继续努力 \ud83d\udcaa');
      setFeedbackOpen(false);
      setFeedbackContent('');
      setFeedbackRating(0);
    } catch {
      message.error('发送失败，请稍后再试');
    } finally {
      setFeedbackLoading(false);
    }
  };

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

  // 移动端底部Tab
  const bottomTabs = [
    { key: '/', icon: <HomeOutlined />, label: '首页' },
    { key: '/chat', icon: <MessageOutlined />, label: 'AI倾诉' },
    { key: '/companions', icon: <TeamOutlined />, label: '社区' },
    { key: '/healing', icon: <HeartOutlined />, label: '疗愈' },
    { key: '/settings', icon: <UserOutlined />, label: '我的' },
  ];

  return (
    <AnxietyProvider>
    <Layout style={{ minHeight: '100vh', background: 'transparent' }}>
      {/* 侧边栏 - 仅桌面端显示 */}
      {!isMobile && (
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
      )}

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
            {/* 移动端菜单按钮 */}
            {isMobile && (
              <Button type="text" icon={<MenuOutlined />} onClick={() => setMobileMenuOpen(true)} style={{ color: '#ff8fab' }} />
            )}
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

        <Content style={{ padding: isMobile ? 12 : 24, overflow: 'auto', paddingBottom: isMobile ? 80 : 24 }}>
          <Outlet />
        </Content>
      </Layout>

      {/* 全局焦虑感知：浮动按钮 + 现实任务弹窗 + 危机干预 */}
      <AnxietyFloatingWidget />
      <RealityTaskModal />
      <CrisisInterventionWidget />

      {/* 反馈浮动按钮 */}
      <div
        onClick={() => setFeedbackOpen(true)}
        style={{
          position: 'fixed', bottom: isMobile ? 80 : 32, right: 24, width: 48, height: 48,
          borderRadius: '50%', background: 'linear-gradient(135deg, #ff8fab 0%, #c084fc 100%)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          cursor: 'pointer', boxShadow: '0 4px 12px rgba(255,143,171,0.4)',
          zIndex: 100, transition: 'transform 0.2s',
        }}
        onMouseEnter={(e) => { e.currentTarget.style.transform = 'scale(1.1)'; }}
        onMouseLeave={(e) => { e.currentTarget.style.transform = 'scale(1)'; }}
        title="反馈建议"
      >
        <SoundOutlined style={{ fontSize: 20, color: '#fff' }} />
      </div>

      {/* 移动端底部Tab */}
      {isMobile && (
        <div style={{
          position: 'fixed', bottom: 0, left: 0, right: 0, height: 56,
          background: 'rgba(255,255,255,0.95)', backdropFilter: 'blur(10px)',
          borderTop: '1px solid rgba(255,182,193,0.15)',
          display: 'flex', alignItems: 'center', justifyContent: 'space-around',
          zIndex: 1000, paddingBottom: 'env(safe-area-inset-bottom)',
        }}>
          {bottomTabs.map(tab => {
            const isActive = location.pathname === tab.key;
            return (
              <div
                key={tab.key}
                onClick={() => navigate(tab.key)}
                style={{
                  display: 'flex', flexDirection: 'column', alignItems: 'center',
                  cursor: 'pointer', padding: '4px 0', minWidth: 48,
                  color: isActive ? '#ff8fab' : '#999',
                  transition: 'color 0.2s',
                }}
              >
                <span style={{ fontSize: 20, marginBottom: 2 }}>{tab.icon}</span>
                <span style={{ fontSize: 10 }}>{tab.label}</span>
              </div>
            );
          })}
        </div>
      )}

      {/* 移动端侧边菜单抽屉 */}
      <Drawer
        placement="left"
        width={260}
        open={mobileMenuOpen}
        onClose={() => setMobileMenuOpen(false)}
        styles={{ body: { padding: 0 } }}
      >
        <div style={{ padding: '24px 16px', textAlign: 'center', borderBottom: '1px solid #f0f0f0' }}>
          <div style={{ fontSize: 36, marginBottom: 4 }}>🌸</div>
          <Text strong style={{ fontSize: 16, color: '#ff8fab' }}>心灵花园</Text>
        </div>
        <Menu
          mode="inline"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={({ key }) => { navigate(key); setMobileMenuOpen(false); }}
          style={{ border: 'none' }}
        />
      </Drawer>

      {/* 反馈弹窗 */}
      <Modal
        open={feedbackOpen}
        onCancel={() => setFeedbackOpen(false)}
        title={<Space><SoundOutlined /> 告诉我们你的想法</Space>}
        footer={null}
        width={480}
      >
        <div style={{ padding: '8px 0' }}>
          <div style={{ marginBottom: 16, textAlign: 'center' }}>
            <Text style={{ display: 'block', marginBottom: 8 }}>你觉得这个平台怎么样？</Text>
            <Rate value={feedbackRating} onChange={setFeedbackRating} />
            {feedbackRating > 0 && (
              <Text type="secondary" style={{ marginLeft: 8, fontSize: 13 }}>
                {feedbackRating <= 2 ? '😔 我们会继续改进' :
                 feedbackRating <= 3 ? '🤔 还有进步空间' :
                 feedbackRating <= 4 ? '😊 谢谢你的认可' : '🎉 太开心了！'}
              </Text>
            )}
          </div>
          <div style={{ marginBottom: 12 }}>
            <Space wrap>
              {[
                { key: 'suggestion', label: '💡 建议' },
                { key: 'bug', label: '🐛 问题' },
                { key: 'feature', label: '✨ 新功能' },
                { key: 'other', label: '💬 其他' },
              ].map(({ key, label }) => (
                <div
                  key={key}
                  onClick={() => setFeedbackType(key)}
                  style={{
                    padding: '4px 14px', borderRadius: 20, cursor: 'pointer', fontSize: 13,
                    background: feedbackType === key ? '#fff0f5' : '#f5f5f5',
                    color: feedbackType === key ? '#ff8fab' : '#666',
                    border: feedbackType === key ? '1px solid #ff8fab' : '1px solid transparent',
                  }}
                >
                  {label}
                </div>
              ))}
            </Space>
          </div>
          <Input.TextArea
            rows={4}
            placeholder="告诉我们你的想法，每一条反馈都会认真看~"
            value={feedbackContent}
            onChange={(e) => setFeedbackContent(e.target.value)}
            style={{ borderRadius: 12, marginBottom: 12 }}
          />
          <Button
            type="primary"
            block
            loading={feedbackLoading}
            disabled={!feedbackContent.trim()}
            onClick={handleSubmitFeedback}
            style={{ borderRadius: 24, background: 'linear-gradient(135deg, #ff8fab 0%, #c084fc 100%)', border: 'none' }}
          >
            提交反馈
          </Button>
        </div>
      </Modal>
    </Layout>
    </AnxietyProvider>
  );
}
