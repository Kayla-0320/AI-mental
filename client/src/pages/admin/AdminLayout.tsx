import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { Layout, Menu, Avatar, Dropdown, Space, Typography } from 'antd';
import {
  DashboardOutlined, UserOutlined, TeamOutlined, MessageOutlined,
  FileTextOutlined, BarChartOutlined, LogoutOutlined, SettingOutlined,
} from '@ant-design/icons';
import { useAuthStore } from '../../store/authStore';

const { Header, Sider, Content } = Layout;
const { Text } = Typography;

const menuItems = [
  { key: '/admin', icon: <DashboardOutlined />, label: '数据概览' },
  { key: '/admin/users', icon: <UserOutlined />, label: '用户管理' },
  { key: '/admin/consultants', icon: <TeamOutlined />, label: '咨询师管理' },
  { key: '/admin/consultations', icon: <MessageOutlined />, label: '咨询记录' },
  { key: '/admin/assessments', icon: <FileTextOutlined />, label: '测评管理' },
  { key: '/admin/stats', icon: <BarChartOutlined />, label: '统计分析' },
];

export default function AdminLayout() {
  const navigate = useNavigate();
  const location = useLocation();
  const { user, logout } = useAuthStore();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const userMenu = {
    items: [
      { key: 'settings', icon: <SettingOutlined />, label: '设置' },
      { type: 'divider' as const },
      { key: 'logout', icon: <LogoutOutlined />, label: '退出登录', onClick: handleLogout },
    ],
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider width={220} style={{ background: '#001529' }}>
        <div style={{ padding: '20px 16px', textAlign: 'center', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
          <Text strong style={{ fontSize: 16, color: '#fff' }}>管理后台</Text>
        </div>
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={({ key }) => navigate(key)}
        />
      </Sider>

      <Layout>
        <Header style={{
          background: '#fff', padding: '0 24px',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          borderBottom: '1px solid #f0f0f0', height: 60,
        }}>
          <Text style={{ fontSize: 16, fontWeight: 500 }}>心理健康平台 - 管理后台</Text>
          <Dropdown menu={userMenu}>
            <Space style={{ cursor: 'pointer' }}>
              <Avatar style={{ backgroundColor: '#001529' }} icon={<UserOutlined />} />
              <Text>{user?.nickname}</Text>
            </Space>
          </Dropdown>
        </Header>

        <Content style={{ padding: 24, background: '#f5f7fa', overflow: 'auto' }}>
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  );
}
