import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Form, Input, Button, Card, message, Typography } from 'antd';
import { UserOutlined, LockOutlined, HeartFilled } from '@ant-design/icons';
import { authApi } from '../services';
import { useAuthStore } from '../store/authStore';

const { Title, Text } = Typography;

export default function Login() {
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const setAuth = useAuthStore((state) => state.setAuth);

  const onFinish = async (values: { email: string; password: string }) => {
    setLoading(true);
    try {
      const res = await authApi.login(values) as any;
      if (res.code === 0) {
        const { user, accessToken, refreshToken } = res.data;
        setAuth(user, accessToken, refreshToken);
        message.success('欢迎回来 ');
        navigate(user.role === 'ADMIN' ? '/admin' : '/');
      }
    } catch (err: any) {
      message.error(err.message || '登录失败');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      background: 'linear-gradient(135deg, #fff0f3 0%, #f0e6ff 30%, #e8f4f8 60%, #fff8f0 100%)',
      position: 'relative',
      overflow: 'hidden',
    }}>
      {/* 装饰云朵 */}
      <div style={{ position: 'absolute', top: 60, left: 80, fontSize: 80, opacity: 0.15, animation: 'float 4s ease-in-out infinite' }}>☁️</div>
      <div style={{ position: 'absolute', top: 120, right: 120, fontSize: 60, opacity: 0.12, animation: 'float 5s ease-in-out infinite 1s' }}>🌸</div>
      <div style={{ position: 'absolute', bottom: 80, left: 150, fontSize: 50, opacity: 0.1, animation: 'float 6s ease-in-out infinite 2s' }}>️</div>
      <div style={{ position: 'absolute', bottom: 120, right: 80, fontSize: 70, opacity: 0.12, animation: 'float 4.5s ease-in-out infinite 0.5s' }}>🌷</div>

      <Card className="cloud-card" style={{
        width: 420,
        borderRadius: 24,
        boxShadow: '0 20px 60px rgba(255,182,193,0.2)',
        border: '1px solid rgba(255,182,193,0.15)',
        zIndex: 1,
      }}>
        <div style={{ textAlign: 'center', marginBottom: 32 }}>
          <div style={{
            fontSize: 56,
            marginBottom: 8,
            display: 'inline-block',
            animation: 'float 3s ease-in-out infinite',
          }}>🌸</div>
          <Title level={2} style={{ marginBottom: 4, color: '#ff8fab' }}>心灵花园</Title>
          <Text style={{ color: '#b0a0c0', fontSize: 14 }}>
            <HeartFilled style={{ color: '#ffb6c1', marginRight: 4 }} />
            温暖陪伴，从心开始
          </Text>
        </div>

        <Form onFinish={onFinish} size="large">
          <Form.Item name="email" rules={[{ required: true, message: '请输入邮箱' }, { type: 'email', message: '邮箱格式不正确' }]}>
            <Input
              prefix={<UserOutlined style={{ color: '#ffb6c1' }} />}
              placeholder="邮箱"
              className="cloud-input"
              style={{ borderRadius: 12 }}
            />
          </Form.Item>
          <Form.Item name="password" rules={[{ required: true, message: '请输入密码' }]}>
            <Input.Password
              prefix={<LockOutlined style={{ color: '#ffb6c1' }} />}
              placeholder="密码"
              className="cloud-input"
              style={{ borderRadius: 12 }}
            />
          </Form.Item>
          <Form.Item style={{ marginBottom: 0 }}>
            <Button
              type="primary"
              htmlType="submit"
              loading={loading}
              block
              className="cloud-btn"
              style={{
                height: 48,
                borderRadius: 50,
                fontSize: 16,
                fontWeight: 600,
                background: 'linear-gradient(135deg, #ffb6c1 0%, #ffc8d6 50%, #ffd1dc 100%)',
                border: 'none',
                boxShadow: '0 8px 32px rgba(255,182,193,0.35)',
                letterSpacing: 4,
              }}
            >
              进入花园
            </Button>
          </Form.Item>
        </Form>

        <div style={{ textAlign: 'center', marginTop: 20 }}>
          <Text style={{ color: '#b0a0c0', fontSize: 13 }}>
            还没有账号？ <Link to="/register" style={{ color: '#ff8fab', fontWeight: 500 }}>立即注册</Link>
          </Text>
        </div>

        {/* 底部装饰 */}
        <div style={{ textAlign: 'center', marginTop: 24, opacity: 0.4 }}>
          <Text style={{ fontSize: 12, color: '#b0a0c0' }}>
            🌿 每一次倾诉，都是勇敢的开始 🌿
          </Text>
        </div>
      </Card>
    </div>
  );
}
