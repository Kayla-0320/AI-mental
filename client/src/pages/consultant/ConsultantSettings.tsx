import { useState, useEffect } from 'react';
import { Card, Typography, Form, Input, Button, Avatar, message, Divider, Space, Tag, Switch, TimePicker, Select, InputNumber } from 'antd';
import { UserOutlined, LockOutlined, MailOutlined, BellOutlined, ClockCircleOutlined, EditOutlined, DollarOutlined, StarOutlined } from '@ant-design/icons';
import { useAuthStore } from '../../store/authStore';
import { expertApi } from '../../services';
import api from '../../services/api';

const { Title, Text } = Typography;

export default function ConsultantSettings() {
  const { user, updateUser } = useAuthStore();
  const [form] = Form.useForm();
  const [passwordForm] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [passwordLoading, setPasswordLoading] = useState(false);
  const [notifyEnabled, setNotifyEnabled] = useState(true);
  const [profileForm] = Form.useForm();
  const [profileLoading, setProfileLoading] = useState(false);
  const [myProfile, setMyProfile] = useState<any>(null);

  useEffect(() => {
    if (user) {
      form.setFieldsValue({ nickname: user.nickname, email: user.email });
      loadMyProfile();
    }
  }, [user]);

  const loadMyProfile = async () => {
    try {
      const res = await expertApi.getMyProfile() as any;
      const data = res.data;
      if (data) {
        setMyProfile(data);
        let specialties: string[] = [];
        try { specialties = JSON.parse(data.specialties || '[]'); } catch {}
        profileForm.setFieldsValue({
          title: data.title,
          specialties: specialties.join('、'),
          introduction: data.introduction,
          pricePerSession: Number(data.pricePerSession),
        });
      }
    } catch {
      // 档案不存在，忽略
    }
  };

  const handleSaveConsultantProfile = async (values: any) => {
    setProfileLoading(true);
    try {
      const specialtiesArr = (values.specialties || '').split(/[、，,\s]+/).filter(Boolean);
      await expertApi.updateMyProfile({
        title: values.title,
        specialties: JSON.stringify(specialtiesArr),
        introduction: values.introduction,
        pricePerSession: values.pricePerSession,
      });
      message.success('职业档案已更新');
    } catch {
      message.error('更新失败');
    } finally {
      setProfileLoading(false);
    }
  };

  const handleSaveProfile = async (values: any) => {
    setLoading(true);
    try {
      const res = await api.put('/auth/profile', { nickname: values.nickname, email: values.email }) as any;
      if (res.data?.user) updateUser(res.data.user);
      message.success('个人信息已更新');
    } catch { message.error('更新失败'); }
    finally { setLoading(false); }
  };

  const handleChangePassword = async (values: any) => {
    setPasswordLoading(true);
    try {
      await api.put('/auth/password', { oldPassword: values.oldPassword, newPassword: values.newPassword });
      message.success('密码修改成功');
      passwordForm.resetFields();
    } catch { message.error('密码修改失败'); }
    finally { setPasswordLoading(false); }
  };

  return (
    <div style={{ maxWidth: 700, margin: '0 auto' }}>
      <Title level={4} style={{ marginBottom: 24 }}>咨询师设置</Title>

      <Card style={{ borderRadius: 12, marginBottom: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
          <Avatar size={72} style={{ backgroundColor: '#1890ff' }} icon={<UserOutlined />}>
            {user?.nickname?.[0]}
          </Avatar>
          <div>
            <Title level={5} style={{ marginBottom: 4 }}>{user?.nickname}</Title>
            <Text type="secondary">{user?.email}</Text>
            <div style={{ marginTop: 8 }}><Tag color="green">咨询师</Tag></div>
          </div>
        </div>
      </Card>

      <Card title="基本信息" style={{ borderRadius: 12, marginBottom: 16 }}>
        <Form form={form} layout="vertical" onFinish={handleSaveProfile}>
          <Form.Item name="nickname" label="昵称" rules={[{ required: true }]}>
            <Input prefix={<UserOutlined />} style={{ borderRadius: 12 }} />
          </Form.Item>
          <Form.Item name="email" label="邮箱" rules={[{ type: 'email' }]}>
            <Input prefix={<MailOutlined />} style={{ borderRadius: 12 }} />
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit" loading={loading} style={{ borderRadius: 24 }}>保存修改</Button>
          </Form.Item>
        </Form>
      </Card>

      <Card
        title={<Space><EditOutlined /> 职业档案（患者可见）</Space>}
        style={{ borderRadius: 12, marginBottom: 16 }}
      >
        <Form form={profileForm} layout="vertical" onFinish={handleSaveConsultantProfile}>
          <Form.Item name="title" label="职称" rules={[{ required: true, message: '请填写职称' }]}>
            <Select placeholder="如：资深心理咨询师" style={{ borderRadius: 12 }} options={[
              { value: '初级心理咨询师', label: '初级心理咨询师' },
              { value: '中级心理咨询师', label: '中级心理咨询师' },
              { value: '资深心理咨询师', label: '资深心理咨询师' },
              { value: '心理治疗师', label: '心理治疗师' },
              { value: '精神科医师', label: '精神科医师' },
            ]} />
          </Form.Item>
          <Form.Item name="specialties" label="擅长领域" rules={[{ required: true, message: '请填写擅长领域' }]}>
            <Input placeholder="用顿号分隔，如：焦虑、抑郁、人际关系" style={{ borderRadius: 12 }} />
          </Form.Item>
          <Form.Item name="introduction" label="个人介绍" rules={[{ required: true, message: '请填写个人介绍' }]}>
            <Input.TextArea rows={4} placeholder="介绍你的从业经历、擅长疗法、帮助过的人群等..." maxLength={500} showCount style={{ borderRadius: 12 }} />
          </Form.Item>
          <Form.Item name="pricePerSession" label="每次咨询价格（元）" rules={[{ required: true, message: '请填写价格' }]}>
            <InputNumber min={0} max={9999} prefix={<DollarOutlined />} style={{ width: '100%', borderRadius: 12 }} />
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit" loading={profileLoading} style={{ borderRadius: 24, background: '#1890ff', borderColor: '#1890ff' }}>保存职业档案</Button>
          </Form.Item>
        </Form>
      </Card>

      <Card title="修改密码" style={{ borderRadius: 12, marginBottom: 16 }}>
        <Form form={passwordForm} layout="vertical" onFinish={handleChangePassword}>
          <Form.Item name="oldPassword" label="原密码" rules={[{ required: true }]}>
            <Input.Password prefix={<LockOutlined />} style={{ borderRadius: 12 }} />
          </Form.Item>
          <Form.Item name="newPassword" label="新密码" rules={[{ required: true }, { min: 6 }]}>
            <Input.Password prefix={<LockOutlined />} style={{ borderRadius: 12 }} />
          </Form.Item>
          <Form.Item name="confirmPassword" label="确认新密码" dependencies={['newPassword']}
            rules={[{ required: true }, ({ getFieldValue }) => ({
              validator(_, value) {
                if (!value || getFieldValue('newPassword') === value) return Promise.resolve();
                return Promise.reject(new Error('两次密码不一致'));
              },
            })]}>
            <Input.Password prefix={<LockOutlined />} style={{ borderRadius: 12 }} />
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit" loading={passwordLoading} style={{ borderRadius: 24 }}>修改密码</Button>
          </Form.Item>
        </Form>
      </Card>

      <Card title="偏好设置" style={{ borderRadius: 12, marginBottom: 16 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Space>
            <BellOutlined style={{ fontSize: 18, color: '#1890ff' }} />
            <div>
              <Text>消息通知</Text>
              <div><Text type="secondary" style={{ fontSize: 12 }}>接收预约提醒和来访者消息</Text></div>
            </div>
          </Space>
          <Switch checked={notifyEnabled} onChange={setNotifyEnabled} />
        </div>
      </Card>
    </div>
  );
}
