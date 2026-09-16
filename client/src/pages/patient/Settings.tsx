import { useState, useEffect } from 'react';
import { Card, Typography, Form, Input, Button, Avatar, message, Divider, Space, Tag, Switch, Alert, Tooltip, Progress, Rate } from 'antd';
import { UserOutlined, LockOutlined, MailOutlined, BellOutlined, MoonOutlined, SafetyCertificateOutlined, EyeOutlined, EyeInvisibleOutlined, DeleteOutlined, DownloadOutlined, InfoCircleOutlined, MessageOutlined, SoundOutlined } from '@ant-design/icons';
import { useAuthStore } from '../../store/authStore';
import api from '../../services/api';

const { Title, Text, Paragraph } = Typography;

export default function Settings() {
  const { user, updateUser } = useAuthStore();
  const [form] = Form.useForm();
  const [passwordForm] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [passwordLoading, setPasswordLoading] = useState(false);
  const [notifyEnabled, setNotifyEnabled] = useState(true);
  const [darkMode, setDarkMode] = useState(false);
  const [privacySettings, setPrivacySettings] = useState({
    sharePeerComparison: true,
    shareResearch: false,
    cameraPermission: false,
    microphonePermission: false,
    dataRetention: '90',  // days
  });
  const [feedbackType, setFeedbackType] = useState('suggestion');
  const [feedbackContent, setFeedbackContent] = useState('');
  const [feedbackRating, setFeedbackRating] = useState(0);
  const [feedbackLoading, setFeedbackLoading] = useState(false);

  useEffect(() => {
    if (user) {
      form.setFieldsValue({
        nickname: user.nickname,
        email: user.email,
      });
    }
  }, [user]);

  const handleSaveProfile = async (values: any) => {
    setLoading(true);
    try {
      const res = await api.put('/auth/profile', {
        nickname: values.nickname,
        email: values.email,
      }) as any;
      if (res.data?.user) {
        updateUser(res.data.user);
      }
      message.success('个人信息已更新');
    } catch {
      message.error('更新失败');
    } finally {
      setLoading(false);
    }
  };

  const handleChangePassword = async (values: any) => {
    setPasswordLoading(true);
    try {
      await api.put('/auth/password', {
        oldPassword: values.oldPassword,
        newPassword: values.newPassword,
      });
      message.success('密码修改成功');
      passwordForm.resetFields();
    } catch {
      message.error('密码修改失败，请检查原密码');
    } finally {
      setPasswordLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 700, margin: '0 auto' }}>
      <Title level={4} style={{ marginBottom: 24 }}>个人设置</Title>

      {/* 头像区域 */}
      <Card style={{ borderRadius: 12, marginBottom: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
          <Avatar size={72} style={{ backgroundColor: '#ffb6c1', boxShadow: '0 2px 8px rgba(255,182,193,0.3)' }}
            icon={<UserOutlined />} src={user?.avatar}>
            {user?.nickname?.[0]}
          </Avatar>
          <div>
            <Title level={5} style={{ marginBottom: 4 }}>{user?.nickname}</Title>
            <Text type="secondary">{user?.email}</Text>
            <div style={{ marginTop: 8 }}>
              <Tag color={user?.role === 'PATIENT' ? 'blue' : user?.role === 'CONSULTANT' ? 'green' : 'red'}>
                {user?.role === 'PATIENT' ? '用户' : user?.role === 'CONSULTANT' ? '咨询师' : '管理员'}
              </Tag>
            </div>
          </div>
        </div>
      </Card>

      {/* 基本信息 */}
      <Card title="基本信息" style={{ borderRadius: 12, marginBottom: 16 }}>
        <Form form={form} layout="vertical" onFinish={handleSaveProfile}>
          <Form.Item name="nickname" label="昵称" rules={[{ required: true, message: '请输入昵称' }]}>
            <Input prefix={<UserOutlined />} placeholder="你的昵称" style={{ borderRadius: 12 }} />
          </Form.Item>
          <Form.Item name="email" label="邮箱" rules={[{ type: 'email', message: '邮箱格式不正确' }]}>
            <Input prefix={<MailOutlined />} placeholder="你的邮箱" style={{ borderRadius: 12 }} />
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit" loading={loading} style={{ borderRadius: 24 }}>
              保存修改
            </Button>
          </Form.Item>
        </Form>
      </Card>

      {/* 修改密码 */}
      <Card title="修改密码" style={{ borderRadius: 12, marginBottom: 16 }}>
        <Form form={passwordForm} layout="vertical" onFinish={handleChangePassword}>
          <Form.Item name="oldPassword" label="原密码" rules={[{ required: true, message: '请输入原密码' }]}>
            <Input.Password prefix={<LockOutlined />} placeholder="输入原密码" style={{ borderRadius: 12 }} />
          </Form.Item>
          <Form.Item name="newPassword" label="新密码" rules={[
            { required: true, message: '请输入新密码' },
            { min: 6, message: '密码至少6个字符' },
          ]}>
            <Input.Password prefix={<LockOutlined />} placeholder="输入新密码" style={{ borderRadius: 12 }} />
          </Form.Item>
          <Form.Item name="confirmPassword" label="确认新密码" dependencies={['newPassword']}
            rules={[
              { required: true, message: '请确认新密码' },
              ({ getFieldValue }) => ({
                validator(_, value) {
                  if (!value || getFieldValue('newPassword') === value) return Promise.resolve();
                  return Promise.reject(new Error('两次密码不一致'));
                },
              }),
            ]}>
            <Input.Password prefix={<LockOutlined />} placeholder="再次输入新密码" style={{ borderRadius: 12 }} />
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit" loading={passwordLoading} style={{ borderRadius: 24 }}>
              修改密码
            </Button>
          </Form.Item>
        </Form>
      </Card>

      {/* 偏好设置 */}
      <Card title="偏好设置" style={{ borderRadius: 12, marginBottom: 16 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <Space>
            <BellOutlined style={{ fontSize: 18, color: '#ff8fab' }} />
            <div>
              <Text>消息通知</Text>
              <div><Text type="secondary" style={{ fontSize: 12 }}>接收系统消息和咨询提醒</Text></div>
            </div>
          </Space>
          <Switch checked={notifyEnabled} onChange={setNotifyEnabled} />
        </div>
        <Divider style={{ margin: '12px 0' }} />
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Space>
            <MoonOutlined style={{ fontSize: 18, color: '#722ed1' }} />
            <div>
              <Text>深色模式</Text>
              <div><Text type="secondary" style={{ fontSize: 12 }}>减少屏幕蓝光，保护眼睛</Text></div>
            </div>
          </Space>
          <Switch checked={darkMode} onChange={setDarkMode} />
        </div>
      </Card>

      {/* 隐私控制 */}
      <Card
        title={<Space><SafetyCertificateOutlined /> 隐私与数据控制</Space>}
        style={{ borderRadius: 12, marginBottom: 16 }}
      >
        <Alert
          type="info"
          showIcon
          message="你的数据你做主"
          description="所有感知数据（键盘节奏、眨眼频率）仅在前端本地处理，不会上传原始数据到服务器。你可以随时控制哪些功能被启用。"
          style={{ marginBottom: 20, borderRadius: 8 }}
        />

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <Space>
            <EyeOutlined style={{ fontSize: 18, color: '#1890ff' }} />
            <div>
              <Text>同龄对比分析</Text>
              <div><Text type="secondary" style={{ fontSize: 12 }}>允许使用你的情绪数据与同龄人对比（仅使用匿名聚合数据）</Text></div>
            </div>
          </Space>
          <Switch
            checked={privacySettings.sharePeerComparison}
            onChange={(v) => setPrivacySettings({ ...privacySettings, sharePeerComparison: v })}
          />
        </div>
        <Divider style={{ margin: '12px 0' }} />

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <Space>
            <EyeInvisibleOutlined style={{ fontSize: 18, color: '#722ed1' }} />
            <div>
              <Text>参与研究计划</Text>
              <div><Text type="secondary" style={{ fontSize: 12 }}>允许匿名化数据用于心理健康研究（完全脱敏，无法识别身份）</Text></div>
            </div>
          </Space>
          <Switch
            checked={privacySettings.shareResearch}
            onChange={(v) => setPrivacySettings({ ...privacySettings, shareResearch: v })}
          />
        </div>
        <Divider style={{ margin: '12px 0' }} />

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <Space>
            <EyeOutlined style={{ fontSize: 18, color: '#52c41a' }} />
            <div>
              <Text>摄像头访问</Text>
              <div><Text type="secondary" style={{ fontSize: 12 }}>允许眨眼频率检测使用摄像头（仅前端本地处理）</Text></div>
            </div>
          </Space>
          <Switch
            checked={privacySettings.cameraPermission}
            onChange={(v) => setPrivacySettings({ ...privacySettings, cameraPermission: v })}
          />
        </div>
        <Divider style={{ margin: '12px 0' }} />

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <Space>
            <EyeOutlined style={{ fontSize: 18, color: '#fa8c16' }} />
            <div>
              <Text>麦克风访问</Text>
              <div><Text type="secondary" style={{ fontSize: 12 }}>允许语音声学特征提取使用麦克风（仅前端本地处理）</Text></div>
            </div>
          </Space>
          <Switch
            checked={privacySettings.microphonePermission}
            onChange={(v) => setPrivacySettings({ ...privacySettings, microphonePermission: v })}
          />
        </div>
        <Divider style={{ margin: '12px 0' }} />

        <div style={{ marginBottom: 20 }}>
          <Text strong style={{ display: 'block', marginBottom: 8 }}>
            数据保留期限
            <Tooltip title="超过此期限的数据将自动删除">
              <InfoCircleOutlined style={{ marginLeft: 4, color: '#bbb' }} />
            </Tooltip>
          </Text>
          <Space>
            {['30', '90', '180', '365'].map(days => (
              <Tag
                key={days}
                color={privacySettings.dataRetention === days ? 'blue' : 'default'}
                style={{ cursor: 'pointer', padding: '4px 12px', fontSize: 13 }}
                onClick={() => setPrivacySettings({ ...privacySettings, dataRetention: days })}
              >
                {days} 天
              </Tag>
            ))}
          </Space>
        </div>

        <Divider style={{ margin: '12px 0' }} />

        <div style={{ marginBottom: 16 }}>
          <Text strong style={{ display: 'block', marginBottom: 8 }}>数据使用透明度</Text>
          <Progress
            percent={privacySettings.sharePeerComparison ? 50 : 25}
            strokeColor="#52c41a"
            format={() => `${privacySettings.sharePeerComparison ? 50 : 25}% 数据被使用`}
            style={{ marginBottom: 8 }}
          />
          <Text type="secondary" style={{ fontSize: 12 }}>
            当前仅使用 {privacySettings.sharePeerComparison ? '情绪画像 + 同龄对比' : '基本服务所需'} 数据
          </Text>
        </div>

        <Divider style={{ margin: '12px 0' }} />

        <Space size="middle">
          <Button icon={<DownloadOutlined />} style={{ borderRadius: 24 }}>
            导出我的数据
          </Button>
          <Button danger icon={<DeleteOutlined />} style={{ borderRadius: 24 }}>
            删除所有数据
          </Button>
        </Space>
      </Card>

      {/* 意见反馈 */}
      <Card
        title={<Space><SoundOutlined /> 意见反馈</Space>}
        style={{ borderRadius: 12, marginBottom: 16 }}
      >
        <div style={{ marginBottom: 16 }}>
          <Text strong style={{ display: 'block', marginBottom: 8 }}>你觉得这个平台怎么样？</Text>
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
          <Text strong style={{ display: 'block', marginBottom: 8 }}>反馈类型</Text>
          <Space wrap>
            {[
              { key: 'suggestion', label: '💡 建议' },
              { key: 'bug', label: '🐛 问题反馈' },
              { key: 'feature', label: '✨ 新功能需求' },
              { key: 'other', label: '💬 其他' },
            ].map(({ key, label }) => (
              <Tag
                key={key}
                color={feedbackType === key ? 'pink' : 'default'}
                style={{ cursor: 'pointer', padding: '4px 12px', borderRadius: 20 }}
                onClick={() => setFeedbackType(key)}
              >
                {label}
              </Tag>
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
          loading={feedbackLoading}
          disabled={!feedbackContent.trim()}
          onClick={async () => {
            setFeedbackLoading(true);
            try {
              await api.post('/feedback', {
                type: feedbackType,
                content: feedbackContent,
                rating: feedbackRating,
              });
              message.success('感谢你的反馈！我们会继续努力 💪');
              setFeedbackContent('');
              setFeedbackRating(0);
            } catch {
              message.error('发送失败，请稍后再试');
            } finally {
              setFeedbackLoading(false);
            }
          }}
          style={{ borderRadius: 24 }}
        >
          提交反馈
        </Button>
      </Card>
    </div>
  );
}
