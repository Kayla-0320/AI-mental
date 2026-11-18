import { useState, useEffect } from 'react';
import { Card, Typography, Form, Input, Button, Avatar, message, Divider, Space, Tag, Switch, Alert, Tooltip, Progress, Rate, Modal, Spin, Empty } from 'antd';
import { UserOutlined, LockOutlined, MailOutlined, BellOutlined, MoonOutlined, SafetyCertificateOutlined, EyeOutlined, EyeInvisibleOutlined, DeleteOutlined, DownloadOutlined, InfoCircleOutlined, MessageOutlined, SoundOutlined, ExclamationCircleOutlined } from '@ant-design/icons';
import { useAuthStore } from '../../store/authStore';
import api from '../../services/api';
import { privacyApi } from '../../services';

const { Title, Text, Paragraph } = Typography;

export default function Settings() {
  const { user, updateUser } = useAuthStore();
  const [form] = Form.useForm();
  const [passwordForm] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [passwordLoading, setPasswordLoading] = useState(false);
  const [notifyEnabled, setNotifyEnabled] = useState(true);
  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem('darkMode') === 'true';
  });

  // 深色模式实际生效
  useEffect(() => {
    if (darkMode) {
      document.body.classList.add('dark-mode');
      document.body.style.filter = 'invert(0.88) hue-rotate(180deg)';
      document.body.style.background = '#1a1a2e';
    } else {
      document.body.classList.remove('dark-mode');
      document.body.style.filter = '';
      document.body.style.background = '';
    }
    localStorage.setItem('darkMode', String(darkMode));
    return () => {
      document.body.style.filter = '';
      document.body.style.background = '';
    };
  }, [darkMode]);
  // ── 数据控制中心（真实后端：/api/privacy）────────────────────────
  // 修复前：privacySettings 是组件内的 useState 字面量（刷新即回默认），
  // 「导出我的数据 / 删除所有数据」两个按钮没有任何 onClick。
  // 现在每一项都读写数据库；加载失败时显示失败态，而不是退回一份看起来正常的默认值。
  const [consentItems, setConsentItems] = useState<any[]>([]);
  const [consentSummary, setConsentSummary] = useState<any>(null);
  const [consentLoading, setConsentLoading] = useState(true);
  const [consentError, setConsentError] = useState(false);
  const [consentSaving, setConsentSaving] = useState<string | null>(null);
  const [retentionDays, setRetentionDays] = useState(90);
  const [dataSummary, setDataSummary] = useState<any>(null);
  const [exporting, setExporting] = useState(false);
  const [eraseOpen, setEraseOpen] = useState(false);
  const [eraseScope, setEraseScope] = useState<'content' | 'account'>('content');
  const [eraseConfirmText, setEraseConfirmText] = useState('');
  const [erasing, setErasing] = useState(false);
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

  // ── 数据控制中心：加载 ──────────────────────────────────────────
  const loadPrivacy = async () => {
    setConsentLoading(true);
    try {
      const [consent, retention, summary] = await Promise.all([
        privacyApi.getConsent() as any,
        privacyApi.getRetention() as any,
        privacyApi.getSummary() as any,
      ]);
      setConsentItems(consent.data?.items || []);
      setConsentSummary(consent.data?.summary || null);
      setRetentionDays(retention.data?.retentionDays ?? 90);
      setDataSummary(summary.data || null);
      setConsentError(false);
    } catch {
      // 失败就说失败：不要退回一份「看起来正常」的默认授权状态
      setConsentError(true);
    } finally {
      setConsentLoading(false);
    }
  };

  useEffect(() => { loadPrivacy(); }, []);

  /** 切换一项授权：先落库，成功后再更新界面（不做乐观更新，避免界面与库不一致） */
  const toggleConsent = async (item: any, granted: boolean) => {
    const key = `${item.dataType}::${item.purpose}`;
    setConsentSaving(key);
    try {
      const res = await privacyApi.setConsent([
        { dataType: item.dataType, purpose: item.purpose, granted },
      ]) as any;
      setConsentItems(res.data?.items || []);
      setConsentSummary(res.data?.summary || null);
      message.success(granted ? `已开启：${item.label}` : `已关闭：${item.label}`);
    } catch (e: any) {
      message.error(e?.message || '保存失败，请重试');
    } finally {
      setConsentSaving(null);
    }
  };

  const handleRetention = async (days: number) => {
    try {
      await privacyApi.setRetention(days);
      setRetentionDays(days);
      message.success(`数据保留期限已设为 ${days} 天`);
    } catch (e: any) {
      message.error(e?.message || '设置失败');
    }
  };

  /** 导出：后端返回 JSON，这里组装成文件下载 */
  const handleExport = async () => {
    setExporting(true);
    try {
      const res = await privacyApi.exportData() as any;
      const payload = res.data;
      if (!payload) throw new Error('导出内容为空');
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `我的数据-${new Date().toISOString().slice(0, 10)}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      message.success('数据已导出为 JSON 文件');
    } catch (e: any) {
      message.error(e?.message || '导出失败，请重试');
    } finally {
      setExporting(false);
    }
  };

  const handleErase = async () => {
    if (eraseConfirmText.trim() !== '删除我的数据') {
      message.warning('请输入「删除我的数据」以确认');
      return;
    }
    setErasing(true);
    try {
      const res = await privacyApi.eraseData(eraseScope, eraseConfirmText.trim()) as any;
      const deleted = res.data?.deleted || {};
      const total = Object.values(deleted).reduce((s: number, v: any) => s + (Number(v) || 0), 0);
      setEraseOpen(false);
      setEraseConfirmText('');
      if (eraseScope === 'account') {
        message.success('账号与全部数据已删除，即将退出登录');
        setTimeout(() => {
          localStorage.removeItem('accessToken');
          localStorage.removeItem('refreshToken');
          window.location.href = '/login';
        }, 1500);
      } else {
        message.success(`已删除 ${total} 条个人数据（账号保留）`);
        await loadPrivacy();
      }
    } catch (e: any) {
      message.error(e?.message || '删除失败，请重试');
    } finally {
      setErasing(false);
    }
  };

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

  // 授权项按 group 分组展示（顺序取目录返回顺序，前端不另排）
  const consentGroups: string[] = [];
  for (const item of consentItems) {
    if (!consentGroups.includes(item.group)) consentGroups.push(item.group);
  }

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

      {/* 隐私与数据控制 —— 读写真实后端 /api/privacy */}
      <Card
        title={<Space><SafetyCertificateOutlined /> 隐私与数据控制</Space>}
        style={{ borderRadius: 12, marginBottom: 16 }}
      >
        <Alert
          type="info"
          showIcon
          message="你的数据你做主"
          description="下面每一项都可以单独开关，改动会立即保存到你的账号。感知数据（语音 / 摄像头 / 行为 / 睡眠）默认关闭，需要你主动开启；对话文本与量表结果用于基础服务。"
          style={{ marginBottom: 20, borderRadius: 8 }}
        />

        {consentError ? (
          <Alert
            type="error"
            showIcon
            style={{ marginBottom: 16, borderRadius: 8 }}
            message="授权状态加载失败"
            description="在成功加载之前，请勿假设这里的开关反映了你的真实授权状态。"
            action={<Button size="small" onClick={loadPrivacy}>重试</Button>}
          />
        ) : consentLoading ? (
          <div style={{ textAlign: 'center', padding: 24 }}><Spin /></div>
        ) : (
          <>
            {consentGroups.map((group) => (
              <div key={group} style={{ marginBottom: 8 }}>
                <Text strong style={{ fontSize: 13, color: '#8c8c8c' }}>{group}</Text>
                <div style={{ marginTop: 8 }}>
                  {consentItems.filter((i) => i.group === group).map((item, idx) => {
                    const key = `${item.dataType}::${item.purpose}`;
                    return (
                      <div key={key}>
                        {idx > 0 && <Divider style={{ margin: '12px 0' }} />}
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 12 }}>
                          <Space align="start">
                            {item.group === '基础服务' && <EyeOutlined style={{ fontSize: 18, color: '#1890ff', marginTop: 2 }} />}
                            {item.group === '共享范围' && <EyeOutlined style={{ fontSize: 18, color: '#52c41a', marginTop: 2 }} />}
                            {item.group === '感知数据' && <EyeOutlined style={{ fontSize: 18, color: '#fa8c16', marginTop: 2 }} />}
                            {item.group === '其他用途' && <EyeInvisibleOutlined style={{ fontSize: 18, color: '#722ed1', marginTop: 2 }} />}
                            <div>
                              <Text>
                                {item.label}
                                {item.required && (
                                  <Tag color="blue" style={{ marginLeft: 6, fontSize: 11 }}>必要</Tag>
                                )}
                              </Text>
                              <div><Text type="secondary" style={{ fontSize: 12 }}>{item.description}</Text></div>
                              {item.required && item.requiredReason && (
                                <div><Text type="secondary" style={{ fontSize: 11, color: '#8c8c8c' }}>依据：{item.requiredReason}</Text></div>
                              )}
                              {!item.granted && item.revokedAt && (
                                <div><Text type="secondary" style={{ fontSize: 11 }}>已于 {new Date(item.revokedAt).toLocaleDateString('zh-CN')} 关闭</Text></div>
                              )}
                            </div>
                          </Space>
                          <Tooltip title={item.required ? '该项为生命健康保护的法律例外，不可关闭' : ''}>
                            <Switch
                              checked={item.granted}
                              disabled={item.required}
                              loading={consentSaving === key}
                              onChange={(v) => toggleConsent(item, v)}
                            />
                          </Tooltip>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </>
        )}

        <Divider style={{ margin: '16px 0' }} />

        <div style={{ marginBottom: 20 }}>
          <Text strong style={{ display: 'block', marginBottom: 8 }}>
            数据保留期限
            <Tooltip title="超过此期限的对话与记录会被自动删除；量表结果与危机干预记录不在此范围内">
              <InfoCircleOutlined style={{ marginLeft: 4, color: '#bbb' }} />
            </Tooltip>
          </Text>
          <Space>
            {[30, 90, 180, 365].map(days => (
              <Tag
                key={days}
                color={retentionDays === days ? 'blue' : 'default'}
                style={{ cursor: 'pointer', padding: '4px 12px', fontSize: 13 }}
                onClick={() => handleRetention(days)}
              >
                {days} 天
              </Tag>
            ))}
          </Space>
          <div><Text type="secondary" style={{ fontSize: 12 }}>当前设置：{retentionDays} 天</Text></div>
        </div>

        <Divider style={{ margin: '12px 0' }} />

        <div style={{ marginBottom: 16 }}>
          <Text strong style={{ display: 'block', marginBottom: 8 }}>平台保存了什么</Text>
          {!dataSummary ? (
            <Text type="secondary" style={{ fontSize: 12 }}>加载中…</Text>
          ) : (
            <>
              <Space wrap size={[6, 6]}>
                {(dataSummary.items || []).map((it: any) => (
                  <Tag key={it.key} style={{ fontSize: 12 }}>{it.label}：{it.count}</Tag>
                ))}
              </Space>
              <div style={{ marginTop: 8 }}>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  {dataSummary.notes?.rawMedia}
                </Text>
                <div>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {dataSummary.notes?.crisisRecords}
                  </Text>
                </div>
              </div>
            </>
          )}
        </div>

        <Divider style={{ margin: '12px 0' }} />

        <Space size="middle" wrap>
          <Button icon={<DownloadOutlined />} loading={exporting} style={{ borderRadius: 24 }} onClick={handleExport}>
            导出我的数据
          </Button>
          <Button danger icon={<DeleteOutlined />} style={{ borderRadius: 24 }} onClick={() => { setEraseConfirmText(''); setEraseOpen(true); }}>
            删除所有数据
          </Button>
        </Space>
      </Card>

      {/* 删除数据二次确认 */}
      <Modal
        open={eraseOpen}
        onCancel={() => setEraseOpen(false)}
        title={<Space><ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />删除数据</Space>}
        okText="确认删除"
        okButtonProps={{ danger: true, disabled: eraseConfirmText.trim() !== '删除我的数据', loading: erasing }}
        cancelText="取消"
        onOk={handleErase}
      >
        <Alert
          type="error"
          showIcon
          style={{ marginBottom: 16, borderRadius: 8 }}
          message="这是不可恢复的删除操作"
          description="删除后数据不会保留在任何可查询的位置，也无法通过任何界面恢复。建议先「导出我的数据」留一份。"
        />
        <div style={{ marginBottom: 12 }}>
          <Text strong style={{ display: 'block', marginBottom: 8 }}>删除范围</Text>
          <Space direction="vertical">
            <Tag.CheckableTag
              checked={eraseScope === 'content'}
              onChange={() => setEraseScope('content')}
            >
              仅清空个人内容（保留账号）
            </Tag.CheckableTag>
            <Tag.CheckableTag
              checked={eraseScope === 'account'}
              onChange={() => setEraseScope('account')}
            >
              删除整个账号（含登录信息）
            </Tag.CheckableTag>
          </Space>
        </div>
        <Text strong style={{ display: 'block', marginBottom: 8 }}>
          请输入「删除我的数据」以确认
        </Text>
        <Input
          value={eraseConfirmText}
          onChange={(e) => setEraseConfirmText(e.target.value)}
          placeholder="删除我的数据"
          style={{ borderRadius: 8 }}
        />
      </Modal>

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
