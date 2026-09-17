import { useState, useEffect } from 'react';
import { Card, Typography, Table, Tag, Space, Button, Modal, Alert, Timeline, Statistic, Row, Col, Badge, Tooltip, message } from 'antd';
import {
  WarningOutlined, PhoneOutlined, SafetyCertificateOutlined,
  CheckCircleOutlined, ClockCircleOutlined, UserOutlined,
  AlertOutlined, NotificationOutlined,
} from '@ant-design/icons';
import { crisisApi } from '../../services';

const { Title, Text, Paragraph } = Typography;

const severityColors: Record<string, string> = { LOW: '#52c41a', MEDIUM: '#faad14', HIGH: '#ff4d4f', CRITICAL: '#cf1322' };
const severityLabels: Record<string, string> = { LOW: '低', MEDIUM: '中', HIGH: '高', CRITICAL: '危急' };
const statusColors: Record<string, string> = { OPEN: 'red', IN_PROGRESS: 'orange', RESOLVED: 'green', CLOSED: 'default' };
const statusLabels: Record<string, string> = { OPEN: '待处理', IN_PROGRESS: '处理中', RESOLVED: '已解决', CLOSED: '已关闭' };

export default function AdminCrisis() {
  const [crisisCases, setCrisisCases] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [detailModal, setDetailModal] = useState(false);
  const [selectedCase, setSelectedCase] = useState<any>(null);
  const [hotlines, setHotlines] = useState<any[]>([]);

  useEffect(() => {
    loadCrisisCases();
    loadHotlines();
  }, []);

  const loadCrisisCases = async () => {
    setLoading(true);
    try {
      // TODO: 实际API接入
      setCrisisCases([
        {
          id: '1', key: '1',
          user: { nickname: '小明', id: 'u1' },
          severity: 'HIGH',
          status: 'IN_PROGRESS',
          trigger: 'PHQ-9第9题自杀意念评分 > 0',
          detectedAt: new Date(Date.now() - 7200000).toISOString(),
          description: '用户在进行心理测评时，PHQ-9第9题（自杀意念）选择了非零值。系统已自动触发关怀弹窗。',
          actions: [
            { type: 'system', content: '系统自动触发关怀弹窗，显示24小时热线', time: new Date(Date.now() - 7200000).toISOString() },
            { type: 'ai', content: 'AI聊天检测到危机关键词"不想活"，已推送安全提示', time: new Date(Date.now() - 5400000).toISOString() },
            { type: 'admin', content: '管理员已查看，正在联系用户', time: new Date(Date.now() - 3600000).toISOString() },
          ],
        },
        {
          id: '2', key: '2',
          user: { nickname: '小红', id: 'u2' },
          severity: 'CRITICAL',
          status: 'OPEN',
          trigger: '聊天关键词检测：自残相关',
          detectedAt: new Date(Date.now() - 1800000).toISOString(),
          description: '用户在AI聊天中发送包含自伤关键词的消息，系统检测到后立即触发危机协议。',
          actions: [
            { type: 'system', content: 'AI聊天检测到危机关键词，推送安全提示', time: new Date(Date.now() - 1800000).toISOString() },
            { type: 'system', content: '已自动通知管理员', time: new Date(Date.now() - 1700000).toISOString() },
          ],
        },
        {
          id: '3', key: '3',
          user: { nickname: '小刚', id: 'u3' },
          severity: 'MEDIUM',
          status: 'RESOLVED',
          trigger: '社区帖子包含危机内容',
          detectedAt: new Date(Date.now() - 86400000).toISOString(),
          description: '用户在社区发布包含消极内容的帖子，已进行人工审核和干预。',
          actions: [
            { type: 'system', content: '社区审核系统标记帖子为高风险', time: new Date(Date.now() - 86400000).toISOString() },
            { type: 'admin', content: '管理员审核帖子，移除危机内容', time: new Date(Date.now() - 82800000).toISOString() },
            { type: 'admin', content: '已私信联系用户，确认安全', time: new Date(Date.now() - 79200000).toISOString() },
            { type: 'admin', content: '用户回复安全，已转介专业咨询师', time: new Date(Date.now() - 72000000).toISOString() },
          ],
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const loadHotlines = async () => {
    try {
      const res = await crisisApi.getHotlines() as any;
      setHotlines(res.data?.hotlines || []);
    } catch {
      setHotlines([
        { name: '全国24小时心理援助热线', phone: '400-161-9995' },
        { name: '北京心理危机研究与干预中心', phone: '010-82951332' },
        { name: '生命热线', phone: '400-821-1215' },
      ]);
    }
  };

  const openCases = crisisCases.filter(c => c.status === 'OPEN' || c.status === 'IN_PROGRESS');
  const criticalCount = crisisCases.filter(c => c.severity === 'CRITICAL' && c.status !== 'RESOLVED' && c.status !== 'CLOSED').length;

  const columns = [
    {
      title: '严重程度', dataIndex: 'severity', key: 'severity', width: 100,
      render: (v: string) => (
        <Tag color={severityColors[v]} style={{ fontWeight: 600 }}>
          {v === 'CRITICAL' ? '🚨 ' : ''}{severityLabels[v]}
        </Tag>
      ),
      sorter: (a: any, b: any) => {
        const order: Record<string, number> = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3 };
        return (order[a.severity] ?? 4) - (order[b.severity] ?? 4);
      },
    },
    {
      title: '来访者', key: 'user', width: 100,
      render: (_: any, r: any) => <Text>{r.user?.nickname || '未知'}</Text>,
    },
    {
      title: '触发原因', dataIndex: 'trigger', key: 'trigger', ellipsis: true,
    },
    {
      title: '状态', dataIndex: 'status', key: 'status', width: 100,
      render: (v: string) => <Tag color={statusColors[v]}>{statusLabels[v]}</Tag>,
    },
    {
      title: '发现时间', dataIndex: 'detectedAt', key: 'detectedAt', width: 140,
      render: (d: string) => new Date(d).toLocaleString('zh-CN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }),
    },
    {
      title: '操作', key: 'action', width: 100,
      render: (_: any, record: any) => (
        <Button type="link" size="small" onClick={() => { setSelectedCase(record); setDetailModal(true); }}>
          查看详情
        </Button>
      ),
    },
  ];

  return (
    <div>
      <Title level={4} style={{ marginBottom: 16 }}>
        <SafetyCertificateOutlined /> 危机管理面板
      </Title>

      {/* 统计卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={12} sm={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic title="待处理" value={openCases.length} prefix={<AlertOutlined />}
              valueStyle={{ color: openCases.length > 0 ? '#ff4d4f' : '#52c41a' }} />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic title="危急案例" value={criticalCount} prefix={<WarningOutlined />}
              valueStyle={{ color: criticalCount > 0 ? '#cf1322' : '#52c41a' }} />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic title="本周新增" value={crisisCases.length} prefix={<ClockCircleOutlined />}
              valueStyle={{ color: '#faad14' }} />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic title="已解决" value={crisisCases.filter(c => c.status === 'RESOLVED').length} prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#52c41a' }} />
          </Card>
        </Col>
      </Row>

      {/* 危机案例列表 */}
      <Card
        title={<Space><WarningOutlined /> 危机案例</Space>}
        extra={criticalCount > 0 && (
          <Badge count={criticalCount} style={{ backgroundColor: '#cf1322' }}>
            <Text type="danger">需立即处理</Text>
          </Badge>
        )}
        style={{ borderRadius: 12, marginBottom: 16 }}
      >
        <Table
          columns={columns}
          dataSource={crisisCases}
          loading={loading}
          pagination={{ pageSize: 10 }}
        />
      </Card>

      {/* 热线资源 */}
      <Card title={<Space><PhoneOutlined /> 危机热线资源</Space>} style={{ borderRadius: 12 }}>
        <Row gutter={[16, 16]}>
          {hotlines.map((h, i) => (
            <Col xs={24} sm={8} key={i}>
              <Card size="small" style={{ borderRadius: 8, textAlign: 'center', borderLeft: '3px solid #ff4d4f' }}>
                <Text type="secondary" style={{ fontSize: 12 }}>{h.name}</Text>
                <Title level={4} style={{ color: '#ff4d4f', margin: '4px 0' }}>{h.phone}</Title>
              </Card>
            </Col>
          ))}
        </Row>
      </Card>

      {/* 案例详情弹窗 */}
      <Modal
        open={detailModal}
        onCancel={() => setDetailModal(false)}
        title={
          <Space>
            <WarningOutlined style={{ color: '#ff4d4f' }} />
            危机案例详情
            {selectedCase && <Tag color={severityColors[selectedCase.severity]}>{severityLabels[selectedCase.severity]}</Tag>}
          </Space>
        }
        width={700}
        footer={
          selectedCase?.status !== 'RESOLVED' && selectedCase?.status !== 'CLOSED' ? (
            <Space>
              <Button onClick={() => { message.success('已标记为已解决'); setDetailModal(false); }}>
                标记为已解决
              </Button>
              <Button type="primary" danger onClick={() => { message.info('已启动危机干预'); setDetailModal(false); }}>
                启动干预
              </Button>
            </Space>
          ) : null
        }
      >
        {selectedCase && (
          <div>
            <Alert
              type={selectedCase.severity === 'CRITICAL' ? 'error' : 'warning'}
              showIcon
              message={selectedCase.description}
              style={{ marginBottom: 16, borderRadius: 8 }}
            />
            <Row gutter={16} style={{ marginBottom: 16 }}>
              <Col span={8}>
                <Text type="secondary">来访者</Text>
                <div><Text strong>{selectedCase.user?.nickname}</Text></div>
              </Col>
              <Col span={8}>
                <Text type="secondary">触发原因</Text>
                <div><Text strong>{selectedCase.trigger}</Text></div>
              </Col>
              <Col span={8}>
                <Text type="secondary">发现时间</Text>
                <div><Text strong>{new Date(selectedCase.detectedAt).toLocaleString()}</Text></div>
              </Col>
            </Row>

            <Title level={5}>干预时间线</Title>
            <Timeline
              items={selectedCase.actions.map((a: any) => ({
                color: a.type === 'system' ? 'blue' : a.type === 'admin' ? 'green' : 'orange',
                dot: a.type === 'system' ? <NotificationOutlined /> : <CheckCircleOutlined />,
                children: (
                  <div>
                    <Tag color={a.type === 'system' ? 'blue' : 'green'} style={{ fontSize: 11 }}>
                      {a.type === 'system' ? '系统' : '管理员'}
                    </Tag>
                    <Text style={{ fontSize: 13 }}>{a.content}</Text>
                    <div><Text type="secondary" style={{ fontSize: 11 }}>{new Date(a.time).toLocaleString()}</Text></div>
                  </div>
                ),
              }))}
            />
          </div>
        )}
      </Modal>
    </div>
  );
}
