import { useState, useEffect } from 'react';
import { Card, Typography, Table, Tag, Space, Button, Modal, Alert, Timeline, Statistic, Row, Col, Badge, Input, message } from 'antd';
import {
  WarningOutlined, PhoneOutlined, SafetyCertificateOutlined,
  CheckCircleOutlined, ClockCircleOutlined,
  AlertOutlined, NotificationOutlined, CheckOutlined,
} from '@ant-design/icons';
import { crisisApi } from '../../services';

const { Title, Text } = Typography;
const { TextArea } = Input;

const severityColors: Record<string, string> = { LOW: '#52c41a', MEDIUM: '#faad14', HIGH: '#ff4d4f', CRITICAL: '#cf1322' };
const severityLabels: Record<string, string> = { LOW: '低', MEDIUM: '中', HIGH: '高', CRITICAL: '危急' };
const statusColors: Record<string, string> = { OPEN: 'red', IN_PROGRESS: 'orange', RESOLVED: 'green', CLOSED: 'default' };
const statusLabels: Record<string, string> = { OPEN: '待接收', IN_PROGRESS: '处理中', RESOLVED: '已解决', CLOSED: '已关闭' };

/** 未接收超时阈值（与后端 CrisisService.ACK_TIMEOUT_MS 一致）：用于把「没人接」显式标出来 */
const ACK_TIMEOUT_MS = 15 * 60 * 1000;

/** 组装一条危机记录的真实时间线。此前这里是 `selectedCase.actions` —— 后端从不返回该字段，所以永远空。 */
function buildTimeline(record: any) {
  const items: any[] = [{
    color: 'red',
    dot: <AlertOutlined />,
    children: (
      <div>
        <Tag color="red" style={{ fontSize: 11 }}>系统</Tag>
        <Text style={{ fontSize: 13 }}>检测到危机信号（{severityLabels[record.severity] || record.severity}）</Text>
        <div><Text type="secondary" style={{ fontSize: 11 }}>{new Date(record.detectedAt).toLocaleString('zh-CN')}</Text></div>
        <div><Text type="secondary" style={{ fontSize: 11 }}>{record.trigger}</Text></div>
      </div>
    ),
  }];

  if (record.status === 'IN_PROGRESS' || record.status === 'RESOLVED' || record.status === 'CLOSED') {
    items.push({
      color: 'blue',
      dot: <CheckOutlined />,
      children: (
        <div>
          <Tag color="blue" style={{ fontSize: 11 }}>真人</Tag>
          <Text style={{ fontSize: 13 }}>
            已确认接收{record.resolvedBy ? `（操作人 ${String(record.resolvedBy).slice(-8)}）` : ''}
          </Text>
        </div>
      ),
    });
  }

  if (record.resolvedAt) {
    items.push({
      color: 'green',
      dot: <CheckCircleOutlined />,
      children: (
        <div>
          <Tag color="green" style={{ fontSize: 11 }}>处置</Tag>
          <Text style={{ fontSize: 13 }}>{record.resolution || '（未填写处置结论）'}</Text>
          <div><Text type="secondary" style={{ fontSize: 11 }}>{new Date(record.resolvedAt).toLocaleString('zh-CN')}</Text></div>
        </div>
      ),
    });
  }

  return items;
}

export default function AdminCrisis() {
  const [crisisCases, setCrisisCases] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [detailModal, setDetailModal] = useState(false);
  const [selectedCase, setSelectedCase] = useState<any>(null);
  const [hotlines, setHotlines] = useState<any[]>([]);
  const [resolutionInput, setResolutionInput] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const [crisisStats, setCrisisStats] = useState({ open: 0, critical: 0, thisWeek: 0, resolved: 0, overdue: 0 });

  useEffect(() => {
    loadCrisisCases();
    loadCrisisStats();
    loadHotlines();
  }, []);

  const loadCrisisStats = async () => {
    try {
      const res = await crisisApi.getStats() as any;
      if (res.data) setCrisisStats(res.data);
    } catch { /* ignore */ }
  };

  const loadCrisisCases = async () => {
    setLoading(true);
    try {
      const res = await crisisApi.getRecords({ pageSize: 50 }) as any;
      const records = res.data?.records || [];
      const cases = records.map((r: any) => ({
        id: r.id, key: r.id,
        user: { nickname: r.userName || '未知', id: r.userId },
        severity: r.severity,
        status: r.status,
        trigger: r.trigger,
        detectedAt: r.detectedAt,
        description: r.description,
        sourceText: r.sourceText,
        resolution: r.resolution,
        resolvedAt: r.resolvedAt,
        resolvedBy: r.resolvedBy,
      }));
      setCrisisCases(cases);
    } catch {
      message.error('加载危机记录失败');
    } finally {
      setLoading(false);
    }
  };

  const loadHotlines = async () => {
    try {
      const res = await crisisApi.getHotlines() as any;
      // 后端 data 本身就是数组（此前按 res.data.hotlines 取，永远取不到 → 一直显示兜底列表）
      const list = Array.isArray(res.data) ? res.data : res.data?.hotlines;
      if (list && list.length > 0) {
        setHotlines(list);
        return;
      }
      throw new Error('empty');
    } catch {
      setHotlines([
        { name: '全国24小时心理援助热线', phone: '400-161-9995' },
        { name: '北京心理危机研究与干预中心', phone: '010-82951332' },
        { name: '生命热线', phone: '400-821-1215' },
      ]);
    }
  };

  /** 确认接收：OPEN → IN_PROGRESS，并记下接收人 */
  const handleAck = async (record: any) => {
    setSubmitting(true);
    try {
      await crisisApi.ackRecord(record.id);
      message.success('已确认接收，该案例转入「处理中」');
      setDetailModal(false);
      await Promise.all([loadCrisisCases(), loadCrisisStats()]);
    } catch (e: any) {
      message.error(e?.message || '接收失败，请重试');
    } finally {
      setSubmitting(false);
    }
  };

  /** 结案：必须填写处置结论，否则「已解决」后面什么都没有 */
  const handleResolve = async (record: any) => {
    const resolution = resolutionInput.trim() || record.resolution;
    if (!resolution) {
      message.warning('请先填写处置结论，再标记为已解决');
      return;
    }
    setSubmitting(true);
    try {
      await crisisApi.updateRecord(record.id, {
        status: record.status === 'OPEN' ? 'RESOLVED' : 'RESOLVED',
        resolution,
      });
      message.success('已记录处置结论并结案');
      setDetailModal(false);
      setResolutionInput('');
      await Promise.all([loadCrisisCases(), loadCrisisStats()]);
    } catch (e: any) {
      message.error(e?.message || '更新失败，请重试');
    } finally {
      setSubmitting(false);
    }
  };

  const openCases = crisisCases.filter(c => c.status === 'OPEN' || c.status === 'IN_PROGRESS');
  const criticalCount = crisisCases.filter(c => c.severity === 'CRITICAL' && c.status !== 'RESOLVED' && c.status !== 'CLOSED').length;
  const isOverdue = (c: any) => c.status === 'OPEN' && Date.now() - new Date(c.detectedAt).getTime() > ACK_TIMEOUT_MS;

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
      // 宽度要放得下「处理中 + 超时未接」两个标签，否则会把后面的时间列挤掉
      title: '状态', dataIndex: 'status', key: 'status', width: 165,
      render: (v: string, r: any) => (
        <Space size={4}>
          <Tag color={statusColors[v]}>{statusLabels[v]}</Tag>
          {isOverdue(r) && <Tag color="volcano" style={{ fontSize: 11 }}>超时未接</Tag>}
        </Space>
      ),
    },
    {
      title: '发现时间', dataIndex: 'detectedAt', key: 'detectedAt', width: 140,
      render: (d: string) => new Date(d).toLocaleString('zh-CN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }),
    },
    {
      title: '操作', key: 'action', width: 130,
      render: (_: any, record: any) => (
        <Space size={0}>
          {record.status === 'OPEN' && (
            <Button type="link" size="small" onClick={() => handleAck(record)}>接收</Button>
          )}
          <Button type="link" size="small" onClick={() => { setSelectedCase(record); setResolutionInput(record.resolution || ''); setDetailModal(true); }}>
            查看详情
          </Button>
        </Space>
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
            <Statistic title="待处理" value={crisisStats.open} prefix={<AlertOutlined />}
              valueStyle={{ color: crisisStats.open > 0 ? '#ff4d4f' : '#52c41a' }} />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic title="危急案例" value={crisisStats.critical} prefix={<WarningOutlined />}
              valueStyle={{ color: crisisStats.critical > 0 ? '#cf1322' : '#52c41a' }} />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic title="超时未接收" value={crisisStats.overdue || 0} prefix={<ClockCircleOutlined />}
              valueStyle={{ color: (crisisStats.overdue || 0) > 0 ? '#fa541c' : '#52c41a' }} />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic title="已解决" value={crisisStats.resolved} prefix={<CheckCircleOutlined />}
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
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 16, borderRadius: 8 }}
          message="处置流程"
          description="检测到危机后系统会通知全部管理员与咨询师；15 分钟无人接收会自动再次提醒。结案必须填写处置结论，否则无法关闭。"
        />
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
          selectedCase && selectedCase.status !== 'RESOLVED' && selectedCase.status !== 'CLOSED' ? (
            <Space>
              {selectedCase.status === 'OPEN' && (
                <Button icon={<CheckOutlined />} loading={submitting} onClick={() => handleAck(selectedCase)}>
                  确认接收
                </Button>
              )}
              <Button type="primary" danger loading={submitting} onClick={() => handleResolve(selectedCase)}>
                记录结论并结案
              </Button>
            </Space>
          ) : <Text type="secondary">该案例已结案</Text>
        }
      >
        {selectedCase && (
          <div>
            <Alert
              type={selectedCase.severity === 'CRITICAL' ? 'error' : 'warning'}
              showIcon
              message={selectedCase.description || selectedCase.trigger}
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
                <div><Text strong>{new Date(selectedCase.detectedAt).toLocaleString('zh-CN')}</Text></div>
              </Col>
            </Row>

            {selectedCase.sourceText && (
              <>
                <Text type="secondary">触发原文</Text>
                <div style={{ background: '#fafafa', padding: 12, borderRadius: 8, margin: '4px 0 16px' }}>
                  <Text style={{ fontSize: 13 }}>{selectedCase.sourceText}</Text>
                </div>
              </>
            )}

            <Title level={5}>处置时间线</Title>
            <Timeline items={buildTimeline(selectedCase)} />

            {selectedCase.status !== 'RESOLVED' && selectedCase.status !== 'CLOSED' && (
              <>
                <Title level={5} style={{ marginTop: 16 }}>处置结论（结案必填）</Title>
                <TextArea
                  rows={3}
                  value={resolutionInput}
                  onChange={(e) => setResolutionInput(e.target.value)}
                  placeholder="例如：已电话联系来访者本人确认安全；已通知紧急联系人；已转介至 XX 咨询师跟进。"
                  style={{ borderRadius: 8 }}
                />
              </>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
}
