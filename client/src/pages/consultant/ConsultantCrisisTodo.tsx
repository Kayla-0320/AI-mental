import { useState, useEffect } from 'react';
import {
  Card, Typography, Table, Tag, Space, Button, Modal, Alert, Input,
  Badge, Row, Col, Statistic, message, Empty,
} from 'antd';
import {
  AlertOutlined, WarningOutlined, CheckOutlined,
  PhoneOutlined, ClockCircleOutlined, UserOutlined,
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { crisisApi } from '../../services';

const { Title, Text } = Typography;
const { TextArea } = Input;

const severityColors: Record<string, string> = { LOW: '#52c41a', MEDIUM: '#faad14', HIGH: '#ff4d4f', CRITICAL: '#cf1322' };
const severityLabels: Record<string, string> = { LOW: '低', MEDIUM: '中', HIGH: '高', CRITICAL: '危急' };
const statusColors: Record<string, string> = { OPEN: 'red', IN_PROGRESS: 'orange' };
const statusLabels: Record<string, string> = { OPEN: '待接收', IN_PROGRESS: '处理中' };

/** 与后端 CrisisService.ACK_TIMEOUT_MS 保持一致 */
const ACK_TIMEOUT_MS = 15 * 60 * 1000;

/**
 * 咨询师端「危机待办」。
 *
 * 这条链路此前是断的：患者端危机弹窗与热线卡可用，算法侧也判出了 CRISIS，
 * 但没有任何界面让真人知道「有一件事在等我」，咨询师端只有一句
 * 「危机干预协议尚未接入后端」的 TODO。
 *
 * 本页只做三件事，且每件都落库：
 *   1. 让未接收的危机可见（含超时标记）
 *   2. 确认接收（OPEN → IN_PROGRESS，记录接收人）
 *   3. 记录处置结论并结案（结案必须写结论）
 *
 * ⚠️ 页面不使用诊断性词汇；严重度是「先处理谁」的排序依据，不是临床分级。
 */
export default function ConsultantCrisisTodo() {
  const navigate = useNavigate();
  const [records, setRecords] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<any>(null);
  const [detailOpen, setDetailOpen] = useState(false);
  const [resolution, setResolution] = useState('');
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    loadTodo();
    const timer = setInterval(loadTodo, 30000);
    return () => clearInterval(timer);
  }, []);

  const loadTodo = async () => {
    try {
      const res = await crisisApi.getTodo() as any;
      setRecords(Array.isArray(res.data) ? res.data : []);
    } catch {
      message.error('加载危机待办失败');
    } finally {
      setLoading(false);
    }
  };

  const isOverdue = (r: any) => r.status === 'OPEN' && Date.now() - new Date(r.detectedAt).getTime() > ACK_TIMEOUT_MS;

  const handleAck = async (record: any) => {
    setSubmitting(true);
    try {
      await crisisApi.ackRecord(record.id);
      message.success('已确认接收，该案例转入「处理中」');
      setDetailOpen(false);
      await loadTodo();
    } catch (e: any) {
      message.error(e?.message || '接收失败，请重试');
    } finally {
      setSubmitting(false);
    }
  };

  const handleResolve = async (record: any) => {
    const text = resolution.trim();
    if (!text) {
      message.warning('请先填写处置结论，再结案');
      return;
    }
    setSubmitting(true);
    try {
      await crisisApi.updateRecord(record.id, { status: 'RESOLVED', resolution: text });
      message.success('已记录处置结论并结案');
      setDetailOpen(false);
      setResolution('');
      await loadTodo();
    } catch (e: any) {
      message.error(e?.message || '结案失败，请重试');
    } finally {
      setSubmitting(false);
    }
  };

  const openCount = records.filter(r => r.status === 'OPEN').length;
  const overdueCount = records.filter(isOverdue).length;
  const inProgress = records.filter(r => r.status === 'IN_PROGRESS').length;

  const columns = [
    {
      title: '严重程度', dataIndex: 'severity', key: 'severity', width: 110,
      render: (v: string, r: any) => (
        <Space size={4}>
          <Tag color={severityColors[v]} style={{ fontWeight: 600 }}>
            {v === 'CRITICAL' ? '🚨 ' : ''}{severityLabels[v] || v}
          </Tag>
          {isOverdue(r) && <Tag color="volcano" style={{ fontSize: 11 }}>超时</Tag>}
        </Space>
      ),
    },
    {
      title: '来访者', key: 'user', width: 120,
      render: (_: any, r: any) => <Space size={4}><UserOutlined /><Text>{r.userName || '未知'}</Text></Space>,
    },
    { title: '触发原因', dataIndex: 'trigger', key: 'trigger', ellipsis: true },
    {
      title: '状态', dataIndex: 'status', key: 'status', width: 100,
      render: (v: string) => <Tag color={statusColors[v]}>{statusLabels[v] || v}</Tag>,
    },
    {
      title: '发现时间', dataIndex: 'detectedAt', key: 'detectedAt', width: 150,
      render: (d: string) => new Date(d).toLocaleString('zh-CN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }),
    },
    {
      title: '操作', key: 'action', width: 150,
      render: (_: any, r: any) => (
        <Space size={0}>
          {r.status === 'OPEN' && (
            <Button type="link" size="small" onClick={() => handleAck(r)}>接收</Button>
          )}
          <Button type="link" size="small" onClick={() => { setSelected(r); setResolution(''); setDetailOpen(true); }}>
            详情
          </Button>
        </Space>
      ),
    },
  ];

  return (
    <div>
      <Title level={4} style={{ marginBottom: 16 }}>
        <AlertOutlined style={{ color: '#ff4d4f' }} /> 危机待办
      </Title>

      <Alert
        type="warning"
        showIcon
        style={{ marginBottom: 16, borderRadius: 8 }}
        message="这些是系统在对话中识别到的高风险信号"
        description="风险等级由对话信号自动判定，不是临床诊断。请先确认接收，再联系来访者或转介；结案必须填写处置结论，否则记录无法关闭。15 分钟未接收会自动升级提醒管理员。"
      />

      {overdueCount > 0 && (
        <Alert
          type="error"
          showIcon
          style={{ marginBottom: 16, borderRadius: 8 }}
          message={`有 ${overdueCount} 条记录超过 15 分钟无人接收`}
          description="请优先处理，或联系管理员改派。"
        />
      )}

      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col xs={12} sm={8}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic title="待接收" value={openCount} prefix={<WarningOutlined />}
              valueStyle={{ color: openCount > 0 ? '#ff4d4f' : '#52c41a' }} />
          </Card>
        </Col>
        <Col xs={12} sm={8}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic title="处理中" value={inProgress} valueStyle={{ color: '#faad14' }} />
          </Card>
        </Col>
        <Col xs={12} sm={8}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic title="超时未接收" value={overdueCount} prefix={<ClockCircleOutlined />}
              valueStyle={{ color: overdueCount > 0 ? '#fa541c' : '#52c41a' }} />
          </Card>
        </Col>
      </Row>

      <Card title={<Badge count={openCount} offset={[8, 0]}><Space><WarningOutlined /> 待处理清单</Space></Badge>} style={{ borderRadius: 12 }}>
        {records.length === 0 && !loading ? (
          <Empty description="当前没有待处理的危机记录" />
        ) : (
          <Table
            columns={columns}
            dataSource={records.map(r => ({ ...r, key: r.id }))}
            loading={loading}
            pagination={{ pageSize: 10 }}
          />
        )}
      </Card>

      <Modal
        open={detailOpen}
        onCancel={() => setDetailOpen(false)}
        title={<Space><WarningOutlined style={{ color: '#ff4d4f' }} />危机案例</Space>}
        width={680}
        footer={
          selected && (
            <Space>
              {selected.status === 'OPEN' && (
                <Button icon={<CheckOutlined />} loading={submitting} onClick={() => handleAck(selected)}>
                  确认接收
                </Button>
              )}
              <Button type="primary" danger loading={submitting} onClick={() => handleResolve(selected)}>
                记录结论并结案
              </Button>
            </Space>
          )
        }
      >
        {selected && (
          <div>
            <Alert
              type={selected.severity === 'CRITICAL' ? 'error' : 'warning'}
              showIcon
              message={`${severityLabels[selected.severity] || selected.severity} · ${selected.trigger}`}
              style={{ marginBottom: 16, borderRadius: 8 }}
            />

            <Row gutter={16} style={{ marginBottom: 12 }}>
              <Col span={8}>
                <Text type="secondary">来访者</Text>
                <div><Text strong>{selected.userName || '未知'}</Text></div>
              </Col>
              <Col span={8}>
                <Text type="secondary">状态</Text>
                <div><Tag color={statusColors[selected.status]}>{statusLabels[selected.status]}</Tag></div>
              </Col>
              <Col span={8}>
                <Text type="secondary">发现时间</Text>
                <div><Text strong>{new Date(selected.detectedAt).toLocaleString('zh-CN')}</Text></div>
              </Col>
            </Row>

            {selected.description && (
              <div style={{ marginBottom: 12 }}>
                <Text type="secondary">系统判定依据</Text>
                <div><Text style={{ fontSize: 13 }}>{selected.description}</Text></div>
              </div>
            )}

            {selected.sourceText && (
              <div style={{ marginBottom: 16 }}>
                <Text type="secondary">触发原文（仅用于研判，请勿外传）</Text>
                <div style={{ background: '#fafafa', padding: 12, borderRadius: 8, marginTop: 4 }}>
                  <Text style={{ fontSize: 13 }}>{selected.sourceText}</Text>
                </div>
              </div>
            )}

            <div style={{ marginBottom: 16 }}>
              <Text type="secondary">紧急资源</Text>
              <div style={{ marginTop: 4 }}>
                <Space wrap>
                  <Tag icon={<PhoneOutlined />} color="red">全国24小时心理援助 400-161-9995</Tag>
                  <Tag icon={<PhoneOutlined />} color="red">北京危机干预中心 010-82951332</Tag>
                  <Tag icon={<PhoneOutlined />} color="red">青少年热线 12355</Tag>
                </Space>
              </div>
              <Text type="secondary" style={{ fontSize: 12 }}>
                若来访者有明确自伤/伤人风险，请按机构流程联系紧急服务，不要仅依赖平台内消息。
              </Text>
            </div>

            <Text type="secondary">处置结论（结案必填）</Text>
            <TextArea
              rows={3}
              value={resolution}
              onChange={(e) => setResolution(e.target.value)}
              placeholder="例如：已电话联系本人确认安全；已通知紧急联系人；已转介至 XX 咨询师并预约 10/12 面谈。"
              style={{ borderRadius: 8, marginTop: 4 }}
            />
          </div>
        )}
      </Modal>

      <div style={{ marginTop: 16 }}>
        <Button type="link" onClick={() => navigate('/consultant/consultations')}>
          查看我的咨询记录
        </Button>
      </div>
    </div>
  );
}
