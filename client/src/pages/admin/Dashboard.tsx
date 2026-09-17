import { useState } from 'react';
import { Row, Col, Card, Typography, Statistic, Space, Tag, Progress, Table, Tabs, Tooltip, Timeline, Badge, Divider } from 'antd';
import {
  UserOutlined, MessageOutlined, TeamOutlined, FileTextOutlined,
  RiseOutlined, SafetyCertificateOutlined, AuditOutlined,
  CheckCircleOutlined, InfoCircleOutlined, WarningOutlined,
  AlertOutlined, ClockCircleOutlined, ThunderboltOutlined,
  DashboardOutlined, ExperimentOutlined, EyeOutlined,
} from '@ant-design/icons';
import { AreaChart, Area, BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer } from 'recharts';

const { Title, Text } = Typography;

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('overview');

  // 模拟数据
  const stats = {
    totalUsers: 1234,
    activeUsers: 456,
    totalConsultants: 28,
    totalConversations: 5678,
    totalAssessments: 890,
    avgRating: 4.6,
  };

  return (
    <div>
      <Title level={4} style={{ marginBottom: 24 }}>数据概览</Title>

      {/* 核心数据 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic title="注册用户" value={stats.totalUsers} prefix={<UserOutlined />}
              valueStyle={{ color: '#6366f1' }} />
            <Text type="secondary" style={{ marginTop: 8 }}>较昨日 +12</Text>
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic title="活跃用户" value={stats.activeUsers} prefix={<RiseOutlined />}
              valueStyle={{ color: '#52c41a' }} />
            <Text type="secondary" style={{ marginTop: 8 }}>今日活跃</Text>
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic title="咨询对话" value={stats.totalConversations} prefix={<MessageOutlined />}
              valueStyle={{ color: '#ec4899' }} />
            <Text type="secondary" style={{ marginTop: 8 }}>累计对话数</Text>
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic title="咨询师" value={stats.totalConsultants} prefix={<TeamOutlined />}
              valueStyle={{ color: '#f59e0b' }} />
            <Text type="secondary" style={{ marginTop: 8 }}>在线 {Math.floor(stats.totalConsultants * 0.6)}</Text>
          </Card>
        </Col>
      </Row>

      {/* 更多数据 */}
      <Tabs activeKey={activeTab} onChange={setActiveTab} style={{ marginBottom: 16 }} items={[
        {
          key: 'overview',
          label: <span><FileTextOutlined /> 平台概况</span>,
          children: (
            <Row gutter={[16, 16]}>
              <Col xs={24} lg={12}>
                <Card title="平台概况" style={{ borderRadius: 12 }}>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Text>累计测评次数</Text>
                      <Text strong>{stats.totalAssessments}</Text>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Text>咨询师平均评分</Text>
                      <Text strong>{stats.avgRating} / 5.0</Text>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Text>高风险用户</Text>
                      <Text strong style={{ color: '#ff4d4f' }}>3</Text>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Text>今日新增对话</Text>
                      <Text strong style={{ color: '#52c41a' }}>67</Text>
                    </div>
                  </Space>
                </Card>
              </Col>
              <Col xs={24} lg={12}>
                <Card title="近期动态" style={{ borderRadius: 12 }}>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    {[
                      { time: '10分钟前', event: '新用户注册' },
                      { time: '30分钟前', event: '完成心理测评 PHQ-9' },
                      { time: '1小时前', event: 'AI咨询对话开始' },
                      { time: '2小时前', event: '预约咨询师成功' },
                      { time: '3小时前', event: '疗愈冥想练习完成' },
                    ].map((item, i) => (
                      <div key={i} style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Text>{item.event}</Text>
                        <Text type="secondary">{item.time}</Text>
                      </div>
                    ))}
                  </Space>
                </Card>
              </Col>
            </Row>
          ),
        },
        {
          key: 'fairness',
          label: <span><SafetyCertificateOutlined /> 公平性报告</span>,
          children: (
            <Row gutter={[16, 16]}>
              <Col xs={24} lg={12}>
                <Card title={<Space><SafetyCertificateOutlined /> 算法公平性指标</Space>}
                  extra={<Tooltip title="基于对抗去偏见训练后的评估结果"><InfoCircleOutlined style={{ color: '#bbb' }} /></Tooltip>}
                  style={{ borderRadius: 12 }}>
                  <Space direction="vertical" style={{ width: '100%' }} size="middle">
                    {[
                      { label: '人口统计均等差 (DPD)', value: 0.04, threshold: 0.10 },
                      { label: '均等赔率差 (EOD)', value: 0.06, threshold: 0.10 },
                      { label: '组间准确率差', value: 0.03, threshold: 0.08 },
                      { label: '情感预测偏见指数', value: 0.02, threshold: 0.05 },
                    ].map(({ label, value, threshold }) => (
                      <div key={label}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                          <Text style={{ fontSize: 13 }}>{label}</Text>
                          <Tag color={value < threshold ? 'green' : 'orange'}>
                            {value < threshold ? '合格' : '待优化'}
                          </Tag>
                        </div>
                        <Progress percent={Math.round(value * 100)} showInfo={false} size="small"
                          strokeColor={value < threshold ? '#52c41a' : '#faad14'} />
                        <Text type="secondary" style={{ fontSize: 11 }}>当前: {value.toFixed(3)} | 阈值: {threshold}</Text>
                      </div>
                    ))}
                  </Space>
                </Card>
              </Col>
              <Col xs={24} lg={12}>
                <Card title={<Space><AuditOutlined /> 分组性能对比</Space>} style={{ borderRadius: 12 }}>
                  <Table
                    size="small"
                    pagination={false}
                    dataSource={[
                      { group: '15-18岁', accuracy: 0.82, recall: 0.78, fairness: '合格', key: '1' },
                      { group: '19-25岁', accuracy: 0.85, recall: 0.81, fairness: '合格', key: '2' },
                      { group: '26-35岁', accuracy: 0.83, recall: 0.79, fairness: '合格', key: '3' },
                      { group: '男性', accuracy: 0.84, recall: 0.80, fairness: '合格', key: '4' },
                      { group: '女性', accuracy: 0.83, recall: 0.79, fairness: '合格', key: '5' },
                    ]}
                    columns={[
                      { title: '分组', dataIndex: 'group', key: 'group' },
                      { title: '准确率', dataIndex: 'accuracy', key: 'accuracy', render: (v: number) => `${(v * 100).toFixed(1)}%` },
                      { title: '召回率', dataIndex: 'recall', key: 'recall', render: (v: number) => `${(v * 100).toFixed(1)}%` },
                      { title: '公平性', dataIndex: 'fairness', key: 'fairness', render: (v: string) => <Tag color="green">{v}</Tag> },
                    ]}
                  />
                  <div style={{ marginTop: 12, padding: '8px 12px', background: '#f6ffed', borderRadius: 8 }}>
                    <Text style={{ fontSize: 12, color: '#52c41a' }}>
                      <CheckCircleOutlined /> 所有分组公平性指标均在可接受范围内
                    </Text>
                  </div>
                </Card>
              </Col>
            </Row>
          ),
        },
        {
          key: 'perf',
          label: <span><DashboardOutlined /> 算法性能监控</span>,
          children: (
            <Row gutter={[16, 16]}>
              {/* 推理延迟分布 */}
              <Col xs={24} lg={12}>
                <Card title={<Space><ClockCircleOutlined /> 模型推理延迟分布</Space>} style={{ borderRadius: 12 }}>
                  <ResponsiveContainer width="100%" height={220}>
                    <AreaChart data={[
                      { time: '00:00', p50: 12, p95: 28, target: 50 },
                      { time: '04:00', p50: 11, p95: 25, target: 50 },
                      { time: '08:00', p50: 18, p95: 42, target: 50 },
                      { time: '10:00', p50: 22, p95: 48, target: 50 },
                      { time: '12:00', p50: 19, p95: 39, target: 50 },
                      { time: '14:00', p50: 21, p95: 44, target: 50 },
                      { time: '16:00', p50: 17, p95: 36, target: 50 },
                      { time: '18:00', p50: 23, p95: 47, target: 50 },
                      { time: '20:00', p50: 15, p95: 33, target: 50 },
                      { time: '22:00', p50: 13, p95: 29, target: 50 },
                    ]}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                      <XAxis dataKey="time" tick={{ fontSize: 11, fill: '#999' }} />
                      <YAxis tick={{ fontSize: 11, fill: '#999' }} unit="ms" />
                      <RechartsTooltip />
                      <Area type="monotone" dataKey="p95" stroke="#fa8c16" fill="#fa8c16" fillOpacity={0.15} name="P95" />
                      <Area type="monotone" dataKey="p50" stroke="#52c41a" fill="#52c41a" fillOpacity={0.2} name="P50" />
                      <Area type="monotone" dataKey="target" stroke="#ff4d4f" fill="none" strokeDasharray="5 5" name="目标 50ms" />
                    </AreaChart>
                  </ResponsiveContainer>
                  <div style={{ display: 'flex', justifyContent: 'space-around', marginTop: 8 }}>
                    <Tag color="green">P50 中位数: 17ms</Tag>
                    <Tag color="orange">P95: 37ms</Tag>
                    <Tag color="green">达标率: 98.2%</Tag>
                  </div>
                </Card>
              </Col>

              {/* 端侧推理覆盖率 + 模态缺失率 */}
              <Col xs={24} lg={12}>
                <Card title={<Space><ExperimentOutlined /> 端侧推理 & 模态覆盖率</Space>} style={{ borderRadius: 12 }}>
                  <Row gutter={16}>
                    <Col xs={12}>
                      <Statistic title="端侧推理覆盖率" value={87.3} suffix="%" valueStyle={{ color: '#52c41a', fontSize: 28 }} />
                      <Progress percent={87.3} showInfo={false} strokeColor="#52c41a" />
                      <Text type="secondary" style={{ fontSize: 11 }}>本地推理 1,247 次 / 总计 1,428 次</Text>
                    </Col>
                    <Col xs={12}>
                      <Statistic title="融合置信度均值" value={0.82} valueStyle={{ color: '#6366f1', fontSize: 28 }} />
                      <Progress percent={82} showInfo={false} strokeColor="#6366f1" />
                      <Text type="secondary" style={{ fontSize: 11 }}>近 24h 多模态融合平均置信度</Text>
                    </Col>
                  </Row>
                  <Divider style={{ margin: '16px 0' }} />
                  <Text strong style={{ fontSize: 13 }}>各模态数据缺失率</Text>
                  <div style={{ marginTop: 8 }}>
                    {[
                      { label: '键盘动力学', missing: 2.1, color: '#52c41a' },
                      { label: '文本语义', missing: 5.8, color: '#1890ff' },
                      { label: '语音声学', missing: 34.2, color: '#fa8c16' },
                      { label: '面部微表情', missing: 18.6, color: '#722ed1' },
                    ].map(m => (
                      <div key={m.label} style={{ marginBottom: 8 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 2 }}>
                          <span>{m.label}</span>
                          <span style={{ color: m.missing > 20 ? '#fa8c16' : '#52c41a' }}>{m.missing}%</span>
                        </div>
                        <Progress percent={m.missing} showInfo={false} size="small" strokeColor={m.color} />
                      </div>
                    ))}
                  </div>
                </Card>
              </Col>

              {/* 模型性能对比 */}
              <Col xs={24} lg={12}>
                <Card title={<Space><EyeOutlined /> 各模型性能指标</Space>} style={{ borderRadius: 12 }}>
                  <Table
                    size="small"
                    pagination={false}
                    dataSource={[
                      { model: 'SProp GNN 去偏', accuracy: '89.2%', latency: '8.3ms', fairness: '0.04', status: '运行中', key: '1' },
                      { model: '多任务风险评估', accuracy: '91.5%', latency: '12.1ms', fairness: '0.06', status: '运行中', key: '2' },
                      { model: '端侧 DistilBERT', accuracy: '86.8%', latency: '23.4ms', fairness: '0.05', status: '运行中', key: '3' },
                      { model: 'MediaPipe FaceMesh', accuracy: '93.1%', latency: '15.2ms', fairness: '0.03', status: '运行中', key: '4' },
                      { model: '声学特征提取', accuracy: '88.7%', latency: '6.8ms', fairness: '0.04', status: '运行中', key: '5' },
                    ]}
                    columns={[
                      { title: '模型', dataIndex: 'model', key: 'model', render: (v: string) => <Text strong style={{ fontSize: 12 }}>{v}</Text> },
                      { title: '准确率', dataIndex: 'accuracy', key: 'accuracy' },
                      { title: '延迟', dataIndex: 'latency', key: 'latency' },
                      { title: '偏见指数', dataIndex: 'fairness', key: 'fairness', render: (v: string) => <Tag color="green">{v}</Tag> },
                      { title: '状态', dataIndex: 'status', key: 'status', render: (v: string) => <Tag color="blue">{v}</Tag> },
                    ]}
                  />
                </Card>
              </Col>

              {/* 推理量趋势 */}
              <Col xs={24} lg={12}>
                <Card title={<Space><RiseOutlined /> 推理请求量趋势 (24h)</Space>} style={{ borderRadius: 12 }}>
                  <ResponsiveContainer width="100%" height={200}>
                    <LineChart data={[
                      { h: '0h', cloud: 12, edge: 45 }, { h: '2h', cloud: 8, edge: 32 },
                      { h: '4h', cloud: 5, edge: 22 }, { h: '6h', cloud: 10, edge: 38 },
                      { h: '8h', cloud: 35, edge: 89 }, { h: '10h', cloud: 52, edge: 124 },
                      { h: '12h', cloud: 48, edge: 118 }, { h: '14h', cloud: 56, edge: 135 },
                      { h: '16h', cloud: 44, edge: 108 }, { h: '18h', cloud: 62, edge: 148 },
                      { h: '20h', cloud: 38, edge: 96 }, { h: '22h', cloud: 18, edge: 58 },
                    ]}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                      <XAxis dataKey="h" tick={{ fontSize: 11, fill: '#999' }} />
                      <YAxis tick={{ fontSize: 11, fill: '#999' }} />
                      <RechartsTooltip />
                      <Line type="monotone" dataKey="edge" stroke="#52c41a" strokeWidth={2} name="端侧推理" dot={false} />
                      <Line type="monotone" dataKey="cloud" stroke="#1890ff" strokeWidth={2} name="云端推理" dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                  <div style={{ display: 'flex', justifyContent: 'space-around', marginTop: 8 }}>
                    <Tag color="green">端侧: 1,013 次</Tag>
                    <Tag color="blue">云端: 388 次</Tag>
                    <Tag>总计: 1,401 次</Tag>
                  </div>
                </Card>
              </Col>
            </Row>
          ),
        },
        {
          key: 'crisis',
          label: <span><AlertOutlined /> 危机升级监控</span>,
          children: (
            <Row gutter={[16, 16]}>
              {/* 危机响应时间统计 */}
              <Col xs={24} lg={8}>
                <Card title={<Space><ThunderboltOutlined /> 危机响应时间</Space>} style={{ borderRadius: 12 }}>
                  <Space direction="vertical" style={{ width: '100%' }} size="middle">
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                        <Text>平均检测时间</Text>
                        <Text strong style={{ color: '#52c41a' }}>2.3s</Text>
                      </div>
                      <Progress percent={23} showInfo={false} size="small" strokeColor="#52c41a" />
                      <Text type="secondary" style={{ fontSize: 11 }}>目标: &lt;10s | 达标</Text>
                    </div>
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                        <Text>平均响应时间</Text>
                        <Text strong style={{ color: '#faad14' }}>45s</Text>
                      </div>
                      <Progress percent={45} showInfo={false} size="small" strokeColor="#faad14" />
                      <Text type="secondary" style={{ fontSize: 11 }}>目标: &lt;60s | 达标</Text>
                    </div>
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                        <Text>咨询师接入时间</Text>
                        <Text strong style={{ color: '#1890ff' }}>3m 12s</Text>
                      </div>
                      <Progress percent={64} showInfo={false} size="small" strokeColor="#1890ff" />
                      <Text type="secondary" style={{ fontSize: 11 }}>目标: &lt;5min | 达标</Text>
                    </div>
                    <Divider style={{ margin: '8px 0' }} />
                    <div style={{ padding: '8px 12px', background: '#f6ffed', borderRadius: 8 }}>
                      <Text style={{ fontSize: 12, color: '#52c41a' }}>
                        <CheckCircleOutlined /> 本月危机响应 SLA 达标率: 96.7%
                      </Text>
                    </div>
                  </Space>
                </Card>
              </Col>

              {/* 危机事件统计 */}
              <Col xs={24} lg={8}>
                <Card title={<Space><WarningOutlined /> 危机事件统计</Space>} style={{ borderRadius: 12 }}>
                  <Space direction="vertical" style={{ width: '100%' }} size="middle">
                    <Row gutter={[12, 12]}>
                      <Col xs={12}>
                        <Statistic title="本月危机触发" value={23} valueStyle={{ fontSize: 24, color: '#ff4d4f' }} />
                      </Col>
                      <Col xs={12}>
                        <Statistic title="已解决" value={21} valueStyle={{ fontSize: 24, color: '#52c41a' }} />
                      </Col>
                      <Col xs={12}>
                        <Statistic title="处理中" value={2} valueStyle={{ fontSize: 24, color: '#faad14' }} />
                      </Col>
                      <Col xs={12}>
                        <Statistic title="转介成功" value={8} valueStyle={{ fontSize: 24, color: '#1890ff' }} />
                      </Col>
                    </Row>
                    <Divider style={{ margin: '8px 0' }} />
                    <div>
                      <Text type="secondary" style={{ fontSize: 12 }}>触发原因分布</Text>
                      <div style={{ marginTop: 8 }}>
                        {[
                          { label: '自伤关键词', pct: 35, color: '#ff4d4f' },
                          { label: '情绪极端波动', pct: 28, color: '#fa8c16' },
                          { label: '连续负面表达', pct: 22, color: '#722ed1' },
                          { label: '其他', pct: 15, color: '#1890ff' },
                        ].map(item => (
                          <div key={item.label} style={{ marginBottom: 6 }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 2 }}>
                              <span>{item.label}</span>
                              <span>{item.pct}%</span>
                            </div>
                            <Progress percent={item.pct} showInfo={false} size="small" strokeColor={item.color} />
                          </div>
                        ))}
                      </div>
                    </div>
                  </Space>
                </Card>
              </Col>

              {/* 最近危机事件时间线 */}
              <Col xs={24} lg={8}>
                <Card title={<Space><ClockCircleOutlined /> 最近危机事件</Space>} style={{ borderRadius: 12 }}>
                  <Timeline
                    items={[
                      { color: 'green', children: (
                        <div>
                          <Text strong style={{ fontSize: 12 }}>用户A 危机解除</Text>
                          <div><Text type="secondary" style={{ fontSize: 11 }}>咨询师已接入，风险下降</Text></div>
                          <div><Text type="secondary" style={{ fontSize: 10 }}>2小时前</Text></div>
                        </div>
                      )},
                      { color: 'orange', children: (
                        <div>
                          <Text strong style={{ fontSize: 12 }}>用户B 危机触发</Text>
                          <div><Text type="secondary" style={{ fontSize: 11 }}>检测到自伤关键词，已通知咨询师</Text></div>
                          <div><Text type="secondary" style={{ fontSize: 10 }}>5小时前</Text></div>
                        </div>
                      )},
                      { color: 'green', children: (
                        <div>
                          <Text strong style={{ fontSize: 12 }}>用户C 危机解除</Text>
                          <div><Text type="secondary" style={{ fontSize: 11 }}>自动降级为低风险</Text></div>
                          <div><Text type="secondary" style={{ fontSize: 10 }}>昨天</Text></div>
                        </div>
                      )},
                      { color: 'blue', children: (
                        <div>
                          <Text strong style={{ fontSize: 12 }}>用户D 风险评估</Text>
                          <div><Text type="secondary" style={{ fontSize: 11 }}>连续3天负面情绪，进入监控</Text></div>
                          <div><Text type="secondary" style={{ fontSize: 10 }}>2天前</Text></div>
                        </div>
                      )},
                    ]}
                  />
                </Card>
              </Col>
            </Row>
          ),
        },
      ]} />
    </div>
  );
}
