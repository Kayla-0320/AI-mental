import { useState } from 'react';
import { Row, Col, Card, Typography, Statistic, Space, Tag, Progress, Table, Tabs, Badge, Tooltip } from 'antd';
import {
  UserOutlined, MessageOutlined, TeamOutlined, FileTextOutlined,
  RiseOutlined, HeartOutlined, SafetyCertificateOutlined, AuditOutlined,
  CheckCircleOutlined, WarningOutlined, InfoCircleOutlined,
} from '@ant-design/icons';

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
                      { label: '人口统计均等差 (DPD)', value: 0.04, threshold: 0.10, status: 'good' },
                      { label: '均等赔率差 (EOD)', value: 0.06, threshold: 0.10, status: 'good' },
                      { label: '组间准确率差', value: 0.03, threshold: 0.08, status: 'good' },
                      { label: '情感预测偏见指数', value: 0.02, threshold: 0.05, status: 'good' },
                    ].map(({ label, value, threshold, status }) => (
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
                      <CheckCircleOutlined /> 所有分组公平性指标均在可接受范围内，对抗去偏见训练效果良好
                    </Text>
                  </div>
                </Card>
              </Col>
            </Row>
          ),
        },
        {
          key: 'community',
          label: <span><AuditOutlined /> 社区审核</span>,
          children: (
            <Row gutter={[16, 16]}>
              <Col xs={24}>
                <Card title={<Space><AuditOutlined /> 社区内容审核</Space>}
                  extra={<Space><Badge count={3} style={{ backgroundColor: '#faad14' }} /><Text type="secondary" style={{ fontSize: 12 }}>待审核</Text></Space>}
                  style={{ borderRadius: 12 }}>
                  <Table
                    size="small"
                    pagination={{ pageSize: 5 }}
                    dataSource={[
                      { id: '1', type: '帖子', content: '分享我的抗抑郁经历...', author: '用户A', status: '已审核', time: '2小时前', risk: 'low' },
                      { id: '2', type: '评论', content: '我觉得生活没有意义...', author: '用户B', status: '待审核', time: '1小时前', risk: 'high' },
                      { id: '3', type: '帖子', content: '推荐一本好书《被讨厌的勇气》', author: '用户C', status: '待审核', time: '45分钟前', risk: 'low' },
                      { id: '4', type: '评论', content: '有人想一起自杀吗...', author: '用户D', status: '待审核', time: '30分钟前', risk: 'crisis' },
                      { id: '5', type: '帖子', content: '冥想一周的感受', author: '用户E', status: '已审核', time: '3小时前', risk: 'low' },
                    ]}
                    columns={[
                      { title: '类型', dataIndex: 'type', key: 'type', width: 80, render: (v: string) => <Tag>{v}</Tag> },
                      { title: '内容', dataIndex: 'content', key: 'content', ellipsis: true },
                      { title: '作者', dataIndex: 'author', key: 'author', width: 80 },
                      { title: '风险', dataIndex: 'risk', key: 'risk', width: 80, render: (v: string) => (
                        <Tag color={v === 'crisis' ? 'red' : v === 'high' ? 'orange' : 'green'}>
                          {v === 'crisis' ? '危机' : v === 'high' ? '高' : '低'}
                        </Tag>
                      )},
                      { title: '状态', dataIndex: 'status', key: 'status', width: 80, render: (v: string) => (
                        <Tag color={v === '已审核' ? 'green' : 'orange'}>{v}</Tag>
                      )},
                      { title: '时间', dataIndex: 'time', key: 'time', width: 100 },
                    ]}
                  />
                  <div style={{ marginTop: 12, padding: '8px 12px', background: '#fff7e6', borderRadius: 8 }}>
                    <Text style={{ fontSize: 12, color: '#d46b08' }}>
                      <WarningOutlined /> 检测到 1 条危机内容，已自动触发危机干预协议并通知管理员
                    </Text>
                  </div>
                </Card>
              </Col>
            </Row>
          ),
        },
      ]} />
    </div>
  );
}
