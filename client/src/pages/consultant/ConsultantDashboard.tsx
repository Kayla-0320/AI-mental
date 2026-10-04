import { Card, Row, Col, Statistic, Typography, List, Tag, Progress, Avatar, Badge } from 'antd';
import { TeamOutlined, CalendarOutlined, MessageOutlined, StarOutlined, RiseOutlined, FallOutlined } from '@ant-design/icons';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';

const { Title, Text } = Typography;

// 模拟数据
const weeklyData = [
  { day: '周一', sessions: 3 }, { day: '周二', sessions: 5 },
  { day: '周三', sessions: 4 }, { day: '周四', sessions: 6 },
  { day: '周五', sessions: 2 }, { day: '周六', sessions: 1 }, { day: '周日', sessions: 0 },
];

const upcomingAppointments = [
  { id: '1', patient: '小明', time: '09:00', topic: '工作压力与焦虑', status: 'confirmed' },
  { id: '2', patient: '小红', time: '10:00', topic: '人际关系困扰', status: 'confirmed' },
  { id: '3', patient: '小刚', time: '14:00', topic: '抑郁情绪管理', status: 'pending' },
];

const riskAlerts = [
  { id: '1', patient: '小刚', risk: 'HIGH', message: '抑郁评分持续上升，建议关注', time: '2小时前' },
  { id: '2', patient: '小丽', risk: 'MEDIUM', message: '睡眠质量下降明显', time: '昨天' },
];

const riskColors: Record<string, string> = { LOW: '#52c41a', MEDIUM: '#faad14', HIGH: '#ff4d4f', CRISIS: '#cf1322' };
const riskLabels: Record<string, string> = { LOW: '低', MEDIUM: '中', HIGH: '高', CRISIS: '危机' };

export default function ConsultantDashboard() {
  return (
    <div>
      <Title level={4} style={{ marginBottom: 16 }}>工作台概览</Title>

      {/* 统计卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic title="累计咨询" value={156} prefix={<TeamOutlined />}
              valueStyle={{ color: '#1890ff' }} suffix="人次" />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic title="本周咨询" value={21} prefix={<CalendarOutlined />}
              valueStyle={{ color: '#52c41a' }} suffix="次" />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic title="待确认预约" value={3} prefix={<MessageOutlined />}
              valueStyle={{ color: '#faad14' }} suffix="个" />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card style={{ borderRadius: 12 }}>
            <Statistic title="综合评分" value={4.8} prefix={<StarOutlined />}
              valueStyle={{ color: '#722ed1' }} suffix="/ 5.0" />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        {/* 本周咨询趋势 */}
        <Col xs={24} lg={14}>
          <Card title="本周咨询趋势" style={{ borderRadius: 12, height: '100%' }}>
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={weeklyData}>
                <XAxis dataKey="day" />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Area type="monotone" dataKey="sessions" stroke="#6366f1" fill="#6366f1" fillOpacity={0.2} name="咨询次数" />
              </AreaChart>
            </ResponsiveContainer>
          </Card>
        </Col>

        {/* 今日预约 */}
        <Col xs={24} lg={10}>
          <Card title="今日预约" style={{ borderRadius: 12, height: '100%' }}
            extra={<Text type="secondary">{upcomingAppointments.length} 个预约</Text>}>
            <List
              dataSource={upcomingAppointments}
              renderItem={(item) => (
                <List.Item>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12, width: '100%' }}>
                    <Avatar style={{ backgroundColor: '#6366f1' }}>{item.patient[0]}</Avatar>
                    <div style={{ flex: 1 }}>
                      <div><Text strong>{item.patient}</Text></div>
                      <Text type="secondary" style={{ fontSize: 12 }}>{item.topic}</Text>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div><Text>{item.time}</Text></div>
                      <Badge status={item.status === 'confirmed' ? 'success' : 'warning'}
                        text={<Text style={{ fontSize: 11 }}>{item.status === 'confirmed' ? '已确认' : '待确认'}</Text>} />
                    </div>
                  </div>
                </List.Item>
              )}
            />
          </Card>
        </Col>

        {/* 风险预警 */}
        <Col xs={24} lg={14}>
          <Card title="来访者风险预警" style={{ borderRadius: 12 }}
            extra={<Tag color="red">2 条预警</Tag>}>
            <List
              dataSource={riskAlerts}
              renderItem={(item) => (
                <List.Item>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12, width: '100%' }}>
                    <Avatar style={{ backgroundColor: riskColors[item.risk] }}>{item.patient[0]}</Avatar>
                    <div style={{ flex: 1 }}>
                      <div>
                        <Text strong>{item.patient}</Text>
                        <Tag color={riskColors[item.risk]} style={{ marginLeft: 8 }}>{riskLabels[item.risk]}风险</Tag>
                      </div>
                      <Text type="secondary" style={{ fontSize: 12 }}>{item.message}</Text>
                    </div>
                    <Text type="secondary" style={{ fontSize: 12 }}>{item.time}</Text>
                  </div>
                </List.Item>
              )}
            />
          </Card>
        </Col>

        {/* 来访者评分分布 */}
        <Col xs={24} lg={10}>
          <Card title="来访者心理评分" style={{ borderRadius: 12 }}>
            {[
              { name: '小明', score: 72 }, { name: '小红', score: 48 },
              { name: '小刚', score: 28 }, { name: '小丽', score: 55 },
            ].map((p) => (
              <div key={p.name} style={{ marginBottom: 16 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <Text>{p.name}</Text>
                  <Text type={p.score > 60 ? 'success' : p.score > 30 ? 'warning' : 'danger'}>{p.score} 分</Text>
                </div>
                <Progress percent={p.score} showInfo={false} size="small"
                  strokeColor={p.score > 60 ? '#52c41a' : p.score > 30 ? '#faad14' : '#ff4d4f'} />
              </div>
            ))}
          </Card>
        </Col>
      </Row>
    </div>
  );
}
