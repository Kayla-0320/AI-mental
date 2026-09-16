import { useState, useEffect } from 'react';
import { Card, Row, Col, Typography, Tag, Space, Progress, Empty, Spin, Tabs } from 'antd';
import { ReadOutlined, PlayCircleOutlined, AudioOutlined, StarFilled } from '@ant-design/icons';
import { learningApi } from '../../services';

const { Title, Text, Paragraph } = Typography;

const categoryLabels: Record<string, string> = {
  emotion: '情绪管理', relationship: '人际关系', sleep: '睡眠改善',
  stress: '压力应对', self_growth: '自我成长', parenting: '亲子教育',
};

const typeIcons: Record<string, any> = {
  article: <ReadOutlined />, video: <PlayCircleOutlined />,
  audio: <AudioOutlined />, course: <StarFilled />,
};

export default function Learning() {
  const [contents, setContents] = useState<any[]>([]);
  const [records, setRecords] = useState<any[]>([]);
  const [stats, setStats] = useState<any>({});
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('explore');

  useEffect(() => { loadData(); }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [contentsRes, recordsRes, statsRes] = await Promise.all([
        learningApi.getContents() as any,
        learningApi.getUserRecords() as any,
        learningApi.getStats() as any,
      ]);
      setContents(contentsRes.data?.contents || []);
      setRecords(recordsRes.data || []);
      setStats(statsRes.data || {});
    } catch {} finally { setLoading(false); }
  };

  if (loading) return <div style={{ textAlign: 'center', padding: 100 }}><Spin size="large" /></div>;

  return (
    <div>
      {/* 学习统计 */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card style={{ borderRadius: 12, textAlign: 'center' }}>
            <div style={{ fontSize: 28, fontWeight: 'bold', color: '#6366f1' }}>{stats.totalLearned || 0}</div>
            <Text type="secondary">已学内容</Text>
          </Card>
        </Col>
        <Col span={6}>
          <Card style={{ borderRadius: 12, textAlign: 'center' }}>
            <div style={{ fontSize: 28, fontWeight: 'bold', color: '#52c41a' }}>{stats.totalCompleted || 0}</div>
            <Text type="secondary">已完成</Text>
          </Card>
        </Col>
        <Col span={6}>
          <Card style={{ borderRadius: 12, textAlign: 'center' }}>
            <div style={{ fontSize: 28, fontWeight: 'bold', color: '#f59e0b' }}>{stats.totalInProgress || 0}</div>
            <Text type="secondary">学习中</Text>
          </Card>
        </Col>
        <Col span={6}>
          <Card style={{ borderRadius: 12, textAlign: 'center' }}>
            <Progress type="dashboard" percent={stats.avgProgress || 0} size={60} />
            <div><Text type="secondary">平均进度</Text></div>
          </Card>
        </Col>
      </Row>

      <Tabs activeKey={activeTab} onChange={setActiveTab} items={[
        { key: 'explore', label: '探索内容', children: (
          <Row gutter={[16, 16]}>
            {contents.map((item: any) => (
              <Col xs={24} sm={12} lg={8} key={item.id}>
                <Card hoverable style={{ borderRadius: 12 }} bodyStyle={{ padding: 20 }}>
                  <Space style={{ marginBottom: 8 }}>
                    {typeIcons[item.type]}
                    <Tag>{categoryLabels[item.category] || item.category}</Tag>
                    <Tag color={item.level === 'beginner' ? 'green' : item.level === 'intermediate' ? 'orange' : 'red'}>
                      {item.level === 'beginner' ? '入门' : item.level === 'intermediate' ? '进阶' : '高级'}
                    </Tag>
                  </Space>
                  <Title level={5} style={{ marginBottom: 4 }}>{item.title}</Title>
                  <Paragraph type="secondary" ellipsis={{ rows: 2 }}>{item.summary}</Paragraph>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Text type="secondary">{item.duration ? `${item.duration}分钟` : ''}</Text>
                    <Text type="secondary">{item.viewCount} 人已学</Text>
                  </div>
                </Card>
              </Col>
            ))}
          </Row>
        )},
        { key: 'my-learning', label: '我的学习', children: (
          records.length === 0 ? <Empty description="暂无学习记录" /> : (
            <Space direction="vertical" style={{ width: '100%' }}>
              {records.map((record: any) => (
                <Card key={record.id} size="small" style={{ borderRadius: 8 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Space>
                      {typeIcons[record.contentType]}
                      <Text strong>{record.title}</Text>
                      {record.completed && <Tag color="success">已完成</Tag>}
                    </Space>
                    <Progress percent={record.progress} size="small" style={{ width: 120 }} />
                  </div>
                </Card>
              ))}
            </Space>
          )
        )},
      ]} />
    </div>
  );
}
