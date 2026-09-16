import { useState, useEffect } from 'react';
import { Card, Typography, Button, Tag, Space, Progress, Timeline, Empty, Spin, Modal, Input, message, Collapse, Divider, List } from 'antd';
import {
  PlusOutlined, ScheduleOutlined, CheckCircleOutlined,
  AimOutlined, CalendarOutlined, FileTextOutlined,
  ExperimentOutlined, ClockCircleOutlined, UserOutlined,
} from '@ant-design/icons';
import { treatmentApi } from '../../services';

const { Title, Text, Paragraph } = Typography;

const statusColors: Record<string, string> = { DRAFT: 'default', ACTIVE: 'processing', COMPLETED: 'success', PAUSED: 'warning' };
const statusLabels: Record<string, string> = { DRAFT: '草稿', ACTIVE: '进行中', COMPLETED: '已完成', PAUSED: '已暂停' };

// 从 goals 字段解析完整方案数据
function parsePlanData(plan: any) {
  try {
    const data = JSON.parse(plan.goals || '{}');
    return {
      goals: data.goals || { shortTerm: [], longTerm: [] },
      phases: data.phases || [],
      dailyTasks: data.dailyTasks || [],
      assessmentNodes: data.assessmentNodes || [],
    };
  } catch {
    return { goals: { shortTerm: [], longTerm: [] }, phases: [], dailyTasks: [], assessmentNodes: [] };
  }
}

export default function Treatment() {
  const [plans, setPlans] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [createModal, setCreateModal] = useState(false);
  const [newTitle, setNewTitle] = useState('');
  const [creating, setCreating] = useState(false);

  useEffect(() => { loadPlans(); }, []);

  const loadPlans = async () => {
    setLoading(true);
    try {
      const res = await treatmentApi.getPlans() as any;
      setPlans(res.data || []);
    } catch (err: any) {
      message.error(err?.message || '加载治疗规划失败');
    } finally { setLoading(false); }
  };

  const handleCreate = async () => {
    if (!newTitle.trim()) {
      message.warning('请输入规划标题');
      return;
    }
    setCreating(true);
    try {
      await treatmentApi.createPlan({ title: newTitle });
      message.success('治疗规划创建成功');
      setCreateModal(false);
      setNewTitle('');
      loadPlans();
    } catch (err: any) {
      message.error(err?.message || '创建失败，请稍后重试');
    } finally {
      setCreating(false);
    }
  };

  if (loading) return <div style={{ textAlign: 'center', padding: 100 }}><Spin size="large" /></div>;

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
        <Title level={4}><ScheduleOutlined /> 治疗规划</Title>
        <Button type="primary" icon={<PlusOutlined />} onClick={() => setCreateModal(true)}>
          新建规划
        </Button>
      </div>

      {plans.length === 0 ? (
        <Empty description="暂无治疗规划">
          <Button type="primary" onClick={() => setCreateModal(true)}>创建第一个规划</Button>
        </Empty>
      ) : (
        plans.map((plan: any) => {
          const { goals, phases, dailyTasks, assessmentNodes } = parsePlanData(plan);

          return (
            <Card key={plan.id} style={{ borderRadius: 12, marginBottom: 16 }}>
              {/* 标题行 */}
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
                <div>
                  <Title level={5} style={{ marginBottom: 4 }}>{plan.title}</Title>
                  <Tag color={statusColors[plan.status]}>{statusLabels[plan.status]}</Tag>
                  <Text type="secondary" style={{ marginLeft: 8, fontSize: 12 }}>
                    创建于 {new Date(plan.createdAt).toLocaleDateString()}
                  </Text>
                </div>
                <Progress type="circle" percent={plan.progress} size={60} />
              </div>

              {/* 治疗目标 */}
              {(goals.shortTerm?.length > 0 || goals.longTerm?.length > 0) && (
                <div style={{ marginBottom: 20 }}>
                  <Title level={5} style={{ marginBottom: 12 }}><AimOutlined /> 治疗目标</Title>
                  {goals.shortTerm?.length > 0 && (
                    <div style={{ marginBottom: 12 }}>
                      <Text strong style={{ color: '#6366f1' }}>短期目标（1-4周）</Text>
                      <ul style={{ margin: '8px 0 0 20px', padding: 0 }}>
                        {goals.shortTerm.map((g: string, i: number) => (
                          <li key={i} style={{ marginBottom: 4, lineHeight: 1.6, fontSize: 14 }}>{g}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {goals.longTerm?.length > 0 && (
                    <div>
                      <Text strong style={{ color: '#722ed1' }}>长期目标（3-6个月）</Text>
                      <ul style={{ margin: '8px 0 0 20px', padding: 0 }}>
                        {goals.longTerm.map((g: string, i: number) => (
                          <li key={i} style={{ marginBottom: 4, lineHeight: 1.6, fontSize: 14 }}>{g}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}

              {/* 治疗阶段 */}
              {phases.length > 0 && (
                <div style={{ marginBottom: 20 }}>
                  <Title level={5} style={{ marginBottom: 12 }}><CalendarOutlined /> 治疗阶段</Title>
                  <Collapse
                    defaultActiveKey={['0']}
                    items={phases.map((phase: any, i: number) => ({
                      key: String(i),
                      label: (
                        <Space>
                          <Tag color={i === 0 ? '#6366f1' : i === phases.length - 1 ? '#52c41a' : '#1890ff'} style={{ borderRadius: 12 }}>
                            第{phase.phase || i + 1}阶段
                          </Tag>
                          <Text strong>{phase.name}</Text>
                          <Tag>{phase.duration}</Tag>
                        </Space>
                      ),
                      children: (
                        <div>
                          <div style={{ marginBottom: 12 }}>
                            <Text type="secondary">核心焦点：</Text>
                            <Paragraph style={{ margin: '4px 0 0 0', fontSize: 14 }}>{phase.focus}</Paragraph>
                          </div>
                          {phase.consultationTopics?.length > 0 && (
                            <div style={{ marginBottom: 12 }}>
                              <Text strong><FileTextOutlined /> 咨询主题建议</Text>
                              <ul style={{ margin: '8px 0 0 20px', padding: 0 }}>
                                {phase.consultationTopics.map((t: string, j: number) => (
                                  <li key={j} style={{ marginBottom: 6, lineHeight: 1.6, fontSize: 13 }}>{t}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                          {phase.therapistRole && (
                            <div style={{ padding: '8px 12px', background: '#f0f5ff', borderRadius: 8, fontSize: 13 }}>
                              <UserOutlined style={{ color: '#6366f1', marginRight: 6 }} />
                              <Text type="secondary">咨询师角色：</Text>
                              <Text style={{ marginLeft: 4 }}>{phase.therapistRole}</Text>
                            </div>
                          )}
                        </div>
                      ),
                    }))}
                  />
                </div>
              )}

              {/* 每日任务 */}
              {dailyTasks.length > 0 && (
                <div style={{ marginBottom: 20 }}>
                  <Title level={5} style={{ marginBottom: 12 }}><ExperimentOutlined /> 每日练习任务</Title>
                  <Collapse
                    items={dailyTasks.map((task: any, i: number) => ({
                      key: String(i),
                      label: (
                        <Space>
                          <CheckCircleOutlined style={{ color: '#52c41a' }} />
                          <Text strong>{task.task}</Text>
                          <Tag color="blue">{task.frequency}</Tag>
                        </Space>
                      ),
                      children: (
                        <div>
                          <Paragraph style={{ fontSize: 14, lineHeight: 1.8, marginBottom: 12 }}>{task.description}</Paragraph>
                          {task.tools?.length > 0 && (
                            <div>
                              <Text type="secondary" style={{ fontSize: 13 }}>推荐工具：</Text>
                              <Space style={{ marginLeft: 8, flexWrap: 'wrap' }}>
                                {task.tools.map((tool: string, j: number) => (
                                  <Tag key={j} color="cyan">{tool}</Tag>
                                ))}
                              </Space>
                            </div>
                          )}
                        </div>
                      ),
                    }))}
                  />
                </div>
              )}

              {/* 评估节点 */}
              {assessmentNodes.length > 0 && (
                <div style={{ marginBottom: 16 }}>
                  <Title level={5} style={{ marginBottom: 12 }}><ClockCircleOutlined /> 评估节点</Title>
                  <Timeline>
                    {assessmentNodes.map((node: any, i: number) => (
                      <Timeline.Item key={i} color={i === 0 ? '#6366f1' : 'gray'}>
                        <Text strong>第 {node.week} 周</Text>
                        <div style={{ marginTop: 4 }}>
                          {node.metrics?.map((m: string, j: number) => (
                            <div key={j} style={{ fontSize: 13, lineHeight: 1.6, padding: '2px 0' }}>
                              <CheckCircleOutlined style={{ color: '#52c41a', marginRight: 6, fontSize: 12 }} />
                              {m}
                            </div>
                          ))}
                        </div>
                        {node.purpose && (
                          <div style={{ marginTop: 6, padding: '6px 10px', background: '#fafafa', borderRadius: 6, fontSize: 12 }}>
                            <Text type="secondary">目的：{node.purpose}</Text>
                          </div>
                        )}
                      </Timeline.Item>
                    ))}
                  </Timeline>
                </div>
              )}
            </Card>
          );
        })
      )}

      <Modal open={createModal} onCancel={() => !creating && setCreateModal(false)} title="新建治疗规划" onOk={handleCreate} confirmLoading={creating} okText={creating ? 'AI 生成中（约需 30-60 秒）...' : '确定'}>
        <Input placeholder="规划标题" value={newTitle} onChange={e => setNewTitle(e.target.value)}
          style={{ marginTop: 16 }} />
        <Paragraph type="secondary" style={{ marginTop: 8 }}>
          AI 将根据你的心理画像自动生成个性化治疗方案
        </Paragraph>
      </Modal>
    </div>
  );
}
