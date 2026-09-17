import { useState, useEffect, useRef } from 'react';
import { Card, Row, Col, Typography, Button, Progress, Space, Tag, Divider, Spin, Alert, Modal, Tabs, Radio, Result, List, Empty, Input, Tooltip, Badge } from 'antd';
import {
  KeyOutlined, CameraOutlined, PlayCircleOutlined, ReloadOutlined,
  ExperimentOutlined, FileTextOutlined, CheckCircleOutlined,
  DashboardOutlined, HeartOutlined, PhoneOutlined,
  SoundOutlined, AudioOutlined, StopOutlined, SendOutlined,
  AudioMutedOutlined, RobotOutlined, ThunderboltOutlined,
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { useKeyboardAnxiety } from '../../hooks/useKeyboardAnxiety';
import { useBlinkDetection } from '../../hooks/useBlinkDetection';
import { usePerception } from '../../hooks/usePerception';
import { profileApi } from '../../services';

const { Title, Text, Paragraph } = Typography;

// ===== 量表定义 =====
const assessments = [
  {
    type: 'PHQ9', title: 'PHQ-9 抑郁症筛查量表',
    desc: '评估过去两周内的抑郁症状严重程度',
    questions: [
      '做事时提不起劲或没有兴趣', '感到心情低落、沮丧或绝望',
      '入睡困难、睡不安稳或睡眠过多', '感觉疲倦或没有活力',
      '食欲不振或吃太多', '觉得自己很糟或觉得自己很失败',
      '对事物专注有困难', '动作或说话慢/烦躁或坐立不安',
      '有不如死掉或用某种方式伤害自己的念头',
    ],
  },
  {
    type: 'GAD7', title: 'GAD-7 广泛性焦虑量表',
    desc: '评估过去两周内的焦虑症状',
    questions: [
      '感觉紧张、焦虑或急切', '不能够停止或控制担忧',
      '对各种各样的事情担忧过多', '很难放松下来',
      '由于不安而无法静坐', '变得容易烦恼或急躁',
      '感到似乎将有可怕的事情发生',
    ],
  },
];

export default function UnifiedAssessment() {
  const navigate = useNavigate();
  // 焦虑感知状态
  const keyboard = useKeyboardAnxiety();
  const blink = useBlinkDetection();
  const [realityTask, setRealityTask] = useState<any>(null);
  const [taskLoading, setTaskLoading] = useState(false);
  const [taskModalOpen, setTaskModalOpen] = useState(false);
  const [taskTimer, setTaskTimer] = useState(0);
  const [taskTimerRunning, setTaskTimerRunning] = useState(false);

  // 测评状态
  const [selectedAssessment, setSelectedAssessment] = useState<any>(null);
  const [answers, setAnswers] = useState<Record<number, number>>({});
  const [result, setResult] = useState<any>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [submitting, setSubmitting] = useState(false);

  // 安全拦截：PHQ-9 第9题（自杀意念）非零时触发
  const [safetyModalOpen, setSafetyModalOpen] = useState(false);
  const safetyTriggeredRef = useRef(false);

  // 多模态感知（仅用于综合焦虑指数计算，不展示手动分析入口）
  const perception = usePerception();

  // 综合焦虑指数 = 键盘 30% + 眨眼 30% + 文本感知 40%
  const textAnxietyIndex = perception.result
    ? Math.round((perception.result.text_emotion_probs[2] || 0) * 100) // 焦虑维度
    : 0;
  const hasPerception = perception.result !== null;
  const combinedAnxiety = hasPerception
    ? Math.round(
        (keyboard.metrics.anxietyIndex * 0.3) +
        (blink.metrics.anxietyIndex * 0.3) +
        (textAnxietyIndex * 0.4)
      )
    : Math.round(
        (keyboard.metrics.anxietyIndex * 0.5) + (blink.metrics.anxietyIndex * 0.5)
      );
  const anxietyLevel = combinedAnxiety > 60 ? 'high' : combinedAnxiety > 30 ? 'medium' : 'low';
  const anxietyColors = { low: '#52c41a', medium: '#faad14', high: '#ff4d4f' };
  const anxietyLabels = { low: '状态良好', medium: '轻度焦虑', high: '焦虑偏高' };

  // PHQ-9 第9题安全拦截：选非零值时立即触发关怀
  useEffect(() => {
    if (
      selectedAssessment?.type === 'PHQ9' &&
      answers[8] !== undefined &&
      answers[8] > 0 &&
      !safetyTriggeredRef.current
    ) {
      safetyTriggeredRef.current = true;
      setSafetyModalOpen(true);
    }
  }, [answers, selectedAssessment]);

  useEffect(() => { loadHistory(); }, []);
  useEffect(() => {
    if (combinedAnxiety > 50 && !realityTask) fetchRealityTask();
  }, [combinedAnxiety]);

  const loadHistory = async () => {
    try {
      const res = await profileApi.getAssessments() as any;
      setHistory(res.data || []);
    } catch {}
  };

  const fetchRealityTask = async () => {
    setTaskLoading(true);
    try {
      const res = await profileApi.generateRealityTask({
        anxietyLevel: combinedAnxiety,
        context: `键盘焦虑指数${keyboard.metrics.anxietyIndex}，眨眼焦虑指数${blink.metrics.anxietyIndex}`,
      }) as any;
      setRealityTask(res.data);
      if (combinedAnxiety > 50) setTaskModalOpen(true);
    } catch {} finally { setTaskLoading(false); }
  };

  const startTaskTimer = () => {
    setTaskTimerRunning(true);
    const duration = (realityTask?.duration || 5) * 60;
    let remaining = duration;
    setTaskTimer(remaining);
    const timer = setInterval(() => {
      remaining--;
      setTaskTimer(remaining);
      if (remaining <= 0) { clearInterval(timer); setTaskTimerRunning(false); }
    }, 1000);
  };

  const formatTime = (s: number) => `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, '0')}`;

  const handleSubmit = async () => {
    if (!selectedAssessment) return;
    setSubmitting(true);
    try {
      const res = await profileApi.submitAssessment({
        type: selectedAssessment.type,
        title: selectedAssessment.title,
        answers,
      }) as any;
      setResult(res.data);
      loadHistory();
    } catch {} finally { setSubmitting(false); }
  };

  const tabItems = [
    {
      key: 'realtime',
      label: <Space><DashboardOutlined />实时焦虑感知</Space>,
      children: (
        <div>
          {/* 综合焦虑指数 */}
          <Card style={{ borderRadius: 16, marginBottom: 24, textAlign: 'center',
            background: `linear-gradient(135deg, ${anxietyColors[anxietyLevel]}15 0%, ${anxietyColors[anxietyLevel]}05 100%)`,
            border: `2px solid ${anxietyColors[anxietyLevel]}30`,
          }}>
            <Text type="secondary" style={{ fontSize: 14 }}>综合焦虑指数</Text>
            {hasPerception && (
              <div style={{ marginTop: 4 }}>
                <Tag color="blue" style={{ fontSize: 11 }}>键盘 30%</Tag>
                <Tag color="cyan" style={{ fontSize: 11 }}>眨眼 30%</Tag>
                <Tag color="purple" style={{ fontSize: 11 }}>文本感知 40%</Tag>
              </div>
            )}
            {!hasPerception && (
              <div style={{ marginTop: 4 }}>
                <Tag color="blue" style={{ fontSize: 11 }}>键盘 50%</Tag>
                <Tag color="cyan" style={{ fontSize: 11 }}>眨眼 50%</Tag>
                <Tag color="default" style={{ fontSize: 11 }}>文本感知未激活</Tag>
              </div>
            )}
            <div style={{ margin: '12px 0' }}>
              <Progress type="dashboard" percent={combinedAnxiety} size={160}
                strokeColor={anxietyColors[anxietyLevel]}
                format={() => (
                  <div>
                    <div style={{ fontSize: 42, fontWeight: 'bold', color: anxietyColors[anxietyLevel] }}>{combinedAnxiety}</div>
                    <Tag color={anxietyColors[anxietyLevel]} style={{ fontSize: 14 }}>{anxietyLabels[anxietyLevel]}</Tag>
                  </div>
                )}
              />
            </div>
            {anxietyLevel === 'high' && (
              <Alert type="warning" showIcon
                message="检测到焦虑水平偏高"
                description="系统已为你准备了一个5分钟现实任务，帮助你从焦虑思维中抽离。"
                style={{ maxWidth: 400, margin: '0 auto', textAlign: 'left' }}
              />
            )}
          </Card>

          <Row gutter={[16, 16]}>
            {/* 键盘检测 */}
            <Col xs={24} md={12}>
              <Card title={<Space><KeyOutlined /> 键盘输入节奏</Space>}
                extra={
                  <Space>
                    {!keyboard.isActive ? (
                      <Button size="small" type="primary" onClick={keyboard.start}>开始监测</Button>
                    ) : (
                      <Button size="small" onClick={keyboard.stop}>停止</Button>
                    )}
                  </Space>
                }
                style={{ borderRadius: 12 }}>
                {keyboard.isActive ? (
                  <>
                    <Row gutter={[12, 12]}>
                      {[
                        { label: '打字速度', value: `${keyboard.metrics.typingSpeed} 字/分`, progress: Math.min(100, keyboard.metrics.typingSpeed), color: '#1890ff' },
                        { label: '删除率', value: `${keyboard.metrics.deletionRate}%`, progress: keyboard.metrics.deletionRate, color: keyboard.metrics.deletionRate > 30 ? '#ff4d4f' : '#52c41a' },
                        { label: '停顿频率', value: `${keyboard.metrics.pauseFrequency}%`, progress: keyboard.metrics.pauseFrequency, color: keyboard.metrics.pauseFrequency > 40 ? '#faad14' : '#52c41a' },
                        { label: '节奏稳定性', value: `${100 - keyboard.metrics.rhythmVariance}%`, progress: 100 - keyboard.metrics.rhythmVariance, color: '#13c2c2' },
                      ].map(({ label, value, progress, color }) => (
                        <Col xs={12} key={label}>
                          <Text type="secondary" style={{ fontSize: 12 }}>{label}</Text>
                          <div style={{ fontWeight: 600, fontSize: 14, color }}>{value}</div>
                          <Progress percent={progress} showInfo={false} size="small" strokeColor={color} />
                        </Col>
                      ))}
                    </Row>
                    <Divider style={{ margin: '12px 0' }} />
                    <div style={{ textAlign: 'center' }}>
                      <Text type="secondary">键盘焦虑指数</Text>
                      <div style={{ fontSize: 28, fontWeight: 'bold', color: anxietyColors[keyboard.metrics.anxietyIndex > 60 ? 'high' : keyboard.metrics.anxietyIndex > 30 ? 'medium' : 'low'] }}>
                        {keyboard.metrics.anxietyIndex}
                      </div>
                    </div>
                  </>
                ) : (
                  <div style={{ textAlign: 'center', padding: 30 }}>
                    <KeyOutlined style={{ fontSize: 36, color: '#d9d9d9', marginBottom: 12 }} />
                    <Text type="secondary">点击上方"开始监测"，然后正常打字</Text>
                  </div>
                )}
              </Card>
            </Col>

            {/* 眨眼检测 */}
            <Col xs={24} md={12}>
              <Card title={<Space><CameraOutlined /> 眨眼频率检测</Space>}
                extra={
                  <Space>
                    {!blink.metrics.isDetecting ? (
                      <Button size="small" type="primary" onClick={blink.start}>开启摄像头</Button>
                    ) : (
                      <Button size="small" danger onClick={blink.stop}>关闭</Button>
                    )}
                  </Space>
                }
                style={{ borderRadius: 12 }}>
                {blink.metrics.error ? (
                  <Alert type="error" message={blink.metrics.error} />
                ) : blink.metrics.isDetecting ? (
                  <>
                    <Row gutter={[12, 12]}>
                      {[
                        { label: '眨眼频率', value: `${blink.metrics.blinkRate} 次/分`, desc: '正常 15-20' },
                        { label: '平均时长', value: `${blink.metrics.avgBlinkDuration} ms`, desc: '正常 100-400' },
                        { label: '长眨眼', value: `${blink.metrics.longBlinkCount} 次`, desc: '>400ms 可能疲劳' },
                      ].map(({ label, value, desc }) => (
                        <Col xs={8} key={label}>
                          <Text type="secondary" style={{ fontSize: 11 }}>{label}</Text>
                          <div style={{ fontWeight: 600, fontSize: 16 }}>{value}</div>
                          <Text type="secondary" style={{ fontSize: 10 }}>{desc}</Text>
                        </Col>
                      ))}
                    </Row>
                    <Divider style={{ margin: '12px 0' }} />
                    <div style={{ textAlign: 'center' }}>
                      <Text type="secondary">眨眼焦虑指数</Text>
                      <div style={{ fontSize: 28, fontWeight: 'bold', color: anxietyColors[blink.metrics.anxietyIndex > 60 ? 'high' : blink.metrics.anxietyIndex > 30 ? 'medium' : 'low'] }}>
                        {blink.metrics.anxietyIndex}
                      </div>
                    </div>
                  </>
                ) : (
                  <div style={{ textAlign: 'center', padding: 30 }}>
                    <CameraOutlined style={{ fontSize: 36, color: '#d9d9d9', marginBottom: 12 }} />
                    <Text type="secondary">点击"开启摄像头"开始眨眼检测</Text>
                    <div style={{ marginTop: 8 }}>
                      <Text type="secondary" style={{ fontSize: 12 }}>仅在前端本地处理，视频不会上传到服务器</Text>
                    </div>
                  </div>
                )}
              </Card>
            </Col>

            {/* 5分钟现实任务 */}
            <Col xs={24}>
              <Card
                title={<Space><PlayCircleOutlined /> 5 分钟现实任务</Space>}
                extra={<Button size="small" icon={<ReloadOutlined />} onClick={fetchRealityTask} loading={taskLoading}>换一个</Button>}
                style={{ borderRadius: 12, background: 'linear-gradient(135deg, #fff7e6 0%, #f0f9ff 100%)' }}
              >
                {realityTask ? (
                  <Row gutter={24} align="middle">
                    <Col xs={24} md={16}>
                      <Tag color="orange" style={{ fontSize: 13, marginBottom: 8 }}>{realityTask.category}</Tag>
                      <Title level={5} style={{ margin: '0 0 8px' }}>{realityTask.title}</Title>
                      <Paragraph style={{ marginBottom: 12 }}>{realityTask.description}</Paragraph>
                      <div>
                        {realityTask.steps?.map((step: string, i: number) => (
                          <div key={i} style={{ padding: '4px 0', fontSize: 14 }}>
                            <CheckCircleOutlined style={{ color: '#52c41a', marginRight: 8 }} />{step}
                          </div>
                        ))}
                      </div>
                    </Col>
                    <Col xs={24} md={8} style={{ textAlign: 'center' }}>
                      <div style={{ background: '#fff', borderRadius: 16, padding: 24, boxShadow: '0 2px 8px rgba(0,0,0,0.06)' }}>
                        <Text type="secondary">倒计时</Text>
                        <div style={{ fontSize: 42, fontWeight: 'bold', color: '#fa8c16', margin: '8px 0' }}>
                          {taskTimerRunning ? formatTime(taskTimer) : `${realityTask.duration}:00`}
                        </div>
                        <Button type="primary" onClick={taskTimerRunning ? () => setTaskTimerRunning(false) : startTaskTimer}
                          style={{ borderRadius: 20 }}>
                          {taskTimerRunning ? '暂停' : taskTimer > 0 && taskTimer < (realityTask.duration || 5) * 60 ? '继续' : '开始任务'}
                        </Button>
                      </div>
                    </Col>
                  </Row>
                ) : (
                  <div style={{ textAlign: 'center', padding: 20 }}>
                    <Spin spinning={taskLoading}>
                      <Text type="secondary">点击"换一个"生成你的专属现实任务</Text>
                    </Spin>
                  </div>
                )}
              </Card>
            </Col>
          </Row>
        </div>
      ),
    },
    {
      key: 'scale',
      label: <Space><FileTextOutlined />标准化量表测评</Space>,
      children: (
        <div>
          {result ? (
            <Card style={{ borderRadius: 12, maxWidth: 600, margin: '0 auto' }}>
              <Result
                status={(result.level === 'normal' ? 'success' : result.level === 'mild' ? 'warning' : 'error') as any}
                title="测评完成"
                subTitle={`总分：${result.totalScore} | 等级：${result.level}`}
                extra={[<Button type="primary" key="back" onClick={() => { setResult(null); setSelectedAssessment(null); setAnswers({}); }}>返回</Button>]}
              >
                <Paragraph>{result.suggestion}</Paragraph>
              </Result>
            </Card>
          ) : selectedAssessment ? (
            <Card style={{ borderRadius: 12 }} title={selectedAssessment.title}
              extra={<Button onClick={() => { setSelectedAssessment(null); setAnswers({}); }}>返回</Button>}>
              <Paragraph type="secondary">{selectedAssessment.desc}</Paragraph>
              <Paragraph>请根据过去<strong>两周</strong>的情况，选择每项的程度：</Paragraph>
              <Paragraph type="secondary">0=完全没有，1=有几天，2=一半以上天数，3=几乎每天</Paragraph>
              <Space direction="vertical" style={{ width: '100%' }} size="middle">
                {selectedAssessment.questions.map((q: string, i: number) => (
                  <Card key={i} size="small" style={{ borderRadius: 8 }}>
                    <Text style={{ display: 'block', marginBottom: 8 }}>{i + 1}. {q}</Text>
                    <Radio.Group value={answers[i]} onChange={e => setAnswers({ ...answers, [i]: e.target.value })}>
                      <Space>{[0, 1, 2, 3].map(v => <Radio.Button key={v} value={v}>{v}</Radio.Button>)}</Space>
                    </Radio.Group>
                  </Card>
                ))}
                <Button type="primary" size="large" block onClick={handleSubmit}
                  loading={submitting}
                  disabled={Object.keys(answers).length < selectedAssessment.questions.length}>
                  提交测评
                </Button>
              </Space>
            </Card>
          ) : (
            <>
              <Space direction="vertical" style={{ width: '100%' }} size="middle">
                {assessments.map(a => (
                  <Card key={a.type} hoverable onClick={() => setSelectedAssessment(a)}
                    style={{ borderRadius: 12 }} bodyStyle={{ padding: 20 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div>
                        <Title level={5} style={{ marginBottom: 4 }}>{a.title}</Title>
                        <Text type="secondary">{a.desc}</Text>
                        <div style={{ marginTop: 8 }}><Tag>{a.questions.length} 题</Tag></div>
                      </div>
                      <Button type="primary">开始测评</Button>
                    </div>
                  </Card>
                ))}
              </Space>
              <Title level={5} style={{ marginTop: 24, marginBottom: 12 }}>历史记录</Title>
              {history.length === 0 ? <Empty description="暂无测评记录" /> : (
                <List dataSource={history} renderItem={(item: any) => (
                  <List.Item>
                    <List.Item.Meta title={item.title}
                      description={`得分：${item.totalScore} | 等级：${item.level} | ${new Date(item.createdAt).toLocaleDateString()}`} />
                    <Tag color={item.level === 'normal' ? 'green' : item.level === 'mild' ? 'orange' : 'red'}>{item.level}</Tag>
                  </List.Item>
                )} />
              )}
            </>
          )}
        </div>
      ),
    },
  ];

  return (
    <div>
      <Title level={4} style={{ marginBottom: 16 }}>
        <ExperimentOutlined /> 了解你的情绪状态
      </Title>
      <Text type="secondary" style={{ display: 'block', marginBottom: 24 }}>
        通过行为小实验、多模态情感分析和经典问卷，帮你更好地了解自己最近的状态。
        {hasPerception && <Tag color="purple" style={{ marginLeft: 8 }}>多模态融合已激活</Tag>}
      </Text>
      <Tabs defaultActiveKey="realtime" items={tabItems} size="large" />

      {/* PHQ-9 安全关怀弹窗 */}
      <Modal
        open={safetyModalOpen}
        onCancel={() => { setSafetyModalOpen(false); setSafetyModalOpen(false); }}
        footer={null}
        width={480}
        centered
      >
        <div style={{ textAlign: 'center', padding: '20px 0' }}>
          <div style={{ fontSize: 48, marginBottom: 12 }}>💛</div>
          <Title level={4} style={{ marginBottom: 8, color: '#5a4a6a' }}>
            谢谢你愿意说出来
          </Title>
          <div style={{
            padding: '16px 20px', background: '#fff7e6', borderRadius: 12,
            textAlign: 'left', marginBottom: 20,
          }}>
            <Typography.Paragraph style={{ fontSize: 15, color: '#5a4a6a', marginBottom: 8 }}>
              你刚才提到的那些想法，让我们很关心你现在的感受。
            </Typography.Paragraph>
            <Typography.Paragraph style={{ fontSize: 14, color: '#8a7a9a', marginBottom: 0 }}>
              请记住，你不是一个人。现在有人愿意听你说，24小时都在：
            </Typography.Paragraph>
          </div>
          <Card style={{ borderRadius: 12, marginBottom: 20, background: '#fff5f5', border: '1px solid #ffccc7' }}>
            <div style={{ marginBottom: 12 }}>
              <Text strong>🆘 24小时心理援助热线</Text>
              <Title level={3} style={{ color: '#ff4d4f', margin: '4px 0' }}>400-161-9995</Title>
            </div>
            <div style={{ marginBottom: 0 }}>
              <Text strong>💬 生命热线</Text>
              <Title level={3} style={{ color: '#6366f1', margin: '4px 0' }}>400-821-1215</Title>
            </div>
          </Card>
          <Space direction="vertical" style={{ width: '100%' }} size="middle">
            <Button
              type="primary"
              block
              size="large"
              onClick={() => { setSafetyModalOpen(false); navigate('/chat'); }}
              style={{ borderRadius: 12 }}
            >
              <HeartOutlined /> 去找 AI 聊聊，它会一直陪着你
            </Button>
            <Button
              block
              size="large"
              onClick={() => { setSafetyModalOpen(false); setSafetyModalOpen(false); }}
              style={{ borderRadius: 12 }}
            >
              我暂时还好，继续测评
            </Button>
          </Space>
        </div>
      </Modal>

      {/* 焦虑超标弹窗 */}
      <Modal open={taskModalOpen} onCancel={() => setTaskModalOpen(false)} footer={null} width={480} centered>
        {realityTask && (
          <div style={{ textAlign: 'center', padding: '20px 0' }}>
            <div style={{ fontSize: 48, marginBottom: 12 }}></div>
            <Title level={4} style={{ marginBottom: 8 }}>检测到焦虑偏高</Title>
            <Text type="secondary">不如先放下手机，试试这个：</Text>
            <Card size="small" style={{ marginTop: 16, textAlign: 'left', borderRadius: 12 }}>
              <Tag color="orange">{realityTask.category}</Tag>
              <Title level={5} style={{ margin: '8px 0' }}>{realityTask.title}</Title>
              <Text>{realityTask.description}</Text>
              <div style={{ marginTop: 12 }}>
                {realityTask.steps?.map((step: string, i: number) => (
                  <div key={i} style={{ padding: '3px 0', fontSize: 13 }}>
                    <CheckCircleOutlined style={{ color: '#52c41a', marginRight: 6 }} />{step}
                  </div>
                ))}
              </div>
            </Card>
            <Button type="primary" size="large" onClick={() => { setTaskModalOpen(false); startTaskTimer(); }}
              style={{ marginTop: 20, borderRadius: 24, padding: '0 40px' }}>
              开始 {realityTask.duration} 分钟
            </Button>
          </div>
        )}
      </Modal>
    </div>
  );
}
