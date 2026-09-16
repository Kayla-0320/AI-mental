import { useState, useEffect } from 'react';
import { Card, Typography, Button, Tag, Space, Radio, Slider, Result, List, Empty, Spin } from 'antd';
import { FileTextOutlined, CheckCircleOutlined } from '@ant-design/icons';
import { profileApi } from '../../services';

const { Title, Text, Paragraph } = Typography;

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

export default function Assessment() {
  const [selectedAssessment, setSelectedAssessment] = useState<any>(null);
  const [answers, setAnswers] = useState<Record<number, number>>({});
  const [result, setResult] = useState<any>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => { loadHistory(); }, []);

  const loadHistory = async () => {
    setLoading(true);
    try {
      const res = await profileApi.getAssessments() as any;
      setHistory(res.data || []);
    } catch {} finally { setLoading(false); }
  };

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

  if (result) {
    const levelColor = result.level === 'normal' ? 'success' : result.level === 'mild' ? 'warning' : 'error';
    return (
      <Card style={{ borderRadius: 12, maxWidth: 600, margin: '0 auto' }}>
        <Result status={levelColor as any} title={`测评完成`}
          subTitle={`总分：${result.totalScore} | 等级：${result.level}`}
          extra={[
            <Button type="primary" key="back" onClick={() => { setResult(null); setSelectedAssessment(null); setAnswers({}); }}>
              返回
            </Button>,
          ]}>
          <Paragraph>{result.suggestion}</Paragraph>
        </Result>
      </Card>
    );
  }

  if (selectedAssessment) {
    return (
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
                <Space>
                  {[0, 1, 2, 3].map(v => (
                    <Radio.Button key={v} value={v}>{v}</Radio.Button>
                  ))}
                </Space>
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
    );
  }

  return (
    <div>
      <Title level={4} style={{ marginBottom: 16 }}><FileTextOutlined /> 心理测评</Title>
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
      {loading ? <Spin /> : history.length === 0 ? <Empty description="暂无测评记录" /> : (
        <List dataSource={history} renderItem={(item: any) => (
          <List.Item>
            <List.Item.Meta title={item.title}
              description={`得分：${item.totalScore} | 等级：${item.level} | ${new Date(item.createdAt).toLocaleDateString()}`} />
            <Tag color={item.level === 'normal' ? 'green' : item.level === 'mild' ? 'orange' : 'red'}>{item.level}</Tag>
          </List.Item>
        )} />
      )}
    </div>
  );
}
