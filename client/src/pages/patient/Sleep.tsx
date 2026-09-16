import { useState, useEffect } from 'react';
import { Card, Row, Col, Typography, Button, Modal, TimePicker, Slider, Input, message, Statistic } from 'antd';
import { ClockCircleOutlined, PlusOutlined } from '@ant-design/icons';
import { extraApi } from '../../services';
import dayjs from 'dayjs';

const { Title, Text } = Typography;
const { TextArea } = Input;

export default function Sleep() {
  const [records, setRecords] = useState<any[]>([]);
  const [stats, setStats] = useState<any>({ avgDuration: 0, avgQuality: 0, total: 0 });
  const [modal, setModal] = useState(false);
  const [bedTime, setBedTime] = useState<dayjs.Dayjs | null>(null);
  const [wakeTime, setWakeTime] = useState<dayjs.Dayjs | null>(null);
  const [quality, setQuality] = useState(5);
  const [notes, setNotes] = useState('');

  const loadData = () => {
    extraApi.getSleepRecords(30).then((res: any) => { if (res.code === 0) setRecords(res.data); });
    extraApi.getSleepStats(30).then((res: any) => { if (res.code === 0) setStats(res.data); });
  };

  useEffect(() => { loadData(); }, []);

  const handleRecord = async () => {
    if (!bedTime || !wakeTime) return message.warning('请选择入睡和起床时间');
    try {
      await extraApi.recordSleep({
        bedTime: bedTime.toISOString(),
        wakeTime: wakeTime.toISOString(),
        quality,
        notes,
      });
      message.success('记录成功');
      setModal(false);
      setBedTime(null);
      setWakeTime(null);
      setQuality(5);
      setNotes('');
      loadData();
    } catch (e: any) { message.error(e.message); }
  };

  const moodEmojis: Record<number, string> = {
    1: '😫', 2: '😣', 3: '😕', 4: '😐', 5: '😌', 6: '😊', 7: '😄', 8: '🥰', 9: '✨', 10: '🌟',
  };

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <Title level={4} style={{ margin: 0, color: '#5a4a6a' }}>睡眠追踪</Title>
        <Button className="cloud-btn" icon={<PlusOutlined />} onClick={() => setModal(true)}>
          记录睡眠
        </Button>
      </div>

      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={8}>
          <Card className="cloud-card" style={{ textAlign: 'center' }}>
            <Statistic title="平均时长" value={stats.avgDuration} suffix="小时" valueStyle={{ color: '#ff8fab', fontSize: 28 }} />
          </Card>
        </Col>
        <Col span={8}>
          <Card className="cloud-card" style={{ textAlign: 'center' }}>
            <Statistic title="平均质量" value={`${stats.avgQuality}/10`} valueStyle={{ color: '#ff8fab', fontSize: 28 }} />
            <div style={{ fontSize: 24, marginTop: 4 }}>{moodEmojis[stats.avgQuality] || '😐'}</div>
          </Card>
        </Col>
        <Col span={8}>
          <Card className="cloud-card" style={{ textAlign: 'center' }}>
            <Statistic title="记录天数" value={stats.total} suffix="天" valueStyle={{ color: '#ff8fab', fontSize: 28 }} />
          </Card>
        </Col>
      </Row>

      <Title level={5} style={{ color: '#5a4a6a', marginBottom: 12 }}>最近记录</Title>
      {records.length === 0 ? (
        <Card className="cloud-card" style={{ textAlign: 'center', padding: 40 }}>
          <Text style={{ color: '#8a7a9a' }}>还没有睡眠记录，点击上方按钮开始记录吧</Text>
        </Card>
      ) : (
        <Row gutter={[16, 16]}>
          {records.map((r: any) => (
            <Col xs={24} sm={12} lg={8} key={r.id}>
              <Card className="cloud-card">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Text strong style={{ color: '#5a4a6a' }}>{dayjs(r.recordDate).format('MM/DD')}</Text>
                  <span style={{ fontSize: 24 }}>{moodEmojis[r.quality] || '😐'}</span>
                </div>
                <div style={{ marginTop: 8 }}>
                  <Text style={{ fontSize: 13, color: '#8a7a9a' }}>
                    <ClockCircleOutlined style={{ marginRight: 4 }} />
                    {r.duration} 小时
                  </Text>
                </div>
                {r.notes && <Text style={{ fontSize: 12, color: '#b0a0c0', display: 'block', marginTop: 4 }}>{r.notes}</Text>}
              </Card>
            </Col>
          ))}
        </Row>
      )}

      <Modal open={modal} onCancel={() => setModal(false)} onOk={handleRecord} title="记录睡眠" okText="保存">
        <div style={{ marginBottom: 16 }}>
          <Text style={{ display: 'block', marginBottom: 8 }}>入睡时间</Text>
          <TimePicker value={bedTime} onChange={setBedTime} style={{ width: '100%', borderRadius: 12 }} format="HH:mm" />
        </div>
        <div style={{ marginBottom: 16 }}>
          <Text style={{ display: 'block', marginBottom: 8 }}>起床时间</Text>
          <TimePicker value={wakeTime} onChange={setWakeTime} style={{ width: '100%', borderRadius: 12 }} format="HH:mm" />
        </div>
        <div style={{ marginBottom: 16 }}>
          <Text style={{ display: 'block', marginBottom: 8 }}>睡眠质量：{quality}/10 {moodEmojis[quality]}</Text>
          <Slider min={1} max={10} value={quality} onChange={setQuality} />
        </div>
        <TextArea rows={2} placeholder="备注（可选）" value={notes} onChange={e => setNotes(e.target.value)} style={{ borderRadius: 12 }} />
      </Modal>
    </div>
  );
}
