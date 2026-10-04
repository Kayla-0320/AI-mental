import { Modal, Typography, Tag, Button, Card } from 'antd';
import { CheckCircleOutlined } from '@ant-design/icons';
import { useAnxiety } from '../context/AnxietyContext';

const { Title, Text } = Typography;

export default function RealityTaskModal() {
  const { realityTask, showTaskModal, setShowTaskModal, fetchRealityTask } = useAnxiety();

  if (!realityTask) return null;

  return (
    <Modal
      open={showTaskModal}
      onCancel={() => setShowTaskModal(false)}
      footer={null}
      width={460}
      centered
      closable={false}
    >
      <div style={{ textAlign: 'center', padding: '16px 0' }}>
        <div style={{ fontSize: 48, marginBottom: 8 }}></div>
        <Title level={4} style={{ marginBottom: 4 }}>检测到焦虑偏高</Title>
        <Text type="secondary" style={{ fontSize: 13 }}>不如先放下手机，试试这个：</Text>

        <Card size="small" style={{ marginTop: 16, textAlign: 'left', borderRadius: 12 }}>
          <Tag color="orange">{realityTask.category}</Tag>
          <Title level={5} style={{ margin: '8px 0 4px' }}>{realityTask.title}</Title>
          <Text style={{ fontSize: 13 }}>{realityTask.description}</Text>
          <div style={{ marginTop: 12 }}>
            {realityTask.steps?.map((step: string, i: number) => (
              <div key={i} style={{ padding: '3px 0', fontSize: 13 }}>
                <CheckCircleOutlined style={{ color: '#52c41a', marginRight: 6 }} />{step}
              </div>
            ))}
          </div>
        </Card>

        <div style={{ marginTop: 20, display: 'flex', gap: 12, justifyContent: 'center' }}>
          <Button onClick={() => { setShowTaskModal(false); fetchRealityTask(); }}>
            换一个
          </Button>
          <Button type="primary" onClick={() => setShowTaskModal(false)}
            style={{ borderRadius: 20, padding: '0 32px' }}>
            我知道了
          </Button>
        </div>
      </div>
    </Modal>
  );
}
