import { useState, useEffect } from 'react';
import { Modal, Button, Space, Typography, Tag, message } from 'antd';
import { PhoneFilled, HeartFilled, CloseOutlined } from '@ant-design/icons';
import { crisisApi } from '../services';

const { Title, Text } = Typography;

export default function CrisisInterventionWidget() {
  const [visible, setVisible] = useState(false);
  const [hotlines, setHotlines] = useState<any[]>([]);
  const [crisisDetected, setCrisisDetected] = useState(false);

  useEffect(() => {
    crisisApi.getHotlines().then((res: any) => {
      if (res.code === 0) setHotlines(res.data);
    }).catch(() => {});
  }, []);

  // 监听 AI 对话中的危机信号
  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail;
      if (detail?.isCrisis) {
        setCrisisDetected(true);
        setVisible(true);
      }
    };
    window.addEventListener('crisis-detected', handler);
    return () => window.removeEventListener('crisis-detected', handler);
  }, []);

  return (
    <>
      {/* 浮动紧急按钮 */}
      <Button
        onClick={() => setVisible(true)}
        style={{
          position: 'fixed',
          bottom: 140,
          right: 20,
          width: 48,
          height: 48,
          borderRadius: '50%',
          background: 'linear-gradient(135deg, #ff4d4f, #ff7875)',
          border: 'none',
          boxShadow: '0 4px 20px rgba(255,77,79,0.4)',
          zIndex: 1000,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: 20,
        }}
        title="紧急求助"
      >
        <HeartFilled style={{ color: '#fff', fontSize: 20 }} />
      </Button>

      <Modal
        open={visible}
        onCancel={() => { setVisible(false); setCrisisDetected(false); }}
        footer={null}
        width={480}
        closable={false}
        style={{ borderRadius: 24 }}
      >
        <div style={{ textAlign: 'center', padding: '16px 0' }}>
          {crisisDetected && (
            <div style={{
              background: '#fff2f0',
              border: '1px solid #ffccc7',
              borderRadius: 16,
              padding: '12px 16px',
              marginBottom: 16,
            }}>
              <Text style={{ color: '#cf1322', fontSize: 14 }}>
                我们注意到你可能正在经历困难时刻。请记住，你并不孤单，有人愿意帮助你。
              </Text>
            </div>
          )}

          <div style={{ fontSize: 48, marginBottom: 8 }}></div>
          <Title level={4} style={{ color: '#5a4a6a', marginBottom: 4 }}>你并不孤单</Title>
          <Text style={{ color: '#8a7a9a', fontSize: 14 }}>
            以下热线由专业心理咨询师值守，24小时免费为你服务
          </Text>
        </div>

        <div style={{ marginTop: 16 }}>
          {hotlines.map((h, i) => (
            <div
              key={i}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '12px 16px',
                background: '#fff5f7',
                borderRadius: 12,
                marginBottom: 8,
                border: '1px solid rgba(255,182,193,0.2)',
              }}
            >
              <div>
                <Text strong style={{ fontSize: 14, color: '#5a4a6a' }}>{h.name}</Text>
                <div>
                  <Tag color="pink" style={{ fontSize: 11, borderRadius: 20 }}>{h.hours}</Tag>
                </div>
              </div>
              <a href={`tel:${h.phone}`} style={{ textDecoration: 'none' }}>
                <Button
                  type="primary"
                  size="small"
                  icon={<PhoneFilled />}
                  style={{
                    borderRadius: 20,
                    background: 'linear-gradient(135deg, #ffb6c1, #ffc8d6)',
                    border: 'none',
                  }}
                >
                  {h.phone}
                </Button>
              </a>
            </div>
          ))}
        </div>

        <div style={{ textAlign: 'center', marginTop: 16 }}>
          <Button
            onClick={() => { setVisible(false); setCrisisDetected(false); }}
            style={{ borderRadius: 20, color: '#8a7a9a' }}
          >
            我知道了
          </Button>
        </div>
      </Modal>
    </>
  );
}
