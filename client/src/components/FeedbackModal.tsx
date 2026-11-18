/**
 * 反馈评价弹窗（受控组件）
 *
 * 调用方：顶栏「更多 → 反馈评价」（见 `pages/patient/PatientLayout.tsx`）。
 * 之所以独立成组件而不是写在页面里：反馈的**唯一可用端点**是 `/extra/feedback`，
 * 这个弹窗是前端提交反馈的唯一实现，将来别的页面要加入口时直接复用，
 * 不该再抄一份（`Settings.tsx` 里那份内联表单打的是不存在的 `/api/feedback`，
 * 属于历史遗留问题，尚未修）。
 *
 * 组件本身不持开关状态（`open` / `onClose` 由调用方给），只持表单状态。
 * 标题沿用后端要求的 `title` 字段，取内容前 20 字 —— 弹窗里不单独问标题，
 * 少一个必填项就少一次"填一半放弃"。
 */
import { useState } from 'react';
import { Modal, Input, Button, Rate, Space, Typography, message } from 'antd';
import { SoundOutlined } from '@ant-design/icons';
import { extraApi } from '../services';

const { Text } = Typography;

const FEEDBACK_TYPES = [
  { key: 'suggestion', label: '💡 建议' },
  { key: 'bug', label: '🐛 问题' },
  { key: 'feature', label: '✨ 新功能' },
  { key: 'other', label: '💬 其他' },
];

const RATING_HINTS: Array<[number, string]> = [
  [2, '😔 我们会继续改进'],
  [3, '🤔 还有进步空间'],
  [4, '😊 谢谢你的认可'],
  [5, '🎉 太开心了！'],
];

export default function FeedbackModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [type, setType] = useState('suggestion');
  const [rating, setRating] = useState(0);
  const [content, setContent] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async () => {
    const text = content.trim();
    if (!text || submitting) return;
    setSubmitting(true);
    try {
      await extraApi.createFeedback({ type, title: text.slice(0, 20), content: text });
      message.success('感谢你的反馈！我们会继续努力 💪');
      setContent('');
      setRating(0);
      setType('suggestion');
      onClose();
    } catch {
      message.error('发送失败，请稍后再试');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Modal
      open={open}
      onCancel={onClose}
      footer={null}
      width={480}
      title={<Space><SoundOutlined /> 反馈评价</Space>}
    >
      <div style={{ padding: '8px 0' }}>
        <div style={{ marginBottom: 16, textAlign: 'center' }}>
          <Text style={{ display: 'block', marginBottom: 8 }}>你觉得这个平台怎么样？</Text>
          <Rate value={rating} onChange={setRating} />
          {rating > 0 && (
            <Text type="secondary" style={{ marginLeft: 8, fontSize: 13 }}>
              {RATING_HINTS.find(([max]) => rating <= max)?.[1]}
            </Text>
          )}
        </div>
        <div style={{ marginBottom: 12 }}>
          <Space wrap>
            {FEEDBACK_TYPES.map(({ key, label }) => (
              <div
                key={key}
                onClick={() => setType(key)}
                style={{
                  padding: '4px 14px', borderRadius: 20, cursor: 'pointer', fontSize: 13,
                  background: type === key ? '#fff0f5' : '#f5f5f5',
                  color: type === key ? '#ff8fab' : '#666',
                  border: type === key ? '1px solid #ff8fab' : '1px solid transparent',
                }}
              >
                {label}
              </div>
            ))}
          </Space>
        </div>
        <Input.TextArea
          rows={4}
          placeholder="告诉我们你的想法，每一条反馈都会认真看~"
          value={content}
          onChange={(e) => setContent(e.target.value)}
          style={{ borderRadius: 12, marginBottom: 12 }}
        />
        <Button
          type="primary"
          block
          loading={submitting}
          disabled={!content.trim()}
          onClick={handleSubmit}
          style={{ borderRadius: 24, background: 'linear-gradient(135deg, #ff8fab 0%, #c084fc 100%)', border: 'none' }}
        >
          提交反馈
        </Button>
      </div>
    </Modal>
  );
}
