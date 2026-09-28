/**
 * 危机资源卡片（施工单 §S6.1）
 *
 * ## 为什么抽成组件
 *
 * 文字对话页（`Chat.tsx`）与通话页（`VoiceCall.tsx`）在危机时**必须给出同一份信息**：
 * 同一句安抚、同一个号码、同一组按钮。两份实现迟早会漂移，而这类漂移出现在
 * 危机场景里是不可接受的（用户看到的号码不一致，或者一边改了文案另一边没改）。
 *
 * ## ⚠️ 只搬不改
 *
 * 下面的文案、热线号码、机构名称与改造前 `Chat.tsx:648-688` 的 `CrisisModal`
 * **一个字都没有动**。任何调整（包括换号码、改语气）都属安全相关改动，
 * 按 AGENTS.md 必须先经人工审核 —— 不要在这一步"顺手润色"。
 *
 * ## 号码为什么只上屏、不朗读
 *
 * 实测（`docs/echo_calibration.md` §4.3）：TTS 把 `400 161 9995` 念成
 * 「四百六十一九九九九五」—— 它当成了一个 11 位数。所以通话里号码**只以卡片呈现**，
 * 并由 `useVoiceCall` 的 `PHONE_LIKE` 守卫确保含号码特征的片段根本不进合成。
 */

import { Button, Card, Space, Typography } from 'antd';
import { HeartOutlined } from '@ant-design/icons';

const { Paragraph, Text, Title } = Typography;

export interface CrisisResourceCardProps {
  /** 主按钮（"继续和 AI 聊聊"）；不传则不显示 */
  onPrimary?: () => void;
  /** 次按钮（"我知道了"）；不传则不显示 */
  onDismiss?: () => void;
  /** 主按钮文案；默认沿用文字对话页里的那句 */
  primaryLabel?: string;
  /**
   * 紧凑模式：通话页是全屏浮层，字号比弹窗略小一点更好看，
   * 但**文案与结构完全一致**（只有排版尺寸变）。
   */
  compact?: boolean;
  /** 顶部再加一行上下文（通话页用来解释"为什么突然停了"）；传空则不显示 */
  context?: string;
}

export default function CrisisResourceCard({
  onPrimary,
  onDismiss,
  primaryLabel = '继续和 AI 聊聊',
  compact = false,
  context,
}: CrisisResourceCardProps) {
  return (
    <div style={{ textAlign: 'center', padding: compact ? '8px 0' : '20px 0' }}>
      <div style={{ fontSize: compact ? 38 : 48, marginBottom: 12 }}>💛</div>
      <Title level={4} style={{ marginBottom: 8, color: '#5a4a6a' }}>
        我们很关心你
      </Title>
      {context && (
        <Paragraph style={{ fontSize: compact ? 13 : 14, color: '#8a7a9a', marginBottom: 12 }}>
          {context}
        </Paragraph>
      )}
      <div
        style={{
          padding: '16px 20px',
          background: '#fff7e6',
          borderRadius: 12,
          textAlign: 'left',
          marginBottom: 20,
        }}
      >
        <Paragraph style={{ fontSize: 15, color: '#5a4a6a', marginBottom: 8 }}>
          你说的话让我们有些担心。不管发生了什么，你都不是一个人。
        </Paragraph>
        <Paragraph style={{ fontSize: 14, color: '#8a7a9a', marginBottom: 0 }}>
          现在有人愿意听你说，24小时都在：
        </Paragraph>
      </div>
      <Card
        style={{
          borderRadius: 12,
          marginBottom: 20,
          background: '#fff5f5',
          border: '1px solid #ffccc7',
        }}
      >
        <div style={{ marginBottom: 12 }}>
          <Text strong>🆘 24小时心理援助热线</Text>
          <Title level={3} style={{ color: '#ff4d4f', margin: '4px 0' }}>
            400-161-9995
          </Title>
        </div>
        <div>
          <Text strong>💬 生命热线</Text>
          <Title level={3} style={{ color: '#6366f1', margin: '4px 0' }}>
            400-821-1215
          </Title>
        </div>
      </Card>
      {(onPrimary || onDismiss) && (
        <Space direction="vertical" style={{ width: '100%' }} size="middle">
          {onPrimary && (
            <Button type="primary" block size="large" onClick={onPrimary} style={{ borderRadius: 12 }}>
              <HeartOutlined /> {primaryLabel}
            </Button>
          )}
          {onDismiss && (
            <Button block size="large" onClick={onDismiss} style={{ borderRadius: 12 }}>
              我知道了
            </Button>
          )}
        </Space>
      )}
    </div>
  );
}
