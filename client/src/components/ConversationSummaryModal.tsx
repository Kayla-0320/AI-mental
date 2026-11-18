/**
 * 对话总结弹窗（元宝纪要风格）
 *
 * ## 职责边界
 *
 * 这个组件**只负责看与点**：拉取已保存的总结、在用户点击时请求生成、渲染五段内容。
 * 生成逻辑全在服务端（`server/src/services/consultation.service.ts`），
 * 这里不做任何文本加工 —— 避免客户端"顺手美化"出与库里不一致的内容。
 *
 * ## 为什么打开弹窗不自动生成
 *
 * 生成一次要调一次大模型（花钱、要等两秒以上）。历史记录是高频浏览动作，
 * 若"点进去看一眼"就等于"烧一次 token"，用户会开始躲着这个按钮点。
 * 所以：**有总结就直接看；没有总结才显示"生成总结"按钮，由用户决定**。
 *
 * ## 安全红线（AGENTS.md）
 *
 * 弹窗内所有文案均不得出现诊断性表述与用药建议。提示词侧已在
 * `server/src/services/summary.prompt.ts` 里做了约束；这里额外保证**界面上不再叠加**
 * 任何判断性标签（例如"你的焦虑指数"）—— 只呈现模型给出的纪要文本。
 */

import { useCallback, useEffect, useState } from 'react';
import { Alert, Button, Empty, Modal, Skeleton, Space, Tag, Tooltip, Typography, message } from 'antd';
import {
  BulbOutlined,
  CommentOutlined,
  FileTextOutlined,
  HistoryOutlined,
  LineChartOutlined,
  ReloadOutlined,
  StarOutlined,
} from '@ant-design/icons';
import { consultationApi } from '../services';
const { Text, Paragraph } = Typography;

/** 与服务端 `ConversationSummaryRecord` 同构（服务端为唯一权威定义处）。 */
export interface ConversationSummaryData {
  overview: string;
  keyPoints: string[];
  emotionalTrend: string;
  helpfulSuggestions: string[];
  followUps: string[];
  generatedAt: string;
  messageCount: number;
  model: string;
}

/** 单个会话的总结载荷（服务端 getConversationSummary 的 data） */
export interface ConversationSummaryPayload {
  conversationId: string;
  title: string;
  messageCount: number;
  summary: ConversationSummaryData | null;
}

export interface ConversationSummaryModalProps {
  /** 要查看总结的会话 id；为 null 时弹窗关闭 */
  conversationId: string | null;
  /** 会话标题（仅在尚未拿到服务端响应时用于标题占位） */
  conversationTitle?: string;
  open: boolean;
  onClose: () => void;
}

/** 一段带图标的小标题 */
function SectionTitle({ icon, text }: { icon: React.ReactNode; text: string }) {
  return (
    <Space size={6} align="center" style={{ marginBottom: 6 }}>
      <span style={{ color: '#6366f1', fontSize: 14 }}>{icon}</span>
      <Text strong style={{ fontSize: 13, color: '#4a4a6a' }}>
        {text}
      </Text>
    </Space>
  );
}

/** 无序要点列表。刻意不用 antd List：这里的行距与缩进需要和正文段落对齐 */
function BulletList({ items, color = '#4a4a6a' }: { items: string[]; color?: string }) {
  return (
    <ul style={{ margin: 0, paddingLeft: 20, color, lineHeight: 1.85, fontSize: 13 }}>
      {items.map((item, index) => (
        <li key={`${index}-${item}`} style={{ marginBottom: 2 }}>
          {item}
        </li>
      ))}
    </ul>
  );
}

export default function ConversationSummaryModal({
  conversationId,
  conversationTitle,
  open,
  onClose,
}: ConversationSummaryModalProps) {
  const [payload, setPayload] = useState<ConversationSummaryPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /** 拉取已保存的总结。**不触发生成** —— 见文件头说明。 */
  const loadSummary = useCallback(async (id: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = (await consultationApi.getConversationSummary(id)) as any;
      setPayload(res?.data ?? null);
    } catch (e: any) {
      setError(e?.message || '读取总结失败，请稍后重试');
      setPayload(null);
    } finally {
      setLoading(false);
    }
  }, []);

  // 打开时拉一次；切换会话时（弹窗不关、直接点另一个会话的总结）也要重拉。
  useEffect(() => {
    if (!open || !conversationId) return;
    setPayload(null);
    void loadSummary(conversationId);
  }, [open, conversationId, loadSummary]);

  /**
   * 生成总结。
   *
   * `force` 只在"重新生成"时传 true：服务端默认幂等（已有总结直接返回旧的），
   * 所以正常路径下这里不会重复调用模型。
   */
  const handleGenerate = useCallback(
    async (force: boolean) => {
      if (!conversationId) return;
      setGenerating(true);
      setError(null);
      try {
        const res = (await consultationApi.generateConversationSummary(conversationId, force)) as any;
        setPayload(res?.data ?? null);
        message.success(res?.data?.generated === false ? '已显示上次生成的总结' : '总结已生成');
      } catch (e: any) {
        // 后端对"还没有内容的对话"返回 400 且带一句人话，直接透出这句话，
        // 不要用通用的"生成失败"盖掉它 —— 用户据此就知道该先去聊几句。
        setError(e?.message || '生成总结失败，请稍后重试');
      } finally {
        setGenerating(false);
      }
    },
    [conversationId],
  );

  const summary = payload?.summary ?? null;
  const title = payload?.title || conversationTitle || '对话总结';
  // 原生 Date 而不是 dayjs：这里只需要"哪一天几点"，不值得为一个格式化再引一个依赖
  const generatedAtText = summary?.generatedAt
    ? new Date(summary.generatedAt).toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
      })
    : '';

  return (
    <Modal
      open={open}
      onCancel={onClose}
      footer={null}
      width={640}
      centered
      destroyOnClose
      title={
        <Space size={8} align="center">
          <FileTextOutlined style={{ color: '#6366f1' }} />
          <span>对话总结</span>
          <Text type="secondary" ellipsis style={{ fontSize: 12, fontWeight: 400, maxWidth: 300 }}>
            {title}
          </Text>
        </Space>
      }
    >
      {/* 元信息条：只显示可核实的事实（条数 / 生成时间 / 模型），不显示任何评分 */}
      {(summary || loading) && (
        <Space size={6} wrap style={{ marginBottom: 12 }}>
          <Tag style={{ fontSize: 11, margin: 0 }} color="purple">
            共 {summary?.messageCount || payload?.messageCount || 0} 条消息
          </Tag>
          {generatedAtText && (
            <Tag style={{ fontSize: 11, margin: 0 }}>
              <HistoryOutlined /> {generatedAtText} 生成
            </Tag>
          )}
          {summary?.model && (
            <Tooltip title="由本平台的 AI 助手生成，仅供你自己回顾，不构成任何医学意见">
              <Tag style={{ fontSize: 11, margin: 0 }}>{summary.model}</Tag>
            </Tooltip>
          )}
        </Space>
      )}

      {loading ? (
        <Skeleton active paragraph={{ rows: 6 }} />
      ) : error ? (
        <Alert
          type="warning"
          showIcon
          message={error}
          action={
            <Button size="small" onClick={() => (conversationId ? void handleGenerate(false) : undefined)}>
              重试
            </Button>
          }
        />
      ) : !summary ? (
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description={
            <span style={{ fontSize: 13, color: '#8a7a9a' }}>
              这段对话还没有总结。生成后会保存在本地，随时可以回来看。
            </span>
          }
        >
          <Button
            type="primary"
            icon={<StarOutlined />}
            loading={generating}
            onClick={() => void handleGenerate(false)}
            style={{ borderRadius: 20 }}
          >
            生成总结
          </Button>
        </Empty>
      ) : (
        <div style={{ maxHeight: '58vh', overflowY: 'auto', paddingRight: 4 }}>
          {/* 概述 */}
          {summary.overview && (
            <div
              style={{
                background: 'linear-gradient(135deg, #f5f3ff, #eef2ff)',
                borderRadius: 12,
                padding: '12px 14px',
                marginBottom: 14,
              }}
            >
              <Paragraph style={{ margin: 0, fontSize: 13.5, lineHeight: 1.9, color: '#3f3d56' }}>
                {summary.overview}
              </Paragraph>
            </div>
          )}

          {/* 关键点 */}
          {summary.keyPoints.length > 0 && (
            <div style={{ marginBottom: 14 }}>
              <SectionTitle icon={<BulbOutlined />} text="聊到的关键点" />
              <BulletList items={summary.keyPoints} />
            </div>
          )}

          {/* 情绪变化 */}
          {summary.emotionalTrend && (
            <div style={{ marginBottom: 14 }}>
              <SectionTitle icon={<LineChartOutlined />} text="情绪变化" />
              <Text style={{ fontSize: 13, lineHeight: 1.85, color: '#4a4a6a' }}>
                {summary.emotionalTrend}
              </Text>
            </div>
          )}

          {/* 有帮助的点 */}
          {summary.helpfulSuggestions.length > 0 && (
            <div style={{ marginBottom: 14 }}>
              <SectionTitle icon={<StarOutlined />} text="可能有帮助的地方" />
              <BulletList items={summary.helpfulSuggestions} />
            </div>
          )}

          {/* 可以接着聊 */}
          {summary.followUps.length > 0 && (
            <div style={{ marginBottom: 4 }}>
              <SectionTitle icon={<CommentOutlined />} text="可以接着聊" />
              <Space direction="vertical" size={6} style={{ width: '100%' }}>
                {summary.followUps.map((item, index) => (
                  <div
                    key={`${index}-${item}`}
                    style={{
                      background: '#fafafa',
                      border: '1px solid #f0f0f0',
                      borderRadius: 10,
                      padding: '8px 12px',
                      fontSize: 13,
                      color: '#4a4a6a',
                      lineHeight: 1.7,
                    }}
                  >
                    {item}
                  </div>
                ))}
              </Space>
            </div>
          )}
        </div>
      )}

      {/* 底部操作：有总结时才出现"重新生成"，并明确说明它会覆盖 */}
      {summary && !loading && !error && (
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginTop: 16,
            paddingTop: 12,
            borderTop: '1px solid #f0f0f0',
            gap: 8,
          }}
        >
          <Text type="secondary" style={{ fontSize: 11 }}>
            总结保存在本地，仅供你自己回顾
          </Text>
          <Tooltip title="重新读一遍这段对话，覆盖当前总结">
            <Button
              size="small"
              icon={<ReloadOutlined />}
              loading={generating}
              onClick={() => void handleGenerate(true)}
            >
              重新生成
            </Button>
          </Tooltip>
        </div>
      )}
    </Modal>
  );
}
