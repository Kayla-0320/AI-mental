import { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Input, Button, List, Avatar, Typography, Spin, Empty, Card, Space, Tag, Tooltip, Popconfirm, Modal, Alert } from 'antd';
import { SendOutlined, PlusOutlined, RobotOutlined, UserOutlined, ThunderboltOutlined, DeleteOutlined, PhoneOutlined, SmileOutlined, VideoCameraOutlined, SoundOutlined, AudioOutlined } from '@ant-design/icons';
import { consultationApi } from '../../services';
import { sendChatMessageStream, ChatStreamTransportError } from '../../services/chatStream';
import { useAnxiety } from '../../context/AnxietyContext';
import { PERCEPTION_API } from '../../config';
import CrisisResourceCard from '../../components/CrisisResourceCard';

const { Text } = Typography;

interface Message {
  id: string;
  role: string;
  content: string;
  createdAt: string;
  sentiment?: string;
  failed?: boolean;
}

// AI 聊天子组件
function AIChatTab() {
  const { reportToServer, toggleCamera, cameraEnabled, voiceMetrics, voiceInputActive, startVoiceInput, stopVoiceInput, voiceFinalSeq, voiceFinalText } = useAnxiety();
  const { conversationId } = useParams();
  const navigate = useNavigate();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sending, setSending] = useState(false);
  const [conversations, setConversations] = useState<any[]>([]);
  const [crisisModalOpen, setCrisisModalOpen] = useState(false);
  const [lastFailed, setLastFailed] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // 流式回复的临时气泡。
  // 刻意**不**把增量写进 messages：流式内容随时可能被安全审计整体改稿
  // （revise 事件），把它混进已定稿的历史列表会让「替换」语义变得难以表达。
  const [streaming, setStreaming] = useState(false);
  const [streamText, setStreamText] = useState('');
  // 组件卸载（切走路由）时中止上游调用，避免继续消耗 token
  const streamAbortRef = useRef<(() => void) | null>(null);

  // 实时情绪分析结果（最近一条用户消息的情绪）
  const [latestEmotion, setLatestEmotion] = useState<{
    text_emotion_probs: number[];
    confidence: number;
    evidence: string[];
  } | null>(null);
  const emotionLabels = ['快乐', '悲伤', '焦虑', '愤怒', '中性'];
  const emotionBarColors = ['#52c41a', '#722ed1', '#ff4d4f', '#fa8c16', '#1890ff'];

  // 键盘监测由浮动窗口的统一开关控制，不再自动启动

  useEffect(() => {
    loadConversations();
    if (conversationId) loadMessages(conversationId);
  }, [conversationId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    // 流式增量也要跟随滚动，否则文字增长时会跑出视口
  }, [messages, streamText]);

  // 组件卸载（切走路由）时中止流式请求：既省 token，也避免 setState 到已卸载组件
  useEffect(() => () => { streamAbortRef.current?.(); }, []);

  // 语音录入：整段录完（用户点"结束"）才定稿一次，这一次的文本整段追加到输入框。
  // 录音过程中"边说边上字"走的是 `asrPartialText`（独立区域展示），不写进输入框 ——
  // 所以不会出现"上屏一份、输入框又一份"的重复。
  //
  // 用递增序号而不是比较文本 —— 两次录入说出完全相同的话时文本不变，会比较不出来。
  //
  // 为什么不能只看 `voiceInputActive`：点"结束语音输入"时 `isRecording` 立刻变 false，
  // 而定稿还在路上（要先让服务端对整段音频跑一次离线识别）。
  // 只按"语音还开着"判断，就会把"点结束前刚说的那句"当过期结果丢掉，
  // 文字于是停在临时上屏区域（"正在录…"下面）进不了输入框。
  //
  // 现在用 `closedAtSeqRef` 记住结束那一刻的序号：比它更新的定稿是正常到达的最后一段，
  // 照样落入输入框；比它旧的是切换前遗留的历史结果，才丢弃。
  const lastVoiceSeqRef = useRef(0);
  const lastAppliedSeqRef = useRef(0);
  const closedAtSeqRef = useRef(0);

  // 记录"结束语音输入"发生在哪个序号
  useEffect(() => {
    if (voiceInputActive) return;
    closedAtSeqRef.current = Math.max(closedAtSeqRef.current, voiceMetrics.asrFinalSeq);
  }, [voiceInputActive, voiceMetrics.asrFinalSeq]);

  useEffect(() => {
    const seq = voiceMetrics.asrFinalSeq;
    const text = voiceMetrics.asrFinalText?.trim();
    if (voiceInputActive) {
      if (seq === lastVoiceSeqRef.current) return;
      lastVoiceSeqRef.current = seq;
    } else {
      // 结束之后最后一段定稿：序号必须比结束时刻更新
      if (seq <= closedAtSeqRef.current) {
        lastVoiceSeqRef.current = seq;
        return;
      }
    }
    if (!text || seq === lastAppliedSeqRef.current) return;
    lastAppliedSeqRef.current = seq;
    setInput(prev => (prev ? `${prev}${text}` : text));
  }, [voiceMetrics.asrFinalSeq, voiceMetrics.asrFinalText, voiceInputActive]);

  // 兜底：`stopVoiceInput()` 是等到最后一段定稿（或超时）才 resolve 的，
  // 若上面那一路因 state 批次没落到输入框，这里用回调序号再补一次。
  useEffect(() => {
    if (!voiceFinalText || voiceFinalSeq === 0) return;
    if (voiceFinalSeq <= closedAtSeqRef.current) return;
    if (voiceFinalSeq === lastAppliedSeqRef.current) return;
    // 若 metrics 的序号已经追上，说明上面那一路已经处理过，交给它，避免重复追加
    if (voiceMetrics.asrFinalSeq === voiceFinalSeq) return;
    lastAppliedSeqRef.current = voiceFinalSeq;
    setInput(prev => (prev ? `${prev}${voiceFinalText}` : voiceFinalText));
  }, [voiceFinalSeq, voiceFinalText, voiceMetrics.asrFinalSeq]);

  // 点"结束"时 `StreamingAsrSession.stop()` 会立刻发 finalize，服务端对**整段**音频
  // 跑一次离线大模型（12s 音频约 90ms，60s 约 0.4s）。这段时间按钮转圈，
  // 避免用户以为点了没反应而反复点击。
  const [stoppingVoice, setStoppingVoice] = useState(false);
  const handleToggleVoice = async () => {
    if (!voiceInputActive) {
      await startVoiceInput();
      return;
    }
    setStoppingVoice(true);
    try {
      // 等最后一段定稿落地；文本由上面的定稿回调落进输入框
      await stopVoiceInput();
    } finally {
      setStoppingVoice(false);
    }
  };

  const loadConversations = async () => {
    try {
      const res = await consultationApi.getConversations() as any;
      setConversations(res.data?.conversations || []);
    } catch {}
  };

  const loadMessages = async (convId: string) => {
    setLoading(true);
    setLastFailed(null);
    try {
      const res = await consultationApi.getMessages(convId) as any;
      setMessages(res.data?.messages || []);
    } catch {} finally { setLoading(false); }
  };

  const handleDeleteConversation = async (e: React.MouseEvent, convId: string) => {
    e.stopPropagation();
    try {
      await consultationApi.deleteConversation(convId);
      setConversations(prev => prev.filter(c => c.id !== convId));
      if (convId === conversationId) {
        navigate('/chat');
        setMessages([]);
      }
    } catch {}
  };

  const handleNewChat = async () => {
    try {
      const res = await consultationApi.createConversation() as any;
      navigate(`/chat/${res.data.id}`);
      setMessages([]);
      setLastFailed(null);
      loadConversations();
    } catch {}
  };

  /**
   * 进 AI 陪伴通话（整屏页，见 `pages/patient/VoiceCall.tsx`）。
   *
   * 已有会话就带着它进去；没有就把"建会话"这件事交给通话页自己做 ——
   * 建会话的接口只留一处实现，不在这里再抄一份。
   */
  const handleStartCall = () => {
    navigate(conversationId ? `/call/${conversationId}` : '/call');
  };

  // 调用感知 API 分析情绪
  const analyzeEmotion = useCallback(async (text: string) => {
    if (!text.trim()) return;
    try {
      const res = await fetch(`${PERCEPTION_API}/api/v1/perception/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const result = await res.json();
      setLatestEmotion(result);
    } catch {
      // 感知服务不可用时静默降级，不影响对话
    }
  }, []);

  const makeAssistantMessage = (content: string, failed = false): Message => ({
    id: failed ? 'error' : 'temp',
    role: 'assistant',
    content,
    createdAt: new Date().toISOString(),
    failed,
  });

  /**
   * 非流式发送（回退路径）。
   *
   * 只有流式链路在**传输层**就失败、且服务端尚未落库用户消息时才会走到这里。
   * 复用既有接口，因此 token 刷新、失败重试等行为与改造前完全一致。
   */
  const sendContentFallback = async (content: string) => {
    if (!conversationId) return;
    setSending(true);
    setStreaming(false);
    setStreamText('');
    try {
      const res = await consultationApi.sendMessage(conversationId, content) as any;
      reportToServer('AI咨询对话');

      // 后端显式标记失败：渲染可重试的失败态，不把它当成一条 AI 回复。
      // 后端不再为失败写入 assistant 消息，因此 aiMessage 为 null。
      if (res.data?.failed) {
        setMessages(prev => [
          ...prev.filter(m => m.id !== 'temp' && m.id !== 'error'),
          makeAssistantMessage(res.data.errorMessage || '抱歉，我暂时无法回应，请稍后再试。', true),
        ]);
        setLastFailed(content);
        return;
      }

      if (res.data) {
        setLastFailed(null);
        setMessages(prev => {
          const filtered = prev.filter(m => m.id !== 'temp' && m.id !== 'error');
          return [...filtered, res.data.userMessage, res.data.aiMessage];
        });

        // 危机信号：由后端 Python 算法层统一判断（62+ 关键词库）
        const isCrisis = res.data.isCrisis === true || res.data.riskLevel === 'crisis';
        if (isCrisis) {
          setCrisisModalOpen(true);
        }
      }
    } catch (e: any) {
      // 网络/HTTP 层失败：同样走可重试的失败态
      setMessages(prev => [
        ...prev.filter(m => m.id !== 'temp' && m.id !== 'error'),
        makeAssistantMessage(e?.message || '网络异常，请稍后重试', true),
      ]);
      setLastFailed(content);
    } finally { setSending(false); }
  };

  const sendContent = async (content: string) => {
    if (!conversationId) return;
    setSending(true);
    setStreaming(true);
    setStreamText('');
    setLastFailed(null);

    let donePayload: any = null;
    let errorPayload: any = null;
    let transportError: unknown = null;

    // 流式链路：user → meta? → delta* → [revise] → done
    const handle = sendChatMessageStream(conversationId, content, {
      onUser: (userMessage) => {
        // 用落库后的真实消息替换本地乐观占位气泡
        setMessages(prev => [
          ...prev.filter(m => m.id !== 'temp' && m.id !== 'error'),
          userMessage as unknown as Message,
        ]);
      },
      onDelta: (text) => setStreamText(prev => prev + text),
      // 安全审计改稿 / 走了降级路径 —— 必须**整体替换**，不能追加
      onRevise: (text) => setStreamText(text),
      onDone: (payload) => { donePayload = payload; },
      onError: (payload) => { errorPayload = payload; },
    });
    streamAbortRef.current = handle.abort;

    try {
      await handle.completed;
    } catch (e) {
      transportError = e;
    } finally {
      streamAbortRef.current = null;
      setStreaming(false);
      setSending(false);
    }

    // ---- 成功 ----
    if (donePayload) {
      setStreamText('');
      if (donePayload.aiMessage) {
        setMessages(prev => [...prev, donePayload.aiMessage as Message]);
      }
      reportToServer('AI咨询对话');
      // 危机信号：由后端 Python 算法层统一判断，与非流式路径同一判据
      if (donePayload.isCrisis === true || donePayload.riskLevel === 'crisis') {
        setCrisisModalOpen(true);
      }
      return;
    }

    // ---- 业务失败（服务端明确告知）----
    if (errorPayload?.failed) {
      setStreamText('');
      setMessages(prev => [
        ...prev.filter(m => m.id !== 'temp' && m.id !== 'error'),
        makeAssistantMessage(errorPayload.errorMessage || '抱歉，我暂时无法回应，请稍后再试。', true),
      ]);
      setLastFailed(content);
      return;
    }

    // ---- 传输层失败：仅在服务端尚未落库用户消息时才安全地重发 ----
    // 若 onUser 已经触发过，说明这条消息已经写进数据库；再走一次非流式接口
    // 会把它重复写一遍（用户侧表现为同一条消息出现两次）。
    const persisted =
      transportError instanceof ChatStreamTransportError && transportError.userPersisted;
    setStreamText('');
    if (persisted) {
      setMessages(prev => [
        ...prev.filter(m => m.id !== 'temp' && m.id !== 'error'),
        makeAssistantMessage('回复生成中断，请点击重试。', true),
      ]);
      setLastFailed(content);
      return;
    }

    await sendContentFallback(content);
  };

  const handleSend = async () => {
    if (!input.trim() || !conversationId) return;
    const content = input.trim();
    setInput('');

    // 实时情绪分析（无感调用，不阻塞对话）
    analyzeEmotion(content);

    const userMsg: Message = { id: 'temp', role: 'user', content, createdAt: new Date().toISOString() };
    setMessages(prev => [...prev, userMsg]);
    await sendContent(content);
  };

  const handleRetry = async () => {
    if (!lastFailed) return;
    await sendContent(lastFailed);
  };

  const emotionColors: Record<string, string> = {
    '焦虑': 'orange', '悲伤': 'blue', '愤怒': 'red', '恐惧': 'purple',
    '喜悦': 'green', '平静': 'cyan', '困惑': 'gold', '孤独': 'geekblue',
    '压力': 'volcano', '无助': 'magenta',
  };

  const renderEmotionTag = (msg: Message) => {
    if (msg.role !== 'user' || !msg.sentiment) return null;
    try {
      const emotion = JSON.parse(msg.sentiment);
      if (!emotion.primaryEmotion) return null;
      return (
        <Tooltip title={`强度: ${emotion.intensity}% | 语气: ${emotion.tone || '未知'}`}>
          <Tag color={emotionColors[emotion.primaryEmotion] || 'default'} style={{ marginTop: 6, fontSize: 11, cursor: 'pointer' }}>
            <ThunderboltOutlined /> {emotion.primaryEmotion} {emotion.intensity > 0 ? `${emotion.intensity}%` : ''}
          </Tag>
        </Tooltip>
      );
    } catch { return null; }
  };

  // AI 消息上的非诊断性元信息。
  //
  // 刻意**只**显示「对话模式」与「已参考的相似经历数」，**不显示认知扭曲类别名**：
  // 对青少年说"你在读心术/乱贴标签"属于贴标签式诊断语言，触碰 AGENTS.md 红线。
  // 完整的 { dialogue_mode, scene_retrieval, cognitive_distortion } 仍写在
  // message.sentiment 里，供咨询师端与审计链路使用。
  const dialogueModeLabels: Record<string, { text: string; color: string }> = {
    SOCRATIC: { text: '苏格拉底引导', color: 'purple' },
    EMPATHY: { text: '共情陪伴', color: 'blue' },
  };

  const renderAiMeta = (msg: Message) => {
    if (msg.role !== 'assistant' || !msg.sentiment) return null;
    try {
      const meta = JSON.parse(msg.sentiment);
      const mode = dialogueModeLabels[meta.dialogue_mode];
      const sceneCount = meta.scene_retrieval?.scenes ?? 0;
      if (!mode && !sceneCount) return null;
      return (
        <>
          {mode && (
            <Tooltip title="当前对话策略">
              <Tag color={mode.color} style={{ marginTop: 6, fontSize: 11, cursor: 'pointer' }}>
                {mode.text}
              </Tag>
            </Tooltip>
          )}
          {sceneCount > 0 && (
            <Tooltip title="已从青少年真实语料中匹配到相似处境，用来让回应更贴合你的具体情况">
              <Tag color="geekblue" style={{ marginTop: 6, fontSize: 11, cursor: 'pointer' }}>
                参考了 {sceneCount} 个相似经历
              </Tag>
            </Tooltip>
          )}
        </>
      );
    } catch { return null; }
  };

  return (
    <div style={{ display: 'flex', gap: 16, height: 'calc(100vh - 200px)' }}>
      {/* 流式回复的闪烁光标。放在组件内而不是全局 CSS：只有这里用得到，
          且避免为了一个 keyframes 去改公共样式文件。 */}
      <style>{`@keyframes blink-caret { 0%, 100% { opacity: 1 } 50% { opacity: 0 } }`}</style>
      <CrisisModal
        open={crisisModalOpen}
        onClose={() => setCrisisModalOpen(false)}
        onGoChat={() => setCrisisModalOpen(false)}
      />
      <Card style={{ width: 280, height: '100%', borderRadius: 12 }}
        title={<Text strong>对话历史</Text>}
        extra={
          <Space size={2}>
            <Tooltip title="和 AI 打电话（语音陪伴，整屏）">
              <Button type="text" icon={<PhoneOutlined />} onClick={handleStartCall} />
            </Tooltip>
            <Button type="text" icon={<PlusOutlined />} onClick={handleNewChat} />
          </Space>
        }
        bodyStyle={{ padding: '8px 0', height: 'calc(100% - 57px)', overflowY: 'auto' }}>
        <List
          dataSource={conversations}
          locale={{ emptyText: <Empty description="还没有对话，点击上方 + 开始" image={Empty.PRESENTED_IMAGE_SIMPLE} /> }}
          renderItem={(item: any) => (
            <List.Item
              onClick={() => navigate(`/chat/${item.id}`)}
              style={{ cursor: 'pointer', padding: '10px 8px', borderRadius: 8, background: item.id === conversationId ? '#f5f3ff' : 'transparent' }}
              actions={[
                <Popconfirm key="del" title="确定删除？" onConfirm={(e) => handleDeleteConversation(e as any, item.id)} onCancel={(e) => e?.stopPropagation()} okText="删除" cancelText="取消" okButtonProps={{ danger: true }}>
                  <DeleteOutlined onClick={(e) => e.stopPropagation()} style={{ color: '#ccc', fontSize: 14, cursor: 'pointer' }}
                    onMouseEnter={(e) => (e.currentTarget.style.color = '#ff4d4f')} onMouseLeave={(e) => (e.currentTarget.style.color = '#ccc')} />
                </Popconfirm>,
              ]}
            >
              <Text ellipsis style={{ fontSize: 13 }}>{item.title || '新对话'}</Text>
            </List.Item>
          )}
        />
      </Card>

      <Card style={{ flex: 1, height: '100%', borderRadius: 12, display: 'flex', flexDirection: 'column' }}
        bodyStyle={{ flex: 1, display: 'flex', flexDirection: 'column', padding: 0, overflow: 'hidden' }}>
        {!conversationId ? (
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'flex-start', padding: 40, overflow: 'auto' }}>
            <div style={{ fontSize: 64, marginBottom: 16 }}>💬</div>
            <Empty description={
              <span style={{ color: '#8a7a9a', fontSize: 15 }}>
                创建新对话，AI 会陪你聊聊
              </span>
            } image={null} />
            <div style={{ marginTop: 24, maxWidth: 400, textAlign: 'center' }}>
              <Text type="secondary" style={{ fontSize: 13, display: 'block', marginBottom: 12 }}>
                不知道说什么？试试这些话题：
              </Text>
              <Space wrap size={[8, 8]} style={{ justifyContent: 'center' }}>
                {['学业压力好大', '和朋友闹别扭了', '最近睡不好', '就是心情不好', '对未来很迷茫', '想找人说说'].map(topic => (
                  <Tag
                    key={topic}
                    style={{
                      cursor: 'pointer', padding: '6px 14px', borderRadius: 20,
                      background: '#f5f3ff', color: '#6366f1', border: 'none', fontSize: 13,
                    }}
                    onClick={handleNewChat}
                  >
                    {topic}
                  </Tag>
                ))}
              </Space>
            </div>
            <Button type="primary" size="large" icon={<PlusOutlined />} onClick={handleNewChat}
              style={{ marginTop: 24, borderRadius: 24, padding: '0 32px' }}>
              开始新对话
            </Button>
          </div>
        ) : (
          <>
            <div style={{ flex: 1, overflow: 'auto', padding: 24 }}>
              {loading ? <div style={{ textAlign: 'center' }}><Spin /></div> : (
                messages.map((msg) => (
                  <div key={msg.id} style={{ display: 'flex', marginBottom: 16, flexDirection: msg.role === 'user' ? 'row-reverse' : 'row' }}>
                    <Avatar icon={msg.role === 'user' ? <UserOutlined /> : <RobotOutlined />}
                      style={{ backgroundColor: msg.role === 'user' ? '#6366f1' : (msg.failed ? '#fff1f0' : '#f0f0f0'), color: msg.role === 'user' ? '#fff' : (msg.failed ? '#ff4d4f' : '#6366f1'), flexShrink: 0 }} />
                    {msg.failed ? (
                      // 失败态：明确区别于正常 AI 回复，并提供重试
                      <div style={{ maxWidth: '70%', margin: '0 12px', padding: '12px 16px', borderRadius: 12,
                        background: '#fff1f0', border: '1px solid #ffccc7', color: '#a8071a', lineHeight: 1.6 }}>
                        <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
                        {lastFailed && (
                          <Button size="small" danger style={{ marginTop: 10, borderRadius: 12 }}
                            loading={sending} onClick={handleRetry}>
                            重试
                          </Button>
                        )}
                      </div>
                    ) : (
                      <div style={{ maxWidth: '70%', margin: '0 12px', padding: '12px 16px', borderRadius: 12,
                        background: msg.role === 'user' ? '#6366f1' : '#f5f5f5', color: msg.role === 'user' ? '#fff' : '#333', whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
                        {msg.content}
                      </div>
                    )}
                    {renderEmotionTag(msg)}
                    {renderAiMeta(msg)}
                  </div>
                ))
              )}
              {/* streaming：流式进行中；sending：回退到非流式路径时的等待。
                  两者都要显示气泡 —— 前者会长出文字，后者只有 spinner。 */}
              {(streaming || sending) && (
                <div style={{ display: 'flex', marginBottom: 16 }}>
                  <Avatar icon={<RobotOutlined />} style={{ backgroundColor: '#f0f0f0', color: '#6366f1' }} />
                  <div style={{ maxWidth: '70%', margin: '0 12px', padding: '12px 16px', borderRadius: 12,
                    background: '#f5f5f5', color: '#333', whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
                    {streamText ? (
                      <>
                        {streamText}
                        {/* 光标：明确告诉用户内容还在增长 */}
                        <span style={{ display: 'inline-block', width: 7, marginLeft: 2,
                          borderRight: '2px solid #6366f1', animation: 'blink-caret 1s step-end infinite' }} />
                      </>
                    ) : (
                      <><Spin size="small" /> <Text type="secondary" style={{ marginLeft: 8 }}>正在思考中...</Text></>
                    )}
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
            <div style={{ padding: '16px 24px', borderTop: '1px solid #f0f0f0' }}>
              {/* 实时情绪感知条 */}
              {latestEmotion && (
                <div style={{ padding: '8px 0 12px', display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
                  <Space size={4}>
                    <SmileOutlined style={{ color: '#6366f1', fontSize: 14 }} />
                    <Text strong style={{ fontSize: 12 }}>AI 实时感知</Text>
                  </Space>
                  {latestEmotion.text_emotion_probs.map((p: number, i: number) => (
                    <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                      <span style={{ fontSize: 11, color: emotionBarColors[i], minWidth: 28 }}>{emotionLabels[i]}</span>
                      <div style={{ width: 40, height: 4, background: '#f0f0f0', borderRadius: 2 }}>
                        <div style={{ height: '100%', width: `${Math.round(p * 100)}%`, background: emotionBarColors[i], borderRadius: 2, transition: 'width 0.5s' }} />
                      </div>
                      <span style={{ fontSize: 10, color: '#999', minWidth: 28 }}>{Math.round(p * 100)}%</span>
                    </div>
                  ))}
                  <Tag color="purple" style={{ fontSize: 10, margin: 0 }}>置信度 {Math.round(latestEmotion.confidence * 100)}%</Tag>
                </div>
              )}
              {/* 语音输入：边说边上字。识别在本机完成，音频不出设备 */}
              {voiceInputActive && (
                <div style={{
                  display: 'flex', alignItems: 'center', gap: 8,
                  padding: '8px 12px', marginBottom: 8,
                  background: 'linear-gradient(135deg, #f0f9ff, #f5f3ff)',
                  border: '1px solid #d6bcfa40', borderRadius: 10,
                }}>
                  <SoundOutlined style={{ color: '#6366f1' }} />
                  <span style={{ fontSize: 12, color: '#6366f1', whiteSpace: 'nowrap' }}>
                    {stoppingVoice ? '正在识别整段语音…' : '正在录…（中途停顿不会打断）'}
                  </span>
                  <span style={{ fontSize: 13, color: '#4a4a6a', flex: 1, minHeight: 18 }}>
                    {voiceMetrics.asrPartialText || <span style={{ color: '#bbb' }}>请开始说话，说完了点右边的按钮结束</span>}
                  </span>
                  {voiceMetrics.asrLatencyMs > 0 && (
                    <Tag color="purple" style={{ fontSize: 10, margin: 0 }}>
                      本地识别 {Math.round(voiceMetrics.asrLatencyMs)}ms
                    </Tag>
                  )}
                  {voiceMetrics.asrRefined && (
                    <Tooltip title="定稿已用离线大模型复识别纠错，并补了标点">
                      <Tag color="green" style={{ fontSize: 10, margin: 0 }}>已纠错</Tag>
                    </Tooltip>
                  )}
                </div>
              )}
              {/* 已点结束、但最后一段还在定稿中：把文字即将落到输入框这件事说清楚 */}
              {!voiceInputActive && stoppingVoice && (
                <div style={{
                  display: 'flex', alignItems: 'center', gap: 8,
                  padding: '8px 12px', marginBottom: 8,
                  background: 'linear-gradient(135deg, #f0f9ff, #f5f3ff)',
                  border: '1px solid #d6bcfa40', borderRadius: 10,
                }}>
                  <Spin size="small" />
                  <span style={{ fontSize: 12, color: '#6366f1' }}>正在识别整段语音，马上填入输入框…</span>
                </div>
              )}
              {voiceMetrics.asrError && (
                <Alert
                  type="warning"
                  showIcon
                  style={{ marginBottom: 8, fontSize: 12 }}
                  message={`语音识别不可用：${voiceMetrics.asrError}`}
                />
              )}
              <Space.Compact style={{ width: '100%' }}>
                <Tooltip title={cameraEnabled ? '关闭摄像头/麦克风' : '开启摄像头/麦克风（面部+语音分析）'}>
                  <Button
                    icon={cameraEnabled ? <VideoCameraOutlined /> : <VideoCameraOutlined />}
                    onClick={toggleCamera}
                    type={cameraEnabled ? 'primary' : 'default'}
                    size="large"
                    style={{ borderRadius: '8px 0 0 8px', background: cameraEnabled ? '#52c41a' : undefined }}
                  />
                </Tooltip>
                <Tooltip title={voiceInputActive ? '结束语音输入（整段一起识别）' : '语音输入（整段录入，说话中途停顿不会被打断；音频不出设备）'}>
                  <Button
                    icon={<AudioOutlined />}
                    onClick={() => void handleToggleVoice()}
                    loading={stoppingVoice}
                    type={voiceInputActive ? 'primary' : 'default'}
                    danger={voiceInputActive}
                    size="large"
                  />
                </Tooltip>
                <Input value={input} onChange={(e) => setInput(e.target.value)} onPressEnter={handleSend}
                  placeholder="说说你的想法..." disabled={sending} size="large" style={{ borderRadius: 0 }} />
                <Button type="primary" icon={<SendOutlined />} onClick={handleSend} loading={sending} size="large" style={{ borderRadius: '0 8px 8px 0' }}>发送</Button>
              </Space.Compact>
            </div>
          </>
        )}
      </Card>
    </div>
  );
}

// 主组件
export default function Chat() {
  return <AIChatTab />;
}

// 危机干预弹窗（内容已抽到 `components/CrisisResourceCard.tsx`，与通话页共用）
//
// ⚠️ 抽出的原因：通话页与文字页在危机时必须给出**同一份**文案与号码。
// 文案本身（含两个热线号码）**一个字都没有改** —— 见该组件的文件头说明。
function CrisisModal({ open, onClose, onGoChat }: { open: boolean; onClose: () => void; onGoChat: () => void }) {
  return (
    <Modal open={open} onCancel={onClose} footer={null} width={460} centered>
      <CrisisResourceCard onPrimary={onGoChat} onDismiss={onClose} />
    </Modal>
  );
}
