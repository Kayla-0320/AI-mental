import { useState, useEffect, useRef, useCallback } from 'react';
import { Navigate, useLocation, useParams, useNavigate, useSearchParams } from 'react-router-dom';
import { Input, Button, List, Avatar, Typography, Spin, Empty, Card, Space, Tag, Tooltip, Popconfirm, Modal, Drawer, Alert, Checkbox, message } from 'antd';
import { SendOutlined, PlusOutlined, RobotOutlined, UserOutlined, DeleteOutlined, PhoneOutlined, SmileOutlined, VideoCameraOutlined, SoundOutlined, AudioOutlined, FileTextOutlined } from '@ant-design/icons';
import { consultationApi } from '../../services';
import { sendChatMessageStream, ChatStreamTransportError } from '../../services/chatStream';
import { useAnxiety } from '../../context/AnxietyContext';
import { PERCEPTION_API } from '../../config';
import CrisisResourceCard from '../../components/CrisisResourceCard';
import RelationshipStageIndicator from '../../components/RelationshipStageIndicator';
import IncongruenceAlert from '../../components/IncongruenceAlert';
import ConversationSummaryModal from '../../components/ConversationSummaryModal';
import SupportUnderstandingCard, { supportStateFromSentiment } from '../../components/SupportUnderstandingCard';
import InnovationStatus from '../../components/InnovationStatus';
import ChatCompanionPanel, { type CompanionTab } from '../../components/ChatCompanionPanel';
import FaceEmotionBlock, { FaceEmotionChips } from '../../components/FaceEmotionBlock';
import { innovationFromSentiment } from '../../services/innovationPresentation';
import { VoiceDraft } from '../../services/voiceDraft';

const { Text } = Typography;

/**
 * 空状态里的六个「话题」入口 —— 每个话题配一句能直接发出去的开口话。
 *
 * 为什么不是「点一下就建一个空对话」：那只是把用户扔进空白的输入框，
 * 用户还得自己想第一句说什么，等于什么忙都没帮上。这里把话题变成**一句真实
 * 的第一人称开场**，点下去就等于用户自己说了这句话，AI 立刻按这个情境回应。
 *
 * 措辞是学生自己的口吻（而不是给模型的元指令）：整句会原样落库成一条用户消息、
 * 也会进对话标题与情绪分析链路，所以它必须读起来就是"这个学生说的话"，
 * 不能是 `请围绕学业压力和我聊天` 这类指令 —— 那既不像人话，也会污染标题与分析。
 *
 * ⚠️ 刻意避开诊断性表述与自伤/危机措辞（见 AGENTS.md 安全红线）：
 * 话题只做"顺情绪的开场"，风险识别仍然交给算法层从真实对话里判断。
 */
const TOPIC_PROMPTS: { topic: string; prompt: string }[] = [
  { topic: '学业压力好大', prompt: '最近学业压力好大，作业和考试一堆，我有点喘不过气。' },
  { topic: '和朋友闹别扭了', prompt: '我和朋友闹别扭了，心里挺别扭的，不知道该怎么面对他。' },
  { topic: '最近睡不好', prompt: '最近总是睡不好，躺下很久都睡不着，白天也没什么精神。' },
  { topic: '就是心情不好', prompt: '说不上具体为什么，就是最近心情一直不太好，做什么都提不起劲。' },
  { topic: '对未来很迷茫', prompt: '一想到未来我就很迷茫，不知道自己以后该干什么。' },
  { topic: '想找人说说', prompt: '我有些话憋在心里很久了，想找个人说说。' },
];

/**
 * 对话历史每页条数（与服务端 getConversations 的默认值保持一致）。
 *
 * 底部"加载更多"按这个步长追加；写成常量而不是散落的 20，是为了以后调大时只改一处。
 */
const CONVERSATION_PAGE_SIZE = 20;

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
  const { reportToServer, toggleCamera, cameraEnabled, getFacialTurnInput, getVisualFrameCount, mediaNotice, voiceMetrics, facialMetrics, voiceInputActive, startVoiceInput, stopVoiceInput, subscribeVoiceInput, commitVoiceInput } = useAnxiety();
  const { conversationId } = useParams();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [wideCompanion, setWideCompanion] = useState(window.innerWidth >= 1350);
  const [companionOpen, setCompanionOpen] = useState(window.innerWidth >= 1350 || searchParams.has('panel'));
  const requestedPanel = searchParams.get('panel');
  const activeCompanionTab: CompanionTab = requestedPanel === 'quality' || requestedPanel === 'tracking' ? requestedPanel : 'fusion';
  useEffect(() => {
    const resize = (): void => setWideCompanion(window.innerWidth >= 1350);
    window.addEventListener('resize', resize);
    return () => window.removeEventListener('resize', resize);
  }, []);
  useEffect(() => { if (requestedPanel) setCompanionOpen(true); }, [requestedPanel]);
  const openCompanion = (tab: CompanionTab): void => {
    setCompanionOpen(true);
    const params = new URLSearchParams(searchParams); params.set('panel', tab);
    setSearchParams(params, { replace: true });
  };
  const closeCompanion = (): void => {
    setCompanionOpen(false);
    const params = new URLSearchParams(searchParams); params.delete('panel');
    setSearchParams(params, { replace: true });
  };
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInputState] = useState('');
  const voiceDraftRef = useRef(new VoiceDraft());
  const setInput = useCallback((value: string | ((previous: string) => string)): void => {
    const draft = voiceDraftRef.current;
    draft.edit(typeof value === 'function' ? value(draft.text) : value);
    setInputState(draft.text);
  }, []);
  const submitLockRef = useRef(false);
  const [voiceSubmitting, setVoiceSubmitting] = useState(false);
  const [supportFeedbackBusy, setSupportFeedbackBusy] = useState(false);
  const [mobileHistoryOpen, setMobileHistoryOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  /**
   * 已经**取回历史**的那个会话 id（取回后为空也算取回）。
   *
   * 空状态的判据是"当前会话没有消息"，而 `messages.length === 0` 这个条件在切会话的
   * 第一帧同样成立（新会话的消息还在路上），两者必须区分开，否则会先闪一下空状态。
   */
  const [loadedConvId, setLoadedConvId] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const [creatingConversation, setCreatingConversation] = useState(false);
  const [conversations, setConversations] = useState<any[]>([]);
  /**
   * 对话历史的分页状态。
   *
   * 后端一页固定 20 条，列表底部按需"加载更多"。之前是"只拉第一页、永远只有 20 条"：
   * 删掉这一页之后任何一次刷新都会把下一页顶上来，而会话标题又大量重复（同一句开场白
   * 被复制了很多条），看起来就像"删掉的记录又回来了"。分页之后，删掉就是删掉，
   * 想看得更早的对话必须自己点"加载更多"。
   */
  const [convTotal, setConvTotal] = useState(0);
  const [convLoadingMore, setConvLoadingMore] = useState(false);
  // 对话历史的多选管理态。刻意不复用 conversations 里的某个字段：
  // 勾选是**界面态**，不进网络，也不该影响会话数据本身。
  const [selectMode, setSelectMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [batchDeleting, setBatchDeleting] = useState(false);
  const [crisisModalOpen, setCrisisModalOpen] = useState(false);
  /**
   * 总结弹窗。
   *
   * 刻意与 `conversationId` 解耦：用户完全可以在**没打开**某个对话的情况下，
   * 直接从历史列表点"查看总结"看它的纪要。所以这里存的是"要看的是哪个会话"，
   * 而不是复用当前打开的会话 —— 复用的结果是"必须先点开对话才能看总结"，
   * 而用户想看纪要恰恰常常是为了**不**重读整段对话。
   */
  const [summaryTarget, setSummaryTarget] = useState<{ id: string; title?: string } | null>(null);
  const [lastFailed, setLastFailed] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  // 当前打开会话的实时值。Modal.confirm 的回调是在**点击那一刻**的闭包里跑的，
  // 而弹窗可能开着好一会儿；这期间用户切到别的会话后，闭包里的 conversationId
  // 已经过期，会漏掉"退回空态"这一步（中间栏于是继续挂着一个已删会话的消息）。
  const activeConvRef = useRef<string | undefined>(conversationId);
  activeConvRef.current = conversationId;

  /**
   * 点「话题」后待发送的开口话。
   *
   * 为什么需要这个中转：建会话拿到 id 之后，`conversationId` 要等 `navigate` 触发
   * 重新渲染才会更新，而 `sendContent` 是**按闭包里的 conversationId** 发请求的 ——
   * 在 `handleTopicClick` 里紧接着发，闭包拿到的还是 undefined（旧会话），消息会丢。
   * 所以这里先把"要发什么、发到哪个会话"存进 ref，等路由参数落到同一个 id 时再发一次。
   *
   * 顺带一个好处：等待期间用户如果切到别的会话，`conversationId` 对不上，
   * 这条待发消息就不会被塞进那个不相干的会话里。
   */
  const pendingPromptRef = useRef<{ convId: string; content: string } | null>(null);

  /**
   * 待发开场白的批次号。
   *
   * 只是 ref 的话，在"复用一个已经空着的会话"这条路径上会漏发：会话 id 没变、
   * `loading` 与 `messages.length` 也都没变，订阅它们的 effect 根本不会再跑一遍。
   * 递增一个 state 让"又排了一句待发"这件事本身成为一次可观察的变化。
   */
  const [pendingSeq, setPendingSeq] = useState(0);

  /**
   * "正在为哪句开场白建会话"。
   *
   * 只在创建请求尚未完成时拦住重复提交；成功和失败都必须解除，
   * 否则下一次开始新对话时会静默丢掉用户的开场白。
   */
  const creatingForRef = useRef<string | null>(null);

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
  // TAMER 框架输出（从后端响应获取）
  const [tamerResult, setTamerResult] = useState<{
    rqi: number;
    mis: number;
    trajectory: string;
    relationship_depth: string;
    incongruence_type?: string;
    /** 是否建议转介真人（算法层：连续高风险轮次达到阈值） */
    should_refer?: boolean;
    consecutive_high_risk?: number;
  } | null>(null);

  /**
   * 转介提示是否被用户关掉。
   *
   * 不用 localStorage 持久化：这是「这一轮对话里的提示」，用户关掉就表示
   * 现在不需要；下次会话再次满足条件时应该重新出现，而不是被永久静音。
   */
  const [referralDismissed, setReferralDismissed] = useState(false);
  const emotionLabels = ['快乐', '悲伤', '焦虑', '愤怒', '中性'];
  const emotionBarColors = ['#52c41a', '#722ed1', '#ff4d4f', '#fa8c16', '#1890ff'];

  // 键盘监测由浮动窗口的统一开关控制，不再自动启动

  // 对话列表只在**挂载时**拉一次；建会话、删会话由对应动作自己维护列表。
  //
  // ⚠️ 之前这条 effect 挂在 [conversationId] 上：每点开一段对话就重拉第一页。
  // 结果就是"删掉一批 → 随手点开一段 → 下一页同名会话顶上来"，用户看到的是
  // "删掉的又全回来了"。列表是导航用的辅助信息，不该跟着选中项来回重建。
  useEffect(() => {
    void loadConversations();
  }, []);

  useEffect(() => {
    if (conversationId) void loadMessages(conversationId);
  }, [conversationId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    // 流式增量也要跟随滚动，否则文字增长时会跑出视口
  }, [messages, streamText]);

  // 组件卸载（切走路由）时中止流式请求：既省 token，也避免 setState 到已卸载组件
  useEffect(() => () => { streamAbortRef.current?.(); }, []);

  // 每次识别更新直接修订草稿；定稿也走同一条通道，避免重复追加。
  useEffect(() => subscribeVoiceInput(event => {
    voiceDraftRef.current.receive(event);
    setInputState(voiceDraftRef.current.text);
  }), [subscribeVoiceInput]);

  const [stoppingVoice, setStoppingVoice] = useState(false);
  const handleToggleVoice = async () => {
    if (!voiceInputActive) { await startVoiceInput(); return; }
    setStoppingVoice(true);
    try { await stopVoiceInput(); }
    finally { setStoppingVoice(false); }
  };

  /**
   * 拉取/刷新对话列表第一页（20 条）。
   *
   * 成功时用服务端列表整体替换本地状态；失败保持现状 —— 历史列表是辅助导航，
   * 不该因为它把整页变成错误页。
   */
  const loadConversations = async () => {
    try {
      const res = await consultationApi.getConversations(1, CONVERSATION_PAGE_SIZE) as any;
      const list = res.data?.conversations || [];
      setConversations(list);
      setConvTotal(Number(res.data?.total) || list.length);
    } catch {
      // 静默：下一次显式刷新会再试
    }
  };

  /**
   * 加载更多：重取"前 N+20 条"，只把新增的追加到尾部。
   *
   * 为什么不用页码往后翻：删除会让后端的 skip 偏移，删掉 3 条后再请求第 2 页，
   * 会永久跳过 3 条没看过的会话。按"已加载条数"重取，天然对齐，重复项按 id 去重。
   */
  const loadMoreConversations = async () => {
    if (convLoadingMore || conversations.length >= convTotal) return;
    setConvLoadingMore(true);
    try {
      const res = await consultationApi.getConversations(1, conversations.length + CONVERSATION_PAGE_SIZE) as any;
      const list: any[] = res.data?.conversations || [];
      setConvTotal(Number(res.data?.total) || convTotal);
      setConversations(prev => {
        const seen = new Set(prev.map((c: any) => c.id));
        return [...prev, ...list.filter((c: any) => !seen.has(c.id))];
      });
    } catch {
      // 失败保持现状，用户再点一次即可
    } finally {
      setConvLoadingMore(false);
    }
  };

  // 列表变化（删除 / 加载更多 / 刷新）后丢弃已经不存在的勾选，否则"已选 N 项"会
  // 包含幽灵会话，点删除时后端会因为对不上数而整体拒绝。
  useEffect(() => {
    const aliveIds = new Set(conversations.map((c: any) => c.id));
    setSelectedIds(prev => prev.filter(id => aliveIds.has(id)));
  }, [conversations]);

  const loadMessages = async (convId: string) => {
    // 先声明"这个会话的历史还没落地"：否则会话 A（有消息）切到会话 B（空会话）的那一帧，
    // messages 里还是 A 的气泡，界面会误判成"B 有内容"而不显示空状态。
    setLoadedConvId(null);
    setLoading(true);
    setLastFailed(null);
    let msgs: Message[] = [];
    try {
      const res = await consultationApi.getMessages(convId) as any;
      msgs = res.data?.messages || [];
    } catch (e: any) {
      // 会话不存在 / 已被删除（404）：退回"还没开始聊"的空态。
      // 服务端现在会拒绝往已删会话里写消息，把用户留在一个打不出字的输入框上
      // 只会让他以为"AI 又坏了"。
      if (e?.status === 404) {
        if (activeConvRef.current === convId) navigate('/chat', { replace: true });
        return;
      }
      // 其他错误（网络抖动等）仍按空会话处理：给出空状态，而不是卡在 loading
    } finally {
      setLoading(false);
    }
    // 请求飞在路上时用户可能已经"开始新对话"（退回 /chat）或切到别的会话：
    // 这时这条响应已经过期，落进 state 会把新页面的消息带回来。按路由实时值丢弃。
    if (activeConvRef.current !== convId) return;
    setMessages(msgs);
    // 标记"这个会话的历史确实取回来了"。空状态靠它区分「这个会话是空的」与
    // 「还没加载完」—— 没有这个区分，切会话时会先闪一下空状态再跳出消息。
    setLoadedConvId(convId);
  };

  /**
   * 把「话题」的开口话真正发出去。
   *
   * 触发时机卡在三个条件同时成立：路由已经切到那个新会话、历史消息已经加载完
   * （`loading` 为 false，否则可能在加载完成前先插一条乐观气泡导致顺序错乱）、
   * 且这个会话确实是空的（消息数为 0）。
   *
   * 「会话必须是空的」这条不是形式检查：它同时挡住了两件事 ——
   * ① React 18 开发模式下的 effect 双跑会把同一句话发两遍；
   * ② 用户在等待期间手动打字发送、把会话"占"了的情况。
   * 因此发之前必须把 ref 清空，做成一次性的。
   */
  useEffect(() => {
    const pending = pendingPromptRef.current;
    if (!pending) return;
    if (!conversationId || conversationId !== pending.convId) return;
    // 必须等到这个会话的历史真的取回来了，才能断定"它是空的、可以写开场白"
    if (loadedConvId !== pending.convId) return;
    if (loading || messages.length > 0) return;
    pendingPromptRef.current = null;
    void sendContent(pending.content).catch(() => {
      // 发送链路自身已把失败渲染成可重试的失败态气泡，这里只吞掉 promise，
      // 避免 unhandled rejection。
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversationId, loadedConvId, loading, messages.length, pendingSeq]);

  /**
   * 把一条消息并入列表：同 id 已存在就**替换**，不存在才追加。
   *
   * 为什么不能用 `[...prev, msg]` 直接追加 —— 话题入口会踩到一个真实的竞态：
   * 建会话后 `sendContent` 发起 POST（服务端随即落库用户消息），而路由变化同时
   * 触发的 `loadMessages` GET 也会把这条刚落库的用户消息带回来。两条链路都往
   * `messages` 里放同一条消息，于是出现两条 `key` 相同的兄弟节点：
   * React 抛 "Encountered two children with the same key"，界面上开场白显示两遍。
   *
   * 修法是让"写入一条消息"这件事**幂等**（按 id upsert），而不是去压制告警。
   * 以后任何"加载 + 乐观追加"并存的时序都不会再产生重复节点。
   */
  const upsertMessage = (prev: Message[], msg: Message): Message[] => {
    const idx = prev.findIndex(m => m.id === msg.id);
    if (idx === -1) return [...prev, msg];
    const next = [...prev];
    next[idx] = msg;
    return next;
  };

  const handleDeleteConversation = async (e: React.MouseEvent, convId: string) => {
    e.stopPropagation();
    try {
      await consultationApi.deleteConversation(convId);
      setConversations(prev => prev.filter((c: any) => c.id !== convId));
      setConvTotal(t => Math.max(0, t - 1));
      if (convId === conversationId) {
        navigate('/chat');
        setMessages([]);
      }
    } catch (err: any) {
      // 以前这里是空的 catch：删失败时用户看不到任何提示，只觉得"删了又回来"。
      // 失败时这一行仍然留在列表里，所以不需要重拉列表把用户已经翻到的位置打回第一页。
      message.error(err?.message || '这次没删掉，你过一会儿再试试。');
    }
  };

  // ---- 对话历史批量删除 ----

  const exitSelectMode = () => {
    setSelectMode(false);
    setSelectedIds([]);
  };

  const toggleSelected = (convId: string) => {
    setSelectedIds(prev =>
      prev.includes(convId) ? prev.filter(id => id !== convId) : [...prev, convId],
    );
  };

  // 只对**当前已加载列表**全选。后端列表是分页的（默认 20 条/页，底部可"加载更多"），
  // 把"全选"说成"删掉全部历史"会超出用户看到的范围 —— 尤其现在删除不再自动补页，
  // 用户点"全选"时看到多少条，就只该删多少条。
  const toggleSelectAll = () => {
    setSelectedIds(prev =>
      prev.length === conversations.length ? [] : conversations.map((c: any) => c.id),
    );
  };

  const handleBatchDelete = () => {
    if (selectedIds.length === 0) return;
    const count = selectedIds.length;
    // 删除前先把 id 复制一份：确认弹窗还开着的时候，列表有可能被后台刷新改写，
    // 用快照才能保证"弹窗里说的三条"和"实际删掉的三条"是同一批。
    const ids = [...selectedIds];

    Modal.confirm({
      title: `删除选中的 ${count} 个对话？`,
      content: '删除后这些对话将从列表中移除，且无法恢复。',
      okText: '删除',
      cancelText: '取消',
      okButtonProps: { danger: true },
      onOk: async () => {
        setBatchDeleting(true);
        try {
          await consultationApi.deleteConversations(ids);
          setConversations(prev => prev.filter(c => !ids.includes(c.id)));
          setConvTotal(t => Math.max(0, t - ids.length));
          // 删掉的会话正开着：退回"未选中会话"的空态，否则中间栏会继续显示
          // 一个已经不在列表里的对话的消息。查的是 ref 里的实时 id，不是闭包快照。
          const openId = activeConvRef.current;
          if (openId && ids.includes(openId)) {
            navigate('/chat');
            setMessages([]);
          }
          exitSelectMode();
          message.success(`已删除 ${count} 个对话`);
        } catch (e: any) {
          // 服务端拒绝时（例如某一批里混进了别的设备已删掉的会话）列表可能
          // 已经变了，这里重新拉一次让界面回到真实状态；用户勾好的选择保留，
          // 直接再点一次删除即可。不关选择态是刻意的。
          message.error(e?.message || '这次没删掉，你过一会儿再试试。');
          void loadConversations();
          // 抛出去让 Modal.confirm 知道失败，保持弹窗可重试（onOk 里 catch 掉
          // 会被当成成功而关窗）。
          throw e;
        } finally {
          setBatchDeleting(false);
        }
      },
    });
  };

  /**
   * 打开某个对话的总结弹窗。
   *
   * `stopPropagation` 由调用方负责（列表整行点击是"切换对话"，
   * 不拦下来会一边切对话一边弹窗，用户看到的是两个动作同时发生）。
   */
  const handleOpenSummary = (convId: string, title?: string) => {
    setSummaryTarget({ id: convId, title });
  };

  /**
   * 「开始新对话」（空状态里的大按钮）与历史卡片右上角的「+」共用的入口：
   * 直接退回**对话页**，不预先建会话。
   *
   * 为什么不在这一步调 `createConversation`：那样会在数据库和左侧「对话历史」里留下
   * 一条没有内容、也没有标题的空对话 —— 用户点两下就多两条"新对话"，历史列表很快被垃圾填满，
   * 而他其实什么都还没说。
   *
   * 现在的时序是：这里只把界面清成"还没开始聊"的状态；等用户真的说第一句（输入框发送，
   * 或点六个话题之一）时，才由 `startConversationWithPrompt` 那一路创建会话。
   * 空会话在界面上和 `/chat` 没有任何区别（都显示小云朵 + 六个话题），所以观感一致。
   *
   * ⚠️ `setLoadedConvId(null)` 是必须的：否则 URL 变成 /chat 之后，它仍然记着上一个会话，
   * 空状态判据会把新页面误判成"那个会话还有内容"。
   */
  const handleStartFresh = () => {
    // 正在生成的回复要先断开，否则它的增量会继续写进这个已经被放弃的页面状态
    streamAbortRef.current?.();
    streamAbortRef.current = null;
    pendingPromptRef.current = null;
    setMessages([]);
    setInput('');
    setStreaming(false);
    setStreamText('');
    setLastFailed(null);
    setSending(false);
    setLoadedConvId(null);
    navigate('/chat');
  };

  /**
   * 建一个会话并把一句开场白发出去（当前**没有**可用会话时走这里）。
   *
   * 用 ref 记住"已经建过、用的是哪句开场白"：
   * `createConversation` 是异步的，期间 `navigate` 会让组件重新渲染，
   * `seed` 回调会再跑一遍。没有这个标记就会连建两个会话（用户看到历史里冒出一条空的）。
   * 失败时清掉标记并回退到空状态，让用户可以再点一次 —— 失败不能是死路。
   */
  const startConversationWithPrompt = async (prompt: string) => {
    if (creatingForRef.current) return;
    creatingForRef.current = prompt;
    setCreatingConversation(true);
    try {
      const res = await consultationApi.createConversation() as any;
      const newId = res.data.id as string;
      pendingPromptRef.current = { convId: newId, content: prompt };
      // 清掉旧气泡：待发逻辑要求"会话是空的"，中间栏挂着上一个会话的气泡会把发送压住。
      setMessages([]);
      setLastFailed(null);
      // 先把这条会话按"首句标题"插进列表，再跳转。
      //
      // 不能在这里 `loadConversations()`：服务端要等第一条消息落库才写标题，立刻重拉
      // 只会拿到"新对话"（截图里中间栏标题一直是"新对话"就是这个原因），而且会把
      // 整个列表重排一次。标题口径与服务端一致（前 30 字 + …）。
      const title = `${prompt.slice(0, 30)}...`;
      setConversations(prev => [
        { id: newId, title, messageCount: 0, isActive: true },
        ...prev.filter((c: any) => c.id !== newId),
      ]);
      setConvTotal(t => t + 1);
      navigate(`/chat/${newId}`);
    } catch {
      pendingPromptRef.current = null;
      setInput(previous => previous ? `${prompt}\n${previous}` : prompt);
      message.error('这次没能开始，你过一会儿再试试。');
    } finally {
      creatingForRef.current = null;
      setCreatingConversation(false);
    }
  };

  /**
   * 点空状态里的「话题」：进入对应情境。
   *
   * 三类情况：
   *   ① `/chat`（还没有会话）→ 建一个会话，把这句开场白发出去；
   *   ② 当前会话是空的 → 直接复用它（不浪费，也不产生垃圾条目）；
   *   ③ 当前会话已经有内容 → 另建一个（不能把新话题塞进正在聊的对话里）。
   */
  const handleTopicClick = (prompt: string) => {
    if (!conversationId) {
      void startConversationWithPrompt(prompt);
      return;
    }
    if (loadedConvId === conversationId && messages.length === 0) {
      pendingPromptRef.current = { convId: conversationId, content: prompt };
      setPendingSeq(n => n + 1);
      return;
    }
    void startConversationWithPrompt(prompt);
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
          makeAssistantMessage(res.data.errorMessage || '我这边一时没接上，你过一会儿再和我说好吗？', true),
        ]);
        setLastFailed(content);
        return;
      }

      if (res.data) {
        setLastFailed(null);
        // 与流式路径同一套语义：先清掉乐观占位，再按 id 幂等并入（见 `upsertMessage`）。
        // 回退路径与刚触发的 `loadMessages` 也可能撞上同一个用户消息。
        setMessages(prev => upsertMessage(
          upsertMessage(prev.filter(m => m.id !== 'temp' && m.id !== 'error'), res.data.userMessage),
          res.data.aiMessage,
        ));

        // 危机信号：由后端 Python 算法层统一判断（62+ 关键词库）
        const isCrisis = res.data.isCrisis === true || res.data.riskLevel === 'crisis';
        if (isCrisis) {
          setCrisisModalOpen(true);
        }

        // 转介信号：非流式路径也要认，否则两条路径行为不一致
        const tamer = (res.data as any).tamerResult ?? (res.data as any).tamer_result;
        if (tamer) {
          setTamerResult(tamer);
          if (tamer.should_refer) setReferralDismissed(false);
        }
      }
    } catch (e: any) {
      // 网络/HTTP 层失败：同样走可重试的失败态
      setMessages(prev => [
        ...prev.filter(m => m.id !== 'temp' && m.id !== 'error'),
        makeAssistantMessage(e?.message || '网络好像断了一下，你过一会儿再试试。', true),
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

    // 请求发出时读取最新在场状态与缩略帧，避免语音定稿期间的旧闭包丢掉画面。
    // CNN 表情摘要可缺失；三张真实画面仍可独立进入像素质量检查，不补造表情概率。
    const facialFeatures = getFacialTurnInput();

    // 流式链路：user → meta? → delta* → [revise] → done
    const handle = sendChatMessageStream(conversationId, content, {
      onUser: (userMessage) => {
        // 用落库后的真实消息替换本地乐观占位气泡；按 id upsert，避免与
        // 同一时间返回的 `loadMessages` 结果叠成两条（见 `upsertMessage` 说明）。
        setMessages(prev => upsertMessage(
          prev.filter(m => m.id !== 'temp' && m.id !== 'error'),
          userMessage as unknown as Message,
        ));
      },
      onDelta: (text) => setStreamText(prev => prev + text),
      // 安全审计改稿 / 走了降级路径 —— 必须**整体替换**，不能追加
      onRevise: (text) => setStreamText(text),
      onDone: (payload) => { donePayload = payload; },
      onError: (payload) => { errorPayload = payload; },
    }, 'text', undefined, facialFeatures);
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
        setMessages(prev => upsertMessage(prev, donePayload.aiMessage as Message));
      }
      // 提取 TAMER 框架输出（RQI / MIS / CUSUM）。
      // ⚠️ 兼容两种口径：最终 done 由 Node 侧构造（camelCase `tamerResult`），
      // 而算法层透传的那次 done 是 snake_case `tamer_result`；消费端两个都认，
      // 避免因取值口径不同而静默拿到 undefined（那样两个可视化组件会退回兜底分支）。
      const tamer = (donePayload as any).tamerResult ?? (donePayload as any).tamer_result;
      if (tamer) {
        setTamerResult(tamer);
        // 连续高风险达到阈值 → 允许再次弹出转介提示（用户此前关掉过也没关系）
        if (tamer.should_refer) setReferralDismissed(false);
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
        makeAssistantMessage(errorPayload.errorMessage || '我这边一时没接上，你过一会儿再和我说好吗？', true),
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
        makeAssistantMessage('刚才那句我没接上，你再点一次好吗？', true),
      ]);
      setLastFailed(content);
      return;
    }

    await sendContentFallback(content);
  };

  const handleSend = async () => {
    if (submitLockRef.current || supportFeedbackBusy || sending || loading || creatingForRef.current || pendingPromptRef.current) return;
    if (!voiceDraftRef.current.text.trim() && !voiceInputActive) return;
    submitLockRef.current = true;
    const boundary = voiceDraftRef.current.beginSend();
    setInputState('');
    setVoiceSubmitting(voiceInputActive);
    try {
      // 定稿只划分音频段，不停止设备；期间的新草稿与本次发送隔离。
      if (voiceInputActive) await commitVoiceInput();
      boundary.finish();
      setInputState(voiceDraftRef.current.text);
      setVoiceSubmitting(false);
      const content = boundary.text().trim();
      if (!content) return;
      analyzeEmotion(content);
      if (!conversationId) {
        await startConversationWithPrompt(content);
        return;
      }
      const userMsg: Message = { id: 'temp', role: 'user', content, createdAt: new Date().toISOString() };
      setMessages(prev => [...prev, userMsg]);
      await sendContent(content);
    } finally {
      boundary.finish();
      setVoiceSubmitting(false);
      submitLockRef.current = false;
    }
  };

  const handleRetry = async () => {
    if (!lastFailed) return;
    await sendContent(lastFailed);
  };

  // 注：消息气泡旁的「情绪识别结果」标签（⚡ 检测到 中性 80% / 平静 30%）已按用户要求
  // **整段移除**（原 `renderEmotionTag`）。情绪数据仍照常写进 `message.sentiment`
  // 供咨询师端与审计链路使用，只是不再在学生的对话界面里贴出来。

  // 注：AI 回复气泡旁的「对话模式」标签（共情陪伴 / 苏格拉底引导）与
  // 「参考了 N 个相似经历」标签也已按用户要求**整段移除**（原 `renderAiMeta`）。
  // 完整的 { dialogue_mode, scene_retrieval, cognitive_distortion } 仍照常写进
  // message.sentiment，供咨询师端与审计链路使用，只是不再在学生的对话界面里贴出来。

  /**
   * 中间栏是显示「小云朵 + 六个话题」的空状态，还是显示消息流。
   *
   * 判据是"当前会话有没有消息"，而**不是**"URL 里有没有会话 id"：
   * 「开始新对话」/「+」现在直接回 `/chat`（不建会话），但刷新、或从左侧历史点进一段
   * 还没说过话的对话时，URL 里是带着 id 的 —— 这几种情况用户看到的都是同一个
   * "还没开始聊"的页面，空状态与六个话题必须照旧在场（见 `handleStartFresh` /
   * `startConversationWithPrompt`）。
   *
   * `loadedConvId === conversationId` 这一环不能省：切到一个**有**消息的会话时，
   * 第一帧 messages 还没换过来（长度同样是 0），只看消息数会先闪一下空状态。
   */
  const emptyStateVisible = !conversationId
    || (loadedConvId === conversationId && messages.length === 0 && !streaming && !sending);

  /**
   * 「小云朵 + 六个话题 + 开始新对话」这一块。
   *
   * 抽成变量是因为它有两个入口：URL 里还没有会话（/chat），以及当前会话是空的。
   * 两处必须是**同一份** JSX —— 之前正是在这里分叉的：只有 `!conversationId`
   * 才渲染它，于是"点开始新对话（URL 变成空会话）"和"刷新一个空会话 URL"
   * 都会让整块内容消失。
   */
  const emptyState = (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'flex-start', padding: '16px 24px 12px', overflow: 'auto' }}>
      {/*
        空状态插图：这里必须显式给图片。
        antd 的 Empty 取值是 `image ?? contextImage ?? defaultEmptyImg`，
        传 image={null} 会被判为「未提供」而回落到内置的灰色「暂无数据」SVG，
        所以之前这一块一直显示灰色信封插画。
        换成本项目的小云朵插图（背景已对齐卡片底色 #FFFBFC，无可见边框）。

        ⚠️ 尺寸必须受控（128→96px，并显式给 img 高度）：这一页顶部有 500px 的图条，
        中间栏卡片从视口 ~520px 处才开始，可见高度只剩 ~380px。原样用 168px 插图 + 40px
        内边距时整块高达 ~600px，在 1440×900 / 1536×864 这类常见屏幕上「六个话题」
        落在折叠线以下 176~212px —— 用户得先滚动才能点到，等于白摆。
        96px 是实测能把「云朵 + 提示语 + 六个话题 + 开始按钮」整块放进可见区的高度。
      */}
      <Empty
        image={<img src="/assets/chat-empty-mascot.png" alt="小云朵" draggable={false} style={{ height: 96, width: 'auto' }} />}
        styles={{ image: { height: 96, marginBottom: 8 } }}
        description={
          <span style={{ color: '#8a7a9a', fontSize: 15 }}>
            不用想好怎么开口，我在。
          </span>
        }
      />
      <div style={{ marginTop: 12, maxWidth: 400, textAlign: 'center' }}>
        <Text type="secondary" style={{ fontSize: 13, display: 'block', marginBottom: 8 }}>
          不知道从哪说起的话，可以挑一句：
        </Text>
        <Space wrap size={[8, 8]} style={{ justifyContent: 'center' }}>
          {TOPIC_PROMPTS.map(({ topic, prompt }) => (
            <Tag
              key={topic}
              // 便于自动化验证与埋点定位到具体是哪一个话题
              data-topic={topic}
              style={{
                cursor: 'pointer', padding: '6px 14px', borderRadius: 20,
                background: '#f5f3ff', color: '#6366f1', border: 'none', fontSize: 13,
              }}
              onClick={() => handleTopicClick(prompt)}
            >
              {topic}
            </Tag>
          ))}
        </Space>
      </div>
      {/*
        直接回 `/chat`（不建会话）：点了它就是要一段全新的、还没开口的对话，
        而左侧历史里不该因此多出一条空的"新对话"（见 `handleStartFresh`）。
      */}
      <Button type="primary" size="large" icon={<PlusOutlined />} onClick={handleStartFresh}
        style={{ marginTop: 14, borderRadius: 24, padding: '0 32px' }}>
        开始聊聊
      </Button>
    </div>
  );

  /**
   * 底部输入区。
   *
   * ⚠️ 它必须在**空状态和消息流两种情况下都渲染**：按现在的时序，用户的第一句话
   * 就是在 `/chat` 的空状态里打到这个输入框、再由 `handleSend` 创建会话发出去的。
   * 之前它只写在消息流那一支里，于是"开始新对话"回到 /chat 后整页没有能打字的地方。
   */
  const composer = (
    <div className="chat-composer" style={{ padding: '16px 24px', borderTop: '1px solid #f0f0f0' }}>
      {/* 实时情绪感知条 */}
      {latestEmotion && (
        <div style={{ padding: '8px 0 12px', display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <Space size={4}>
            <SmileOutlined style={{ color: '#6366f1', fontSize: 14 }} />
            <Text strong style={{ fontSize: 12 }}>我听到的情绪</Text>
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
          <Tag color="purple" style={{ fontSize: 10, margin: 0 }}>我的把握 {Math.round(latestEmotion.confidence * 100)}%</Tag>
        </div>
      )}
      {/* 创新点②：矛盾检测提示（MIS 可视化） */}
      {latestEmotion?.text_emotion_probs && (
        <div style={{ padding: '4px 0 8px' }}>
          <IncongruenceAlert
            textEmotionProbs={latestEmotion.text_emotion_probs}
            tamerResult={tamerResult}
            // 只有本轮真的采到了语音才传：`duration > 0` 才说明录过音，
            // 否则会把上一次录音的残留指标当成"本轮声音"。
            voiceFeatures={voiceMetrics?.duration ? voiceMetrics : undefined}
          />
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
            {stoppingVoice ? '正在关闭语音输入…' : '持续语音输入中'}
          </span>
          <span style={{ fontSize: 13, color: '#4a4a6a', flex: 1, minHeight: 18 }}>
            文字会直接进入输入框；点击发送提交这一轮，麦克风继续听。
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
          <span style={{ fontSize: 12, color: '#6366f1' }}>我在听你刚才说的，马上写下来…</span>
        </div>
      )}
      {voiceMetrics.asrError && (
        <Alert
          type="warning"
          showIcon
          style={{ marginBottom: 8, fontSize: 12 }}
          message={`这段语音我没听清：${voiceMetrics.asrError}`}
        />
      )}
      <Space.Compact style={{ width: '100%' }}>
        <Tooltip title={cameraEnabled ? '关闭摄像头/麦克风' : '开启摄像头/麦克风；每轮三张缩略帧送项目算法服务作临时图文参考，原图不保存，不参与风险判断'}>
          <Button
            icon={<VideoCameraOutlined />}
            onClick={toggleCamera} disabled={voiceSubmitting}
            type={cameraEnabled ? 'primary' : 'default'}
            size="large"
            style={{ borderRadius: '8px 0 0 8px', background: cameraEnabled ? '#52c41a' : undefined }}
          />
        </Tooltip>
        <Tooltip title={voiceInputActive ? '关闭麦克风；发送消息无需关闭' : '开启持续语音输入：边说边进入输入框，点击发送提交这一轮'}>
          <Button
            icon={<AudioOutlined />}
            onClick={() => void handleToggleVoice()}
            loading={stoppingVoice} disabled={voiceSubmitting}
            type={voiceInputActive ? 'primary' : 'default'}
            danger={voiceInputActive}
            size="large"
          />
        </Tooltip>
        <Input value={input} onChange={(e) => setInput(e.target.value)} onPressEnter={handleSend}
          placeholder="我在听，慢慢说…" disabled={sending || creatingConversation} size="large" style={{ borderRadius: 0 }} />
        <Button type="primary" icon={<SendOutlined />} onClick={handleSend} loading={sending || creatingConversation || voiceSubmitting} disabled={supportFeedbackBusy || loading} size="large" style={{ borderRadius: '0 8px 8px 0' }}>发送</Button>
      </Space.Compact>
    </div>
  );

  const currentMessage = conversationId && loadedConvId === conversationId ? messages[messages.length - 1] : undefined;
  const currentReference = currentMessage?.role === 'assistant' && !currentMessage.failed ? innovationFromSentiment(currentMessage.sentiment) : null;
  const companionPanel = <ChatCompanionPanel reference={currentReference} recordedAt={currentMessage?.role === 'assistant' ? currentMessage.createdAt : undefined}
    liveQuality={{ enabled: cameraEnabled, detecting: facialMetrics.isDetecting, facePresent: facialMetrics.facePresent, reason: facialMetrics.unavailableReason || mediaNotice, getSampleCount: getVisualFrameCount }}
    waiting={sending || streaming || currentMessage?.role === 'user'} activeTab={activeCompanionTab} onTabChange={openCompanion}
    refreshKey={`${conversationId || ''}:${currentMessage?.id || ''}:${currentMessage?.sentiment || ''}`} />;

  return (
    <div className={`chat-layout${wideCompanion && companionOpen ? ' chat-layout--companion' : ''}`} style={{ display: 'flex', gap: 16, height: 'calc(100vh - 200px)' }}>
      {/* 流式回复的闪烁光标。放在组件内而不是全局 CSS：只有这里用得到，
          且避免为了一个 keyframes 去改公共样式文件。 */}
      <style>{`
        @keyframes blink-caret { 0%, 100% { opacity: 1 } 50% { opacity: 0 } }
        .chat-history-toggle { display: none; }
        @media (max-width: 767px) {
          .chat-layout { flex-direction: column; height: auto !important; gap: 12px !important; }
          .chat-history { width: 100% !important; height: auto !important; flex-shrink: 0; }
          .chat-history-toggle { display: inline-block; }
          .chat-history:not([data-expanded="true"]) > .ant-card-body { display: none; }
          .chat-history > .ant-card-body { height: auto !important; max-height: 160px; }
          .chat-main { width: 100%; height: 640px !important; min-height: 640px; }
          .chat-composer { padding: 12px !important; }
          .chat-main [data-testid="support-understanding"] { margin-left: 0 !important; }
        }
      `}</style>
      <CrisisModal
        open={crisisModalOpen}
        onClose={() => setCrisisModalOpen(false)}
        onGoChat={() => setCrisisModalOpen(false)}
      />
      {/* 对话总结弹窗（元宝纪要风格）。挂在页面级而不是列表项里：
          列表项会被 List 复用时重挂，弹窗状态跟着列表滚动丢掉。
          `destroyOnClose` 在组件内部已开，所以关掉后不会留着上一个会话的纪要。 */}
      <ConversationSummaryModal
        open={summaryTarget !== null}
        conversationId={summaryTarget?.id ?? null}
        conversationTitle={summaryTarget?.title}
        onClose={() => setSummaryTarget(null)}
      />
      <Card className="chat-history" data-expanded={mobileHistoryOpen} style={{ width: 280, height: '100%', borderRadius: 12 }}
        title={<Text strong>我们聊过的</Text>}
        extra={
          <Space size={2}>
            <Button className="chat-history-toggle" type="text" size="small" aria-expanded={mobileHistoryOpen}
              onClick={() => setMobileHistoryOpen(open => !open)}>{mobileHistoryOpen ? '收起' : '展开'}</Button>
            <Tooltip title="和 AI 打电话（语音陪伴，整屏）">
              <Button type="text" icon={<PhoneOutlined />} onClick={handleStartCall} />
            </Tooltip>
            {/* 管理入口只在有对话时出现：空列表下"批量删除"没有意义 */}
            {conversations.length > 0 && (
              <Tooltip title={selectMode ? '退出管理' : '管理（全选后批量删除已加载的对话）'}>
                <Button type="text" size="small" onClick={() => (selectMode ? exitSelectMode() : setSelectMode(true))}
                  style={{ fontSize: 12, color: selectMode ? '#6366f1' : undefined, paddingInline: 4 }}>
                  {selectMode ? '完成' : '管理'}
                </Button>
              </Tooltip>
            )}
            <Button type="text" icon={<PlusOutlined />} onClick={handleStartFresh} />
          </Space>
        }
        bodyStyle={{ padding: '8px 0', height: 'calc(100% - 57px)', overflowY: 'auto' }}>
        {/* 批量操作条：只有进入管理模式才出现，平时完全不占位、不干扰单条删除 */}
        {selectMode && conversations.length > 0 && (
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            padding: '4px 12px 8px', marginBottom: 4, borderBottom: '1px solid #f0f0f0',
          }}>
            <Checkbox
              checked={conversations.length > 0 && selectedIds.length === conversations.length}
              indeterminate={selectedIds.length > 0 && selectedIds.length < conversations.length}
              onChange={toggleSelectAll}
            >
              <span style={{ fontSize: 12 }}>
                全选{selectedIds.length > 0 ? `（${selectedIds.length}）` : ''}
              </span>
            </Checkbox>
            {/* 确认只有一处：点击后弹 Modal.confirm（见 handleBatchDelete）。
                这里不再套 Popconfirm —— 两层确认框对同一个动作是纯负担。 */}
            <Tooltip title={selectedIds.length === 0 ? '先勾选要删除的对话' : `删除选中的 ${selectedIds.length} 个对话`}>
              <Button type="text" danger size="small" icon={<DeleteOutlined />}
                disabled={selectedIds.length === 0}
                loading={batchDeleting}
                onClick={handleBatchDelete}
                style={{ fontSize: 12, paddingInline: 4 }}>
                删除
              </Button>
            </Tooltip>
          </div>
        )}
        <List
          dataSource={conversations}
          locale={{ emptyText: <Empty description="还没有聊过。点上面的 +，我们从第一句开始。" image={Empty.PRESENTED_IMAGE_SIMPLE} /> }}
          renderItem={(item: any) => (
            <List.Item
              onClick={() => (selectMode ? toggleSelected(item.id) : navigate(`/chat/${item.id}`))}
              style={{ cursor: 'pointer', padding: '10px 8px', borderRadius: 8, background: item.id === conversationId ? '#f5f3ff' : 'transparent' }}
              actions={selectMode ? undefined : [
                // 总结入口用 Tooltip + 文字图标：历史列表是窄条，塞两个常驻图标会把
                // 标题挤没。删除保持原样（一键 + 二次确认），手感不因为新增功能而变慢。
                <Tooltip key="summary" title="查看这段对话的总结">
                  <Button
                    type="text"
                    size="small"
                    data-action="conversation-summary"
                    icon={<FileTextOutlined style={{ color: '#8c8aa7', fontSize: 14 }} />}
                    onClick={(e) => {
                      // 整行点击 = 切换对话。必须拦下来，否则会一边切走一边弹窗。
                      e.stopPropagation();
                      handleOpenSummary(item.id, item.title);
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.color = '#6366f1')}
                    onMouseLeave={(e) => (e.currentTarget.style.color = '')}
                    style={{ paddingInline: 2 }}
                  />
                </Tooltip>,
                <Popconfirm key="del" title="确定删除？" onConfirm={(e) => handleDeleteConversation(e as any, item.id)} onCancel={(e) => e?.stopPropagation()} okText="删除" cancelText="取消" okButtonProps={{ danger: true }}>
                  <DeleteOutlined onClick={(e) => e.stopPropagation()} style={{ color: '#ccc', fontSize: 14, cursor: 'pointer' }}
                    onMouseEnter={(e) => (e.currentTarget.style.color = '#ff4d4f')} onMouseLeave={(e) => (e.currentTarget.style.color = '#ccc')} />
                </Popconfirm>,
              ]}
            >
              {selectMode && (
                <Checkbox
                  checked={selectedIds.includes(item.id)}
                  // 勾选框自己不再处理点击：整行点击已经能切换勾选，
                  // 两处都挂 handler 会在冒泡时把状态切两次（等于没切）。
                  style={{ marginRight: 8, pointerEvents: 'none' }}
                />
              )}
              <Text ellipsis style={{ fontSize: 13 }}>{item.title || '新对话'}</Text>
            </List.Item>
          )}
        />
        {/* 分页入口：后端一页 20 条，想看得更早的对话必须自己点这里。
            刻意不做"到底自动加载"：删除之后列表不该自己悄悄补一批上来 ——
            那正是"删掉的又回来了"的观感来源。 */}
        {conversations.length > 0 && conversations.length < convTotal && (
          <div style={{ padding: '4px 12px 8px' }}>
            <Button block size="small" loading={convLoadingMore} onClick={loadMoreConversations}
              style={{ fontSize: 12 }}>
              加载更多（还有 {convTotal - conversations.length} 条）
            </Button>
          </div>
        )}
      </Card>

      <Card className="chat-main"
        style={{ flex: 1, minWidth: 0, height: '100%', borderRadius: 12, display: 'flex', flexDirection: 'column' }}
        // 中间栏补一个标题 + 总结入口：历史列表里的图标按行找，当前会话的总结
        // 应该在"正在看的这段对话"顶上就能点到，而不是回到左侧去找那一行。
        title={
          conversationId ? (
            <Text ellipsis style={{ fontSize: 14 }}>
              {conversations.find((c: any) => c.id === conversationId)?.title || '对话'}
            </Text>
          ) : (
            <Text type="secondary" style={{ fontSize: 14 }}>我在这儿</Text>
          )
        }
        extra={
          conversationId ? (
            <Tooltip title="把这段对话收成一页，留在这里，想回看时再看。">
              <Button
                type="text"
                size="small"
                data-action="current-conversation-summary"
                icon={<FileTextOutlined />}
                onClick={() =>
                  handleOpenSummary(
                    conversationId,
                    conversations.find((c: any) => c.id === conversationId)?.title,
                  )
                }
                style={{ fontSize: 12, color: '#6366f1' }}
              >
                回顾我们聊的
              </Button>
            </Tooltip>
          ) : undefined
        }
        bodyStyle={{ flex: 1, display: 'flex', flexDirection: 'column', padding: 0, overflow: 'hidden' }}>
        <div className="chat-companion-toolbar">
          <span className="chat-companion-toolbar-label">我会这样陪你</span>
          <Space size={2} wrap>{([
            ['fusion', '图文理解'], ['quality', '质量纠偏'], ['tracking', '持续陪伴'],
          ] as const).map(([key, label]) => <Button key={key} size="small" type={companionOpen && activeCompanionTab === key ? 'primary' : 'text'}
            aria-label={`打开${label}面板`} onClick={() => openCompanion(key)}>{label}</Button>)}</Space>
          {/* 右栏收起（或窄屏下变成抽屉）时，三种情绪读数在这里补一行 ——
              否则"看得到右栏才有读数"，等于又把这个功能藏起来了。 */}
          {!(wideCompanion && companionOpen) && <FaceEmotionChips />}
        </div>
        {emptyStateVisible ? (
          emptyState
        ) : (
          <>
            {/* 展示实际交流轮次，不把研究性 RQI 当作信任程度。 */}
            {loadedConvId === conversationId && messages.length > 0 && (
              <div style={{ padding: '12px 24px 0' }}>
                <RelationshipStageIndicator key={conversationId} messages={messages} />
              </div>
            )}

            {/*
              转介提示：算法层（TAMER）在连续高风险轮次达到阈值时给出 should_refer。
              此前这个信号在算法层算完、Node 层也透传了类型，但没有任何界面消费它 ——
              「AI 知道该转给真人」这件事对用户不可见。这里把它变成一步可点击的动作。
              ⚠️ 措辞刻意避开诊断与恐吓，只说「也许更有帮助」，并保留关闭入口。
            */}
            {tamerResult?.should_refer && !referralDismissed && (
              <div style={{ padding: '12px 24px 0' }}>
                <Alert
                  type="info"
                  showIcon
                  closable
                  onClose={() => setReferralDismissed(true)}
                  style={{ borderRadius: 12 }}
                  message="也许现在找一位真人咨询师聊聊会更有帮助"
                  description={
                    <div>
                      <div style={{ marginBottom: 10 }}>
                        我听得出来，你最近几轮都挺沉的。我可以陪你聊，但替代不了真人咨询。
                        平台上有公益咨询师可以预约；如果现在就需要有人说话，也可以拨打下面这条热线。
                      </div>
                      <Space wrap>
                        <Button type="primary" size="small" style={{ borderRadius: 16 }}
                          onClick={() => navigate('/experts')}>
                          预约咨询师
                        </Button>
                        <Tag icon={<PhoneOutlined />} color="red" style={{ fontSize: 12 }}>
                          全国24小时心理援助 400-161-9995
                        </Tag>
                      </Space>
                    </div>
                  }
                />
              </div>
            )}
            <div style={{ flex: 1, overflow: 'auto', padding: 24 }}>
              {loading ? <div style={{ textAlign: 'center' }}><Spin /></div> : (
                messages.map((msg) => (
                  <div key={msg.id} data-msg-id={msg.id} data-msg-role={msg.role} style={{ display: 'flex', marginBottom: 16, flexDirection: msg.role === 'user' ? 'row-reverse' : 'row' }}>
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
                      <><Spin size="small" /> <Text type="secondary" style={{ marginLeft: 8 }}>我在想怎么回你…</Text></>
                    )}
                  </div>
                </div>
              )}
              {!sending && !streaming && conversationId && (() => {
                const latest = messages[messages.length - 1];
                const state = latest?.role === 'assistant' && !latest.failed ? supportStateFromSentiment(latest.sentiment) : null;
                if (!latest || latest.role !== 'assistant' || latest.failed) return null;
                // 陪伴方式只在开头轻轻问一次；之后用户可以直接说，或到「我理解对了吗」里改。
                const userTurns = messages.filter(msg => msg.role === 'user').length;
                return <><InnovationStatus sentiment={latest.sentiment} />{state ? <SupportUnderstandingCard key={latest.id} conversationId={conversationId} messageId={latest.id} state={state}
                  showGoalOffer={userTurns <= 2}
                  onBusyChange={setSupportFeedbackBusy}
                  onSaved={updated => setMessages(previous => previous.map(msg => {
                    if (msg.id !== latest.id) return msg;
                    const sentiment = JSON.parse(msg.sentiment || '{}');
                    return { ...msg, sentiment: JSON.stringify({ ...sentiment, support_state: updated }) };
                  }))} /> : null}</>;
              })()}
              <div ref={messagesEndRef} />
            </div>
          </>
        )}
        {/*
          输入区**抽到分支外面**：空状态（/chat，或一段还没说过话的对话）也要有输入框。
          之前它在消息流那一支里，于是"开始新对话"回到 /chat 后整页没有可以打字的地方 ——
          而按现在的时序，用户的第一句话正是在这个输入框里发出来、并由此创建会话的。
        */}
        {composer}
      </Card>
      {wideCompanion ? <aside className="chat-companion-aside" hidden={!companionOpen} aria-label="倾诉中的陪伴与感知">
        <div className="chat-companion-heading"><strong>陪伴与感知</strong><Button type="text" size="small" onClick={closeCompanion}>收起</Button></div>
        {/* 三种表情读数钉在滚动区**外面**：切换 Tab、往下翻「持续陪伴」时它都还在，
            这才是原来左下角浮层"始终可见"的真正替代，而不是被卷走的一行。 */}
        <div className="chat-companion-emotion"><FaceEmotionBlock /></div>
        <div className="chat-companion-scroll">{companionPanel}</div>
      </aside> : <Drawer title="AI 倾诉 · 陪伴与感知" open={companionOpen} onClose={closeCompanion} width={Math.min(400, window.innerWidth)}>
        <div className="chat-companion-emotion chat-companion-emotion--drawer"><FaceEmotionBlock /></div>
        {companionPanel}
      </Drawer>}
    </div>
  );
}

// 主组件
export default function Chat() {
  return <AIChatTab />;
}

/** 旧的独立入口进入同一个倾诉工作区，保留所请求的能力面板。 */
export function LegacyCompanionRedirect() {
  const location = useLocation();
  const panel = location.hash.slice(1);
  return <Navigate to={`/chat?panel=${panel === 'quality' || panel === 'tracking' ? panel : 'fusion'}`} replace />;
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
