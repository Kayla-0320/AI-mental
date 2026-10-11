/**
 * AI 陪伴通话 —— 整屏页面
 *
 * ## 为什么是整屏、且刻意不进 `PatientLayout`
 *
 * 通话的视觉隐喻是"电话"：全屏、没有侧边栏、没有导航。挂上侧边栏就变回"网页"了。
 * 因此路由与 `/login` 同级（见 `App.tsx`），而不是挂在 `/` 下面。
 *
 * ## 页面负责什么 / 不负责什么
 *
 * 这个文件**只负责看和点**：布局、状态词、字幕、按钮、计时、跳转。
 * 轮次编排（ASR → 定稿 → SSE → TTS → 打断）全在 `useVoiceCall` 里；
 * 通话期视觉（存在感 / 光线 / 本机表情读数）全在 `useCallVision` 里 ——
 * 那两个 hook 才是需要写测试的地方。
 *
 * ## 面部读数怎么进对话（2026-10-11）
 *
 * 用户在通话里开了摄像头时，本机 ONNX 表情模型（happy/neutral/sad）的读数会被打成
 * 一个**实时状态快照**，随本轮消息作为**临时提示词**上行给算法层（见
 * `services/facialLiveState.ts` 与 `useVoiceCall` 的 `getFacialLiveState`）。
 * 三条边界不变：**不进风险等级、不进融合权重、不写画像分数**；没开摄像头就整段不存在。
 * 界面上的文案必须如实说明这一点（"会作为 AI 回应的内部参考"），不许只说"只给你自己看"。
 *
 * ## 「打断并发送」按钮（2026-10-11）
 *
 * 用户反馈"说话被截断太快"，因此把服务端的自动判停放宽到 1.8 s 静音
 * （见 `algorithm/asr/streaming.py` 的 rule2），并补上这个手动出口：
 * **立刻停掉小安的声音 + 把已经说出口的话马上定稿发出去**，不用等那 1.8 s。
 *
 * 它只是调用 `useVoiceCall().interruptAndSend`（打断与定稿都在 hook 里，
 * 这个文件仍然"只负责看和点"）。插话本身仍走 F-4 的既有语义：
 * 本轮不 abort、插话排队等 `done` 之后作为下一轮发 —— 按钮**不**绕过服务端顺序。
 *
 * ## 为什么"开始通话"必须点一下（不是偷懒，是两个硬约束的交点）
 *
 * 1. `AudioContext` 必须在**用户手势**里 `resume()`，否则自动播放策略会让第一句话没声音，
 *    而且事后无法补救。
 * 2. 麦克风若在 effect 里申请，会撞上 React 18 StrictMode 的"假卸载"：先 cleanup
 *    （`release`）再重新 effect（`acquire`），而 `mediaStreamManager` 的引用计数是
 *    **按消费者名去重的 Set** —— 同一个名字记不了两次，于是后一次拿到的可能是
 *    **已经被 stop 掉的轨道**（麦看起来是开的、界面什么都不报，实际全是静音）。
 *    放进点击回调就只会申请一次，从根上绕开这个竞态。
 *
 * 摄像头同理：它也必须由点击打开（`useCallVision.start()` 会 `getUserMedia`）。
 *
 * ## `?debug=1` 或 `Ctrl+Shift+D`
 *
 * 打开一个**只显示真实测量值**的诊断面板（轮次耗时、ASR/TTS 引擎、回声判定、
 * 打断计数、麦克风**实际生效**的 AEC 值、视觉读数）。它不是演示数据 ——
 * 里面每个数字都是这一通电话里实测出来的，用来给 S4/S5/S6/S7 的闸门取数。
 *
 * ⚠️ 两个开关都留着，但**排查通话中必须能开**：手改地址栏要刷新页面，而刷新会
 * 直接挂断正在进行的通话；`Ctrl+Shift+D` 不用刷新、不用记住会话 id（2026-10-11 加）。
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';
import { Alert, Button, Card, Popover, Space, Tag, Tooltip, Typography } from 'antd';
import {
  ArrowLeftOutlined,
  AudioMutedOutlined,
  AudioOutlined,
  PhoneFilled,
  SendOutlined,
  VideoCameraOutlined,
} from '@ant-design/icons';
import { consultationApi } from '../../services';
import { mediaStreamManager } from '../../services/mediaStreamManager';
import { useVoiceCall, type CallPhase } from '../../hooks/useVoiceCall';
import { useCallVision } from '../../hooks/useCallVision';
import { FACE_EMOTION_LABELS_ZH } from '../../services/faceEmotionClassifier';
import { groupTranscript, type CallTranscriptEntry } from '../../services/callTranscript';
import CrisisResourceCard from '../../components/CrisisResourceCard';

const { Text, Title } = Typography;

/**
 * sessionStorage 键：通话期持久化当前会话 id。
 *
 * 覆盖两种"通话被打断但没挂断"的场景：
 *   * 用户中途刷新页面；
 *   * 误关标签页 / 浏览器崩溃后重新打开 `/call`（此时 URL 里没有 id）。
 *
 * ⚠️ 只在**挂断**（phase === 'ended'）时清除，见下面的 effect —— 否则下次进
 * `/call` 会被送回一通已经结束的通话。
 */
const CALL_SESSION_KEY = 'voice-call:active-conversation';

const PHASE_LABEL: Record<CallPhase, string> = {
  idle: '准备就绪',
  connecting: '正在接通…',
  listening: '正在聆听',
  thinking: '正在思考…',
  speaking: '正在回应…',
  crisis: '已切换为文字陪伴',
  ended: '通话已结束',
  error: '需要注意',
};

/** 状态点的颜色：绿=在听，紫=在想/在说，橙=危机，红=异常。 */
const PHASE_DOT: Record<CallPhase, string> = {
  idle: '#a5b4fc',
  connecting: '#a5b4fc',
  listening: '#52c41a',
  thinking: '#8b5cf6',
  speaking: '#6366f1',
  crisis: '#fa8c16',
  ended: '#bfbfbf',
  error: '#ff4d4f',
};

/**
 * 头像律动。**纯装饰** —— 说话时快而明显、聆听时慢而轻。
 * 真正的信息在状态词与字幕里，所以读屏用户不会漏掉任何东西。
 */
const AVATAR_ANIMATION: Record<CallPhase, string> = {
  idle: 'call-breathe 4s ease-in-out infinite',
  connecting: 'call-breathe 2.4s ease-in-out infinite',
  listening: 'call-breathe 4s ease-in-out infinite',
  thinking: 'call-breathe 1.6s ease-in-out infinite',
  speaking: 'call-pulse 0.9s ease-in-out infinite',
  crisis: 'none',
  ended: 'none',
  error: 'none',
};

/** `mm:ss` */
function formatDuration(totalSeconds: number): string {
  const safe = Math.max(0, Math.floor(totalSeconds));
  return `${String(Math.floor(safe / 60)).padStart(2, '0')}:${String(safe % 60).padStart(2, '0')}`;
}

/** 一行的诊断读数 */
function DebugRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: 'flex', gap: 8, fontSize: 12 }}>
      <span style={{ color: '#8c8aa7', minWidth: 104 }}>{label}</span>
      <span data-debug={label} style={{ color: '#3f3d56', fontVariantNumeric: 'tabular-nums' }}>
        {value}
      </span>
    </div>
  );
}

/** 视觉小窗里那句"人话"（每一句都能对上一条实测状态，没有形容词撑场面） */
function presenceText(v: {
  paused: boolean;
  unavailableReason: string;
  frameFresh: boolean;
  present: boolean;
}): string {
  if (v.paused) return '采集已暂停（页面在后台）';
  if (v.unavailableReason) return v.unavailableReason;
  if (!v.frameFresh) return '画面已停住，读数暂空';
  return v.present ? '你在画面里' : '暂时看不到你';
}

/** 光线分档 → 一句人话 */
function lightText(level: 'ok' | 'dim' | 'dark' | 'unknown'): string {
  switch (level) {
    case 'ok':
      return '光线正常';
    case 'dim':
      return '光线偏暗';
    case 'dark':
      return '光线太暗，看不清';
    default:
      return '光线 —';
  }
}

/** 时钟时刻 `HH:mm:ss`（通话可能跨零点，所以不用相对秒数代替它） */
function formatClock(tsMs: number): string {
  const d = new Date(tsMs);
  const pad = (n: number): string => String(n).padStart(2, '0');
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

/**
 * 相对通话开始的 `mm:ss`。`startedAtMs <= 0`（还没接通）时退化成 `00:00`。
 *
 * ⚠️ 与顶栏计时同一个基准：页面在 `phase !== 'idle'` 时把 `startedAtRef` 打上时间戳。
 */
function relativeClock(tsMs: number, startedAtMs: number): string {
  if (startedAtMs <= 0) return '00:00';
  return formatDuration(Math.max(0, Math.round((tsMs - startedAtMs) / 1000)));
}

/**
 * 一条对话气泡 —— 与文字对话页同一种语言：用户靠右、紫色实底；小安靠左、白底浮起。
 *
 * 比文字页多两样东西，都是这次"把每轮都弄出来"才可能显示出来的：
 *   · `meta`：这条消息在**本轮第几秒**、以及真实时刻；
 *   · `revised` / `unspoken`：安全审计改过稿、或这一截**没有被念出来**。
 *     ⚠️ 后者不是装饰 —— 打断/危机切换会停掉本轮剩余音频，而字幕按"内容不藏"照旧上屏，
 *     不标出来用户就会以为"屏幕上这段是听到的"。
 */
function CallBubble({ v, meta }: { v: CallTranscriptEntry; meta: string }) {
  const isUser = v.speaker === 'user';
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: isUser ? 'flex-end' : 'flex-start' }}>
      <span
        style={{
          fontSize: 11,
          color: '#a6a3bb',
          margin: isUser ? '0 4px 3px 0' : '0 0 3px 4px',
          fontVariantNumeric: 'tabular-nums',
        }}
      >
        {meta}
      </span>
      <div
        style={{
          maxWidth: '86%',
          padding: isUser ? '9px 14px' : '10px 14px',
          borderRadius: isUser ? '14px 14px 4px 14px' : '14px 14px 14px 4px',
          background: isUser ? '#6366f1' : '#fff',
          boxShadow: isUser ? 'none' : '0 2px 10px rgba(99,102,241,0.10)',
          color: isUser ? '#fff' : '#3f3d56',
          lineHeight: 1.7,
          fontSize: 15,
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
        }}
      >
        {!isUser && (
          <Text type="secondary" style={{ fontSize: 11, display: 'block', marginBottom: 2 }}>
            小安{v.revised ? '（安全审计已改稿）' : ''}
          </Text>
        )}
        {v.text}
      </div>
      {v.unspoken && (
        <span style={{ fontSize: 11, color: '#a8600a', margin: isUser ? '3px 4px 0 0' : '3px 0 0 4px' }}>
          这一截没有说话（本轮语音已停），只有文字
        </span>
      )}
    </div>
  );
}

/**
 * 过程记录里的一轮：轮次 + 相对时刻 + 真实时刻。
 *
 * 用户口径是"**每轮**对话过程"，所以按轮插一条分隔；小屏幕上信息密度不能太高，
 * 于是分隔行只用一行 11px 的小字（比每条气泡都挂时间戳更省地方，也更好读）。
 */
function turnLabel(group: { turn: number; tsMs: number }, startedAtMs: number): string {
  return `第 ${group.turn} 轮 · 通话第 ${relativeClock(group.tsMs, startedAtMs)} · ${formatClock(group.tsMs)}`;
}

export default function VoiceCall() {
  const { conversationId } = useParams<{ conversationId?: string }>();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  /**
   * 诊断面板的第二个开关：通话中按 `Ctrl+Shift+D`（见文件头说明）。
   *
   * 只影响**显示**：不碰任何采集、合成、播放路径，开着面板的通话与关着的一模一样。
   */
  const [debugForced, setDebugForced] = useState(false);
  /** 「导出缓冲日志」的输出（面板里就地展示；见诊断面板的按钮） */
  const [bufferLogText, setBufferLogText] = useState('');
  const debugEnabled = searchParams.get('debug') === '1' || debugForced;

  useEffect(() => {
    const onKey = (e: KeyboardEvent): void => {
      if (!e.ctrlKey || !e.shiftKey) return;
      if (e.key !== 'D' && e.key !== 'd') return;
      e.preventDefault();
      setDebugForced((v) => !v);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const [conversationError, setConversationError] = useState<string | null>(null);
  /**
   * 地址栏里的会话 id 已经不存在（旧链接 / 会话被删 / 换过账号）→ 已自动换成新会话。
   * 单独一个状态而不是塞进 `conversationError`：后者会**藏掉"开始通话"按钮**。
   */
  const [sessionReplaced, setSessionReplaced] = useState(false);
  const [retryToken, setRetryToken] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  /**
   * 通话开始的墙钟时刻（`0` = 还没接通）。
   *
   * 用 state 而不是 ref：过程记录里"第几秒说的"要用它算，值变了必须触发一次重渲染，
   * 否则所有时间都会停在 `00:00`（计时器那条 effect 本来就在 setState，所以这里不额外增开销）。
   */
  const [startedAtMs, setStartedAtMs] = useState(0);
  const [muted, setMuted] = useState(false);
  const [hangingUp, setHangingUp] = useState(false);
  const [crisisCardDismissed, setCrisisCardDismissed] = useState(false);
  const [visionError, setVisionError] = useState<string | null>(null);
  const [visionBusy, setVisionBusy] = useState(false);
  /**
   * 网络断了（施工单 §S9.3 走查项："断网 5 秒 → 提示网络不稳定，正在重连…，已说出的字幕不丢"）。
   *
   * 为什么用浏览器的 `offline` / `online` 事件而不是等 WebSocket 报错：
   * 断网到 WS 真正超时之间有十几秒的静默期，用户那段时间看到的是一个"没反应"的界面。
   * `navigator.onLine` 是立刻的、并且是**真话**（它反映的就是系统网络状态）。
   *
   * ⚠️ 字幕**不清空**：已说出的内容留着，重连回来接着往上加。
   */
  const [networkDown, setNetworkDown] = useState(
    typeof navigator !== 'undefined' ? navigator.onLine === false : false,
  );

  useEffect(() => {
    const goOffline = (): void => setNetworkDown(true);
    const goOnline = (): void => setNetworkDown(false);
    window.addEventListener('offline', goOffline);
    window.addEventListener('online', goOnline);
    return () => {
      window.removeEventListener('offline', goOffline);
      window.removeEventListener('online', goOnline);
    };
  }, []);

  const bootstrapRef = useRef<Promise<string> | null>(null);
  /** 过程记录列表末尾的哨兵：新消息进来时滚到它（见下面的自动贴底 effect） */
  const transcriptEndRef = useRef<HTMLDivElement | null>(null);
  /** 过程记录滚动容器（"贴底"判断与"回到底部"按钮都用它） */
  const transcriptBoxRef = useRef<HTMLDivElement | null>(null);
  /**
   * 用户是否手动往上滚看过历史。
   *
   * 通话中**不断**有新内容进来，如果无条件贴底，用户想回看上一轮就会被反复拽回底部。
   * 规则（与聊天页一致的手感）：贴底时自动跟；用户自己滚上去了就停下来，
   * 直到他重新滚回底部（或点"回到底部"）。
   */
  const stickToBottomRef = useRef(true);
  /** 用户正在回看历史（脱离底部）→ 显示"回到底部"按钮 */
  const [showBackToBottom, setShowBackToBottom] = useState(false);
  /**
   * 通话开始的墙钟时刻（`0` = 还没接通）。
   *
   * **ref 是真源**；同名的 state 只是为了让"通话第几秒"这种派生文本在基准变化时
   * 重新渲染一次（见下面的计时 effect）。
   */
  const startedAtRef = useRef(0);

  /**
   * 建会话 / 恢复会话后要把地址 replace 成 `/call/<id>` —— **必须把地址栏里原有的
   * 查询串带上**，否则 `?debug=1` 会在那次 replace 里被悄悄抹掉：从 `/call?debug=1`
   * 进来，会话建好了、通话也正常，但诊断面板永远不出现（2026-10-11 实测）。
   *
   * 用 ref 而不是把 `searchParams` 放进 effect 依赖：那个 effect 的重跑条件只能是
   * `conversationId` / `retryToken`，地址栏多一个参数不该让它再建一个会话。
   */
  const keepQueryRef = useRef('');
  const search = searchParams.toString();
  keepQueryRef.current = search ? `?${search}` : '';

  /**
   * 通话期视觉。
   *
   * ⚠️ 它只是"把读数摆出来"：**不进风险、不写库**（施工单 §S7 / F7）。
   * 硬验收：开/关摄像头两次，落库的 `riskLevel` 完全一致。
   *
   * 2026-10-11 起它也多了一条本机三分类表情读数（用户开摄像头才加载模型）：
   * 该读数经 `vision.getFacialLiveState()` 随本轮消息作为**临时提示词**上行，
   * 但**仍然不进风险、不进融合权重** —— 上面的硬验收依然成立。
   */
  const vision = useCallVision({ ageGroup: null });

  /**
   * 开场预缓冲的**本次通话生效值**（毫秒，0~2000）。
   *
   * 口径（2026-10-11 用户真机试听后定档）：
   *   · **不带参数 = 用默认档 300ms**（`ttsPlayer` 的 `DEFAULT_PREBUFFER_MS`）；
   *   · `?prebuffer=0` = **显式关掉**（回到"收到第一块就播"的旧行为）；
   *   · `?prebuffer=600` = 试更强的档。
   *
   * 试档时最要紧的是"我知道自己在试哪一档"，所以这个值：
   *   1. 顶栏**常驻一个标签**显示它（不用开诊断面板）；
   *   2. 与实际生效值（`call.outputHealth.prebufferAppliedMs`）并排显示 ——
   *      "设定 600 / 生效 300"这种不一致一眼就能看出来，不会白试一通。
   * ⚠️ 只在**通话建立**时读取（`startCall` 里建 TTS 通道时用）：通话中途改地址栏无效，
   *    要换档必须刷新页面重开一通。
   */
  const DEFAULT_PREBUFFER_MS_UI = 300;
  const prebufferParam = searchParams.get('prebuffer');
  const requestedPrebufferMs = (() => {
    if (prebufferParam === null) return DEFAULT_PREBUFFER_MS_UI; // 不带参数 ⇒ 默认档
    const raw = Number(prebufferParam);
    if (!Number.isFinite(raw) || raw < 0) return DEFAULT_PREBUFFER_MS_UI;
    return Math.min(2000, Math.round(raw)); // 显式给值（含 0 = 关掉）照给
  })();
  /** 是否由地址栏显式指定（决定提示文案与"恢复默认"链接） */
  const prebufferExplicit = prebufferParam !== null;

  /**
   * **等 `speak` 计划的宽限期**（毫秒）：`?grace=250`。不带参数 = 默认 350。
   *
   * 为什么要旋钮：实测 `speak` 计划从未在宽限期内到达，350ms 基本是纯等待；
   * 但缩短它可能让首段更短、片段数变多 —— 用户要求**先验收预缓冲修复**，
   * 下一轮再做 350 → 250 的 A/B，所以这里只把旋钮做出来，**默认值不动**。
   * 对比要看：`首个 say` / `首段出声` / `本轮合成段数` / `缓冲·见底`（都在诊断面板里）。
   */
  const DEFAULT_GRACE_MS_UI = 350;
  const graceParam = searchParams.get('grace');
  const requestedGraceMs = (() => {
    if (graceParam === null) return DEFAULT_GRACE_MS_UI;
    const raw = Number(graceParam);
    if (!Number.isFinite(raw) || raw < 0) return DEFAULT_GRACE_MS_UI;
    return Math.min(1000, Math.round(raw)); // 显式给值（含 0）照给
  })();
  const graceExplicit = graceParam !== null;

  /**
   * 生成某一档的可复制链接（保留当前会话 id 与其它查询参数，只换 `prebuffer`）。
   *
   * 用途：试档时不用手打地址栏 —— 顶栏那个「预缓冲」标签点开就是几条链接。
   * `v = 300` 时**移除**该参数（回到默认档，链接更干净）；
   * `v = 0` 时写 `prebuffer=0` —— 那是"显式关掉"，与"不带参数"语义不同，必须写出来。
   */
  const prebufferUrl = useCallback(
    (v: number): string => {
      const next = new URLSearchParams(searchParams);
      if (v === DEFAULT_PREBUFFER_MS_UI) next.delete('prebuffer');
      else next.set('prebuffer', String(v));
      const qs = next.toString();
      const base = conversationId ? `/call/${conversationId}` : '/call';
      return `${window.location.origin}${base}${qs ? `?${qs}` : ''}`;
    },
    [conversationId, searchParams],
  );

  const call = useVoiceCall({
    conversationId,
    // 身份稳定（`useCallback([])` 内部读 ref），不会让轮次回调随 200ms 采样重建
    getFacialLiveState: vision.getFacialLiveState,
    // 开场预缓冲（见上面 requestedPrebufferMs 的说明）；0 = 既有行为
    prebufferMs: requestedPrebufferMs,
    // 等 speak 计划的宽限期；默认 350 不动，下一轮 A/B 才动它
    graceMs: requestedGraceMs,
  });
  const { phase, userLines, partialText, aiLines, revised, notice, voiceMetrics, ttsStatus, turn, transcript, ttsLedger } = call;

  // ── 对话过程（按轮）──────────────────────────────────────────────────────
  //
  // 用户口径（2026-10-11）："现在 AI 电话看不到完整的对话过程，我需要把每轮对话过程
  // 都像文字对话那样弄出来。" 于是这里把 hook 记下的时间轴按轮分组，带上"说话人 +
  // 通话第几秒 + 真实时刻"渲染成一个**能回看的消息流**（不是原来那种只显示当前字幕的
  // 扁平列表）。分组与"哪条属于哪一轮"的判据都在 `services/callTranscript.ts`（纯函数）。
  const transcriptGroups = groupTranscript(transcript);
  const hasTranscript = transcript.length > 0;

  /**
   * 新消息进来自动滚到底（**只在用户本来就在底部时**）。
   *
   * 依赖 `transcript` 的**引用**：每条消息都是新数组（纯函数返回新数组），所以内容
   * 一变就滚；`partialText` 也带上，因为"边说边出字"的中间结果同样在长高。
   */
  useEffect(() => {
    if (!stickToBottomRef.current) return;
    setShowBackToBottom(false);
    const el = transcriptEndRef.current;
    if (el) el.scrollIntoView({ block: 'end', behavior: 'smooth' });
  }, [transcript, partialText]);

  /**
   * 用户手动滚动：离底部还有一点距离就认为"他在回看历史"，停止自动贴底。
   *
   * 阈值 48px 是留给"平滑滚动尚未落到底"的余量 —— 否则每次自动滚动都会被自己的
   * 滚动事件判成"用户滚上去了"，然后自动贴底就永久失效了。
   */
  const handleTranscriptScroll = useCallback(() => {
    const box = transcriptBoxRef.current;
    if (!box) return;
    const distance = box.scrollHeight - box.scrollTop - box.clientHeight;
    const atBottom = distance <= 48;
    stickToBottomRef.current = atBottom;
    // 只在真正离开底部时显示"回到底部"：贴着底还挂一个按钮是纯干扰
    setShowBackToBottom(!atBottom);
  }, []);

  /** "回到底部"：重新开始自动贴底并立刻滚下去 */
  const handleBackToBottom = useCallback(() => {
    stickToBottomRef.current = true;
    setShowBackToBottom(false);
    transcriptEndRef.current?.scrollIntoView({ block: 'end', behavior: 'smooth' });
  }, []);

  // ── 会话生命周期：没有 `:conversationId` 就建一个，成功后 replace 到带 id 的地址 ──
  //
  // 用 ref 缓存这个 promise：StrictMode 会把 effect 跑两遍，
  // 若直接在 effect 里 await 建会话，就会建出**两个**空会话 ——
  // 一个挂着当前通话、另一个永远没人用，却留在用户的对话历史列表里。
  useEffect(() => {
    let alive = true;

    /** 建一个新会话并 replace 到带 id 的地址（恢复失败 / 地址里的 id 已失效时都走这里）。 */
    const bootstrapNew = (): void => {
      if (!bootstrapRef.current) {
        setConversationError(null);
        bootstrapRef.current = consultationApi.createConversation().then((res) => {
          const id = (res as { data?: { id?: string } })?.data?.id;
          if (!id) throw new Error('服务端没有返回会话 id');
          return id;
        });
      }
      bootstrapRef.current
        .then((id) => {
          if (alive) navigate(`/call/${id}${keepQueryRef.current}`, { replace: true });
        })
        .catch((err: unknown) => {
          if (!alive) return;
          bootstrapRef.current = null; // 允许重试
          setConversationError(err instanceof Error ? err.message : String(err));
        });
    };

    if (conversationId) {
      // 已经有 id 了，缓存作废（也顺带避免"退回 /call 又被送回旧会话"）；
      // 同时持久化，供刷新 / 误关标签后恢复。
      bootstrapRef.current = null;
      try {
        sessionStorage.setItem(CALL_SESSION_KEY, conversationId);
      } catch {
        /* 隐私模式等场景下不可写，忽略 */
      }

      // ⚠️ 地址栏里的 id **不一定还在**：旧链接 / 旧截图、会话被删、换过账号。
      //    不校验的话页面一切正常、能按"开始通话"、能说话，但**每一轮**都会在流里
      //    收到「对话不存在」（服务端 SSE 已 writeHead(200)，只能作为 error 事件下发），
      //    看着就是"AI 不回复"，而且麦克风已经申请了（2026-10-11 实测）。
      //    这里用同一次只读拉取做存在性校验 —— 与下面恢复 sessionStorage 的口径一致。
      //
      // ⚠️ 成功时刻意**不**把 `sessionReplaced` 复位：自动换会话之后地址会被 replace
      //    成新 id，这个 effect 会带着新 id 再跑一遍并成功 —— 若在这里复位，
      //    提示会在用户看到之前就被自己抹掉。
      consultationApi
        .getMessages(conversationId, 1)
        .catch((err: unknown) => {
          // 只处理"确实没有这个会话"。网络抖动/500 不在这里兜：那属于服务端问题，
          // 换一个会话并不能让它变好，而且会莫名其妙丢掉用户原本的会话。
          if (!alive) return;
          if ((err as { status?: number } | null)?.status !== 404) return;
          try {
            sessionStorage.removeItem(CALL_SESSION_KEY);
          } catch {
            /* 忽略 */
          }
          setSessionReplaced(true);
          bootstrapNew();
        });

      return () => {
        alive = false;
      };
    }

    // 没有 URL 参数：先看 sessionStorage 里有没有"上次没挂断"的会话。
    let stored: string | null = null;
    try {
      stored = sessionStorage.getItem(CALL_SESSION_KEY);
    } catch {
      /* 读不到就当没有 */
    }

    if (!stored) {
      bootstrapNew();
      return () => {
        alive = false;
      };
    }

    // ⚠️ 必须**先验证会话还在**再跳：id 可能对应一个已被删除的会话（用户清理过
    //    历史）或不属于当前用户，直接 navigate 会把手机会话带到一个空壳上 ——
    //    通话页拿不到会话、麦克风却已经申请了。服务端对不存在的会话返回 404，
    //    所以这里用一次消息拉取做存在性校验；失败就清键并回退到新建。
    consultationApi
      .getMessages(stored, 1)
      .then(() => {
        if (alive) navigate(`/call/${stored}${keepQueryRef.current}`, { replace: true });
      })
      .catch(() => {
        try {
          sessionStorage.removeItem(CALL_SESSION_KEY);
        } catch {
          /* 忽略 */
        }
        if (alive) bootstrapNew();
      });

    return () => {
      alive = false;
    };
  }, [conversationId, navigate, retryToken]);

  // ── 刷新恢复：挂断时清掉 sessionStorage，避免下次进 /call 被送回已结束的通话 ──
  useEffect(() => {
    if (phase === 'ended') {
      try {
        sessionStorage.removeItem(CALL_SESSION_KEY);
      } catch {
        /* 忽略 */
      }
    }
  }, [phase]);

  // 通话计时：从"接通"开始（`idle` 一旦离开就开始）
  const active = phase !== 'idle';
  useEffect(() => {
    if (!active) {
      // 回到 idle（重试前的一次失败交接）时清零，下一次接通重新开始计时
      startedAtRef.current = 0;
      setStartedAtMs(0);
      return;
    }
    if (startedAtRef.current === 0) {
      // ⚠️ 与过程记录**同一个时间基准**：气泡上的"通话第几秒"就是用它算的
      startedAtRef.current = Date.now();
      setStartedAtMs(startedAtRef.current);
    }
    const id = window.setInterval(() => {
      setElapsed(Math.floor((Date.now() - startedAtRef.current) / 1000));
    }, 1000);
    return () => window.clearInterval(id);
  }, [active]);

  /**
   * 静音 = 关掉**共享的那一条**麦克风轨道。
   *
   * 通话页自己不持设备引用（ASR 的 `'perception-voice'` 就是那一份），
   * 所以这里按名字取当下正在用的流，而不是自己再 `acquire` 一次 ——
   * 同一份流被两个消费者各持一次会给释放带来歧义（施工单 §S4 取舍 5）。
   *
   * **静音不等于挂断**：会话与安全监测继续；只是没有音频就没有文本，
   * 此时不做任何风险推断。
   */
  const handleToggleMute = useCallback(() => {
    const stream = mediaStreamManager.getStream('microphone');
    if (!stream) return;
    const next = !muted;
    stream.getAudioTracks().forEach((track) => {
      track.enabled = !next;
    });
    setMuted(next);
  }, [muted]);

  /** 挂断：冲刷最后一段定稿 → 等本轮落库 → 回文字会话。 */
  const handleHangup = useCallback(async () => {
    if (hangingUp) return;
    setHangingUp(true);
    try {
      // 摄像头先关（它不参与落库，先还设备）
      try {
        vision.stop();
      } catch {
        /* 忽略 */
      }
      await call.endCall();
    } finally {
      // 用 replace：避免用户按浏览器后退又回到通话页（会重新申请麦克风）
      navigate(conversationId ? `/chat/${conversationId}` : '/chat', { replace: true });
    }
  }, [call, conversationId, hangingUp, navigate, vision]);

  const handleStartCall = useCallback(() => {
    if (conversationError) return;
    void call.startCall();
  }, [call, conversationError]);

  /**
   * 摄像头开关。
   *
   * **失败绝不能让通话中断**（`video_call_build_plan.md` §3.6）：
   * 拒权限 / 被抢走 / 设备忙 → 明确提示 + 语音通话照常，且**绝不显示"已开启"**。
   */
  const handleToggleCamera = useCallback(async () => {
    if (visionBusy) return;
    setVisionBusy(true);
    setVisionError(null);
    try {
      if (vision.enabled || vision.detecting) {
        vision.stop();
      } else {
        await vision.start();
      }
    } catch (err) {
      setVisionError(
        err instanceof Error ? `摄像头不可用：${err.message}（已改为纯语音通话）` : '摄像头不可用（已改为纯语音通话）',
      );
    } finally {
      setVisionBusy(false);
    }
  }, [vision, visionBusy]);

  const handleRetryConversation = useCallback(() => setRetryToken((t) => t + 1), []);

  const handleCrisisGoChat = useCallback(() => {
    void handleHangup();
  }, [handleHangup]);

  // "接通中"也要留着按钮（用 loading 态显示进度）—— 否则一点下去按钮就消失，
  // 用户会以为点漏了而反复点。
  const showStartButton =
    !conversationError && (phase === 'idle' || phase === 'connecting' || phase === 'error');

  const cameraOn = vision.enabled || vision.detecting;
  const showCrisisCard = phase === 'crisis' && !crisisCardDismissed;

  return (
    <div
      data-page="voice-call"
      style={{
        position: 'fixed',
        inset: 0,
        display: 'flex',
        flexDirection: 'column',
        padding: '18px 26px 22px',
        boxSizing: 'border-box',
        background: 'linear-gradient(165deg, #fdf4ff 0%, #eef2ff 46%, #e6f4ff 100%)',
        overflow: 'hidden',
      }}
    >
      <style>{`
        @keyframes call-breathe { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.045); } }
        @keyframes call-pulse {
          0%, 100% { transform: scale(1); box-shadow: 0 16px 38px rgba(129,140,248,0.38), 0 0 0 0 rgba(99,102,241,0.30); }
          50% { transform: scale(1.06); box-shadow: 0 16px 38px rgba(129,140,248,0.38), 0 0 0 20px rgba(99,102,241,0); }
        }
        @keyframes call-blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
        /* 过程记录是新出现的滚动区：细滚动条不抢戏，hover 时才明显 */
        .vc-transcript-scroll { scroll-behavior: smooth; }
        .vc-transcript-scroll::-webkit-scrollbar { width: 6px; }
        .vc-transcript-scroll::-webkit-scrollbar-track { background: transparent; }
        .vc-transcript-scroll::-webkit-scrollbar-thumb { background: #d9d5ec; border-radius: 3px; }
        .vc-transcript-scroll:hover::-webkit-scrollbar-thumb { background: #c3bce4; }
      `}</style>

      {/* ── 顶栏 ─────────────────────────────────────────────── */}
      <header
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 16,
          flexShrink: 0,
        }}
      >
        <Button
          type="text"
          icon={<ArrowLeftOutlined />}
          loading={hangingUp}
          onClick={() => void handleHangup()}
          style={{ color: '#5a4a6a' }}
        >
          返回
        </Button>
        <Space size={8} wrap>
          {phase !== 'idle' && (
            <Tag
              style={{
                fontSize: 13,
                fontVariantNumeric: 'tabular-nums',
                background: '#fff',
                border: 'none',
                color: '#5a4a6a',
                padding: '4px 12px',
                borderRadius: 20,
              }}
            >
              {formatDuration(elapsed)}
            </Tag>
          )}
          <Tooltip title="语音识别与语音合成都在本机完成，通话音频不上传服务器">
            <Tag style={{ fontSize: 12, background: '#ffffffb0', border: 'none', color: '#7c7a99', borderRadius: 20 }}>
              本机识别 · 音频不出设备
            </Tag>
          </Tooltip>
          {/* ── 试档标识（2026-10-11 用户："我怎么知道我打的是哪一种"）──────────
              **常驻显示**当前是哪一档，不藏在诊断面板里；并且不只信地址栏 ——
              这里把"设定值"和"通道里实际生效值"并排显示：
              `设定 600 · 生效 300` 这种不一致一眼可见，不会白试一通。
              点一下弹出各档的可复制链接。
              ⚠️ 生效值来自 TTS 通道（点「开始通话」时按设定值建通道），
                 通话中途改地址栏不会生效 —— 换档必须刷新页面重开一通。 */}
          <Popover
            trigger="click"
            title="开场预缓冲（卡顿 ↔ 首声延迟的取舍）"
            content={
              <Space direction="vertical" size={6} style={{ maxWidth: 360 }}>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  默认档 300ms 是真机试听后定的。换档要刷新页面重开一通 ——
                  本通已按设定值建好通道，中途改地址栏无效。复制下面任一链接打开即可：
                </Text>
                {[
                  { v: DEFAULT_PREBUFFER_MS_UI, label: '默认 300ms' },
                  { v: 0, label: '关掉 0ms' },
                  { v: 600, label: '更强 600ms' },
                ].map(({ v, label }) => (
                  <Space key={v} size={8} align="start">
                    <Text style={{ fontSize: 12, minWidth: 96 }}>{label}</Text>
                    <Text
                      copyable={{ text: prebufferUrl(v) }}
                      style={{ fontSize: 12, color: '#6366f1', wordBreak: 'break-all' }}
                    >
                      {prebufferUrl(v)}
                    </Text>
                  </Space>
                ))}
                <Text type="secondary" style={{ fontSize: 11 }}>
                  判据看这里的读数：缓冲·见底（句内/跨句）与缓冲·水位 ——
                  卡顿变少、首声多等的时间可接受，就是合适的档。
                </Text>
              </Space>
            }
          >
            <Tag
              data-tag="prebuffer"
              style={{
                cursor: 'pointer',
                fontSize: 12,
                background: prebufferExplicit ? '#fff7e6' : '#ffffffb0',
                border: prebufferExplicit ? '1px solid #ffd591' : 'none',
                color: prebufferExplicit ? '#ad4e00' : '#7c7a99',
                borderRadius: 20,
                fontVariantNumeric: 'tabular-nums',
              }}
            >
              {`预缓冲 ${requestedPrebufferMs}ms`}
              {prebufferExplicit ? '（试档）' : '（默认）'}
              {call.outputHealth ? ` · 生效 ${call.outputHealth.prebufferAppliedMs}ms` : ' · 待接通'}
            </Tag>
          </Popover>
          {cameraOn && (
            <Tooltip title="画面在本机浏览器内分析，只用于「你在不在画面里」和光线提示；不参与风险评估、不上报">
              <Tag
                data-tag="vision"
                style={{ fontSize: 12, background: '#f0fdf4', border: 'none', color: '#3f6212', borderRadius: 20 }}
              >
                画面本机分析 · 不进风险
              </Tag>
            </Tooltip>
          )}
        </Space>
      </header>

      {/* ── 提示条 ───────────────────────────────────────────── */}
      {(conversationError || sessionReplaced || notice || voiceMetrics.asrError || visionError || networkDown) && (
        <div style={{ maxWidth: 620, width: '100%', margin: '10px auto 0', flexShrink: 0 }}>
          {networkDown && (
            <Alert
              data-alert="network"
              type="warning"
              showIcon
              style={{ borderRadius: 12, marginBottom: 8 }}
              message="网络不稳定，正在重连…"
              description="已经说出来的内容不会丢，恢复后接着聊。"
            />
          )}
          {sessionReplaced && (
            <Alert
              data-alert="session-replaced"
              type="warning"
              showIcon
              closable
              onClose={() => setSessionReplaced(false)}
              style={{ borderRadius: 12, marginBottom: 8 }}
              message="原来的通话会话已不存在，已为你新建一个"
              description="常见原因是打开了旧的通话链接，或那个会话已经被删除。"
            />
          )}
          {conversationError && (
            <Alert
              type="error"
              showIcon
              style={{ borderRadius: 12 }}
              message="无法建立通话会话"
              description={conversationError}
              action={
                <Button size="small" onClick={handleRetryConversation}>
                  重试
                </Button>
              }
            />
          )}
          {notice && (
            <Alert
              type={phase === 'crisis' ? 'warning' : 'info'}
              showIcon
              closable
              onClose={call.dismissNotice}
              style={{ borderRadius: 12, marginTop: conversationError ? 8 : 0 }}
              message={notice}
            />
          )}
          {voiceMetrics.asrError && (
            <Alert
              type="warning"
              showIcon
              style={{ borderRadius: 12, marginTop: 8 }}
              message={`语音识别不可用：${voiceMetrics.asrError}`}
            />
          )}
          {visionError && (
            <Alert
              type="warning"
              showIcon
              closable
              onClose={() => setVisionError(null)}
              style={{ borderRadius: 12, marginTop: 8 }}
              message={visionError}
            />
          )}
        </div>
      )}

      {/* ── 舞台：小安 + 状态 + **完整的对话过程** ────────────────────
          2026-10-11 起从"只显示当前字幕"改成"按轮铺开的完整过程"（用户口径：
          "把每轮对话过程都像文字对话那样弄出来"）。顶栏那一行（头像 + 状态 + 开始键）
          高度固定，剩下的整块留给消息流 —— 消息流自己滚，控制栏永远在屏幕里。 */}
      <main
        style={{
          flex: 1,
          minHeight: 0,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 8,
          padding: '6px 0 0',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexShrink: 0 }}>
          <div
            style={{
              width: 68,
              height: 68,
              borderRadius: '50%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: 'linear-gradient(140deg, #a5b4fc 0%, #818cf8 48%, #f0abfc 100%)',
              boxShadow: '0 12px 28px rgba(129,140,248,0.38)',
              animation: AVATAR_ANIMATION[phase],
              flexShrink: 0,
            }}
          >
            <PhoneFilled style={{ fontSize: 24, color: '#fff' }} />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <Title level={5} style={{ margin: 0, color: '#4a4a6a' }}>
              小安 · 心理陪伴
            </Title>
            <Space size={8} align="center">
              <span
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  background: PHASE_DOT[phase],
                  display: 'inline-block',
                  animation: phase === 'listening' ? 'call-blink 2s ease-in-out infinite' : undefined,
                }}
              />
              <Text data-phase={phase} style={{ fontSize: 14, color: '#5a4a6a' }}>
                {PHASE_LABEL[phase]}
              </Text>
            </Space>
          </div>
          {showStartButton && (
            <Button
              type="primary"
              size="large"
              icon={<PhoneFilled />}
              loading={phase === 'connecting'}
              disabled={phase === 'connecting'}
              onClick={handleStartCall}
              style={{ marginLeft: 6, borderRadius: 24, padding: '0 30px', height: 42 }}
            >
              {phase === 'error' ? '重试' : '开始通话'}
            </Button>
          )}
        </div>

        {/*
          ── 对话过程（按轮）──────────────────────────────────────────────
          这里回答的是"这一通电话聊了什么"，与"现在该显示什么字"分开：
          每一条都带讲话人、时刻与轮次；已说出的内容只增不删（断网重连、
          打断、改稿都不会清空）。数据来自 `useVoiceCall().transcript`。
        */}
        <div
          ref={transcriptBoxRef}
          onScroll={handleTranscriptScroll}
          data-transcript
          className="vc-transcript-scroll"
          style={{
            width: 'min(760px, 100%)',
            flex: 1,
            minHeight: 0,
            overflowY: 'auto',
            display: 'flex',
            flexDirection: 'column',
            gap: 10,
            padding: '6px 4px 12px',
          }}
        >
          {!hasTranscript && !partialText && (
            <div
              style={{
                flex: 1,
                minHeight: 160,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Text type="secondary" style={{ fontSize: 13, textAlign: 'center' }}>
                {active
                  ? '直接说话就行 —— 每一轮说了什么、小安怎么回应，都会按顺序留在这里'
                  : '开始通话后，每一轮对话都会像文字聊天一样留在这里，随时能往上翻'}
              </Text>
            </div>
          )}

          {transcriptGroups.map((group) => (
            <div key={`turn-${group.turn}`} style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {/* 轮次分隔：读长通话时一眼能看出"这是第几轮、什么时候说的" */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span style={{ flex: 1, height: 1, background: '#e6e3f5' }} />
                <span
                  data-turn={group.turn}
                  style={{
                    fontSize: 11,
                    color: '#a6a3bb',
                    whiteSpace: 'nowrap',
                    fontVariantNumeric: 'tabular-nums',
                  }}
                >
                  {turnLabel(group, startedAtMs)}
                </span>
                <span style={{ flex: 1, height: 1, background: '#e6e3f5' }} />
              </div>
              {group.messages.map((m) => (
                <div key={m.id} data-caption={m.speaker === 'user' ? 'user' : 'ai'}>
                  <CallBubble
                    v={m}
                    meta={`${relativeClock(m.tsMs, startedAtMs)} · ${formatClock(m.tsMs)}`}
                  />
                </div>
              ))}
            </div>
          ))}

          {/* 中间结果：边说边出字。刻意用浅色 + 斜体，与"已定稿"明确区分 */}
          {partialText && (
            <div data-caption="partial" style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
              <span style={{ fontSize: 11, color: '#a6a3bb', margin: '0 4px 3px 0' }}>正在说…</span>
              <div
                style={{
                  maxWidth: '86%',
                  padding: '9px 14px',
                  borderRadius: '14px 14px 4px 14px',
                  background: '#c7d2fe',
                  color: '#3730a3',
                  lineHeight: 1.65,
                  fontSize: 15,
                  fontStyle: 'italic',
                  wordBreak: 'break-word',
                }}
              >
                {partialText}
              </div>
            </div>
          )}

          {/* 贴到底的哨兵（自动滚动与"回到底部"都以它为目标） */}
          <div ref={transcriptEndRef} />
        </div>

        {/* 用户在回看历史时给一个明确的"回到底部"出口（否则新消息只会悄悄堆在下面）。
            位置贴右下角（也在视口边缘，不会跟居中的控制栏打架）：高 z-index 的
            视觉小窗在它上方、诊断面板盖在它上面，三者互不遮挡。 */}
        {showBackToBottom && (
          <Button
            size="small"
            data-action="back-to-bottom"
            onClick={handleBackToBottom}
            style={{
              position: 'fixed',
              right: 12,
              bottom: 10,
              zIndex: 16,
              borderRadius: 16,
              boxShadow: '0 6px 18px rgba(99,102,241,0.22)',
            }}
          >
            回到底部 ↓
          </Button>
        )}

        {/* ── 舞台下方的一句状态提示（都还没发生的事：能不能打断、AEC 窗口、危机切换）── */}
        {call.aiSpeaking && (
          <Text
            data-hint="barge-in"
            type="secondary"
            style={{ fontSize: 12, textAlign: 'center', flexShrink: 0 }}
          >
            随时可以打断 —— 直接开口说话（要连着说上一句，小安才确认不是在咳嗽），
            或者点【打断并发送】立刻抢话
          </Text>
        )}

        {call.forwardingPaused && (
          <Text
            data-hint="half-duplex-window"
            type="secondary"
            style={{ fontSize: 11, textAlign: 'center', flexShrink: 0 }}
          >
            正在出声，先让 AEC 收敛 0.4 秒
          </Text>
        )}

        {phase === 'crisis' && (
          <Text
            style={{
              fontSize: 13,
              color: '#a8600a',
              textAlign: 'center',
              maxWidth: 520,
              flexShrink: 0,
            }}
          >
            这一轮不再用语音，避免听到不合适的内容。你也可以回到文字对话继续。
          </Text>
        )}

        {/*
          ── 「哪一段没念出来」的如实标注（2026-10-11）─────────────────────
          打断/改稿/危机/挂断都会停掉本轮剩余音频，而字幕按"内容不藏"的既有策略
          仍会把剩下的字刷上屏。两者叠加 = 屏幕上是一整段、耳朵里只有前半句
          —— 用户报告的"回复和语音内容完全不匹配"就是它。
          这里不藏字，但明确说清楚：这一截**没有被念出来**。
          （过程记录里对应那条消息也会带上同一条标注，见 `CallBubble`。）
        */}
        {call.unspokenTail && (
          <Text
            data-hint="unspoken-tail"
            style={{
              fontSize: 12,
              color: '#a8600a',
              textAlign: 'center',
              maxWidth: 520,
              flexShrink: 0,
            }}
          >
            这一段没有说话（本轮语音已停），只有文字：{call.unspokenTail}
          </Text>
        )}
      </main>

      {/* ── 视觉小窗（只在开了摄像头时出现；**不进风险、不上报**）──────── */}
      {cameraOn && (
        <div
          data-panel="vision"
          style={{
            position: 'fixed',
            right: 22,
            bottom: 92,
            width: 176,
            borderRadius: 14,
            background: '#fff',
            boxShadow: '0 10px 30px rgba(99,102,241,0.20)',
            padding: 10,
            zIndex: 15,
          }}
        >
          <div
            ref={(el) => vision.attachPreview(el)}
            data-vision-preview
            style={{
              width: '100%',
              height: 118,
              borderRadius: 10,
              overflow: 'hidden',
              background: '#111827',
              marginBottom: 8,
            }}
          />
          <Space direction="vertical" size={2} style={{ width: '100%' }}>
            <Text data-vision="presence" style={{ fontSize: 12, color: '#3f3d56' }}>
              {presenceText(vision)}
            </Text>
            <Text data-vision="light" style={{ fontSize: 12, color: '#8c8aa7' }}>
              {lightText(vision.lightLevel)}
              {vision.brightness === null ? '' : `（${vision.brightness}/255）`}
            </Text>
            <Text data-vision="head" type="secondary" style={{ fontSize: 11, color: '#a6a3bb' }}>
              头部 {vision.headPitchDeg === null ? '—' : `${vision.headPitchDeg}°`} · 低头
              {Math.round(vision.headDownRatio * 100)}%
            </Text>
            {/* 本机三分类表情读数（用户开摄像头才加载模型）。措辞与画像页/对话页一致，
                只说"读到了什么表情"，不做任何心理状态断言。 */}
            <Text data-vision="face-emotion" style={{ fontSize: 11, color: '#8c8aa7' }}>
              {vision.faceEmotionStatus === 'ready'
                ? (vision.faceEmotion
                    ? `表情 ${FACE_EMOTION_LABELS_ZH[vision.faceEmotion.dominant]} ${Math.round(vision.faceEmotion.confidence * 100)}%`
                    : '表情 等待人脸')
                : vision.faceEmotionStatus === 'loading'
                  ? '表情 模型加载中'
                  : vision.faceEmotionStatus === 'unavailable'
                    ? '表情 不可用（不影响通话）'
                    : '表情 未开启'}
            </Text>
            <Text type="secondary" style={{ fontSize: 11, color: '#a6a3bb' }}>
              只给你自己看 · 不进风险 · 会作为 AI 回应的内部参考
            </Text>
          </Space>
        </div>
      )}

      {/* ── 控制栏 ───────────────────────────────────────────── */}
      <footer
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 14,
          flexWrap: 'wrap',
          flexShrink: 0,
        }}
      >
        <Button
          size="large"
          shape="round"
          icon={muted ? <AudioMutedOutlined /> : <AudioOutlined />}
          type={muted ? 'primary' : 'default'}
          danger={muted}
          disabled={phase === 'idle' || phase === 'connecting' || phase === 'ended'}
          onClick={handleToggleMute}
          style={{ height: 46, padding: '0 22px' }}
        >
          {muted ? '取消静音' : '静音'}
        </Button>

        <Tooltip
          title={
            cameraOn
              ? '关闭摄像头：视觉读数停止，通话继续（视觉从不参与风险判定）'
              : '打开摄像头：读取"你在不在画面里"、光线、本机表情读数；表情读数会作为 AI 回应的内部参考，但从不参与风险判定'
          }
        >
          <Button
            size="large"
            shape="round"
            data-action="toggle-camera"
            icon={<VideoCameraOutlined />}
            type={cameraOn ? 'primary' : 'default'}
            loading={visionBusy}
            disabled={phase === 'idle' || phase === 'connecting' || phase === 'ended'}
            onClick={() => void handleToggleCamera()}
            style={{ height: 46, padding: '0 22px' }}
          >
            {cameraOn ? '关闭摄像头' : '摄像头'}
          </Button>
        </Tooltip>

        <Tooltip
          title={
            '说完了就点它：小安立刻停下来，你已经说出口的话马上发出去，' +
            '不用等停顿判停（自动判停会等到 1.8 秒静音）'
          }
        >
          <Button
            size="large"
            shape="round"
            data-action="interrupt-send"
            icon={<SendOutlined />}
            /* 聆听时高亮：这是"我说完了，该你了"的主按钮 */
            type={phase === 'listening' ? 'primary' : 'default'}
            disabled={phase === 'idle' || phase === 'connecting' || phase === 'ended'}
            onClick={call.interruptAndSend}
            style={{ height: 46, padding: '0 22px' }}
          >
            打断并发送
          </Button>
        </Tooltip>

        <Button
          size="large"
          shape="round"
          type="primary"
          danger
          loading={hangingUp}
          icon={<PhoneFilled style={{ transform: 'rotate(135deg)' }} />}
          onClick={() => void handleHangup()}
          style={{ height: 46, padding: '0 26px' }}
        >
          挂断
        </Button>
      </footer>

      {/* ── 危机卡片（S6）：复用文字页那份文案与号码，**电话号码只上屏、绝不朗读** ── */}
      {showCrisisCard && (
        <div
          data-overlay="crisis"
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(24,20,40,0.55)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: 20,
            zIndex: 40,
          }}
        >
          <Card style={{ width: 'min(440px, 100%)', borderRadius: 16, maxHeight: '92vh', overflowY: 'auto' }}>
            <CrisisResourceCard
              compact
              context="语音已暂停，我们改用文字继续陪着你。"
              onPrimary={handleCrisisGoChat}
              onDismiss={() => setCrisisCardDismissed(true)}
            />
          </Card>
        </div>
      )}

      {/* ── `?debug=1` 或 Ctrl+Shift+D：只显示**实测**读数，不是演示数据 ── */}
      {debugEnabled && (
        <Card
          size="small"
          title="诊断（实测读数）"
          extra={
            <Space size={8}>
              {/* 「导出缓冲日志」：把逐块水位明细 + 见底现场贴进面板（同时写控制台）。
                  排查卡顿时要的就是"卡顿那一刻"的这几行，截图即可，不用开控制台。 */}
              <Button
                size="small"
                data-action="dump-buffer-log"
                onClick={() => setBufferLogText(call.dumpBufferLog(14))}
              >
                导出缓冲日志
              </Button>
              <Text type="secondary" style={{ fontSize: 11 }}>
                Ctrl+Shift+D 关闭
              </Text>
            </Space>
          }
          style={{
            position: 'fixed',
            right: 20,
            bottom: 20,
            width: 340,
            zIndex: 20,
            borderRadius: 14,
            boxShadow: '0 10px 30px rgba(99,102,241,0.18)',
            maxHeight: '72vh',
            overflowY: 'auto',
          }}
        >
          <Space direction="vertical" size={4} style={{ width: '100%' }}>
            <DebugRow label="状态" value={phase} />
            {/* 缓冲日志明细（点上面「导出缓冲日志」后出现在这里；同时写进控制台） */}
            {bufferLogText && (
              <pre
                data-buffer-log
                style={{
                  margin: 0,
                  maxHeight: 220,
                  overflow: 'auto',
                  background: '#f7f6fc',
                  borderRadius: 8,
                  padding: 8,
                  fontSize: 10,
                  lineHeight: 1.5,
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-all',
                }}
              >
                {bufferLogText}
              </pre>
            )}
            <DebugRow
              label="ASR"
              value={
                voiceMetrics.asrEngine
                  ? `${voiceMetrics.asrEngine}${voiceMetrics.asrError ? ` · ${voiceMetrics.asrError}` : ''}`
                  : '未启动'
              }
            />
            <DebugRow
              label="TTS"
              value={
                ttsStatus
                  ? `${ttsStatus.engine ?? '-'} · ${ttsStatus.sampleRate}Hz · ${ttsStatus.numSpeakers}音色`
                  : '未连接'
              }
            />
            {/* ── 载荷级音频账（2026-10-11 "两次都没有声音"排查加的）──────────────
                `say` = 我们发出去的合成请求句数；`PCM` = 服务端回回来的音频帧数/字节。
                判据（一眼可判）：
                  · say > 0 而 PCM 帧 = 0    → 服务端一个字节都没回（"没有声音"的直接证据）
                  · PCM 帧 > 0 且丢弃 ≈ 帧数 → 被我们自己的取消丢光（打断/改稿掐的）
                  · PCM 帧 > 0 且没听见       → 音频到了客户端，问题在播放层
                ⚠️ 这三行与时间读数（首声/首块/RTF）看的是两件事：时间回答"慢不慢"，
                   数量回答"**到底出没出声音**"。 */}
            <DebugRow
              label="TTS·say 发出"
              value={ttsLedger ? `${ttsLedger.sayFrames} 句` : '—'}
            />
            <DebugRow
              label="TTS·PCM 收到"
              value={
                ttsLedger
                  ? `${ttsLedger.pcmFrames} 帧 · ${Math.round(ttsLedger.pcmBytes / 1024)}KB` +
                    ` · 约 ${Math.round(ttsLedger.pcmBytes / 48)}ms`
                  : '—'
              }
            />
            <DebugRow
              label="TTS·丢弃/空音"
              value={
                ttsLedger
                  ? `${ttsLedger.droppedFrames} 帧 · 空音 ${ttsLedger.emptyAudioSegments} 句`
                  : '—'
              }
            />
            <DebugRow
              label="TTS·start/end"
              value={ttsLedger ? `${ttsLedger.startFrames} / ${ttsLedger.endFrames}` : '—'}
            />
            {/* ── "AI 偶尔说出乱话"的两条证据（2026-10-11）───────────────────────
                `文本不一致` = 服务端回的 `spoken_text` 与本次请求的文本不符
                （术语规范化那类预期差异也在里面，所以看下面的样例才知道是不是真异常）。
                `短单元` = 本轮有哪几段"不到 4 个字"被**当成独立单元**送去合成 ——
                delta 回退时手上只有「嗯」「那个」这种时就会出现，听感正是
                "先冒出一句莫名其妙的短话，然后才说正文"。 */}
            <DebugRow
              label="TTS·文本不一致"
              value={ttsLedger ? `${ttsLedger.textMismatch} 句` : '—'}
            />
            {ttsLedger && ttsLedger.textMismatchSamples.length > 0 && (
              <div
                data-tts-mismatch
                style={{
                  fontSize: 10,
                  lineHeight: 1.45,
                  background: '#fff7e6',
                  border: '1px solid #ffd591',
                  borderRadius: 8,
                  padding: 6,
                  color: '#8a4b00',
                  wordBreak: 'break-all',
                }}
              >
                {ttsLedger.textMismatchSamples.slice(-3).map((s) => (
                  <div key={`${s.seq}-${s.reason}`}>
                    seq={s.seq} {s.reason}
                    <br />请求「{s.requested}」
                    <br />念出「{s.spoken}」
                  </div>
                ))}
              </div>
            )}
            {turn && (
              <DebugRow
                label="短单元(&lt;4字)"
                value={
                  turn.shortUnits === 0 ? '0 个' : `${turn.shortUnits} 个 · 最短「${turn.shortestUnit}」`
                }
              />
            )}
            {/* 短单元守卫是否真的拦下过（见 useVoiceCall 的 MIN_FIRST_UNIT_CHARS）：
                为"是"说明本轮差点把两三个字单独念出来 —— 那正是"先说一句莫名其妙的
                话、然后才说正文"的机制。 */}
            {turn && (
              <DebugRow
                label="短单元已拦住"
                value={turn.shortHeld ? '是（首段太短，已等凑齐）' : '否'}
              />
            )}
            {/* 输出侧：AudioContext 是否真的在跑、有几个块交给了声卡。
                ⚠️ `suspended` 时音频排了程也不会响（自动播放策略），而所有时间读数都正常。 */}
            {/* ── 播放缓冲区（2026-10-11 "小卡顿"排查的主读数）────────────────────
                用户口径：先分清"卡在句子中间"还是"卡在两句之间"。这两个数就是答案：
                  · 句内见底 = 供音速度跟不上播放（缓冲被吃空）→ 该预缓冲 / 该优化推理
                  · 跨句见底 = 片段之间的调度空档 → 该提前发下一句的合成请求
                `水位` 是当前缓冲剩余毫秒：它一路贴 0 就是前者。 */}
            <DebugRow
              label="缓冲·见底"
              value={
                call.bufferStats
                  ? `句内 ${call.bufferStats.withinSentenceStarves} / 跨句 ${call.bufferStats.gapStarves}`
                    + ` · 累计 ${call.bufferStats.starveTotalMs}ms`
                  : '—'
              }
            />
            <DebugRow
              label="缓冲·水位"
              value={
                call.bufferStats
                  ? `${call.bufferStats.bufferMs} ms · 已入库 ${call.bufferStats.chunks} 块`
                  : '—'
              }
            />
            <DebugRow
              label="TTS·输出设备"
              value={
                call.outputHealth
                  ? `${call.outputHealth.ctxState} · 已排程 ${call.outputHealth.scheduledChunks} 块 · ${call.outputHealth.scheduledAudioMs}ms`
                    + (call.outputHealth.resumeAttempts > 0
                      ? ` · **被挂起 ${call.outputHealth.resumeAttempts} 次已自愈**`
                      : '')
                  : '—'
              }
            />
            <DebugRow
              label="TTS·预缓冲"
              value={
                call.outputHealth
                  ? `生效 ${call.outputHealth.prebufferAppliedMs}ms / 设定 ${call.outputHealth.prebufferMs}ms`
                  : '—'
              }
            />
            {/* ⚠️ 这一行是"每轮只生效一次"的**现场证据**（2026-10-11 用户要求核查的缺陷）：
                累计生效次数应当 **≤ 轮次**（`过程记录` 的轮数）；大于它说明又退了回去。
                `欠` = 本轮是否还欠一次（开场前应为"是"，排上音频后应为"否"）。 */}
            <DebugRow
              label="TTS·预缓冲生效"
              value={
                call.outputHealth
                  ? `${call.outputHealth.prebufferActivations} 次（应 ≤ 轮数）`
                    + ` · 本轮${call.outputHealth.prebufferPending ? '欠一次' : '已用掉'}`
                  : '—'
              }
            />
            <DebugRow
              label="麦克风实际"
              value={
                call.micSettings
                  ? `AEC=${call.micSettings.echoCancellation ? '开' : '关'} NS=${
                      call.micSettings.noiseSuppression ? '开' : '关'
                    } AGC=${call.micSettings.autoGainControl ? '开' : '关'} ${call.micSettings.sampleRate}Hz`
                  : '未启动'
              }
            />
            <DebugRow label="送帧暂停" value={call.forwardingPaused ? '是（半双工窗口）' : '否'} />
            {/* 宽限期（下一轮 A/B 的主变量）：当前值 + 实测"宽限触发"在第几毫秒。
                对比时看四项：首个 say / 首段出声 / 本轮合成段数 / 缓冲·见底。 */}
            <DebugRow
              label="宽限期"
              value={`${requestedGraceMs} ms${graceExplicit ? '（试档）' : '（默认）'}`}
            />
            <DebugRow label="定稿句数" value={String(userLines.length)} />
            {/* 过程记录的自检读数：**条数 / 轮数 / 已改稿 / 有未朗读**。
                它是"每轮都上屏了"的实测证据（屏幕上看不出漏记，这里能看出来）。 */}
            <DebugRow
              label="过程记录"
              value={`${transcript.length} 条 · ${transcriptGroups.length} 轮`}
            />
            <DebugRow
              label="改稿/未朗读"
              value={`${transcript.filter((m) => m.revised).length} / ${transcript.filter((m) => m.unspoken).length}`}
            />
            <DebugRow label="被丢弃定稿" value={String(call.discardedFinals)} />
            <DebugRow
              label="判为回声丢弃"
              value={`${call.droppedEchoes}${call.lastEchoReason ? ` · ${call.lastEchoReason}` : ''}`}
            />
            <DebugRow label="打断次数" value={String(call.bargeIns)} />
            <DebugRow label="手动打断并发送" value={String(call.manualSends)} />
            {/* 「卡不卡」的第一手读数：空洞 = 音频来晚了被听成断一下。
                空洞为 0 而首段出声很大 ⇒ 只是整体慢，不是卡。判据见 ttsPlayer.ts。 */}
            <DebugRow
              label="音频空洞"
              value={
                call.audioStats
                  ? `${call.audioStats.holes} 次 · 共 ${Math.round(call.audioStats.totalHoleMs)}ms` +
                    `${call.audioStats.maxHoleMs > 0 ? ` · 最长 ${Math.round(call.audioStats.maxHoleMs)}ms` : ''}` +
                    `${call.audioStats.holeRatio > 0 ? ` · 占比 ${(call.audioStats.holeRatio * 100).toFixed(1)}%` : ''}`
                  : '—'
              }
            />
            <DebugRow label="并发轮次峰值" value={String(call.maxConcurrentTurns)} />
            <DebugRow label="跳过号码片段" value={String(call.skippedPhoneUnits)} />
            <DebugRow label="复述用户未登记" value={String(call.parrotSkips)} />
            <DebugRow label="纯标点未合成" value={String(call.skippedUnspeakableUnits)} />
            <DebugRow label="服务端判风险" value={call.lastRisk ?? '—'} />
            {turn ? (
              <>
                {/* 「首段排队 / 首段到达 / 服务端首块」是 2026-10-11 加的**分辨读数**：
                    真机某轮首声 6159 ms，而离线实测合成本身只要 1.4~1.7 s ——
                    排队大 ⇒ 慢在"前面还有音频没播完"；排队 0 而服务端首块大 ⇒ 慢在合成。
                    三个数分开看，才知道该修哪一头（见 docs/voice_call_audio_stutter.md §1.2）。 */}
                <DebugRow
                  label="首段排队"
                  value={
                    turn.queueAheadAtFirstSayMs === null
                      ? '—'
                      : `${turn.queueAheadAtFirstSayMs} ms · 队列 ${turn.pendingUnitsAtFirstSay ?? 0} 句`
                  }
                />
                <DebugRow
                  label="首段到达"
                  value={turn.firstPcmAtMs === null ? '—' : `${turn.firstPcmAtMs} ms`}
                />
                <DebugRow label="首段合成" value={turn.firstPcmReadyMs === null ? '—' : `${turn.firstPcmReadyMs} ms`} />
                <DebugRow label="首段出声" value={turn.firstSoundMs === null ? '—' : `${turn.firstSoundMs} ms`} />
                <DebugRow
                  label="服务端首块"
                  value={
                    turn.firstServerFirstChunkMs === null
                      ? '—'
                      : `${Math.round(turn.firstServerFirstChunkMs)} ms` +
                        (turn.firstRtf === null ? '' : ` · RTF ${turn.firstRtf.toFixed(2)}`)
                  }
                />
                <DebugRow label="首个 delta" value={turn.firstDeltaMs === null ? '—' : `${turn.firstDeltaMs} ms`} />
                {/* ── 350ms 宽限的取证三格（2026-10-11 新增，只记录、不改常量）──
                    协议实测：`speak` 总在全部 delta 之后才到。所以"宽限到底等到了
                    计划没有"要靠 ②③ 两个时刻比出来；"这 350ms 花在哪"看 ①④ 的差。 */}
                <DebugRow
                  label="宽限触发"
                  value={turn.graceFiredAtMs === null ? '—' : `本轮第 ${turn.graceFiredAtMs} ms`}
                />
                <DebugRow
                  label="speak 到达"
                  value={turn.speakAtMs === null ? '未收到' : `本轮第 ${turn.speakAtMs} ms`}
                />
                <DebugRow
                  label="首个 say"
                  value={turn.firstSayAtMs === null ? '—' : `本轮第 ${turn.firstSayAtMs} ms`}
                />
                <DebugRow
                  label="delta→首个 say"
                  value={
                    turn.firstSayAtMs === null || turn.firstDeltaMs === null
                      ? '—'
                      : `${turn.firstSayAtMs - turn.firstDeltaMs} ms`
                  }
                />
                <DebugRow
                  label="宽限等到了计划?"
                  value={
                    turn.speakAtMs === null
                      ? '未收到计划（delta 驱动）'
                      : turn.graceFiredAtMs === null
                        ? '—'
                        : turn.speakAtMs <= turn.graceFiredAtMs
                          ? '是'
                          : '否（纯等待）'
                  }
                />
                <DebugRow
                  label="插话→静音"
                  value={turn.interruptStopMs === null ? '—' : `${turn.interruptStopMs} ms`}
                />
                <DebugRow
                  label="插话时刻"
                  value={turn.interruptMs === null ? '—' : `本轮第 ${turn.interruptMs} ms`}
                />
                <DebugRow label="整轮" value={turn.doneMs === null ? '—' : `${turn.doneMs} ms`} />
                <DebugRow
                  label="本轮音频空洞"
                  value={
                    turn.audioHoleCount === 0
                      ? '0 次'
                      : `${turn.audioHoleCount} 次 · 共 ${Math.round(turn.audioHoleMs)}ms`
                  }
                />
                {/* 「掐断未播」≠「音频空洞」：这一格是**我们主动丢掉**的句数
                    （打断 / 改稿 / 危机）。有数字不会弹错误；真的合成不出音频才会。 */}
                <DebugRow label="本轮掐断未播" value={`${turn.audioStoppedCount} 句`} />
                <DebugRow label="合成段数" value={String(turn.units)} />
                <DebugRow
                  label="首 delta 时通道"
                  value={
                    turn.ttsPresentAtFirstDelta === null
                      ? '—'
                      : `${turn.ttsPresentAtFirstDelta ? '已就绪' : '未就绪'} / 允许出声=${String(turn.voiceAllowedAtFirstDelta)}`
                  }
                />
                <DebugRow label="首 delta 原文" value={turn.firstDeltaText || '—'} />
                <DebugRow
                  label="meta.streaming"
                  value={turn.streaming === null ? '无 meta（非流式回退）' : String(turn.streaming)}
                />
                <DebugRow label="被改稿" value={String(turn.revised)} />
              </>
            ) : (
              <DebugRow label="轮次" value="尚未发生" />
            )}

            {/*
              ── 文本 ↔ 语音对账（A/B/C/D，2026-10-11）────────────────────
              排查"回复和语音不匹配"时**先看这里**：exceptions 直接写出第一处不一致
              在哪一层。A=页面显示，B=speak 计划，C=实际发出的 say，D=实际出声的
              spoken_text。有意跳过的号码/纯标点算预期例外，单独列。判据见
              `services/ttsSplit.ts` 与 `buildVoiceTextLedger`。
            */}
            {call.voiceLedger ? (
              <>
                <DebugRow label="对账·轮次" value={`#${call.voiceLedger.turnId} · 来源=${call.voiceLedger.source}`} />
                <DebugRow
                  label="对账·A 显示"
                  value={`${call.voiceLedger.shown.length} 字｜${call.voiceLedger.shown.slice(0, 18)}`}
                />
                <DebugRow
                  label="对账·B 计划"
                  value={
                    call.voiceLedger.planText === null
                      ? '无（未收到 speak）'
                      : `${call.voiceLedger.planText.length} 字 / ${call.voiceLedger.planSegments?.length ?? 0} 段`
                  }
                />
                <DebugRow
                  label="对账·C 发出"
                  value={`${call.voiceLedger.said.length} 字 / ${call.voiceLedger.saidUnits.length} 段`}
                />
                <DebugRow
                  label="对账·D 出声"
                  value={`${call.voiceLedger.spoken.length} 字 / ${call.voiceLedger.spokenUnits.length} 段`}
                />
                <DebugRow
                  label="对账·跳过"
                  value={
                    call.voiceLedger.skipped.length === 0
                      ? '无'
                      : call.voiceLedger.skipped
                          .map((s) => `${s.reason}:${s.text.slice(0, 8)}`)
                          .slice(0, 3)
                          .join(' ')
                  }
                />
                <DebugRow
                  label="对账·不一致"
                  value={
                    call.voiceLedger.exceptions.length === 0
                      ? '无（四份文本一致）'
                      : call.voiceLedger.exceptions.join('；')
                  }
                />
              </>
            ) : (
              <DebugRow label="对账" value="尚未发生" />
            )}
            <DebugRow label="视觉" value={cameraOn ? '已开启' : '未开启'} />
            {cameraOn && (
              <>
                <DebugRow label="采样帧率" value={`${vision.fps} Hz`} />
                <DebugRow
                  label="画面新鲜"
                  value={vision.frameFresh ? '是' : '否（读数不计）'}
                />
                <DebugRow label="在画面里" value={vision.present ? '是' : '否'} />
                <DebugRow
                  label="亮度"
                  value={vision.brightness === null ? '—' : `${vision.brightness}/255`}
                />
                <DebugRow
                  label="头部俯仰"
                  value={vision.headPitchDeg === null ? '—' : `${vision.headPitchDeg}°`}
                />
                <DebugRow
                  label="低头占比"
                  value={`${Math.round(vision.headDownRatio * 100)}%（>25°且>3s）`}
                />
              </>
            )}
          </Space>
        </Card>
      )}
    </div>
  );
}
