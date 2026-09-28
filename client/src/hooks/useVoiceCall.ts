/**
 * AI 陪伴通话的**轮次编排** —— 通话页的心脏（施工单 S4 / S5 / S6）
 *
 * ## 它负责什么
 *
 * ```
 *   麦克风 ──► 本地流式 ASR ──┬─ partial（每 ~330ms）─► 打断判据（S5）
 *                            │
 *                            └─ 定稿 ─┐
 *                                     ▼
 *                     sendChatMessageStream（走 Node 的 SSE，**不能**直连算法服务：F-1）
 *                                     │
 *             meta.streaming=false ───┼──► crisis：硬禁用语音 + 展示热线卡（F-3 / S6）
 *                                     │
 *             delta（已是过闸的完整句）┼──► AI 字幕 + 送 TTS 合成（F-2：不需要切句器）
 *                                     │
 *             revise（审计改稿）──────┼──► 停播 + 只改字幕，**不重说**
 *                                     │
 *             done ───────────────────┼──► 等播放队列排空 ──► 回到 listening
 *                                     └──► 有插话排队 → 顺序发下一轮（不并发）
 * ```
 *
 * ## 五个刻意的取舍（都写在对应位置的注释里）
 *
 * 1. **绝不 abort SSE（F-4）**：`consultation.service.ts:360` 收到 abort 后会回退到
 *    非流式 `smartChat`（**第二次完整 LLM 调用**，且不带中止信号），并在 `:432`
 *    落库一份用户从没听过的回复。所以挂断/打断时**不**调用 `abort()`：
 *    打断只清本地播放，本轮让它跑完，插话排队等 `done` 之后作为**下一轮**发出去。
 * 2. **打断靠 partial，不靠 final（G-1）**：partial 每 ~330ms 一次，
 *    final 要等端点 0.45–1.2 s，用 final 会觉得"插话没反应"。
 * 3. **回声过滤是文本级的（S5.2）**：阈值与时间窗都是 `docs/echo_calibration.md`
 *    里 40 组真实配对标定出来的，不是拍的。
 * 4. **revise 只改字幕不重说**：话已经播出去收不回来（`voice_call_plan.md` §4.4）。
 *    完整策略（比对最长公共前缀后决定"补说剩余部分"）是**产品判断**，
 *    施工单 §S4.5 标了 ⚠️ 需人工过眼；在有人拍板之前取保守的这半边。
 * 5. **设备归属**：ASR 走 `useVoiceAnalysis`（消费者名 `'perception-voice'`），
 *    它自己 `mediaStreamManager.acquire('microphone', ...)` —— 通话页**不再**单独持一份
 *    `'call'` 引用（同一份流被两个消费者各持一次是多余的，且给静音与释放增加歧义）。
 *    静音按钮改由页面直接切那一条共享轨道的 `enabled`。
 *
 * ## 为什么不用 `useAnxiety()`
 *
 * 那个 context 只在 `PatientLayout` 里挂载，而通话页是刻意的整屏独立页（不在
 * PatientLayout 内），`useAnxiety()` 在它会直接抛错。`useVoiceAnalysis` 是自包含 hook，
 * 正是这里该用的东西（施工单 §S4.2 的接线表默认了 context 可用，这一步已修正）。
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { sendChatMessageStream } from '../services/chatStream';
import { TtsChannel, type TtsStreamStatus } from '../services/ttsStream';
import { EchoFilter, ECHO_OVERLAP_THRESHOLD, scoreAgainstSpoken } from '../services/echoFilter';
import { useVoiceAnalysis } from './useVoiceAnalysis';
import type { VoiceAnalysis } from '../types/multimodal.types';

/** 通话状态 */
export type CallPhase =
  | 'idle'
  | 'connecting'
  | 'listening'
  | 'thinking'
  | 'speaking'
  /** 高风险：语音硬禁用，文字照常，并展示热线卡（S6） */
  | 'crisis'
  | 'ended'
  | 'error';

/** 一轮的实测耗时，供 S4.6/S5.5 闸门与 `?debug=1` 诊断面板读 */
export interface TurnMetrics {
  userText: string;
  /** 用户停口（定稿）→ 首个 `delta` 到达 */
  firstDeltaMs: number | null;
  /** 本轮第一段「请求 → 音频到手」—— **首声预算该看的就是这个**（施工单 §1 F-7） */
  firstPcmReadyMs: number | null;
  /** 本轮第一段真正出声（含排队） */
  firstSoundMs: number | null;
  /** 定稿 → 本轮 `done` */
  doneMs: number | null;
  /** 插话（partial）被判定为真打断 → **本地已经静音** 的耗时（S5.5 判据：< 200 ms） */
  interruptStopMs: number | null;
  /** 本轮开始 → 用户插话的时刻（信息性质：说明用户是在第几秒插进来的） */
  interruptMs: number | null;
  /** 送合成了几段 */
  units: number;
  /** 是否被安全审计改稿 */
  revised: boolean;
  /** `meta.streaming`；null 表示这一轮没收到 meta 事件 */
  streaming: boolean | null;
  /** 没收到 meta ⇒ 服务端走了非流式回退 */
  fellBack: boolean;
  /**
   * 第一个 `delta` 到达时，**出声通道是否已经就绪**。
   *
   * 加这两个字段是因为踩过一次"AI 字幕正常上屏、却一个 PCM 帧都没有"：
   * 当时所有环节都"看起来对"，只能靠这两个读数把范围缩到"是通道没就绪，
   * 还是通道就绪了但没送合成"。
   */
  ttsPresentAtFirstDelta: boolean | null;
  voiceAllowedAtFirstDelta: boolean | null;
  /** 第一个 `delta` 的原文（截断），用于确认切分输入长什么样 */
  firstDeltaText: string;
}

/** 挂断时等这一轮跑完的上限；超时也要让用户走得了 */
const END_CALL_WAIT_MS = 8000;

/**
 * 每次 TTS **开始出声**之后关掉送帧的时长（施工单 §S5.4）。
 *
 * 为什么是 400ms：AEC 需要一小段时间收敛到新的声学场景（"扬声器里突然出现人声"）。
 * 这段窗口里送出去的音频主要是 AI 自己的声音，让回声进来只会污染识别。
 *
 * ⚠️ 窗口内如果用户开口，开头会被切掉 —— 这是施工单 §S5.5 明确接受的范围
 * （"余 1 次落在 400ms 半双工窗口内，**这是设计接受范围内的**，要如实告知"）。
 */
const HALF_DUPLEX_WINDOW_MS = 400;

/**
 * 送 TTS 前的最后一道保险（施工单 §S6.3）：**含电话号码特征的片段一律不合成**。
 *
 * 为什么必须落成代码而不是靠自觉：实测（`docs/echo_calibration.md` §4.3）
 * `400 161 9995` 被念成「四百六十一九九九九五」—— TTS 把它当成一个 11 位数读，
 * 而不是分组电话号码。危机场景下念错热线号码的后果不可接受，所以号码**只以文字卡片呈现**。
 *
 * 比施工单里那条正则多收了几个字符：全角括号与破折号（中文排版里很常见），
 * 以及"7 位以上连续数字"（不含分隔符的写法）。
 */
export const PHONE_LIKE = /\d[\d\s\-—–()（）]{5,}\d|\d{7,}/;

/**
 * "这句话里有能发音的东西吗" —— 只有标点/符号的片段不送合成。
 *
 * 实测（网关日志）：AI 的回复里带书名号，`splitSpeakable` 会把 `「` `」` 切成
 * **独立的 1 字片段**送合成，而 sherpa 的词表里没有这两个符号：
 *
 * ```
 * Ignore OOV '「' / Ignore OOV '」'  →  Failed to convert 」 to token IDs
 * [TTS] 合成结果为空白音频：seq=3（文本 1 字），已跳过
 * ```
 *
 * 每一轮都会白发 2 次注定失败的请求（还各带回一次 `error` 回调）。
 * 汉字 / 字母 / 数字都算"能发音"；纯标点直接不入队。
 */
const SPEAKABLE = /[\p{Script=Han}\p{L}\p{N}]/u;

/**
 * 把一段 AI 文本切成"送合成的批次"（施工单 §S4.4）。
 *
 * 安全闸门已经按句切过，所以正常情况**不用切**；这里只处理两件事：
 *   1. 一段太长 → 在**逗号处**贪心打包成若干 ≤ MAX 字的块；
 *   2. **本轮第一段** → 只取极短的一小截，让音频尽早出声。
 *
 * ⚠️ MAX=36 是**显存**定的，不是延迟定的：实测一段 80 字的文本会让
 * `generate_voice_design` 在 RTX 5060 8GB 上抛 `CUDA out of memory`
 * （模型常驻 3.7 GB，合成期峰值随文本长度线性上涨；29 字实测安全、80 字必炸）。
 *
 * ⚠️ 也不要退回 VITS 时代的 30 字硬切：那会把完整语义单位劈成两段，
 * 而每多一段就多一次完整的合成等待，用户听到明显停顿。
 * 现在的做法是"逗号优先、打包到上限"，绝大多数句子原样成段。
 *
 * 顺带纠正一条过期结论：文件里"TTS 很慢（RTF~2.1）"的说法已经不成立。
 * 实测（2026-09-27，`_probe_tts_clock.js`，sherpa-onnx VITS）三段 15/14/20 字：
 * 合成 661 / 451 / 575 ms，对应音频 4229 / 3787 / 5241 ms —— **RTF≈0.12**，
 * 合成比播放快一个数量级。真正支配"出声时刻"的是**音频时长**，不是合成耗时；
 * 所以字幕若还是"首段出声时一次全刷"，就会比声音提前一整套音频的长度。
 */
function pack(text: string, limit: number): string[] {
  if (text.length <= limit) return [text];
  const parts = text.split(/(?<=[，,、：:])/).filter((s) => s.trim());
  const out: string[] = [];
  let buf = '';
  for (const p of parts) {
    if (p.length > limit) {
      // 单个小句本身就超长（无逗号的长句）——只能按字数硬切
      if (buf) { out.push(buf); buf = ''; }
      out.push(...(p.match(new RegExp(`.{1,${limit}}`, 'gs')) ?? [p]));
      continue;
    }
    if (buf && buf.length + p.length > limit) { out.push(buf); buf = p; }
    else buf += p;
  }
  if (buf) out.push(buf);
  return out;
}

export function splitSpeakable(unit: string, isFirst = false): string[] {
  // MAX=36 是**显存**定的（>60 字 OOM），也恰好让绝大多数对话句**整句成段不切断**。
  // 之前为降延迟砍到 18 + 首段硬切 4 字，实测把完整语义单位劈碎：
  // 用户听到"句子念不完整"，且段数翻倍 → 每段各等一次 RTF~1.8，间隙更多。
  // 韵律完整性 > 首字延迟，故回退。isFirst 不再特殊处理。
  void isFirst;
  return pack(unit, 36);
}

/** 一轮进行中的累计量（放 ref：TTS 回调与 SSE 回调不在同一作用域） */
interface TurnAccumulator {
  userText: string;
  startedAt: number;
  firstDeltaMs: number | null;
  firstPcmReadyMs: number | null;
  firstSoundMs: number | null;
  /** 本轮第一段的 seq —— 只有它算"首段"（见施工单 §1 F-7） */
  firstSeq: number | null;
  units: number;
  sawRevise: boolean;
  sawMeta: boolean;
  voiceAllowed: boolean;
  /** 本轮是否还在飞（**防并发轮次**） */
  active: boolean;
  /** 本轮 `done` 的时刻（相对 `startedAt`）；未结束为 null */
  doneMs: number | null;
  /** 插话被识别（partial 到达）→ 本地静音完成 的耗时（毫秒）—— **S5.5 的判据就是这个** */
  interruptStopMs: number | null;
  /** 插话 → 本轮开始 的耗时（毫秒）；信息性质，说明"用户在 AI 说到第几秒时插的话" */
  interruptMs: number | null;
  ttsPresentAtFirstDelta: boolean | null;
  voiceAllowedAtFirstDelta: boolean | null;
  firstDeltaText: string;
  /** 本轮字幕与音频的逐段对齐状态（见 {@link SubtitleSyncState}） */
  subtitle: SubtitleSyncState;
}

function newAccumulator(): TurnAccumulator {
  return {
    userText: '',
    startedAt: 0,
    firstDeltaMs: null,
    firstPcmReadyMs: null,
    firstSoundMs: null,
    firstSeq: null,
    units: 0,
    sawRevise: false,
    sawMeta: false,
    voiceAllowed: true,
    active: false,
    doneMs: null,
    interruptStopMs: null,
    interruptMs: null,
    ttsPresentAtFirstDelta: null,
    voiceAllowedAtFirstDelta: null,
    firstDeltaText: '',
    subtitle: newSubtitleSync(),
  };
}

/** 半双工窗口（见 {@link HALF_DUPLEX_WINDOW_MS}）之外，等 `speak` 计划的宽限时长。
 *
 * 服务端的 `speak` 计划必须在文本流完之后才算得出来，所以它通常出现在**最后一个
 * delta 之后几十毫秒**。而客户端原先"收到 delta 就立刻送合成"，于是同一段文本
 * 会被合成两遍（实测 delta 3867ms / speak 4061ms，相差 194ms）。这里对"第一个
 * delta 之后到计划到达之前"的送合成动作**扣住一小段时间**：
 *
 *   · 宽限期内收到计划 → 取消扣住的文本，完全由计划驱动（不重复、文本也已是定稿）；
 *   · 宽限期内没收到   → 按 delta 正常送合成（纯文字端、非流式回退、计划不可信时
 *                        的行为与改动前**完全一致**）。
 *
 * 取 350 ms：足够覆盖实测的 delta→speak 间隔（~200ms），又不至于让"没有计划"的
 * 路径明显变慢。
 */
const SPEAK_PLAN_GRACE_MS = 350;

/**
 * 字幕与音频的**逐段对齐**状态（通话语音模式下）。
 *
 * `full` 是本轮已到达的全文；`shownChars` 是已经上屏到的位置（对同一段文本的
 * 字符偏移）。`aiLines` 里最后一行就是"正在生长"的那一行，所以揭示一个字就是
 * 把最后一行替换成 `full.slice(0, shownChars)` —— 不需要额外的行/序号映射。
 */
interface SubtitleSyncState {
  /** 本轮已到达的增量文本（拼接后即模型全文） */
  full: string;
  /** 已经上屏到的字符数 */
  shownChars: number;
  /** 因找不到段边界而整段补屏的次数（诊断用：非 0 说明与 TTS 的切分粒度不一致） */
  unmatchedSegments: number;
}

function newSubtitleSync(): SubtitleSyncState {
  return { full: '', shownChars: 0, unmatchedSegments: 0 };
}
export { newSubtitleSync };

/**
 * 一个音频段开始出声时，算出字幕该揭示到第几个字。
 *
 * 用**字面子串**在"尚未上屏的剩余文本"里定位 `spokenText`：
 *   * 命中 → 揭示到该片段结尾，字幕与音频采样级同步；
 *   * 未命中（服务端改写过文本 / 播完的段又出现）→ 整段补屏，
 *     宁可多显示也**绝不把字丢掉**，并计数以便诊断。
 *
 * @param state 当前字幕游标状态（原地更新）
 * @param spokenText 这一段实际被念出来的原文（`TtsChannel.onStart` 的第 2 个参数）
 * @returns 新的"已上屏字符数"；与调用前相同表示这一句没有可揭示的内容
 */
export function advanceSubtitleOnSegment(state: SubtitleSyncState, spokenText: string): number {
  const seg = (spokenText || '').trim();
  if (!seg) return state.shownChars;
  const rest = state.full.slice(state.shownChars);
  const at = rest.indexOf(seg);
  if (at === -1) {
    state.unmatchedSegments += 1;
    // 找不到边界：不能再按字符推，把当前已到达的全部补上屏（安全侧：不吞字）
    state.shownChars = state.full.length;
    return state.shownChars;
  }
  state.shownChars += at + seg.length;
  return state.shownChars;
}

/** 剩余未上屏的原文（轮的收尾/改稿/打断时兑现，保证不丢字） */
export function subtitleRemainder(state: SubtitleSyncState): string {
  return state.full.slice(state.shownChars);
}

/** 麦克风轨道**实际生效**的约束（不是请求的约束 —— 见 S5.3） */
export interface MicSettings {
  echoCancellation: boolean;
  noiseSuppression: boolean;
  autoGainControl: boolean;
  sampleRate: number;
  channelCount: number;
  deviceId: string;
}

export interface UseVoiceCallOptions {
  conversationId?: string;
}

export interface UseVoiceCallResult {
  phase: CallPhase;
  /** 用户侧字幕：定稿后的每一句（临时的中间结果在 `partialText`） */
  userLines: string[];
  /** 用户正在说、还没定稿的中间结果（"边说边出字"） */
  partialText: string;
  /** AI 侧字幕：每个 `delta` 是一句（过闸的完整句） */
  aiLines: string[];
  /** AI 字幕是否被审计改稿整体替换过 */
  revised: boolean;
  notice: string | null;
  dismissNotice: () => void;
  voiceMetrics: VoiceAnalysis;
  ttsStatus: TtsStreamStatus | null;
  /** 不在可接受状态时被丢弃的定稿句数（如已挂断）；**不再**包含"AI 说话时" */
  discardedFinals: number;
  /** 被判为"AI 自己的回声"而丢弃的定稿句数（判据见 `echoFilter.ts`） */
  droppedEchoes: number;
  /** 最近一次回声判定的理由，供 `?debug=1` 与后续标定 */
  lastEchoReason: string | null;
  /** 触发过几次打断（用户真的插上话了） */
  bargeIns: number;
  /** 因"含号码特征"而被跳过合成（只上屏）的片段数 */
  skippedPhoneUnits: number;
  /** 纯标点片段（`「」`、破折号）被跳过合成的数量 —— 合成它们只会得到空白音频 */
  skippedUnspeakableUnits: number;
  /**
   * 因"AI 这句是在复述用户"而**没有**登记成回声指纹的句数。
   * 它不是错误计数 —— 是"这句放弃回声保护，换用户能插上话"的次数。
   */
  parrotSkips: number;
  /** 危机分支是否已经触发过（一旦触发，这一通电话后面都不再出声） */
  crisisTriggered: boolean;
  /**
   * 同时进行的轮次数**峰值**。
   *
   * 恒为 1 是 S5.5 的"无并发轮次"闸门。做成一个可读的读数而不是口头约定，
   * 是因为"插话排队"这条路径最容易被写错（在 `finish()` 里递归开下一轮，
   * 一不小心就会在上一轮还没收尾时并发出去）。
   */
  maxConcurrentTurns: number;
  /** 最近一轮服务端判定的风险等级（危机轮没有 meta，只能靠这个如实展示） */
  lastRisk: string | null;
  /** 麦克风实际生效的设置；未启动时为 null */
  micSettings: MicSettings | null;
  turn: TurnMetrics | null;
  startCall: () => Promise<void>;
  /** 挂断：冲刷最后一段定稿 → 等本轮跑完（有上限）→ 释放资源 */
  endCall: () => Promise<void>;
  /** AI 正在出声（UI 据此显示"小安在说话"而不是"暂停收音"） */
  aiSpeaking: boolean;
  /** 半双工窗口是否正开着（`?debug=1` 用；窗口内不向服务端送帧） */
  forwardingPaused: boolean;
}

export function useVoiceCall(options: UseVoiceCallOptions): UseVoiceCallResult {
  const { conversationId } = options;

  const [phase, setPhase] = useState<CallPhase>('idle');
  const [userLines, setUserLines] = useState<string[]>([]);
  const [aiLines, setAiLines] = useState<string[]>([]);
  const [revised, setRevised] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [ttsStatus, setTtsStatus] = useState<TtsStreamStatus | null>(null);
  const [discardedFinals, setDiscardedFinals] = useState(0);
  const [droppedEchoes, setDroppedEchoes] = useState(0);
  const [lastEchoReason, setLastEchoReason] = useState<string | null>(null);
  const [bargeIns, setBargeIns] = useState(0);
  const [skippedPhoneUnits, setSkippedPhoneUnits] = useState(0);
  /** 纯标点片段（「」——……）不送合成的计数；合成它们只会得到空白音频 */
  const [skippedUnspeakableUnits, setSkippedUnspeakableUnits] = useState(0);
  const [crisisTriggered, setCrisisTriggered] = useState(false);
  const [micSettings, setMicSettings] = useState<MicSettings | null>(null);
  const [forwardingPaused, setForwardingPaused] = useState(false);
  const [maxConcurrentTurns, setMaxConcurrentTurns] = useState(0);
  /**
   * 最近一轮 `done` 上报的风险等级（服务端判的，不是客户端猜的）。
   *
   * 危机路径**不会**发 `meta`（实测：危机轮无 meta、无 delta，直接 `done`），
   * 所以面板上只看 `meta.streaming` 会显示"无 meta"，看不出到底判成了什么。
   * 这个读数就是那件事的如实来源。
   */
  const [lastRisk, setLastRisk] = useState<string | null>(null);
  const [turn, setTurn] = useState<TurnMetrics | null>(null);
  /** 正在飞的轮次数（与 `maxConcurrentTurns` 一起构成"无并发轮次"的硬证据） */
  const activeTurnsRef = useRef(0);
  const maxConcurrentTurnsRef = useRef(0);

  /**
   * `phase` 的**真源**。
   *
   * state 更新是异步的，而定稿回调与 TTS 的 `onDrained` 可能在同一次事件里连续触发 ——
   * 只读 state 会读到上一帧的值，于是"AI 正在说话"被当成"正在聆听"，
   * 把回声当插话发出去。所有状态迁移都必须走 {@link applyPhase}。
   */
  const phaseRef = useRef<CallPhase>('idle');
  const applyPhase = useCallback((next: CallPhase) => {
    phaseRef.current = next;
    setPhase(next);
  }, []);

  const ttsRef = useRef<TtsChannel | null>(null);
  /** 送合成的段序号，整通电话内只增不减 */
  const ttsSeqRef = useRef(0);
  const accRef = useRef<TurnAccumulator>(newAccumulator());
  /** 本轮结束时兑现，供挂断等待 */
  const turnDoneRef = useRef<{ done: Promise<void>; resolve: () => void } | null>(null);
  /** 当前轮是否被审计改稿过（改稿后不再出声） */
  const mutedByReviseRef = useRef(false);
  /** 当前轮是否被用户插话打断过（打断后本轮剩下的句子不再出声） */
  const mutedByBargeInRef = useRef(false);
  /**
   * 字幕随音频**逐段**上屏的游标（替代原先的"首段出声时一次性全刷"）。
   *
   * ## 为什么要改（实测数据，2026-09-27，`_probe_tts_clock.js`）
   *
   * 原实现把整轮 delta 都压在这里，直到**第一段音频出声**才 `splice(0)` 全刷。
   * 本机实测（3 句：15/14/20 字）：
   *
   * ```
   *   delta 到达       0 / 81 / 135 ms      ← LLM 流式很快
   *   音频真正出声     670 / 1125 / 1708 ms ← 出声由「音频时长」支配
   *   音频时长         4229 / 3787 / 5241 ms（合成只需 451~661 ms，RTF≈0.12）
   * ```
   *
   * 全刷发生在 670 ms：于是**第 2、3 句字幕比它们自己的声音早 455 / 1038 ms**，
   * 用户看到整段文字"哗"地铺满屏幕、而声音才念到第一句 —— 这就是"语音和文本不同步"。
   * 注意不是"文字比声音早几秒"（合成并不慢），而是**整轮字幕被一次性地提前交付**。
   *
   * ## 改成什么
   *
   * 每个 delta 到达时只把文本**累积**进 `pending`，**按它将要被念到的位置**
   * 逐字揭示（`release`）。`TtsChannel.onStart(seq, spokenText)` 给的
   * `spoken_text` 是服务端实际送去合成的原文，用字面子串定位就能拿到
   * "这一句念到第几个字"，与音频采样级对齐（`seq` 的排序即出声顺序）。
   *
   * ⚠️ 安全侧：**一个字都不改文案、不丢字**。整轮 `done` / 改稿 / 插话 / 挂断时
   * 一律把剩余部分原样兑现（见 `flushSubtitleRemainder`），所以"字幕最终等于
   * 模型说的全文"这条不变量仍然成立；改变的只是**上屏时刻**。
   */
  /** 字幕随音频上屏的可用性判断（见 {@link SubtitleSyncState}） */
  const subtitleRef = useRef<SubtitleSyncState>(newSubtitleSync());
  /**
   * 出声通道当前是否真的可用。
   *
   * 字幕现在**由音频段驱动**上屏（逐段同步），所以必须显式判"到底有没有音频"：
   * 通道不可用时若还按"有声"处理，字幕就会一直压着，直到轮 `done` 才整段跳出来
   * —— 那是比不同步更糟的退化（用户中途什么都看不到）。
   */
  const ttsStatusRef = useRef<TtsStreamStatus | null>(null);
  /**
   * 本轮服务端下发的**合成计划**（`speak` 事件）。
   *
   * 有它时，TTS 与字幕都从这份**定稿文本**出发，而不是从流式 delta 出发 ——
   * 这是"听到的 = 看到的"的根保证：定稿规则（一轮只问一个等）已经在服务端
   * 生效完毕，不可能再出现"音频已经合成、文本才被删掉"的时序。
   *
   * 没有它时（纯文字端、非流式回退、计划与全文对不上、高风险）走原有路径：
   * 按 delta 自行切分。两条路径的**安全闸门与审计完全一致**，差别只是切分来源。
   */
  const speakPlanRef = useRef<{ text: string; segments: string[] } | null>(null);
  /** 已按 `speak` 计划送合成的段数（用于把计划里的句序映射到 seq） */
  const speakCursorRef = useRef(0);
  /**
   * 是否按 `speak` 计划送过合成了 —— **一旦送过，delta 就永久不许再送合成**。
   *
   * 这是"重复读句"的正面防线。实测（`_probe_call_e2e.js`，2026-09-27）出现过
   * 同一句话被合成**两遍**：
   *
   * ```
   *   SAY seq=1..4  ← 按 delta 切的（第一段 3867ms）
   *   SAY seq=5,6   ← 按 speak 计划切的（4061ms，同一段文本）
   * ```
   *
   * 两次相隔 194 ms：`speak` 是在文本流完之后才由服务端算出来的，客户端这里
   * **delta 与 speak 之间存在天然竞态**。两条来源同时往 TTS 队列里塞，
   * 用户就听到"这句话念两遍"。
   */
  const spokeFromPlanRef = useRef(false);
  /**
   * 等待 `speak` 计划期间被**扣住**的 delta 文本（见 {@link SPEAK_PLAN_GRACE_MS}）。
   */
  const deferredDeltaRef = useRef('');
  const deferTimerRef = useRef(0);
  /**
   * 轮的收尾：把还没随音频上屏的部分兑现。
   *
   * 四种收尾路径都必须调用它，否则会吞字：轮 `done`、审计改稿、用户插话打断、
   * 进入危机分支、挂断（见各调用点的注释）。
   */
  const flushSubtitleRemainder = useCallback((state?: SubtitleSyncState) => {
    const st = state ?? subtitleRef.current;
    const rest = subtitleRemainder(st);
    if (!rest) return;
    st.shownChars = st.full.length;
    setAiLines((prev) => (prev.length === 0 ? [st.full] : [...prev.slice(0, -1), st.full]));
  }, []);
  /** `runTurn` 的 ref —— 定稿回调需要调用它，而它定义在后面 */
  const runTurnRef = useRef<(text: string) => void>(() => undefined);

  // ── S5：回声过滤与打断 ──────────────────────────────────────────────────
  /**
   * 文本级回声过滤器。**只有它带时间**：窗口内（说话中 + 说完 2 秒）的 AI 文本
   * 才参与比对，阈值 0.40 来自 `docs/echo_calibration.md` 的实测标定。
   */
  const echoRef = useRef<EchoFilter | null>(null);
  if (!echoRef.current) echoRef.current = new EchoFilter();
  /** 本轮是否已经被插话打断过（一次打断只算一次） */
  const bargedInRef = useRef(false);
  /**
   * 插话队列：本轮还在飞的时候收到的用户定稿，按顺序等本轮 `done` 之后再发。
   * **不并发**（S5.5 闸门项），也**不丢弃**（那是用户真说的话）。
   */
  const pendingRef = useRef<string[]>([]);
  /** 一旦进入危机分支，这一通电话后面都不再出声 */
  const crisisRef = useRef(false);
  /**
   * 用户自己最近说过的话（最多 3 句）。
   *
   * 用来判断"AI 这一句是不是在复述用户"—— 见 {@link isParrotingUser}。
   */
  const userSaidRef = useRef<string[]>([]);
  /** 因为"是复述用户的话"而**没有**登记成回声指纹的句数（诊断用） */
  const [parrotSkips, setParrotSkips] = useState(0);

  const voiceSetForwardingRef = useRef<(on: boolean) => void>(() => undefined);
  const forwardingTimerRef = useRef(0);
  const voiceGetMicSettingsRef = useRef<() => MediaTrackSettings | null>(() => null);

  /**
   * 半双工窗口：TTS 开始出声 → 停送帧 400ms → 恢复（施工单 §S5.3/S5.4）。
   *
   * ⚠️ 停的是"送帧"，不是"麦克风"：`track.enabled = false` 会让服务端收到一段
   * 纯静音，端点检测照样触发，反而定稿出一个空段。
   */
  const openHalfDuplexWindow = useCallback(() => {
    if (phaseRef.current === 'ended') return;
    voiceSetForwardingRef.current(false);
    setForwardingPaused(true);
    window.clearTimeout(forwardingTimerRef.current);
    forwardingTimerRef.current = window.setTimeout(() => {
      forwardingTimerRef.current = 0;
      if (phaseRef.current === 'ended') return;
      voiceSetForwardingRef.current(true);
      setForwardingPaused(false);
    }, HALF_DUPLEX_WINDOW_MS);
  }, []);

  /**
   * AI 这一句是不是**在复述用户刚说的话**？
   *
   * ## 为什么必须排除这种情况（闸门实测抓到的）
   *
   * 回声过滤的原理是"我们知道自己说了什么"，但如果 AI 那句**本身就是用户的话**，
   * 这个指纹就没有区分力了 —— 用户继续把同一件事说下去时，那些字会与它高度重合，
   * 于是**用户真说的话被判成回声丢掉**。
   *
   * 这不是假想：安全审计的改稿会把回复改写成复述用户（实测那一轮 AI 说的是
   * 「你最近总是睡不着，心里特别烦。这是什么情况啊？」，与用户那句
   * 「我最近总是睡不着，心里特别烦」重叠 **92%**），结果那一轮**打断完全没有生效**。
   *
   * ## 为什么宁可漏掉这句的回声保护
   *
   * 两种错误的代价不对称：
   *   · 把用户的话当回声丢掉 → 用户的话**根本没进安全闸门**（危急的话也会被吞掉），
   *     而且用户看到的是"我说了它没反应"；
   *   · 把 AI 的回声当用户的话收下 → 多一轮莫名其妙的对话，安全上偏保守。
   * 与施工单 §S5.1 的原则一致：**判不准时放行用户的话**。
   */
  const isParrotingUser = useCallback((spokenText: string): boolean => {
    const said = userSaidRef.current;
    if (said.length === 0) return false;
    const s = scoreAgainstSpoken(spokenText, said);
    return s.containment || s.overlap >= ECHO_OVERLAP_THRESHOLD;
  }, []);

  // ── 状态收敛：队列空了 **且** 本轮结束，才回到聆听 ──────────────────────
  const settleToListening = useCallback(() => {
    if (accRef.current.active) return;
    const p = phaseRef.current;
    if (p === 'crisis' || p === 'ended' || p === 'error') return;
    if (ttsRef.current?.isSpeaking) return;
    applyPhase('listening');
  }, [applyPhase]);

  // ⚠️ `onDrained` 在"自然播完"与"被 stopAll 掐断"时都会来 ——
  //    所以这里只做幂等的状态收敛，不当成"播完了"的独有信号。
  //    S5 起还兼一个职责：把**被打断的那半句**从回声窗口里退场
  //    （`stopAll` 不会触发播放器的 `onEnd`，见 EchoFilter 的注释）。
  const handleDrained = useCallback(() => {
    echoRef.current?.retireAll(performance.now());
    settleToListening();
  }, [settleToListening]);

  // ── 打断：partial 判定（S5.4 / G-1）──────────────────────────────────────
  const handleVoicePartial = useCallback(
    (text: string) => {
      const t = (text || '').trim();
      // 只有"本轮还在飞"的时候收到的语音才可能是打断。其余时刻 partial 只是上屏数据。
      if (!accRef.current.active || !t) return;
      if (phaseRef.current === 'ended') return;

      const now = performance.now();
      const verdict = echoRef.current?.judge(t, now);
      if (verdict?.echo) {
        // AI 自己的声音被听到了 —— 既不是插话，也不该上屏
        setLastEchoReason(`partial：${verdict.reason}`);
        return;
      }
      if (bargedInRef.current) return;

      // ── 真打断：G-2 —— 先清本地队列，再谈别的，绝不等服务端回执 ──
      bargedInRef.current = true;
      // 本轮剩下的句子**不再出声**：已经"立刻静音"了，几百毫秒后又自己说起来
      // 会让人以为打断失败了。字幕照旧追加（内容不藏），回答由队列里的插话开新一轮。
      mutedByBargeInRef.current = true;
      // 打断前还没随音频上屏的字幕不会再有音频了，立刻兑现（内容不藏）
      flushSubtitleRemainder(accRef.current.subtitle);
      accRef.current.interruptMs = Math.round(now - accRef.current.startedAt);
      // S5.5 的判据：**从"判定为真打断"到"本地已经不响了"**（这一段全是客户端可控的），
      // 要求 < 200ms。`stopAll()` 同步掐断，不 await、不回调。
      const stopAt = performance.now();
      ttsRef.current?.stopAll();
      accRef.current.interruptStopMs = Math.round(performance.now() - stopAt);
      echoRef.current?.retireAll(now);
      setBargeIns((n) => n + 1);
      // 相位回到聆听：这样"边说边出字"的临时候选会显示出来，
      // 用户能立刻看到自己的话被听见了（本轮 SSE 仍在跑，F-4）
      applyPhase('listening');
    },
    [applyPhase],
  );

  // ── 定稿 = 一轮的起点（或插话队列的入口）────────────────────────────────
  const handleVoiceFinal = useCallback((text: string) => {
    const t = text.trim();
    if (!t) return;

    const p = phaseRef.current;
    // 没接通 / 已挂断：确实无处安放，如实计数
    if (p === 'idle' || p === 'connecting' || p === 'ended' || p === 'error') {
      setDiscardedFinals((n) => n + 1);
      return;
    }

    // ① 自回声过滤（**放在最前面**）。
    //    最危险的窗口恰恰是"AI 刚说完、相位已经回到 listening"那一秒：
    //    此时麦克风收进来的多半是 AI 自己的声音，不清掉就会落库一条假的用户消息。
    const verdict = echoRef.current?.judge(t, performance.now());
    if (verdict?.echo) {
      setDroppedEchoes((n) => n + 1);
      setLastEchoReason(`${verdict.reason}${verdict.matched ? `｜${verdict.matched.slice(0, 20)}` : ''}`);
      return;
    }

    setUserLines((prev) => [...prev, t]);
    // 记住用户自己说了什么：AI 复述用户时，那句不能当回声指纹（见 isParrotingUser）
    userSaidRef.current = [t, ...userSaidRef.current].slice(0, 3);

    // ② 本轮还在飞 ⇒ 插话：排队，等 `done` 之后作为**下一轮**发（F-4，不并发）
    if (accRef.current.active) {
      pendingRef.current.push(t);
      return;
    }
    // ③ 正常路径：立刻开一轮
    runTurnRef.current(t);
  }, []);

  // ── 语音输入：直接复用既有 hook（它就是"聊天页语音录入"那条链路）────────
  const voice = useVoiceAnalysis(null, null, handleVoiceFinal, handleVoicePartial);  const voiceStopRef = useRef(voice.stop);
  voiceStopRef.current = voice.stop;
  const voiceStartRef = useRef(voice.start);
  voiceStartRef.current = voice.start;
  const voiceMetricsRef = useRef(voice.metrics);
  voiceMetricsRef.current = voice.metrics;
  voiceSetForwardingRef.current = voice.setForwarding;
  voiceGetMicSettingsRef.current = voice.getMicSettings;

  /**
   * 把本轮累计量发布成 `turn` 状态。
   *
   * ⚠️ **不能在 `done` 时只发布一次**：首段的「请求→到手 / 真正出声」这两个读数
   * 是在**播放器真的开始播之后**才产生的，而那时 `done` 通常已经过去了
   * （合成在 SSE 还没结束时就已经入队并开播）。只在 `done` 快照一次，
   * 这两个格子会永远是 `—` —— 实测就是这样：PCM 明明到了 2 帧、`say` 发了 3 次，
   * 面板上"首段合成"却是空的。所以 `onStart` 里也要再发布一次。
   */
  const publishTurn = useCallback(() => {
    const acc = accRef.current;
    setTurn({
      userText: acc.userText,
      firstDeltaMs: acc.firstDeltaMs,
      firstPcmReadyMs: acc.firstPcmReadyMs,
      firstSoundMs: acc.firstSoundMs,
      doneMs: acc.doneMs,
      interruptStopMs: acc.interruptStopMs,
      interruptMs: acc.interruptMs,
      units: acc.units,
      revised: acc.sawRevise,
      streaming: acc.sawMeta ? acc.voiceAllowed : null,
      fellBack: !acc.sawMeta,
      ttsPresentAtFirstDelta: acc.ttsPresentAtFirstDelta,
      voiceAllowedAtFirstDelta: acc.voiceAllowedAtFirstDelta,
      firstDeltaText: acc.firstDeltaText,
    });
  }, []);

  /**
   * 送合成 —— **S6.3 的最后一道保险就这里**。
   *
   * 含号码特征的片段只上屏、不合成。实测 `400 161 9995` 会被念成
   * 「四百六十一九九九九五」，危机场景下念错热线号码不可接受。
   */
  const ttsSay = useCallback((seq: number, unit: string) => {
    const tts = ttsRef.current;
    if (!tts) return;
    if (!SPEAKABLE.test(unit)) {
      // 纯标点（「」——……）：合成只会得到空白音频，直接不入队
      setSkippedUnspeakableUnits((n) => n + 1);
      return;
    }
    if (PHONE_LIKE.test(unit)) {
      console.warn('[通话] 片段含号码特征，跳过合成（只上屏）:', unit.slice(0, 30));
      setSkippedPhoneUnits((n) => n + 1);
      return;
    }
    tts.say(seq, unit);
  }, []);

  // ── 一轮：发文本 → 收流 → 出声 ─────────────────────────────────────────
  const runTurn = useCallback(
    (userText: string) => {
      const convId = conversationId;
      if (!convId || accRef.current.active) return;

      const acc = newAccumulator();
      acc.userText = userText;
      acc.startedAt = performance.now();
      acc.active = true;
      accRef.current = acc;
      mutedByReviseRef.current = false;
      mutedByBargeInRef.current = false;
      bargedInRef.current = false;
      acc.subtitle = newSubtitleSync();
      speakPlanRef.current = null;
      speakCursorRef.current = 0;
      spokeFromPlanRef.current = false;
      deferredDeltaRef.current = '';
      window.clearTimeout(deferTimerRef.current);
      deferTimerRef.current = 0;
      // "无并发轮次"的硬证据：进一轮 +1，收一轮 -1，峰值恒为 1
      activeTurnsRef.current += 1;
      if (activeTurnsRef.current > maxConcurrentTurnsRef.current) {
        maxConcurrentTurnsRef.current = activeTurnsRef.current;
        setMaxConcurrentTurns(activeTurnsRef.current);
      }

      let resolveDone!: () => void;
      const done = new Promise<void>((r) => {
        resolveDone = r;
      });
      turnDoneRef.current = { done, resolve: resolveDone };

      const finish = (): void => {
        acc.active = false;
        acc.doneMs = Math.round(performance.now() - acc.startedAt);
        activeTurnsRef.current = Math.max(0, activeTurnsRef.current - 1);
        publishTurn();
        resolveDone();
        settleToListening();
        // 插话排队：**顺序**发下一轮，绝不并发（S5.5 闸门项）
        const next = pendingRef.current.shift();
        if (next && phaseRef.current !== 'ended') runTurnRef.current(next);
      };

      applyPhase('thinking');

      sendChatMessageStream(convId, userText, {
        onMeta: (meta) => {
          acc.sawMeta = true;
          // F-3：零新增阈值 —— 高风险时 `streaming` 恒为 false
          acc.voiceAllowed = meta.streaming && !crisisRef.current;
          if (!meta.streaming) enterCrisis(acc);
        },
        onSpeak: (plan) => {
          // 定稿合成计划：**按它合成、也按它上屏**。
          //
          // ⚠️ 为什么字幕在这里就整段上屏，而不是再等音频逐段揭示：
          // 服务端的 `delta` 是**闸门放行的原文**，它可能比定稿长（例如模型一轮问了
          // 两个问题，定稿按"只能问一个"删掉尾部整句）。实测第 2 轮：delta 共 51 字，
          // 定稿 37 字。若字幕仍按"音频念到哪儿显示到哪儿"，被删掉的那 14 字就永远不会
          // 上屏 —— 服务端下了 `speak`，客户端却没显示，等于**丢字**。
          //
          // 现在以 `plan.text`（服务端定稿全文，与 `done.text` 逐字相同）为唯一来源：
          // 文本从收到计划起就是完整的；音频按 `plan.segments` 顺序出声。
          // 代价是文字先于声音出现（已知的、可接受的取舍），换来的是
          // "绝不丢字 + 念出来的一定是屏幕上这段"。
          speakPlanRef.current = plan;
          speakCursorRef.current = 0;
          // 计划到了：撤掉"按 delta 送合成"的宽限定时器与已扣住的文本 ——
          // 本轮的字幕与音频从这一刻起**只由计划驱动**（见 spokeFromPlanRef）。
          window.clearTimeout(deferTimerRef.current);
          deferTimerRef.current = 0;
          deferredDeltaRef.current = '';
          acc.subtitle.full = plan.text;
          acc.subtitle.shownChars = plan.text.length;
          setAiLines([plan.text]);
          const willSpeak = acc.voiceAllowed && !mutedByReviseRef.current
            && !mutedByBargeInRef.current && !crisisRef.current
            && ttsStatusRef.current?.available === true;
          if (!willSpeak) return;   // 纯文字模式：字幕已在上方上屏，没有音频要送
          spokeFromPlanRef.current = true;
          // 立即按计划送合成 —— 不再等 delta，也不再按 delta 切分
          for (const seg of plan.segments) {
            for (const unit of splitSpeakable(seg, acc.units === 0)) {
              if (!unit.trim()) continue;
              acc.units += 1;
              ttsSeqRef.current += 1;
              const seq = ttsSeqRef.current;
              if (acc.firstSeq === null) acc.firstSeq = seq;
              speakCursorRef.current += 1;
              ttsSay(seq, unit);
            }
          }
        },
        onDelta: (text) => {
          if (acc.firstDeltaMs === null) {
            acc.firstDeltaMs = Math.round(performance.now() - acc.startedAt);
          }
          if (acc.ttsPresentAtFirstDelta === null) {
            acc.ttsPresentAtFirstDelta = ttsRef.current !== null;
            acc.voiceAllowedAtFirstDelta = acc.voiceAllowed;
            acc.firstDeltaText = text.slice(0, 60);
          }
          // 有合成计划时：delta 只用来做"首字延迟"之类的观测，**不再**驱动字幕与合成
          // —— 文本与音频都已由 plan 定死（见 speakPlanRef 的注释）。
          if (speakPlanRef.current) return;

          // 无计划（纯文字端 / 非流式回退 / 计划不可信）：按 delta 自行切分。
          // 语音模式下字幕仍**随音频逐段**上屏（不再在 delta 到达时抢跑）。
          const willSpeak =
            acc.voiceAllowed &&
            !mutedByReviseRef.current &&
            !mutedByBargeInRef.current &&
            !crisisRef.current &&
            ttsStatusRef.current?.available === true;
          if (willSpeak) {
            // 只累积；真正的上屏由 TtsChannel.onStart（这一段音频真的出声）触发
            acc.subtitle.full += text;
          } else {
            setAiLines((prev) => [...prev, text]);
          }

          if (!willSpeak) return;

          // ⚠️ 按 `speak` 计划送过合成之后，delta **永久**不许再送 —— 否则同一段
          // 文本会被合成两遍，用户听到"这句话念两遍"（见 spokeFromPlanRef）。
          if (spokeFromPlanRef.current) return;

          const flushByDelta = () => {
            deferTimerRef.current = 0;
            const pending = deferredDeltaRef.current;
            deferredDeltaRef.current = '';
            if (!pending) return;
            for (const unit of splitSpeakable(pending, acc.units === 0)) {
              if (!unit.trim()) continue;
              acc.units += 1;
              ttsSeqRef.current += 1;
              const seq = ttsSeqRef.current;
              if (acc.firstSeq === null) acc.firstSeq = seq;
              ttsSay(seq, unit);
            }
          };

          // 第一个 delta 到达后先扣住一小段时间等 `speak` 计划；之后（宽限期已过、
          // 计划仍没来）就回到原来的"边到边送"，延迟不变。
          if (deferTimerRef.current === 0 && !spokeFromPlanRef.current) {
            deferredDeltaRef.current += text;
            deferTimerRef.current = window.setTimeout(flushByDelta, SPEAK_PLAN_GRACE_MS);
            return;
          }
          if (deferredDeltaRef.current) {
            deferredDeltaRef.current += text;
            return;
          }
          for (const unit of splitSpeakable(text, acc.units === 0)) {
            if (!unit.trim()) continue;
            acc.units += 1;
            ttsSeqRef.current += 1;
            const seq = ttsSeqRef.current;
            if (acc.firstSeq === null) acc.firstSeq = seq;
            // ⚠️ 非阻塞入队：绝不能 await 合成结果，否则整轮被首句合成卡住
            ttsSay(seq, unit);
          }
        },
        onRevise: (text) => {
          // 已经播出去的声音收不回来：**停播 + 只改字幕，不重说**（文件头取舍 4）
          acc.sawRevise = true;
          mutedByReviseRef.current = true;
          // 改稿文本整体取代已到达的流式文本：游标重置到全文
          acc.subtitle.full = text;
          acc.subtitle.shownChars = text.length;
          ttsRef.current?.stopAll();
          echoRef.current?.retireAll(performance.now());
          setAiLines([text]);
          setRevised(true);
        },
        onDone: (payload) => {
          const risk = String((payload as { riskLevel?: string }).riskLevel ?? '');
          setLastRisk(risk || null);
          const isCrisis =
            (payload as { isCrisis?: boolean }).isCrisis === true || risk === 'crisis';
          if (isCrisis) enterCrisis(acc);
          // 轮已结束：不可能再有音频段来揭示剩余字幕了，一次兑现，绝不吞字。
          // （`finish()` 之前调用：`finish` 会 publishTurn 并把本轮标成非活跃。）
          flushSubtitleRemainder(acc.subtitle);
          finish();
        },
        onError: (err) => {
          setNotice(err.errorMessage || '这一轮没能生成回复。麦克风仍然开着，你可以再说一次。');
          finish();
        },
      });
      // 刻意不保存/调用 handle.abort()：见文件头取舍 1（F-4）

      /** 危机分支：立刻静音 + 会话切纯文字（S6.2） */
      function enterCrisis(turnAcc: TurnAccumulator): void {
        turnAcc.voiceAllowed = false;
        crisisRef.current = true;
        setCrisisTriggered(true);
        // 1) 立刻 stopAll + 中止在途合成
        ttsRef.current?.stopAll();
        echoRef.current?.retireAll(performance.now());
        // 缓冲的字幕不会再有音频了，立刻兑现
        flushSubtitleRemainder(turnAcc.subtitle);
        // 2) 会话切纯文字；麦克风保持可用（用户还要说话）
        applyPhase('crisis');
        setNotice('这一轮涉及较高风险，已暂停语音，继续用文字陪伴你。');
      }
    },
    [applyPhase, conversationId, publishTurn, settleToListening, ttsSay],
  );
  runTurnRef.current = runTurn;

  // ── 开始 / 结束 ────────────────────────────────────────────────────────
  const startingRef = useRef(false);

  const startCall = useCallback(async () => {
    if (!conversationId) return;
    // ⚠️ 同步重入锁：连点时 React 还没重渲染，只靠 `loading` 拦不住第二次，
    //    会建出第二个 TtsChannel 并覆盖 ref（第一个的 WS 与 AudioContext 从此无人可达）。
    if (startingRef.current || phaseRef.current !== 'idle') return;
    startingRef.current = true;
    try {
      applyPhase('connecting');
      setNotice(null);
      crisisRef.current = false;
      setCrisisTriggered(false);

      // 1) 出声通道 —— 必须在**这个用户手势里**唤醒 AudioContext，
      //    否则自动播放策略会让第一句话没声音，且事后无法补救。
      const tts = new TtsChannel({
        onStatus: (status) => {
          ttsStatusRef.current = status;
          setTtsStatus(status);
          if (!status.available) {
            setNotice(`语音合成不可用：${status.reason ?? '服务未就绪'}（文字字幕不受影响）`);
          }
        },
        onStart: (seq, text) => {
          // 字幕同步（问题②的修复点）：**这一段音频真的出声了**，才把它对应的
          // 那部分文字揭示出来。原先是在这里把整轮缓冲一次性 flush ——
          // 实测会让第 2、3 句的字幕比它们自己的声音早 455 / 1038 ms（见
          // `SubtitleSyncState` 的实测数据），用户看到的是"文字哗地铺满、声音才念第一句"。
          //
          // `text` 是服务端实际送去合成的原文（`spoken_text`），用它在本轮已到达的
          // 文本里做字面子串定位，即可得到"念到第几个字"。`seq` 的递增顺序就是出声顺序。
          //
          // 安全不变式：这里**只揭示、不删字、不改字**；找不到边界时整段补屏
          // （见 `advanceSubtitleOnSegment`），轮结束/打断/改稿/挂断时剩余部分
          // 一律由 `flushSubtitleRemainder` 兑现 —— 字幕最终必然等于模型全文。
          const acc = accRef.current;
          if (!mutedByReviseRef.current && !mutedByBargeInRef.current && !crisisRef.current) {
            const before = acc.subtitle.shownChars;
            const after = advanceSubtitleOnSegment(acc.subtitle, text);
            if (after > before) {
              setAiLines((prev) =>
                prev.length === 0 ? [acc.subtitle.full] : [...prev.slice(0, -1), acc.subtitle.full],
              );
            }
          }
          // S5：登记"这句话已经进了房间"（回声过滤的唯一输入，必须用 spoken_text）
          //
          // ⚠️ 但**复述用户的那句不登记**：它的指纹与用户自己的话高度重合，
          //    登记了就会把用户继续说下去的话判成回声（实测导致打断完全失效）。
          if (isParrotingUser(text)) {
            setParrotSkips((n) => n + 1);
          } else {
            echoRef.current?.note(text, performance.now());
          }
          // S5：开半双工窗口，覆盖 AEC 收敛期
          openHalfDuplexWindow();
          // `thinking` → `speaking`：**真的出声了**才算在说话。
          // （S4 里这个相位其实从没被置起来过 —— 状态词一直显示"正在思考…"，
          //   而用户听到的明明是声音。S5 顺手修正。）
          if (phaseRef.current === 'thinking') applyPhase('speaking');
          // 只记本轮**第一段**：第 2 段起的 onStart 是"排到它出声了"，
          // 天然包含前面几段的音频时长（实测差 20 倍，施工单 §1 F-7）。
          if (seq !== acc.firstSeq || acc.firstSoundMs !== null) return;
          const pcm = tts.takePcmReadyMs(seq);
          const sound = tts.takeFirstSoundMs(seq);
          if (pcm !== undefined) acc.firstPcmReadyMs = pcm;
          if (sound !== undefined) acc.firstSoundMs = Math.round(sound);
          if (pcm !== undefined || sound !== undefined) publishTurn();
        },
        onEnd: (_seq, text) => {
          // 这一句说完了：再记 2 秒就退出回声窗口（`ECHO_TAIL_MS`）
          echoRef.current?.retire(text, performance.now());
        },
        onDrained: handleDrained,
        onError: (message) => setNotice(`语音合成出错：${message}`),
      });
      ttsRef.current = tts;
      try {
        await tts.start();
      } catch (err) {
        setNotice(`语音合成通道启动失败：${err instanceof Error ? err.message : String(err)}`);
      }

      // 2) 语音输入（含本地流式 ASR，麦克风也在这里被借入）
      try {
        await voiceStartRef.current();
      } catch (err) {
        applyPhase('error');
        setNotice(err instanceof Error ? err.message : '麦克风启动失败，请检查浏览器权限');
        return;
      }
      const asrError = voiceMetricsRef.current.asrError;
      if (asrError) setNotice(`语音识别不可用：${asrError}`);

      // 3) 麦克风**实际生效**的设置 —— 绝不假设 AEC 开了（S5.3）
      const s = voiceGetMicSettingsRef.current();
      if (s) {
        setMicSettings({
          echoCancellation: s.echoCancellation === true,
          noiseSuppression: s.noiseSuppression === true,
          autoGainControl: s.autoGainControl === true,
          sampleRate: s.sampleRate ?? 0,
          channelCount: s.channelCount ?? 0,
          deviceId: s.deviceId ?? '',
        });
      }

      applyPhase('listening');
    } finally {
      startingRef.current = false;
    }
  }, [applyPhase, conversationId, handleDrained, isParrotingUser, openHalfDuplexWindow, publishTurn]);

  const endCall = useCallback(async () => {
    applyPhase('ended');
    window.clearTimeout(forwardingTimerRef.current);
    forwardingTimerRef.current = 0;
    // 挂断时把"等 speak 计划"的宽限定时器也停掉：迟到的 flush 只会往已结束的
    // 通话里塞音频（`ttsSay` 对已 dispose 的通道是空操作，但没必要留着跑）。
    window.clearTimeout(deferTimerRef.current);
    deferTimerRef.current = 0;
    deferredDeltaRef.current = '';
    echoRef.current?.clear();
    pendingRef.current = [];
    // 挂断时把还没来得及随音频上屏的字幕全部兑现（用户挂断前可能还有没看到的文字）
    flushSubtitleRemainder(accRef.current.subtitle);
    // 冲刷最后一段定稿：`stop()` 会等服务端 finalize + 离线复识别（~600ms）再交文本。
    // ⚠️ 顺序很关键：**先**把状态置成 ended（定稿回调因此丢弃），**再**取返回值 ——
    //    否则同一段会被"回调 + 返回值"送两次，落库两条用户消息。
    let lastText = '';
    try {
      lastText = (await voiceStopRef.current()) ?? '';
    } catch {
      /* 冲刷失败就当作没有最后一段 */
    }
    const trimmed = lastText.trim();
    if (trimmed && !accRef.current.active) {
      setUserLines((prev) => [...prev, trimmed]);
      runTurnRef.current(trimmed);
    }

    // 等这一轮（以及它后面排队的插话轮次）落库完再走：
    // 中途 abort 会触发服务端非流式回退 + 幽灵回复（F-4）
    const deadline = performance.now() + END_CALL_WAIT_MS;
    let guard = 0;
    while (performance.now() < deadline && guard < 8) {
      guard += 1;
      const pending = turnDoneRef.current?.done;
      if (!pending) break;
      const remaining = Math.max(0, deadline - performance.now());
      let timer = 0;
      const timeout = new Promise<'timeout'>((r) => {
        timer = window.setTimeout(() => r('timeout'), remaining);
      });
      const raced = await Promise.race([pending.then(() => 'done' as const), timeout]);
      window.clearTimeout(timer);
      if (raced === 'timeout') break;
      // 这一轮结束后可能又起来了下一轮（插话排队）；没有新轮次才收工
      if (turnDoneRef.current?.done === pending) break;
    }

    const tts = ttsRef.current;
    ttsRef.current = null;
    if (tts) {
      try {
        tts.stopAll();
        tts.dispose();
      } catch {
        /* 正在挂断，出声与连接都已经不重要 */
      }
    }
  }, [applyPhase]);

  // ── 卸载清理（幂等）────────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      window.clearTimeout(forwardingTimerRef.current);
      forwardingTimerRef.current = 0;
      echoRef.current?.clear();
      const tts = ttsRef.current;
      ttsRef.current = null;
      try {
        tts?.dispose();
      } catch {
        /* 忽略 */
      }
      // ASR 会话 / 16kHz AudioContext / 麦克风引用由 `useVoiceAnalysis` 自己的
      // 卸载清理负责（注册在同一个组件上）。这里**不**重复 release ——
      // 重复 release 会让引用计数提前归零，把别的消费者的流一起停掉。
    };
  }, []);

  return {
    phase,
    userLines,
    // 中间结果在"正在聆听"时上屏：AI 说话时麦克风照开，但那些字多半是它自己的回声
    partialText: phase === 'listening' ? voice.metrics.asrPartialText : '',
    aiLines,
    revised,
    notice,
    dismissNotice: () => setNotice(null),
    voiceMetrics: voice.metrics,
    ttsStatus,
    discardedFinals,
    droppedEchoes,
    lastEchoReason,
    bargeIns,
    skippedPhoneUnits,
    skippedUnspeakableUnits,
    parrotSkips,
    crisisTriggered,
    micSettings,
    maxConcurrentTurns,
    lastRisk,
    turn,
    startCall,
    endCall,
    aiSpeaking: phase === 'speaking',
    forwardingPaused,
  };
}
