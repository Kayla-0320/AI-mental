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
 *             revise（审计改稿）──────┼──► 保留已听全的前缀 + **补念剩余部分**（见取舍 4）
 *                                     │
 *             done ───────────────────┼──► 等播放队列排空 ──► 回到 listening
 *                                     └──► 有插话排队 → 顺序发下一轮（不并发）
 * ```
 *
 * ## 六个刻意的取舍（都写在对应位置的注释里）
 *
 * 1. **绝不 abort SSE（F-4）**：`consultation.service.ts:360` 收到 abort 后会回退到
 *    非流式 `smartChat`（**第二次完整 LLM 调用**，且不带中止信号），并在 `:432`
 *    落库一份用户从没听过的回复。所以挂断/打断时**不**调用 `abort()`：
 *    打断只清本地播放，本轮让它跑完，插话排队等 `done` 之后作为**下一轮**发出去。
 * 2. **打断靠 partial，不靠 final（G-1）**：partial 每 ~330ms 一次，
 *    final 要等端点 0.45–1.2 s，用 final 会觉得"插话没反应"。
 *    2026-10-11 起自动打断多了一个 `BARGE_IN_CONFIRM_MS`（0.6 s）的**确认窗口**：
 *    第一条 partial 不再立刻掐停 AI（一声咳嗽也会先吐一条 partial），
 *    要持续说话才打断；急着抢话的人用控制栏的「打断并发送」（零等待 + 立刻定稿）。
 * 3. **回声过滤是文本级的（S5.2）**：阈值与时间窗都是 `docs/echo_calibration.md`
 *    里 40 组真实配对标定出来的，不是拍的。
 * 4. **revise 也要出声（2026-10-11 用户拍板）**：口径是"**每一次输出的内容都需要
 *    语音合成**" —— 屏幕上有的字，耳朵里必须有。所以不再"只改字幕不重念"，改成
 *    **前缀对齐后补念剩余部分**（`planReviseSpeech`）：已完整播完且仍是新稿前缀的段
 *    不重念，从第一段没播完/对不上的地方撤回，只合成新稿剩下没念的部分。
 *    旧的保守半边（`voice_call_plan.md` §4.4 的"已播出去收不回来"）仍然成立 ——
 *    撤回的那半句收不回来，所以对齐点只取"**听全了**"的段，宁可重念半句也不吞字。
 * 5. **设备归属**：ASR 走 `useVoiceAnalysis`（消费者名 `'perception-voice'`），
 *    它自己 `mediaStreamManager.acquire('microphone', ...)` —— 通话页**不再**单独持一份
 *    `'call'` 引用（同一份流被两个消费者各持一次是多余的，且给静音与释放增加歧义）。
 *    静音按钮改由页面直接切那一条共享轨道的 `enabled`。
 * 6. **自动判停保守、手动发送即时（2026-10-11）**：服务端的端点静音阈值放宽到
 *    rule1/rule2 = 2.0/1.8 s（原来 1.6/0.8），让"我想想……"式的停顿不再被切成两段；
 *    代价由 {@link interruptAndSend}（「打断并发送」按钮）补上 —— 说完了就点，
 *    不用等那 1.8 s。真机参数可用 `ASR_RULE1_SILENCE_S` / `ASR_RULE2_SILENCE_S`
 *    覆盖，不必改代码；完整记录见 `docs/voice_call_interrupt_tuning.md`。
 *
 * ## 为什么不用 `useAnxiety()`
 *
 * 那个 context 只在 `PatientLayout` 里挂载，而通话页是刻意的整屏独立页（不在
 * PatientLayout 内），`useAnxiety()` 在它会直接抛错。`useVoiceAnalysis` 是自包含 hook，
 * 正是这里该用的东西（施工单 §S4.2 的接线表默认了 context 可用，这一步已修正）。
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { sendChatMessageStream } from '../services/chatStream';
import {
  TtsChannel,
  type TtsStreamStatus,
  type TtsPayloadLedger,
} from '../services/ttsStream';
import type { TtsUnderrunStats } from '../services/ttsPlayer';
import {
  buildVoiceTextLedger as buildLedger,
  type SpeakPlanMode,
  type VoiceTextLedger,
} from '../services/voiceLedger';
import { EchoFilter, ECHO_OVERLAP_THRESHOLD, scoreAgainstSpoken } from '../services/echoFilter';
import { useVoiceAnalysis } from './useVoiceAnalysis';
import type { FacialLiveState } from '../services/facialLiveState';
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
  /**
   * 首段 `say` 发出时，播放器里**还排着**的音频时长（毫秒）与**还没合成完**的句数。
   *
   * 为什么要有这两个数（2026-10-11 真机：某轮首声 **6159 ms**，而合成本身只要
   * 1.4~1.7 s）：用户听到的"停顿"有两个来源，对策完全不同 ——
   *   · 排队 → 本字段大（上一轮的句子还没播完）；
   *   · 合成慢 → `firstServerFirstChunkMs` 大（排队为 0 时才是它）。
   * 只报一个 `firstSoundMs` 时，这两种情况长得一模一样。
   */
  queueAheadAtFirstSayMs: number | null;
  pendingUnitsAtFirstSay: number | null;
  /** 首段「`say` → 第一块 PCM 到手」（客户端观测） */
  firstPcmAtMs: number | null;
  /**
   * 首段服务端自报的合成读数（`end` 帧带来）：请求→第一块、整段耗时、实时率。
   *
   * 它们与客户端观测（`firstPcmAtMs` / `firstPcmReadyMs`）相减，就能把
   * "慢在服务端合成"与"慢在下行/客户端"分开 —— 离线实测 CosyVoice3 首块
   * 1.44~1.68 s、RTF 0.89~1.01（9 次采样），所以首声远大于它时**不该**怪合成。
   */
  firstServerFirstChunkMs: number | null;
  firstServerSynthMs: number | null;
  firstRtf: number | null;
  /**
   * `speak` 计划到达的时刻（相对本轮开始）；未收到为 null。
   *
   * 为什么要单独记它：协议实测顺序是 `meta → delta* → speak → [revise] → done`，
   * 计划**总是在全部 delta 之后**才到。而客户端在第一个 delta 后只等
   * `SPEAK_PLAN_GRACE_MS`（350ms）就启动 delta 回退合成 —— 所以这 350ms 到底
   * "等到了计划"还是"白等"，只能靠 ③ 与 ① 两个时刻比出来（见下面两格）。
   */
  speakAtMs: number | null;
  /** 宽限计时器**触发**的时刻（相对本轮开始）；没走 delta 回退则为 null */
  graceFiredAtMs: number | null;
  /** 本轮第一个 `say` 真正发出去的时刻（相对本轮开始）—— "用户说完 → 送合成"的终点 */
  firstSayAtMs: number | null;
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
  /**
   * 本轮播放里出现的**音频空洞**次数与累计时长（毫秒）。
   *
   * 判据见 `ttsPlayer.ts` 的 `TtsUnderrunStats`：一块音频来晚了、播放头已经越过
   * 上一块结尾，中间那段没人补 —— 就是耳朵里的"卡一下"。
   * **这是"卡顿"与"反应慢"的分界线**：空洞多 ⇒ 音频断裂；
   * 空洞为 0 而 `firstSoundMs` 大 ⇒ 只是整体慢，不是卡。
   */
  audioHoleCount: number;
  audioHoleMs: number;
  /**
   * 本轮被**我们自己掐断**、因此没出声的句数（打断 / 改稿 / 危机切换）。
   *
   * ⚠️ 与 `audioHoleCount` 不是一回事：空洞是"该有声音却没有"（缺陷），
   * 这里是"有意丢掉"（设计）。它单独占一格的意义是：面板上出现数字时，
   * 用户能分辨"这一句是我不让念的"还是"这一句根本合成不出来" ——
   * 后者才会弹「语音合成出错」（见 `TtsStreamSession.stoppedSeqs`）。
   */
  audioStoppedCount: number;
  /**
   * 本轮**载荷级**音频账：发出了几句 `say`、服务端回了多少 PCM 字节。
   *
   * ⚠️ 这是"没有声音"这类问题的**唯一直接证据**（时间读数在那情况下全是 `—`）：
   *   · `says > 0 && pcmBytes === 0` → 服务端一个字节都没回（合成/OOV/取消）
   *   · `pcmBytes > 0` 却没听到   → 音频到了客户端，问题在播放层
   * 见 `ttsStream.ts` 的 {@link TtsPayloadLedger}。
   */
  says: number;
  pcmBytes: number;
  /**
   * 本轮**送合成过短的单元**个数（< {@link SHORT_UNIT_CHARS} 字）与最短的那一段。
   *
   * ## 为什么量它（2026-10-11 用户："AI 回复时偶尔会说出一些乱说的话，然后才能说出回复的内容"）
   *
   * `onDelta` 的 delta 回退路径会把**宽限期到点时手上已有的 delta** 送去合成。
   * 模型流式吐字是逐字的，所以宽限期到点时手上可能只有「嗯」「我」「那个」这么两三个字
   * —— 那段字会被当成一个**独立合成单元**念出来，听感就是"先说一句莫名其妙的短话，
   * 然后才说正文"。这不是模型乱说，是**切分切在了一个不成立的位置**。
   *
   * 服务端只对"一段太长"负责（`splitSpeakable` 按字数硬切），**管不了"太短"**；
   * 而"太短的单元"只有这里能观察到，所以先把证据摆出来。
   */
  shortUnits: number;
  /** 最短的那个单元原文（便于一眼看出是什么被打碎念了） */
  shortestUnit: string;
  /**
   * 本轮是否出现过"首段太短、先不送合成、等凑齐"（见 `flushByDelta` 的短单元守卫）。
   *
   * 这是"AI 先说一句莫名其妙的话"那件事的**处理记录**：为 1 说明本轮真的差点
   * 把两三个字单独念出来，被守卫拦住了。
   */
  shortHeld: boolean;
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
 * 自动打断的**确认窗口**（S5.4 修订）：用户开口后要持续这么久才真把 AI 掐停。
 *
 * 为什么需要它：原实现是"第一条非回声 partial 就打断"。而 partial 是**解码器
 * 刚听到一点点就吐出来的中间结果** —— 一声咳嗽、桌椅响、或者用户只是"嗯"了一下
 * 换气，都会先吐一条 partial。结果是 AI 的句子被切在半句上，用户还得再听一遍。
 * （实测数据见 `docs/echo_calibration.md`：partial 首条约在开口后 700ms，
 * 之后每 ~330ms 一条。）
 *
 * 0.6s ≈ 两条 partial 的间隔：要求"用户确实在连续说话"，而不是"响了一声"。
 * 这段时间里 AI 只会多说半句 —— 与"被无缘无故切掉半句"相比，后者更伤体验。
 *
 * ⚠️ 它只延缓**自动**打断；用户真想立刻打断时用控制栏的「打断并发送」按钮，
 * 那条路径是**立即**停播、零延迟（见 `interruptAndSend`）。
 */
const BARGE_IN_CONFIRM_MS = 600;

/**
 * 送 TTS 前的最后一道保险（施工单 §S6.3）：**含电话号码特征的片段不合成**。
 *
 * 为什么必须落成代码而不是靠自觉：实测（`docs/echo_calibration.md` §4.3）
 * `400 161 9995` 被念成「四百六十一九九九九五」—— TTS 把它当成一个 11 位数读，
 * 而不是分组电话号码。危机场景下念错热线号码的后果不可接受，所以号码**只以文字卡片呈现**。
 *
 * ⚠️ 2026-10-11 修正粒度：旧实现是在**整段（最多 100 字、按逗号打包）**上测这个正则，
 * 命中就把**整段**丢掉 —— 一句里带号码，等于整句都没有声音（而字幕是完整的），
 * 这正是"语音输出不全"的一个真凶。现在正则只用来**摘出号码本身**，
 * 前后文照常合成，被跳过的片段连原因一起记进对账链（见 `splitForSpeech`）。
 */
export const PHONE_LIKE = PHONE_LIKE_REGEX;
export { PHONE_LIKE_SOURCE } from '../services/ttsSplit';

/**
 * 把一段 AI 文本切成"送合成的批次"（施工单 §S4.4），并逐片段记账。
 *
 * 实现已抽到 `services/ttsSplit.ts`（纯函数、可单测），这里 import 进来供本文件
 * 使用，并原样转出，保持既有调用点与外部引用不变。
 */
import {
  splitSpeakable,
  splitForSpeech,
  planJoinMatchesText,
  unitsToCancelForPlan,
  planReviseSpeech,
  PHONE_LIKE as PHONE_LIKE_REGEX,
  type SkippedSpeechFragment,
} from '../services/ttsSplit';

export { splitSpeakable, MAX_TTS_UNIT_CHARS } from '../services/ttsSplit';

/**
 * 通话过程记录（时间轴上的消息流）。
 *
 * 为什么要有它：页面原来只有 `userLines` / `aiLines` 两个**扁平字符串数组** ——
 * 它们回答"现在显示什么字"，回答不了"这一通电话聊了什么"（没有讲话人、没有时刻、
 * 两个数组各自追加因而看不出轮次）。用户口径是"每轮对话过程都像文字对话那样"，
 * 所以这里旁路再记一条**带讲话人与时刻**的时间轴，页面按轮渲染。
 *
 * ⚠️ 它**只做编排**：文本取的是同一个字符串（改稿后的文本原样进来），不改字、不丢字；
 * 也不参与风险等级 / 融合权重 / 画像分数。实现与回归见 `services/callTranscript.ts`
 * 与 `scripts/verify_call_transcript.ts`。
 */
import {
  appendAssistantDelta,
  appendUserMessage,
  createTranscript,
  nextTranscriptId,
  type CallTranscriptEntry,
} from '../services/callTranscript';

/** 一轮进行中的累计量（放 ref：TTS 回调与 SSE 回调不在同一作用域） */
interface TurnAccumulator {
  userText: string;
  startedAt: number;
  firstDeltaMs: number | null;
  firstPcmReadyMs: number | null;
  firstSoundMs: number | null;
  /** 首段 `say` 时"前面还排着多少音频 / 几句没合成完"（排队 vs 合成慢的分界线） */
  queueAheadAtFirstSayMs: number | null;
  pendingUnitsAtFirstSay: number | null;
  /** 首段「`say` → 第一块 PCM」（客户端） */
  firstPcmAtMs: number | null;
  /** 首段服务端自报：请求→第一块 / 整段合成 / 实时率 */
  firstServerFirstChunkMs: number | null;
  firstServerSynthMs: number | null;
  firstRtf: number | null;
  /** `speak` 计划到达时刻（相对本轮开始）；未收到为 null —— 与 `graceFiredAtMs` 比出"白等" */
  speakAtMs: number | null;
  /** 宽限计时器触发时刻（相对本轮开始）；没走 delta 回退为 null */
  graceFiredAtMs: number | null;
  /** 本轮第一个 `say` 发出时刻（相对本轮开始） */
  firstSayAtMs: number | null;
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
  /**
   * 本轮开始时播放器的**累计**空洞读数（`TtsPlayer.underrun`）。
   *
   * 单轮增量 = 收尾时的累计值 − 这里的快照。为什么不直接用播放器自己的计数：
   * 播放器是整通电话一个实例，而 `TurnMetrics` 是**按轮**的读数。
   */
  holeBaseCount: number;
  holeBaseMs: number;
  /** 本轮开始时"被我们掐断未播"的累计读数（口径同上，见 TurnMetrics.audioStoppedCount） */
  stoppedBaseCount: number;
  /**
   * 本轮是否已经因为"首段太短"扣过一次（见 `TurnMetrics.shortHeld`）。
   * 每个轮次清一次；用来保证"扣一次、最多再等一个宽限期就照发"，绝不长期扣住文本。
   */
  shortHeld: boolean;
  /**
   * 本轮开始时**载荷级账**的基线（`say` 句数 / 收到的 PCM 字节数）。
   *
   * 为什么要它（2026-10-11 用户："我两次说我感觉好一些了都没有声音出来"）：
   * 时间读数（首声/首块/RTF）只能回答"慢不慢"；"**完全没有声音**"必须靠数量回答——
   * 是没发 `say`、还是发了但服务端没回音频、还是回了被取消丢掉。
   * 见 `ttsStream.ts` 的 {@link TtsPayloadLedger}。
   */
  sayBase: number;
  pcmBytesBase: number;

  // ── 文本 ↔ 语音对账链（见 VoiceTextLedger）──────────────────────────────
  /** 本轮序号（单调递增，只在日志/面板里用来把四份文本对上号） */
  turnId: number;
  /**
   * 本轮合成由谁驱动。**同一轮只允许一个来源**（见 `planModeRef` 的注释）：
   * `pending` 还在等计划、`speak-plan` 由定稿计划驱动、`delta-fallback` 由 delta 驱动。
   */
  planMode: SpeakPlanMode;
  /** B：`speak` 计划（没收到为 null） */
  planText: string | null;
  planSegments: string[] | null;
  /** C：实际发出的每个 `say`（按序）—— 拼接起来就是"文本层认为要念的内容" */
  saidUnits: { seq: number; text: string }[];
  /** D：实际出声的 `spoken_text`（按播放顺序） */
  spokenUnits: { seq: number; text: string }[];
  /** 有意跳过的片段（号码 / 纯标点）：对账时算**预期例外** */
  skippedFragments: SkippedSpeechFragment[];
  /** 计划迟到/重复/违约等异常（人能读的一句话） */
  ledgerNotes: string[];
  /**
   * 被打断/改稿时**没有念出来**的那一段（页面据此明确标注，不再让字幕假装念过）。
   * null 表示本轮从头到尾都念完了。
   */
  unspokenTail: string | null;
}

/** 单轮合成的来源（同一轮只允许一个） */
export type { SpeakPlanMode, VoiceTextLedger } from '../services/voiceLedger';

/**
 * 把"响应头写出之前"的流式失败翻译成用户能行动的一句话。
 *
 * `onError` 只覆盖**流内部**下发的 `error` 事件；服务端在 `writeHead(200)` 之前失败
 * （会话已被删除 → 404、鉴权失效、上游 500）时，`completed` 是 reject 的，
 * 必须另接一条 catch —— 否则本轮永远停在"正在思考…"，控制台只有一条
 * unhandled rejection（2026-10-11 实测）。
 */
function describeStreamFailure(err: unknown): string {
  const status = (err as { status?: number } | null)?.status;
  if (status === 404) {
    return '这通电话的会话已经不存在了（可能已被删除）。请返回后重新发起通话。';
  }
  if (status === 401 || status === 403) {
    return '登录状态已失效，请重新登录后再发起通话。';
  }
  const detail = err instanceof Error ? err.message : String(err);
  return `这一轮没能生成回复（${detail}）。麦克风仍然开着，你可以再说一次。`;
}

function newAccumulator(): TurnAccumulator {
  return {
    userText: '',
    startedAt: 0,
    firstDeltaMs: null,
    firstPcmReadyMs: null,
    firstSoundMs: null,
    queueAheadAtFirstSayMs: null,
    pendingUnitsAtFirstSay: null,
    firstPcmAtMs: null,
    firstServerFirstChunkMs: null,
    firstServerSynthMs: null,
    firstRtf: null,
    speakAtMs: null,
    graceFiredAtMs: null,
    firstSayAtMs: null,
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
    holeBaseCount: 0,
    holeBaseMs: 0,
    stoppedBaseCount: 0,
    sayBase: 0,
    pcmBytesBase: 0,
    shortHeld: false,
    turnId: 0,
    planMode: 'pending',
    planText: null,
    planSegments: null,
    saidUnits: [],
    spokenUnits: [],
    skippedFragments: [],
    ledgerNotes: [],
    unspokenTail: null,
  };
}

/**
 * 把一轮的对账记录整理成"人能读的差异清单"。
 *
 * 实现是**纯函数**，放在 `services/voiceLedger.ts`（不依赖 React/DOM/网络），
 * 因此能在 Node 里直接跑回归：`scripts/verify_tts_text_ledger.ts`。
 *
 * @param acc 本轮累计量
 * @param shown 页面最终显示的文本（A）
 * @returns 对账记录
 */
export function buildVoiceTextLedger(acc: TurnAccumulator, shown: string): VoiceTextLedger {
  return buildLedger({
    turnId: acc.turnId,
    source: acc.planMode,
    shown,
    planText: acc.planText,
    planSegments: acc.planSegments,
    saidUnits: acc.saidUnits,
    spokenUnits: acc.spokenUnits,
    skipped: acc.skippedFragments,
    notes: acc.ledgerNotes,
    unspokenTail: acc.unspokenTail,
    sawRevise: acc.sawRevise,
  });
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
 *
 * ⚠️ **2026-10-11：这个值已可逐通话覆盖**（`UseVoiceCallOptions.graceMs`，
 * 通话页用 `?grace=250`）。原因是排查发现 **`speak` 计划实际上从来没在宽限期内到达过**
 * （真机 `speak 到达 = 未收到`），也就是这 350ms 基本是**纯等待**：
 * 用户决定下一轮做 350 → 250 的 A/B，所以先把旋钮做出来，**默认值仍然是 350 不动**。
 * 注意缩短它有可能让首段更早送去合成、但**首段更短、片段数变多**，反而加重句中卡顿 ——
 * 所以对比要看四项读数：`首个 say` / 首段出声 / `本轮合成段数` / `缓冲·见底`。
 */
const SPEAK_PLAN_GRACE_MS = 350;

/** 宽限期的允许范围（防止误配把首段切得太碎或白等太久） */
const MIN_SPEAK_PLAN_GRACE_MS = 0;
const MAX_SPEAK_PLAN_GRACE_MS = 1000;

/**
 * "太短的合成单元"阈值（字符数）。
 *
 * 为什么用 4：中文里一格完整的语义单位极少只有 1~3 个字（"嗯""那个""我想"），
 * 而**独立合成**这几个字会读出一段突兀的短话 —— 用户听到的"先冒出一句乱话"
 * 最可能就是它。这里只**计量并展示**，不改变切分行为（改切分是产品判断，见
 * `TurnMetrics.shortUnits` 的说明）。
 */
const SHORT_UNIT_CHARS = 4;

/**
 * **首段合成的最小字数**：不足它就先扣住、再等一个宽限期凑齐（见 `flushByDelta`）。
 *
 * 为什么需要（2026-10-11 用户："AI 回复时偶尔会说出一些乱说的话，然后才能说出回复的内容"）：
 * delta 是**逐字**到达的，宽限期到点时手上可能只有「嗯」「不过」这种两三个字。
 * 原实现会把它当**一个独立合成单元**送去念 —— 于是用户先听到一句莫名其妙的短话，
 * 然后才是正文。这与"模型乱说"完全不同：文本没错，是**切在了一个不成立的位置**。
 *
 * 边界（**一个字都不丢**）：
 *   · 扣住的文本放回队列，下一个 delta 到达时继续累积；
 *   · 若又等满一个宽限期还没凑齐，**照原样送出去**（`allowShort`），只是白等 ≤1 个宽限期；
 *   · 一轮只扣一次（`TurnAccumulator.shortHeld`），不会把文本长期压住。
 */
const MIN_FIRST_UNIT_CHARS = 6;

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
  /**
   * 取**当前**面部实时状态（通话页在用户开了摄像头时由 `useCallVision` 提供）。
   *
   * 为什么是 getter 而不是值：本轮要在"用户说完话那一刻"取快照，而视觉采样是
   * 200ms 一拍；把值当依赖传进来会让 `runTurn` 每 200ms 重建一次。
   *
   * @returns 快照；没有新鲜读数时返回 `null`（本轮就**不**带面部字段上行）
   */
  getFacialLiveState?: () => FacialLiveState | null;
  /**
   * **开场预缓冲**（毫秒）：第一块音频到达后先攒这么久再出声，用来吸收
   * CosyVoice3 阵发式供音（块间隔 1.5~3.2s）造成的短时断档。
   *
   * 实测依据与默认值见 `services/ttsPlayer.ts` 的 `DEFAULT_PREBUFFER_MS`
   * （**默认 300ms**，用户真机试听后定的）。通话页可用 `?prebuffer=0` 关掉、
   * `?prebuffer=600` 加强。**不传 = 用 ttsPlayer 的默认值**（不要在这里写 0 兜底，
   * 那会把默认值静默吃掉）。
   */
  prebufferMs?: number;
  /**
   * **等 `speak` 计划的宽限期**（毫秒）。不传就用 {@link SPEAK_PLAN_GRACE_MS}（350）。
   *
   * 为什么要可覆盖：实测 `speak` 计划从未在宽限期内到达（真机 `speak 到达 = 未收到`），
   * 这 350ms 基本是纯等待。但它不是"越小越好"：扣住的 delta 越早放出去，
   * 首段可能越短、合成片段数越多，反而加重句中卡顿。所以做成旋钮做 A/B，
   * **默认值不动**（用户要求先验收预缓冲修复、再单独测这一项）。
   */
  graceMs?: number;
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
  /** 用户点「打断并发送」且控制帧真的发出去了的次数（诊断用） */
  manualSends: number;
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
  /**
   * **载荷级** TTS 账（整通电话累计）：`say` 发出几句、服务端回了多少 PCM 字节、
   * 有多少帧/字节被取消丢掉、有多少句"合成成功但没音频"。
   *
   * 与 `turn.says / turn.pcmBytes`（单轮增量）配着看：
   * 整通累计回答"到底出没出过声音"，单轮增量回答"是哪一轮没出声"。
   * 见 `ttsStream.ts` 的 {@link TtsPayloadLedger}。
   */
  ttsLedger: TtsPayloadLedger | null;
  /**
   * **输出侧**健康读数：浏览器 `AudioContext` 的实际状态 + 累计排程的音频块数与时长。
   *
   * 它是"音频到了客户端却没听见"与"根本没合成出音频"的分界线：
   * `suspended` 表示自动播放策略把播放器挂起了（排了程也不会响），
   * 而这时所有时间读数（首声/首块/RTF）看起来都正常。
   */
  outputHealth: {
    ctxState: string;
    scheduledChunks: number;
    scheduledAudioMs: number;
    resumeAttempts: number;
    prebufferMs: number;
    prebufferAppliedMs: number;
    prebufferPending: boolean;
    prebufferActivations: number;
  } | null;
  /**
   * 播放缓冲区统计（`?debug=1` 面板）：**句内见底 vs 跨句见底**。
   *
   * 用户口径（2026-10-11）："小卡顿优先排查供音速度是否跟得上播放，以及两个片段之间
   * 有没有短暂断档" —— 这两个数就是那两件事的判据：
   *   · `withinSentenceStarves` 大 ⇒ 供音跟不上播放（该预缓冲 / 该优化推理）；
   *   · `gapStarves` 大 ⇒ 片段之间空着等（该提前发下一句的合成请求）；
   *   · `bufferMs` 是当前水位，它一路贴 0 就是前者。
   */
  bufferStats: {
    chunks: number;
    withinSentenceStarves: number;
    gapStarves: number;
    starveTotalMs: number;
    bufferMs: number;
  } | null;
  /** 缓冲区日志明细（逐块水位 + 见底现场）—— 面板上的「导出缓冲日志」按钮用它 */
  dumpBufferLog: (recentChunks?: number) => string;
  /** 整通电话累计的播放空洞统计（`?debug=1` 面板；判据见 `TtsUnderrunStats`） */
  audioStats: TtsUnderrunStats | null;
  /**
   * **整通电话的完整过程**：时间轴上的每一条消息（谁 + 什么时刻 + 第几轮 + 说了什么）。
   *
   * 与 `userLines` / `aiLines` 的关系：后两者是"字幕区现在该显示什么"的扁平视图，
   * 本字段是"这一通电话聊了什么"的可读记录 —— 页面按轮铺开、按时间读（用户口径：
   * "把每轮对话过程都像文字对话那样弄出来"）。已说出的内容**只增不删**，
   * 断网重连、打断、改稿都不会清空它。
   */
  transcript: CallTranscriptEntry[];
  /**
   * 最近一轮「文本 ↔ 语音」对账（A 显示 / B 计划 / C 发出 / D 出声 + 差异清单）。
   *
   * 这是排查"回复和语音不匹配"的第一手数据：先看 `exceptions`，
   * 它直接写出**第一处不一致在哪一层**（见 `buildVoiceTextLedger`）。
   */
  voiceLedger: VoiceTextLedger | null;
  /**
   * 本轮**没有被念出来**的那一段（打断/改稿/危机/挂断时非空）。
   *
   * 页面据此明确标注「以下未朗读」—— 字一个不丢，但界面不再假装它出过声。
   */
  unspokenTail: string | null;
  startCall: () => Promise<void>;
  /** 挂断：冲刷最后一段定稿 → 等本轮跑完（有上限）→ 释放资源 */
  endCall: () => Promise<void>;
  /**
   * 「打断并发送」：立刻停掉 AI 的声音，并把**已经说出口的话**马上定稿发出去，
   * 不等服务端的静音判停（rule2 = 1.8 s）。
   *
   * 语义边界：插话仍走 F-4 的既有路径 —— 本轮不 abort、插话排队等 `done` 之后
   * 作为下一轮发。所以它是"不用等静音判停"，不是"绕过服务端顺序"。
   */
  interruptAndSend: () => void;
  /** AI 正在出声（UI 据此显示"小安在说话"而不是"暂停收音"） */
  aiSpeaking: boolean;
  /** 半双工窗口是否正开着（`?debug=1` 用；窗口内不向服务端送帧） */
  forwardingPaused: boolean;
}

export function useVoiceCall(options: UseVoiceCallOptions): UseVoiceCallResult {
  const { conversationId, getFacialLiveState, prebufferMs, graceMs } = options;
  /**
   * 本次通话实际生效的宽限期。放 ref：`runTurn` 是个大 `useCallback`，
   * 不想因为一个数值变化就把它重建（它与轮次编排绑在一起）。
   */
  const graceMsRef = useRef(
    Math.max(
      MIN_SPEAK_PLAN_GRACE_MS,
      Math.min(MAX_SPEAK_PLAN_GRACE_MS, Math.round(graceMs ?? SPEAK_PLAN_GRACE_MS)),
    ),
  );
  graceMsRef.current = Math.max(
    MIN_SPEAK_PLAN_GRACE_MS,
    Math.min(MAX_SPEAK_PLAN_GRACE_MS, Math.round(graceMs ?? SPEAK_PLAN_GRACE_MS)),
  );

  const [phase, setPhase] = useState<CallPhase>('idle');
  const [userLines, setUserLines] = useState<string[]>([]);
  const [aiLines, setAiLines] = useState<string[]>([]);
  /**
   * 通话过程记录（见 `UseVoiceCallResult.transcript`）。
   *
   * ⚠️ 与 `userLines` / `aiLines` **同源同刻**更新：凡是往那两个字幕数组写的地方，
   * 这里都跟着写一条。字幕数组是既有行为（不动），本字段是旁路的可读记录。
   */
  const [transcript, setTranscript] = useState<CallTranscriptEntry[]>(() => createTranscript());
  /** 消息 id 计数器（整通电话单调递增；重置时机见下面的 startCall 清空 effect） */
  const transcriptIdRef = useRef(0);
  /**
   * 用户开口的次数（从 1 开始）。AI 的一条回复与前一条用户消息**同号**，
   * 页面据此把"一次提问 + 它的一次回答"分成同一轮。
   */
  const transcriptTurnRef = useRef(0);
  /** 本次通话是否已经开过一次（真源 `phase !== 'idle'`；见下面清空时间轴的 effect） */
  const transcriptSessionRef = useRef(false);
  const [revised, setRevised] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [ttsStatus, setTtsStatus] = useState<TtsStreamStatus | null>(null);
  const [discardedFinals, setDiscardedFinals] = useState(0);
  const [droppedEchoes, setDroppedEchoes] = useState(0);
  const [lastEchoReason, setLastEchoReason] = useState<string | null>(null);
  const [bargeIns, setBargeIns] = useState(0);
  /** 用户点「打断并发送」且控制帧真的发出去了的次数（诊断用） */
  const [manualSends, setManualSends] = useState(0);
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
  /**
   * 播放空洞的**整通电话累计**读数（`?debug=1` 面板看它）。
   *
   * 与 `turn.audioHoleMs`（单轮增量）的区别：轮次读数会被下一轮覆盖，
   * 而"这通电话一共卡了多少下"需要累计量才对得上用户的体感。
   */
  const [audioStats, setAudioStats] = useState<TtsUnderrunStats | null>(null);
  /** 载荷级 TTS 账（见 {@link TtsPayloadLedger}）—— 面板上回答"到底出没出过声音" */
  const [ttsLedger, setTtsLedger] = useState<TtsPayloadLedger | null>(null);
  /** 输出侧健康（见 {@link UseVoiceCallResult.outputHealth}）—— 面板上区分"没合成"与"没听见" */
  const [outputHealth, setOutputHealth] = useState<{
    ctxState: string;
    scheduledChunks: number;
    scheduledAudioMs: number;
    resumeAttempts: number;
    prebufferMs: number;
    prebufferAppliedMs: number;
    prebufferPending: boolean;
    prebufferActivations: number;
  } | null>(null);
  /** 播放缓冲区统计（句内/跨句见底）—— 见 {@link UseVoiceCallResult.bufferStats} */
  const [bufferStats, setBufferStats] = useState<{
    chunks: number;
    withinSentenceStarves: number;
    gapStarves: number;
    starveTotalMs: number;
    bufferMs: number;
  } | null>(null);
  /**
   * 最近一轮的「文本 ↔ 语音」对账记录（A/B/C/D 四份文本 + 差异清单）。
   *
   * 用户要求的第一条就是这个：**先让数据指出是哪一层不一致**，而不是猜。
   * 见 {@link buildVoiceTextLedger} 与 `docs/voice_call_text_voice_ledger.md`。
   */
  const [voiceLedger, setVoiceLedger] = useState<VoiceTextLedger | null>(null);
  /**
   * 「本轮哪一段没被念出来」—— 与台账同源，但单独放 state：
   * 页面要在**打断发生的当下**立刻标注，而不是等这一轮结束后的对账快照。
   */
  const [unspokenTailState, setUnspokenTailState] = useState<string | null>(null);
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

  /**
   * 通话开始时清空过程记录（**只在"从 idle 进入通话"这一次**）。
   *
   * 为什么按 `phase !== 'idle'` 判：这正是页面计时器的口径（顶栏 `mm:ss` 从
   * "离开 idle"开始走）。若在 `startCall` 里清，`connecting → error → 再点一次`
   * 会把上一次失败通话的残留也带进来；而"重试"（error → connecting）时
   * `phase` 一直是非 idle，所以**不会**清掉用户已经看到的过程 —— 与页面计时一致。
   */
  useEffect(() => {
    if (phase === 'idle') {
      transcriptSessionRef.current = false;
      return;
    }
    if (transcriptSessionRef.current) return;
    transcriptSessionRef.current = true;
    transcriptIdRef.current = 0;
    transcriptTurnRef.current = 0;
    setTranscript(createTranscript());
  }, [phase]);

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
   * 本轮在**过程记录**里的轮次号（1 起）。
   *
   * 口径：AI 的回复跟前一条用户消息同号；若这一轮还没归属用户消息（异常路径），
   * 退化成"用户已开口数 + 1"，至少保证单调不串轮。读 ref，身份恒定。
   */
  const accTurnNo = useCallback((): number => {
    const n = accRef.current.turnId;
    return n > 0 ? n : transcriptTurnRef.current + 1;
  }, []);

  /**
   * 轮的收尾：把还没随音频上屏的部分兑现。
   *
   * 四种收尾路径都必须调用它，否则会吞字：轮 `done`、审计改稿、用户插话打断、
   * 进入危机分支、挂断（见各调用点的注释）。
   */
  const flushSubtitleRemainder = useCallback(
    (state?: SubtitleSyncState, flags: { unspoken?: boolean } = {}) => {
      const st = state ?? subtitleRef.current;
      const rest = subtitleRemainder(st);
      if (!rest) return;
      st.shownChars = st.full.length;
      const full = st.full;
      setAiLines((prev) => (prev.length === 0 ? [full] : [...prev.slice(0, -1), full]));
      // 过程记录：字幕尾部被整体兑现成全文 —— 时间轴上也是**替换**尾部那条 AI 消息
      // （同一条回复的后续增量，不是新的一条）。`unspoken` 表示这一轮剩下的字
      // 确实不会再出声了（打断 / 改稿 / 危机 / 挂断）。
      transcriptIdRef.current += 1;
      setTranscript((prev) =>
        appendAssistantDelta(
          prev,
          full,
          accTurnNo(),
          Date.now(),
          nextTranscriptId(transcriptIdRef.current),
          'replace',
          flags.unspoken === true ? { unspoken: true } : {},
        ),
      );
    },
    // `accTurnNo` 读 ref、身份恒定（写成依赖不改变行为，只为与既有风格一致）
    [accTurnNo],
  );

  /**
   * 「这一段不会再出声了」—— 先记账，再照旧把剩余字幕兑现上屏。
   *
   * ## 为什么必须有它（用户报告："回复和语音内容完全不匹配"）
   *
   * 打断 / 审计改稿 / 危机切换都会**立刻停掉**本轮剩余的音频。而字幕的既有策略是
   * "内容不藏"—— 停播后仍把剩下的字一次性刷上屏。两者叠加的后果是：
   * **屏幕上是一整段完整回复，耳朵里只听到了前半句**，用户读到的和听到的不是一回事。
   *
   * 现在把"未朗读的那一段"记进本轮台账（`TurnAccumulator.unspokenTail`），
   * 页面据此明确标注「以下未朗读」；字一个都不丢（仍然是原样上屏），
   * 但界面不再假装它被念过。
   *
   * @param reason 停播原因（打断 / 改稿 / 危机 / 挂断），写进台账便于复盘
   * @returns 未朗读的文本（没有剩余时为空串）
   */
  const markUnspokenTail = useCallback(
    (reason: 'barge-in' | 'revise' | 'crisis' | 'hangup'): string => {
      const acc = accRef.current;
      const rest = subtitleRemainder(acc.subtitle);
      if (!rest) return '';
      acc.unspokenTail = rest;
      acc.ledgerNotes.push(`停播原因=${reason}，未朗读 ${rest.length} 字`);
      setUnspokenTailState(rest);
      // 兑现上屏的同时把"这一段没念出来"记进过程记录（两者同一次写入，不会错位）
      flushSubtitleRemainder(acc.subtitle, { unspoken: true });
      return rest;
    },
    [flushSubtitleRemainder],
  );

  /**
   * AI 文本到达时写进**过程记录**（时间轴）。
   *
   * 与页面字幕同步改写：`mode='delta'` 是流式增量（同一轮追加到尾部那条），
   * `mode='replace'` 是整轮替换（`speak` 计划 / 审计改稿 / 分段揭示的全文）。
   * 轮次取 {@link accTurnNo} —— 与"前一条用户消息同号"。
   */
  const pushAiTranscript = useCallback(
    (
      text: string,
      mode: 'delta' | 'replace',
      flags: { revised?: boolean; unspoken?: boolean } = {},
    ) => {
      if (!text) return;
      transcriptIdRef.current += 1;
      setTranscript((prev) =>
        appendAssistantDelta(
          prev,
          text,
          accTurnNo(),
          Date.now(),
          nextTranscriptId(transcriptIdRef.current),
          mode,
          flags,
        ),
      );
    },
    [accTurnNo],
  );
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
   * 自动打断的**候选时刻**（本轮的相对毫秒数；0 = 当前没有候选）。
   *
   * 收到第一条非回声 partial 时置上，`BARGE_IN_CONFIRM_MS` 之后才真打断 ——
   * 这样"一声响 / 一次换气"不会把 AI 的整句切掉。见 `handleVoicePartial`。
   */
  const bargeCandidateAtRef = useRef(0);
  /** 上面那个确认窗口的定时器 id（换轮/打断/卸载时必须清掉） */
  const bargeConfirmTimerRef = useRef(0);
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
    // 队列空了：这一刻的空洞累计量是"这一轮最终的"读数，刷新给诊断面板
    setAudioStats(ttsRef.current?.underrun ?? null);
    settleToListening();
  }, [settleToListening]);

  // ── 打断：partial 判定（S5.4 / G-1）──────────────────────────────────────
  /**
   * 真打断：把 AI 掐停，并且本轮剩下的句子不再出声。**立即生效，零等待**。
   *
   * 两条调用路径共用它，保证"自动"与"手动"的收尾完全一致：
   *   ① 自动打断的确认窗口到期（`handleVoicePartial` 起的定时器）；
   *   ② 用户点控制栏的「打断并发送」（`interruptAndSend`）。
   *
   * @returns 是否真的发生了打断（本轮已经是打断态 / 本轮已结束时为 false）。
   */
  const commitBargeIn = useCallback((): boolean => {
    const acc = accRef.current;
    if (!acc.active || bargedInRef.current) return false;
    if (phaseRef.current === 'ended') return false;
    window.clearTimeout(bargeConfirmTimerRef.current);
    bargeConfirmTimerRef.current = 0;
    bargeCandidateAtRef.current = 0;

    const now = performance.now();
    // ── G-2 —— 先清本地队列，再谈别的，绝不等服务端回执 ──
    bargedInRef.current = true;
    // 本轮剩下的句子**不再出声**：已经"立刻静音"了，几百毫秒后又自己说起来
    // 会让人以为打断失败了。字幕照旧追加（内容不藏），回答由队列里的插话开新一轮。
    mutedByBargeInRef.current = true;
    // 打断前还没随音频上屏的字幕不会再有音频了：兑现上屏 + **记账为"未朗读"**
    // （否则屏幕上是一整段、耳朵里只有前半句 —— 用户报告的就是这个）
    markUnspokenTail('barge-in');
    acc.interruptMs = Math.round(now - acc.startedAt);
    // S5.5 的判据：**从"判定为真打断"到"本地已经不响了"**（这一段全是客户端可控的），
    // 要求 < 200ms。`stopFrom()` 同步掐断，不 await、不回调。
    //
    // ⚠️ 用 `stopFrom` 而不是 `stopAll`：只清本地队列的话，**服务端仍在合成这一轮
    // 剩下的句子**（一句一个 seq，客户端在收到计划时就全发出去了），几百毫秒后
    // PCM 到齐会被照常播出来 —— 用户听到"停了一下又自己说起来"。
    // `stopFrom` 同时发 WS `cancel`（服务端从待合成队列摘掉 + 让在途合成提前结束），
    // 并让已经在路上的帧因查不到 segment 被丢弃。见 `TtsStreamSession.cancelFrom`。
    const stopAt = performance.now();
    ttsRef.current?.stopFrom(ttsSeqRef.current);
    acc.interruptStopMs = Math.round(performance.now() - stopAt);
    echoRef.current?.retireAll(now);
    setBargeIns((n) => n + 1);
    // 相位回到聆听：这样"边说边出字"的临时候选会显示出来，
    // 用户能立刻看到自己的话被听见了（本轮 SSE 仍在跑，F-4）
    applyPhase('listening');
    return true;
  }, [applyPhase, markUnspokenTail]);

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

      // ── 打断确认窗口（S5.4 修订）──────────────────────────────────────
      // 第一条非回声 partial **先不打断**：partial 是解码器"刚听到一点"就吐出来的
      // 中间结果，一声咳嗽也会先吐一条。等 `BARGE_IN_CONFIRM_MS` 才能确认
      // "用户确实在说话"（细节与取舍见常量注释）。
      if (!bargeCandidateAtRef.current) {
        bargeCandidateAtRef.current = now;
        window.clearTimeout(bargeConfirmTimerRef.current);
        bargeConfirmTimerRef.current = window.setTimeout(() => {
          bargeConfirmTimerRef.current = 0;
          bargeCandidateAtRef.current = 0;
          commitBargeIn();
        }, BARGE_IN_CONFIRM_MS);
        return;
      }
      // 候选期间又来了非回声 partial：用户还在说 —— 但**仍然等满确认窗口**，
      // 不提前打断（"打断间隔改长一点"就是这一步）。计时器刻意不重置：
      // 窗口从"第一次开口"算起，否则一个一直说话的插话者会让 AI 永远说不完。
    },
    [commitBargeIn],
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
    // 过程记录：用户的每一次开口是时间轴上的一条消息，也是新一轮的开始
    // （AI 的回复会带着同一个轮次号跟上来，页面据此按轮铺开）。
    transcriptTurnRef.current += 1;
    transcriptIdRef.current += 1;
    setTranscript((prev) =>
      appendUserMessage(
        prev,
        t,
        transcriptTurnRef.current,
        Date.now(),
        nextTranscriptId(transcriptIdRef.current),
      ),
    );
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
  const voiceFlushRef = useRef(voice.flush);
  voiceFlushRef.current = voice.flush;
  /** 对账编号：整通电话单调递增（不重置，避免日志里两轮同名） */
  const turnIdRef = useRef(0);

  /**
   * 「打断并发送」—— 用户手动抢话的按钮（控制栏那个）。
   *
   * ## 它做两件事，顺序不能反
   *
   * 1. **立刻打断**：`commitBargeIn()` 停播 + 本轮剩余句子不再出声。
   *    只有"本轮还在飞"时才有这一步；AI 没在说话时点它就是纯"发送"。
   * 2. **立刻定稿当前这一段**：`flush()` 让服务端马上把已经收到的音频定稿，
   *    **跳过端点检测那条 rule2 等待**（服务端默认 1.8 s 静音才判"说完了"）。
   *
   * ## 为什么顺序是"先打断再定稿"
   *
   * 打断会把相位切回 `listening`，并且（若处于半双工窗口）停掉送帧；
   * 而定稿依赖的音频**已经送到服务端了**，不受影响。反过来先定稿同样可行，
   * 但那样会让"用户按下按钮 → AI 还在响"多存在一次网络往返 ——
   * 手动打断的体感要求是**零等待**（与自动打断同一条 `stopAll()` 路径）。
   *
   * ## 定稿文本去哪了
   *
   * 服务端定稿后回 `final`，走的是 `handleVoiceFinal`：
   *   · 本轮还在飞（刚被打断，SSE 没收完）⇒ **排队**，等 `done` 之后作为下一轮发
   *     （F-4：不 abort、不并发 —— 这条不能因为"用户很急"而破例）；
   *   · 本轮已经结束 ⇒ 立刻开新一轮。
   * 所以按钮的语义是"**你的话一秒钟都不会丢，而且不用等静音判停**"，
   * 而不是"绕过服务端插队"。
   */
  const interruptAndSend = useCallback(() => {
    if (phaseRef.current === 'ended') return;
    // ① 手动打断（本轮没在飞时是空操作，返回 false）
    commitBargeIn();
    // ② 立刻定稿。没接通/连接没就绪时 `flush()` 返回 false —— 此时什么都不做，
    //    不弹提示：按下去没反应的最可能原因是"还没说话"，而不是出了错。
    if (voiceFlushRef.current()) setManualSends((n) => n + 1);
  }, [commitBargeIn]);

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
    const holes = ttsRef.current?.underrun;
    setTurn({
      userText: acc.userText,
      firstDeltaMs: acc.firstDeltaMs,
      firstPcmReadyMs: acc.firstPcmReadyMs,
      firstSoundMs: acc.firstSoundMs,
      queueAheadAtFirstSayMs: acc.queueAheadAtFirstSayMs,
      pendingUnitsAtFirstSay: acc.pendingUnitsAtFirstSay,
      firstPcmAtMs: acc.firstPcmAtMs,
      firstServerFirstChunkMs: acc.firstServerFirstChunkMs,
      firstServerSynthMs: acc.firstServerSynthMs,
      firstRtf: acc.firstRtf,
      speakAtMs: acc.speakAtMs,
      graceFiredAtMs: acc.graceFiredAtMs,
      firstSayAtMs: acc.firstSayAtMs,
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
      // 单轮空洞 = 累计值 − 本轮开始时的快照（见 TurnAccumulator.holeBaseCount）
      audioHoleCount: holes ? Math.max(0, holes.holes - acc.holeBaseCount) : 0,
      audioHoleMs: holes
        ? Math.max(0, Math.round((holes.totalHoleMs - acc.holeBaseMs) * 10) / 10)
        : 0,
      // 单轮"掐断未播" = 累计值 − 本轮基线（打断/改稿时故意丢掉的句数）
      audioStoppedCount: Math.max(
        0,
        (ttsRef.current?.diagnostics.stoppedWithoutAudio ?? 0) - acc.stoppedBaseCount,
      ),
      // 载荷级：本轮发出去的 `say` 句数与服务端回回来的 PCM 字节数
      says: Math.max(0, (ttsRef.current?.payloadLedger.sayFrames ?? 0) - acc.sayBase),
      pcmBytes: Math.max(0, (ttsRef.current?.payloadLedger.pcmBytes ?? 0) - acc.pcmBytesBase),
      // 本轮送合成里"太短"的单元（见 TurnMetrics.shortUnits：这是"先冒出一句乱话"的候选）
      ...(() => {
        const units = acc.saidUnits.map((u) => u.text);
        const short = units.filter((t) => t.trim().length < SHORT_UNIT_CHARS);
        return {
          shortUnits: short.length,
          shortestUnit: [...units].sort((a, b) => a.length - b.length)[0] ?? '',
          shortHeld: acc.shortHeld,
        };
      })(),
    });
    // 累计读数一并刷新（面板用；与单轮增量是两个口径，见 audioStats 的注释）
    setAudioStats(holes ?? null);
    // 载荷级累计账（"到底出没出过声音"的直接证据）
    setTtsLedger(ttsRef.current?.payloadLedger ?? null);
    // 输出侧：AudioContext 是否真的在跑、有几个块交给了声卡
    setOutputHealth(ttsRef.current?.outputHealth ?? null);
    // 播放缓冲区：句内/跨句见底次数与当前水位（卡顿病因的分界线）
    setBufferStats(ttsRef.current?.bufferStats ?? null);
    // 文本 ↔ 语音对账（A/B/C/D）：每一轮都把四份文本摆出来，差异写成人能读的清单。
    // 一条 console 也打出来 —— 真机排查时"哪一层先不一致"要能一眼看到。
    const ledger = buildVoiceTextLedger(acc, acc.subtitle.full);
    setVoiceLedger(ledger);
    if (ledger.exceptions.length > 0) {
      console.warn(`[通话][对账] 第 ${ledger.turnId} 轮：`, ledger.exceptions, {
        A_shown: ledger.shown,
        B_plan: ledger.planText,
        C_said: ledger.said,
        D_spoken: ledger.spoken,
        skipped: ledger.skipped,
        notes: ledger.notes,
      });
    }
  }, []);

  /**
   * 送合成 —— **S6.3 的最后一道保险就这里**，同时也是对账链的 C 端点。
   *
   * 一个"文本单元"可能被切成多段送合成（号码被摘掉之后前后文分开、超长句按字数硬切），
   * **每段一个独立 seq** —— 服务端按 seq 区分句子，复用 seq 会让两段音频互相覆盖
   * （`segments` 是 Map，同 seq 后写覆盖先写）。
   *
   * 台账：
   *   · 真正送出去的每段写进 `acc.saidUnits`（C：实际发出去的文本）；
   *   · 有意跳过的片段（号码 / 纯标点）写进 `acc.skippedFragments`，带原因 ——
   *     对账时算**预期例外**，不是故障。**这一步不能吞掉整句**（旧实现就会）。
   *
   * @param raw 一个合成单元（可能含号码/纯标点）
   * @returns 实际发出的 seq 列表（按序）
   */
  const speakUnit = useCallback((raw: string): number[] => {
    const tts = ttsRef.current;
    const acc = accRef.current;
    const { spoken, skipped } = splitForSpeech(raw);
    for (const s of skipped) {
      acc.skippedFragments.push(s);
      if (s.reason === 'phone') {
        console.warn('[通话] 号码片段只上屏不朗读:', s.text.slice(0, 20));
        setSkippedPhoneUnits((n) => n + 1);
      } else if (s.reason === 'unspeakable') {
        setSkippedUnspeakableUnits((n) => n + 1);
      }
    }
    const seqs: number[] = [];
    if (!tts) return seqs;
    for (const unit of spoken) {
      acc.units += 1;
      ttsSeqRef.current += 1;
      const seq = ttsSeqRef.current;
      if (acc.firstSeq === null) acc.firstSeq = seq;
      if (acc.units === 1) {
        // 第一段即将送出去之前的两条"排队"读数（**必须在 `tts.say` 之前取**：
        // `say` 之后本句也会进队列，就分不出"前面排着的"和"本句"了）。
        // 它们把"首声 6159 ms"拆成「排队」还是「合成慢」—— 见 TurnMetrics 的字段说明。
        acc.queueAheadAtFirstSayMs = tts.pendingAudioMs();
        acc.pendingUnitsAtFirstSay = tts.pendingUnits();
      }
      acc.saidUnits.push({ seq, text: unit });
      seqs.push(seq);
      // 本轮**第一个** `say` 发出的时刻（面板读数④）：用户说完 → 真正开始合成。
      // 与「首个 delta」「宽限触发」并排看，就能算出这 350ms 宽限值不值。
      if (acc.firstSayAtMs === null) {
        acc.firstSayAtMs = Math.round(performance.now() - acc.startedAt);
      }
      tts.say(seq, unit);
    }
    return seqs;
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
      // 本轮对账编号（面板/日志用它把 A/B/C/D 四份文本对上号）
      turnIdRef.current += 1;
      acc.turnId = turnIdRef.current;
      acc.planMode = 'pending';
      // 新一轮开始：上一轮的"未朗读标注"清空（它属于上一轮）
      setUnspokenTailState(null);
      // 本轮空洞计数的基线（见 TurnAccumulator.holeBaseCount）
      const holeBase = ttsRef.current?.underrun;
      acc.holeBaseCount = holeBase?.holes ?? 0;
      acc.holeBaseMs = holeBase?.totalHoleMs ?? 0;
      // "被我们掐断未播"的基线（口径同上；这一格用来区分有意丢弃与合成失败）
      acc.stoppedBaseCount = ttsRef.current?.diagnostics.stoppedWithoutAudio ?? 0;
      // 载荷级基线（见 TurnMetrics.pcmBytes）：本轮"发了几句 / 回了多少音频字节"的起点
      acc.sayBase = ttsRef.current?.payloadLedger.sayFrames ?? 0;
      acc.pcmBytesBase = ttsRef.current?.payloadLedger.pcmBytes ?? 0;
      accRef.current = acc;
      mutedByReviseRef.current = false;
      mutedByBargeInRef.current = false;
      bargedInRef.current = false;
      // 上一轮如果还挂着"等确认窗口"的定时器，必须先清掉：它属于上一轮，
      // 晚了 600ms 触发会把这一轮刚开的口误判成插话打断（跨轮串味的来源）。
      window.clearTimeout(bargeConfirmTimerRef.current);
      bargeConfirmTimerRef.current = 0;
      bargeCandidateAtRef.current = 0;
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
        // ── 本轮的音频裁决（**只报事实，不改行为**）──────────────────────────
        // "屏幕上有字、耳朵里没声音"这件事在时间读数上是一片 `—`，只能靠载荷数判：
        // `says > 0` 而 `pcmBytes === 0` ⇒ 服务端一个字节都没回（合成/OOV/取消），
        // 这条日志就是那件事的第一手证据（真机排查 2026-10-11）。
        {
          const led = ttsRef.current?.payloadLedger;
          if (led) {
            const says = Math.max(0, led.sayFrames - acc.sayBase);
            const bytes = Math.max(0, led.pcmBytes - acc.pcmBytesBase);
            const dropped = Math.max(0, led.droppedFrames);
            const shown = acc.subtitle.full.length;
            if (says > 0 && bytes === 0) {
              // 用 `error` 级：控制台默认要展开看警告，而这条是"没有声音"的第一手判据，
              // 必须一眼可见、能直接复制给排查的人（见 docs/voice_call_no_audio_probe.md）。
              console.error(
                `[无声音排查] 第 ${acc.turnId} 轮：发出 ${says} 句合成请求，服务端回了 0 字节音频。`
                + `字幕 ${shown} 字 · 累计丢弃帧 ${dropped} · 请把这一行和诊断面板一起发出`,
              );
            } else if (says > 0) {
              const out = ttsRef.current?.outputHealth;
              console.log(
                `[通话音频] 第 ${acc.turnId} 轮：发出 ${says} 句 · 收到 ${bytes} 字节`
                + `（约 ${Math.round(bytes / 48)} ms @24k/16bit）· 字幕 ${shown} 字`
                + `${out ? ` · 输出 ${out.ctxState}/已排程 ${out.scheduledChunks} 块` : ''}`,
              );
            }
          }
        }
        // 本轮结束：确认窗口作废（否则它会在下一轮里凭空打断一次）
        window.clearTimeout(bargeConfirmTimerRef.current);
        bargeConfirmTimerRef.current = 0;
        bargeCandidateAtRef.current = 0;
        activeTurnsRef.current = Math.max(0, activeTurnsRef.current - 1);
        publishTurn();
        resolveDone();
        settleToListening();
        // 插话排队：**顺序**发下一轮，绝不并发（S5.5 闸门项）
        const next = pendingRef.current.shift();
        if (next && phaseRef.current !== 'ended') runTurnRef.current(next);
      };

      applyPhase('thinking');

      // ── 开场预缓冲：**每轮只生效一次**的"欠账"就在这里挂上 ──────────────
      // 判据与反例见 `services/prebufferGate.ts`：绝不能挂在"队列播空"上
      // （阵发式供音的间隙里队列也会空，那会让每次断档后又多等一整个预缓冲）。
      ttsRef.current?.armPrebufferTurn();

      // ── 提取声学特征摘要，随本轮消息一起上报（供风险研判）──────────────
      // 只取聚合统计量，不传原始帧数据（体积大且含 PII）。
      const vm = voiceMetricsRef.current;
      const voiceFeatures: Record<string, unknown> | undefined = vm ? {
        pitch: { mean: vm.pitch.mean, std: vm.pitch.std, range: vm.pitch.range },
        energy: { mean: vm.energy.mean, std: vm.energy.std, peakRatio: vm.energy.peakRatio },
        rate: { speechRate: vm.rate.speechRate, tempo: vm.rate.tempo, regularity: vm.rate.regularity },
        pauses: { ratio: vm.pauses.ratio, count: vm.pauses.count, avgDuration: vm.pauses.avgDuration, longPauseCount: vm.pauses.longPauseCount },
        spectral: { centroid: vm.spectral.centroid, rolloff: vm.spectral.rolloff, flatness: vm.spectral.flatness },
        riskScore: vm.riskScore,
        detectedEmotion: vm.detectedEmotion,
      } : undefined;

      // ── 本轮的面部实时状态（**只在用户开了摄像头时才有**）────────────────────
      // 与声学特征同一口径：只传聚合快照，不传原始帧、不传图像。
      // 拿不到（没开摄像头 / 画面没人脸 / 读数过期）就是 undefined —— 本轮不带该字段，
      // 算法层也就不注入面部那一段。**绝不用中性值兜底**。
      // ⚠️ 它只是算法层的低权重内部参考，不进风险等级、不进融合权重。
      const facialFeatures = getFacialLiveState?.() ?? undefined;

      const pendingTurn = sendChatMessageStream(convId, userText, {
        onMeta: (meta) => {
          acc.sawMeta = true;
          // F-3：零新增阈值 —— 高风险时 `streaming` 恒为 false
          acc.voiceAllowed = meta.streaming && !crisisRef.current;
          if (!meta.streaming) enterCrisis(acc);
        },
        onSpeak: (plan) => {
          // 定稿合成计划：**按它合成、也按它上屏**。
          //
          // ## 服务端的真实时序（2026-10-11 核对，与"计划在第一个 delta 之前"的说法相反）
          //
          // `_chat_stream_events` 先流式吐 delta（LLM 逐字），**等整段生成完**才调
          // `_plan_streamed_reply` 定稿、再下发 `speak`。所以实际顺序是：
          //
          //     meta → delta* → speak{text,segments} → [revise] → done
          //
          // 也就是说：计划总是**最后一个 delta 之后几十毫秒**才到，而 delta 可能已经
          // 流了好几秒。`SPEAK_PLAN_GRACE_MS`（350ms）只够覆盖"计划紧跟在 delta 后面"
          // 的短回复；长回复必然先走 delta 合成。**所以"同一轮只能有一个合成来源"
          // 必须显式管起来**（见 planModeRef / SpeakPlanMode），否则同一段文本会被
          // 合成两遍（delta 一遍 + 计划一遍），用户听到"这句话念两遍"。
          //
          // 现在三条规则：
          //   ① `pending`（delta 还没送出去）→ 计划成为唯一来源，扣住的 delta 作废；
          //   ② `delta-fallback`（delta 已经送出去了）→ **绝不把计划再入队**；
          //      若定稿文本是"已发文本的前缀"（删尾），就把多出来的尾段撤掉
          //      （`unitsToCancelForPlan` + `stopFrom`），让听到的回到定稿文本；
          //   ③ 计划自相矛盾（`segments` 拼接 ≠ `text`）→ 记录并**拒收**，
          //      不能让一份对不上的计划驱动 TTS。
          const acc2 = accRef.current;
          // `speak` 计划**到达**的时刻（面板读数②）—— 协议上它总在全部 delta 之后到，
          // 记下来就能和「宽限触发」比：计划先到 ⇒ 宽限有用；后到/没到 ⇒ 纯等待。
          if (acc2.speakAtMs === null) {
            acc2.speakAtMs = Math.round(performance.now() - acc2.startedAt);
          }
          const mode = acc2.planMode;

          // ③ 契约校验：对不上就拒收（记录原因，别静默）
          if (!planJoinMatchesText(plan.text, plan.segments)) {
            acc2.ledgerNotes.push('speak 计划违约：segments 拼接 ≠ text，已拒收（退回 delta 路径）');
            console.warn('[通话] speak 计划自相矛盾，已拒收:', {
              text: plan.text, segments: plan.segments,
            });
            return;
          }
          if (acc2.sawRevise) {
            // 改稿先到：那一轮的真话已经在 `onRevise` 里按新稿念过了。
            // 这份计划是**改稿前**的文本，绝不能入队（否则旧说法会被念一遍），
            // 也不能拿它去撤回 —— 它和补念的尾段本就对不上。
            acc2.ledgerNotes.push('speak 计划在改稿之后才到：已按改稿补念，未再入队');
            return;
          }
          if (mode !== 'pending') {
            // ② 计划迟到：不再入队，但可以用它把"多念的尾段"收回来
            acc2.ledgerNotes.push(`speak 计划迟到（当前来源=${mode}），未再入队`);
            acc2.planText = plan.text;
            acc2.planSegments = plan.segments;
            const cancelFrom = unitsToCancelForPlan(acc2.saidUnits, plan.text);
            if (cancelFrom !== null) {
              acc2.ledgerNotes.push(`按定稿文本撤回 seq≥${cancelFrom} 的多余尾段`);
              ttsRef.current?.stopFrom(cancelFrom);
              // 只把**确实没出声**的那些段从台账里划掉：
              // 已经进了 D（真的响过）的段留在 C 里，否则 C≠D 会误报成"串单"。
              const heard = new Set(acc2.spokenUnits.map((u) => u.seq));
              acc2.saidUnits = acc2.saidUnits.filter((u) => u.seq < cancelFrom || heard.has(u.seq));
            }
            return;
          }

          // ① 正常路径：计划就是唯一来源
          acc2.planMode = 'speak-plan';
          acc2.planText = plan.text;
          acc2.planSegments = [...plan.segments];
          speakPlanRef.current = plan;
          speakCursorRef.current = 0;
          // 计划到了：撤掉"按 delta 送合成"的宽限定时器与扣住的文本 ——
          // 扣住的 delta 一个字都没送出去，直接作废（这正是"只能有一个来源"）。
          window.clearTimeout(deferTimerRef.current);
          deferTimerRef.current = 0;
          deferredDeltaRef.current = '';
          acc2.subtitle.full = plan.text;
          acc2.subtitle.shownChars = plan.text.length;
          setAiLines([plan.text]);
          // 过程记录：定稿计划就是本轮回复的**全文**（整体替换尾部那条 AI 消息）
          pushAiTranscript(plan.text, 'replace');
          const willSpeak = acc2.voiceAllowed && !mutedByReviseRef.current
            && !mutedByBargeInRef.current && !crisisRef.current
            && ttsStatusRef.current?.available === true;
          if (!willSpeak) return;   // 纯文字模式：字幕已在上方上屏，没有音频要送
          spokeFromPlanRef.current = true;
          // 立即按计划送合成 —— 不再等 delta，也不再按 delta 切分
          for (const seg of plan.segments) {
            speakCursorRef.current += 1;
            speakUnit(seg);
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
          // 有合成计划（或计划已迟到处置过）时：delta 只用来做"首字延迟"之类的观测，
          // **不再**驱动字幕与合成 —— 同一轮只能有一个合成来源（见 onSpeak 的注释）。
          if (acc.planMode === 'speak-plan') return;

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
            // 过程记录：没有音频驱动的纯文字路径 —— 每个 delta 都是本轮回复的**增量**
            pushAiTranscript(text, 'delta');
          }

          if (!willSpeak) return;

          // 已经按 `speak` 计划送过合成的轮次，delta **永久**不许再送（见 spokeFromPlanRef）
          if (spokeFromPlanRef.current) return;

          const flushByDelta = (target: TurnAccumulator, allowShort = false) => {
            deferTimerRef.current = 0;
            // 宽限计时器**触发**的时刻（面板读数③）—— 与 `speakAtMs` 比就知道
            // 这 350ms 到底等到了计划，还是纯等待（见 TurnMetrics 的说明）。
            if (target.graceFiredAtMs === null) {
              target.graceFiredAtMs = Math.round(performance.now() - target.startedAt);
            }
            const pending = deferredDeltaRef.current;
            deferredDeltaRef.current = '';
            if (!pending) return;
            // ⚠️ 宽限期内**改稿（或计划）已经把这一轮的字送出去合成过了**：
            // 这份扣住的 delta 是改稿前的旧文本，再送一遍就是"同一段念两遍"
            // （用户口径：每一次输出的内容都要合成，但**不是**每份来源都合成）。
            // 判据用 `spokeFromPlanRef`（改稿/计划路径会置上），与 `onDelta` 里
            // 哨兵 `if (spokeFromPlanRef.current) return` 同一口径。
            if (spokeFromPlanRef.current) {
              target.ledgerNotes.push('宽限期扣住的 delta 已被改稿/计划接管，未再送合成');
              return;
            }
            // ── 太短的单元**不单独送合成**（2026-10-11 "AI 偶尔说出乱话"的候选真因）────
            // delta 是逐字到达的：宽限期到点时手上可能只有「嗯」「不过」这么两三个字，
            // 而**一个独立合成单元**会把它们念成一句突兀的短话 ——
            // 听感正是用户报的"先说一句莫名其妙的，然后才说正文"。
            // ⚠️ 什么都不丢：把短文本**放回队列**、再等一个宽限期凑齐；
            // 若一直凑不齐，`allowShort`（第二次触发）时照样送出去，文本一个字都不少。
            const short = pending.trim().length < MIN_FIRST_UNIT_CHARS;
            if (short && !allowShort && !target.shortHeld) {
              target.shortHeld = true;
              target.ledgerNotes.push(
                `首段只有 ${pending.trim().length} 字（<${MIN_FIRST_UNIT_CHARS}），先不送合成、等凑齐`,
              );
              deferredDeltaRef.current = pending;
              deferTimerRef.current = window.setTimeout(
                () => flushByDelta(target, true),
                graceMsRef.current,
              );
              return;
            }
            if (short && allowShort) {
              target.ledgerNotes.push(`首段仍只有 ${pending.trim().length} 字，为不丢字照原样送合成`);
            }
            // 宽限期到点仍没有计划 ⇒ 本轮正式进入 delta-fallback（此后计划只会用来收尾）
            target.planMode = 'delta-fallback';
            target.ledgerNotes.push(
              `speak 计划未在 ${graceMsRef.current}ms 内到达，音频改由 delta 驱动`,
            );
            for (const unit of splitSpeakable(pending, target.units === 0)) {
              if (!unit.trim()) continue;
              speakUnit(unit);
            }
          };

          // 第一个 delta 到达后先扣住一小段时间等 `speak` 计划；之后（宽限期已过、
          // 计划仍没来）就回到原来的"边到边送"，延迟不变。
          // ⚠️ `spokeFromPlanRef` 已置上（本轮的音频已由改稿/计划驱动）时**不要**再
          // 起这个定时器：那份 delta 永远不会被送出去，留着只会白等 350ms。
          if (deferTimerRef.current === 0 && acc.planMode === 'pending' && !spokeFromPlanRef.current) {
            deferredDeltaRef.current += text;
            deferTimerRef.current = window.setTimeout(() => flushByDelta(acc), graceMsRef.current);
            return;
          }
          if (acc.planMode === 'pending' && deferredDeltaRef.current && !spokeFromPlanRef.current) {
            deferredDeltaRef.current += text;
            return;
          }
          acc.planMode = 'delta-fallback';
          for (const unit of splitSpeakable(text, acc.units === 0)) {
            if (!unit.trim()) continue;
            speakUnit(unit);
          }
        },
        onRevise: (text) => {
          // ── 用户口径（2026-10-11）：「**每一次输出的内容都需要语音合成**」────────
          // 所以旧的"停播 + 只改字幕、不重念"被换掉：屏幕上有的字，耳朵里必须有。
          // 但整段重念会让用户把开头听第二遍，于是做**前缀对齐**：
          //   ① 保留"已经**完整播完**、且仍是新稿前缀"的那几段 —— 不重念；
          //   ② 从第一段"没播完 / 与新稿对不上"的地方起全部撤回（含它本身，
          //      它后半句没听到，重念比吞字安全）；
          //   ③ 剩余部分按正常 `speakUnit` 路径补念。
          // 判据与边界见 `ttsSplit.ts::planReviseSpeech` 与真机回归
          // `scripts/verify_tts_text_ledger.ts`（组 5）。
          acc.sawRevise = true;
          const canSpeak =
            acc.voiceAllowed &&
            !crisisRef.current &&
            !mutedByBargeInRef.current &&      // 用户已经在插话：别再抢回话轮
            ttsRef.current !== null &&
            ttsStatusRef.current?.available === true;

          // 改稿文本整体取代已到达的流式文本（"内容不藏"：屏幕上的字一个不丢）
          acc.subtitle.full = text;
          acc.subtitle.shownChars = text.length;

          if (!canSpeak) {
            // 纯文字模式 / 已被打断 / 危机切换：本轮本来就没有音频，如实标注即可
            mutedByReviseRef.current = true;
            acc.ledgerNotes.push('安全审计改稿：本轮没有语音（纯文字或已切换），新稿只上屏');
            acc.unspokenTail = null;
            ttsRef.current?.stopFrom(ttsSeqRef.current);
            echoRef.current?.retireAll(performance.now());
            setAiLines([text]);
            setRevised(true);
            // 过程记录：改稿后的新稿整体替换本轮那条 AI 消息（事实位记下来）
            pushAiTranscript(text, 'replace', { revised: true });
            return;
          }

          const plan = planReviseSpeech(acc.saidUnits, text);
          if (plan.cancelFrom !== null) {
            // 停播 + 中止在途合成（服务端从待合成队列摘掉 + 让在途合成提前结束）
            ttsRef.current?.stopFrom(plan.cancelFrom);
            acc.ledgerNotes.push(`改稿：从 seq≥${plan.cancelFrom} 撤回未播完/对不上的段`);
          }
          acc.ledgerNotes.push(
            plan.keptChars > 0
              ? `改稿：前 ${plan.keptChars} 字沿用已播出的说法，不重念`
              : '改稿：没有已听全的前缀，新稿从头补念',
          );
          echoRef.current?.retireAll(performance.now());
          // 改稿之后的语音由这里驱动：delta 不许再送（否则新稿和 delta 稿会各念一遍）
          spokeFromPlanRef.current = true;
          setAiLines([text]);
          setRevised(true);
          // 过程记录：同上 —— 改稿全文整体替换本轮那条 AI 消息
          pushAiTranscript(text, 'replace', { revised: true });
          // ③ 补念剩余部分：走同一条 `speakUnit`（新 seq、进 C/D 台账、同样受回声过滤）
          const tail = plan.tail;
          if (!tail.trim()) {
            acc.ledgerNotes.push('改稿：新稿已被已播内容覆盖，无需补念');
            return;
          }
          // 补念的文本也交给同一个切分器：号码照样只上屏、超长照样按字数切
          for (const unit of splitSpeakable(tail, acc.units === 0)) {
            if (unit.trim()) speakUnit(unit);
          }
          acc.ledgerNotes.push(`改稿：未播出的剩余部分已按新稿补念（${tail.length} 字）`);
          // 补念的音频同样要驱动字幕定位（shownChars 已在上面推到全文，不会回退）
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
      }, 'voice', voiceFeatures, facialFeatures);
      // ⚠️ 传输层失败（响应头之前就 404/401/500，或压根没连上）不会走上面的 onError，
      //    只会让 `completed` reject。不接住的话本轮既不落通知也不复位状态 ——
      //    界面永远停在"正在思考…"，录音却还在继续（实测就是这样"卡死"的）。
      void pendingTurn.completed.catch((err: unknown) => {
        if (!acc.active) return; // onError 已经收过尾（流内 error 事件），不要重复 finish
        setNotice(describeStreamFailure(err));
        finish();
      });
      // 句柄只用来接 `completed`；**刻意不调用** `handle.abort()`：见文件头取舍 1（F-4）

      /** 危机分支：立刻静音 + 会话切纯文字（S6.2） */
      function enterCrisis(turnAcc: TurnAccumulator): void {
        turnAcc.voiceAllowed = false;
        crisisRef.current = true;
        setCrisisTriggered(true);
        // 1) 立刻 stopAll + 中止在途合成（S6.2 第 1 步：客户端从没做过后半句，
        //    于是危机轮的剩余句子仍会被合成推来并播出去 —— 见 TtsStreamSession.cancelFrom）
        ttsRef.current?.stopFrom(ttsSeqRef.current);
        echoRef.current?.retireAll(performance.now());
        // 缓冲的字幕不会再有音频了，立刻兑现 + 记账为"未朗读"
        markUnspokenTail('crisis');
        // 2) 会话切纯文字；麦克风保持可用（用户还要说话）
        applyPhase('crisis');
        setNotice('这一轮涉及较高风险，已暂停语音，继续用文字陪伴你。');
      }
    },
    [applyPhase, conversationId, publishTurn, settleToListening, speakUnit, markUnspokenTail, getFacialLiveState, pushAiTranscript],  );
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
          // 对账链的 **D**：这一段**真的出声了**、实际念的是 `spoken_text`（服务端术语替换后的文本）.
          // 按出声顺序追加 —— `said`（C）与 `spoken`（D）不一致就是"发出去的和念出来的不是一回事"。
          if (acc.spokenUnits.length < 200) acc.spokenUnits.push({ seq, text });
          if (!mutedByReviseRef.current && !mutedByBargeInRef.current && !crisisRef.current) {
            const before = acc.subtitle.shownChars;
            const after = advanceSubtitleOnSegment(acc.subtitle, text);
            if (after > before) {
              setAiLines((prev) =>
                prev.length === 0 ? [acc.subtitle.full] : [...prev.slice(0, -1), acc.subtitle.full],
              );
              // 过程记录：字幕随音频**逐段**揭示 —— 每次揭示都是"本轮回复的当前全文"
              // （整体替换，不是新增一条），与上面 setAiLines 的语义严格一致。
              pushAiTranscript(acc.subtitle.full, 'replace');
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
          const firstPcm = tts.takeFirstPcmMs(seq);
          const sound = tts.takeFirstSoundMs(seq);
          if (pcm !== undefined) acc.firstPcmReadyMs = pcm;
          if (firstPcm !== undefined) acc.firstPcmAtMs = firstPcm;
          if (sound !== undefined) acc.firstSoundMs = Math.round(sound);
          if (pcm !== undefined || firstPcm !== undefined || sound !== undefined) publishTurn();
        },
        onSegment: (meta) => {
          // 服务端 `end` 帧自报的合成读数（请求→第一块 / 整段 / RTF）——**只记不判断**。
          // ⚠️ 必须用 `meta.seq === firstSeq` 认领：`end` 可能在**下一轮已经开始之后**
          //    才到（服务端串行合成），不认领就会把上一段的读数记到新轮上。
          const a = accRef.current;
          if (a.firstSeq === null || meta.seq !== a.firstSeq) return;
          if (a.firstServerFirstChunkMs === null) a.firstServerFirstChunkMs = meta.firstChunkMs;
          if (a.firstServerSynthMs === null) a.firstServerSynthMs = meta.synthMs;
          if (a.firstRtf === null) a.firstRtf = meta.rtf;
          const pcm = tts.takePcmReadyMs(meta.seq);
          if (pcm !== undefined && a.firstPcmReadyMs === null) a.firstPcmReadyMs = pcm;
          publishTurn();
          // 面板要能"边听边看"：每句收尾都刷新一次载荷级账、输出侧与缓冲区读数
          setTtsLedger(tts.payloadLedger);
          setOutputHealth(tts.outputHealth);
          setBufferStats(tts.bufferStats);
        },
        onEnd: (_seq, text) => {
          // 这一句说完了：再记 2 秒就退出回声窗口（`ECHO_TAIL_MS`）
          echoRef.current?.retire(text, performance.now());
        },
        onDrained: handleDrained,
        onError: (message) => setNotice(`语音合成出错：${message}`),
        },
        // 开场预缓冲（见 UseVoiceCallOptions.prebufferMs 与 ttsPlayer 的实测依据）
        prebufferMs,
      );
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
  }, [applyPhase, conversationId, handleDrained, isParrotingUser, openHalfDuplexWindow, publishTurn, prebufferMs]);

  const endCall = useCallback(async () => {
    applyPhase('ended');
    window.clearTimeout(forwardingTimerRef.current);
    forwardingTimerRef.current = 0;
    // 挂断：打断确认窗口也一并作废（否则它会在通话结束后再"打断"一次）
    window.clearTimeout(bargeConfirmTimerRef.current);
    bargeConfirmTimerRef.current = 0;
    bargeCandidateAtRef.current = 0;
    // 挂断时把"等 speak 计划"的宽限定时器也停掉：迟到的 flush 只会往已结束的
    // 通话里塞音频（`ttsSay` 对已 dispose 的通道是空操作，但没必要留着跑）。
    window.clearTimeout(deferTimerRef.current);
    deferTimerRef.current = 0;
    deferredDeltaRef.current = '';
    echoRef.current?.clear();
    pendingRef.current = [];
    // 挂断时把还没来得及随音频上屏的字幕全部兑现（用户挂断前可能还有没看到的文字），
    // 同时记账为"未朗读" —— 挂断后那半句确实不会再出声了。
    if (accRef.current.active) markUnspokenTail('hangup');
    else flushSubtitleRemainder(accRef.current.subtitle);
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
      // 过程记录：挂断前冲刷出来的最后一句同样是这条时间轴上的一条用户消息
      transcriptTurnRef.current += 1;
      transcriptIdRef.current += 1;
      setTranscript((prev) =>
        appendUserMessage(
          prev,
          trimmed,
          transcriptTurnRef.current,
          Date.now(),
          nextTranscriptId(transcriptIdRef.current),
        ),
      );
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
  }, [applyPhase, markUnspokenTail, flushSubtitleRemainder]);

  // ── 卸载清理（幂等）────────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      window.clearTimeout(forwardingTimerRef.current);
      forwardingTimerRef.current = 0;
      window.clearTimeout(bargeConfirmTimerRef.current);
      bargeConfirmTimerRef.current = 0;
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
    transcript,
    revised,
    notice,
    dismissNotice: () => setNotice(null),
    voiceMetrics: voice.metrics,
    ttsStatus,
    discardedFinals,
    droppedEchoes,
    lastEchoReason,
    bargeIns,
    manualSends,
    skippedPhoneUnits,
    skippedUnspeakableUnits,
    parrotSkips,
    crisisTriggered,
    micSettings,
    maxConcurrentTurns,
    lastRisk,
    turn,
    ttsLedger,
    outputHealth,
    bufferStats,
    /**
     * 导出缓冲区日志明细：既写控制台（可直接复制），也把字符串返回给调用方
     * （页面上的「导出缓冲日志」按钮把它渲染在面板里，供截图）。
     */
    dumpBufferLog: (recentChunks = 12) => {
      const text = ttsRef.current?.bufferLogDump(recentChunks) ?? '（通道未建立，暂无缓冲日志）';
      console.log(text);
      return text;
    },
    audioStats,
    voiceLedger,
    unspokenTail: unspokenTailState,
    startCall,
    endCall,
    interruptAndSend,
    aiSpeaking: phase === 'speaking',
    forwardingPaused,
  };
}
