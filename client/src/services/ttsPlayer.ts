/**
 * TTS 播放队列（Web Audio）
 *
 * 职责单一：**把一段段 PCM 按顺序、无缝地播出来，并能被立刻掐断。**
 * 网络与协议在 `ttsStream.ts`，两者由 `TtsChannel` 组合。
 *
 * ## 为什么用 `AudioBufferSourceNode` 而不是 `<audio>`
 *
 * 1. **`<audio>` 每句之间有可听的接缝** —— 通话里一秒钟一个缝，很突兀。
 *    用 `nextStartTime` 排程则是采样级连续的。
 * 2. **`AudioBufferSourceNode.stop()` 是采样级精确的**，`<audio>.pause()` 有延迟。
 *    打断的体感全靠这个。
 * 3. **`<audio>` 会被系统媒体会话接管**（锁屏控制、蓝牙路由、音量键），
 *    行为不可控；Web Audio 不会。
 * 4. 需要 `recentSpoken()` —— "我刚刚说了什么"。这是文本级自回声过滤
 *    （`voice_call_plan.md` §6.2 的方案 D）唯一的输入，用自己的播放器才拿得到。
 *
 * ## 采样率
 *
 * 模型的输出是 24 kHz（CosyVoice3），而设备通常是 44.1/48 kHz。这里**不**把 AudioContext
 * 钉在 24 kHz（部分设备会直接构造失败），而是把 24 kHz 的 PCM 装进
 * `AudioBuffer`，由浏览器在播放时重采样。`buffer.duration` 仍是它自己的
 * 自然时长，所以排程计算不受影响。
 *
 * ## 与回声消除（AEC）的关系
 *
 * 音频走 `ctx.destination`，也就是**默认输出设备** —— 这正是 Chromium 的 AEC
 * 取参考信号的地方，所以播放路径本身不会让 AEC 失效。
 * ⚠️ 但 AEC 是否真的生效仍要以 `track.getSettings().echoCancellation` 为准，
 * **不要假设**（`voice_call_plan.md` §6.1）。
 */

import { TtsBufferLog } from './ttsBufferLog';
import { PrebufferGate } from './prebufferGate';

/** 某一句的出声时刻 */
export interface TtsStartTiming {
  /**
   * 排程的播放时钟起点（毫秒，**AudioContext 域**）。
   *
   * 这是精确值 —— 它是排程时算出来的，不受回调触发时刻影响。
   * 校验"句间无缝隙"必须用它：在 `onStart` 里现读 `currentTime` 会带上
   * `setTimeout` 的抖动（无头浏览器实测 ±35ms），把精确的排程误判成有缝。
   */
  ctxStartMs: number;
  /**
   * 预估的墙钟出声时刻（毫秒，`performance.now` 域）。
   *
   * ⚠️ 只是**近似**：早期 AudioContext 的时钟可能还没起步
   * （实测 `#1 enqueue@1048(ctx 0)` —— 第一个 source 开始渲染前 ctx 恒为 0），
   * 此时按"ctx 与墙钟 1:1"外推会凭空多出几百毫秒。
   * 用于"用户感知的首声延迟"这类指标是可以的，不要用它做时序校验。
   */
  startsAtWallMs: number;
}

/** 播放器事件 */
export interface TtsPlayerCallbacks {
  /**
   * 某一句**真正开始出声**（不是排程时刻）。
   *
   * 为什么要区分：非首句是在 `nextStartTime` 排到未来的，排程那一刻还没出声。
   * 半双工窗口（S5 的 400ms 静音期）与端到端首声测量都必须按**真实出声**算，
   * 否则窗口会提前开在上一句还在播的时候。
   */
  onStart?: (seq: number, text: string, timing: TtsStartTiming) => void;
  /** 某一句播完（文本用于回声过滤登记） */
  onEnd?: (seq: number, text: string) => void;
  /**
   * 队列彻底空了。
   *
   * ⚠️ **自然播完**与**被 `stopAll()` 掐断**都会触发 —— 因为两种情况下列队
   * 都是空的，这是事实而不是事件。调用方（`useVoiceCall`）必须把它当成
   * 幂等的状态收敛，而不是"播完了"的独有信号。
   */
  onDrained?: () => void;
}

/** 正在播/刚播过的一句 */
interface SpokenEntry {
  seq: number;
  text: string;
  /** 开始播放的墙钟时间（`performance.now()`） */
  startedAt: number;
}

/**
 * 播放**空洞（欠载）**的实测统计。
 *
 * ## 为什么要有这个读数（2026-10-11）
 *
 * 用户的反馈是"AI 电话有时候还是会卡顿"。卡顿有三种完全不同的病因，靠猜分不开：
 *
 * | 读数 | 说明 |
 * |---|---|
 * | `holes` / `totalHoleMs` | **音频空洞**：一块音频来晚了，播放头已经越过上一块的结尾 → 耳朵里的"卡一下" |
 * | ASR 定稿延迟 | 说话被截断/判停晚（看 `?debug=1` 的定稿句数、端点参数） |
 * | 首声/首 delta | 整轮反应慢（看 `firstSoundMs`），不是音频断裂 |
 *
 * 这里的判定是**纯客户端、纯时间轴**的：`enqueueChunk` 时若"本该接上的位置"
 * 已经在 `ctx.currentTime` 之前，那就是一段没人补的空洞 —— 它的时长就是
 * `currentTime` 与那个位置之差。**不需要听音频**，所以能在真机上直接量。
 *
 * 只统计"流中间"的空洞：队列本来就空（上一句已经自然播完）不算 ——
 * 那时用户听到的是刻意的留白，不是卡顿。
 */
export interface TtsUnderrunStats {
  /** 发生过的空洞次数（块 ≥ 1 块来晚） */
  holes: number;
  /** 空洞累计时长（毫秒）—— 用户听到的"卡顿"总量 */
  totalHoleMs: number;
  /** 单次最长空洞（毫秒） */
  maxHoleMs: number;
  /** 最近一次空洞的时长（毫秒）；没有过为 null */
  lastHoleMs: number | null;
  /** 最近一次空洞发生在哪一句（seq）；没有过为 null */
  lastHoleSeq: number | null;
  /** 空洞总时长占已播音频总时长的比例（0~1）；没有音频时为 0 */
  holeRatio: number;
}

/** 一个**音频块**在队列里的排程记录（一句 = 若干块） */
interface ChunkEntry {
  seq: number;
  source: AudioBufferSourceNode;
  /** 这一块自己的增益节点（跨句交叉淡化用；永远是 `source → gain → destination`） */
  gain: GainNode;
  /** 这一块的音频时长（秒，用于限制交叠量） */
  durationSec: number;
  /** 排程的结束时刻（`ctx.currentTime` 域，秒） */
  endsAtCtx: number;
  /** 是否已经播完/被掐断，不再是可交叠的"上一块" */
  finished: boolean;
}

/**
 * 一句（seq）的聚合状态。
 *
 * ⚠️ 存在的理由：**音频要一块一块地播，但 `onStart`/`onEnd`/`recentSpoken` 必须
 * 按"句"上报**。服务端一块就有 0.17~4 s 音频，一句有十几到几十块；
 * 若按块触发回调，字幕会重复揭示、回声过滤会重复登记、半双工窗口会被反复重开。
 */
interface OpenSegment {
  seq: number;
  text: string;
  /** 已排程、还没播完的块数 */
  live: number;
  /** 累计收到多少块（0 表示这一句没合成出音频） */
  chunks: number;
  /** 是否已收到 `end`（之后不会再有新块） */
  ended: boolean;
  /** `onStart` 是否已触发 */
  started: boolean;
  /** `onEnd` 是否已触发 */
  finished: boolean;
  /** 到点触发 `onStart` 的定时器；已触发或已清理时为 0 */
  startTimer: number;
  /** 墙钟时间的预计开始/结束时刻（`performance.now()` 域），供 `recentSpoken` 用 */
  startsAt: number;
  endsAt: number;
}

/** `recentSpoken()` 默认回溯窗口：覆盖"AI 刚说完"到"用户插话被识别"的间隔 */
const DEFAULT_SPOKEN_WINDOW_MS = 8000;

/** 文本记录上限，防长通话里无限增长 */
const MAX_SPOKEN_LOG = 40;

// ── 逐块播放与段间接缝 ────────────────────────────────────────────────────
//
// ## 为什么必须逐块播（2026-10-05 实测）
//
// 之前是"攒完整段（收到 end）才 `enqueue` 一次"。而服务端**串行合成**：一段 3 句的
// 回复里，第 1 句播完时第 2 句还没合成完。实测真实链路（`_probe_playback_timeline.py`）：
//
//     现状（攒整段播）  首声 2558 ms   句中空洞 2 处 / 合计 1911 ms（686 + 1225 ms）
//     逐块播            首声 1635 ms   句中空洞 1 处 / 合计  141 ms
//
// 那 1~2 秒的空洞就是用户听到的"句子中间卡顿"。逐块播等于把服务端已经流式出来的
// 音频立刻排上，不再等整段合成完。
//
// ## 块与句的关系
//
// * **同一句内部的块**是同一段连续波形的前后片段（服务端逐块 yield 的），
//   必须**严丝合缝**地接上：不重叠、不淡化（重叠会吃掉采样）。
// * **跨句的接缝**才是两次独立合成，波形相位不连续 —— 那里做交叉淡化。
//
// ## 关于一句尾部的静音
//
// 早先的探针脚本报过"段尾静音 640 ms、接缝处共 820 ms 死气"，**那是脚本 bug**
// （静音检测把 RMS 写成了 |均值|，而 |均值| 对零均值音频天然偏小，把语音也误判成
// 静音）。真 RMS 重测（`_probe_seam_recheck.py`）：真实单元尾部静音只有 0~150 ms，
// 接缝处 max|一阶差分| 落在全段分布的 18%~36%（click 判据 >99.5%）。
// 所以**不做静音裁剪** —— 能裁的只有几十毫秒，却要白担一次啃掉弱音节的风险。

/** 跨句交叉淡化时长（毫秒）。取很短：两句是独立合成，交叠太久会互相盖住 */
const CROSSFADE_MS = 12;

/**
 * **开场预缓冲默认值**（毫秒）：第一块音频到达之后，先攒够这么长的排程再出声。
 *
 * ## 为什么需要它（2026-10-11 实测，`scripts/probe_tts_supply.mjs`）
 *
 * 原先的模型是"**收到第一块就立刻排到 `ctx.currentTime`**"，于是第一块一响，
 * 播放头就开始消耗；而 CosyVoice3 的**供音是阵发式的**：
 *
 * ```
 * 一句 11 字：首块 1.55s 到（音频 2.4s），随后块间隔 0.6 ~ 1.5s
 * 三句回复：块间隔 1.3 ~ 2.4s，整轮有效 RTF 0.55（平均快于播放，但每阵只补 0.3~0.8s）
 * 一句长回复：间隔 1.5 ~ 3.2s ×7，有效 RTF 0.82
 * ```
 *
 * 三句回复实测**整轮缓冲区见底 296~1211ms**（连跑 5 次，每次都见底），
 * 与用户"中途还有些小卡顿"对得上。
 *
 * ## 为什么默认是 300ms（**用户真机试听后定的，不是算出来的**）
 *
 * 离线档位对比（同一条时间线重算 0/300/600/1000/1500ms）只看水位时，
 * 300ms 只是"减轻"（5 次里 4 次仍见底）、600ms 更干净。但**真机上用户试听后
 * 明确反馈 300ms 比 600ms 好**，那一通的诊断面板是：
 *
 * ```
 * 缓冲·见底 句内 0 / 跨句 0 · 累计 0ms       ← 一次都没见底
 * TTS·PCM 收到 281 帧 · 2214KB · 约 47233ms   ← 47 秒音频全部交给声卡
 * 首段到达 1721ms（合成段 ≈0.65s）
 * ```
 *
 * 原因是离线模型漏了一件真机上成立的事：**开头那一块很大**（≈1.5s 音频一次到达），
 * 所以 300ms 之后开播时水位本来就有一秒多，首个阵间隔被它盖住了；
 * 而 600ms 只是让首声再白等 300ms。**体感结论优先于离线水位。**
 *
 * ## 它治什么、不治什么
 *
 * * 治：开场那几下（以及紧随其后的第一个阵间隔）；
 * * 不治：中段比水位更长的阵间隔（1.5~3.2s）—— 那要动推理或播放侧策略，
 *   不在本常量职责内。**不要**用"预缓冲能吸收 X ms"这类经验数字评估档位：
 *   它动的是播放起点，能否消掉某次断档必须拿同一条时间线重算。
 *
 * ⚠️ 这个值只影响"第一块什么时候开始播"，不碰 CROSSFADE、不碰 PCM、不碰切分规则、
 * 不碰任何音质参数。要现场试档：`?prebuffer=0`（关）/ `?prebuffer=600`（更强）。
 */
const DEFAULT_PREBUFFER_MS = 300;

/** 预缓冲上限：再大就只是白等（实测最险一段 2.4s，但那是极端值） */
const MAX_PREBUFFER_MS = 2000;

export class TtsPlayer {
  private ctx: AudioContext | null = null;
  private readonly cb: TtsPlayerCallbacks;
  /**
   * 开场预缓冲（毫秒）。见 {@link DEFAULT_PREBUFFER_MS} 的实测依据。
   *
   * 只在"**队列从空到有**"的那一刻生效（也就是每一轮回复的第一块），
   * 之后的块照旧紧接排程 —— 所以它不会把整轮音频整体推后。
   */
  private readonly prebufferMs: number;

  /** 下一块的排程起点（`ctx.currentTime` 域）。0 表示队列空。 */
  private nextStartTime = 0;
  /** 按句聚合的状态；插入序 = 入队序 */
  private readonly open = new Map<number, OpenSegment>();
  /** 所有已排程、还没播完的块 */
  private readonly entries = new Set<ChunkEntry>();
  /** 最近一次入队的块；跨句交叉淡化要拿它的包络做淡出 */
  private lastEntry: ChunkEntry | null = null;
  private spokenLog: SpokenEntry[] = [];
  private disposed = false;

  // ── 欠载（空洞）统计：见 TtsUnderrunStats ──────────────────────────────
  private holes = 0;
  private totalHoleMs = 0;
  private maxHoleMs = 0;
  private lastHoleMs: number | null = null;
  private lastHoleSeq: number | null = null;
  /** 累计排程的音频总时长（毫秒），用于算空洞占比 */
  private scheduledAudioMs = 0;
  /** 累计排程的**块数**（"到底有没有东西交给声卡"的数量证据，见 outputHealth） */
  private scheduledChunks = 0;
  /** AudioContext 被挂起后**自动恢复**的尝试次数（>0 说明它真的被挂起过，见 resume） */
  private resumeAttempts = 0;
  /** 最近一次入库的 seq（判断"这一块与上一块是否同一句" ⇒ 句内 vs 跨句断档） */
  private lastChunkSeq: number | null = null;
  /** 播放缓冲区日志（见 `ttsBufferLog.ts`）：供音断档的现场记录，只记不决策 */
  private readonly bufferLog = new TtsBufferLog();
  /** 上一次开场预缓冲实际生效的毫秒数（面板读数用） */
  private prebufferAppliedMs = 0;
  /**
   * 预缓冲的**开场判定状态机**（见 `prebufferGate.ts`）。
   *
   * ⚠️ 它是"每轮只生效一次"的唯一真源。不要退回"看 `nextStartTime` 是否为零"
   * 那种写法 —— 那会在每次队列抽干后重新触发一次预缓冲（2026-10-11 用户指出的缺陷）。
   */
  private readonly prebufferGate = new PrebufferGate(true);
  /** 最近一次预缓冲实际生效的开关次数（诊断用：一轮只该 +1，多出来就是缺陷复发） */
  private prebufferActivations = 0;

  constructor(cb: TtsPlayerCallbacks = {}, prebufferMs?: number) {
    this.cb = cb;
    // ⚠️ 用 `??` 而不是默认参数：调用方（`TtsChannel`）会把 `undefined` **显式传进来**
    // （它自己不设默认值、由这里兜底），而默认参数对显式 `undefined` **不会**生效 ——
    // 那样预缓冲会静默变成 0，"改了没生效"。
    const wanted = prebufferMs ?? DEFAULT_PREBUFFER_MS;
    // 夹紧：负数当 0，过大只会白等（上限见 MAX_PREBUFFER_MS）
    this.prebufferMs = Math.max(0, Math.min(MAX_PREBUFFER_MS, Math.round(wanted)));
  }

  /** 当前生效的开场预缓冲（毫秒）—— 面板/日志如实上报，避免"改了没生效" */
  get prebuffer(): number {
    return this.prebufferMs;
  }

  /**
   * **新一轮开始** —— 让预缓冲重新有资格生效。
   *
   * ⚠️ 只能由"真正开一轮"的地方调用（`useVoiceCall.runTurn`）；
   * **绝不能**挂在"队列播空 / 所有设备静音"这类事件上 —— 那正是原缺陷的成因：
   * 阵发式供音的间隙里队列也会空，一挂上去就会在断档后再等一次预缓冲。
   */
  armPrebufferTurn(): void {
    this.prebufferGate.armTurn();
  }

  /** 本轮是否还欠一次预缓冲（诊断用） */
  get prebufferPending(): boolean {
    return this.prebufferGate.pending;
  }

  /** 预缓冲累计生效次数（诊断用：一轮只该 +1） */
  get prebufferActivationsCount(): number {
    return this.prebufferActivations;
  }

  // ── 状态 ────────────────────────────────────────────────────────────────
  /** 是否有音频正在播（已排程但还没结束） */
  get isPlaying(): boolean {
    return this.entries.size > 0;
  }

  /** 队列里还有几句（含正在合成、只到了一部分块的） */
  get pendingCount(): number {
    return this.open.size;
  }

  /** AudioContext 的实际采样率（设备值，不是模型值）；未创建时为 0 */
  get deviceSampleRate(): number {
    return this.ctx?.sampleRate ?? 0;
  }

  /**
   * 输出侧一行的健康读数（`?debug=1` 面板）——**"音频到了但听不见"只有这里能看出来**。
   *
   * 2026-10-11 用户："我两次说我感觉好一些了都没有声音出来。" 排查时必须能分开三种情况：
   *   · `ctxState = 'suspended'` → 浏览器自动播放策略把 AudioContext 挂起了：
   *     音频**排了程也不会响**（`enqueueChunk` 里的 `resume()` 被拒绝时就是这个），
   *     与合成链路完全无关，时间读数却一片正常；
   *   · `ctxState = 'running'` 且 `scheduledChunks > 0` → 我们确实把音频交给了声卡，
   *     此时"没声音"在页面之外（系统音量/输出设备/静音键）；
   *   · `scheduledChunks = 0` → 一个块都没排上（上游没给音频，见 TtsPayloadLedger）。
   *
   * `scheduledAudioMs` 是**累计排程的音频时长**：它与 `pcmBytes` 同源互相印证
   * （`pcmBytes / 48 ≈ scheduledAudioMs` @24k/16bit），两者不一致说明丢在排程前。
   */
  get outputHealth(): {
    ctxState: string;
    scheduledChunks: number;
    scheduledAudioMs: number;
    resumeAttempts: number;
    prebufferMs: number;
    prebufferAppliedMs: number;
    prebufferPending: boolean;
    prebufferActivations: number;
  } {
    return {
      ctxState: this.ctx ? this.ctx.state : 'none',
      scheduledChunks: this.scheduledChunks,
      scheduledAudioMs: Math.round(this.scheduledAudioMs),
      // >0 = 这通电话里 AudioContext 真的被挂起过（每一次都会自动尝试恢复）
      resumeAttempts: this.resumeAttempts,
      // 当前生效 / 最近一次实际生效的开场预缓冲（"改了没生效"最怕看不出来）
      prebufferMs: this.prebufferMs,
      prebufferAppliedMs: this.prebufferAppliedMs,
      // 本轮是否还欠一次（false 且 activations 已 +1 = 正常"只用一次"）
      prebufferPending: this.prebufferGate.pending,
      prebufferActivations: this.prebufferActivations,
    };
  }

  /**
   * 播放缓冲区日志的汇总（`?debug=1` 面板）。
   *
   * `withinSentenceStarves` 与 `gapStarves` 的**比值**就是用户要的那个结论：
   * 卡在句子中间（供音跟不上）还是卡在两句之间（片段调度）。
   * 明细（逐块水位 + 见底现场）在 `bufferLogDump()`，见 `ttsBufferLog.ts`。
   */
  get bufferStats(): {
    chunks: number;
    withinSentenceStarves: number;
    gapStarves: number;
    starveTotalMs: number;
    bufferMs: number;
  } {
    return this.bufferLog.summary();
  }

  /** 缓冲区日志明细（逐块水位 + 每次见底的现场）—— 排查卡顿时把它打出来 */
  bufferLogDump(recentChunks = 12): string {
    return this.bufferLog.dump(recentChunks);
  }

  /**
   * AudioContext 的播放时钟（毫秒）；未创建时为 0。
   *
   * 排程全部在这个时钟域里做，所以它是排查"句间有缝 / 排程错位"的第一手依据：
   * 把这个值与 `performance.now()` 放在一起看，就能判断音频时钟是否与墙钟同步
   * （无头浏览器里音频渲染线程可能被节流，二者并不同步）。
   */
  get currentTimeMs(): number {
    return (this.ctx?.currentTime ?? 0) * 1000;
  }

  /**
   * 播放空洞（欠载）的实测统计；**整通电话累计**，不是单轮。
   *
   * 单轮的量由 `useVoiceCall` 取两次快照相减得到（`TurnMetrics.audioHoleMs`）。
   * 定义与判据见 {@link TtsUnderrunStats}。
   */
  get underrun(): TtsUnderrunStats {
    return {
      holes: this.holes,
      totalHoleMs: Math.round(this.totalHoleMs * 10) / 10,
      maxHoleMs: Math.round(this.maxHoleMs * 10) / 10,
      lastHoleMs: this.lastHoleMs === null ? null : Math.round(this.lastHoleMs * 10) / 10,
      lastHoleSeq: this.lastHoleSeq,
      holeRatio: this.scheduledAudioMs > 0
        ? Math.round((this.totalHoleMs / this.scheduledAudioMs) * 1000) / 1000
        : 0,
    };
  }

  /** 清空欠载统计（新通话开始时用；不要在日常轮次之间调，那会丢掉累计证据） */
  resetUnderrunStats(): void {
    this.holes = 0;
    this.totalHoleMs = 0;
    this.maxHoleMs = 0;
    this.lastHoleMs = null;
    this.lastHoleSeq = null;
    this.scheduledAudioMs = 0;
  }

  // ── 生命周期 ────────────────────────────────────────────────────────────
  /**
   * 确保 AudioContext 存在并处于运行态。
   *
   * 浏览器要求音频必须由**用户手势**触发。通话页是由点击进入的，所以正常路径
   * 不会遇到挂起；但 `enqueue` 里仍会兜底调用一次 —— 若被浏览器拒绝，
   * 表现为"没声音"，而不是抛错中断通话。
   */
  async resume(): Promise<void> {
    if (this.disposed) return;
    if (!this.ctx) {
      const Ctor = window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!Ctor) return;
      this.ctx = new Ctor();
      // ── 挂起自愈（2026-10-11）────────────────────────────────────────────
      // 浏览器会在这些时刻把 AudioContext 置成 `suspended`：系统音频设备切换、
      // 来电/其它应用抢占、标签页被后台节流后回来、以及自动播放策略在**非用户手势**
      // 里给的那一次 resume 被拒。此时音频**排了程也不会响**，而通话里所有时间读数
      // （首声/首块/RTF）看起来完全正常 —— 表现就是用户说的"没有声音出来"，
      // 且征兆只有 `ctx.state`（原来的面板上根本看不到）。
      //
      // 这里挂一个监听：一旦发现被挂起就立刻尝试恢复，并把尝试次数记进读数
      // （`resumeAttempts`），于是"是不是它"在面板上有一手证据。
      this.ctx.addEventListener?.('statechange', () => {
        if (this.disposed) return;
        const c = this.ctx;
        if (c && c.state === 'suspended') {
          this.resumeAttempts += 1;
          void c.resume().catch(() => {
            /* 被自动播放策略拒绝：保持静默，面板上有读数可查 */
          });
        }
      });
    }
    if (this.ctx.state === 'suspended') {
      try {
        await this.ctx.resume();
      } catch {
        /* 被自动播放策略拒绝：静默，表现为无声，不打断通话 */
      }
    }
  }

  // ── 播放 ────────────────────────────────────────────────────────────────
  /**
   * 追加**一个音频块**并立即排程（逐块播放的主入口）。
   *
   * 服务端会把一句的 PCM 分成十几到几十块流式回推；每块一到就调这里，立刻排进
   * 时间轴 —— **不要攒到 `end` 再一次性入队**，那会把服务端串行合成的等待全部
   * 暴露成句中空洞（实测 686 + 1225 ms，见 `_probe_playback_timeline.py`）。
   *
   * @param seq 所属句号，用于把 `onStart`/`onEnd` 与请求对上
   * @param pcm int16 小端 PCM（服务端二进制帧的原样内容）
   * @param sampleRate PCM 的采样率
   * @param text 这一句**实际被念出来的文本**（服务端 `spoken_text`）
   * @returns 是否成功排程。false 表示这块音频被丢弃（空/已销毁/无 AudioContext）
   */
  enqueueChunk(seq: number, pcm: Int16Array, sampleRate: number, text: string): boolean {
    if (this.disposed || pcm.length === 0 || sampleRate <= 0) return false;

    void this.resume();
    const ctx = this.ctx;
    if (!ctx) return false;

    let seg = this.open.get(seq);
    if (!seg) {
      seg = {
        seq,
        text,
        live: 0,
        chunks: 0,
        ended: false,
        started: false,
        finished: false,
        startTimer: 0,
        startsAt: 0,
        endsAt: 0,
      };
      this.open.set(seq, seg);
    } else if (text && !seg.started) {
      // `onStart` 之前允许更新文本（服务端可能后到更准的 spoken_text）
      seg.text = text;
    }

    // int16 → float32 [-1, 1]
    const buffer = ctx.createBuffer(1, pcm.length, sampleRate);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) {
      channel[i] = pcm[i] / 32768;
    }

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    // ⚠️ 必须 source → gain → destination（而不是 source 直连）：
    // 跨句交叉淡化要靠各自的增益节点做互补斜坡。多一个 GainNode 不影响
    // AEC 的参考信号 —— 参考取的是默认输出设备，见文件头。
    const gain = ctx.createGain();
    source.connect(gain);
    gain.connect(ctx.destination);

    // 交叉淡化**只发生在跨句接缝**上：同一句内部的块是同一段连续波形的前后片段，
    // 必须严丝合缝地接上（重叠会吃掉采样）。跨句才是两次独立合成、相位不连续。
    const prev = this.lastEntry;
    const crossSeg = prev !== null && !prev.finished && prev.seq !== seq
      && prev.endsAtCtx > ctx.currentTime;
    const overlap = crossSeg && prev
      ? Math.max(0, Math.min(CROSSFADE_MS / 1000, buffer.duration / 2, prev.durationSec / 2))
      : 0;

    // ── 开场预缓冲（判据见 `prebufferGate.ts`，别在这里重写条件）────────
    // ⚠️ 曾经用 `nextStartTime <= 0` 判"开场"，那是**错的**：`nextStartTime` 在
    // 队列每次被抽干时都会归零，于是阵发间隙里又来一次 300ms —— 把要治的卡顿放大。
    // 现在由 `PrebufferGate` 严格控制为"**每轮只生效一次**"：
    //   · 是否欠一次预缓冲：只由 `armTurn()`（新一轮开始）决定；
    //   · 是否清账：看"队列是否真的起飞了"（queuedAudioMs > 0），不看单个块。
    const queuedAudioMs = Math.max(0, (this.nextStartTime - ctx.currentTime) * 1000);
    const wantPrebuffer =
      this.prebufferMs > 0 && this.prebufferGate.decide(queuedAudioMs);
    const base = wantPrebuffer
      ? ctx.currentTime + this.prebufferMs / 1000
      : this.nextStartTime > 0
        ? this.nextStartTime
        : ctx.currentTime;
    const startAt = Math.max(ctx.currentTime, base - overlap);
    this.nextStartTime = startAt + buffer.duration;
    // 把预缓冲记进读数：只说"多等了多少"，不改变判定口径（见 TtsUnderrunStats）
    if (wantPrebuffer) {
      this.prebufferAppliedMs = this.prebufferMs;
      this.prebufferActivations += 1;
    }

    // ── 欠载判定（见 TtsUnderrunStats）──────────────────────────────────
    // `base > currentTime` 说明"上一块还没播完、这一块本该接在它后面"；
    // 而 `startAt` 被 Math.max 抬到了"现在" ⇒ 中间那一段没人补：这就是空洞。
    // 队列本来就空（base === currentTime，上一句已自然播完）不算 ——
    // 那时用户听到的是刻意的留白。
    if (base > ctx.currentTime && startAt > base) {
      const holeMs = (startAt - base) * 1000;
      this.holes += 1;
      this.totalHoleMs += holeMs;
      if (holeMs > this.maxHoleMs) this.maxHoleMs = holeMs;
      this.lastHoleMs = holeMs;
      this.lastHoleSeq = seq;
    }
    this.scheduledAudioMs += buffer.duration * 1000;
    this.scheduledChunks += 1;

    // ── 播放缓冲区日志（见 ttsBufferLog.ts）──────────────────────────────
    // 记"这块入库后缓冲区还剩多少毫秒"，并把**句内 / 跨句**分开 ——
    // 这正是用户那两个候选病因的分界线。只记录，不改排程。
    {
      const starved = base > ctx.currentTime && startAt > base;
      const starveMs = starved ? (startAt - base) * 1000 : 0;
      const prevSeq = this.lastChunkSeq;
      this.bufferLog.onChunk({
        atMs: performance.now(),
        seq,
        chunkMs: buffer.duration * 1000,
        // 入库后的水位：从 startAt 到"已排程末尾"还剩多少音频
        bufferMs: Math.max(0, (this.nextStartTime - ctx.currentTime) * 1000),
        sameSeq: prevSeq === seq,
        starved,
        starveMs,
      });
      this.lastChunkSeq = seq;
      // 真的见底了：把现场（最近若干块 + 见底事件）打出来。
      // ⚠️ 只在见底时打，正常播放一行都不打 —— 否则一次回复几十块会淹掉控制台。
      if (starved) {
        console.warn(
          `[播放缓冲] seq=${seq} 见底 ${Math.round(starveMs)}ms`
          + `（${prevSeq === seq ? '句内供给不足' : '跨句接缝'}）\n`
          + this.bufferLog.dump(),
        );
      }
    }

    if (overlap > 0 && prev) {
      gain.gain.setValueAtTime(0, startAt);
      gain.gain.linearRampToValueAtTime(1, startAt + overlap);
      // 上一块此时是满幅的（它自己的淡入早已结束），直接补一条淡出斜坡
      prev.gain.gain.setValueAtTime(1, startAt);
      prev.gain.gain.linearRampToValueAtTime(0, startAt + overlap);
    } else {
      gain.gain.setValueAtTime(1, startAt);
    }

    const entry: ChunkEntry = {
      seq,
      source,
      gain,
      durationSec: buffer.duration,
      endsAtCtx: startAt + buffer.duration,
      finished: false,
    };
    this.entries.add(entry);
    this.lastEntry = entry;
    seg.live += 1;
    seg.chunks += 1;

    const nowWall = performance.now();
    const startsAtWall = nowWall + (startAt - ctx.currentTime) * 1000;
    if (seg.chunks === 1) {
      seg.startsAt = startsAtWall;
      // `onStart` 必须按**真实出声**时刻触发（见回调文档），而且**一句只触发一次**。
      // 首块 startAt ≈ currentTime ⇒ 延迟 ≈ 0，等价于立即触发。
      seg.startTimer = window.setTimeout(() => {
        seg.startTimer = 0;
        if (seg.finished || seg.started || !this.open.has(seq)) return;
        seg.started = true;
        this.cb.onStart?.(seq, seg.text, {
          ctxStartMs: startAt * 1000, // 精确：排程时算出来的
          startsAtWallMs: startsAtWall, // 近似：见 TtsStartTiming 的文档
        });
      }, Math.max(0, startsAtWall - performance.now()));
    }
    seg.endsAt = startsAtWall + buffer.duration * 1000;

    source.onended = () => {
      // 被 stopAll() 掐断时这里也会触发（`onended` 在 stop 后照常派发）。
      // 若这块已经被清理过，就什么都不做 —— 幂等。
      if (!this.entries.delete(entry)) return;
      entry.finished = true;
      if (this.lastEntry === entry) this.lastEntry = null;
      seg.live -= 1;
      this.settleSegment(seg);
    };

    source.start(startAt);
    return true;
  }

  /**
   * 声明某一句**不会再有新块了**（收到服务端 `end`）。
   *
   * `onEnd` 会在这一句**最后一个块播完**之后才触发，而不是现在 ——
   * 否则调用方会以为"说完了"，而其实还有几秒音频排在队列里。
   *
   * @param seq 句号
   * @returns 这一句累计收到的音频块数；**0 表示没合成出音频**
   */
  endSegment(seq: number): number {
    const seg = this.open.get(seq);
    if (!seg) return 0;
    seg.ended = true;
    this.settleSegment(seg);
    return seg.chunks;
  }

  /** 一句收尾：`end` 已到 **且** 所有块都播完 → 触发 `onEnd`（幂等） */
  private settleSegment(seg: OpenSegment): void {
    if (seg.finished) return;
    if (!seg.ended || seg.live > 0) return;
    seg.finished = true;
    if (seg.startTimer) {
      window.clearTimeout(seg.startTimer);
      seg.startTimer = 0;
    }
    this.open.delete(seg.seq);
    this.logSpoken(seg.seq, seg.text);
    this.cb.onEnd?.(seg.seq, seg.text);
    if (this.entries.size === 0 && this.open.size === 0) {
      this.nextStartTime = 0;
      this.cb.onDrained?.();
    }
  }

  /**
   * 追加**整句**音频并立即排程（兼容入口：等价于一块 + 立刻 `endSegment`）。
   *
   * @param seq 序号
   * @param pcm int16 小端 PCM
   * @param sampleRate PCM 的采样率
   * @param text **实际被念出来的文本**
   * @returns 是否成功排程
   */
  enqueue(seq: number, pcm: Int16Array, sampleRate: number, text: string): boolean {
    if (!this.enqueueChunk(seq, pcm, sampleRate, text)) return false;
    this.endSegment(seq);
    return true;
  }

  /**
   * 当前还挂在播放器里（`end` 已到但还没播完，或正在收块）的句号。
   *
   * 必须在 `stopAll()` **之前**取：`stopAll()` 会清空 `open`，之后那些
   * "已经 `start` 过、`end` 还没到"的句子就查不到了 —— 而它们的 `end`
   * 稍后仍会到达，`endSegment()` 会返回 0，被误判成"没合成出音频"。
   * 调用方（`TtsChannel.stopFrom`）据此把"我们掐断的"登记给会话层。
   */
  openSeqs(): number[] {
    return [...this.open.keys()];
  }

  /**
   * **还排在队列里、没播到**的音频总时长（毫秒）。0 = 当下没有待播音频。
   *
   * 为什么需要它（2026-10-11 真机：一轮的首声 **6159 ms**，而合成本身只有
   * 1.5 s 左右）：用户听到的"停顿"有两个来源，对策完全不同 ——
   *   · **合成慢**：服务端自己报的 `first_chunk_ms` 就大；
   *   · **前面还排着音频**：本函数大（上一轮的句子还没播完就被排到了前面）。
   * 只把两者分开量，才能知道该修哪一头。
   */
  pendingAudioMs(): number {
    if (!this.ctx || this.nextStartTime <= 0) return 0;
    return Math.max(0, Math.round((this.nextStartTime - this.ctx.currentTime) * 1000));
  }

  /**
   * 立刻掐断全部播放（**打断路径**）。
   *
   * 先 `stop()` 再 `disconnect()`，同步完成、不等待、不返回 Promise ——
   * 打断的体感全在这里，任何 await 都会让它变慢。
   * 实测 `AudioBufferSourceNode.stop()` 的端到端延迟 < 50 ms
   * （按渲染量子推算，`voice_call_plan.md` 未验证项 12）。
   *
   * 被掐断的句子**也**会进 `recentSpoken()`：它确实已经说出去了一部分，
   * 回声过滤必须知道这件事，否则会把"AI 刚说的半句"当成用户插话。
   */
  stopAll(): void {
    const chunks = [...this.entries];
    const segs = [...this.open.values()];
    this.entries.clear();
    this.open.clear();
    this.nextStartTime = 0;
    this.lastEntry = null;

    for (const e of chunks) {
      try {
        e.finished = true;
        e.source.onended = null; // 避免 stop() 触发 onended 再走一遍收尾逻辑
        e.source.stop();
        e.source.disconnect();
        e.gain.disconnect();
      } catch {
        /* 已结束的 source 再 stop 会抛，忽略 */
      }
    }
    // 被掐断的句子**也**要按句登记进 `recentSpoken()`：它确实已经说出去了一部分，
    // 回声过滤必须知道这件事，否则会把"AI 刚说的半句"当成用户插话。
    for (const s of segs) {
      if (s.startTimer) {
        window.clearTimeout(s.startTimer);
        s.startTimer = 0;
      }
      s.finished = true;
      this.logSpoken(s.seq, s.text);
    }

    if (chunks.length > 0 || segs.length > 0) this.cb.onDrained?.();
  }

  // ── 自回声过滤的支持 ────────────────────────────────────────────────────
  /**
   * 最近一段时间内**已经播出去**的文本（按时间从旧到新）。
   *
   * 这是 `EchoFilter`（S5）唯一的输入：把 ASR 的中间结果与这些文本做模糊匹配，
   * 命中即判定为"AI 自己的声音被麦克风收回来"，不当成用户插话。
   *
   * 关键点：入参的 `text` 必须用服务端回传的 `spoken_text`（术语替换**之后**的），
   * 否则 `CBT → 认知行为疗法` 这类替换会让比对失配，
   * 进而把 AI 自己的声音误判成用户在说话。
   *
   * @param withinMs 回溯窗口；默认 8 秒（覆盖"AI 说完"到"插话被定稿"的间隔）
   * @returns 文本列表；当前**正在播**的那句也在里面
   */
  recentSpoken(withinMs: number = DEFAULT_SPOKEN_WINDOW_MS): string[] {
    const now = performance.now();
    // 已经进了播放队列的（含正在播的与排在后面的，**含只到齐一半块的**）：
    // 它们都会经由扬声器进入麦克风，所以都要参与过滤 ——
    // 宁可多滤一点，也不要漏滤后把 AI 的声音当成用户插话。
    // 按**句**去重（一句有几十块，不去重会返回几十条同样的文本）。
    const pending = [...this.open.values()]
      .filter((s) => now <= s.endsAt)
      .sort((a, b) => a.startsAt - b.startsAt)
      .map((s) => s.text);
    return [...this.spokenLog.filter((e) => now - e.startedAt <= withinMs).map((e) => e.text), ...pending];
  }

  /**
   * 释放：掐断播放并关闭 AudioContext。
   *
   * ⚠️ 关闭后本实例**不可复用**。通话页卸载时调用。
   */
  dispose(): void {
    if (this.disposed) return;
    this.stopAll();
    this.disposed = true;
    this.spokenLog = [];
    const ctx = this.ctx;
    this.ctx = null;
    if (ctx && ctx.state !== 'closed') {
      // ⚠️ 必须 `.catch()`：`AudioContext.close()` 返回 Promise，对**已经关过**的
      // context 会**拒绝**（InvalidStateError: Cannot close a closed AudioContext）。
      // 只写 `void ctx.close()` 的话 try/catch 抓不到它 —— 它会变成一个
      // 未处理的 rejection，而"离开通话页后未处理 rejection 为 0"是 S3/S4 的闸门项。
      try {
        void ctx.close().catch(() => undefined);
      } catch {
        /* 同步抛（正在关的过程中）同样不该冒到上层 */
      }
    }
  }

  /** 是否已销毁（调用方据此避免往已释放的播放器里塞音频） */
  get isDisposed(): boolean {
    return this.disposed;
  }

  // ── 内部 ────────────────────────────────────────────────────────────────
  private logSpoken(seq: number, text: string): void {
    if (!text) return;
    this.spokenLog.push({ seq, text, startedAt: performance.now() });
    if (this.spokenLog.length > MAX_SPOKEN_LOG) {
      this.spokenLog.splice(0, this.spokenLog.length - MAX_SPOKEN_LOG);
    }
  }
}

export { DEFAULT_SPOKEN_WINDOW_MS };
