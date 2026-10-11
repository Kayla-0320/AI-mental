/**
 * 本地 TTS 传输（WebSocket）+ 与播放器的组合
 *
 * 分工：
 *   `TtsStreamSession` —— 只说协议：连 `/algorithm/api/v1/tts/ws`、发待合成文本、
 *                          收二进制 PCM 帧、把控制帧翻译成回调。
 *   `TtsPlayer`        —— 只说播放（见 `ttsPlayer.ts`）。
 *   `TtsChannel`       —— 把两者接起来，**通话页直接用这个**。
 *
 * ## 协议（与 `algorithm/api/tts.py` 同步）
 *
 * 上行（文本帧 JSON）::
 *
 *     {"type":"say","seq":1,"text":"听起来这段时间你挺不容易的。"}
 *     {"type":"cancel","seq":1}
 *     {"type":"close"}
 *
 * 下行::
 *
 *     {"type":"status","available":true,"engine":"sherpa-onnx-vits",
 *      "sample_rate":16000,"num_speakers":187,"num_threads":8,"error":""}
 *     {"type":"start","seq":1,"sample_rate":16000}
 *     二进制帧 —— int16 小端单声道 PCM
 *     {"type":"end","seq":1,"spoken_text":"...","audio_ms":4101.1,
 *      "first_chunk_ms":443.4,"synth_ms":443.4,"rtf":0.108,"cancelled":false}
 *     {"type":"cancelled","seq":1}
 *     {"type":"error","seq":1,"message":"……"}
 *
 * ⚠️ **实测：服务端每一句只回一个二进制帧**（VITS 不流式，见
 * `docs/tts_benchmark.md` §1 发现一）。这里的接收逻辑仍写成"累积到 `end` 为止"，
 * 这样将来换成真流式 TTS 时不用改客户端 —— 但**不要**依赖"帧数 > 1 才能工作"。
 *
 * ⚠️ **`spoken_text` 必须原样带给播放器**：它是术语替换**之后**真正被念出来的
 * 文本。回声过滤拿它比对；若用原始请求文本，`CBT → 认知行为疗法` 这类替换
 * 会让比对失配，把 AI 自己的声音误判成用户插话。
 */

import { TtsPlayer, type TtsPlayerCallbacks, type TtsStartTiming, type TtsUnderrunStats } from './ttsPlayer';
import { normalizeForCompare } from './ttsSplit';

/** 服务端握手回来的状态 */
export interface TtsStreamStatus {
  available: boolean;
  engine?: string;
  sampleRate: number;
  numSpeakers: number;
  numThreads: number;
  /** 不可用时的原因（供 UI 如实展示） */
  reason?: string;
}

/** 一句合成结束时的元数据 */
export interface TtsSegmentMeta {
  seq: number;
  /** 真正被念出来的文本（已过术语替换） */
  spokenText: string;
  sampleRate: number;
  audioMs: number;
  /** 服务端合成耗时；流式引擎下它是**整段**合成耗时（含全部块） */
  synthMs: number;
  /**
   * 服务端自报的「请求 → 第一块音频」耗时（CosyVoice3 流式路径才有意义）。
   *
   * 与客户端 {@link TtsStreamSession.takeFirstPcmMs} 相减 = 下行/客户端那一段。
   */
  firstChunkMs: number;
  /** 实时率；判据 < 0.4 */
  rtf: number;
}

/** TTS 通道的**对账/隔离**诊断（`?debug=1` 与真机排查用） */
export interface TtsStreamDiagnostics {
  /** 取消后仍在路上、被丢弃的残留帧数（>0 说明取消确实截住了旧音频） */
  drainedFrames: number;
  /** 被丢弃的残留字节数 */
  drainedBytes: number;
  /** 收到过多少个"水位线之下"的陈旧 `start`（旧轮次的回声） */
  staleStarts: number;
  /**
   * 疑似**串单**次数：某一句收到的 `spoken_text` 与请求文本不符，
   * **且**与另一句待播文本相符（能对上别人 = 音频串到了别的片段）。
   * 术语规范化（`CBT → 认知行为疗法`）导致的正常差异不计入这里。
   */
  crossTalk: number;
  /** 术语规范化等"预期差异"的次数（同上条件但不指向别的片段） */
  normalizedText: number;
  /** 取消水位线：`seq <= cancelledThrough` 的音频一律不再接收/播放 */
  cancelledThrough: number;
  /**
   * 被**我们自己掐断**、因此没有音频可播的句数（打断 / 改稿 / 危机）。
   *
   * 与"真的没合成出音频"必须分开算：前者是我们主动丢的，不该弹错误；
   * 后者要如实上报。判据见 {@link TtsStreamSession.stoppedSeqs}。
   */
  stoppedWithoutAudio: number;
}

/**
 * 「这一通电话到底发出去了几个字、收回来了多少个音频字节」——**载荷级**读数。
 *
 * ## 为什么必须有它（2026-10-11 用户："我两次说我感觉好一些了都没有声音出来"）
 *
 * 现有读数全是**时间**（首声 / 首块 / RTF），只能回答"慢不慢"；而"完全没有声音"
 * 是一个**数量**问题：是根本没发 `say`，还是发了但服务端一个 PCM 字节都没回，
 * 还是收到了却被我们自己的取消丢掉了。三者的处置完全不同，而光看时间读数是
 * 一片 `—`，分不出来（真机截图就是这种情况）。
 *
 * 判据（面板上并排看）：
 *   · `say` > 0 而 `收帧` = 0        → 服务端没回音频（合成/OOV/取消）
 *   · `收帧` > 0 而 `丢帧` ≈ `收帧`  → 被取消水位线丢光（我们自己掐的）
 *   · `收帧` > 0 且 `丢帧` = 0        → 音频到了客户端，问题在播放层（见 `underrun`）
 *
 * ⚠️ `cancelled` 帧也计进 `endFrames`（它就是服务端对"我们取消了"的确认），
 * 所以 `endFrames` 与 `sayFrames` 应当逐句对齐。
 */
export interface TtsPayloadLedger {
  /** 发出去的 `say` 句数 */
  sayFrames: number;
  /** 收到的二进制 PCM 帧数与总字节数（**服务端真的产出了音频**的证据） */
  pcmFrames: number;
  pcmBytes: number;
  /** 收到的 `start` / `end`（含 `cancelled`）句数 */
  startFrames: number;
  endFrames: number;
  /** 服务端自报"合成成功但一个字节都没出"的句数 */
  emptyAudioSegments: number;
  /** 因为取消水位线/残留排空而**被我们丢弃**的帧数与字节数 */
  droppedFrames: number;
  droppedBytes: number;
  /**
   * **文本对不上**的次数：服务端 `spoken_text` 与本次 `say` 请求的文本不一致
   * （术语规范化那种预期差异已排除）。
   *
   * 这是"AI 偶尔说出乱话"最直接的证据类型 —— 耳朵里那段话**送出去的时候就是那句**，
   * 而不是播放错位。见 {@link TtsStreamSession.auditSpokenText}。
   */
  textMismatch: number;
  /**
   * 最近几组"请求文本 → 实际念出的文本"（供面板展示）。
   *
   * ⚠️ 只在**疑似异常**时填（文本对不上 / 疑似串单），正常合成不占内存、不进面板。
   */
  textMismatchSamples: Array<{ seq: number; requested: string; spoken: string; reason: string }>;
}

/** `TtsChannel` 的回调 */
export interface TtsChannelCallbacks extends TtsPlayerCallbacks {
  /** 连接状态变化（含不可用原因），UI 据此如实展示 */
  onStatus?: (status: TtsStreamStatus) => void;
  /** 单句合成完成（已入播放队列） */
  onSegment?: (meta: TtsSegmentMeta) => void;
  /** 出错；`seq` 缺省表示连接级错误 */
  onError?: (message: string, seq?: number) => void;
  /**
   * 对账/隔离诊断发生变化（串单、丢弃残留帧等）。
   *
   * 单独一个回调而不是塞进 `onError`：这些多半**不是错误**（术语规范化、
   * 取消后正常排空），但必须能在面板上看到 —— 否则"文字与语音不匹配"
   * 只能靠人耳猜。
   */
  onDiagnostics?: (info: TtsStreamDiagnostics) => void;
}

/** 连接建立前的待发上限；超出丢最旧的，避免无限增长 */
const MAX_PENDING_SAYS = 10;
/** 通话比一次语音输入长得多，重连次数放宽（ASR 那边是 3） */
const MAX_RECONNECTS = 5;
/** 连接超时 */
const CONNECT_TIMEOUT_MS = 5000;

/** 依据当前页面推出 WS 地址（走 vite 代理的 /algorithm 前缀） */
function resolveWsUrl(): string {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.host}/algorithm/api/v1/tts/ws`;
}

/** int16 小端字节 → Int16Array（复制一份，避免共享已被回收的 buffer） */
function bytesToInt16(buf: ArrayBuffer): Int16Array {
  return new Int16Array(buf.slice(0));
}

/** 合并同一句的多个帧（当前恒为 1 帧，保留是为了兼容将来的流式实现） */
function concatInt16(parts: Int16Array[]): Int16Array {
  if (parts.length === 1) return parts[0];
  const total = parts.reduce((n, p) => n + p.length, 0);
  const out = new Int16Array(total);
  let offset = 0;
  for (const p of parts) {
    out.set(p, offset);
    offset += p.length;
  }
  return out;
}

/** 一句的接收缓冲 */
interface SegmentBuffer {
  /** 客户端**发出去**的原文（未过术语替换） */
  text: string;
  /**
   * 服务端回传的 `spoken_text`：**实际被念出来**的文本（已过术语替换）。
   *
   * ⚠️ `start` 事件就带着它（`api/tts.py` 在 `say` 时就已算好），所以
   * **第一块音频到达前**就有值 —— 这是"边收边播"能保持 `onStart(seq, spokenText)`
   * 语义不变的关键。若服务端没给（旧版本），退回原文。
   */
  spokenText: string;
  sampleRate: number;
  /**
   * 采样率未知时用来兜底的攒帧缓冲。
   *
   * 正常路径**不会用到**：`start` 事件总在二进制帧之前到达且带 `sample_rate`。
   * 保留是为了防御"协议被改坏时整句丢音"变成"整句静音"。
   */
  frames: Int16Array[];
  /** 客户端发出 `say` 的时刻，用于算"从请求到出声"的真实延迟 */
  sentAt: number;
}

/**
 * 只负责协议的 TTS 会话。
 *
 * 为什么直接用原始 WebSocket：与 `streamingAsr.ts` 同一个理由 ——
 * 需要"一条持续通道 + 服务端主动推"，HTTP 一问一答会把每次请求的
 * 建连/头部开销放大好几倍，而通话里每句话都要合成一次。
 */
export class TtsStreamSession {
  private ws: WebSocket | null = null;
  private pendingSays: Array<{ seq: number; text: string }> = [];
  private segments = new Map<number, SegmentBuffer>();
  private reconnects = 0;
  private closed = false;
  private closing = false;
  /** 最近一个收到 `start` 的 seq —— 二进制帧靠它归属（服务端串行合成，所以唯一） */
  private lastStartedSeq: number | null = null;
  /**
   * **取消水位线**：`seq <= cancelledThrough` 的音频一律不再接收、不再播放。
   *
   * 为什么需要它（2026-10-11，"文字与语音不匹配"的一条真因）：
   * 取消是**异步**的 —— `cancel` 控制帧发出去之后，服务端那边**已经在队列里/正在发**
   * 的 PCM 还会到达。二进制帧本身**不带 seq**，客户端只能按"最近一个 `start`"归属，
   * 于是一旦旧帧和新一轮的 `start` 撞在一起，旧音频就会被当成**新一轮的语音**播出来
   * —— 听起来就是"说的话和字幕对不上、还缺一块"。
   *
   * 水位线 + {@link awaitingFreshStart} 一起把这段窗口封死：
   * 取消之后，**必须先看到一个水位线之上的 `start`**，才重新接受二进制帧。
   */
  private cancelledThrough = 0;
  /** 取消后是否还在"排空残留"：true 时到达的二进制帧一律丢弃 */
  private awaitingFreshStart = false;
  private drainedFrames = 0;
  private drainedBytes = 0;
  private staleStarts = 0;
  private crossTalk = 0;
  private normalizedText = 0;
  private stoppedWithoutAudio = 0;
  /** 载荷级账（见 {@link TtsPayloadLedger}）：发了几句、回了多少字节、丢了多少 */
  private ledger: TtsPayloadLedger = {
    sayFrames: 0,
    pcmFrames: 0,
    pcmBytes: 0,
    startFrames: 0,
    endFrames: 0,
    emptyAudioSegments: 0,
    droppedFrames: 0,
    droppedBytes: 0,
    textMismatch: 0,
    textMismatchSamples: [],
  };
  /**
   * 被**我们自己掐断**（播放器 `stopAll()`）的句号。
   *
   * 为什么需要它（2026-10-11 用户截图："语音合成出错：这一句没有合成出音频"）：
   * 一句回复是**一次把所有句都 `say` 出去**的，服务端串行合成。收到 `revise`
   * （或打断）时 `TtsChannel.stopFrom()` 先 `player.stopAll()` —— 播放器的 `open`
   * 表被清空，于是那些**已经 `start` 过、`end` 还没到**的句子后来走 `end` 分支时
   * `endSegment()` 查不到记录、返回 0，被当成"合成成功但没音频"弹了一条错误。
   * 其实那几句的音频是**我们主动丢的**（用户听到的正是被掐断的效果）。
   * 记下这批句号：`end` 到了只计入 {@link stoppedWithoutAudio} 读数，不报错。
   */
  private stoppedSeqs = new Set<number>();
  /**
   * `connect()` 的兑现函数。**由服务端的 `status` 帧触发**，不是 `onopen`。
   *
   * 为什么不能只在 `onopen` 兑现：`onopen` 只说明 TCP/WS 握手成功，
   * 此时 `status` 还没到，调用方会读到 `available:false / sampleRate:0` 的空状态 ——
   * 于是"服务端到底能不能合成"这个问题在第一句话之前无从判断。
   * （这是浏览器实测抓到的：连接明明通了、5 句都合成出来了，`ch.status` 却全是 0。）
   */
  private settleConnect: ((ok: boolean) => void) | null = null;

  private status: TtsStreamStatus = {
    available: false,
    sampleRate: 0,
    numSpeakers: 0,
    numThreads: 0,
  };

  /**
   * 每个 seq 从 `say()` 到**最后一个 PCM 字节到手**的耗时（毫秒）。
   *
   * 为什么必须单独量它，而不是拿播放器的 `onStart` 凑：
   * `onStart` 是**排程时刻**（"该它出声了"），第 2 段起天然包含前面几段的音频时长。
   * 实测（`_probe_player_clock.js`）：3 段 2 秒音频一起入队时，第 3 段的 `onStart`
   * 落在入队后 3.9 s —— 那不是"慢"，那是它本来就要排队。
   * 所以「请求 → 音频到手」只能在这里量，它才是 S4 闸门要的那个数
   * （「用户停口 → 第一个 PCM **到达播放器** < 2.4 s」）。
   */
  private readonly pcmReadyMs = new Map<number, number>();

  /**
   * 每个 seq 的**第一块** PCM 到达时刻（毫秒，相对 `say()`）。
   *
   * 与 {@link pcmReadyMs}（最后一块到手）配成一对：
   *   · 第一块 = "听到声音要等多久"的下限（还需要播放器排程）；
   *   · `end` 消息自带服务端自报的 `first_chunk_ms` —— 两者一减，
   *     就能看出慢在**服务端合成**还是慢在下行/客户端。
   * （2026-10-11 真机：某轮首声 6159 ms，而离线实测合成本身只要 1.4~1.7 s，
   *   必须把这段差额拆开，见 docs/voice_call_audio_stutter.md §1.2。）
   */
  private readonly firstPcmMs = new Map<number, number>();

  constructor(
    private readonly cb: TtsChannelCallbacks,
    private readonly player: TtsPlayer,
  ) {}

  get isOpen(): boolean {
    return !!this.ws && this.ws.readyState === WebSocket.OPEN;
  }

  get currentStatus(): TtsStreamStatus {
    return { ...this.status };
  }

  // ── 连接 ────────────────────────────────────────────────────────────────
  connect(): Promise<boolean> {
    return new Promise((resolve) => {
      let url: string;
      try {
        url = resolveWsUrl();
      } catch (err) {
        this.fail(`无法解析 WebSocket 地址：${(err as Error)?.message || err}`);
        resolve(false);
        return;
      }

      let ws: WebSocket;
      try {
        ws = new WebSocket(url);
      } catch (err) {
        this.fail(`创建 WebSocket 失败：${(err as Error)?.message || err}`);
        resolve(false);
        return;
      }
      ws.binaryType = 'arraybuffer';
      this.ws = ws;

      let settled = false;
      const settle = (ok: boolean): void => {
        if (settled) return;
        settled = true;
        window.clearTimeout(openTimer);
        this.settleConnect = null;
        resolve(ok);
      };
      this.settleConnect = settle;

      const openTimer = window.setTimeout(() => {
        try {
          ws.close();
        } catch {
          /* 忽略 */
        }
        this.fail('连接语音合成服务超时（本机算法服务是否在 8001 运行？）');
        settle(false);
      }, CONNECT_TIMEOUT_MS);

      ws.onopen = () => {
        // 刻意**不**在这里兑现 —— 要等服务端的 status 帧（见 settleConnect 的注释）
        this.reconnects = 0;
        // 把建立连接期间攒下的待合成文本补发出去，避免丢开头
        const queued = this.pendingSays;
        this.pendingSays = [];
        for (const item of queued) this.sendSay(item.seq, item.text);
      };

      ws.onmessage = (ev) => this.onMessage(ev);

      ws.onerror = () => {
        // onerror 之后必然跟 onclose，统一在 onclose 里处理，避免重复上报
      };

      ws.onclose = () => {
        window.clearTimeout(openTimer);
        this.ws = null;
        if (this.closed || this.closing) {
          settle(false);
          return;
        }
        this.reconnects += 1;
        if (this.reconnects >= MAX_RECONNECTS) {
          this.fail(`语音合成连接中断，已重试 ${this.reconnects} 次后停止`);
          settle(false);
          return;
        }
        // 在途的句子取不回来了：如实上报，让通话逻辑决定要不要重说
        for (const [seq] of this.segments) {
          this.cb.onError?.('连接中断，这一句没能合成', seq);
        }
        this.segments.clear();
        settle(false);
        // 退避重连：0.5s / 1s / 1.5s …
        window.setTimeout(() => {
          if (!this.closed && !this.closing) void this.connect();
        }, 500 * this.reconnects);
      };
    });
  }

  // ── 收发 ────────────────────────────────────────────────────────────────
  private onMessage(ev: MessageEvent): void {
    if (typeof ev.data !== 'string') {
      // 二进制帧 = 某一句的 PCM。归到自己在等的 seq 上：
      // 服务端串行合成，所以"最近一个 start 过的 seq"就是它。
      const seq = this.lastStartedSeq;
      if (seq === null || !this.ws) return;
      // ── 取消后的残留排空（见 cancelledThrough / awaitingFreshStart）──
      // 这两道闸合起来保证：取消那一轮还在路上的 PCM **绝不会**被当成下一轮的语音播出去。
      if (this.awaitingFreshStart || seq <= this.cancelledThrough) {
        this.drainedFrames += 1;
        this.drainedBytes += (ev.data as ArrayBuffer).byteLength ?? 0;
        // 载荷级账：这一帧**收到了**却被我们丢了（与"服务端没回"必须分开）
        this.ledger.droppedFrames += 1;
        this.ledger.droppedBytes += (ev.data as ArrayBuffer).byteLength ?? 0;
        this.emitDiagnostics();
        return;
      }
      const seg = this.segments.get(seq);
      if (!seg) return;
      // 载荷级账：服务端真的产出了音频（这是"没声音"排查里最关键的一个数）
      this.ledger.pcmFrames += 1;
      this.ledger.pcmBytes += (ev.data as ArrayBuffer).byteLength ?? 0;
      // 第一块到手（面板"首段到达"读数）：只记一次
      if (!this.firstPcmMs.has(seq)) {
        this.firstPcmMs.set(seq, Math.round(performance.now() - seg.sentAt));
      }
      const int16 = bytesToInt16(ev.data as ArrayBuffer);
      const sr = seg.sampleRate || this.status.sampleRate;
      if (sr <= 0) {
        // 采样率还没到（协议异常）：攒着，等 end 时兜底处理，别把音频丢掉
        seg.frames.push(int16);
        return;
      }
      // **边收边播**：立刻把这一块排进播放时间轴，不要攒到 `end`。
      // 攒到 end 会把服务端串行合成的等待暴露成句中空洞（实测 686 + 1225 ms）。
      this.player.enqueueChunk(seq, int16, sr, seg.spokenText || seg.text);
      return;
    }

    let msg: Record<string, unknown>;
    try {
      msg = JSON.parse(ev.data);
    } catch {
      return;
    }

    const type = String(msg.type ?? '');

    if (type === 'status') {
      this.status = {
        available: Boolean(msg.available),
        engine: msg.engine ? String(msg.engine) : undefined,
        sampleRate: Number(msg.sample_rate ?? 0),
        numSpeakers: Number(msg.num_speakers ?? 0),
        numThreads: Number(msg.num_threads ?? 0),
        reason: msg.error ? String(msg.error) : undefined,
      };
      if (!this.status.available) {
        this.fail(this.status.reason || '服务端语音合成引擎不可用');
        this.settleConnect?.(false);
      } else {
        this.cb.onStatus?.(this.status);
        // 服务端自述可用 —— 到这里 `ch.status` 才是可信的（见 settleConnect 注释）
        this.settleConnect?.(true);
      }
      return;
    }

    if (type === 'error') {
      const seq = msg.seq === undefined ? undefined : Number(msg.seq);
      this.cb.onError?.(String(msg.message ?? '未知错误'), seq);
      return;
    }

    const seq = Number(msg.seq ?? -1);
    if (seq < 0) return;

    if (type === 'start') {
      // 水位线之下的 `start` = 被取消那一轮的残留，直接忽略（见 cancelledThrough）
      if (seq <= this.cancelledThrough) {
        this.staleStarts += 1;
        this.emitDiagnostics();
        return;
      }
      this.lastStartedSeq = seq;
      // 载荷级账：这一句服务端开始合成
      this.ledger.startFrames += 1;
      // 看到了水位线之上的新段 ⇒ 残留已排空，重新接受二进制帧
      if (this.awaitingFreshStart) this.awaitingFreshStart = false;
      const sr = Number(msg.sample_rate ?? this.status.sampleRate ?? 0);
      const existing = this.segments.get(seq);
      const spokenText = String(msg.spoken_text ?? existing?.spokenText ?? existing?.text ?? '');
      const requested = existing?.text ?? '';
      this.auditSpokenText(seq, requested, spokenText);
      this.segments.set(seq, {
        text: requested,
        // 服务端**在 start 就带上 spoken_text**（术语替换后的真文本）。
        // 没有它，边收边播时 onStart 只能拿到原文，字幕定位与回声过滤会失配。
        spokenText,
        sampleRate: sr,
        frames: existing?.frames ?? [],
        sentAt: existing?.sentAt ?? performance.now(),
      });
      return;
    }

    if (type === 'cancelled') {
      this.segments.delete(seq);
      this.stoppedSeqs.delete(seq);
      if (this.lastStartedSeq === seq) this.lastStartedSeq = null;
      return;
    }

    if (type === 'end') {
      const seg = this.segments.get(seq);
      this.segments.delete(seq);
      if (this.lastStartedSeq === seq) this.lastStartedSeq = null;
      if (!seg) return;

      const sampleRate = Number(msg.sample_rate ?? seg.sampleRate ?? this.status.sampleRate ?? 0);
      const spokenText = String(msg.spoken_text ?? seg.spokenText ?? seg.text);

      const meta: TtsSegmentMeta = {
        seq,
        spokenText,
        sampleRate,
        audioMs: Number(msg.audio_ms ?? 0),
        synthMs: Number(msg.synth_ms ?? 0),
        firstChunkMs: Number(msg.first_chunk_ms ?? 0),
        rtf: Number(msg.rtf ?? 0),
      };

      // 「请求 → 音频到手」：在收尾**之前**记，所以不受播放队列长度影响。
      this.pcmReadyMs.set(seq, Math.round(performance.now() - seg.sentAt));

      // 兜底：采样率一直没拿到，帧只能攒着（正常路径不会走到这里）
      if (seg.frames.length > 0) {
        const pcm = concatInt16(seg.frames);
        seg.frames = [];
        if (sampleRate > 0 && pcm.length > 0) {
          this.player.enqueue(seq, pcm, sampleRate, spokenText);
        }
      }

      // 收尾：这一句不会再有新块了。`onEnd` 由播放器在**最后一块播完**时触发。
      const chunks = this.player.endSegment(seq);
      // ⚠️ 先判"这一句是不是我们主动掐断的"，再决定要不要报"没合成出音频"：
      // 掐断过的句子在播放器里已经查不到记录（`stopAll` 清了表），`chunks` 必然是 0
      // —— 那是**故意丢弃**，不是合成失败。混在一起报会每次都弹一条假错误
      // （2026-10-11 截图：改稿停播后同一轮的句子被误报）。
      const stoppedByUs = this.stoppedSeqs.delete(seq)
        || seq <= this.cancelledThrough
        || this.awaitingFreshStart;
      // 载荷级账：这一句收尾（含被取消的句 —— 服务端的 `cancelled` 帧也记在这里）
      this.ledger.endFrames += 1;
      if (chunks === 0) {
        if (stoppedByUs) {
          this.stoppedWithoutAudio += 1;
          this.emitDiagnostics();
        } else {
          // 合成成功但没音频（空文本 / 全被 OOV 吞掉）。如实上报，不假装播过。
          this.ledger.emptyAudioSegments += 1;
          this.emitDiagnostics();
          this.cb.onError?.('这一句没有合成出音频', seq);
        }
      }
      this.cb.onSegment?.(meta);
      // 真机排查"没有声音"的落点：这一句到底**有没有音频**、被我丢了没有。
      // （`chunks` 是播放器里这一句收到的块数；0 且不是我们掐的 = 服务端没出音频。）
      console.log(
        `[TTS] end seq=${seq} audio=${meta.audioMs}ms chunks=${chunks} stoppedByUs=${stoppedByUs}`,
      );
    }
  }

  /** 最近一个收到 `start` 的 seq —— 二进制帧靠它归属 */
  private sendSay(seq: number, text: string): void {
    const ws = this.ws;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    try {
      ws.send(JSON.stringify({ type: 'say', seq, text }));
    } catch {
      this.cb.onError?.('发送合成请求失败', seq);
    }
  }

  /**
   * 请求合成一句。
   *
   * 通道没就绪时先缓存（上限 {@link MAX_PENDING_SAYS}），连上后自动补发 ——
   * 与 `streamingAsr.ts` 对开头音频的处理同一个思路。
   */
  say(seq: number, text: string): void {
    if (this.closed || !text.trim()) return;
    this.ledger.sayFrames += 1;
    this.segments.set(seq, {
      text,
      // 先按原文占位；`start` 事件到达时会换成服务端的 spoken_text
      spokenText: text,
      sampleRate: this.status.sampleRate,
      frames: [],
      sentAt: performance.now(),
    });
    if (this.isOpen) {
      this.sendSay(seq, text);
      return;
    }
    this.pendingSays.push({ seq, text });
    if (this.pendingSays.length > MAX_PENDING_SAYS) this.pendingSays.shift();
  }

  /** 取消某一句（清服务端待合成队列 + 丢弃在途结果 + 不发音频） */
  cancel(seq: number): void {
    this.segments.delete(seq);
    this.pendingSays = this.pendingSays.filter((p) => p.seq !== seq);
    const ws = this.ws;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    try {
      ws.send(JSON.stringify({ type: 'cancel', seq }));
    } catch {
      /* 忽略：取消丢了最多是白合成一句，不影响正确性 */
    }
  }

  /**
   * 中止「seq ≥ `minSeq`」的**全部**在途合成 —— 打断 / 改稿 / 危机用（S5 / S6）。
   *
   * ## 为什么需要它（2026-10-11 补的洞）
   *
   * 通话页在收到 `speak` 计划时**一次把所有句子都 `say` 出去**（一句话一个 seq），
   * 服务端串行合成、边合成边推 PCM。而客户端原先的"打断"只做了 `stopAll()`
   * ——那只清掉了**本地已经排上时间轴**的音频。于是：
   *
   * ```
   *   打断 → stopAll() 静音
   *        → 服务端还在合成第 3、4、5 句（客户端从没告诉它别做了）
   *        → 几百毫秒后 PCM 到 → 客户端照常入队播放
   *        → 用户听到"AI 停了一下又自己说起来"，且是从半句开始的
   * ```
   *
   * `docs/call_implementation_steps.md` 的 S5/S6 一直写着"中止在途合成（WS 发 cancel）"，
   * 但客户端从来没调用过 `cancel` —— 这条不变式一直只写在文档里。
   *
   * 现在两步一起做：① 服务端 `cancel`（`api/tts.py` 会从待合成队列里摘掉、并让
   * 正在跑的合成回调返回非 0 提前结束）；② 本地 `segments.delete`，
   * 于是**已经在路上的**二进制帧到了也会被 `onMessage` 丢掉（它查不到 segment）。
   *
   * 两条路径都不依赖对方：WS 断了也能保证本地不再出声（代价是服务端白合成几句）。
   *
   * @param minSeq 起算句号（含）。调用方传"当前已发出的最大 seq"即可中止整轮。
   * @returns 被中止的句号列表（空数组表示本来就没有在途音频）
   */
  cancelFrom(minSeq: number): number[] {
    const targets = [...this.segments.keys()].filter((seq) => seq >= minSeq);
    if (targets.length === 0) return targets;
    for (const seq of targets) this.cancel(seq);
    // 抬高水位线 + 进入"排空残留"：取消前已经上路的那几帧，绝不能当成下一轮的语音。
    // （二进制帧不带 seq，只能靠"先看到水位线之上的 start"来判断残留排空了。）
    this.cancelledThrough = Math.max(this.cancelledThrough, minSeq);
    this.awaitingFreshStart = true;
    this.emitDiagnostics();
    return targets;
  }

  /**
   * 登记"**被我们自己掐断**"的句号（调用方：`TtsChannel.stopAll` / `stopFrom`
   * 在 `player.stopAll()` 之前把播放器当时还挂着的句号取下来传进来）。
   *
   * 这些句子的 `end` 稍后仍会到达，但播放器里的记录已经被 `stopAll()` 清掉了，
   * 不登记就会被误报成"这一句没有合成出音频"（见 {@link stoppedSeqs}）。
   * 只登记 seq **小于**取消水位线的那批；`cancelFrom` 覆盖的 seq 会收到
   * `cancelled` 帧，走另一条分支（那一条本来就不报错）。
   *
   * @param seqs 被掐断时仍在播放队列里的句号
   */
  markPlayerStopped(seqs: number[]): void {
    for (const seq of seqs) this.stoppedSeqs.add(seq);
  }

  close(): void {
    this.closed = true;
    this.closing = true;
    this.pendingSays = [];
    this.segments.clear();
    this.pcmReadyMs.clear();
    this.firstPcmMs.clear();
    this.stoppedSeqs.clear();
    const ws = this.ws;
    if (ws && ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(JSON.stringify({ type: 'close' }));
      } catch {
        /* 忽略 */
      }
    }
    try {
      ws?.close();
    } catch {
      /* 忽略 */
    }
    this.ws = null;
  }

  /**
   * 取某一句的「请求 → 音频到手」耗时（毫秒）；还没合成完时为 undefined。
   *
   * 与 `TtsPlayer` 的 `onStart`（真正出声）的区别，以及该用哪个：
   *   · 本方法 —— 服务端合成 + 下行，**与播放队列无关**。
   *     判断"TTS 慢不慢"、S4 闸门的首声预算，都用它。
   *   · `onStart` —— 真正出声的时刻，**含排队等待**。
   *     判断"用户什么时候听到声音"用它，但它天然随队列变长而变大，
   *     把第 3 句的读数当"首声"报会凭空多出前两句的时长。
   */
  takePcmReadyMs(seq: number): number | undefined {
    const v = this.pcmReadyMs.get(seq);
    this.pcmReadyMs.delete(seq);
    return v;
  }

  /** 取某一句「请求 → **第一块** PCM 到手」的耗时；还没到过任何块时为 undefined */
  takeFirstPcmMs(seq: number): number | undefined {
    const v = this.firstPcmMs.get(seq);
    this.firstPcmMs.delete(seq);
    return v;
  }

  /**
   * 已发出 `say`、但服务端还没回 `end` 的句数 —— "TTS 队列里还堆着几句"。
   *
   * 服务端是**串行**合成：这个数 > 0 就说明我们要等前面那几句合成完，
   * 与"合成本身慢"是两件事。
   */
  pendingUnits(): number {
    return this.segments.size;
  }

  private fail(reason: string): void {
    this.status = { ...this.status, available: false, reason };
    this.cb.onStatus?.(this.status);
    this.cb.onError?.(reason);
  }

  // ── 对账 / 隔离（见 TtsStreamDiagnostics）────────────────────────────────
  private emitDiagnostics(): void {
    this.cb.onDiagnostics?.(this.diagnostics);
  }

  /** 当前对账/隔离读数（面板用；不触发回调） */
  get diagnostics(): TtsStreamDiagnostics {
    return {
      drainedFrames: this.drainedFrames,
      drainedBytes: this.drainedBytes,
      staleStarts: this.staleStarts,
      crossTalk: this.crossTalk,
      normalizedText: this.normalizedText,
      cancelledThrough: this.cancelledThrough,
      stoppedWithoutAudio: this.stoppedWithoutAudio,
    };
  }

  /** 载荷级账（"发了几句 / 回了多少字节 / 丢了多少"）—— 见 {@link TtsPayloadLedger} */
  get payloadLedger(): TtsPayloadLedger {
    return { ...this.ledger };
  }

  /**
   * 校验 `start.spoken_text` 与**本 seq 请求的文本**是否对得上。
   *
   * 为什么要单独判一次：这一步是"文字与语音不匹配"唯一能在客户端本地抓住的地方。
   *
   *   · 两种情况都算**正常**：术语规范化（服务端把 `CBT` 读成"认知行为疗法"）、
   *     服务端在 `prepare_for_speech` 里做的其它规范化。差别记进读数，不当作故障。
   *   · 真正危险的是**串单**：这一段收到的文本，恰好是**另一个待播 seq** 的请求文本
   *     —— 那说明音频串到了别的片段上（或 seq 关联错了）。此时**不能照常播放**：
   *     删掉本段，它的 PCM 会被 `!seg` 那一关丢掉，宁可这一句没声音，
   *     也不要让用户听到"另一句话的声音配这几句字幕"。
   */
  private auditSpokenText(seq: number, requested: string, spokenText: string): void {
    if (!requested || !spokenText) return;
    const a = normalizeForCompare(requested);
    const b = normalizeForCompare(spokenText);
    if (a === b) return;
    // 是不是"别人的文本"？
    for (const [other, seg] of this.segments) {
      if (other === seq) continue;
      const otherText = normalizeForCompare(seg.text || seg.spokenText);
      if (otherText && otherText === b) {
        this.crossTalk += 1;
        this.ledger.textMismatch += 1;
        this.recordMismatch(seq, requested, spokenText, '疑似串单：收到的是别的 seq 的文本');
        console.warn(
          `[TTS] 疑似串单：seq=${seq} 收到的是 seq=${other} 的文本`,
          { requested, spokenText },
        );
        this.segments.delete(seq);
        if (this.lastStartedSeq === seq) this.lastStartedSeq = null;
        this.emitDiagnostics();
        return;
      }
    }
    this.normalizedText += 1;
    // 术语规范化之外的差异也要记账 —— "AI 偶尔说出乱话"时，
    // 这一格就是"送出去的就是那句"还是"念错了"的分界线。
    this.ledger.textMismatch += 1;
    this.recordMismatch(seq, requested, spokenText, '文本与请求不一致（非串单）');
    this.emitDiagnostics();
  }

  /** 记一组"请求 → 实际念出"（上限 5 组，面板展示用） */
  private recordMismatch(seq: number, requested: string, spoken: string, reason: string): void {
    const clip = (s: string): string => (s.length > 48 ? `${s.slice(0, 48)}…` : s);
    this.ledger.textMismatchSamples.push({
      seq,
      requested: clip(requested),
      spoken: clip(spoken),
      reason,
    });
    if (this.ledger.textMismatchSamples.length > 5) this.ledger.textMismatchSamples.shift();
  }
}

/**
 * 通话页直接用的门面：**合成 + 播放**。
 *
 * 它把"说一句话"收敛成一次 `say(seq, text)` 调用，并对外暴露
 * `recentSpoken()`（回声过滤）与 `stopAll()`（打断）。
 */
export class TtsChannel {
  private readonly player: TtsPlayer;
  private readonly stream: TtsStreamSession;
  private readonly cb: TtsChannelCallbacks;
  private playerReady = false;

  /** 每个 seq 从 `say()` 到**真正出声**的耗时（毫秒）—— 端到端首声，最该被量的数 */
  private readonly firstSoundMs = new Map<number, number>();
  private readonly saySentAt = new Map<number, number>();

  /**
   * @param cb 回调
   * @param prebufferMs **开场预缓冲**（毫秒）。不传就用 {@link DEFAULT_PREBUFFER_MS}
   *   （300ms，用户真机试听后定的档位）；实测依据与边界见 `ttsPlayer.ts` 的同名常量。
   */
  constructor(cb: TtsChannelCallbacks = {}, prebufferMs?: number) {
    this.cb = cb;
    this.player = new TtsPlayer({
      onStart: (seq, text, timing: TtsStartTiming) => {
        const sentAt = this.saySentAt.get(seq);
        if (sentAt !== undefined) {
          // 端到端首声 = 从 `say()` 到**真正出声**（用墙钟，因为用户感知的是墙钟）。
          // ⚠️ 不能用 timing.ctxStartMs 减 ctx 域的时间：早期音频时钟可能还没起步
          // （实测第一个 source 开始渲染前 ctx 恒为 0），那样会低估成几十毫秒。
          this.firstSoundMs.set(seq, performance.now() - sentAt);
          this.saySentAt.delete(seq);
        }
        this.cb.onStart?.(seq, text, timing);
      },
      onEnd: (seq, text) => this.cb.onEnd?.(seq, text),
      onDrained: () => this.cb.onDrained?.(),
    }, prebufferMs);
    this.stream = new TtsStreamSession(cb, this.player);
  }

  /** 建立通道 + 唤醒 AudioContext（应在用户手势里调用） */
  async start(): Promise<boolean> {
    await this.player.resume();
    this.playerReady = true;
    return this.stream.connect();
  }

  /** 合成并播放一句。非阻塞 —— 绝不能 await，否则会把轮次编排卡住。 */
  say(seq: number, text: string): void {
    if (!this.playerReady) {
      // 静默返回是原来最难查的一种"没声音"：看着什么错都没有，其实一个字都没发。
      // 真机排查（2026-10-11 "两次都没有声音"）需要能一眼看到它，所以出一条日志。
      console.warn(`[TTS] 通道还没就绪，这一句没有送合成（seq=${seq}）：`, text.slice(0, 40));
      return;
    }
    this.saySentAt.set(seq, performance.now());
    console.log(`[TTS] say seq=${seq}（${text.length} 字）：`, text.slice(0, 60));
    this.stream.say(seq, text);
  }

  /** 取消某一句（还没播到的那句） */
  cancel(seq: number): void {
    this.stream.cancel(seq);
    this.saySentAt.delete(seq);
  }

  /** 立刻掐断全部播放 —— **打断路径**，同步完成 */
  stopAll(): void {
    // 与 stopFrom 同一个理由：先把还在播放器里的句号登记成"我们掐断的"，
    // 否则它们稍后到达的 `end` 会被误报成"没合成出音频"。
    const inFlight = this.player.openSeqs();
    this.player.stopAll();
    this.stream.markPlayerStopped(inFlight);
  }

  /**
   * 「这一轮到此为止」—— 本地立刻静音 **且** 通知服务端取消在途合成。
   *
   * 打断（自动/手动）、审计改稿、危机切换都用它，替代裸的 `stopAll()`：
   * 只用 `stopAll()` 时服务端仍会把这一轮剩下的句子合成完再推过来，
   * 客户端会把它们照常播出来 —— 用户听到的是"停一下又自己说起来"。
   * 判据与两条路径的细节见 {@link TtsStreamSession.cancelFrom}。
   *
   * ⚠️ 两步必须在**同一个同步块**里：中间的 await/事件回调会让一个已经在路上的
   * PCM 帧挤进来重新入队（JS 单线程，同步块内不可能发生）。
   *
   * @param minSeq 起算句号（含）；调用方传"当前已发出的最大 seq"即中止整轮。
   * @returns 被中止的在途句数（0 = 本来就没有在途音频）
   */
  stopFrom(minSeq: number): number {
    // ⚠️ 顺序不能变：**先**取"此刻还挂在播放器里的句号"，**再** stopAll()（它会清表）。
    // 这批句号里 `seq < minSeq` 的那几句不会被 cancelFrom 取消 —— 它们的 `end` 稍后
    // 仍会到达，必须让会话层知道"是我们掐断的"，否则每条都会误报
    // "这一句没有合成出音频"（真机截图 2026-10-11，见 TtsStreamSession.stoppedSeqs）。
    const inFlight = this.player.openSeqs();
    this.player.stopAll();
    this.stream.markPlayerStopped(inFlight);
    const cancelled = this.stream.cancelFrom(minSeq);
    // 被取消的句子不会再出声，通道级的两个计时 Map 也要跟着清 —— 不清就是
    // 每打断一次留几条永不回收的条目（长通话里会一直涨）。
    for (const seq of cancelled) {
      this.saySentAt.delete(seq);
      this.firstSoundMs.delete(seq);
    }
    return cancelled.length;
  }

  /** 最近播出去的文本，供文本级自回声过滤比对 */
  recentSpoken(withinMs?: number): string[] {
    return this.player.recentSpoken(withinMs);
  }

  /** 取某一句的端到端首声耗时（毫秒）；还没出声时为 undefined */
  takeFirstSoundMs(seq: number): number | undefined {
    const v = this.firstSoundMs.get(seq);
    this.firstSoundMs.delete(seq);
    return v;
  }

  /** 取某一句的「请求 → 音频到手」耗时（毫秒）；见 `TtsStreamSession.takePcmReadyMs` 的取舍说明 */
  takePcmReadyMs(seq: number): number | undefined {
    return this.stream.takePcmReadyMs(seq);
  }

  /** 取某一句的「请求 → **第一块** PCM 到手」耗时（毫秒）；还没到过块时为 undefined */
  takeFirstPcmMs(seq: number): number | undefined {
    return this.stream.takeFirstPcmMs(seq);
  }

  /**
   * 播放器里**还排着**的音频时长（毫秒）。
   *
   * 「首声为什么这么晚」的两个候选之一（另一个是服务端 `first_chunk_ms`）：
   * 本值大 = 前面还有别的音频要播完，用户听到的等待是**排队**，不是合成慢。
   */
  pendingAudioMs(): number {
    return this.player.pendingAudioMs();
  }

  /** 已 `say` 但服务端还没回 `end` 的句数（服务端串行合成，>0 就是还堆着几句） */
  pendingUnits(): number {
    return this.stream.pendingUnits();
  }

  get isSpeaking(): boolean {
    return this.player.isPlaying;
  }

  /**
   * 播放空洞（欠载）的**累计**实测统计 —— 「通话卡不卡」的第一手读数。
   *
   * 判据、字段含义与"为什么不需要听音频就能量"写在
   * `ttsPlayer.ts` 的 {@link TtsUnderrunStats} 上。单轮的增量由调用方取两次快照相减。
   */
  get underrun(): TtsUnderrunStats {
    return this.player.underrun;
  }

  /**
   * 合成链路的**对账/隔离**读数（串单次数、取消后丢弃的残留帧/字节、陈旧 start）。
   *
   * 与 `underrun`（播放侧空洞）分开：这两个读数一条量"音频与文本对不对得上"，
   * 另一条量"音频有没有断"。用户看到的"不匹配 + 输出不全"正好是这两条线上的病。
   */
  get diagnostics(): TtsStreamDiagnostics {
    return this.stream.diagnostics;
  }

  /**
   * 载荷级账（发了几句 / 回了多少字节 / 丢了多少）—— 见 {@link TtsPayloadLedger}。
   *
   * 这是"**完全没声音**"和"慢"的分界线：时间读数在那一种情况下只会是一片 `—`。
   */
  get payloadLedger(): TtsPayloadLedger {
    return this.stream.payloadLedger;
  }

  /** 是否已释放；释放后 `say()` 是空操作（不会静默重建连接） */
  get isDisposed(): boolean {
    return this.player.isDisposed;
  }

  get status(): TtsStreamStatus {
    return this.stream.currentStatus;
  }

  get deviceSampleRate(): number {
    return this.player.deviceSampleRate;
  }

  /**
   * 输出侧健康读数（AudioContext 状态 + 累计排程块数/时长）——见 `TtsPlayer.outputHealth`。
   *
   * "音频到了客户端却没听见"与"根本没合成出音频"的**分界线**就在这里。
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
    return this.player.outputHealth;
  }

  /**
   * **新一轮开始**：让开场预缓冲重新有资格生效一次。
   *
   * 调用点必须是"真正开一轮"的地方（`useVoiceCall.runTurn`）。见
   * `prebufferGate.ts` 顶部那段：挂到"队列播空"上会在阵发间隙里重复触发。
   */
  armPrebufferTurn(): void {
    this.player.armPrebufferTurn();
  }

  /** 当前生效的开场预缓冲（毫秒）。0 = 收到第一块就播（既有行为） */
  get prebufferMs(): number {
    return this.player.prebuffer;
  }

  /**
   * 播放缓冲区统计（见 `ttsBufferLog.ts`）：句内见底 vs 跨句见底 —— 卡顿病因的分界线。
   *
   * `bufferMs` 是**当前水位**：它一路贴着 0 就是"供音跟不上播放"（该预缓冲或提速），
   * 只有跨句那一下空就是"片段调度"（该提前发下一句）。
   */
  get bufferStats(): {
    chunks: number;
    withinSentenceStarves: number;
    gapStarves: number;
    starveTotalMs: number;
    bufferMs: number;
  } {
    return this.player.bufferStats;
  }

  /** 缓冲区日志明细（逐块水位 + 见底现场）；用户要"卡顿那一刻的日志"就是它 */
  bufferLogDump(recentChunks = 12): string {
    return this.player.bufferLogDump(recentChunks);
  }

  /** AudioContext 播放时钟（毫秒）—— 与墙钟一起看可判断排程是否错位 */
  get currentTimeMs(): number {
    return this.player.currentTimeMs;
  }

  dispose(): void {
    this.stream.close();
    this.player.dispose();
    this.firstSoundMs.clear();
    this.saySentAt.clear();
  }
}
