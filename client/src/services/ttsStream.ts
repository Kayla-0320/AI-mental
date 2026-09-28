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

import { TtsPlayer, type TtsPlayerCallbacks, type TtsStartTiming } from './ttsPlayer';

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
  /** 服务端合成耗时；实测等于 first_chunk_ms（VITS 不流式） */
  synthMs: number;
  /** 实时率；判据 < 0.4 */
  rtf: number;
}

/** `TtsChannel` 的回调 */
export interface TtsChannelCallbacks extends TtsPlayerCallbacks {
  /** 连接状态变化（含不可用原因），UI 据此如实展示 */
  onStatus?: (status: TtsStreamStatus) => void;
  /** 单句合成完成（已入播放队列） */
  onSegment?: (meta: TtsSegmentMeta) => void;
  /** 出错；`seq` 缺省表示连接级错误 */
  onError?: (message: string, seq?: number) => void;
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
  text: string;
  sampleRate: number;
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
      const seg = this.segments.get(seq);
      if (!seg) return;
      seg.frames.push(bytesToInt16(ev.data as ArrayBuffer));
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
      this.lastStartedSeq = seq;
      const sr = Number(msg.sample_rate ?? this.status.sampleRate ?? 0);
      const existing = this.segments.get(seq);
      this.segments.set(seq, {
        text: existing?.text ?? '',
        sampleRate: sr,
        frames: existing?.frames ?? [],
        sentAt: existing?.sentAt ?? performance.now(),
      });
      return;
    }

    if (type === 'cancelled') {
      this.segments.delete(seq);
      if (this.lastStartedSeq === seq) this.lastStartedSeq = null;
      return;
    }

    if (type === 'end') {
      const seg = this.segments.get(seq);
      this.segments.delete(seq);
      if (this.lastStartedSeq === seq) this.lastStartedSeq = null;
      if (!seg) return;

      const sampleRate = Number(msg.sample_rate ?? seg.sampleRate ?? this.status.sampleRate ?? 0);
      const spokenText = String(msg.spoken_text ?? seg.text);
      const pcm = concatInt16(seg.frames);

      const meta: TtsSegmentMeta = {
        seq,
        spokenText,
        sampleRate,
        audioMs: Number(msg.audio_ms ?? 0),
        synthMs: Number(msg.synth_ms ?? 0),
        rtf: Number(msg.rtf ?? 0),
      };

      // 「请求 → 音频到手」：在送入播放器**之前**记，所以不受播放队列长度影响。
      this.pcmReadyMs.set(seq, Math.round(performance.now() - seg.sentAt));

      if (pcm.length === 0) {
        // 合成成功但没音频（空文本 / 全被 OOV 吞掉）。如实上报，不假装播过。
        this.cb.onError?.('这一句没有合成出音频', seq);
        this.cb.onSegment?.(meta);
        return;
      }

      // 交给播放器。文本用 spokenText —— 回声过滤依赖它。
      this.player.enqueue(seq, pcm, sampleRate, spokenText);
      this.cb.onSegment?.(meta);
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
    this.segments.set(seq, {
      text,
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

  close(): void {
    this.closed = true;
    this.closing = true;
    this.pendingSays = [];
    this.segments.clear();
    this.pcmReadyMs.clear();
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

  private fail(reason: string): void {
    this.status = { ...this.status, available: false, reason };
    this.cb.onStatus?.(this.status);
    this.cb.onError?.(reason);
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
  private playerReady = false;

  /** 每个 seq 从 `say()` 到**真正出声**的耗时（毫秒）—— 端到端首声，最该被量的数 */
  private readonly firstSoundMs = new Map<number, number>();
  private readonly saySentAt = new Map<number, number>();

  constructor(private readonly cb: TtsChannelCallbacks = {}) {
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
    });
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
    if (!this.playerReady) return;
    this.saySentAt.set(seq, performance.now());
    this.stream.say(seq, text);
  }

  /** 取消某一句（还没播到的那句） */
  cancel(seq: number): void {
    this.stream.cancel(seq);
    this.saySentAt.delete(seq);
  }

  /** 立刻掐断全部播放 —— **打断路径**，同步完成 */
  stopAll(): void {
    this.player.stopAll();
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

  get isSpeaking(): boolean {
    return this.player.isPlaying;
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
