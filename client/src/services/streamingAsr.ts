/**
 * 本地流式语音识别客户端（边说边出字，像手机语音输入）
 *
 * 与 `localAsr.ts` 的区别：
 *   localAsr.ts      VAD 攒完一整句 → HTTP 整段识别。延迟 = 句末 + ~50ms，准一点
 *   本文件            AudioWorklet 每 100ms 推一块 → WebSocket → 边解码边回推
 *                    中间结果。首条中间结果约 700ms 出现，之后每 ~330ms 刷新
 *
 * 为什么用原始 WebSocket 而不是 HTTP：
 *   流式要的是"一条持续通道 + 服务端主动推"。HTTP 一问一答做不到，
 *   而每 300ms 发一次 POST 会把连接建立/头部/JSON 开销放大好几倍。
 *
 * 音频路径：麦克风 → AudioContext(16kHz) → AudioWorklet(攒 100ms)
 *        → 掐掉开头静音 → int16 PCM 二进制帧
 *        → ws://…/algorithm/api/v1/asr/ws
 *        → Python 本机 sherpa-onnx（离线 Paraformer / 流式 zipformer）→ 中间结果 / 定稿
 *
 * 两种结束语义（`AsrMode`）：
 *   'utterance'（聊天页语音输入）停顿**不定稿**，整段音频一路累积，用户点"结束"才
 *               定稿；中间结果由服务端用离线大模型对已累积音频周期性解码。
 *   'streaming'（通话页）      说话停顿即定稿，一句一轮 —— 通话需要"一句话一轮"
 *               的交互，断句在那儿是交互需求，不是识别需要。
 *   模式在连接建立后用首帧 `{"type":"hello","mode":…}` 交给服务端。
 *
 * ⚠️ 这里**不再**过滤静音（原先有一道 `SpeechGate` 逐块 VAD）。整段录入要靠句间
 *   停顿维持上下文：实测 12s、句间停顿 1.2s 的音频，端点切段后每一段都错
 *   （"重点呢"→"重点来"、"动荡"→"动量"），整段识别全对。
 *   只保留"开头那段静音不计入"—— 那是唯一会把假文本喂出来的部分。
 *
 * 关闭时的音频不外发：整条链路都在本机（浏览器 → 本机 8001）。
 */

/** 结束语义。见文件头。 */
export type AsrMode = 'utterance' | 'streaming';

/** 服务端回推的一条更新 */
export interface StreamingAsrMessage {
  type: 'partial' | 'final' | 'status' | 'error';
  text?: string;
  elapsed_ms?: number;
  latency_ms?: number;
  /** 仅 final：endpoint（端点检测自动定稿）| finalize（用户主动结束） */
  reason?: string;
  engine?: string;
  /** 仅 final：定稿文本是否经过离线大模型复识别纠错（标点是定稿时无条件尝试的） */
  refined?: boolean;
  available?: boolean;
  message?: string;
}

export interface StreamingAsrMeta {
  elapsedMs: number;
  latencyMs: number;
  reason: string;
  engine: string;
  /** 定稿是否经过离线大模型纠错 */
  refined: boolean;
}

export interface StreamingAsrCallbacks {
  /** 中间结果：会被后续结果覆盖，只用于上屏 */
  onPartial?: (text: string, meta: StreamingAsrMeta) => void;
  /** 定稿：这一段结束，可以送入下游文本/语义分析 */
  onFinal?: (text: string, meta: StreamingAsrMeta) => void;
  /** 状态变化：可用性 / 最近一次错误，供 UI 如实展示 */
  onStatus?: (status: { available: boolean; reason?: string; engine?: string }) => void;
}

const WORKLET_URL = '/worklets/pcm-capture-processor.js';
const SAMPLE_RATE = 16000;
const BLOCK_MS = 100;
/** 连接建立前最多缓存多少块音频（10 × 100ms = 1s），避免无限增长 */
const MAX_PENDING_BLOCKS = 10;
/** 断线重连上限；达到后如实上报不可用，不再无限重试 */
const MAX_RECONNECTS = 3;

/** float 波形 [-1,1] 的 RMS 能量（用于掐掉开头静音） */
export function rmsOf(samples: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return Math.sqrt(sum / samples.length);
}

/** Float32 [-1,1] → int16 小端（服务端按 int16 解析，比 base64 省 33% 带宽） */
export function floatToInt16(samples: Float32Array): ArrayBuffer {
  const buf = new ArrayBuffer(samples.length * 2);
  const view = new DataView(buf);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buf;
}

/** 依据当前页面推出 WebSocket 地址（走 vite 代理的 /algorithm 前缀） */
function resolveWsUrl(): string {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.host}/algorithm/api/v1/asr/ws`;
}

export class StreamingAsrSession {
  private ctx: AudioContext | null = null;
  private node: AudioWorkletNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private sink: GainNode | null = null;

  private ws: WebSocket | null = null;
  private pending: ArrayBuffer[] = [];
  private reconnects = 0;
  private stopped = false;
  private closing = false;

  /** 结束后语义：整段录入 or 说话停顿即定稿。见文件头。 */
  private readonly mode: AsrMode;
  /**
   * 开头静音闸：语音开始之前不送音频。
   *
   * 与已删除的 `SpeechGate` 的区别：它**只关开头那一次**，一旦判定"开始说话了"
   * 就永久打开，句间停顿照常上行 —— 整段录入必须保住停顿的上下文。
   * 阈值取 0.012（≈ -38dBFS）：实测真人说话 rms ≈ 0.088，空调/风扇级底噪
   * -45dBFS ≈ 0.0056，因此留了 ~19dB 余量。
   */
  private speechStarted = false;
  private preroll: ArrayBuffer[] = [];
  private static readonly SPEECH_RMS = 0.012;
  /** 判定"开始说话"需要连续几块超阈（100ms/块） */
  private static readonly CONFIRM_BLOCKS = 2;
  private static readonly PREROLL_BLOCKS = 3;
  private aboveRun = 0;
  /**
   * 是否还允许把音频块送给服务端。
   *
   * 用户点"结束"时立刻置 false 并马上发 `finalize` 冲刷：这样最后一段定稿只等
   * 一次服务端往返（约 100ms），而不是等端点检测把 0.9s 尾随静音收完。
   * 之所以能这么做，是因为 `finalize` 对已收到的语音做最后一次解码 + 离线复识别，
   * 不需要额外补静音（实测 speech_0：立刻冲刷同样得到完整句子）。
   */
  private forwarding = true;

  private readonly cb: StreamingAsrCallbacks;

  constructor(cb: StreamingAsrCallbacks = {}, mode: AsrMode = 'streaming') {
    this.cb = cb;
    this.mode = mode;
  }

  /** 是否处于可用状态（采集已就绪、未停止、未因重连超限而放弃） */
  get active(): boolean {
    return !!this.ctx && !this.stopped && this.reconnects < MAX_RECONNECTS;
  }

  // ── 建立连接 ────────────────────────────────────────────────────────────
  private connect(): Promise<void> {
    return new Promise((resolve) => {
      let url: string;
      try {
        url = resolveWsUrl();
      } catch (err) {
        this.fail(`无法解析 WebSocket 地址：${(err as Error)?.message || err}`);
        resolve();
        return;
      }

      let ws: WebSocket;
      try {
        ws = new WebSocket(url);
      } catch (err) {
        this.fail(`创建 WebSocket 失败：${(err as Error)?.message || err}`);
        resolve();
        return;
      }
      ws.binaryType = 'arraybuffer';
      this.ws = ws;

      // 首次连接/重连都要给个上限，否则后端没起来时会一直挂着
      const openTimer = window.setTimeout(() => {
        if (ws.readyState !== WebSocket.OPEN) {
          try {
            ws.close();
          } catch {
            /* 忽略 */
          }
          this.fail('连接流式识别服务超时（本机算法服务是否在 8001 运行？）');
          resolve();
        }
      }, 5000);

      ws.onopen = () => {
        window.clearTimeout(openTimer);
        this.reconnects = 0;
        // 首帧必须是 hello：服务端据此决定"说话停顿要不要自动定稿"。
        // 必须在任何音频之前发出，否则服务端会按默认的端点模式建会话。
        try {
          ws.send(JSON.stringify({ type: 'hello', mode: this.mode }));
        } catch {
          /* 忽略：服务端会退回默认端点模式，最坏是断句方式不同 */
        }
        // 把建立连接期间攒下的音频补发出去，避免丢开头
        for (const buf of this.pending) {
          try {
            ws.send(buf);
          } catch {
            /* 忽略 */
          }
        }
        this.pending = [];
        resolve();
      };

      ws.onmessage = (ev) => this.onMessage(ev);

      ws.onerror = () => {
        // onerror 之后必然跟 onclose，统一在 onclose 里处理，避免重复上报
      };

      ws.onclose = () => {
        window.clearTimeout(openTimer);
        this.ws = null;
        if (this.stopped || this.closing) return;
        this.reconnects += 1;
        if (this.reconnects >= MAX_RECONNECTS) {
          this.fail(`流式识别连接中断，已重试 ${this.reconnects} 次后停止`);
          return;
        }
        // 退避重连：0.5s / 1s / 1.5s
        window.setTimeout(() => {
          if (!this.stopped && !this.closing) void this.connect();
        }, 500 * this.reconnects);
      };
    });
  }

  private onMessage(ev: MessageEvent): void {
    let msg: StreamingAsrMessage;
    try {
      msg = JSON.parse(typeof ev.data === 'string' ? ev.data : '');
    } catch {
      return;
    }

    if (msg.type === 'status') {
      if (msg.available) {
        this.cb.onStatus?.({ available: true, engine: msg.engine });
      } else {
        this.fail(msg.message || '服务端流式识别引擎不可用');
      }
      return;
    }

    if (msg.type === 'error') {
      this.cb.onStatus?.({ available: this.active, reason: msg.message });
      return;
    }

    const meta: StreamingAsrMeta = {
      elapsedMs: msg.elapsed_ms ?? 0,
      latencyMs: msg.latency_ms ?? 0,
      reason: msg.reason ?? '',
      engine: msg.engine ?? '',
      refined: msg.refined ?? false,
    };

    if (msg.type === 'partial') {
      this.cb.onPartial?.(msg.text ?? '', meta);
    } else if (msg.type === 'final') {
      this.cb.onFinal?.(msg.text ?? '', meta);
    }
  }

  private fail(reason: string): void {
    this.cb.onStatus?.({ available: false, reason });
  }

  // ── 生命周期 ────────────────────────────────────────────────────────────
  async start(stream: MediaStream): Promise<void> {
    const Ctor = window.AudioContext || (window as any).webkitAudioContext;
    if (!Ctor || !(window as any).AudioWorkletNode) {
      this.fail('浏览器不支持 WebAudio / AudioWorklet');
      return;
    }

    this.stopped = false;
    this.closing = false;
    this.reconnects = 0;
    this.forwarding = true;
    this.speechStarted = false;
    this.preroll = [];
    this.aboveRun = 0;

    // 先连通道，再开音频图；这样能尽早把"引擎不可用"报出来
    await this.connect();

    const ctx: AudioContext = new Ctor({ sampleRate: SAMPLE_RATE });
    await ctx.audioWorklet.addModule(WORKLET_URL);

    const source = ctx.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(ctx, 'pcm-capture');
    const sink = ctx.createGain();
    sink.gain.value = 0; // 只为让节点被调度，不出声

    node.port.onmessage = (e: MessageEvent) => {
      const data = e.data;
      if (!data || data.type !== 'block' || !data.samples) return;
      // 已经点结束：后续块一律丢掉，避免它们在服务端冲刷之后又开出一个新段
      if (!this.forwarding) return;
      const samples = data.samples as Float32Array;
      const pcm = floatToInt16(samples);
      const rms = typeof data.rms === 'number' ? data.rms : rmsOf(samples);
      this.pushBlock(pcm, rms);
    };

    source.connect(node);
    node.connect(sink);
    sink.connect(ctx.destination);

    if (ctx.state === 'suspended') {
      try {
        await ctx.resume();
      } catch {
        /* 忽略 */
      }
    }

    this.ctx = ctx;
    this.node = node;
    this.source = source;
    this.sink = sink;
  }

  /**
   * 开头静音闸：语音开始之前只攒不送；一旦确认开始说话，预滚 + 之后的**所有**块
   * 原样上行（含句间停顿，整段录入需要那个上下文）。
   *
   * 为什么只掐开头：把静音喂给声学模型会得到"好的的的的"这类假文本（见
   * `looksLikeDecoderLoop` 的说明）。而句间停顿必须保留 —— 实测端点切段会让
   * 每一段的上下文都不足。
   */
  private pushBlock(pcm: ArrayBuffer, rms: number): void {
    if (this.speechStarted) {
      this.dispatch(pcm);
      return;
    }

    this.preroll.push(pcm);
    while (this.preroll.length > StreamingAsrSession.PREROLL_BLOCKS) this.preroll.shift();

    if (rms >= StreamingAsrSession.SPEECH_RMS) {
      this.aboveRun += 1;
      if (this.aboveRun >= StreamingAsrSession.CONFIRM_BLOCKS) {
        this.speechStarted = true;
        // 把预滚一起发出去，避免吃掉首字（但绝不能因此把整段开头静音都发出去）
        for (const buf of this.preroll) this.dispatch(buf);
        this.preroll = [];
      }
    } else {
      this.aboveRun = 0;
    }
  }

  /** 发一块音频；通道没就绪时先缓存（超过上限则丢弃最旧的） */
  private dispatch(buf: ArrayBuffer): void {
    const ws = this.ws;
    if (ws && ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(buf);
      } catch {
        /* 发送失败交由 onclose 处理 */
      }
      return;
    }
    this.pending.push(buf);
    if (this.pending.length > MAX_PENDING_BLOCKS) this.pending.shift();
  }

  /** 发一条 JSON 控制帧（finalize / reset），通道没就绪时静默跳过 */
  private sendControl(type: 'finalize' | 'reset'): void {
    const ws = this.ws;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    try {
      ws.send(JSON.stringify({ type }));
    } catch {
      /* 忽略：控制帧丢失最多是晚一次定稿，不影响识别内容 */
    }
  }

  /**
   * 用户主动结束。
   *
   * 顺序刻意如此：
   *   1. 先掐断送音频（`forwarding = false`）
   *   2. 立刻发 `finalize` 让服务端冲刷已收到的语音并定稿
   *   3. 留一小段时间收定稿，再拆音频图与连接
   *
   * 等待时长按模式给：整段录入模式下服务端要拿**整段**音频跑一次离线大模型
   * （60s 音频约 0.4s），给 600ms 会偶发地把定稿关在门外。
   * 全程没人说话时只发 `reset`：那段音频里没有任何语音证据，让声学模型去
   * "冲刷残余"只会换回一串假字（实测：纯静音 6s →「我们我我的时呢。」）。
   */
  async stop(): Promise<void> {
    if (this.stopped) return;
    this.stopped = true;
    this.closing = true;
    this.forwarding = false;

    const forwarded = this.speechStarted;
    const ws = this.ws;
    if (ws && ws.readyState === WebSocket.OPEN) {
      this.sendControl(forwarded ? 'finalize' : 'reset');
      const waitMs = !forwarded ? 80 : this.mode === 'utterance' ? 1500 : 600;
      await new Promise((r) => setTimeout(r, waitMs));
    }

    try {
      this.node?.port.close();
      this.node?.disconnect();
      this.source?.disconnect();
      this.sink?.disconnect();
      await this.ctx?.close();
    } catch {
      /* 忽略释放失败 */
    }
    try {
      ws?.close();
    } catch {
      /* 忽略 */
    }

    this.ctx = null;
    this.node = null;
    this.source = null;
    this.sink = null;
    this.ws = null;
    this.pending = [];
    this.speechStarted = false;
    this.preroll = [];
    this.aboveRun = 0;
  }

  /** 丢弃当前这一段（不清空已上屏文本） */
  resetSegment(): void {
    this.sendControl('reset');
  }

  /**
   * 暂停 / 恢复"把音频帧送给服务端"（施工单 §S5.3，半双工窗口用）。
   *
   * ⚠️ **不要用 `track.enabled = false` 代替它**。那会让麦克风真的输出静音，
   * 而服务端收到的是一段连续的静音 —— 端点检测照样会在这段静音之后触发，
   * 于是定稿出一个**空段**，把一段真实语音从中间劈成两半。必须停的是
   * "送帧"这个动作本身：服务端收不到音频，就不会有端点，也不会有假定稿。
   *
   * 暂停期间音频块**直接丢弃**（不缓存、不补发）。这是刻意的：
   * 调用它的场景是"TTS 刚出声的头 400ms"，那一段音频里主要是 AI 自己的声音，
   * AEC 还没收敛；补发只会把回声重新塞进识别链路。代价是如果用户恰好在这
   * 400ms 里开口，开头会被切掉 —— 这是施工单 §S5.5 明确接受的范围。
   *
   * 停轨道与 `AudioContext` 都不受影响，恢复即接着送。
   */
  setForwarding(on: boolean): void {
    if (this.forwarding === on) return;
    this.forwarding = on;
  }

  /** 当前是否在向服务端送帧（`?debug=1` 面板与 S5 闸门读它） */
  get isForwarding(): boolean {
    return this.forwarding;
  }
}

export { BLOCK_MS };
