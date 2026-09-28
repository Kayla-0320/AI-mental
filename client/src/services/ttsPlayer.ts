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
 * 模型的输出是 16 kHz，而设备通常是 44.1/48 kHz。这里**不**把 AudioContext
 * 钉在 16 kHz（部分设备会直接构造失败），而是把 16 kHz 的 PCM 装进
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

/** 一句在队列里的排程记录 */
interface ScheduledEntry {
  seq: number;
  text: string;
  source: AudioBufferSourceNode;
  /** 墙钟时间的预计开始/结束时刻（`performance.now()` 域） */
  startsAt: number;
  endsAt: number;
  /** 到点触发 `onStart` 的定时器；已触发或已清理时为 0 */
  startTimer: number;
}

/** `recentSpoken()` 默认回溯窗口：覆盖"AI 刚说完"到"用户插话被识别"的间隔 */
const DEFAULT_SPOKEN_WINDOW_MS = 8000;

/** 文本记录上限，防长通话里无限增长 */
const MAX_SPOKEN_LOG = 40;

export class TtsPlayer {
  private ctx: AudioContext | null = null;
  private readonly cb: TtsPlayerCallbacks;

  /** 下一句的排程起点（`ctx.currentTime` 域）。0 表示队列空。 */
  private nextStartTime = 0;
  private readonly scheduled = new Map<number, ScheduledEntry>();
  private spokenLog: SpokenEntry[] = [];
  private disposed = false;

  constructor(cb: TtsPlayerCallbacks = {}) {
    this.cb = cb;
  }

  // ── 状态 ────────────────────────────────────────────────────────────────
  /** 是否有句子正在播（已排程但还没结束） */
  get isPlaying(): boolean {
    return this.scheduled.size > 0;
  }

  /** 队列里还有几句 */
  get pendingCount(): number {
    return this.scheduled.size;
  }

  /** AudioContext 的实际采样率（设备值，不是模型值）；未创建时为 0 */
  get deviceSampleRate(): number {
    return this.ctx?.sampleRate ?? 0;
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
   * 追加一句音频并立即排程。
   *
   * 排程用 `max(now, nextStartTime)`，所以：
   *   * 队列空 → 立刻播（首声延迟 = 0，不额外等待）；
   *   * 队列非空 → 紧接上一句结束（句间无缝隙）。
   *
   * @param seq 序号，用于把 `onStart`/`onEnd` 与请求对上
   * @param pcm int16 小端 PCM（服务端二进制帧的原样内容）
   * @param sampleRate PCM 的采样率（服务端 `status` 里报的模型采样率）
   * @param text **实际被念出来的文本**（服务端 `spoken_text`，已过术语替换）
   * @returns 是否成功排程。false 表示这段音频被丢弃（空/已销毁/无 AudioContext）
   */
  enqueue(seq: number, pcm: Int16Array, sampleRate: number, text: string): boolean {
    if (this.disposed || pcm.length === 0 || sampleRate <= 0) return false;

    void this.resume();
    const ctx = this.ctx;
    if (!ctx) return false;

    // int16 → float32 [-1, 1]
    const buffer = ctx.createBuffer(1, pcm.length, sampleRate);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) {
      channel[i] = pcm[i] / 32768;
    }

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);

    const startAt = Math.max(ctx.currentTime, this.nextStartTime);
    this.nextStartTime = startAt + buffer.duration;

    const nowWall = performance.now();
    const startsAt = nowWall + (startAt - ctx.currentTime) * 1000;
    const entry: ScheduledEntry = {
      seq,
      text,
      source,
      startsAt,
      endsAt: startsAt + buffer.duration * 1000,
      startTimer: 0,
    };
    this.scheduled.set(seq, entry);

    // `onStart` 必须按**真实出声**时刻触发（见回调文档）。
    // 首句 startAt ≈ currentTime ⇒ 延迟 ≈ 0，等价于立即触发。
    const delayMs = Math.max(0, startsAt - performance.now());
    entry.startTimer = window.setTimeout(() => {
      entry.startTimer = 0;
      if (!this.scheduled.has(seq)) return; // 已被 stopAll 掐断
      this.cb.onStart?.(seq, text, {
        ctxStartMs: startAt * 1000, // 精确：排程时算出来的
        startsAtWallMs: startsAt,   // 近似：见 TtsStartTiming 的文档
      });
    }, delayMs);

    source.onended = () => {
      // 被 stopAll() 掐断时这里也会触发（`onended` 在 stop 后照常派发）。
      // 若这条已经被清理过，就什么都不做 —— 幂等。
      if (!this.scheduled.delete(seq)) return;
      if (entry.startTimer) {
        window.clearTimeout(entry.startTimer);
        entry.startTimer = 0;
      }
      this.logSpoken(seq, text);
      this.cb.onEnd?.(seq, text);
      if (this.scheduled.size === 0) {
        this.nextStartTime = 0;
        this.cb.onDrained?.();
      }
    };

    source.start(startAt);
    return true;
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
    const entries = [...this.scheduled.values()];
    this.scheduled.clear();
    this.nextStartTime = 0;

    for (const e of entries) {
      try {
        if (e.startTimer) {
          window.clearTimeout(e.startTimer);
          e.startTimer = 0;
        }
        e.source.onended = null; // 避免 stop() 触发 onended 再走一遍收尾逻辑
        e.source.stop();
        e.source.disconnect();
      } catch {
        /* 已结束的 source 再 stop 会抛，忽略 */
      }
      this.logSpoken(e.seq, e.text);
    }

    if (entries.length > 0) this.cb.onDrained?.();
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
    // 已排程的（含正在播的与排在后面的）：它们都会经由扬声器进入麦克风，
    // 所以都要参与过滤 —— 宁可多滤一点，也不要漏滤后把 AI 的声音当成用户插话。
    const scheduled = [...this.scheduled.values()]
      .filter((e) => now <= e.endsAt)
      .sort((a, b) => a.startsAt - b.startsAt)
      .map((e) => e.text);
    return [...this.spokenLog.filter((e) => now - e.startedAt <= withinMs).map((e) => e.text), ...scheduled];
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
