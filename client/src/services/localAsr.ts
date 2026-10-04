/**
 * 本地语音转文字（Local ASR）—— 离线整段识别路径
 *
 * ⚠️ 当前**不是默认路径**：`useVoiceAnalysis` 已改用 `streamingAsr.ts` 的流式识别
 *    （边说边出字）。本文件保留的是"VAD 攒完整句 → HTTP 整段识别"这条更准但更慢的
 *    路径，对应的后端端点 `POST /api/v1/asr/transcribe` 仍然在线且经过验证。
 *    目前没有任何调用方；若确定不做"录完再识别"的批处理模式，可连同后端端点一起删除。
 *
 * 与 `streamingAsr.ts` 的分工：
 *   本文件         VAD 分句后整段识别，句末 + ~50ms 出文本；用 78 MB Paraformer，更准
 *   streamingAsr   100ms 分片推流，~700ms 出首条中间结果；用 24 MB zipformer，更快
 *
 * 替换掉浏览器 `webkitSpeechRecognition` 的原因：它并非本地能力 —— Edge 走微软
 * Azure、Chrome 走谷歌云（微软策略文档 `SpeechRecognitionEnabled` 原文：
 * "The Microsoft Edge implementation of the Web Speech API uses Azure
 * Cognitive Services, so voice data leaves the machine"），既把未成年人音频送出
 * 设备、又依赖外网；旧实现的 `onend` 零延迟重启还会把一次瞬时失败放大成永久报错风暴。
 *
 * 音频只在「浏览器 → 本机后端」之间流转，不出这台机器，也不依赖外网。
 */

import api from './api';

export const ASR_SAMPLE_RATE = 16000;

/** 请求体：与后端 `algorithm/api/asr.py` 的接口约定一致 */
export interface AsrTranscribeRequest {
  audio_base64: string;
  sample_rate: number;
}

/** 后端返回（与 `server/src/services/algorithm-bridge.ts` 的 AsrTranscribeResponse 对齐） */
export interface AsrTranscribeResult {
  text: string;
  engine: string;
  duration_ms: number;
  latency_ms: number;
  /** 非空表示本次识别失败（引擎不可用 / 音频无法解析），由 UI 如实展示 */
  error: string;
}

// ── VAD 参数（可在运行期调）────────────────────────────────────────────────
const SPEECH_RMS = 0.012; // 判定「有人在说话」的短时能量阈值
const SILENCE_MS = 400; // 连续静音多久算一句说完
const MIN_SPEECH_MS = 1000; // 一句话短于此不送识别（避免空转）
const MAX_SPEECH_MS = 8000; // 一句话长于此强制切分（Paraformer 非流式，需控长）
const PREROLL_MS = 240; // 触发瞬间向前多留一点，避免吃掉首字
const BLOCK_MS = 100;

const WORKLET_URL = '/worklets/pcm-capture-processor.js';

export interface LocalAsrCallbacks {
  /** 识别出一段文本。`text` 已 trim，空串不会回调。 */
  onText?: (text: string, meta: AsrTranscribeResult) => void;
  /** 状态变化：可用性 / 最近一次错误原因，供 UI 如实展示 */
  onStatus?: (status: { available: boolean; reason?: string }) => void;
}

interface Block {
  samples: Float32Array;
  rms: number;
}

/** Float32(-1..1) → 16bit PCM WAV */
export function encodeWav16(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buf);
  const str = (off: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  };
  str(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  str(8, 'WAVE');
  str(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // 单声道
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // 位深
  str(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  let off = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    off += 2;
  }
  return buf;
}

/** ArrayBuffer → base64（分块拼接，避免 apply 参数上限） */
export function toBase64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  const CHUNK = 0x8000;
  let bin = '';
  for (let i = 0; i < bytes.length; i += CHUNK) {
    bin += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(bin);
}

/**
 * 一次采集会话。生命周期与 `useVoiceAnalysis.start()/stop()` 对齐。
 */
export class LocalAsrSession {
  private ctx: AudioContext | null = null;
  private node: AudioWorkletNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private sink: GainNode | null = null;
  private readonly cb: LocalAsrCallbacks;

  // VAD 状态
  private state: 'idle' | 'speech' = 'idle';
  private speechBlocks: Block[] = [];
  private preroll: Block[] = [];
  private speechSamples = 0;
  private silenceMs = 0;

  private stopped = false;
  /** 连续失败计数：达到上限后停止尝试，避免旧实现那种无限报错风暴 */
  private failures = 0;
  private static readonly MAX_FAILURES = 3;

  constructor(cb: LocalAsrCallbacks = {}) {
    this.cb = cb;
  }

  /** 采集是否已就绪且未因连续失败而停用 */
  get active(): boolean {
    return !!this.ctx && !this.stopped && this.failures < LocalAsrSession.MAX_FAILURES;
  }

  /**
   * 启动。音频图：source → worklet → 静音 Gain → destination。
   * 必须接到 destination（哪怕是 0 增益）才会被浏览器持续调度。
   */
  async start(stream: MediaStream): Promise<void> {
    const Ctor = window.AudioContext || (window as any).webkitAudioContext;
    if (!Ctor) {
      this.cb.onStatus?.({ available: false, reason: '浏览器不支持 WebAudio' });
      return;
    }
    if (!(window as any).AudioWorkletNode) {
      this.cb.onStatus?.({ available: false, reason: '浏览器不支持 AudioWorklet' });
      return;
    }

    // 16kHz：让浏览器替我们重采样，worklet 里就不必再处理采样率转换
    const ctx: AudioContext = new Ctor({ sampleRate: ASR_SAMPLE_RATE });
    await ctx.audioWorklet.addModule(WORKLET_URL);

    const source = ctx.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(ctx, 'pcm-capture');
    const sink = ctx.createGain();
    sink.gain.value = 0; // 只为让节点被调度，不出声

    node.port.onmessage = (e: MessageEvent) => this.onMessage(e.data);
    source.connect(node);
    node.connect(sink);
    sink.connect(ctx.destination);

    // 部分浏览器创建后处于 suspended，需要一次用户手势后 resume
    if (ctx.state === 'suspended') {
      try {
        await ctx.resume();
      } catch {
        /* 忽略：仍可能正常工作 */
      }
    }

    this.ctx = ctx;
    this.node = node;
    this.source = source;
    this.sink = sink;
    this.stopped = false;
    this.cb.onStatus?.({ available: true });
  }

  private onMessage(data: any): void {
    if (!data) return;
    if (data.type === 'flushed') {
      if (data.samples?.length) this.pushBlock({ samples: data.samples, rms: 0 });
      this.emitPending(true);
      return;
    }
    if (data.type === 'block') {
      this.pushBlock({ samples: data.samples, rms: data.rms });
    }
  }

  /** 把一块音频喂进 VAD，必要时切句送识别 */
  private pushBlock(b: Block): void {
    if (this.state === 'idle') {
      this.preroll.push(b);
      const keep = Math.ceil(PREROLL_MS / BLOCK_MS);
      if (this.preroll.length > keep) this.preroll.shift();

      if (b.rms >= SPEECH_RMS) {
        // 检测到说话：把 preroll 一并带上，避免吃掉首字
        this.state = 'speech';
        this.speechBlocks = this.preroll.slice();
        this.speechSamples = this.speechBlocks.reduce((n, x) => n + x.samples.length, 0);
        this.preroll = [];
        this.silenceMs = 0;
      }
      return;
    }

    this.speechBlocks.push(b);
    this.speechSamples += b.samples.length;
    if (b.rms >= SPEECH_RMS) this.silenceMs = 0;
    else this.silenceMs += BLOCK_MS;

    const speechMs = (this.speechSamples / ASR_SAMPLE_RATE) * 1000;
    if (speechMs >= MAX_SPEECH_MS || (this.silenceMs >= SILENCE_MS && speechMs >= MIN_SPEECH_MS)) {
      this.emitPending(false);
    }
  }

  /** 把当前句子的音频送去识别，并重置 VAD 状态 */
  private emitPending(final: boolean): void {
    const blocks = this.speechBlocks;
    const total = this.speechSamples;
    this.state = 'idle';
    this.speechBlocks = [];
    this.speechSamples = 0;
    this.silenceMs = 0;

    const minSamples = (MIN_SPEECH_MS / 1000) * ASR_SAMPLE_RATE;
    if (total < minSamples) return;

    const merged = new Float32Array(total);
    let off = 0;
    for (const blk of blocks) {
      merged.set(blk.samples, off);
      off += blk.samples.length;
    }
    void this.transcribe(merged, final);
  }

  private async transcribe(samples: Float32Array, final: boolean): Promise<void> {
    if (!this.active) return;
    try {
      const wav = encodeWav16(samples, ASR_SAMPLE_RATE);
      const payload: AsrTranscribeRequest = {
        audio_base64: toBase64(wav),
        sample_rate: ASR_SAMPLE_RATE,
      };
      const res = (await api.post('/algorithm/asr/transcribe', payload)) as any;
      const data: AsrTranscribeResult | undefined = res?.data ?? undefined;
      if (!data) {
        this.noteFailure('算法服务返回空响应');
        return;
      }
      this.failures = 0;
      const text = (data.text || '').trim();
      if (text) this.cb.onText?.(text, data);
    } catch (err: any) {
      this.noteFailure(err?.message || String(err));
    }
    void final;
  }

  private noteFailure(reason: string): void {
    this.failures += 1;
    if (this.failures >= LocalAsrSession.MAX_FAILURES) {
      this.cb.onStatus?.({
        available: false,
        reason: `本地识别连续失败 ${this.failures} 次，已停止尝试：${reason}`,
      });
    } else {
      this.cb.onStatus?.({ available: true, reason });
    }
  }

  /** 通知 worklet 把残余样本吐出来，然后关闭音频图 */
  async stop(): Promise<void> {
    this.stopped = true;
    try {
      this.node?.port.postMessage({ type: 'flush' });
    } catch {
      /* 忽略 */
    }
    // 给 flush 一点时间回传，再拆图
    await new Promise((r) => setTimeout(r, 150));
    try {
      this.node?.port.close();
      this.node?.disconnect();
      this.source?.disconnect();
      this.sink?.disconnect();
      await this.ctx?.close();
    } catch {
      /* 忽略释放失败 */
    }
    this.ctx = null;
    this.node = null;
    this.source = null;
    this.sink = null;
    this.state = 'idle';
    this.speechBlocks = [];
    this.preroll = [];
    this.speechSamples = 0;
  }
}
