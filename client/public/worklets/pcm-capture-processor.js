/**
 * PCM 采集 AudioWorklet —— 为本地 ASR 提供 16kHz 单声道音频块
 *
 * 为什么需要它：
 *   浏览器的 Web Speech API 会把音频送到厂商云服务（Edge→微软、Chrome→谷歌），
 *   既不可控也会泄漏未成年人音频。改用本地 ASR 后，音频必须由我们自己按块取出。
 *
 * 设计：
 *   - 运行在音频渲染线程，不阻塞主线程（`ScriptProcessorNode` 已废弃，不用）
 *   - **不在这里做重采样**：上游用 `new AudioContext({ sampleRate: 16000 })` 创建，
 *     浏览器会把麦克风重采样到 16kHz 再送进来，因此这里拿到的就是 16kHz 单声道
 *   - 只负责「攒够 100ms 就投递一块」，语音端点检测（VAD）与分片策略放在主线程，
 *     便于单独测试与调整，也让这个文件保持无状态
 *
 * 消息协议：
 *   主线程 → worklet : { type: 'flush' }   立即投递残余样本并清空
 *   worklet → 主线程 : { type: 'block', samples: Float32Array, rms: number }
 *                      { type: 'flushed', samples: Float32Array }
 */
const BLOCK_SAMPLES = 1600; // 16000 Hz × 100 ms

class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf = new Float32Array(BLOCK_SAMPLES);
    this._n = 0;
    this.port.onmessage = (e) => {
      if (e.data && e.data.type === 'flush') this._flush();
    };
  }

  _emit(samples) {
    if (!samples.length) return;
    let sum = 0;
    for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
    const rms = Math.sqrt(sum / samples.length);
    // 复制一份再转移，避免与内部缓冲共享内存
    const out = samples.slice();
    this.port.postMessage({ type: 'block', samples: out, rms }, [out.buffer]);
  }

  _flush() {
    if (this._n > 0) {
      const out = this._buf.slice(0, this._n);
      this.port.postMessage({ type: 'flushed', samples: out }, [out.buffer]);
      this._n = 0;
    } else {
      this.port.postMessage({ type: 'flushed', samples: new Float32Array(0) });
    }
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true; // 无输入时保持存活
    const ch = input[0];

    let off = 0;
    while (off < ch.length) {
      const room = BLOCK_SAMPLES - this._n;
      const take = Math.min(room, ch.length - off);
      this._buf.set(ch.subarray(off, off + take), this._n);
      this._n += take;
      off += take;
      if (this._n === BLOCK_SAMPLES) {
        this._emit(this._buf);
        this._n = 0;
      }
    }
    return true;
  }
}

registerProcessor('pcm-capture', PcmCaptureProcessor);
