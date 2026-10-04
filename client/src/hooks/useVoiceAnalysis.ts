/**
 * 声音声学分析 + 语音识别 Hook
 *
 * 理论依据：
 * - Scherer (2003) — 语音情感计算的声学标志物
 * - Cummins et al. (2015) — 语音作为抑郁 biomarker
 * - Jiang et al. (2020) — 青少年语音情感识别
 *
 * 采集维度（使用 Web Audio API）：
 * - 基频特征 (F0/Pitch)：均值、标准差、范围 → 情绪唤醒度
 * - 能量特征 (Energy)：均值、波动、峰值比 → 情绪强度
 * - 语速特征 (Rate)：音节速率、规律性 → 焦虑/抑郁指标
 * - 停顿特征 (Pause)：停顿比、次数、长停顿 → 犹豫/思维迟缓
 * - 频谱特征 (Spectral)：质心、滚降、平坦度 → 情绪色彩
 *
 * 语音识别（本地 sherpa-onnx Paraformer）：
 * - 音频经 AudioWorklet 按 16kHz 分片，送**本机**算法服务识别，全程不出设备
 * - 将识别文本传递给文本分析模块
 * - 刻意不使用浏览器 Web Speech API：那条路径会把音频送到厂商云服务
 *   （Edge→微软 Azure、Chrome→谷歌），既依赖外网也会泄漏未成年人音频
 */
import { useEffect, useRef, useState, useCallback } from 'react';
import type { VoiceAnalysis } from '../types/multimodal.types';
import { defaultVoice } from '../types/multimodal.types';
import type { AgeGroup } from './ageConfig';
import { getVoiceConfig } from './ageConfig';
import { StreamingAsrSession } from '../services/streamingAsr';
import { looksLikeDecoderLoop } from '../services/speechGate';
import { mediaStreamManager, MediaUnavailableError } from '../services/mediaStreamManager';

/** 音频能量回调：RMS 能量值（供呼吸模式分析使用） */
export type AudioEnergyCallback = (rmsEnergy: number) => void;

/**
 * 等最后一段定稿的上限。
 *
 * 通话页（端点模式）服务端冲刷 + 离线复识别纠错正常在 100ms 内完成，
 * `StreamingAsrSession.stop()` 自身最多等 600ms；
 * 聊天页（整段录入）要对**整段**音频跑一次离线大模型，60s 音频约 0.4s，
 * `stop()` 因此最多等 1.5s。这里在它之上再留 1s 余量，超时就不再让调用方继续
 * 挂着（返回空串，不假装拿到了文本）。
 */
const LAST_FINAL_TIMEOUT_MS = 2500;

/** 语音识别结束语义。见 `StreamingAsrSession` 的文件头。 */
export type VoiceAsrMode = 'utterance' | 'streaming';

/**
 * 一段语音定稿回调。
 *
 * 与直接读 `metrics.asrFinalText` 的区别：这个回调在定稿发生的**那一刻**触发，
 * 不依赖调用方的 `voiceInputActive` 是否还开着。聊天页"点结束 → 最后一段文字
 * 没能落进输入框"就是因为定稿比 `isRecording=false` 晚（撤销端点检测 + 离线复识别
 * 约 600ms），只靠订阅 state 会漏掉这一段。
 *
 * @param text 定稿文本（已 trim，非空）
 * @param seq  递增序号，用于调用方区分"新的一段"（两段内容相同时文本不变）
 */
export type VoiceFinalCallback = (text: string, seq: number) => void;

/**
 * 中间结果回调（施工单 §S5.4）。
 *
 * 打断必须靠**中间结果**而不是定稿：中间结果每 ~330ms 就有一次，
 * 而定稿要等端点检测（0.45–1.2 s），手感差一个数量级（G-1）。
 */
export type VoicePartialCallback = (text: string) => void;

export function useVoiceAnalysis(
  ageGroup: AgeGroup | null = null,
  onAudioEnergy?: AudioEnergyCallback | null,
  onVoiceFinal?: VoiceFinalCallback | null,
  onVoicePartial?: VoicePartialCallback | null,
  /**
   * 结束语义。
   * - `'streaming'`（默认）说话停顿即定稿，一句一轮 —— 通话页
   * - `'utterance'` 停顿不定稿，用户点结束才定稿 —— 聊天页语音输入
   *
   * 默认取 `'streaming'` 是为了不改变通话页的既有行为；聊天页显式传 `'utterance'`。
   */
  asrMode: VoiceAsrMode = 'streaming',
) {
  const [metrics, setMetrics] = useState<VoiceAnalysis>({ ...defaultVoice });
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const isActiveRef = useRef(false);
  /**
   * 正在进行的 `start()`，用于**重入保护**。
   * `isActiveRef` 要等初始化走完才置 true，只靠它挡不住"上一次还在 await"的第二次调用；
   * 那种情况下会开出两个 AudioContext + 两个 rAF 循环 + 两个 ASR 会话。
   */
  const startingRef = useRef<Promise<void> | null>(null);
  const startTimeRef = useRef(0);
  const frameDataRef = useRef<Float32Array[]>([]);
  const energyHistoryRef = useRef<number[]>([]);
  const silenceCountRef = useRef(0);
  const pauseCountRef = useRef(0);
  const frameCountRef = useRef(0);
  const rafRef = useRef(0);

  // 本地流式语音识别会话（sherpa-onnx streaming zipformer，经本机 8001 的 WebSocket）
  const asrSessionRef = useRef<StreamingAsrSession | null>(null);
  const speechTextHistoryRef = useRef<string[]>([]);
  /** 定稿序号，由 Hook 自己发号（不依赖 setState 的 prev，回调与 state 才能对齐） */
  const finalSeqRef = useRef(0);
  /** 定稿回调放在 ref 里，避免每秒重渲染时重建 ASR 会话的闭包 */
  const onVoiceFinalRef = useRef(onVoiceFinal);
  onVoiceFinalRef.current = onVoiceFinal;
  /** 中间结果回调同理（打断判据靠它） */
  const onVoicePartialRef = useRef(onVoicePartial);
  onVoicePartialRef.current = onVoicePartial;
  /**
   * 结束语义放 ref：`asrMode` 变了不该重开一路麦克风，而且 `start` 的依赖数组
   * 刻意只有 `analyzeFrame`（它是稳定引用），所以只靠闭包捕获会读到旧值。
   */
  const asrModeRef = useRef(asrMode);
  asrModeRef.current = asrMode;
  /**
   * 等待"最后一段定稿"的 resolve。
   *
   * 为什么需要：定稿要等撤销端点检测（0.8s 静音）+ 离线复识别纠错（约 40ms），
   * 因此 `stop()` 之后还有一段文字在路上。调用方（聊天页）必须等它落地，
   * 否则点结束后输入框里永远少最后一句。
   */
  const pendingFinalRef = useRef<{ resolve: (text: string) => void; timer: number } | null>(null);

  /** 取出并清空"等待最后一段定稿"的悬挂 Promise */
  const settlePendingFinal = (text: string): void => {
    const pending = pendingFinalRef.current;
    if (!pending) return;
    pendingFinalRef.current = null;
    window.clearTimeout(pending.timer);
    pending.resolve(text);
  };

  /**
   * 关掉 16kHz 的 ASR AudioContext —— **幂等、不抛**。
   *
   * 为什么不能直接 `ctx.close()`：
   *   1. `AudioContext.close()` 对**已经关过**的 context 会返回一个**被拒绝的 Promise**
   *      （`InvalidStateError: Cannot close a closed AudioContext`），
   *      原来的写法没有 `catch`，于是它变成一个**未处理的 rejection**
   *      （实测在通话页被闸门抓到两条：`stop()` 与卸载清理各一次）。
   *   2. 关掉之后必须立刻把 ref 置空，否则再进来一次就是重复关。
   */
  const closeAudioContext = useCallback((): void => {
    const ctx = audioContextRef.current;
    audioContextRef.current = null;
    if (!ctx || ctx.state === 'closed') return;
    try {
      void ctx.close().catch(() => undefined);
    } catch {
      // 某些实现会同步抛（已经在关的过程中）：同样不该冒到上层
    }
  }, []);

  // 年龄校准阈值
  const voiceConfig = getVoiceConfig(ageGroup);

  const analyzeFrame = useCallback(() => {
    if (!isActiveRef.current || !analyserRef.current) return;

    const analyser = analyserRef.current;
    const bufferLength = analyser.fftSize;
    const dataArray = new Float32Array(bufferLength);
    analyser.getFloatTimeDomainData(dataArray);
    frameDataRef.current.push(dataArray);

    // 计算 RMS 能量
    let sum = 0;
    for (let i = 0; i < dataArray.length; i++) {
      sum += dataArray[i] * dataArray[i];
    }
    const rms = Math.sqrt(sum / dataArray.length);
    const energyDb = 20 * Math.log10(rms + 1e-10);
    energyHistoryRef.current.push(energyDb);
    frameCountRef.current++;

    // 通知呼吸模式分析 Hook
    if (onAudioEnergy) onAudioEnergy(rms);

    // 检测停顿 (能量 < -50dB = 静音)
    const isSilence = energyDb < -50;
    if (isSilence) {
      silenceCountRef.current++;
      if (silenceCountRef.current === 10) { // 连续10帧静音 = 一次停顿
        pauseCountRef.current++;
      }
    } else {
      silenceCountRef.current = 0;
    }

    // 每2秒计算一次综合指标
    if (frameCountRef.current % 60 === 0 && frameCountRef.current > 0) {
      const energies = energyHistoryRef.current;
      const duration = (Date.now() - startTimeRef.current) / 1000;

      // 能量统计
      const energyMean = energies.reduce((a, b) => a + b, 0) / energies.length;
      const energyStd = Math.sqrt(
        energies.reduce((sum, e) => sum + (e - energyMean) ** 2, 0) / energies.length
      );
      const energyPeak = energies.filter(e => e > energyMean + energyStd * 2).length / energies.length;

      // 基频估计 (使用自相关法)
      const sampleRate = audioContextRef.current?.sampleRate || 44100;
      const midStart = Math.floor(dataArray.length * 0.25);
      const midEnd = Math.floor(dataArray.length * 0.75);
      let bestLag = 0;
      let maxCorr = 0;
      const minLag = Math.floor(sampleRate / 500); // 500Hz
      const maxLag = Math.floor(sampleRate / 65);  // 65Hz

      for (let lag = minLag; lag < Math.min(maxLag, midEnd - midStart); lag++) {
        let corr = 0;
        for (let i = midStart; i < midEnd - lag; i++) {
          corr += dataArray[i] * dataArray[i + lag];
        }
        if (corr > maxCorr) {
          maxCorr = corr;
          bestLag = lag;
        }
      }
      const pitchEstimate = bestLag > 0 ? sampleRate / bestLag : 0;

      // 频谱质心 (使用频域数据)
      const freqData = new Float32Array(analyser.frequencyBinCount);
      analyser.getFloatFrequencyData(freqData);
      let spectralSum = 0;
      let spectralWeight = 0;
      const nyquist = sampleRate / 2;
      for (let i = 0; i < freqData.length; i++) {
        const freq = (i / freqData.length) * nyquist;
        const mag = Math.pow(10, freqData[i] / 20); // 转换回线性
        spectralSum += freq * mag;
        spectralWeight += mag;
      }
      const spectralCentroid = spectralWeight > 0 ? spectralSum / spectralWeight : 0;

      // 频谱滚降点 (85% 能量集中点)
      let cumulativeEnergy = 0;
      const totalEnergy = freqData.reduce((s, v) => s + Math.pow(10, v / 10), 0);
      let rolloffFreq = 0;
      for (let i = 0; i < freqData.length; i++) {
        cumulativeEnergy += Math.pow(10, freqData[i] / 10);
        if (cumulativeEnergy >= totalEnergy * 0.85) {
          rolloffFreq = (i / freqData.length) * nyquist;
          break;
        }
      }

      // 频谱平坦度 (几何均值/算术均值)
      let logSum = 0;
      let linSum = 0;
      let validBins = 0;
      for (let i = 1; i < freqData.length; i++) {
        const mag = Math.pow(10, freqData[i] / 20);
        if (mag > 1e-10) {
          logSum += Math.log(mag);
          linSum += mag;
          validBins++;
        }
      }
      const geometricMean = Math.exp(logSum / Math.max(validBins, 1));
      const arithmeticMean = linSum / Math.max(validBins, 1);
      const spectralFlatness = arithmeticMean > 0 ? geometricMean / arithmeticMean : 0;

      // 语速估计 (基于能量过零率)
      let zeroCrossings = 0;
      for (let i = 1; i < dataArray.length; i++) {
        if ((dataArray[i] >= 0) !== (dataArray[i-1] >= 0)) zeroCrossings++;
      }
      const speechRate = zeroCrossings / (dataArray.length / sampleRate) / 100;

      // 停顿比
      const pauseRatio = silenceCountRef.current > 0 ? silenceCountRef.current / frameCountRef.current : 0;

      // 映射到情绪概率 [快乐, 悲伤, 焦虑, 愤怒, 中性]（使用年龄校准阈值）
      const emotionProbs = [0.2, 0.2, 0.2, 0.2, 0.2];
      // 低基频 + 低能量 → 悲伤（年龄校准）
      if (pitchEstimate < voiceConfig.f0DepressionThreshold && energyMean < -35) { emotionProbs[1] += 0.3; }
      // 高基频 + 高语速 → 焦虑（年龄校准）
      if (pitchEstimate > voiceConfig.f0AnxietyThreshold || speechRate > voiceConfig.speechRateAnxietyThreshold) { emotionProbs[2] += 0.3; }
      // 高能量 + 高波动 → 愤怒
      if (energyMean > -20 && energyStd > 8) { emotionProbs[3] += 0.3; }
      // 高停顿 → 悲伤/焦虑（年龄校准）
      if (pauseRatio > voiceConfig.pauseDepressionThreshold) { emotionProbs[1] += 0.15; emotionProbs[2] += 0.1; }
      // 正常 → 中性
      if (Math.max(...emotionProbs) < 0.3) { emotionProbs[4] += 0.4; }
      const probSum = emotionProbs.reduce((a, b) => a + b, 0);
      const normalizedProbs = emotionProbs.map(p => Math.round((p / probSum) * 100) / 100);

      const dominantIdx = normalizedProbs.indexOf(Math.max(...normalizedProbs));
      const detectedEmotion = ['快乐', '悲伤', '焦虑', '愤怒', '中性'][dominantIdx];
      const riskScore = Math.round((normalizedProbs[1] + normalizedProbs[2] + normalizedProbs[3]) * 100) / 100;

      setMetrics(prev => ({
        ...prev,
        pitch: {
          mean: Math.round(pitchEstimate),
          std: Math.round(energyStd * 10) / 10,
          range: Math.round((pitchEstimate > 0 ? pitchEstimate * 0.3 : 0) * 10) / 10,
          min: Math.round(pitchEstimate * 0.8),
          max: Math.round(pitchEstimate * 1.2),
        },
        energy: {
          mean: Math.round(energyMean * 10) / 10,
          std: Math.round(energyStd * 10) / 10,
          peakRatio: Math.round(energyPeak * 100) / 100,
        },
        rate: {
          speechRate: Math.round(speechRate * 10) / 10,
          tempo: speechRate > 5 ? 'fast' : speechRate < 2.5 ? 'slow' : 'normal',
          regularity: Math.round(Math.max(0, 1 - energyStd / 20) * 100) / 100,
        },
        pauses: {
          ratio: Math.round(pauseRatio * 100) / 100,
          count: pauseCountRef.current,
          avgDuration: Math.round(duration * 100) / 100,
          longPauseCount: Math.max(0, pauseCountRef.current - 5),
        },
        spectral: {
          centroid: Math.round(spectralCentroid),
          rolloff: Math.round(rolloffFreq),
          flatness: Math.round(spectralFlatness * 1000) / 1000,
        },
        emotionProbs: normalizedProbs,
        riskScore,
        detectedEmotion,
        duration: Math.round(duration),
        timestamp: Date.now(),
      }));
    }

    rafRef.current = requestAnimationFrame(analyzeFrame);
  }, []);

  /**
   * 开始采集与分析（麦克风 + 本地流式识别）。
   *
   * **幂等**：已在运行、或上一次 `start()` 还没结束时直接复用 ——
   * 否则会开出第二个 `AudioContext`、第二个 rAF 循环、第二个 ASR 会话。
   * （`isActiveRef` 要等初始化走完才置 true，只靠它挡不住那段 await 期间的第二次调用。）
   *
   * **失败会抛出** `MediaUnavailableError`（带 `reason`），不再静默吞掉：
   * 麦克风没打开却被当成"在录音"，是比报错更糟的状态。
   */
  const start = useCallback(async (): Promise<void> => {
    if (isActiveRef.current) return;
    if (startingRef.current) return startingRef.current;

    // 用一个"占位" promise 在**同步**阶段就把重入窗口关上：
    // 任何并发的第二次调用只会等第一次结束，不会再开一路麦克风。
    let releaseGate!: () => void;
    startingRef.current = new Promise<void>((resolve) => { releaseGate = resolve; });

    try {
      // 走统一的设备管理器：麦克风可能正被通话页共用，引用计数保证不会互相打死
      const stream = await mediaStreamManager.acquire('microphone', 'perception-voice');
      streamRef.current = stream;

      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.8;
      source.connect(analyser);
      analyserRef.current = analyser;

      // 重置计数器
      energyHistoryRef.current = [];
      silenceCountRef.current = 0;
      pauseCountRef.current = 0;
      frameCountRef.current = 0;
      startTimeRef.current = Date.now();
      isActiveRef.current = true;

      // 上一次会话若还挂着"等最后一段定稿"，先结清，避免它一直悬着
      settlePendingFinal('');

      setMetrics(prev => ({ ...prev, isRecording: true }));
      rafRef.current = requestAnimationFrame(analyzeFrame);

      // ── 本地流式语音转文字（sherpa-onnx streaming zipformer）────────────
      //
      // 这里替换过两次，记录一下为什么：
      //   1. 最早是 `webkitSpeechRecognition` —— 它并非本地能力，Edge 走微软
      //      Azure、Chrome 走谷歌云（微软策略文档原文："The Microsoft Edge
      //      implementation of the Web Speech API uses Azure Cognitive Services,
      //      so voice data leaves the machine"），既把未成年人音频送出设备、
      //      又依赖外网；旧实现的 `onend` 零延迟重启还会把一次瞬时失败放大成
      //      永久报错风暴。
      //   2. 第二版是"VAD 攒完整句再整段识别"（LocalAsrSession）—— 音频不出
      //      设备了，但必须等一句话说完才出文本，不像手机语音输入。
      //   3. 现在这版：AudioContext(16kHz) → AudioWorklet(100ms/块) → WebSocket
      //      → 本机 streaming zipformer → 中间结果边说边上屏 + 端点定稿。
      //
      // 中间结果与定稿分开：中间结果只上屏；只有定稿才写进 `speechText`，
      // 因此下游的文本情感 / 认知扭曲 / 语音语义分析不会被半句话反复触发。
      asrSessionRef.current?.stop();
      const asrSession = new StreamingAsrSession({
        onPartial: (text, meta) => {
          // 打断判据要的是"刚听到什么"，所以原样透出（含空串清屏）
          onVoicePartialRef.current?.(text);
          setMetrics(prev => ({
            ...prev,
            asrPartialText: text,
            isSpeechRecognizing: true,
            asrEngine: meta.engine || prev.asrEngine,
            asrError: '',
            asrLatencyMs: meta.latencyMs,
          }));
        },
        onFinal: (text, meta) => {
          const finalText = text.trim();
          // 兜底：解码器打转产生的假文本（"好的的的的。我的我的没的。"）直接丢弃，
          // 不写进 speechText / asrFinalText，否则聊天页会把它自动追加进输入框。
          // 主要防线在采集侧：`StreamingAsrSession` 掐掉开头静音，不让模型拿到
          // "没有真实语音证据"的音频。
          if (finalText && looksLikeDecoderLoop(finalText)) {
            console.warn('[语音识别] 丢弃疑似解码打转的假文本:', finalText);
            setMetrics(prev => ({ ...prev, asrPartialText: '', isSpeechRecognizing: true }));
            return;
          }
          if (finalText) {
            // 序号在这里自增：既写进 metrics，也通过回调带出去。
            // 回调走的是"定稿那一刻"，因此点结束之后再到达的最后一段文字不会丢。
            const seq = ++finalSeqRef.current;
            onVoiceFinalRef.current?.(finalText, seq);
            settlePendingFinal(finalText);
            setMetrics(prev => ({
              ...prev,
              asrPartialText: '',
              isSpeechRecognizing: true,
              asrEngine: meta.engine || prev.asrEngine,
              asrError: '',
              asrLatencyMs: meta.latencyMs,
              speechText: finalText,
              speechTextHistory: [finalText, ...prev.speechTextHistory].slice(0, 5),
              asrFinalText: finalText,
              asrFinalSeq: seq,
              // 定稿是否经离线大模型纠错（标点是定稿无条件尝试的）
              asrRefined: meta.refined,
            }));
            return;
          }
          // 定稿但文本为空（例如刚开口就结束）：只清掉临时上屏文本
          setMetrics(prev => ({
            ...prev,
            asrPartialText: '',
            isSpeechRecognizing: true,
            asrEngine: meta.engine || prev.asrEngine,
            asrError: '',
            asrLatencyMs: meta.latencyMs,
          }));
        },
        // 如实上报：引擎不可用时必须能看出来，不允许静默变成空文本
        onStatus: (st) => {
          setMetrics(prev => ({
            ...prev,
            asrEngine: st.available ? st.engine || 'sherpa-onnx' : 'unavailable',
            asrError: st.available ? '' : st.reason || '本地流式识别不可用',
            isSpeechRecognizing: st.available,
          }));
        },
      }, asrModeRef.current);
      asrSessionRef.current = asrSession;
      try {
        await asrSession.start(stream);
        console.log(
          `[语音识别] 本地 ASR 已启动（mode=${asrModeRef.current}，` +
            (asrModeRef.current === 'utterance'
              ? '停顿不定稿，用户点结束才定稿'
              : '停顿即定稿') +
            '）',
        );
      } catch (err) {
        console.warn('[语音识别] 本地流式 ASR 启动失败:', err);
        setMetrics(prev => ({
          ...prev,
          asrEngine: 'unavailable',
          asrError: `本地流式 ASR 启动失败：${(err as Error)?.message || err}`,
          isSpeechRecognizing: false,
        }));
      }
    } catch (err) {
      // 失败路径要自己收干净：半开的流 + 半初始化的音频图比"没开"更糟
      isActiveRef.current = false;
      cancelAnimationFrame(rafRef.current);
      mediaStreamManager.release('microphone', 'perception-voice');
      streamRef.current = null;
      analyserRef.current = null;
      const ctx = audioContextRef.current;
      audioContextRef.current = null;
      if (ctx) {
        try {
          await ctx.close();
        } catch {
          // 已经关掉了就算了
        }
      }

      const reason = err instanceof MediaUnavailableError
        ? err.message
        : `麦克风启动失败（${(err as Error)?.message || err}）`;
      console.warn('[语音分析] 启动失败:', reason);
      throw err instanceof MediaUnavailableError
        ? err
        : new MediaUnavailableError('microphone', 'unknown', reason);
    } finally {
      releaseGate();
      startingRef.current = null;
    }
  }, [analyzeFrame]);

  /**
   * 停止采集。
   *
   * @returns 最后一段定稿文本（若在超时内没有新定稿则为空串）。
   *   **必须 await 它再把"等待"状态收掉**，否则最后一句会落在界面之后。
   */
  const stop = useCallback(async (): Promise<string> => {
    isActiveRef.current = false;
    cancelAnimationFrame(rafRef.current);
    // 交还给设备管理器，**不要**直接 stop 轨道：通话页可能正共用同一路麦克风。
    mediaStreamManager.release('microphone', 'perception-voice');
    closeAudioContext();
    audioContextRef.current = null;
    analyserRef.current = null;
    streamRef.current = null;

    // 先挂上等待钩子再停会话：`streamingAsr.stop()` 会先让服务端冲刷残余并定稿，
    // 那段定稿可能就发生在这几毫秒里，顺序反了会漏掉它。
    const pending = new Promise<string>((resolve) => {
      const timer = window.setTimeout(() => {
        pendingFinalRef.current = null;
        resolve('');
      }, LAST_FINAL_TIMEOUT_MS);
      pendingFinalRef.current = { resolve, timer };
    });

    // 停止本地语音转文字（会先 flush 残余音频再拆音频图）
    if (asrSessionRef.current) {
      void asrSessionRef.current.stop();
      asrSessionRef.current = null;
    }

    setMetrics(prev => ({ ...prev, isRecording: false, isSpeechRecognizing: false }));
    return pending;
  }, []);

  const reset = useCallback(() => {
    void stop();
    setMetrics({ ...defaultVoice });
  }, [stop]);

  /**
   * 暂停 / 恢复"把音频帧送给服务端"（施工单 §S5.3 的半双工窗口用）。
   *
   * 只是把调用转给当前的 ASR 会话；**没有会话时是空操作**（不是错误）——
   * 通话页在 TTS 出声时才开窗，而那时会话一定在，但重连/未启动期间也可能是空的。
   *
   * ⚠️ 不要用 `track.enabled = false` 代替它：那会送出一段真静音，
   * 服务端端点检测照样触发，反而定稿出一个空段。详见
   * `StreamingAsrSession.setForwarding` 的注释。
   */
  const setForwarding = useCallback((on: boolean): void => {
    asrSessionRef.current?.setForwarding(on);
  }, []);

  /** 当前是否在向服务端送帧（`?debug=1` 面板与 S5 闸门读它） */
  const isForwarding = useCallback((): boolean => {
    return asrSessionRef.current?.isForwarding ?? false;
  }, []);

  /** 麦克风轨道**实际生效**的约束（不假设 AEC 开了 —— 见 S5.3） */
  const getMicSettings = useCallback((): MediaTrackSettings | null => {
    const track = streamRef.current?.getAudioTracks()[0];
    if (!track) return null;
    try {
      return track.getSettings();
    } catch {
      return null;
    }
  }, []);

  useEffect(() => {
    return () => {
      isActiveRef.current = false;
      cancelAnimationFrame(rafRef.current);
      mediaStreamManager.releaseAll('perception-voice');
      closeAudioContext();
      // 卸载时也要拆掉 ASR 会话，否则 16kHz AudioContext 与 worklet 会泄漏
      if (asrSessionRef.current) {
        void asrSessionRef.current.stop();
        asrSessionRef.current = null;
      }
      // 组件没了就别让"等最后一段定稿"的 Promise 一直悬着
      settlePendingFinal('');
    };
  }, [closeAudioContext]);

  return {
    metrics,
    start,
    stop,
    reset,
    setForwarding,
    isForwarding,
    getMicSettings,
    audioContextRef,
    analyserRef,
  };
}
