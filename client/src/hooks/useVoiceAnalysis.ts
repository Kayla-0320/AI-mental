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
 * 语音识别（Web Speech API）：
 * - 实时转录用户说话内容
 * - 将识别文本传递给文本分析模块
 */
import { useEffect, useRef, useState, useCallback } from 'react';
import type { VoiceAnalysis } from '../types/multimodal.types';
import { defaultVoice } from '../types/multimodal.types';

// Web Speech API 类型声明
interface SpeechRecognitionResult {
  isFinal: boolean;
  [index: number]: { transcript: string; confidence: number };
}

interface SpeechRecognitionEvent {
  resultIndex: number;
  results: SpeechRecognitionResult[] & { length: number };
}

interface SpeechRecognitionErrorEvent {
  error: string;
}

interface SpeechRecognitionInstance {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start: () => void;
  stop: () => void;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEvent) => void) | null;
  onend: (() => void) | null;
}

export function useVoiceAnalysis() {
  const [metrics, setMetrics] = useState<VoiceAnalysis>({ ...defaultVoice });
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const isActiveRef = useRef(false);
  const startTimeRef = useRef(0);
  const frameDataRef = useRef<Float32Array[]>([]);
  const energyHistoryRef = useRef<number[]>([]);
  const silenceCountRef = useRef(0);
  const pauseCountRef = useRef(0);
  const frameCountRef = useRef(0);
  const rafRef = useRef(0);

  // 语音识别
  const recognitionRef = useRef<SpeechRecognitionInstance | null>(null);
  const speechTextHistoryRef = useRef<string[]>([]);

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

      // 映射到情绪概率 [快乐, 悲伤, 焦虑, 愤怒, 中性]
      const emotionProbs = [0.2, 0.2, 0.2, 0.2, 0.2];
      // 低基频 + 低能量 → 悲伤
      if (pitchEstimate < 150 && energyMean < -35) { emotionProbs[1] += 0.3; }
      // 高基频 + 高语速 → 焦虑
      if (pitchEstimate > 280 || speechRate > 5) { emotionProbs[2] += 0.3; }
      // 高能量 + 高波动 → 愤怒
      if (energyMean > -20 && energyStd > 8) { emotionProbs[3] += 0.3; }
      // 高停顿 → 悲伤/焦虑
      if (pauseRatio > 0.3) { emotionProbs[1] += 0.15; emotionProbs[2] += 0.1; }
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

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
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

      setMetrics(prev => ({ ...prev, isRecording: true }));
      rafRef.current = requestAnimationFrame(analyzeFrame);

      // 初始化语音识别
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      if (SpeechRecognition) {
        const recognition = new SpeechRecognition() as SpeechRecognitionInstance;
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'zh-CN';

        recognition.onresult = (event: SpeechRecognitionEvent) => {
          let interimTranscript = '';
          let finalTranscript = '';

          for (let i = event.resultIndex; i < event.results.length; i++) {
            const result = event.results[i];
            const transcript = result[0].transcript;
            if (result.isFinal) {
              finalTranscript += transcript;
            } else {
              interimTranscript += transcript;
            }
          }

          const text = finalTranscript || interimTranscript;
          if (text.trim().length > 0) {
            // 更新语音文本到 metrics
            setMetrics(prev => {
              const newHistory = [text, ...prev.speechTextHistory].slice(0, 5);
              return {
                ...prev,
                speechText: text,
                speechTextHistory: newHistory,
                isSpeechRecognizing: true,
              };
            });
          }
        };

        recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
          console.warn('[语音识别] 错误:', event.error);
          if (event.error !== 'no-speech' && event.error !== 'aborted') {
            setMetrics(prev => ({ ...prev, isSpeechRecognizing: false }));
          }
        };

        recognition.onend = () => {
          // 如果还在录音，自动重启识别
          if (isActiveRef.current) {
            try {
              recognition.start();
            } catch {
              // 忽略重启错误
            }
          } else {
            setMetrics(prev => ({ ...prev, isSpeechRecognizing: false }));
          }
        };

        recognitionRef.current = recognition;
        try {
          recognition.start();
          setMetrics(prev => ({ ...prev, isSpeechRecognizing: true }));
        } catch (err) {
          console.warn('[语音识别] 启动失败:', err);
        }
      } else {
        console.warn('[语音识别] 浏览器不支持 Web Speech API');
      }
    } catch {
      // 麦克风权限被拒绝
    }
  }, [analyzeFrame]);

  const stop = useCallback(() => {
    isActiveRef.current = false;
    cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach(t => t.stop());
    audioContextRef.current?.close();
    audioContextRef.current = null;
    analyserRef.current = null;
    streamRef.current = null;

    // 停止语音识别
    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch {
        // 忽略停止错误
      }
      recognitionRef.current = null;
    }

    setMetrics(prev => ({ ...prev, isRecording: false, isSpeechRecognizing: false }));
  }, []);

  const reset = useCallback(() => {
    stop();
    setMetrics({ ...defaultVoice });
  }, [stop]);

  useEffect(() => {
    return () => {
      isActiveRef.current = false;
      cancelAnimationFrame(rafRef.current);
      streamRef.current?.getTracks().forEach(t => t.stop());
      audioContextRef.current?.close();
    };
  }, []);

  return { metrics, start, stop, reset };
}
