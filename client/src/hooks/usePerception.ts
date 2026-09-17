import { useState, useRef, useCallback } from 'react';

const PERCEPTION_API = 'http://localhost:8001';

export interface PerceptionResult {
  text_emotion_probs: number[];   // [快乐, 悲伤, 焦虑, 愤怒, 中性]
  audio_risk_prob: number | null;
  confidence: number;
  timestamp: number;
  evidence: string[];
}

export interface VoiceMetrics {
  isRecording: boolean;
  duration: number;          // 录音时长（秒）
  volume: number;            // 实时音量 0-100
  audioBlob: Blob | null;    // 录音文件
}

const emotionLabels = ['快乐', '悲伤', '焦虑', '愤怒', '中性'];

export function usePerception() {
  const [result, setResult] = useState<PerceptionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 语音录制相关
  const [voiceMetrics, setVoiceMetrics] = useState<VoiceMetrics>({
    isRecording: false,
    duration: 0,
    volume: 0,
    audioBlob: null,
  });
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const timerRef = useRef<number>(0);
  const volumeTimerRef = useRef<number>(0);

  // ===== 文本情感分析 =====
  const analyzeText = useCallback(async (text: string, behaviorFeatures?: Record<string, unknown>) => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const body: Record<string, unknown> = { text };
      if (behaviorFeatures) body.behavior_features = behaviorFeatures;
      const res = await fetch(`${PERCEPTION_API}/api/v1/perception/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: PerceptionResult = await res.json();
      setResult(data);
    } catch (err: any) {
      console.warn('文本情感分析失败:', err);
      setError('感知服务未连接，请确保算法服务已启动');
    } finally {
      setLoading(false);
    }
  }, []);

  // ===== 语音录制 =====
  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioContext = new AudioContext();
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      audioContextRef.current = audioContext;
      analyserRef.current = analyser;

      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      mediaRecorder.start();
      const startTime = Date.now();

      // 录音时长计时
      timerRef.current = window.setInterval(() => {
        setVoiceMetrics(prev => ({
          ...prev,
          isRecording: true,
          duration: Math.floor((Date.now() - startTime) / 1000),
        }));
      }, 1000);

      // 实时音量监测
      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      volumeTimerRef.current = window.setInterval(() => {
        analyser.getByteFrequencyData(dataArray);
        const avg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
        setVoiceMetrics(prev => ({ ...prev, volume: Math.min(100, Math.round(avg * 2)) }));
      }, 100);

      setVoiceMetrics(prev => ({ ...prev, isRecording: true, duration: 0 }));
    } catch (err: any) {
      console.error('麦克风访问失败:', err);
      setError('无法访问麦克风，请检查浏览器权限设置');
    }
  }, []);

  const stopRecording = useCallback((): Promise<Blob | null> => {
    return new Promise((resolve) => {
      const mediaRecorder = mediaRecorderRef.current;
      if (!mediaRecorder || mediaRecorder.state === 'inactive') {
        resolve(null);
        return;
      }

      mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        setVoiceMetrics(prev => ({ ...prev, isRecording: false, audioBlob: blob }));

        // 清理
        clearInterval(timerRef.current);
        clearInterval(volumeTimerRef.current);
        audioContextRef.current?.close();
        mediaRecorder.stream.getTracks().forEach(t => t.stop());

        resolve(blob);
      };

      mediaRecorder.stop();
    });
  }, []);

  // ===== 语音情感分析（发送录音到后端） =====
  const analyzeVoice = useCallback(async (text: string, behaviorFeatures?: Record<string, unknown>) => {
    const blob = await stopRecording();
    if (!blob) return;

    setLoading(true);
    setError(null);
    try {
      // 将录音转为 WAV 格式上传（后端需要 wav_path 或 base64）
      // 由于后端 API 接受 wav_path，这里先发送文本+行为特征
      // 语音文件通过 FormData 上传到独立端点
      const formData = new FormData();
      formData.append('audio', blob, 'recording.webm');
      formData.append('text', text);
      if (behaviorFeatures) {
        formData.append('behavior_features', JSON.stringify(behaviorFeatures));
      }

      // 尝试上传音频并分析
      const uploadRes = await fetch(`${PERCEPTION_API}/api/v1/perception/analyze-audio`, {
        method: 'POST',
        body: formData,
      });

      if (uploadRes.ok) {
        const data: PerceptionResult = await uploadRes.json();
        setResult(data);
      } else {
        // 降级：仅分析文本+行为特征
        await analyzeText(text, behaviorFeatures);
      }
    } catch {
      // 降级为纯文本分析
      await analyzeText(text, behaviorFeatures);
    } finally {
      setLoading(false);
    }
  }, [stopRecording, analyzeText]);

  // ===== 综合多模态分析（文本+语音+行为融合） =====
  const analyzeMultimodal = useCallback(async (
    text: string,
    options?: {
      includeVoice?: boolean;
      facialFeatures?: Record<string, unknown>;
      behaviorFeatures?: Record<string, unknown>;
    }
  ) => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const body: Record<string, unknown> = { text };
      if (options?.facialFeatures) body.facial_features = options.facialFeatures;
      if (options?.behaviorFeatures) body.behavior_features = options.behaviorFeatures;

      const res = await fetch(`${PERCEPTION_API}/api/v1/perception/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: PerceptionResult = await res.json();
      setResult(data);
    } catch (err: any) {
      console.warn('多模态分析失败:', err);
      setError('感知服务未连接');
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    result,
    loading,
    error,
    voiceMetrics,
    emotionLabels,
    analyzeText,
    analyzeMultimodal,
    startRecording,
    stopRecording,
    analyzeVoice,
  };
}
