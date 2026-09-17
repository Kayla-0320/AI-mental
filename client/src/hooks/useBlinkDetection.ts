import { useEffect, useRef, useState, useCallback } from 'react';

export interface BlinkMetrics {
  blinkRate: number;         // 每分钟眨眼次数
  avgBlinkDuration: number;  // 平均眨眼时长 ms
  longBlinkCount: number;    // 长眨眼次数（>400ms，可能表示疲劳/焦虑）
  anxietyIndex: number;      // 基于眨眼模式的焦虑指数 0-100
  isDetecting: boolean;
  error?: string;
}

const defaultBlinkMetrics: BlinkMetrics = {
  blinkRate: 0,
  avgBlinkDuration: 0,
  longBlinkCount: 0,
  anxietyIndex: 0,
  isDetecting: false,
};

export function useBlinkDetection() {
  const [metrics, setMetrics] = useState<BlinkMetrics>(defaultBlinkMetrics);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animFrameRef = useRef<number>(0);
  const streamRef = useRef<MediaStream | null>(null);

  const blinkEventsRef = useRef<{ time: number; duration: number }[]>([]);
  const isBlinkingRef = useRef(false);
  const blinkStartRef = useRef(0);
  const sessionStartRef = useRef(0);
  const frameCountRef = useRef(0);

  // 滚动亮度历史（用于自适应阈值）
  const brightnessHistoryRef = useRef<number[]>([]);
  const MAX_HISTORY = 30; // 保留最近30帧的亮度值

  const getEyeRegionBrightness = (ctx: CanvasRenderingContext2D, width: number, height: number): number => {
    // 取面部中心偏上区域（大致是眼睛位置），缩小采样区域减少噪声
    const eyeRegionWidth = Math.floor(width * 0.35);
    const eyeRegionHeight = Math.floor(height * 0.12);
    const startX = Math.floor((width - eyeRegionWidth) / 2);
    const startY = Math.floor(height * 0.2);

    const imageData = ctx.getImageData(startX, startY, eyeRegionWidth, eyeRegionHeight);
    const data = imageData.data;
    let totalBrightness = 0;
    const pixelCount = data.length / 4;

    for (let i = 0; i < data.length; i += 4) {
      totalBrightness += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    }

    return totalBrightness / pixelCount;
  };

  const detectFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState !== 4) {
      animFrameRef.current = requestAnimationFrame(detectFrame);
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    const brightness = getEyeRegionBrightness(ctx, canvas.width, canvas.height);
    frameCountRef.current++;

    // 更新滚动亮度历史
    const history = brightnessHistoryRef.current;
    history.push(brightness);
    if (history.length > MAX_HISTORY) history.shift();

    const now = Date.now();

    // 前10帧用于建立基线，不检测眨眼
    if (frameCountRef.current < 10) {
      animFrameRef.current = requestAnimationFrame(detectFrame);
      return;
    }

    // 自适应阈值：使用滚动平均值和标准差
    const avg = history.reduce((s, v) => s + v, 0) / history.length;
    const variance = history.reduce((s, v) => s + (v - avg) ** 2, 0) / history.length;
    const stdDev = Math.sqrt(variance);

    // 眨眼判定：亮度低于平均值超过1.5个标准差
    const blinkThreshold = avg - Math.max(stdDev * 1.5, 5);

    if (brightness < blinkThreshold) {
      if (!isBlinkingRef.current) {
        isBlinkingRef.current = true;
        blinkStartRef.current = now;
      }
    } else {
      if (isBlinkingRef.current) {
        const duration = now - blinkStartRef.current;
        // 有效眨眼：80ms ~ 600ms
        if (duration >= 80 && duration < 600) {
          blinkEventsRef.current.push({ time: now, duration });
        }
        isBlinkingRef.current = false;
      }
    }

    // 每 3 秒更新一次指标
    const elapsed = (now - sessionStartRef.current) / 1000;
    if (elapsed > 3) {
      const blinks = blinkEventsRef.current;
      // 只统计最近60秒内的眨眼
      const recentBlinks = blinks.filter(b => now - b.time < 60000);
      const blinkRate = Math.round((recentBlinks.length / Math.max(elapsed, 1)) * 60);
      const avgDuration = recentBlinks.length > 0
        ? Math.round(recentBlinks.reduce((s, b) => s + b.duration, 0) / recentBlinks.length)
        : 0;
      const longBlinks = recentBlinks.filter(b => b.duration > 400).length;

      // 焦虑指数计算
      let anxietyIndex = 0;
      if (blinkRate > 25) anxietyIndex += Math.min(40, (blinkRate - 25) * 4);
      else if (blinkRate < 10 && blinkRate > 0) anxietyIndex += Math.min(30, (10 - blinkRate) * 3);
      if (longBlinks > 2) anxietyIndex += Math.min(30, longBlinks * 10);
      if (avgDuration > 350) anxietyIndex += 15;
      anxietyIndex = Math.min(100, anxietyIndex);

      setMetrics({
        blinkRate,
        avgBlinkDuration: avgDuration,
        longBlinkCount: longBlinks,
        anxietyIndex,
        isDetecting: true,
      });
    }

    animFrameRef.current = requestAnimationFrame(detectFrame);
  }, []);

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 320 }, height: { ideal: 240 } },
      });

      streamRef.current = stream;

      const video = document.createElement('video');
      video.srcObject = stream;
      video.autoplay = true;
      video.playsInline = true;
      // 镜像翻转，让用户看到正常的自己
      video.style.transform = 'scaleX(-1)';
      videoRef.current = video;

      const canvas = document.createElement('canvas');
      canvasRef.current = canvas;

      await new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error('摄像头超时')), 5000);
        video.onloadedmetadata = () => {
          clearTimeout(timeout);
          video.play().then(resolve).catch(reject);
        };
        video.onerror = () => {
          clearTimeout(timeout);
          reject(new Error('视频加载失败'));
        };
      });

      // 重置状态
      blinkEventsRef.current = [];
      isBlinkingRef.current = false;
      brightnessHistoryRef.current = [];
      frameCountRef.current = 0;
      sessionStartRef.current = Date.now();

      setMetrics(prev => ({ ...prev, isDetecting: true, error: undefined }));
      animFrameRef.current = requestAnimationFrame(detectFrame);
    } catch (err: any) {
      setMetrics(prev => ({
        ...prev,
        isDetecting: false,
        error: err.name === 'NotAllowedError' ? '摄像头权限被拒绝' : err.message || '无法访问摄像头',
      }));
      throw err;
    }
  }, [detectFrame]);

  const stop = useCallback(() => {
    cancelAnimationFrame(animFrameRef.current);
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    videoRef.current = null;
    setMetrics(prev => ({ ...prev, isDetecting: false }));
  }, []);

  useEffect(() => {
    return () => {
      cancelAnimationFrame(animFrameRef.current);
      streamRef.current?.getTracks().forEach(t => t.stop());
    };
  }, []);

  return { metrics, start, stop };
}
