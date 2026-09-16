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
  const lastFrameBrightnessRef = useRef(0);
  const sessionStartRef = useRef(0);
  const thresholdRef = useRef(0);

  const getEyeRegionBrightness = (ctx: CanvasRenderingContext2D, width: number, height: number): number => {
    // 取面部中心偏上区域（大致是眼睛位置）
    const eyeRegionWidth = Math.floor(width * 0.5);
    const eyeRegionHeight = Math.floor(height * 0.2);
    const startX = Math.floor((width - eyeRegionWidth) / 2);
    const startY = Math.floor(height * 0.25);

    const imageData = ctx.getImageData(startX, startY, eyeRegionWidth, eyeRegionHeight);
    const data = imageData.data;
    let totalBrightness = 0;
    const pixelCount = data.length / 4;

    for (let i = 0; i < data.length; i += 4) {
      // 灰度值 = 0.299R + 0.587G + 0.114B
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

    // 自适应阈值（基于初始几帧的平均值）
    if (thresholdRef.current === 0) {
      thresholdRef.current = brightness * 0.85;
    }

    const now = Date.now();

    // 检测眨眼：亮度突然下降（眼睛闭合时进入更多暗色区域）
    if (brightness < thresholdRef.current * 0.7) {
      if (!isBlinkingRef.current) {
        // 开始眨眼
        isBlinkingRef.current = true;
        blinkStartRef.current = now;
      }
    } else {
      if (isBlinkingRef.current) {
        // 眨眼结束
        const duration = now - blinkStartRef.current;
        // 过滤掉过短的检测（<50ms 可能是噪声）
        if (duration >= 50 && duration < 1000) {
          blinkEventsRef.current.push({ time: now, duration });
        }
        isBlinkingRef.current = false;
      }
    }

    lastFrameBrightnessRef.current = brightness;

    // 每 3 秒更新一次指标
    const elapsed = (now - sessionStartRef.current) / 1000;
    if (elapsed > 3) {
      const blinks = blinkEventsRef.current;
      const blinkRate = Math.round((blinks.length / elapsed) * 60);
      const avgDuration = blinks.length > 0
        ? Math.round(blinks.reduce((s, b) => s + b.duration, 0) / blinks.length)
        : 0;
      const longBlinks = blinks.filter(b => b.duration > 400).length;

      // 焦虑指数计算：
      // 正常眨眼频率 15-20 次/分钟
      // 焦虑时可能增加到 25-30+ 或减少到 <10
      // 长眨眼增多也表示疲劳/焦虑
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
      videoRef.current = video;

      const canvas = document.createElement('canvas');
      canvasRef.current = canvas;

      await new Promise<void>((resolve) => {
        video.onloadedmetadata = () => {
          video.play();
          resolve();
        };
      });

      blinkEventsRef.current = [];
      isBlinkingRef.current = false;
      thresholdRef.current = 0;
      sessionStartRef.current = Date.now();

      setMetrics(prev => ({ ...prev, isDetecting: true, error: undefined }));
      animFrameRef.current = requestAnimationFrame(detectFrame);
    } catch (err: any) {
      setMetrics(prev => ({
        ...prev,
        isDetecting: false,
        error: err.name === 'NotAllowedError' ? '摄像头权限被拒绝' : '无法访问摄像头',
      }));
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
