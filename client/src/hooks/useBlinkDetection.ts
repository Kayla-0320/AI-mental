/**
 * 眨眼检测 Hook —— MediaPipe FaceLandmarker blendshape 版
 *
 * 背景（为什么要重写）：
 *   原实现用「画面固定矩形亮度」推断眨眼：取 width*0.35 × height*0.12 的矩形区域
 *   （起点 y = height*0.2），算灰度均值，低于「滚动均值 − max(1.5σ, 5)」即判为闭眼。
 *   它不做人脸检测，矩形位置写死，因此：
 *     - 头一动、镜头一偏、光照一变就误判；
 *     - 戴眼镜反光、房间明暗变化会被当成眨眼；
 *     - 无法区分「眨眼」与「低头 / 离开画面」。
 *   而项目里 useFacialAnalysis 已经加载了 MediaPipe FaceLandmarker 且显式开启了
 *   outputFaceBlendshapes，其中 eyeBlinkLeft / eyeBlinkRight 就是 MediaPipe 直接给出的
 *   眼睑闭合度（0~1），此前从未被读取过。
 *
 * 本版实现：
 *   优先使用 blendshape（eyeBlinkLeft/eyeBlinkRight 取双眼较小值，保证单眼眨动/眯眼
 *   不被计为眨眼）；MediaPipe 不可用时**显式降级**回原亮度法，并通过 metrics.method
 *   如实上报本次实际使用的检测方式，避免降级结果被当成模型输出使用。
 *
 * 判定规则见 blinkLogic.ts（纯逻辑，可独立测试）。
 */
import { useEffect, useRef, useState, useCallback } from 'react';
import {
  BLENDSHAPE_PARAMS,
  BlinkStateMachine,
  brightnessThreshold,
  type BlinkMethod,
} from './blinkLogic';

export type { BlinkMethod } from './blinkLogic';

export interface BlinkMetrics {
  blinkRate: number;         // 每分钟眨眼次数
  avgBlinkDuration: number;  // 平均眨眼时长 ms
  longBlinkCount: number;    // 长眨眼次数（>400ms，可能表示疲劳/焦虑）
  anxietyIndex: number;      // 基于眨眼模式的焦虑指数 0-100
  isDetecting: boolean;
  method: BlinkMethod;       // 检测方式：blendshape=模型眼睑闭合度，brightness=亮度降级
  error?: string;
}

const defaultBlinkMetrics: BlinkMetrics = {
  blinkRate: 0,
  avgBlinkDuration: 0,
  longBlinkCount: 0,
  anxietyIndex: 0,
  isDetecting: false,
  method: 'none',
};

const MAX_HISTORY = 30;    // 亮度滚动窗口
const WARMUP_MS = 500;     // 模型热启动期，期间不计数
const METRIC_INTERVAL_MS = 3000;

export function useBlinkDetection() {
  const [metrics, setMetrics] = useState<BlinkMetrics>(defaultBlinkMetrics);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animFrameRef = useRef<number>(0);
  const streamRef = useRef<MediaStream | null>(null);

  // MediaPipe FaceLandmarker（懒加载，失败则保持 null 走亮度降级）
  const faceLandmarkerRef = useRef<any>(null);
  const machineRef = useRef<BlinkStateMachine>(new BlinkStateMachine(BLENDSHAPE_PARAMS));

  const sessionStartRef = useRef(0);
  const frameCountRef = useRef(0);
  const lastPublishRef = useRef(0);
  const brightnessHistoryRef = useRef<number[]>([]);

  /** 初始化 MediaPipe FaceLandmarker（与 useFacialAnalysis 使用同一模型与 CDN） */
  const initFaceLandmarker = useCallback(async (): Promise<boolean> => {
    try {
      const vision = await import('@mediapipe/tasks-vision');
      const { FaceLandmarker, FilesetResolver } = vision;
      const filesetResolver = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm',
      );
      faceLandmarkerRef.current = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numFaces: 1,
        outputFaceBlendshapes: true,
      });
      console.log('[眨眼检测] ✅ 使用 MediaPipe blendshape（eyeBlinkLeft/Right）');
      return true;
    } catch (err) {
      console.warn('[眨眼检测] ⚠️ MediaPipe 不可用，降级为亮度法:', err);
      faceLandmarkerRef.current = null;
      return false;
    }
  }, []);

  const getEyeRegionBrightness = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
  ): number => {
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

  /** 每 3 秒发布一次滚动指标 */
  const publishMetrics = useCallback((now: number, method: BlinkMethod) => {
    if (now - lastPublishRef.current < METRIC_INTERVAL_MS) return;
    lastPublishRef.current = now;

    const s = machineRef.current.stats(now, sessionStartRef.current);
    setMetrics({
      blinkRate: s.blinkRate,
      avgBlinkDuration: s.avgBlinkDuration,
      longBlinkCount: s.longBlinkCount,
      anxietyIndex: s.anxietyIndex,
      isDetecting: true,
      method,
    });
  }, []);

  const detectFrame = useCallback(() => {
    const video = videoRef.current;
    if (!video || video.readyState < 2) {
      animFrameRef.current = requestAnimationFrame(detectFrame);
      return;
    }

    const now = Date.now();
    const landmarker = faceLandmarkerRef.current;

    if (landmarker) {
      // ── 主路径：MediaPipe blendshape ──
      try {
        const result = landmarker.detectForVideo(video, performance.now());
        if (result?.faceBlendshapes?.length) {
          const cats = result.faceBlendshapes[0].categories;
          let left = 0;
          let right = 0;
          for (const c of cats) {
            if (c.categoryName === 'eyeBlinkLeft') left = c.score;
            else if (c.categoryName === 'eyeBlinkRight') right = c.score;
          }
          // 取双眼较小值：单眼眨动/眯眼不算眨眼
          const blinkScore = Math.min(left, right);

          if (now - sessionStartRef.current > WARMUP_MS) {
            const closed = machineRef.current.isClosed(blinkScore);
            machineRef.current.update(closed, now);
          }
          publishMetrics(now, 'blendshape');
        } else {
          // 画面中没有人脸：结束当前闭眼区间，避免「离开画面」被计成一次长闭眼
          machineRef.current.breakClosed(now);
        }
      } catch (err) {
        console.warn('[眨眼检测] blendshape 推理失败，本次跳过:', err);
      }
      animFrameRef.current = requestAnimationFrame(detectFrame);
      return;
    }

    // ── 降级路径：原有亮度法（无人脸检测，仅作兜底）──
    const canvas = canvasRef.current;
    if (!canvas) {
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

    const history = brightnessHistoryRef.current;
    history.push(brightness);
    if (history.length > MAX_HISTORY) history.shift();

    if (frameCountRef.current >= 10) {
      const threshold = brightnessThreshold(history);
      machineRef.current.update(brightness < threshold, now);
    }
    publishMetrics(now, 'brightness');

    animFrameRef.current = requestAnimationFrame(detectFrame);
  }, [publishMetrics]);

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
      video.muted = true; // 避免部分浏览器因未静音而阻塞自动播放
      video.style.transform = 'scaleX(-1)';
      videoRef.current = video;

      canvasRef.current = document.createElement('canvas');

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
      machineRef.current.reset();
      brightnessHistoryRef.current = [];
      frameCountRef.current = 0;
      sessionStartRef.current = Date.now();
      lastPublishRef.current = 0;

      const usingBlendshape = await initFaceLandmarker();

      setMetrics((prev) => ({
        ...prev,
        isDetecting: true,
        error: undefined,
        method: usingBlendshape ? 'blendshape' : 'brightness',
      }));
      animFrameRef.current = requestAnimationFrame(detectFrame);
    } catch (err: any) {
      setMetrics((prev) => ({
        ...prev,
        isDetecting: false,
        method: 'none',
        error:
          err.name === 'NotAllowedError' ? '摄像头权限被拒绝' : err.message || '无法访问摄像头',
      }));
      throw err;
    }
  }, [detectFrame, initFaceLandmarker]);

  const stop = useCallback(() => {
    cancelAnimationFrame(animFrameRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    videoRef.current = null;
    setMetrics((prev) => ({ ...prev, isDetecting: false }));
  }, []);

  useEffect(() => {
    return () => {
      cancelAnimationFrame(animFrameRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
      try {
        faceLandmarkerRef.current?.close?.();
      } catch {
        /* 忽略释放失败 */
      }
      faceLandmarkerRef.current = null;
    };
  }, []);

  return { metrics, start, stop };
}
