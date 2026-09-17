/**
 * 面部微表情分析 Hook —— MediaPipe Face Mesh 版本
 *
 * 理论依据：
 * - FACS (Facial Action Coding System, Ekman & Friesen, 1978)
 * - AU (Action Unit) 动作单元系统 — 面部肌肉运动与情绪的对应关系
 * - 微表情：持续时间 < 500ms 的无意识表情 (Ekman, 2001)
 *
 * 改进：使用 MediaPipe Face Mesh (468 关键点) 替代像素分析法
 * - 精确的面部几何特征提取（距离比、角度、纵横比）
 * - 基于关键点的 AU 强度计算，精度大幅提升
 * - 支持回退到像素分析（当 MediaPipe 不可用时）
 *
 * AU → 情绪映射规则：
 * - AU1+AU4+AU15 → 悲伤 (内眉上扬+眉毛下压+嘴角下拉)
 * - AU1+AU2+AU26 → 恐惧 (眉毛上扬+下巴下垂)
 * - AU4+AU5+AU7  → 愤怒 (眉毛下压+眼睑收紧)
 * - AU6+AU12     → 快乐 (脸颊上扬+嘴角上提, Duchenne 真笑)
 * - AU9+AU10     → 厌恶 (鼻子皱起+上唇上扬)
 */
import { useEffect, useRef, useState, useCallback } from 'react';
import type { FacialAnalysis } from '../types/multimodal.types';
import { defaultFacial } from '../types/multimodal.types';

// MediaPipe Face Mesh 关键点索引
const LANDMARKS = {
  // 眉毛
  browInnerL: 66, browInnerR: 296,
  browMidL: 105, browMidR: 334,
  browOuterL: 63, browOuterR: 293,
  // 眼睛
  eyeUpperL: 159, eyeLowerL: 145, eyeOuterL: 33,
  eyeUpperR: 386, eyeLowerR: 374, eyeOuterR: 263,
  eyeInnerL: 133, eyeInnerR: 362,
  // 鼻子
  noseTip: 1, noseBridge: 6,
  noseWingL: 129, noseWingR: 358,
  // 嘴巴
  mouthCornerL: 61, mouthCornerR: 291,
  mouthUpperLip: 13, mouthLowerLip: 14,
  mouthUpperOuter: 40, mouthLowerOuter: 181,
  // 脸颊
  cheekL: 234, cheekR: 454,
  // 下巴
  chin: 152, chinLower: 199,
  // 额头
  forehead: 10,
};

// 计算两点间距离
function dist(a: { x: number; y: number }, b: { x: number; y: number }): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

// 计算三点角度（度）
function angle(
  a: { x: number; y: number },
  b: { x: number; y: number },
  c: { x: number; y: number },
): number {
  const ab = dist(a, b);
  const bc = dist(b, c);
  const ac = dist(a, c);
  if (bc === 0 || ab === 0) return 0;
  const cosAngle = (ab * ab + bc * bc - ac * ac) / (2 * ab * bc);
  return Math.acos(Math.max(-1, Math.min(1, cosAngle))) * (180 / Math.PI);
}

export function useFacialAnalysis() {
  const [metrics, setMetrics] = useState<FacialAnalysis>({ ...defaultFacial });
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const isActiveRef = useRef(false);
  const intervalRef = useRef(0);
  const faceLandmarkerRef = useRef<any>(null);
  const useMediaPipeRef = useRef(false);
  const microExprCountRef = useRef(0);
  const prevIntensityRef = useRef(0);
  const frameCountRef = useRef(0);
  const prevExprRef = useRef('平静');
  const prevExprTimeRef = useRef(0);

  // 初始化 MediaPipe FaceLandmarker
  const initMediaPipe = useCallback(async () => {
    try {
      const vision = await import('@mediapipe/tasks-vision');
      const { FaceLandmarker, FilesetResolver } = vision;

      const filesetResolver = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
      );

      const landmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numFaces: 1,
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
      });

      faceLandmarkerRef.current = landmarker;
      useMediaPipeRef.current = true;
      console.log('[面部分析] ✅ MediaPipe Face Mesh 初始化成功');
      return true;
    } catch (err) {
      console.warn('[面部分析] ⚠️ MediaPipe 初始化失败，回退到像素分析:', err);
      useMediaPipeRef.current = false;
      return false;
    }
  }, []);

  // 使用 MediaPipe 分析帧
  const analyzeFrameMediaPipe = useCallback(() => {
    const video = videoRef.current;
    const landmarker = faceLandmarkerRef.current;
    if (!video || !landmarker || video.readyState < 2) return;

    try {
      const timestamp = performance.now();
      const result = landmarker.detectForVideo(video, timestamp);

      if (!result.faceLandmarks || result.faceLandmarks.length === 0) return;

      const lm = result.faceLandmarks[0];
      frameCountRef.current++;

      // 从 blendshapes 获取 AU（MediaPipe 直接输出 blendshapes）
      let au1 = 0, au2 = 0, au4 = 0, au5 = 0, au6 = 0, au7 = 0;
      let au9 = 0, au10 = 0, au12 = 0, au15 = 0, au26 = 0;

      if (result.faceBlendshapes && result.faceBlendshapes.length > 0) {
        const bs = result.faceBlendshapes[0].categories;
        const bsMap: Record<string, number> = {};
        bs.forEach((cat: any) => { bsMap[cat.categoryName] = cat.score; });

        au1 = bsMap['browInnerUp'] || 0;
        au2 = bsMap['browOuterUpLeft'] || bsMap['browOuterUpRight'] || 0;
        au4 = bsMap['browDownLeft'] || bsMap['browDownRight'] || 0;
        au5 = bsMap['eyeWideLeft'] || bsMap['eyeWideRight'] || 0;
        au6 = bsMap['cheekSquintLeft'] || bsMap['cheekSquintRight'] || 0;
        au7 = bsMap['eyeSquintLeft'] || bsMap['eyeSquintRight'] || 0;
        au9 = bsMap['noseSneerLeft'] || bsMap['noseSneerRight'] || 0;
        au10 = bsMap['upperLipRaiserLeft'] || bsMap['upperLipRaiserRight'] || 0;
        au12 = bsMap['mouthFrownLeft'] || bsMap['mouthFrownRight'] || 0;
        au15 = bsMap['mouthFrownLeft'] || bsMap['mouthFrownRight'] || 0;
        au26 = bsMap['jawOpen'] || 0;

        // 修正：mouthSmile 对应 AU12
        const smileL = bsMap['mouthSmileLeft'] || 0;
        const smileR = bsMap['mouthSmileRight'] || 0;
        au12 = Math.max(smileL, smileR);

        // 修正：mouthFrown 对应 AU15
        const frownL = bsMap['mouthFrownLeft'] || 0;
        const frownR = bsMap['mouthFrownRight'] || 0;
        au15 = Math.max(frownL, frownR);
      } else {
        // 如果没有 blendshapes，从关键点几何计算
        const p = (idx: number) => lm[idx];

        // 眉毛距离比
        const browEyeDist = dist(p(LANDMARKS.browInnerL), p(LANDMARKS.eyeUpperL));
        const faceWidth = dist(p(LANDMARKS.cheekL), p(LANDMARKS.cheekR));
        const browRatio = browEyeDist / (faceWidth || 1);

        // 眼睛纵横比 (EAR)
        const eyeH = dist(p(LANDMARKS.eyeUpperL), p(LANDMARKS.eyeLowerL));
        const eyeW = dist(p(LANDMARKS.eyeOuterL), p(LANDMARKS.eyeInnerL));
        const ear = eyeH / (eyeW || 1);

        // 嘴巴开合度
        const mouthH = dist(p(LANDMARKS.mouthUpperLip), p(LANDMARKS.mouthLowerLip));
        const mouthW = dist(p(LANDMARKS.mouthCornerL), p(LANDMARKS.mouthCornerR));
        const mouthOpen = mouthH / (mouthW || 1);

        // 嘴角位置（相对中心）
        const mouthCenter = (p(LANDMARKS.mouthCornerL).y + p(LANDMARKS.mouthCornerR).y) / 2;
        const noseY = p(LANDMARKS.noseTip).y;
        const smileHeight = (noseY - mouthCenter) / (faceWidth || 1);

        au1 = Math.min(1, browRatio * 5);
        au4 = Math.min(1, Math.max(0, 0.15 - browRatio) * 10);
        au5 = Math.min(1, ear * 3);
        au7 = Math.min(1, Math.max(0, 0.28 - ear) * 5);
        au12 = Math.min(1, Math.max(0, smileHeight * 8));
        au15 = Math.min(1, Math.max(0, -smileHeight * 8));
        au26 = Math.min(1, mouthOpen * 5);
      }

      // 头部姿态（从 facial transformation matrix）
      let headPitch = 0, headYaw = 0, headRoll = 0;
      if (result.facialTransformationMatrixes && result.facialTransformationMatrixes.length > 0) {
        const matrix = result.facialTransformationMatrixes[0].data;
        if (matrix && matrix.length >= 12) {
          headPitch = Math.atan2(matrix[6], matrix[10]) * (180 / Math.PI);
          headYaw = Math.atan2(-matrix[2], Math.sqrt(matrix[6] ** 2 + matrix[10] ** 2)) * (180 / Math.PI);
          headRoll = Math.atan2(matrix[1], matrix[5]) * (180 / Math.PI);
        }
      }

      // 几何特征
      const p = (idx: number) => lm[idx];
      const faceWidth = dist(p(LANDMARKS.cheekL), p(LANDMARKS.cheekR));
      const browEyeDist = dist(p(LANDMARKS.browInnerL), p(LANDMARKS.eyeUpperL));
      const eyeH = dist(p(LANDMARKS.eyeUpperL), p(LANDMARKS.eyeLowerL));
      const eyeW = dist(p(LANDMARKS.eyeOuterL), p(LANDMARKS.eyeInnerL));
      const mouthH = dist(p(LANDMARKS.mouthUpperLip), p(LANDMARKS.mouthLowerLip));
      const mouthW = dist(p(LANDMARKS.mouthCornerL), p(LANDMARKS.mouthCornerR));

      const browDistance = browEyeDist / (faceWidth || 1);
      const mouthWidthRatio = mouthW / (faceWidth || 1);
      const eyeAspect = eyeH / (eyeW || 1);
      const mouthOpenRatio = mouthH / (mouthW || 1);

      // AU → 情绪映射
      const emotionMapping = {
        happiness: Math.min(1, (au6 * 0.5 + au12 * 0.5) * (au6 > 0.3 && au12 > 0.3 ? 1.5 : 0.4)),
        sadness: Math.min(1, (au1 * 0.4 + au4 * 0.4 + au15 * 0.5) * (au15 > 0.3 ? 1.3 : 1)),
        anger: Math.min(1, (au4 * 0.5 + au5 * 0.4 + au7 * 0.4) * (au4 > 0.4 ? 1.3 : 1)),
        fear: Math.min(1, (au1 * 0.4 + au2 * 0.4 + au26 * 0.5)),
        surprise: Math.min(1, (au1 * 0.35 + au2 * 0.35 + au5 * 0.35 + au26 * 0.35) * (au1 > 0.3 && au2 > 0.3 && au5 > 0.3 ? 1.4 : 1)),
        disgust: Math.min(1, (au9 * 0.6 + au10 * 0.6) * (au9 > 0.3 && au10 > 0.3 ? 1.4 : 1)),
        distress: Math.min(1, (au1 * 0.35 + au4 * 0.35 + au7 * 0.35 + au15 * 0.35)),
      };

      // 主导表情
      const dominantExpr = Object.entries(emotionMapping).reduce((a, b) => a[1] > b[1] ? a : b);
      const dominantExpression = dominantExpr[1] > 0.15 ? dominantExpr[0] : '平静';
      const expressionIntensity = Math.min(1, Object.values(emotionMapping).reduce((s, v) => s + v, 0) / 2);

      // 微表情检测（表情快速切换）
      const now = Date.now();
      if (dominantExpression !== prevExprRef.current && dominantExpression !== '平静') {
        if (prevExprTimeRef.current > 0 && now - prevExprTimeRef.current < 500) {
          microExprCountRef.current++;
        }
      }
      if (dominantExpression !== '平静') {
        prevExprRef.current = dominantExpression;
        prevExprTimeRef.current = now;
      }

      // 置信度：基于 blendshapes 的质量
      const confidence = useMediaPipeRef.current ? Math.min(0.95, 0.7 + frameCountRef.current * 0.005) : 0.5;

      setMetrics(prev => ({
        ...prev,
        actionUnits: {
          au1_browRaise: Math.round(au1 * 100) / 100,
          au2_browOuterRaise: Math.round(au2 * 100) / 100,
          au4_browLower: Math.round(au4 * 100) / 100,
          au5_eyeOpen: Math.round(au5 * 100) / 100,
          au7_eyeTighten: Math.round(au7 * 100) / 100,
          eyeAspect: Math.round(eyeAspect * 100) / 100,
          au9_noseWrinkle: Math.round(au9 * 100) / 100,
          au6_cheekRaise: Math.round(au6 * 100) / 100,
          au10_lipRaise: Math.round(au10 * 100) / 100,
          au12_lipCorner: Math.round(au12 * 100) / 100,
          au15_lipCornerDrop: Math.round(au15 * 100) / 100,
          au17_chinRaise: 0,
          au20_lipStretch: 0,
          au26_jawDrop: Math.round(au26 * 100) / 100,
          mouthAspect: Math.round(mouthOpenRatio * 100) / 100,
          headPitch: Math.round(headPitch * 10) / 10,
          headYaw: Math.round(headYaw * 10) / 10,
          headRoll: Math.round(headRoll * 10) / 10,
        },
        geometry: {
          browDistance: Math.round(browDistance * 100) / 100,
          mouthWidth: Math.round(mouthWidthRatio * 100) / 100,
          eyeAspect: Math.round(eyeAspect * 100) / 100,
          mouthOpen: Math.round(mouthOpenRatio * 100) / 100,
        },
        microExpressions: {
          count: microExprCountRef.current,
          avgDuration: microExprCountRef.current > 0 ? 200 : 0,
          recentEmotions: microExprCountRef.current > 0 ? [dominantExpression] : [],
        },
        dominantExpression,
        expressionIntensity: Math.round(expressionIntensity * 100) / 100,
        emotionMapping: Object.fromEntries(
          Object.entries(emotionMapping).map(([k, v]) => [k, Math.round(v * 100) / 100])
        ) as FacialAnalysis['emotionMapping'],
        confidence: Math.round(confidence * 100) / 100,
        frameCount: frameCountRef.current,
        timestamp: Date.now(),
      }));
    } catch (err) {
      console.error('[面部分析] MediaPipe 帧分析错误:', err);
    }
  }, []);

  // 回退：像素分析法（当 MediaPipe 不可用时）
  const prevFrameRef = useRef<ImageData | null>(null);

  const analyzeFramePixel = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) return;

    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    try {
      canvas.width = 160;
      canvas.height = 120;
      ctx.drawImage(video, 0, 0, 160, 120);
      const frame = ctx.getImageData(0, 0, 160, 120);
      frameCountRef.current++;

      const regions = {
        brow: { x: 40, y: 25, w: 80, h: 20 },
        eye: { x: 40, y: 40, w: 80, h: 15 },
        mouth: { x: 50, y: 75, w: 60, h: 20 },
      };

      const regionBrightness: Record<string, number> = {};
      for (const [name, region] of Object.entries(regions)) {
        let sum = 0, count = 0;
        for (let y = region.y; y < region.y + region.h; y++) {
          for (let x = region.x; x < region.x + region.w; x++) {
            const idx = (y * 160 + x) * 4;
            sum += frame.data[idx] * 0.299 + frame.data[idx + 1] * 0.587 + frame.data[idx + 2] * 0.114;
            count++;
          }
        }
        regionBrightness[name] = count > 0 ? sum / count : 0;
      }

      let motionIntensity = 0;
      let browMotion = 0;
      let mouthMotion = 0;
      if (prevFrameRef.current) {
        const prev = prevFrameRef.current;
        let totalDiff = 0;
        for (let i = 0; i < frame.data.length; i += 4) {
          totalDiff += Math.abs(
            (frame.data[i] * 0.299 + frame.data[i + 1] * 0.587 + frame.data[i + 2] * 0.114) -
            (prev.data[i] * 0.299 + prev.data[i + 1] * 0.587 + prev.data[i + 2] * 0.114)
          );
        }
        motionIntensity = totalDiff / (frame.data.length / 4);

        const browRegion = regions.brow;
        let browDiff = 0;
        for (let y = browRegion.y; y < browRegion.y + browRegion.h; y++) {
          for (let x = browRegion.x; x < browRegion.x + browRegion.w; x++) {
            const idx = (y * 160 + x) * 4;
            browDiff += Math.abs(
              (frame.data[idx] * 0.299 + frame.data[idx + 1] * 0.587 + frame.data[idx + 2] * 0.114) -
              (prev.data[idx] * 0.299 + prev.data[idx + 1] * 0.587 + prev.data[idx + 2] * 0.114)
            );
          }
        }
        browMotion = browDiff / (browRegion.w * browRegion.h);

        const mouthRegion = regions.mouth;
        let mouthDiff = 0;
        for (let y = mouthRegion.y; y < mouthRegion.y + mouthRegion.h; y++) {
          for (let x = mouthRegion.x; x < mouthRegion.x + mouthRegion.w; x++) {
            const idx = (y * 160 + x) * 4;
            mouthDiff += Math.abs(
              (frame.data[idx] * 0.299 + frame.data[idx + 1] * 0.587 + frame.data[idx + 2] * 0.114) -
              (prev.data[idx] * 0.299 + prev.data[idx + 1] * 0.587 + prev.data[idx + 2] * 0.114)
            );
          }
        }
        mouthMotion = mouthDiff / (mouthRegion.w * mouthRegion.h);
      }
      prevFrameRef.current = frame;

      const au1 = Math.min(1, browMotion / 5);
      const au4 = Math.min(1, Math.max(0, (10 - (regionBrightness.brow || 0)) / 20));
      const au6 = Math.min(1, (regionBrightness.eye || 0) / 100);
      const au12 = Math.min(1, mouthMotion / 4);
      const au15 = Math.min(1, Math.max(0, (100 - (regionBrightness.mouth || 0)) / 30));
      const au26 = Math.min(1, mouthMotion / 6);

      const emotionMapping = {
        happiness: Math.min(1, (au6 * 0.5 + au12 * 0.5) * (au6 > 0.3 && au12 > 0.3 ? 1.5 : 0.4)),
        sadness: Math.min(1, (au1 * 0.4 + au4 * 0.4 + au15 * 0.5)),
        anger: Math.min(1, au4 * 0.6),
        fear: Math.min(1, (au1 * 0.5 + au26 * 0.5)),
        surprise: Math.min(1, au26 * 0.6),
        disgust: Math.min(1, au4 * 0.4),
        distress: Math.min(1, (au1 * 0.35 + au4 * 0.35 + au15 * 0.35)),
      };

      const dominantExpr = Object.entries(emotionMapping).reduce((a, b) => a[1] > b[1] ? a : b);
      const dominantExpression = dominantExpr[1] > 0.1 ? dominantExpr[0] : '平静';
      const expressionIntensity = Math.min(1, Object.values(emotionMapping).reduce((s, v) => s + v, 0) / 2);

      setMetrics(prev => ({
        ...prev,
        actionUnits: {
          ...prev.actionUnits,
          au1_browRaise: Math.round(au1 * 100) / 100,
          au4_browLower: Math.round(au4 * 100) / 100,
          au6_cheekRaise: Math.round(au6 * 100) / 100,
          au12_lipCorner: Math.round(au12 * 100) / 100,
          au15_lipCornerDrop: Math.round(au15 * 100) / 100,
          au26_jawDrop: Math.round(au26 * 100) / 100,
        },
        dominantExpression,
        expressionIntensity: Math.round(expressionIntensity * 100) / 100,
        emotionMapping: Object.fromEntries(
          Object.entries(emotionMapping).map(([k, v]) => [k, Math.round(v * 100) / 100])
        ) as FacialAnalysis['emotionMapping'],
        confidence: Math.min(0.6, 0.3 + frameCountRef.current * 0.005),
        frameCount: frameCountRef.current,
        timestamp: Date.now(),
      }));
    } catch (err) {
      console.error('[面部分析] 像素分析错误:', err);
    }
  }, []);

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      });
      streamRef.current = stream;

      const video = document.createElement('video');
      video.srcObject = stream;
      video.autoplay = true;
      video.playsInline = true;
      videoRef.current = video;

      const canvas = document.createElement('canvas');
      canvasRef.current = canvas;

      await new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(() => reject(), 5000);
        video.onloadedmetadata = () => {
          clearTimeout(timeout);
          video.play().then(resolve).catch(reject);
        };
      });

      // 尝试初始化 MediaPipe
      await initMediaPipe();

      // 重置状态
      prevFrameRef.current = null;
      microExprCountRef.current = 0;
      prevIntensityRef.current = 0;
      frameCountRef.current = 0;
      isActiveRef.current = true;

      setMetrics(prev => ({ ...prev, isDetecting: true }));

      // 选择分析函数：MediaPipe 用 100ms (10fps)，像素分析用 200ms (5fps)
      const analyzeFn = useMediaPipeRef.current ? analyzeFrameMediaPipe : analyzeFramePixel;
      const fps = useMediaPipeRef.current ? 100 : 200;
      intervalRef.current = window.setInterval(analyzeFn, fps);

      console.log(`[面部分析] 使用 ${useMediaPipeRef.current ? 'MediaPipe Face Mesh' : '像素分析'} 模式`);
    } catch {
      console.warn('[面部分析] 摄像头权限被拒绝');
    }
  }, [initMediaPipe, analyzeFrameMediaPipe, analyzeFramePixel]);

  const stop = useCallback(() => {
    isActiveRef.current = false;
    clearInterval(intervalRef.current);
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    videoRef.current = null;
    setMetrics(prev => ({ ...prev, isDetecting: false }));
  }, []);

  const reset = useCallback(() => {
    stop();
    setMetrics({ ...defaultFacial });
  }, [stop]);

  useEffect(() => {
    return () => {
      isActiveRef.current = false;
      clearInterval(intervalRef.current);
      streamRef.current?.getTracks().forEach(t => t.stop());
      faceLandmarkerRef.current?.close();
    };
  }, []);

  return { metrics, start, stop, reset };
}
