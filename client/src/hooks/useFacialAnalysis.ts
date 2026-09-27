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
import type { AgeGroup } from './ageConfig';
import { getFaceConfig } from './ageConfig';
import { mediaStreamManager, MediaUnavailableError } from '../services/mediaStreamManager';
import { EyePatchExtractor, type EyePatchPixels } from './eyePatchCanvas';
import {
  DEFAULT_BLINK_FPS,
  METRICS_CHANNEL_HZ,
  intervalMsFor,
  measuredFps,
  metricsEveryN,
  nextBlinkRate,
  shouldComputeMetrics,
} from './blinkChannel';

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

/** 每帧信号：供下游 Hook 复用本帧的推理结果，避免重复开摄像头/重复推理 */export interface FacialFrameSignal {
  /** 绿色通道均值（供 rPPG 使用，当前 rPPG 未启用） */
  greenMean: number;
  /** 眼睛纵横比 EAR（左眼，blendshape 缺失时的眨眼降级路径） */
  eyeAspect: number;
  /** 头部俯仰（度） */
  headPitch: number;
  /** 头部偏航（度） */
  headYaw: number;
  /**
   * MediaPipe eyeBlinkLeft/Right 取双眼较小值（0~1）。
   * 单眼眨动/眯眼会被 min 滤掉；模型未输出 blendshape 时为 null，由下游显式降级。
   */
  blinkScore: number | null;
  /**
   * 眨眼通道当前的实际帧率（自适应档位 30/15/10，跑不动会自动降档）。
   * 下游据此如实上报「本次测量的时长量化粒度」= 1000/该值 ms。
   */
  blinkChannelFps?: number;
  /** 本拍「推理 + 信号下发」的耗时 ms（用于解释为什么降档） */
  inferMs?: number;
}

export type FrameSignalCallback = (signal: FacialFrameSignal) => void;

/** 眼 patch 回调：仅在指标通道那一拍产生（默认 10fps），供浏览器内 ONNX 二次判定使用 */
export type EyePatchCallback = (patches: EyePatchPixels) => void;

/** 32×32 画面中央区域的绿通道均值（供 rPPG）。
 *  画布通过 holder **复用**：原实现每帧 `document.createElement('canvas')`，
 *  10fps 时是每秒 10 个垃圾对象，提到 30fps 后会是 30 个。 */
function readGreenMean(
  video: HTMLVideoElement,
  holder: { current: HTMLCanvasElement | null },
): number {
  let canvas = holder.current;
  if (!canvas) {
    canvas = document.createElement('canvas');
    canvas.width = 32;
    canvas.height = 32;
    holder.current = canvas;
  }
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return 0;
  ctx.drawImage(video, 0, 0, 32, 32);
  const faceRegion = ctx.getImageData(8, 4, 16, 12);
  let gSum = 0;
  let gCount = 0;
  for (let i = 0; i < faceRegion.data.length; i += 4) {
    gSum += faceRegion.data[i + 1];
    gCount++;
  }
  return gCount > 0 ? gSum / gCount : 0;
}

export function useFacialAnalysis(
  ageGroup: AgeGroup | null = null,
  onFrameSignal?: FrameSignalCallback | null,
  onEyePatchFrame?: EyePatchCallback | null,
) {
  const [metrics, setMetrics] = useState<FacialAnalysis>({ ...defaultFacial });
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const isActiveRef = useRef(false);
  const intervalRef = useRef(0);
  /**
   * 正在进行的 `start()`。用于**重入保护**：
   * `isActiveRef` 是在初始化成功之后才置 true 的，所以如果只看它，
   * 两次相邻的 `start()`（第一次还没 await 完）都会走到"打开设备"那一支，
   * 结果是两个 `setInterval` + 两个 `<video>` —— 即使底层流是同一份也会泄漏循环。
   */
  const startingRef = useRef<Promise<void> | null>(null);
  const faceLandmarkerRef = useRef<any>(null);
  const useMediaPipeRef = useRef(false);
  const microExprCountRef = useRef(0);
  const prevIntensityRef = useRef(0);
  const frameCountRef = useRef(0);
  const prevExprRef = useRef('平静');
  const prevExprTimeRef = useRef(0);

  // ── 眨眼通道（30fps 自适应）与眼 patch 通道（10fps）相关状态 ──
  /** 上一拍是否还在跑：跑不完就跳过本拍，避免 setInterval 排队导致时间戳抖动 */
  const busyRef = useRef(false);
  /** 拍计数：只有每 metricsEvery 拍才做昂贵的指标计算与 React 更新 */
  const tickRef = useRef(0);
  /** 当前眨眼通道帧率（会在 30/15/10 之间自适应） */
  const blinkFpsRef = useRef<number>(DEFAULT_BLINK_FPS);
  const consecutiveSlowRef = useRef(0);
  const consecutiveFastRef = useRef(0);
  /** 近期眨眼通道时间戳，用于如实上报实测帧率 */
  const blinkTimesRef = useRef<number[]>([]);
  /** 32×32 绿通道均值画布：**复用**，不再每帧 createElement（原实现每帧新建一个 canvas） */
  const greenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  /** 眼 patch 提取器（懒创建，只在需要时占用两个离屏画布） */
  const patchExtractorRef = useRef<EyePatchExtractor | null>(null);
  /** 当前定时器句柄与实测帧率（供自适应换档时重建定时器） */
  const currentIntervalRef = useRef(0);
  /** 当前分析函数（换档时重建定时器要用它，用 ref 避免 useCallback 循环依赖） */
  const analyzeFnRef = useRef<(() => void) | null>(null);

  // 年龄校准配置
  const faceConfig = getFaceConfig(ageGroup);

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
    // 上一拍还没跑完就跳过本拍：30fps 下最怕 setInterval 排队 —— 时间戳会被推后，
    // 而眨眼时长完全依赖时间戳，排队等于把时长测歪。
    if (busyRef.current) return;
    const video = videoRef.current;
    const landmarker = faceLandmarkerRef.current;
    if (!video || !landmarker || video.readyState < 2) return;
    busyRef.current = true;
    const tickStart = performance.now();

    try {
      const timestamp = performance.now();
      const result = landmarker.detectForVideo(video, timestamp);

      if (!result.faceLandmarks || result.faceLandmarks.length === 0) {
        // 「没有脸」不等于「有脸但表情平静」。
        // 旧实现这里直接 return，于是界面保留上一帧读数，并用旧的 confidence
        // 继续参与融合 —— 人已经离开画面，系统还在"读"他的情绪。
        // 现在明确上报本帧无有效人脸；返回同一个对象引用时 React 会跳过重渲染。
        setMetrics(prev =>
          prev.facePresent || prev.confidence !== null
            ? { ...prev, facePresent: false, confidence: null }
            : prev,
        );
        return;
      }

      const lm = result.faceLandmarks[0];
      frameCountRef.current++;

      // 从 blendshapes 获取 AU（MediaPipe 直接输出 blendshapes）
      let au1 = 0, au2 = 0, au4 = 0, au5 = 0, au6 = 0, au7 = 0;
      let au9 = 0, au10 = 0, au12 = 0, au15 = 0, au26 = 0;
      // 眼睑闭合度：blendshape 缺失时必须保持 null，不能默认为 0（否则等价于「永远睁眼」）
      let blinkScore: number | null = null;
      /** 本帧的 AU 是否来自模型真输出（blendshape）—— 置信度的第一因子。 */
      let hasBlendshapes = false;

      if (result.faceBlendshapes && result.faceBlendshapes.length > 0) {
        hasBlendshapes = true;
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

        // 眼睑闭合度（眨眼主路径）：双眼取较小值，单眼眨动/眯眼不计为眨眼
        const blinkL = bsMap['eyeBlinkLeft'];
        const blinkR = bsMap['eyeBlinkRight'];
        blinkScore =
          blinkL !== undefined && blinkR !== undefined
            ? Math.min(blinkL, blinkR)
            : null;
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
      let hasMatrix = false;
      if (result.facialTransformationMatrixes && result.facialTransformationMatrixes.length > 0) {
        const matrix = result.facialTransformationMatrixes[0].data;
        if (matrix && matrix.length >= 12) {
          hasMatrix = true;
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

      // ── 眨眼通道：**每一拍都下发**（这正是把眨眼提到 30fps 的意义）────────
      //
      // 眨眼的准确性只取决于两件事：闭眼判定是否可信、以及**时间戳有多密**。
      // 这里把「每拍都做的事」（取本帧 blendshape/EAR/头姿 → 下发）与
      // 「每 N 拍才做的事」（AU/情绪映射 + React setState + 眼 patch）拆开，
      // 于是眨眼拿到 30fps 的时间分辨率，而 UI 与其余指标仍按 10fps 更新。
      const tick = tickRef.current++;
      const blinkFps = blinkFpsRef.current;
      const metricsEvery = metricsEveryN(blinkFps);

      // 实测帧率：用真实到达时间算，不用"设定值"（设定值只是目标）
      const times = blinkTimesRef.current;
      times.push(timestamp);
      if (times.length > 240) times.shift();
      const actualBlinkFps = measuredFps(times, 3000) || blinkFps;

      if (onFrameSignal) {
        const greenMean = readGreenMean(video, greenCanvasRef);
        onFrameSignal({
          greenMean,
          eyeAspect,
          headPitch,
          headYaw,
          blinkScore,
          blinkChannelFps: actualBlinkFps,
          inferMs: Math.round((performance.now() - tickStart) * 10) / 10,
        });
      }

      // 非指标拍：眨眼已经在上面拿到数据，这里直接结束，省掉情绪映射与 React 更新
      if (!shouldComputeMetrics(tick, metricsEvery)) return;

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

      // 微表情检测（表情快速切换，阈值按年龄校准）
      const microThreshold = faceConfig.microExpressionThreshold * 1000; // 转换为 ms
      const now = Date.now();
      if (dominantExpression !== prevExprRef.current && dominantExpression !== '平静') {
        if (prevExprTimeRef.current > 0 && now - prevExprTimeRef.current < microThreshold) {
          microExprCountRef.current++;
        }
      }
      if (dominantExpression !== '平静') {
        prevExprRef.current = dominantExpression;
        prevExprTimeRef.current = now;
      }

      // ── 置信度：只由本帧可测量的信号质量决定 ─────────────────────────────
      //
      // 旧实现是 `Math.min(0.95, 0.7 + frameCount * 0.005)` —— 一个**帧计数常量**：
      // 摄像头开着满 50 帧（5 秒）就恒为 0.95，画面全黑、没有人在镜头里也一样。
      // 而它会一路进融合权重并改写落库的 `patientProfile.riskLevel`，
      // 所以这里换成三个**本帧真实测得的因子**：
      //
      //   · 来源：AU 来自模型 blendshape 输出(1.0)，还是退化成人工关键点比值(0.5)
      //   · 尺度：脸在画面里太小 → 关键点噪声大（faceWidth 是脸宽占画面宽度的比例）
      //   · 姿态：侧脸 / 大角度俯仰时 AU 与头姿估计都不可靠
      //
      // 上限 0.9：不假装 1.0 —— "模型输出 + 正面 + 足够大"已是这条链路的合理上限。
      const sourceFactor = hasBlendshapes ? 1 : 0.5;
      // 脸宽占比 0.10 → 0；0.32 → 1（640×480 下人脸的常见区间）
      const sizeFactor = Math.max(0, Math.min(1, (faceWidth - 0.1) / 0.22));
      const yawFactor = Math.max(0, 1 - Math.abs(headYaw) / 45);
      const pitchFactor = Math.max(0, 1 - Math.abs(headPitch) / 40);
      // 头姿本身也来自模型输出时才全额计入
      const poseFactor = Math.min(yawFactor, pitchFactor) * (hasMatrix ? 1 : 0.7);
      const confidence = Math.min(0.9, sourceFactor * sizeFactor * poseFactor);

      // 提取绿色通道均值（供 rPPG 使用）+ 把本帧结果交给下游（眨眼/头姿）的动作
      // 已经挪到上面的「眨眼通道」块里，并且画布改为复用（greenCanvasRef），
      // 不再每帧 createElement —— 原来 10fps 时每帧新建一个 canvas，
      // 提到 30fps 后那会变成每秒 30 个垃圾对象。

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
          recentEmotions: microExprCountRef.current > 0 ? [dominantExpression] : [],
        },
        dominantExpression,
        expressionIntensity: Math.round(expressionIntensity * 100) / 100,
        emotionMapping: Object.fromEntries(
          Object.entries(emotionMapping).map(([k, v]) => [k, Math.round(v * 100) / 100])
        ) as FacialAnalysis['emotionMapping'],
        confidence: Math.round(confidence * 100) / 100,
        facePresent: true,
        frameCount: frameCountRef.current,
        timestamp: Date.now(),
        unavailableReason: '',
      }));

      // ── 眼 patch 通道（默认 10fps）：供浏览器内 ONNX 每眼二次判定 ──────────
      // 只在指标拍做，因为裁两张图 + getImageData 比眨眼信号贵得多。
      // 视频不出设备：整条链路在本机 canvas + WASM 内完成。
      if (onEyePatchFrame) {
        if (!patchExtractorRef.current) patchExtractorRef.current = new EyePatchExtractor();
        const patches = patchExtractorRef.current.extract(video, lm);
        if (patches) onEyePatchFrame(patches);
      }
    } catch (err) {
      console.error('[面部分析] MediaPipe 帧分析错误:', err);
    } finally {
      // ── 自适应换档：跑不动就 30 → 15 → 10，跑得轻松再升回去 ──
      // 判据是「本拍推理 + 下发」的实测耗时，不是猜测；换档后重建定时器。
      const inferMs = performance.now() - tickStart;
      busyRef.current = false;

      const decision = nextBlinkRate({
        currentFps: blinkFpsRef.current,
        inferMs,
        consecutiveSlow: consecutiveSlowRef.current,
        consecutiveFast: consecutiveFastRef.current,
      });
      consecutiveSlowRef.current = decision.consecutiveSlow;
      consecutiveFastRef.current = decision.consecutiveFast;

      if (decision.fps !== blinkFpsRef.current) {
        console.log(
          `[面部分析] 眨眼通道换档 ${blinkFpsRef.current} → ${decision.fps}fps` +
          `（本拍耗时 ${inferMs.toFixed(1)}ms，时长量化粒度 ${intervalMsFor(decision.fps)}ms）`,
        );
        blinkFpsRef.current = decision.fps;
        if (currentIntervalRef.current) window.clearInterval(currentIntervalRef.current);
        currentIntervalRef.current = window.setInterval(
          analyzeFnRef.current ?? (() => {}),
          intervalMsFor(decision.fps),
        );
      }
    }
  }, [onFrameSignal, onEyePatchFrame]);

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
        // 像素回退路径**没有做人脸检测**（在 160×120 上按几个写死的矩形算亮度和帧差），
        // 也没有任何模型输出，因此它无法提供"检测置信度"，只能如实报 null。
        // 旧实现这里是 `Math.min(0.6, 0.3 + frameCount * 0.005)` —— 同样是帧计数常量。
        // 报 null 的后果是下游会把 facial 这个模态剔除，而这正是应有的行为：
        // MediaPipe 不可用时我们**没有**可信的面部读数，不该硬凑一个出来。
        confidence: null,
        facePresent: false,
        frameCount: frameCountRef.current,
        timestamp: Date.now(),
        unavailableReason: 'MediaPipe 不可用，已降级为像素法（无面部读数）',
      }));
    } catch (err) {
      console.error('[面部分析] 像素分析错误:', err);
    }
  }, []);

  /**
   * 开始采集与分析。
   *
   * **幂等**：已经在运行、或上一次 `start()` 还没结束时直接复用，不会开出第二个
   * `<video>` 和第二个定时器 —— 即使底层流是同一份，循环泄漏一样是泄漏。
   *
   * **失败会抛出** `MediaUnavailableError`（带 `reason`），不再静默吞掉。
   * 调用方必须能区分"开了"和"没开成"，否则界面会在摄像头没打开时显示"已开启"。
   */
  const start = useCallback(async (): Promise<void> => {
    if (isActiveRef.current) return;
    if (startingRef.current) return startingRef.current;

    const run = async (): Promise<void> => {
      try {
        // 走统一的设备管理器：同一个 consumer 重复借用拿到的是同一份流，
        // 也不会和通话页抢设备（引用计数在这里兜底）。
        const stream = await mediaStreamManager.acquire('camera', 'perception-facial', {
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
          const timeout = setTimeout(() => reject(new Error('摄像头画面加载超时')), 5000);
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
        // 眨眼通道复位（否则上一段会话的帧率/时间戳会污染本次统计）
        tickRef.current = 0;
        busyRef.current = false;
        blinkFpsRef.current = DEFAULT_BLINK_FPS;
        consecutiveSlowRef.current = 0;
        consecutiveFastRef.current = 0;
        blinkTimesRef.current = [];
        isActiveRef.current = true;

        setMetrics(prev => ({ ...prev, isDetecting: true, unavailableReason: '' }));

        // 两条通道：
        //   · MediaPipe 可用 → 眨眼通道按 30fps 起（会自适应降档），其余指标每 3 拍算一次（10fps）
        //   · MediaPipe 不可用 → 退回像素分析，维持原来的 200ms（5fps）
        const analyzeFn = useMediaPipeRef.current ? analyzeFrameMediaPipe : analyzeFramePixel;
        const intervalMs = useMediaPipeRef.current ? intervalMsFor(DEFAULT_BLINK_FPS) : 200;
        analyzeFnRef.current = analyzeFn;
        intervalRef.current = window.setInterval(analyzeFn, intervalMs);
        currentIntervalRef.current = intervalRef.current;

        console.log(
          `[面部分析] 使用 ${useMediaPipeRef.current ? `MediaPipe Face Mesh（眨眼通道 ${DEFAULT_BLINK_FPS}fps / 指标通道 ${METRICS_CHANNEL_HZ}fps）` : '像素分析（5fps）'} 模式`,
        );
      } catch (err) {
        // 失败路径要自己收干净：半开的流 + 半初始化的状态比"没开"更糟
        isActiveRef.current = false;
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = 0;
        }
        mediaStreamManager.release('camera', 'perception-facial');
        streamRef.current = null;
        videoRef.current = null;
        useMediaPipeRef.current = false;

        const reason = err instanceof MediaUnavailableError
          ? err.message
          : `摄像头启动失败（${(err as Error)?.message || err}）`;
        console.warn('[面部分析] 启动失败:', reason);
        // 同步写进 metrics：即使调用方忽略了 rejection，界面也必须能看到真实原因
        setMetrics(prev => ({
          ...prev,
          isDetecting: false,
          facePresent: false,
          confidence: null,
          unavailableReason: reason,
        }));
        throw err instanceof MediaUnavailableError
          ? err
          : new MediaUnavailableError('camera', 'unknown', reason);
      }
    };

    const pending = run().finally(() => {
      startingRef.current = null;
    });
    startingRef.current = pending;
    return pending;
  }, [initMediaPipe, analyzeFrameMediaPipe, analyzeFramePixel]);

  const stop = useCallback(() => {
    isActiveRef.current = false;
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = 0;
    }
    // 自适应换档会重建定时器，所以两个句柄都要清（它们是同一个，但换档后可能不同步）
    if (currentIntervalRef.current) {
      clearInterval(currentIntervalRef.current);
      currentIntervalRef.current = 0;
    }
    analyzeFnRef.current = null;
    busyRef.current = false;
    // 交还给设备管理器，**不要**直接 `getTracks().forEach(stop)`：
    // 通话页可能正共用同一份摄像头流，直接 stop 会把它一起打死；
    // 管理器按引用计数决定是否真的停轨道。这正是以前"两套清理逻辑互相打架"的根源。
    mediaStreamManager.release('camera', 'perception-facial');
    streamRef.current = null;
    videoRef.current = null;
    setMetrics(prev => ({
      ...prev,
      isDetecting: false,
      facePresent: false,
      confidence: null,
    }));
  }, []);

  const reset = useCallback(() => {
    stop();
    setMetrics({ ...defaultFacial });
  }, [stop]);

  useEffect(() => {
    return () => {
      isActiveRef.current = false;
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (currentIntervalRef.current) clearInterval(currentIntervalRef.current);
      analyzeFnRef.current = null;
      mediaStreamManager.releaseAll('perception-facial');
      faceLandmarkerRef.current?.close();
    };
  }, []);

  return { metrics, start, stop, reset, videoRef, streamRef };
}
