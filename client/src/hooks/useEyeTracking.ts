/**
 * 眼动 / 眨眼分析 Hook
 *
 * ## 眨眼信号链（本次优化的重点）
 *
 *   主路径 = `useFacialAnalysis` 已经算出来的 MediaPipe eyeBlinkLeft/Right 眼睑闭合度
 *   （双眼取较小值），交给 `blinkLogic.BlinkStateMachine` 做
 *   「迟滞 0.5/0.35 → 时长窗 60~800ms → 最小间隔 100ms」判定。
 *
 *   与面部分析共用同一路摄像头、同一份推理结果，不再单独开一路 video + FaceLandmarker。
 *   只有模型确实没输出 blendshape 时才降级为关键点纵横比 EAR 阈值法，
 *   并通过 `metrics.blinkMethod` 如实上报本次实际使用的来源。
 *
 * ## 为什么改成「按实测帧率」而不是写死常数
 *
 * 眨眼是本链路里**唯一由时间分辨率决定精度**的指标：
 *   - 闭眼时长只能测到采样间隔的整数倍 → 10fps 是 100ms、30fps 是 33ms；
 *   - 而 `avgBlinkDuration`（>350ms 加 15 分）与 `longBlinkCount`（>400ms 每次加最多 30 分）
 *     直接给 `anxietyIndex` 加分（见 `blinkLogic.stats`）。
 *
 * 原实现有两处与真实不符的常数：
 *   1. `EXPECTED_FPS = 5`，但实际面部分析是 10fps → 覆盖率恒被夹到 1，`signalQuality` 失去意义；
 *   2. 没有暴露「本次时长的量化粒度」，于是界面无法说明"这个时长是 100ms 分辨率的"。
 * 现在两者都按**实测帧率**给出，跑不动降档时也会如实下降。
 *
 * ⚠️ 口径说明：downwardGazeRatio / attentionScatter 由 **头部姿态** 推算
 *   （本平台未接入虹膜注视跟踪），分别是「低头帧占比」和「头部偏航波动」，
 *   是注视方向的代理指标而非眼球注视方向。字段名沿用历史接口以保证兼容。
 *
 * 年龄差异化：不同年龄段眨眼频率基线不同（ageConfig.getEyeConfig）。
 */
import { useState, useCallback, useRef } from 'react';
import type { EyeMovementAnalysis } from '../types/multimodal.types';
import { defaultEyeMovement } from '../types/multimodal.types';
import type { AgeGroup } from './ageConfig';
import { getEyeConfig } from './ageConfig';
import { BLENDSHAPE_PARAMS, BlinkStateMachine, type BlinkMethod } from './blinkLogic';
import {
  DEFAULT_BLINK_FPS,
  blinkChannelQuality,
  durationQuantumMs,
  measuredFps,
} from './blinkChannel';
import type { FacialFrameSignal } from './useFacialAnalysis';

/** 面部帧信号中与本 Hook 相关的部分（不需要 rPPG 的绿色通道均值） */
export type EyeFrameSignal = Omit<FacialFrameSignal, 'greenMean'>;

/** 无 blendshape 时的 EAR 闭眼阈值（左眼关键点纵横比） */
const EAR_CLOSED_THRESHOLD = 0.15;
/** 统计窗口：与 BlinkStateMachine 的 60 秒滚动窗口一致 */
const WINDOW_MS = 60000;
/** 头部姿态历史的保留上限（30fps × 10 秒） */
const MAX_POSE_SAMPLES = 300;
/** blendshape / EAR 两种来源的信号质量基准分 */
const QUALITY_BASE: Record<'blendshape' | 'ear', number> = { blendshape: 0.9, ear: 0.5 };

export function useEyeTracking(ageGroup: AgeGroup | null = null) {
  const [metrics, setMetrics] = useState<EyeMovementAnalysis>({ ...defaultEyeMovement });
  const machineRef = useRef<BlinkStateMachine>(new BlinkStateMachine(BLENDSHAPE_PARAMS));
  const frameTimesRef = useRef<number[]>([]);
  const poseHistoryRef = useRef<{ yaw: number; pitch: number }[]>([]);
  const sessionStartRef = useRef(0);
  /** 上游上报的实测眨眼通道帧率（缺失时由本 Hook 自己按到帧时间估） */
  const reportedFpsRef = useRef(0);
  /**
   * 第二判定通道（浏览器内 ONNX 每眼分类器）的最近结果与滚动一致率。
   * ⚠️ **只上报，不参与 riskScore** —— 见 useEyeStateModel 的说明与
   * `docs/blink_signal_source_selection.md` 的步 B/C。
   */
  const eyeStateRef = useRef<{
    closedLeft: number;
    closedRight: number;
    closedBoth: number;
    latencyMs: number;
    status: string;
    samples: number;
    agree: number;
  }>({ closedLeft: 0, closedRight: 0, closedBoth: 0, latencyMs: 0, status: 'disabled', samples: 0, agree: 0 });
  const config = getEyeConfig(ageGroup);

  /**
   * 接收第二判定通道的结果。**只更新 ref**，实际发布在下一次
   * `updateFromFacialFrame` 里一起 setMetrics —— 避免一拍两次 React 更新。
   */
  const updateFromEyeStateModel = useCallback((r: {
    closedLeft: number;
    closedRight: number;
    closedBoth: number;
    latencyMs: number;
    blendshapeClosed: boolean;
    status?: string;
  }) => {
    const st = eyeStateRef.current;
    st.closedLeft = Math.round(r.closedLeft * 1000) / 1000;
    st.closedRight = Math.round(r.closedRight * 1000) / 1000;
    st.closedBoth = Math.round(r.closedBoth * 1000) / 1000;
    st.latencyMs = r.latencyMs;
    if (r.status) st.status = r.status;
    // 一致率 = 两路"是否闭眼"判定相同的比例（模型侧阈值取 0.5；上线前需按实测重标定）
    const modelClosed = r.closedBoth > 0.5;
    st.samples += 1;
    if (modelClosed === r.blendshapeClosed) st.agree += 1;
  }, []);

  /** 第二通道的加载状态变化（disabled/loading/ready/unavailable） */
  const setEyeStateStatus = useCallback((status: string) => {
    eyeStateRef.current.status = status;
  }, []);

  // 从面部分析帧更新眨眼 / 头姿指标（同一帧只推理一次）
  const updateFromFacialFrame = useCallback((frame: EyeFrameSignal) => {
    const now = Date.now();
    if (sessionStartRef.current === 0) sessionStartRef.current = now;

    const { eyeAspect, headPitch, headYaw, blinkScore } = frame;

    // ── 眨眼：闭眼判定 → 状态机（迟滞 + 时长窗 + 最小间隔）──
    // 连续分数一并发给状态机：插值默认关闭（见 BlinkParams.interpolateDurations 的实测说明），
    // 但传入分数让 `interpolatedShare` / `sampleIntervalMs` 能被如实统计出来。
    const method: BlinkMethod = blinkScore !== null ? 'blendshape' : 'ear';
    const closed =
      blinkScore !== null
        ? machineRef.current.isClosed(blinkScore)
        : eyeAspect < EAR_CLOSED_THRESHOLD;
    machineRef.current.update(closed, now, blinkScore ?? undefined);

    // ── 帧到达统计（供覆盖率与实测帧率）──
    const frameTimes = frameTimesRef.current;
    frameTimes.push(now);
    while (frameTimes.length > 0 && now - frameTimes[0] > WINDOW_MS) frameTimes.shift();

    if (typeof frame.blinkChannelFps === 'number' && frame.blinkChannelFps > 0) {
      reportedFpsRef.current = frame.blinkChannelFps;
    }
    const channelFps = reportedFpsRef.current > 0
      ? reportedFpsRef.current
      : measuredFps(frameTimes, 3000);

    const elapsedSec = Math.max(0, (now - sessionStartRef.current) / 1000);
    // 期望帧数按**实际目标帧率**算：原来写死 5fps，而实际是 10fps，
    // 于是 coverage 恒 ≥2 被夹到 1，signalQuality 实际只反映"来源基准分"。
    const effectiveTarget = channelFps > 0 ? channelFps : DEFAULT_BLINK_FPS;
    const expectedFrames = Math.max(1, Math.min(elapsedSec, WINDOW_MS / 1000) * effectiveTarget);
    const coverage = Math.min(1, frameTimes.length / expectedFrames);
    const qualityBase = method === 'blendshape' ? QUALITY_BASE.blendshape : QUALITY_BASE.ear;
    // 质量 = 来源基准 × 帧率达成率 × 覆盖率；跑到 15fps 就如实折半，而不是判成 0 或 1
    const signalQuality = blinkChannelQuality({
      sourceBase: qualityBase,
      measuredFps: channelFps,
      targetFps: DEFAULT_BLINK_FPS,
      coverage,
    });

    // ── 头部姿态历史（注视方向代理）──
    const poseHistory = poseHistoryRef.current;
    poseHistory.push({ yaw: headYaw, pitch: headPitch });
    if (poseHistory.length > MAX_POSE_SAMPLES) {
      poseHistory.splice(0, poseHistory.length - MAX_POSE_SAMPLES);
    }

    // 低头帧占比
    const downwardCount = poseHistory.filter(p => p.pitch > 10).length;
    const downwardRatio = poseHistory.length > 0 ? downwardCount / poseHistory.length : 0;

    // 头姿波动（偏航标准差）；不足 30 帧时不给值，避免把噪声当分散
    let scatter = 0;
    if (poseHistory.length > 30) {
      const yaws = poseHistory.map(p => p.yaw);
      const mean = yaws.reduce((a, b) => a + b, 0) / yaws.length;
      const variance = yaws.reduce((s, y) => s + (y - mean) ** 2, 0) / yaws.length;
      scatter = Math.min(1, Math.sqrt(variance) / 15);
    }

    const s = machineRef.current.stats(now, sessionStartRef.current);

    // ── 风险计算（公式与优化前逐项一致；本次只改采样与上报口径）──
    let riskScore = 0;
    if (s.blinkRate > 0 && s.blinkRate < config.lowBlinkThreshold) {
      riskScore += (config.lowBlinkThreshold - s.blinkRate) / config.lowBlinkThreshold * 0.35;
    } else if (s.blinkRate > config.highBlinkThreshold) {
      riskScore += Math.min(0.35, (s.blinkRate - config.highBlinkThreshold) / 15 * 0.35);
    }
    if (downwardRatio > 0.5) riskScore += Math.min(0.35, (downwardRatio - 0.5) / 0.3 * 0.35);
    if (scatter > 0.6) riskScore += Math.min(0.3, (scatter - 0.6) / 0.3 * 0.3);

    setMetrics({
      riskScore: Math.round(Math.min(1, riskScore) * 100) / 100,
      blinkRate: s.blinkRate,
      downwardGazeRatio: Math.round(downwardRatio * 100) / 100,
      attentionScatter: Math.round(scatter * 100) / 100,
      signalQuality,
      blinkMethod: method,
      avgBlinkDuration: s.avgBlinkDuration,
      longBlinkCount: s.longBlinkCount,
      blinkSampleCount: s.sampleCount,
      // ── 本次优化新增的「如实上报」字段 ──
      blinkChannelFps: Math.round(channelFps * 10) / 10,
      blinkDurationQuantumMs: durationQuantumMs(channelFps || DEFAULT_BLINK_FPS),
      blinkSampleIntervalMs: s.sampleIntervalMs,
      blinkInterpolatedShare: s.interpolatedShare,
      // ── 第二判定通道（只上报，不进 riskScore）──
      eyeStateModelStatus: eyeStateRef.current.status,
      eyeStateModelClosedBoth: eyeStateRef.current.closedBoth,
      eyeStateModelClosedLeft: eyeStateRef.current.closedLeft,
      eyeStateModelClosedRight: eyeStateRef.current.closedRight,
      eyeStateModelLatencyMs: eyeStateRef.current.latencyMs,
      eyeStateModelSamples: eyeStateRef.current.samples,
      eyeStateModelAgreement: eyeStateRef.current.samples > 0
        ? Math.round((eyeStateRef.current.agree / eyeStateRef.current.samples) * 100) / 100
        : 0,
      poseProxy: true,
      isMeasuring: true,
      timestamp: now,
    });
  }, [config]);

  const reset = useCallback(() => {
    machineRef.current.reset();
    frameTimesRef.current = [];
    poseHistoryRef.current = [];
    sessionStartRef.current = 0;
    reportedFpsRef.current = 0;
    eyeStateRef.current = {
      closedLeft: 0, closedRight: 0, closedBoth: 0, latencyMs: 0,
      status: 'disabled', samples: 0, agree: 0,
    };
    setMetrics({ ...defaultEyeMovement });
  }, []);

  return { metrics, updateFromFacialFrame, updateFromEyeStateModel, setEyeStateStatus, reset };
}
