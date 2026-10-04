/**
 * 每眼「睁闭眼」第二判定通道（浏览器内 ONNX）
 *
 * 位置与职责：
 *   `useFacialAnalysis`（指标拍，10fps）裁出左右眼 patch
 *     → 本 Hook 跑 open-closed-eye-0001（46 KB，Apache-2.0）
 *     → 得到每眼闭眼概率，与同帧的 MediaPipe blendshape 判定做**一致性统计**
 *     → 由 `useEyeTracking` 只做**上报**（`eyeStateModel*` 字段），**不参与 riskScore**
 *
 * 为什么先只上报：
 *   模型训练域是近红外灰度眼图（MRL Eye），在本平台 RGB 摄像头画面上存在域偏移，
 *   实测 AUC 0.92–0.97 但没有任何阈值能让闭眼召回达到 0.9。因此必须先采集
 *   真实设备上的分数分布来标定阈值，再谈让它影响风险 —— 详见
 *   `docs/blink_signal_source_selection.md` 的步 B/C（后者需人工审核）。
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import type { EyePatchPixels } from './eyePatchCanvas';
import { EyeStateClassifier, type EyeStateResult } from '../services/eyeStateClassifier';

export type EyeStateModelStatus = 'disabled' | 'loading' | 'ready' | 'unavailable';

export interface EyeStateModelReport extends EyeStateResult {
  /** 同帧 MediaPipe blendshape 是否判为闭眼（用于一致率） */
  blendshapeClosed: boolean;
}

export interface UseEyeStateModelResult {
  status: EyeStateModelStatus;
  /** 是否已加载好模型（可显示"第二意见：可用"） */
  isReady: boolean;
  /** 供 `useFacialAnalysis` 的 `onEyePatchFrame` 直接使用 */
  onEyePatches: (patches: EyePatchPixels) => void;
  /** 供 `onFrameSignal` 调用，记录同帧的 blendshape 判定 */
  observeBlendshape: (closed: boolean) => void;
  /** 最近一次成功结果（可能为 null） */
  latest: EyeStateResult | null;
}

/** blendshape 判"闭眼"用的阈值 —— 与线上状态机的 onThreshold 一致 */
const BLENDSHAPE_CLOSED_THRESHOLD = 0.5;

export function useEyeStateModel(
  enabled: boolean,
  onReport?: ((r: EyeStateModelReport) => void) | null,
): UseEyeStateModelResult {
  const [status, setStatus] = useState<EyeStateModelStatus>(enabled ? 'loading' : 'disabled');
  const [latest, setLatest] = useState<EyeStateResult | null>(null);
  const classifierRef = useRef<EyeStateClassifier | null>(null);
  /** 同帧 blendshape 判定（在 onFrameSignal 里更新，patch 回调里读取） */
  const blendshapeClosedRef = useRef<boolean | null>(null);
  /** 串行化推理：patch 以 10fps 到来，若上一拍还没跑完就直接丢帧，不排队 */
  const inFlightRef = useRef(false);
  /** 上报回调（放在 ref 里，避免调用方每次渲染换函数导致推理闭包抖动） */
  const reportRef = useRef<((r: EyeStateModelReport) => void) | null>(null);
  reportRef.current = onReport ?? null;

  useEffect(() => {
    if (!enabled) {
      setStatus('disabled');
      return undefined;
    }
    const classifier = new EyeStateClassifier();
    classifierRef.current = classifier;
    let cancelled = false;
    classifier.load().then((ok) => {
      if (!cancelled) setStatus(ok ? 'ready' : 'unavailable');
    });
    return () => {
      cancelled = true;
      classifier.dispose();
      classifierRef.current = null;
    };
  }, [enabled]);

  const onEyePatches = useCallback((patches: EyePatchPixels) => {
    const classifier = classifierRef.current;
    if (!classifier || !classifier.isReady) return;
    // 越界（眼部 patch 出画面）时不予采信：画布会补黑边，判出来的分数没有意义
    if (!patches.fullyInsideFrame) return;
    if (inFlightRef.current) return;
    inFlightRef.current = true;

    const blendshapeClosed = blendshapeClosedRef.current ?? false;
    classifier
      .classify(patches.left, patches.right)
      .then((result) => {
        if (!result) return;
        setLatest(result);
        reportRef.current?.({ ...result, blendshapeClosed });
      })
      .finally(() => {
        inFlightRef.current = false;
      });
  }, []);

  const observeBlendshape = useCallback((closed: boolean) => {
    blendshapeClosedRef.current = closed;
  }, []);

  return { status, isReady: status === 'ready', onEyePatches, observeBlendshape, latest };
}

/** 供上层直接判断 blendshape 分数是否算"闭眼" */
export function isBlendshapeClosed(score: number | null | undefined): boolean {
  return typeof score === 'number' && Number.isFinite(score) && score > BLENDSHAPE_CLOSED_THRESHOLD;
}
