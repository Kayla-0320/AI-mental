/**
 * 眼动模式分析 Hook —— 通过面部关键点追踪眨眼和注视方向
 * 年龄差异化：不同年龄段眨眼频率基线不同
 * 注：此 Hook 从 useFacialAnalysis 的面部关键点中提取眼动指标
 */
import { useState, useCallback, useRef } from 'react';
import type { EyeMovementAnalysis } from '../types/multimodal.types';
import { defaultEyeMovement } from '../types/multimodal.types';
import type { AgeGroup } from './ageConfig';
import { getEyeConfig } from './ageConfig';

export function useEyeTracking(ageGroup: AgeGroup | null = null) {
  const [metrics, setMetrics] = useState<EyeMovementAnalysis>({ ...defaultEyeMovement });
  const blinkTimestampsRef = useRef<number[]>([]);
  const gazeHistoryRef = useRef<{ x: number; y: number }[]>([]);
  const prevEyeAspectRef = useRef(0.3);
  const config = getEyeConfig(ageGroup);

  // 从面部分析帧更新眼动指标
  const updateFromFacialFrame = useCallback((eyeAspect: number, headPitch: number) => {
    const now = Date.now();

    // 眨眼检测：eyeAspect 突然下降后恢复
    if (eyeAspect < 0.15 && prevEyeAspectRef.current > 0.2) {
      blinkTimestampsRef.current.push(now);
    }
    prevEyeAspectRef.current = eyeAspect;

    // 保留最近 60 秒的眨眼记录
    const cutoff = now - 60000;
    blinkTimestampsRef.current = blinkTimestampsRef.current.filter(t => t > cutoff);

    // 眨眼频率
    const blinkRate = blinkTimestampsRef.current.length; // 次/分钟

    // 注视方向（基于头部姿态估计）
    gazeHistoryRef.current.push({ x: headPitch, y: headPitch });
    if (gazeHistoryRef.current.length > 300) gazeHistoryRef.current = gazeHistoryRef.current.slice(-300);

    // 向下注视比例
    const downwardCount = gazeHistoryRef.current.filter(g => g.y > 10).length;
    const downwardRatio = gazeHistoryRef.current.length > 0
      ? downwardCount / gazeHistoryRef.current.length : 0;

    // 注意力分散度
    let scatter = 0;
    if (gazeHistoryRef.current.length > 30) {
      const xs = gazeHistoryRef.current.map(g => g.x);
      const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
      const variance = xs.reduce((s, x) => s + (x - mean) ** 2, 0) / xs.length;
      scatter = Math.min(1, Math.sqrt(variance) / 15);
    }

    // 风险计算
    let riskScore = 0;
    if (blinkRate > 0 && blinkRate < config.lowBlinkThreshold) {
      riskScore += (config.lowBlinkThreshold - blinkRate) / config.lowBlinkThreshold * 0.35;
    } else if (blinkRate > config.highBlinkThreshold) {
      riskScore += Math.min(0.35, (blinkRate - config.highBlinkThreshold) / 15 * 0.35);
    }
    if (downwardRatio > 0.5) riskScore += Math.min(0.35, (downwardRatio - 0.5) / 0.3 * 0.35);
    if (scatter > 0.6) riskScore += Math.min(0.3, (scatter - 0.6) / 0.3 * 0.3);

    setMetrics({
      riskScore: Math.round(Math.min(1, riskScore) * 100) / 100,
      blinkRate,
      downwardGazeRatio: Math.round(downwardRatio * 100) / 100,
      attentionScatter: Math.round(scatter * 100) / 100,
      signalQuality: Math.round(Math.min(1, blinkTimestampsRef.current.length > 0 ? 0.7 : 0.2) * 100) / 100,
      isMeasuring: true,
      timestamp: now,
    });
  }, [config]);

  const reset = useCallback(() => {
    blinkTimestampsRef.current = [];
    gazeHistoryRef.current = [];
    prevEyeAspectRef.current = 0.3;
    setMetrics({ ...defaultEyeMovement });
  }, []);

  return { metrics, updateFromFacialFrame, reset };
}
