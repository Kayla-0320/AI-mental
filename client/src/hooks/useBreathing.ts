/**
 * 呼吸模式分析 Hook —— 通过麦克风能量变化检测呼吸节律
 * 年龄差异化：不同年龄段正常呼吸频率不同
 */
import { useState, useCallback, useRef } from 'react';
import type { BreathingAnalysis } from '../types/multimodal.types';
import { defaultBreathing } from '../types/multimodal.types';
import type { AgeGroup } from './ageConfig';
import { getBreathingConfig } from './ageConfig';

export function useBreathing(ageGroup: AgeGroup | null = null) {
  const [metrics, setMetrics] = useState<BreathingAnalysis>({ ...defaultBreathing });
  const energyHistoryRef = useRef<number[]>([]);
  const sighCountRef = useRef(0);
  const config = getBreathingConfig(ageGroup);

  const updateFromAudioEnergy = useCallback((rmsEnergy: number) => {
    energyHistoryRef.current.push(rmsEnergy);
    if (energyHistoryRef.current.length > 600) energyHistoryRef.current = energyHistoryRef.current.slice(-600);

    const history = energyHistoryRef.current;
    if (history.length < 60) return; // 需要至少 6 秒数据

    // 检测呼吸周期：能量波动的峰谷交替
    const smoothed = history.map((_, i) => {
      const start = Math.max(0, i - 5);
      const slice = history.slice(start, i + 1);
      return slice.reduce((a, b) => a + b, 0) / slice.length;
    });

    // 计算过零率（呼吸周期指标）
    const mean = smoothed.reduce((a, b) => a + b, 0) / smoothed.length;
    let crossings = 0;
    for (let i = 1; i < smoothed.length; i++) {
      if ((smoothed[i] - mean) * (smoothed[i-1] - mean) < 0) crossings++;
    }

    // 呼吸频率估计（假设 10fps 采样）
    const breathsPerMinute = (crossings / 2) / (smoothed.length / 10) * 60;

    // 规律性
    const intervals: number[] = [];
    let lastCrossing = 0;
    for (let i = 1; i < smoothed.length; i++) {
      if ((smoothed[i] - mean) * (smoothed[i-1] - mean) < 0) {
        if (lastCrossing > 0) intervals.push(i - lastCrossing);
        lastCrossing = i;
      }
    }
    const intervalMean = intervals.length > 0 ? intervals.reduce((a, b) => a + b, 0) / intervals.length : 1;
    const cv = intervals.length > 1
      ? Math.sqrt(intervals.reduce((s, v) => s + (v - intervalMean) ** 2, 0) / intervals.length) / intervalMean
      : 0;

    // 叹气检测（突然的大幅度能量下降后恢复）
    for (let i = 10; i < history.length; i++) {
      const before = history.slice(i - 10, i).reduce((a, b) => a + b, 0) / 10;
      const during = history[i];
      if (during < before * 0.3 && before > 0.01) sighCountRef.current++;
    }

    // 风险计算
    let riskScore = 0;
    if (breathsPerMinute > config.anxietyRateThreshold) {
      riskScore += Math.min(0.5, (breathsPerMinute - config.anxietyRateThreshold) / 8);
    } else if (breathsPerMinute > 0 && breathsPerMinute < config.depressionRateThreshold) {
      riskScore += Math.min(0.5, (config.depressionRateThreshold - breathsPerMinute) / 5);
    }
    if (cv > 0.25) riskScore += Math.min(0.3, (cv - 0.25) / 0.3 * 0.3);

    const signalQuality = Math.min(1, history.reduce((s, v) => s + v, 0) / history.length * 100);

    setMetrics({
      riskScore: Math.round(Math.min(1, riskScore) * 100) / 100,
      breathingRate: Math.round(breathsPerMinute * 10) / 10,
      regularityCV: Math.round(cv * 100) / 100,
      sighCount: sighCountRef.current,
      signalQuality: Math.round(signalQuality * 100) / 100,
      isMeasuring: true,
      timestamp: Date.now(),
    });
  }, [config]);

  const reset = useCallback(() => {
    energyHistoryRef.current = [];
    sighCountRef.current = 0;
    setMetrics({ ...defaultBreathing });
  }, []);

  return { metrics, updateFromAudioEnergy, reset };
}
