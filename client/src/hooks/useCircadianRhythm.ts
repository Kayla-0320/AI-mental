/**
 * 昼夜节律分析 Hook —— 基于时间戳的活动模式监测
 *
 * ⚠️ 本模块**不是睡眠检测**：它只记录用户在 App 内产生活动的时间点，
 * 用于提示「深夜仍在使用」和「使用时间是否规律」。
 * 睡眠时长、入睡时间、睡眠质量一律不由此推导（见 perceptionCapabilities.ts）。
 *
 * 分析维度：深夜活动风险、活动时间规律性
 * 年龄差异化：不同年龄段的理想就寝时间不同
 */
import { useState, useCallback, useRef, useEffect } from 'react';
import type { CircadianAnalysis } from '../types/multimodal.types';
import { defaultCircadian } from '../types/multimodal.types';
import type { AgeGroup } from './ageConfig';
import { getCircadianConfig } from './ageConfig';

export function useCircadianRhythm(ageGroup: AgeGroup | null = null) {
  const [metrics, setMetrics] = useState<CircadianAnalysis>({ ...defaultCircadian });
  const activityTimestampsRef = useRef<number[]>([]);
  const config = getCircadianConfig(ageGroup);

  const recordActivity = useCallback(() => {
    const now = Date.now();
    activityTimestampsRef.current.push(now);
    // 保留最近 200 条
    if (activityTimestampsRef.current.length > 200) {
      activityTimestampsRef.current = activityTimestampsRef.current.slice(-200);
    }
  }, []);

  const updateMetrics = useCallback(() => {
    const now = new Date();
    const hour = now.getHours() + now.getMinutes() / 60;
    const timestamps = activityTimestampsRef.current;

    // 深夜风险
    let lateNightRisk = 0;
    if (hour >= config.lateActivityRiskHour || (hour < 5 && hour < 6.5)) {
      const hoursPast = hour >= config.lateActivityRiskHour
        ? hour - config.lateActivityRiskHour
        : (24 - config.lateActivityRiskHour) + hour;
      lateNightRisk = Math.min(1, 0.3 + hoursPast * 0.1);
    }

    // 活动时间规律性（基于最近活动时间分布）
    // 样本不足 10 条时不给结论，避免用两三个点算出一个"规律性百分比"
    const MIN_SAMPLES_FOR_REGULARITY = 10;
    let regularityScore = 0.5;
    let hasRegularity = false;
    if (timestamps.length >= MIN_SAMPLES_FOR_REGULARITY) {
      const hours = timestamps.map(t => new Date(t).getHours() + new Date(t).getMinutes() / 60);
      const mean = hours.reduce((a, b) => a + b, 0) / hours.length;
      const variance = hours.reduce((s, h) => s + (h - mean) ** 2, 0) / hours.length;
      const std = Math.sqrt(variance);
      regularityScore = Math.max(0, Math.min(1, 1 - std / 6));
      hasRegularity = true;
    }

    // 缺少规律性样本时，只保留"深夜活动"这一个可判定事实
    const riskScore = hasRegularity
      ? Math.min(1, lateNightRisk * 0.6 + (1 - regularityScore) * 0.3)
      : Math.min(1, lateNightRisk * 0.6);

    setMetrics({
      riskScore: Math.round(riskScore * 100) / 100,
      lateNightRisk: Math.round(lateNightRisk * 100) / 100,
      regularityScore: Math.round(regularityScore * 100) / 100,
      lastActivityHour: Math.round(hour * 10) / 10,
      activitySampleCount: timestamps.length,
      isActive: timestamps.length > 0,
      timestamp: Date.now(),
    });
  }, [config]);

  useEffect(() => {
    const interval = setInterval(updateMetrics, 60000); // 每分钟更新
    return () => clearInterval(interval);
  }, [updateMetrics]);

  const reset = useCallback(() => {
    activityTimestampsRef.current = [];
    setMetrics({ ...defaultCircadian });
  }, []);

  return { metrics, recordActivity, reset };
}
