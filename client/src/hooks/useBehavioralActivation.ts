/**
 * 行为激活水平 Hook —— 数字表型行为监测
 * 年龄差异化：不同年龄段社交需求和探索基线不同
 */
import { useState, useCallback, useRef } from 'react';
import type { BehavioralActivationAnalysis } from '../types/multimodal.types';
import { defaultBehavioralActivation } from '../types/multimodal.types';
import type { AgeGroup } from './ageConfig';
import { getBehavioralActConfig } from './ageConfig';

export function useBehavioralActivation(ageGroup: AgeGroup | null = null) {
  const [metrics, setMetrics] = useState<BehavioralActivationAnalysis>({ ...defaultBehavioralActivation });
  const interactionCountRef = useRef(0);
  const visitedModulesRef = useRef<Set<string>>(new Set());
  const dailyHistoryRef = useRef<number[]>([]);
  const config = getBehavioralActConfig(ageGroup);

  const recordInteraction = useCallback(() => {
    interactionCountRef.current++;
  }, []);

  const recordModuleVisit = useCallback((moduleName: string) => {
    visitedModulesRef.current.add(moduleName);
  }, []);

  const updateMetrics = useCallback(() => {
    const count = interactionCountRef.current;
    const explorationRate = visitedModulesRef.current.size / 10; // 假设 10 个功能模块

    // 社交退缩评估
    let withdrawalScore = 0;
    if (count < config.minDailyInteractions) {
      withdrawalScore = Math.min(1, (1 - count / config.minDailyInteractions) * 0.7);
    }

    // 趋势分析
    dailyHistoryRef.current.push(count);
    if (dailyHistoryRef.current.length > 7) dailyHistoryRef.current = dailyHistoryRef.current.slice(-7);
    let trendDirection: 'improving' | 'stable' | 'declining' = 'stable';
    if (dailyHistoryRef.current.length >= 4) {
      const mid = Math.floor(dailyHistoryRef.current.length / 2);
      const firstHalf = dailyHistoryRef.current.slice(0, mid).reduce((a, b) => a + b, 0) / mid;
      const secondHalf = dailyHistoryRef.current.slice(mid).reduce((a, b) => a + b, 0) / (dailyHistoryRef.current.length - mid);
      if (firstHalf > 0) {
        const change = (secondHalf - firstHalf) / firstHalf;
        if (change < -0.3) trendDirection = 'declining';
        else if (change > 0.3) trendDirection = 'improving';
      }
    }

    const riskScore = Math.min(1,
      withdrawalScore * 0.4
      + (explorationRate < config.explorationRateBaseline * 0.5 ? 0.3 : 0)
      + (trendDirection === 'declining' ? 0.2 : 0)
    );

    setMetrics({
      riskScore: Math.round(riskScore * 100) / 100,
      dailyInteractionCount: count,
      explorationRate: Math.round(explorationRate * 100) / 100,
      socialWithdrawalScore: Math.round(withdrawalScore * 100) / 100,
      trendDirection,
      isActive: true,
      timestamp: Date.now(),
    });
  }, [config]);

  const reset = useCallback(() => {
    interactionCountRef.current = 0;
    visitedModulesRef.current.clear();
    dailyHistoryRef.current = [];
    setMetrics({ ...defaultBehavioralActivation });
  }, []);

  return { metrics, recordInteraction, recordModuleVisit, updateMetrics, reset };
}
