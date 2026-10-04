/**
 * 行为激活水平 Hook —— 数字表型行为监测（仅统计平台内互动）
 * 年龄差异化：不同年龄段社交需求和探索基线不同
 *
 * ⚠️ 这里的「互动」只指平台内操作次数，不代表真实社交状况；
 * 没有互动时不声称正在采集，也不把单日数据当成趋势。
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
  const lastDayRef = useRef<string>('');
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

    // 社交退缩评估（仅基于平台内互动次数，不等同于真实社交状况）
    let withdrawalScore = 0;
    if (count < config.minDailyInteractions) {
      withdrawalScore = Math.min(1, (1 - count / config.minDailyInteractions) * 0.7);
    }

    // 趋势分析：必须有跨天记录才谈趋势，且不把"今天更少"直接当成社交退缩
    const todayKey = new Date().toDateString();
    if (lastDayRef.current !== todayKey) {
      lastDayRef.current = todayKey;
      dailyHistoryRef.current.push(count);
      if (dailyHistoryRef.current.length > 7) dailyHistoryRef.current = dailyHistoryRef.current.slice(-7);
    }
    let trendDirection: 'improving' | 'stable' | 'declining' = 'stable';
    // 至少 4 天记录才判定趋势，否则同一会话内的累计值会伪造出"下降趋势"
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
      // 没有任何平台内互动时不得声称"行为激活采集中"
      isActive: count > 0,
      timestamp: Date.now(),
    });
  }, [config]);

  const reset = useCallback(() => {
    interactionCountRef.current = 0;
    visitedModulesRef.current.clear();
    dailyHistoryRef.current = [];
    lastDayRef.current = '';
    setMetrics({ ...defaultBehavioralActivation });
  }, []);

  return { metrics, recordInteraction, recordModuleVisit, updateMetrics, reset };
}
