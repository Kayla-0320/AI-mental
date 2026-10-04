/**
 * 认知扭曲深度分析 Hook —— 基于 CBT 理论的七类认知扭曲检测
 * 年龄差异化：不同年龄段的认知扭曲表达关键词不同
 */
import { useState, useCallback } from 'react';
import type { CognitiveDistortionAnalysis } from '../types/multimodal.types';
import { defaultCognitiveDistortion } from '../types/multimodal.types';
import type { AgeGroup } from './ageConfig';
import { getCognitiveConfig } from './ageConfig';

export function useCognitiveDistortion(ageGroup: AgeGroup | null = null) {
  const [metrics, setMetrics] = useState<CognitiveDistortionAnalysis>({ ...defaultCognitiveDistortion });
  const config = getCognitiveConfig(ageGroup);

  const analyzeText = useCallback((text: string) => {
    if (!text.trim()) return;
    const t = text.toLowerCase();
    const maxPerCat = 0.15;

    const countMatches = (keywords: string[]) => {
      const matched = keywords.filter(kw => t.includes(kw));
      return Math.min(maxPerCat, matched.length * maxPerCat / 3);
    };

    const categories = {
      catastrophizing: countMatches(config.catastrophizing),
      blackAndWhite: countMatches(config.blackAndWhite),
      overgeneralization: countMatches(['每次都', '从来不', '所有人', '没有人', '到处']),
      selfBlame: countMatches(config.selfBlame),
      hopelessness: countMatches(config.hopelessness),
      mindReading: countMatches(config.mindReading),
      shouldStatements: countMatches(config.shouldStatements),
    };

    const riskScore = Math.min(1, Object.values(categories).reduce((s, v) => s + v, 0));
    const totalCount = Object.values(categories).filter(v => v > 0).length;

    setMetrics({
      riskScore: Math.round(riskScore * 100) / 100,
      categories,
      totalDistortionCount: totalCount,
      timestamp: Date.now(),
    });
  }, [config]);

  const reset = useCallback(() => setMetrics({ ...defaultCognitiveDistortion }), []);
  return { metrics, analyzeText, reset };
}
