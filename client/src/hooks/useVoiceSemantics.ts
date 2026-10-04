/**
 * 语音深层语义 Hook —— 对语音识别文本的深度语言分析
 * 分析维度：第一人称代词密度、绝对化表达、自我指涉、消极情感词比例
 * 年龄差异化：不同年龄段的语言模式阈值不同
 */
import { useState, useCallback } from 'react';
import type { VoiceSemanticsAnalysis } from '../types/multimodal.types';
import { defaultVoiceSemantics } from '../types/multimodal.types';
import type { AgeGroup } from './ageConfig';

export function useVoiceSemantics(ageGroup: AgeGroup | null = null) {
  const [metrics, setMetrics] = useState<VoiceSemanticsAnalysis>({ ...defaultVoiceSemantics });

  const analyzeSpeechText = useCallback((text: string) => {
    if (!text || text.trim().length < 5) return;
    const t = text;
    const len = Math.max(t.length, 1);

    // 第一人称单数（"我"但不含"我们"）
    const fpsMatches = t.match(/我(?!们)/g) || [];
    const fpsRatio = fpsMatches.length / len;

    // 第一人称复数
    const fppMatches = t.match(/我们/g) || [];
    const fppRatio = fppMatches.length / len;

    // 绝对化表达
    const absMatches = t.match(/总是|从不|永远|绝对|一定|所有|全部|完全|根本|每次|必须|应该/g) || [];
    const absRatio = absMatches.length / len;

    // 自我指涉
    const selfMatches = t.match(/我|自己|本人|我的|俺|咱/g) || [];
    const selfDensity = selfMatches.length / len;

    // 消极/积极情感词
    const negMatches = t.match(/难过|伤心|焦虑|害怕|恐惧|愤怒|痛苦|绝望|孤独|寂寞|空虚|无聊|烦|累|疲|丧|衰|暗|灰|冷|痛|哭|死|恨|厌/g) || [];
    const posMatches = t.match(/开心|快乐|高兴|幸福|棒|好|喜欢|爱|感恩|温暖|阳光|笑|甜|美|亮|暖/g) || [];
    const totalAffect = negMatches.length + posMatches.length;
    const negRatio = totalAffect > 0 ? negMatches.length / totalAffect : 0;

    // 风险计算
    const thresholds = ageGroup === 'young_adult'
      ? { fps: 0.12, abs: 0.04, self: 0.20, neg: 0.18 }
      : { fps: 0.15, abs: 0.05, self: 0.25, neg: 0.20 };

    let riskScore = 0;
    if (fpsRatio > thresholds.fps) riskScore += Math.min(0.25, (fpsRatio - thresholds.fps) / 0.1 * 0.25);
    if (fppRatio < 0.02 && len > 20) riskScore += 0.1;
    if (absRatio > thresholds.abs) riskScore += Math.min(0.15, (absRatio - thresholds.abs) / 0.03 * 0.15);
    if (selfDensity > thresholds.self) riskScore += Math.min(0.25, (selfDensity - thresholds.self) / 0.15 * 0.25);
    if (negRatio > thresholds.neg && totalAffect > 0) riskScore += Math.min(0.25, (negRatio - thresholds.neg) / 0.3 * 0.25);

    setMetrics({
      riskScore: Math.round(Math.min(1, riskScore) * 100) / 100,
      firstPersonSingularRatio: Math.round(fpsRatio * 10000) / 10000,
      firstPersonPluralRatio: Math.round(fppRatio * 10000) / 10000,
      absolutistRatio: Math.round(absRatio * 10000) / 10000,
      selfReferentialDensity: Math.round(selfDensity * 10000) / 10000,
      negativeAffectRatio: Math.round(negRatio * 100) / 100,
      timestamp: Date.now(),
    });
  }, [ageGroup]);

  const reset = useCallback(() => setMetrics({ ...defaultVoiceSemantics }), []);
  return { metrics, analyzeSpeechText, reset };
}
