/**
 * 干预效果闭环 Hook —— 量化干预前后的多模态变化
 *
 * 临床依据：
 * - Martell et al. (2001) —— 行为激活治疗的效果评估
 * - CBT 认知行为疗法 —— 干预前后认知指标对比
 *
 * 工作流程：
 * 1. 检测到风险 → 自动拍摄基线快照
 * 2. 触发干预（CBT/正念/呼吸练习）
 * 3. 干预结束后再次拍摄
 * 4. 计算各维度改善幅度 → 生成疗效反馈
 */
import { useState, useCallback, useRef } from 'react';

/** 干预基线快照 */
export interface InterventionBaseline {
  timestamp: number;
  /** 融合情绪概率 [快乐, 悲伤, 焦虑, 愤怒, 中性] */
  emotionProbs: number[];
  /** 综合风险分 0-1 */
  riskScore: number;
  /** 各模态风险分 */
  modalityRisks: {
    text: number;
    voice: number;
    facial: number;
    keyboard: number;
    circadian: number;
    cognitive: number;
    hrv: number;
    breathing: number;
    behavioralAct: number;
    eye: number;
    voiceSemantics: number;
  };
  /** 主导情绪 */
  dominantEmotion: string;
  /** 干预类型 */
  interventionType: string;
}

/** 干预效果报告 */
export interface InterventionReport {
  /** 干预持续时间 (秒) */
  durationSeconds: number;
  /** 各维度变化 (正=改善, 负=恶化) */
  deltas: {
    riskScore: number;
    anxietyProb: number;
    sadnessProb: number;
    dominantEmotionShift: string;
    /** 改善最显著的模态 */
    topImprovedModality: string;
    /** 改善幅度 */
    topImprovement: number;
  };
  /** 整体效果评级 */
  effectiveness: 'significant' | 'moderate' | 'minimal' | 'worsened';
  /** 温暖反馈文案 */
  feedbackMessage: string;
  baseline: InterventionBaseline;
  postIntervention: InterventionBaseline;
}

/** 单次干预记录（用于历史统计） */
export interface InterventionRecord {
  timestamp: number;
  interventionType: string;      // 'breathing' | 'cbt' | 'reality_task' | 'mindfulness' | ...
  preAnxiety: number;            // 干预前焦虑概率
  postAnxiety: number;           // 干预后焦虑概率
  delta: number;                 // 变化量（正=改善）
  effectiveness: 'significant' | 'moderate' | 'minimal' | 'worsened';
}

/** 干预历史统计 */
export interface InterventionHistory {
  records: InterventionRecord[];   // 最近 30 条
  effectivenessByType: Record<string, { avg: number; count: number }>;
  recommendedType: string;          // 最有效的干预类型
}

const HISTORY_STORAGE_KEY = 'intervention_history';
const MAX_HISTORY_RECORDS = 30;

/** localStorage 持久化 */
function loadHistory(): InterventionRecord[] {
  try {
    const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) return parsed;
    }
  } catch {}
  return [];
}

function saveHistory(records: InterventionRecord[]): void {
  try {
    localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(records));
  } catch {}
}

/** 计算干预历史统计 */
function computeHistoryStats(records: InterventionRecord[]): InterventionHistory {
  const byType: Record<string, { total: number; count: number }> = {};
  for (const r of records) {
    if (!byType[r.interventionType]) byType[r.interventionType] = { total: 0, count: 0 };
    byType[r.interventionType].total += r.delta;
    byType[r.interventionType].count += 1;
  }
  const effectivenessByType: Record<string, { avg: number; count: number }> = {};
  let bestType = '';
  let bestAvg = -Infinity;
  for (const [type, { total, count }] of Object.entries(byType)) {
    const avg = count > 0 ? total / count : 0;
    effectivenessByType[type] = { avg: Math.round(avg * 100) / 100, count };
    if (avg > bestAvg && count >= 2) { bestAvg = avg; bestType = type; }
  }
  return { records, effectivenessByType, recommendedType: bestType };
}

export function useInterventionFeedback() {
  const [baseline, setBaseline] = useState<InterventionBaseline | null>(null);
  const [lastReport, setLastReport] = useState<InterventionReport | null>(null);
  const [history, setHistory] = useState<InterventionRecord[]>(loadHistory);
  const interventionStartRef = useRef<number>(0);

  /** 拍摄干预前基线 */
  const captureBaseline = useCallback((
    emotionProbs: number[],
    riskScore: number,
    modalityRisks: InterventionBaseline['modalityRisks'],
    dominantEmotion: string,
    interventionType: string = 'general',
  ) => {
    const snap: InterventionBaseline = {
      timestamp: Date.now(),
      emotionProbs: [...emotionProbs],
      riskScore,
      modalityRisks: { ...modalityRisks },
      dominantEmotion,
      interventionType,
    };
    setBaseline(snap);
    interventionStartRef.current = Date.now();
    console.log('[干预闭环] 基线快照已拍摄:', snap);
    return snap;
  }, []);

  /** 计算干预效果 */
  const computeReport = useCallback((
    currentEmotionProbs: number[],
    currentRiskScore: number,
    currentModalityRisks: InterventionBaseline['modalityRisks'],
    currentDominantEmotion: string,
  ): InterventionReport | null => {
    if (!baseline) return null;

    const durationSeconds = Math.round((Date.now() - interventionStartRef.current) / 1000);

    // 计算各维度变化
    const riskDelta = baseline.riskScore - currentRiskScore; // 正=改善
    const anxietyDelta = baseline.emotionProbs[2] - currentEmotionProbs[2]; // 正=改善
    const sadnessDelta = baseline.emotionProbs[1] - currentEmotionProbs[1];

    // 找改善最显著的模态
    let topModality = 'none';
    let topImprovement = 0;
    const modalityEntries = Object.entries(currentModalityRisks) as [string, number][];
    const baselineEntries = Object.entries(baseline.modalityRisks) as [string, number][];
    for (let i = 0; i < modalityEntries.length; i++) {
      const [key, currentVal] = modalityEntries[i];
      const [, baselineVal] = baselineEntries[i];
      const improvement = baselineVal - currentVal;
      if (improvement > topImprovement) {
        topImprovement = improvement;
        topModality = key;
      }
    }

    // 情绪转换描述
    const emotionShift = baseline.dominantEmotion !== currentDominantEmotion
      ? `${baseline.dominantEmotion} → ${currentDominantEmotion}`
      : `维持 ${currentDominantEmotion}`;

    // 效果评级
    const avgImprovement = (riskDelta + anxietyDelta) / 2;
    let effectiveness: InterventionReport['effectiveness'];
    if (avgImprovement > 0.2) effectiveness = 'significant';
    else if (avgImprovement > 0.1) effectiveness = 'moderate';
    else if (avgImprovement > 0) effectiveness = 'minimal';
    else effectiveness = 'worsened';

    // 温暖反馈文案
    const feedbackMessages: Record<string, string[]> = {
      significant: [
        '干预效果非常显著！你的状态有了明显的改善，继续加油 💪',
        '你做得很棒！这些练习对你的帮助很大，记住这种感觉 🌟',
      ],
      moderate: [
        '干预有一定效果，你的状态在慢慢好转 🌱',
        '每一步都算数，你的内心正在慢慢平静下来 🍃',
      ],
      minimal: [
        '改变需要时间，你已经在努力了，这本身就很重要 🌸',
        '也许效果还不明显，但请相信每一次尝试都有意义 💛',
      ],
      worsened: [
        '有时候情绪会反复，这不是你的错。试试换一种方式，或者找信任的人聊聊 🤗',
        '感到困难是正常的，你不需要独自面对。深呼吸，慢慢来 🌈',
      ],
    };
    const msgs = feedbackMessages[effectiveness];
    const feedbackMessage = msgs[Math.floor(Math.random() * msgs.length)];

    const postSnap: InterventionBaseline = {
      timestamp: Date.now(),
      emotionProbs: [...currentEmotionProbs],
      riskScore: currentRiskScore,
      modalityRisks: { ...currentModalityRisks },
      dominantEmotion: currentDominantEmotion,
      interventionType: baseline.interventionType,
    };

    const report: InterventionReport = {
      durationSeconds,
      deltas: {
        riskScore: Math.round(riskDelta * 100) / 100,
        anxietyProb: Math.round(anxietyDelta * 100) / 100,
        sadnessProb: Math.round(sadnessDelta * 100) / 100,
        dominantEmotionShift: emotionShift,
        topImprovedModality: topModality,
        topImprovement: Math.round(topImprovement * 100) / 100,
      },
      effectiveness,
      feedbackMessage,
      baseline,
      postIntervention: postSnap,
    };

    setLastReport(report);
    console.log('[干预闭环] 效果报告:', report);

    // 记录到干预历史
    const record: InterventionRecord = {
      timestamp: Date.now(),
      interventionType: baseline.interventionType,
      preAnxiety: baseline.emotionProbs[2],
      postAnxiety: currentEmotionProbs[2],
      delta: Math.round(anxietyDelta * 100) / 100,
      effectiveness,
    };
    setHistory(prev => {
      const next = [...prev, record].slice(-MAX_HISTORY_RECORDS);
      saveHistory(next);
      return next;
    });

    return report;
  }, [baseline]);

  const reset = useCallback(() => {
    setBaseline(null);
    setLastReport(null);
    interventionStartRef.current = 0;
  }, []);

  /** 获取干预历史统计 */
  const getHistory = useCallback((): InterventionHistory => {
    return computeHistoryStats(history);
  }, [history]);

  /** 获取最有效的干预类型 */
  const getRecommendedType = useCallback((): string => {
    const stats = computeHistoryStats(history);
    return stats.recommendedType;
  }, [history]);

  return { baseline, lastReport, captureBaseline, computeReport, reset, getHistory, getRecommendedType, history };
}
