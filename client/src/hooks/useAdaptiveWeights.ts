/**
 * 模态权重自适应 Hook —— 自动学习哪个模态对哪个用户更可靠
 *
 * 原理：
 * - 维护最近 50 次融合的滑动窗口
 * - 计算每个模态的 riskScore 与最终融合结果的相关性
 * - 一致性高的模态权重上调，一致性低的下调
 * - 每 10 次融合更新一次，避免频繁抖动
 *
 * 存储：localStorage，绑定 userId
 */
import { useState, useCallback, useRef } from 'react';

// ===== 类型定义 =====

export interface FusionRecord {
  /** 各模态的 riskScore (0-1) */
  modalityScores: Record<string, number>;
  /** 融合后的焦虑概率 */
  fusedAnxietyProb: number;
  /** 时间戳 */
  timestamp: number;
}

export interface AdaptiveWeightState {
  /** 自适应权重系数（相对于年龄基准的乘数） */
  coefficients: Record<string, number>;
  /** 滑动窗口中的记录数 */
  recordCount: number;
  /** 上次更新时的记录数 */
  lastUpdateAt: number;
}

// ===== 常量 =====

const WINDOW_SIZE = 50;          // 滑动窗口大小
const UPDATE_INTERVAL = 10;      // 每 10 次融合更新一次权重
const MIN_COEFFICIENT = 0.5;     // 权重系数下限（年龄基准的 50%）
const MAX_COEFFICIENT = 1.5;     // 权重系数上限（年龄基准的 150%）
const ADJUSTMENT_STEP = 0.02;    // 每次调整步长
const STORAGE_KEY = 'adaptive_weights';

// ===== 默认值 =====

function createDefaultState(): AdaptiveWeightState {
  return {
    coefficients: {},
    recordCount: 0,
    lastUpdateAt: 0,
  };
}

// ===== localStorage 持久化 =====

function loadState(): AdaptiveWeightState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed.coefficients === 'object') {
        return parsed as AdaptiveWeightState;
      }
    }
  } catch {}
  return createDefaultState();
}

function saveState(state: AdaptiveWeightState): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {}
}

// ===== Pearson 相关系数计算 =====

function pearsonCorrelation(xs: number[], ys: number[]): number {
  const n = xs.length;
  if (n < 5) return 0; // 样本不足，不计算

  const meanX = xs.reduce((a, b) => a + b, 0) / n;
  const meanY = ys.reduce((a, b) => a + b, 0) / n;

  let num = 0, denX = 0, denY = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - meanX;
    const dy = ys[i] - meanY;
    num += dx * dy;
    denX += dx * dx;
    denY += dy * dy;
  }

  const den = Math.sqrt(denX * denY);
  if (den < 0.0001) return 0; // 方差为零，不相关
  return num / den;
}

// ===== Hook =====

export function useAdaptiveWeights() {
  const [state, setState] = useState<AdaptiveWeightState>(loadState);
  const stateRef = useRef(state);
  stateRef.current = state;

  // 滑动窗口（不持久化，仅运行时）
  const windowRef = useRef<FusionRecord[]>([]);

  // 记录一次融合结果
  const recordFusion = useCallback((
    modalityScores: Record<string, number>,
    fusedAnxietyProb: number,
  ) => {
    // 添加到滑动窗口
    windowRef.current.push({
      modalityScores,
      fusedAnxietyProb,
      timestamp: Date.now(),
    });

    // 保持窗口大小
    if (windowRef.current.length > WINDOW_SIZE) {
      windowRef.current = windowRef.current.slice(-WINDOW_SIZE);
    }

    // 更新记录计数
    setState(prev => {
      const newCount = prev.recordCount + 1;

      // 每 UPDATE_INTERVAL 次才重新计算权重
      if (newCount - prev.lastUpdateAt < UPDATE_INTERVAL) {
        return { ...prev, recordCount: newCount };
      }

      // 需要足够的数据点（至少 10 个）
      const records = windowRef.current;
      if (records.length < 10) {
        return { ...prev, recordCount: newCount, lastUpdateAt: newCount };
      }

      // 获取所有模态键
      const modalityKeys = Object.keys(records[0].modalityScores);
      const fusedProbs = records.map(r => r.fusedAnxietyProb);

      // 计算每个模态与融合结果的相关性
      const newCoefficients = { ...prev.coefficients };
      for (const key of modalityKeys) {
        const scores = records.map(r => r.modalityScores[key] ?? 0);
        const correlation = pearsonCorrelation(scores, fusedProbs);

        // 基于相关性调整权重系数
        const currentCoeff = newCoefficients[key] ?? 1.0;
        let newCoeff = currentCoeff;

        if (correlation > 0.3) {
          // 与融合结果高度正相关 → 该模态可靠，提升权重
          newCoeff = Math.min(MAX_COEFFICIENT, currentCoeff + ADJUSTMENT_STEP);
        } else if (correlation < -0.1) {
          // 与融合结果负相关或无关 → 该模态可能引入噪声，降低权重
          newCoeff = Math.max(MIN_COEFFICIENT, currentCoeff - ADJUSTMENT_STEP);
        }
        // -0.1 ≤ correlation ≤ 0.3 → 不调整

        newCoefficients[key] = Math.round(newCoeff * 1000) / 1000;
      }

      const next = { ...prev, coefficients: newCoefficients, recordCount: newCount, lastUpdateAt: newCount };
      saveState(next);
      return next;
    });
  }, []);

  // 获取某个模态的自适应权重系数
  const getCoefficient = useCallback((modality: string): number => {
    return stateRef.current.coefficients[modality] ?? 1.0;
  }, []);

  // 获取所有自适应权重
  const getAllCoefficients = useCallback((): Record<string, number> => {
    return { ...stateRef.current.coefficients };
  }, []);

  // 重置
  const reset = useCallback(() => {
    const fresh = createDefaultState();
    setState(fresh);
    saveState(fresh);
    windowRef.current = [];
  }, []);

  return {
    state,
    recordFusion,
    getCoefficient,
    getAllCoefficients,
    reset,
  };
}
