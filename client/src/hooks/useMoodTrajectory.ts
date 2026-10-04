/**
 * 情绪时间轨迹 Hook —— 从单次快照升级为趋势洞察
 *
 * 原理：
 * - 每次融合后记录一个数据点
 * - 同一天只保留中位数（去重）
 * - 用简化版 Mann-Kendall 趋势检验判断方向
 * - 存储上限 336 个点（14天 × 24小时），超出自动清理
 *
 * 存储：localStorage，绑定 userId
 */
import { useState, useCallback, useRef } from 'react';

// ===== 类型定义 =====

export interface TrajectoryPoint {
  timestamp: number;
  fusionScore: number;       // 融合焦虑概率 (0-1)
  anxietyProb: number;       // 焦虑概率
  dominantEmotion: string;   // 主导情绪
  activeModalityCount: number; // 活跃模态数
}

export type TrendDirection = 'improving' | 'declining' | 'stable';

export interface MoodTrajectory {
  points: TrajectoryPoint[];
  trend: TrendDirection;
  trendStrength: number;     // 0-1，趋势置信度
  weeklyChange: number;      // 本周 vs 上周变化百分比
}

// ===== 常量 =====

const MAX_POINTS = 336;       // 14天 × 24小时
const STORAGE_KEY = 'mood_trajectory';

// ===== 默认值 =====

function createDefaultTrajectory(): MoodTrajectory {
  return {
    points: [],
    trend: 'stable',
    trendStrength: 0,
    weeklyChange: 0,
  };
}

// ===== localStorage 持久化 =====

function loadTrajectory(): MoodTrajectory {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (parsed && Array.isArray(parsed.points)) {
        return parsed as MoodTrajectory;
      }
    }
  } catch {}
  return createDefaultTrajectory();
}

function saveTrajectory(trajectory: MoodTrajectory): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(trajectory));
  } catch {}
}

// ===== 工具函数 =====

/** 获取日期键（YYYY-MM-DD） */
function getDayKey(timestamp: number): string {
  const d = new Date(timestamp);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

/** 计算中位数 */
function median(arr: number[]): number {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * 简化版 Mann-Kendall 趋势检验
 * 计算 S 统计量：正数表示上升趋势，负数表示下降趋势
 * 返回 { direction, strength }
 */
function mannKendallTrend(values: number[]): { direction: TrendDirection; strength: number } {
  const n = values.length;
  if (n < 5) return { direction: 'stable', strength: 0 };

  let s = 0;
  for (let i = 0; i < n - 1; i++) {
    for (let j = i + 1; j < n; j++) {
      const diff = values[j] - values[i];
      s += diff > 0 ? 1 : diff < 0 ? -1 : 0;
    }
  }

  // 归一化 S 到 [-1, 1]
  const maxS = n * (n - 1) / 2;
  const normalizedS = maxS > 0 ? s / maxS : 0;
  const strength = Math.abs(normalizedS);

  let direction: TrendDirection = 'stable';
  if (normalizedS < -0.15) direction = 'improving';   // 焦虑概率下降 = 改善
  else if (normalizedS > 0.15) direction = 'declining'; // 焦虑概率上升 = 恶化

  return { direction, strength: Math.round(strength * 100) / 100 };
}

// ===== Hook =====

export function useMoodTrajectory() {
  const [trajectory, setTrajectory] = useState<MoodTrajectory>(loadTrajectory);
  const trajectoryRef = useRef(trajectory);
  trajectoryRef.current = trajectory;

  // 记录一个数据点
  const recordPoint = useCallback((
    fusionScore: number,
    anxietyProb: number,
    dominantEmotion: string,
    activeModalityCount: number,
  ) => {
    setTrajectory(prev => {
      const now = Date.now();
      const dayKey = getDayKey(now);
      const newPoint: TrajectoryPoint = {
        timestamp: now,
        fusionScore,
        anxietyProb,
        dominantEmotion,
        activeModalityCount,
      };

      // 去重逻辑：同一天内，如果已有多个点，保留最新的（近似中位数策略）
      let points = [...prev.points];
      const existingDayPoints = points.filter(p => getDayKey(p.timestamp) === dayKey);

      if (existingDayPoints.length >= 3) {
        // 同一天超过 3 个点，用中位数替代
        const otherPoints = points.filter(p => getDayKey(p.timestamp) !== dayKey);
        const dayAnxietyProbs = [...existingDayPoints.map(p => p.anxietyProb), anxietyProb];
        const medianProb = median(dayAnxietyProbs);
        // 找到最接近中位数的点
        const closestPoint = existingDayPoints.reduce((best, p) =>
          Math.abs(p.anxietyProb - medianProb) < Math.abs(best.anxietyProb - medianProb) ? p : best
        );
        // 用最新数据更新最接近中位数的点
        const updatedPoint = { ...closestPoint, ...newPoint };
        points = [...otherPoints, updatedPoint];
      } else {
        points.push(newPoint);
      }

      // 按时间排序
      points.sort((a, b) => a.timestamp - b.timestamp);

      // 超出上限则清理最旧的
      if (points.length > MAX_POINTS) {
        points = points.slice(points.length - MAX_POINTS);
      }

      // 重新计算趋势
      const allProbs = points.map(p => p.anxietyProb);
      const { direction, strength } = mannKendallTrend(allProbs);

      // 计算周变化：最近 7 天 vs 前 7 天
      const sevenDaysAgo = now - 7 * 24 * 60 * 60 * 1000;
      const fourteenDaysAgo = now - 14 * 24 * 60 * 60 * 1000;
      const thisWeek = points.filter(p => p.timestamp >= sevenDaysAgo).map(p => p.anxietyProb);
      const lastWeek = points.filter(p => p.timestamp >= fourteenDaysAgo && p.timestamp < sevenDaysAgo).map(p => p.anxietyProb);

      let weeklyChange = 0;
      if (thisWeek.length > 0 && lastWeek.length > 0) {
        const thisAvg = thisWeek.reduce((a, b) => a + b, 0) / thisWeek.length;
        const lastAvg = lastWeek.reduce((a, b) => a + b, 0) / lastWeek.length;
        weeklyChange = lastAvg > 0.01 ? Math.round(((thisAvg - lastAvg) / lastAvg) * 100) : 0;
      }

      const next: MoodTrajectory = {
        points,
        trend: direction,
        trendStrength: strength,
        weeklyChange,
      };

      saveTrajectory(next);
      return next;
    });
  }, []);

  // 获取完整轨迹
  const getTrajectory = useCallback((): MoodTrajectory => {
    return trajectoryRef.current;
  }, []);

  // 获取最近 7 天的每日数据
  const getDailyPoints = useCallback((): TrajectoryPoint[] => {
    const now = Date.now();
    const sevenDaysAgo = now - 7 * 24 * 60 * 60 * 1000;
    return trajectoryRef.current.points
      .filter(p => p.timestamp >= sevenDaysAgo)
      .sort((a, b) => a.timestamp - b.timestamp);
  }, []);

  // 生成周报摘要
  const getWeeklyReport = useCallback((): string => {
    const t = trajectoryRef.current;
    if (t.points.length < 5) return '数据不足，暂无法生成周报。';

    const trendLabels: Record<TrendDirection, string> = {
      improving: '改善',
      declining: '恶化',
      stable: '平稳',
    };

    const parts: string[] = [];
    parts.push(`过去一周情绪趋势：${trendLabels[t.trend]}（置信度 ${Math.round(t.trendStrength * 100)}%）`);

    if (t.weeklyChange !== 0) {
      const direction = t.weeklyChange > 0 ? '上升' : '下降';
      parts.push(`焦虑水平比上周${direction} ${Math.abs(t.weeklyChange)}%`);
    }

    const dailyPts = t.points.slice(-7);
    if (dailyPts.length > 0) {
      const avgAnxiety = dailyPts.reduce((s, p) => s + p.anxietyProb, 0) / dailyPts.length;
      parts.push(`近 7 天平均焦虑概率：${Math.round(avgAnxiety * 100)}%`);
    }

    return parts.join('。') + '。';
  }, []);

  // 重置
  const reset = useCallback(() => {
    const fresh = createDefaultTrajectory();
    setTrajectory(fresh);
    saveTrajectory(fresh);
  }, []);

  return {
    trajectory,
    recordPoint,
    getTrajectory,
    getDailyPoints,
    getWeeklyReport,
    reset,
  };
}
