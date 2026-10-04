/**
 * 个人基线校准 Hook —— 用指数移动平均（EMA）建立每个用户的个人常态画像
 *
 * 原理：
 * - 群体基线（如心率 72bpm）无法判断个体异常
 * - 个人基线（如张三均值 65±3bpm）才能准确识别偏离
 * - EMA 对近期数据更敏感，同时保留历史记忆
 *
 * 存储：localStorage，绑定 userId，注销时清除
 *
 * 后端同步：
 * - 每次 updateBaseline() 后，节流 30 秒同步一次到后端
 * - 同步失败不阻塞前端，记录到 console
 * - localStorage 中 sync_baseline_enabled=false 时不同步（开发者开关）
 */
import { useState, useCallback, useRef } from 'react';
import { baselineApi } from '../services';

// ===== 类型定义 =====

export interface BaselineMetric {
  mean: number;
  std: number;
  n: number;          // 样本数
}

export interface PersonalBaseline {
  heartRate: BaselineMetric;
  breathingRate: BaselineMetric;
  typingSpeed: BaselineMetric;
  blinkRate: BaselineMetric;
  voiceF0: BaselineMetric;
  speechRate: BaselineMetric;
  emotionBaseline: number[];  // [快乐, 悲伤, 焦虑, 愤怒, 中性] 的 EMA 均值
  createdAt: number;
  lastUpdated: number;
  sampleCount: number;
}

export interface BaselineDeviation {
  heartRate: number;       // z-score
  breathingRate: number;
  typingSpeed: number;
  blinkRate: number;
  voiceF0: number;
  speechRate: number;
  maxDeviation: number;    // 最大偏离绝对值
  calibrated: boolean;     // 基线是否已校准（样本 >= 20）
}

// ===== 常量 =====

const EMA_ALPHA = 0.05;           // EMA 平滑系数（越小越稳定）
const CALIBRATION_SAMPLES = 20;   // 基线校准所需最小样本数
const STORAGE_KEY = 'personal_baseline';
const SYNC_THROTTLE_MS = 30000;   // 后端同步节流间隔（30 秒）
const SYNC_ENABLED_KEY = 'sync_baseline_enabled'; // 开发者开关

// ===== 默认值 =====

function emptyMetric(): BaselineMetric {
  return { mean: 0, std: 0, n: 0 };
}

function createDefaultBaseline(): PersonalBaseline {
  return {
    heartRate: emptyMetric(),
    breathingRate: emptyMetric(),
    typingSpeed: emptyMetric(),
    blinkRate: emptyMetric(),
    voiceF0: emptyMetric(),
    speechRate: emptyMetric(),
    emotionBaseline: [0.2, 0.2, 0.2, 0.2, 0.2],  // 均匀初始值
    createdAt: 0,
    lastUpdated: 0,
    sampleCount: 0,
  };
}

// ===== EMA 更新算法 =====

function updateMetric(metric: BaselineMetric, value: number): BaselineMetric {
  if (metric.n === 0) {
    // 第一个样本：直接初始化
    return { mean: value, std: 0, n: 1 };
  }

  const { mean, std, n } = metric;
  const variance = std * std;

  // EMA 均值更新
  const newMean = EMA_ALPHA * value + (1 - EMA_ALPHA) * mean;
  // EMA 方差更新：alpha * (x - mean_old)^2 + (1 - alpha) * var_old
  const newVariance = EMA_ALPHA * (value - mean) ** 2 + (1 - EMA_ALPHA) * variance;
  const newStd = Math.sqrt(Math.max(0, newVariance));

  return {
    mean: Math.round(newMean * 100) / 100,
    std: Math.round(newStd * 100) / 100,
    n: n + 1,
  };
}

function updateEmotionBaseline(baseline: number[], probs: number[]): number[] {
  if (baseline.length !== 5 || probs.length !== 5) return baseline;
  return probs.map((p, i) => {
    const old = baseline[i];
    const updated = EMA_ALPHA * p + (1 - EMA_ALPHA) * old;
    return Math.round(updated * 1000) / 1000;
  });
}

function calcZScore(metric: BaselineMetric, value: number): number {
  if (metric.n < 2 || metric.std < 0.001) return 0;
  return (value - metric.mean) / metric.std;
}

// ===== localStorage 持久化 =====

function loadBaseline(): PersonalBaseline {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      // 验证结构完整性
      if (parsed && typeof parsed.heartRate === 'object' && parsed.emotionBaseline?.length === 5) {
        return parsed as PersonalBaseline;
      }
    }
  } catch {}
  return createDefaultBaseline();
}

function saveBaseline(baseline: PersonalBaseline): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(baseline));
  } catch {}
}

// ===== Hook =====

export function usePersonalBaseline() {
  const [baseline, setBaseline] = useState<PersonalBaseline>(loadBaseline);
  const baselineRef = useRef(baseline);
  baselineRef.current = baseline;

  // 后端同步节流控制
  const lastSyncTimeRef = useRef(0);

  /**
   * 同步基线到后端（节流 30 秒）
   * 
   * 同步内容：当前各模态的最新观测值（均值），后端用 EMA 更新
   * 同步失败不阻塞前端，只记录到 console
   */
  const syncToBackend = useCallback(() => {
    // 开发者开关检查
    if (localStorage.getItem(SYNC_ENABLED_KEY) === 'false') {
      return;
    }

    // 节流检查
    const now = Date.now();
    if (now - lastSyncTimeRef.current < SYNC_THROTTLE_MS) {
      return;
    }
    lastSyncTimeRef.current = now;

    const b = baselineRef.current;
    
    // 构建增量观测数据（当前各模态的均值作为最新观测值）
    const observation: Record<string, number | number[]> = {};
    if (b.heartRate.n > 0) observation.heartRate = b.heartRate.mean;
    if (b.breathingRate.n > 0) observation.breathingRate = b.breathingRate.mean;
    if (b.typingSpeed.n > 0) observation.typingSpeed = b.typingSpeed.mean;
    if (b.blinkRate.n > 0) observation.blinkRate = b.blinkRate.mean;
    if (b.voiceF0.n > 0) observation.voiceF0 = b.voiceF0.mean;
    if (b.speechRate.n > 0) observation.speechRate = b.speechRate.mean;
    if (b.emotionBaseline.length === 5) {
      observation.emotionProbs = b.emotionBaseline;
    }

    // 至少有一个模态有数据才同步
    if (Object.keys(observation).length === 0) {
      return;
    }

    // 异步同步，不阻塞前端
    baselineApi.syncBaseline(observation)
      .then(() => {
        console.debug('[Baseline] 同步成功', observation);
      })
      .catch((err) => {
        console.warn('[Baseline] 同步失败，已降级:', err?.message || err);
      });
  }, []);

  // 更新基线：每次融合后调用
  const updateBaseline = useCallback((data: {
    heartRate?: number;
    breathingRate?: number;
    typingSpeed?: number;
    blinkRate?: number;
    voiceF0?: number;
    speechRate?: number;
    emotionProbs?: number[];
  }) => {
    setBaseline(prev => {
      const next = { ...prev };
      let changed = false;

      if (data.heartRate !== undefined && data.heartRate > 0) {
        next.heartRate = updateMetric(prev.heartRate, data.heartRate);
        changed = true;
      }
      if (data.breathingRate !== undefined && data.breathingRate > 0) {
        next.breathingRate = updateMetric(prev.breathingRate, data.breathingRate);
        changed = true;
      }
      if (data.typingSpeed !== undefined && data.typingSpeed > 0) {
        next.typingSpeed = updateMetric(prev.typingSpeed, data.typingSpeed);
        changed = true;
      }
      if (data.blinkRate !== undefined && data.blinkRate > 0) {
        next.blinkRate = updateMetric(prev.blinkRate, data.blinkRate);
        changed = true;
      }
      if (data.voiceF0 !== undefined && data.voiceF0 > 0) {
        next.voiceF0 = updateMetric(prev.voiceF0, data.voiceF0);
        changed = true;
      }
      if (data.speechRate !== undefined && data.speechRate > 0) {
        next.speechRate = updateMetric(prev.speechRate, data.speechRate);
        changed = true;
      }
      if (data.emotionProbs && data.emotionProbs.length === 5) {
        next.emotionBaseline = updateEmotionBaseline(prev.emotionBaseline, data.emotionProbs);
        changed = true;
      }

      if (changed) {
        next.sampleCount = prev.sampleCount + 1;
        next.lastUpdated = Date.now();
        if (prev.createdAt === 0) next.createdAt = Date.now();
        saveBaseline(next);
        // 触发后端同步（节流控制）
        // 注意：这里在 setState 回调中不能直接调用外部函数，需要用 setTimeout
        setTimeout(() => syncToBackend(), 0);
      }

      return next;
    });
  }, [syncToBackend]);

  // 计算 z-score
  const getZScore = useCallback((modality: keyof Omit<PersonalBaseline, 'emotionBaseline' | 'createdAt' | 'lastUpdated' | 'sampleCount'>, value: number): number => {
    const metric = baseline[modality] as BaselineMetric | undefined;
    if (!metric) return 0;
    return Math.round(calcZScore(metric, value) * 100) / 100;
  }, [baseline]);

  // 获取所有模态的偏离程度
  const getBaselineDeviation = useCallback((): BaselineDeviation => {
    const b = baselineRef.current;
    const heartRateZ = calcZScore(b.heartRate, b.heartRate.mean);  // 当前值即最新均值
    const breathingZ = calcZScore(b.breathingRate, b.breathingRate.mean);
    const typingZ = calcZScore(b.typingSpeed, b.typingSpeed.mean);
    const blinkZ = calcZScore(b.blinkRate, b.blinkRate.mean);
    const voiceZ = calcZScore(b.voiceF0, b.voiceF0.mean);
    const speechZ = calcZScore(b.speechRate, b.speechRate.mean);

    const deviations = [
      Math.abs(heartRateZ), Math.abs(breathingZ), Math.abs(typingZ),
      Math.abs(blinkZ), Math.abs(voiceZ), Math.abs(speechZ),
    ];

    return {
      heartRate: Math.round(heartRateZ * 100) / 100,
      breathingRate: Math.round(breathingZ * 100) / 100,
      typingSpeed: Math.round(typingZ * 100) / 100,
      blinkRate: Math.round(blinkZ * 100) / 100,
      voiceF0: Math.round(voiceZ * 100) / 100,
      speechRate: Math.round(speechZ * 100) / 100,
      maxDeviation: Math.round(Math.max(...deviations) * 100) / 100,
      calibrated: b.sampleCount >= CALIBRATION_SAMPLES,
    };
  }, []);

  // 获取某个模态的当前 z-score（传入实时值）
  const getZScoreForValue = useCallback((modality: keyof Omit<PersonalBaseline, 'emotionBaseline' | 'createdAt' | 'lastUpdated' | 'sampleCount'>, value: number): number => {
    const metric = baselineRef.current[modality] as BaselineMetric | undefined;
    if (!metric || metric.n < 2) return 0;
    return Math.round(calcZScore(metric, value) * 100) / 100;
  }, []);

  // 基线是否已校准
  const isCalibrated = useCallback((): boolean => {
    return baselineRef.current.sampleCount >= CALIBRATION_SAMPLES;
  }, []);

  // 重置基线
  const reset = useCallback(() => {
    const fresh = createDefaultBaseline();
    setBaseline(fresh);
    saveBaseline(fresh);
  }, []);

  // 持久化（外部触发保存）
  const persist = useCallback(() => {
    saveBaseline(baselineRef.current);
  }, []);

  // 加载（从 localStorage 恢复）
  const load = useCallback(() => {
    const saved = loadBaseline();
    setBaseline(saved);
  }, []);

  return {
    baseline,
    updateBaseline,
    syncToBackend,
    getZScore,
    getZScoreForValue,
    getBaselineDeviation,
    isCalibrated,
    reset,
    persist,
    load,
  };
}
