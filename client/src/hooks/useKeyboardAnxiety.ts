import { useEffect, useRef, useState, useCallback } from 'react';

export interface AnxietyMetrics {
  typingSpeed: number;       // 字/分钟
  deletionRate: number;      // 删除率 0-100
  pauseFrequency: number;    // 停顿频率 0-100
  rhythmVariance: number;    // 节奏方差 0-100
  anxietyIndex: number;      // 综合焦虑指数 0-100
  sampleSize: number;        // 采样次数
}

const defaultMetrics: AnxietyMetrics = {
  typingSpeed: 0,
  deletionRate: 0,
  pauseFrequency: 0,
  rhythmVariance: 0,
  anxietyIndex: 0,
  sampleSize: 0,
};

export function useKeyboardAnxiety() {
  const [metrics, setMetrics] = useState<AnxietyMetrics>(defaultMetrics);
  const [isActive, setIsActive] = useState(false);

  const keyTimesRef = useRef<number[]>([]);
  const deletionCountRef = useRef(0);
  const totalKeysRef = useRef(0);
  const pausesRef = useRef<number[]>([]);
  const lastKeyTimeRef = useRef<number>(0);
  const sessionStartRef = useRef<number>(0);
  const wordBoundariesRef = useRef<number[]>([]);

  const analyze = useCallback(() => {
    const times = keyTimesRef.current;
    const totalKeys = totalKeysRef.current;
    const deletions = deletionCountRef.current;
    const pauses = pausesRef.current;

    if (times.length < 5) return;

    // 1. 打字速度（字/分钟）
    const elapsed = (times[times.length - 1] - sessionStartRef.current) / 1000 / 60;
    const typingSpeed = elapsed > 0 ? Math.round(times.length / elapsed) : 0;

    // 2. 删除率
    const deletionRate = totalKeys > 0 ? Math.round((deletions / totalKeys) * 100) : 0;

    // 3. 停顿频率（>1.5秒的间隔视为停顿）
    const longPauses = pauses.filter(p => p > 1500).length;
    const pauseFrequency = pauses.length > 0 ? Math.round((longPauses / pauses.length) * 100) : 0;

    // 4. 节奏方差（打字间隔的标准差，越大越不稳定）
    const intervals: number[] = [];
    for (let i = 1; i < times.length; i++) {
      intervals.push(times[i] - times[i - 1]);
    }
    const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
    const variance = intervals.reduce((sum, v) => sum + Math.pow(v - avgInterval, 2), 0) / intervals.length;
    const stdDev = Math.sqrt(variance);
    // 标准化到 0-100（假设 stdDev > 2000ms 为极高不稳定）
    const rhythmVariance = Math.min(100, Math.round((stdDev / 2000) * 100));

    // 5. 综合焦虑指数（加权计算）
    // 高删除率 + 高停顿频率 + 高节奏方差 = 高焦虑
    // 打字速度过快或过慢都可能表示焦虑
    const speedFactor = typingSpeed > 80 ? 30 : typingSpeed < 20 && typingSpeed > 0 ? 20 : 10;
    const anxietyIndex = Math.min(100, Math.round(
      deletionRate * 0.35 +
      pauseFrequency * 0.30 +
      rhythmVariance * 0.20 +
      speedFactor * 0.15
    ));

    setMetrics({
      typingSpeed,
      deletionRate,
      pauseFrequency,
      rhythmVariance,
      anxietyIndex,
      sampleSize: times.length,
    });
  }, []);

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (!isActive) return;
    const now = Date.now();

    if (lastKeyTimeRef.current > 0) {
      const interval = now - lastKeyTimeRef.current;
      pausesRef.current.push(interval);
    }

    lastKeyTimeRef.current = now;
    totalKeysRef.current++;

    if (e.key === 'Backspace' || e.key === 'Delete') {
      deletionCountRef.current++;
    }

    keyTimesRef.current.push(now);

    // 每 10 次按键分析一次
    if (keyTimesRef.current.length % 10 === 0) {
      analyze();
    }
  }, [isActive, analyze]);

  const start = useCallback(() => {
    keyTimesRef.current = [];
    deletionCountRef.current = 0;
    totalKeysRef.current = 0;
    pausesRef.current = [];
    lastKeyTimeRef.current = 0;
    sessionStartRef.current = Date.now();
    setIsActive(true);
    setMetrics(defaultMetrics);
  }, []);

  const stop = useCallback(() => {
    setIsActive(false);
    analyze();
  }, [analyze]);

  const reset = useCallback(() => {
    setMetrics(defaultMetrics);
    keyTimesRef.current = [];
    deletionCountRef.current = 0;
    totalKeysRef.current = 0;
    pausesRef.current = [];
    lastKeyTimeRef.current = 0;
  }, []);

  useEffect(() => {
    if (isActive) {
      window.addEventListener('keydown', handleKeyDown);
      return () => window.removeEventListener('keydown', handleKeyDown);
    }
  }, [isActive, handleKeyDown]);

  return { metrics, isActive, start, stop, reset };
}
