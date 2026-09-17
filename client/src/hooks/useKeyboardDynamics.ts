/**
 * 键盘动力学分析 Hook
 *
 * 理论依据：
 * - Keystroke Dynamics for Biometric Identification (Jain et al., 2011)
 * - Key-press patterns reflect cognitive load (Airas, 2008)
 * - Typing rhythm changes correlate with anxiety/depression (Yano et al., 2019)
 *
 * 采集维度：
 * - 打字节奏：键间间隔、速度、变异系数
 * - 停顿分析：长停顿(>1.5s)次数与占比
 * - 修正行为：删除率、修正率、连续删除爆发
 * - 错误与压力：错误率估计、按键压力指数
 */
import { useEffect, useRef, useState, useCallback } from 'react';
import type { KeyboardDynamics } from '../types/multimodal.types';
import { defaultKeyboard } from '../types/multimodal.types';

export function useKeyboardDynamics() {
  const [metrics, setMetrics] = useState<KeyboardDynamics>({ ...defaultKeyboard });
  const isActiveRef = useRef(false);

  // 按键时间序列
  const keyTimesRef = useRef<number[]>([]);
  const keyIntervalsRef = useRef<number[]>([]);
  const pauseDurationsRef = useRef<number[]>([]);
  const lastKeyTimeRef = useRef(0);
  const deletionCountRef = useRef(0);
  const totalKeysRef = useRef(0);
  const consecutiveDelRef = useRef(0); // 连续删除计数
  const backspaceBurstRef = useRef(0); // 删除爆发次数

  const PAUSE_THRESHOLD = 1500; // 长停顿阈值 1.5s

  const computeMetrics = useCallback(() => {
    const intervals = keyIntervalsRef.current;
    const pauses = pauseDurationsRef.current;
    const total = totalKeysRef.current;
    const deletions = deletionCountRef.current;

    if (total < 5) return;

    // 打字速度
    const elapsed = (Date.now() - (keyTimesRef.current[0] || Date.now())) / 60000; // 分钟
    const typingSpeed = elapsed > 0 ? Math.round(total / elapsed) : 0;

    // 键间间隔统计
    const validIntervals = intervals.filter(i => i > 0 && i < 10000);
    const avgInterval = validIntervals.length > 0
      ? validIntervals.reduce((a, b) => a + b, 0) / validIntervals.length : 0;
    const intervalMean = avgInterval || 1;
    const intervalVariance = validIntervals.length > 1
      ? Math.sqrt(validIntervals.reduce((sum, v) => sum + (v - intervalMean) ** 2, 0) / validIntervals.length) / intervalMean
      : 0; // 变异系数 CV

    // 节奏规律性 (1 - CV，越规律越高)
    const rhythmScore = Math.max(0, Math.min(1, 1 - intervalVariance));

    // 停顿分析
    const pauseCount = pauses.length;
    const totalPauseTime = pauses.reduce((a, b) => a + b, 0);
    const totalTime = keyTimesRef.current.length > 0
      ? keyTimesRef.current[keyTimesRef.current.length - 1] - keyTimesRef.current[0] : 0;
    const pauseRatio = totalTime > 0 ? totalPauseTime / totalTime : 0;
    const avgPauseDuration = pauseCount > 0 ? totalPauseTime / pauseCount : 0;

    // 修正行为
    const deletionRate = total > 0 ? deletions / total : 0;
    const correctionRate = total > 0 ? Math.min(1, deletions / (total * 0.3)) : 0;
    const backspaceBurst = backspaceBurstRef.current;

    // 错误率估计 (基于删除率 + 速度变化的综合指标)
    const speedFactor = typingSpeed > 80 ? 0.3 : typingSpeed < 15 && typingSpeed > 0 ? 0.2 : 0;
    const errorRate = Math.min(1, deletionRate * 0.6 + intervalVariance * 0.3 + speedFactor);

    // 按键压力指数 (速度↑ + 删除↓ + 节奏乱 → 高压力)
    const speedPressure = typingSpeed > 60 ? (typingSpeed - 60) / 60 : 0;
    const deletionPressure = deletionRate < 0.05 && total > 20 ? 0.3 : 0; // 几乎不删除=紧张
    const rhythmPressure = 1 - rhythmScore;
    const pressureIndex = Math.min(1, speedPressure * 0.4 + deletionPressure + rhythmPressure * 0.3 + intervalVariance * 0.3);

    // 综合焦虑指数
    const pauseStress = pauseRatio > 0.3 ? (pauseRatio - 0.3) * 100 : 0;
    const burstStress = Math.min(30, backspaceBurst * 6);
    const anxietyIndex = Math.min(100, Math.round(
      pressureIndex * 30 + intervalVariance * 25 + pauseStress * 0.3 + burstStress + errorRate * 15
    ));

    setMetrics(prev => {
      const newMetrics = {
        ...prev,
        typingSpeed, avgInterval: Math.round(avgInterval),
        intervalVariance: Math.round(intervalVariance * 100) / 100,
        rhythmScore: Math.round(rhythmScore * 100) / 100,
        pauseCount, pauseRatio: Math.round(pauseRatio * 100) / 100,
        avgPauseDuration: Math.round(avgPauseDuration),
        deletionRate: Math.round(deletionRate * 100) / 100,
        correctionRate: Math.round(correctionRate * 100) / 100,
        backspaceBurst,
        errorRate: Math.round(errorRate * 100) / 100,
        pressureIndex: Math.round(pressureIndex * 100) / 100,
        anxietyIndex,
        sampleSize: total,
        isCollecting: isActiveRef.current,
      };
      console.log('[键盘分析] 指标更新:', { sampleSize: total, typingSpeed, isCollecting: isActiveRef.current });
      return newMetrics;
    });
  }, []);

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (!isActiveRef.current) return;
    const now = Date.now();
    totalKeysRef.current++;
    console.log('[键盘分析] 按键捕获:', e.key, '总计:', totalKeysRef.current);
  
    // 记录键间间隔
    if (lastKeyTimeRef.current > 0) {
      const interval = now - lastKeyTimeRef.current;
      keyIntervalsRef.current.push(interval);
  
      // 检测长停顿
      if (interval > PAUSE_THRESHOLD) {
        pauseDurationsRef.current.push(interval);
      }
    }
  
    keyTimesRef.current.push(now);
    lastKeyTimeRef.current = now;
  
    // 删除键行为分析
    if (e.key === 'Backspace' || e.key === 'Delete') {
      deletionCountRef.current++;
      consecutiveDelRef.current++;
      if (consecutiveDelRef.current >= 3) {
        backspaceBurstRef.current++; // 连续删除≥3 次 = 一次爆发
      }
    } else {
      consecutiveDelRef.current = 0; // 重置连续删除
    }
  }, []);

  const intervalRef = useRef(0);

  const start = useCallback(() => {
    isActiveRef.current = true;
    keyTimesRef.current = [];
    keyIntervalsRef.current = [];
    pauseDurationsRef.current = [];
    lastKeyTimeRef.current = 0;
    deletionCountRef.current = 0;
    totalKeysRef.current = 0;
    consecutiveDelRef.current = 0;
    backspaceBurstRef.current = 0;
    setMetrics(prev => ({ ...prev, isCollecting: true }));
    window.addEventListener('keydown', handleKeyDown);
    // 直接在这里启动 interval，确保分析运行
    intervalRef.current = window.setInterval(computeMetrics, 3000);
    console.log('[键盘分析] 已启动');
  }, [handleKeyDown, computeMetrics]);

  const stop = useCallback(() => {
    isActiveRef.current = false;
    window.removeEventListener('keydown', handleKeyDown);
    clearInterval(intervalRef.current);
    setMetrics(prev => ({ ...prev, isCollecting: false }));
    console.log('[键盘分析] 已停止');
  }, [handleKeyDown]);

  const reset = useCallback(() => {
    stop();
    setMetrics({ ...defaultKeyboard });
  }, [stop]);
  
  // 组件卸载时清理
  useEffect(() => {
    return () => {
      isActiveRef.current = false;
      window.removeEventListener('keydown', handleKeyDown);
      clearInterval(intervalRef.current);
    };
  }, [handleKeyDown]);

  return { metrics, start, stop, reset };
}
