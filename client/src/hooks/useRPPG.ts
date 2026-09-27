/**
 * rPPG 心率变异性 Hook —— 通过摄像头面部皮肤颜色微变化提取心率
 * 年龄差异化：不同年龄段心率基线和HRV正常范围不同
 *
 * ⚠️ 当前 RPPG_AVAILABLE = false：本设备不具备可信的心率采集能力。
 * 该 Hook 只保留数据结构与接口，不再输出任何心率/HRV 数值，
 * 以免把摄像头噪声伪装成生理测量（见 perceptionCapabilities.ts）。
 */
import { useState, useCallback, useRef } from 'react';
import type { HRVAnalysis } from '../types/multimodal.types';
import { defaultHRV } from '../types/multimodal.types';
import type { AgeGroup } from './ageConfig';
import { getHRVConfig } from './ageConfig';
import { RPPG_AVAILABLE } from './perceptionCapabilities';

export function useRPPG(ageGroup: AgeGroup | null = null) {
  const [metrics, setMetrics] = useState<HRVAnalysis>({ ...defaultHRV });
  const hrHistoryRef = useRef<number[]>([]);
  const config = getHRVConfig(ageGroup);

  // 从面部视频帧提取 rPPG 信号（简化版：绿色通道分析）
  const updateFromFrame = useCallback((greenChannelMean: number, timestamp: number) => {
    // 不具备心率采集能力：不采样、不估算、不更新任何指标
    if (!RPPG_AVAILABLE) return;

    // rPPG 简化估计：通过绿色通道变化估计心率
    // 实际应用中需要更复杂的信号处理（FFT/自相关）
    hrHistoryRef.current.push(greenChannelMean);
    if (hrHistoryRef.current.length > 180) hrHistoryRef.current = hrHistoryRef.current.slice(-180);

    // 每 30 个采样点计算一次
    if (hrHistoryRef.current.length < 30) return;

    const signal = hrHistoryRef.current;
    // 简单峰值检测估计心率
    let peaks = 0;
    for (let i = 2; i < signal.length - 2; i++) {
      if (signal[i] > signal[i-1] && signal[i] > signal[i-2] &&
          signal[i] > signal[i+1] && signal[i] > signal[i+2]) {
        peaks++;
      }
    }
    // 估算心率（假设采样率约 10fps，30 个采样点 = 3 秒）
    const durationSec = 30 / 10;
    const estimatedHR = (peaks / durationSec) * 60;

    if (estimatedHR < 40 || estimatedHR > 180) return; // 异常值过滤

    // HRV 简化估计（峰间间隔标准差）
    const hrvRmssd = Math.max(5, 50 - Math.abs(estimatedHR - config.restingHRBaseline) * 0.5);

    // 信号质量
    const signalVariance = signal.reduce((s, v) => s + (v - signal.reduce((a, b) => a + b, 0) / signal.length) ** 2, 0) / signal.length;
    const signalQuality = Math.min(1, signalVariance * 1000);

    // 风险计算
    let riskScore = 0;
    if (estimatedHR > config.anxietyHRThreshold) {
      riskScore += Math.min(0.5, (estimatedHR - config.anxietyHRThreshold) / 30);
    }
    if (hrvRmssd < config.hrvLowRiskThreshold) {
      riskScore += Math.min(0.5, (config.hrvLowRiskThreshold - hrvRmssd) / config.hrvLowRiskThreshold * 0.3);
    }

    setMetrics({
      riskScore: Math.round(Math.min(1, riskScore) * 100) / 100,
      heartRate: Math.round(estimatedHR),
      hrvRmssd: Math.round(hrvRmssd * 10) / 10,
      signalQuality: Math.round(signalQuality * 100) / 100,
      isMeasuring: true,
      timestamp: Date.now(),
    });
  }, [config]);

  const reset = useCallback(() => {
    hrHistoryRef.current = [];
    setMetrics({ ...defaultHRV });
  }, []);

  return { metrics, updateFromFrame, reset };
}
