import React, { createContext, useContext, useState, useCallback, useRef, useEffect, ReactNode } from 'react';
import { useKeyboardAnxiety, AnxietyMetrics as KeyboardMetrics } from '../hooks/useKeyboardAnxiety';
import { useBlinkDetection, BlinkMetrics } from '../hooks/useBlinkDetection';
import { profileApi } from '../services';

export interface CombinedAnxietyState {
  keyboard: KeyboardMetrics;
  blink: BlinkMetrics;
  combinedIndex: number;
  level: 'low' | 'medium' | 'high';
  isMonitoring: boolean;
  keyboardActive: boolean;
  blinkActive: boolean;
}

interface AnxietyContextType {
  state: CombinedAnxietyState;
  startKeyboard: () => void;
  stopKeyboard: () => void;
  startBlink: () => void;
  stopBlink: () => void;
  reportToServer: (context?: string) => Promise<void>;
  realityTask: any;
  showTaskModal: boolean;
  setShowTaskModal: (v: boolean) => void;
  fetchRealityTask: () => Promise<void>;
  reset: () => void;
}

const defaultState: CombinedAnxietyState = {
  keyboard: { typingSpeed: 0, deletionRate: 0, pauseFrequency: 0, rhythmVariance: 0, anxietyIndex: 0, sampleSize: 0 },
  blink: { blinkRate: 0, avgBlinkDuration: 0, longBlinkCount: 0, anxietyIndex: 0, isDetecting: false },
  combinedIndex: 0,
  level: 'low',
  isMonitoring: false,
  keyboardActive: false,
  blinkActive: false,
};

const AnxietyContext = createContext<AnxietyContextType | null>(null);

export function AnxietyProvider({ children }: { children: ReactNode }) {
  const keyboard = useKeyboardAnxiety();
  const blink = useBlinkDetection();
  const [realityTask, setRealityTask] = useState<any>(null);
  const [showTaskModal, setShowTaskModal] = useState(false);
  const lastReportRef = useRef(0);
  const hasTriggeredRef = useRef(false);
  const combinedIndexRef = useRef(0);

  const combinedIndex = Math.round(
    (keyboard.metrics.anxietyIndex * 0.5) + (blink.metrics.anxietyIndex * 0.5)
  );
  combinedIndexRef.current = combinedIndex;
  const level = combinedIndex > 60 ? 'high' : combinedIndex > 30 ? 'medium' : 'low';
  const isMonitoring = keyboard.isActive || blink.metrics.isDetecting;

  const fetchRealityTask = useCallback(async () => {
    try {
      const res = await profileApi.generateRealityTask({
        anxietyLevel: combinedIndexRef.current,
        context: `键盘${keyboard.metrics.anxietyIndex} 眨眼${blink.metrics.anxietyIndex}`,
      }) as any;
      setRealityTask(res.data);
      setShowTaskModal(true);
    } catch {}
  }, []);

  // 焦虑超标时自动获取任务并弹窗（只触发一次）
  useEffect(() => {
    if (combinedIndex > 50 && !hasTriggeredRef.current && isMonitoring) {
      hasTriggeredRef.current = true;
      fetchRealityTask();
    }
    if (combinedIndex <= 30) {
      hasTriggeredRef.current = false;
    }
  }, [combinedIndex, isMonitoring, fetchRealityTask]);

  const reportToServer = useCallback(async (context?: string) => {
    const now = Date.now();
    if (now - lastReportRef.current < 10000) return; // 最多每10秒上报一次
    lastReportRef.current = now;

    try {
      await fetch('/api/profile/profile/anxiety-report', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
        },
        body: JSON.stringify({
          anxietyIndex: combinedIndex,
          level,
          keyboard: keyboard.metrics,
          blink: blink.metrics,
          context,
        }),
      });
    } catch {}
  }, [combinedIndex, level, keyboard.metrics, blink.metrics]);

  // 定期上报
  useEffect(() => {
    if (!isMonitoring) return;
    const interval = setInterval(() => reportToServer(), 15000);
    return () => clearInterval(interval);
  }, [isMonitoring, reportToServer]);

  const startKeyboard = useCallback(() => keyboard.start(), [keyboard]);
  const stopKeyboard = useCallback(() => keyboard.stop(), [keyboard]);
  const startBlink = useCallback(() => blink.start(), [blink]);
  const stopBlink = useCallback(() => blink.stop(), [blink]);

  const reset = useCallback(() => {
    keyboard.reset();
    hasTriggeredRef.current = false;
    setRealityTask(null);
    setShowTaskModal(false);
  }, [keyboard]);

  const value: AnxietyContextType = {
    state: {
      keyboard: keyboard.metrics,
      blink: blink.metrics,
      combinedIndex,
      level,
      isMonitoring,
      keyboardActive: keyboard.isActive,
      blinkActive: blink.metrics.isDetecting,
    },
    startKeyboard,
    stopKeyboard,
    startBlink,
    stopBlink,
    reportToServer,
    realityTask,
    showTaskModal,
    setShowTaskModal,
    fetchRealityTask,
    reset,
  };

  return (
    <AnxietyContext.Provider value={value}>
      {children}
    </AnxietyContext.Provider>
  );
}

export function useAnxiety() {
  const ctx = useContext(AnxietyContext);
  if (!ctx) throw new Error('useAnxiety must be used within AnxietyProvider');
  return ctx;
}
