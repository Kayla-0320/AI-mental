/**
 * 眨眼判定的纯逻辑（不依赖 React / DOM）
 *
 * 从 useBlinkDetection 中抽出，目的是让判定规则可被独立测试 —— 之前这段逻辑
 * 埋在 requestAnimationFrame 回调里，除了接真摄像头没有别的验证手段。
 *
 * 判定链：连续信号（blendshape 闭合度 或 亮度）→ 布尔「闭眼」→ 迟滞去抖
 *        → 时长合理性过滤 → 眨眼事件 → 滚动指标
 */

export type BlinkMethod = 'blendshape' | 'brightness' | 'none';

export interface BlinkEvent {
  /** 睁眼（闭眼区间结束）时刻，ms */
  time: number;
  /** 闭眼持续时长，ms */
  duration: number;
}

export interface BlinkParams {
  /** 进入闭眼的上阈值 */
  onThreshold: number;
  /** 退出闭眼的阈值（< onThreshold，构成迟滞） */
  offThreshold: number;
  /** 计入眨眼的最短闭眼时长，ms */
  minMs: number;
  /** 计入眨眼的最长闭眼时长，ms（更长视为休息性闭眼或检测丢失） */
  maxMs: number;
  /** 两次眨眼之间的最小间隔，ms（去抖） */
  minGapMs: number;
}

/** MediaPipe blendshape（eyeBlinkLeft/Right）参数 */
export const BLENDSHAPE_PARAMS: BlinkParams = {
  onThreshold: 0.5,
  offThreshold: 0.35,
  minMs: 60,
  maxMs: 800,
  minGapMs: 100,
};

/** 亮度降级法参数（阈值由滚动统计动态给出，此处只约束时长与去抖） */
export const BRIGHTNESS_PARAMS: BlinkParams = {
  onThreshold: 0,
  offThreshold: 0,
  minMs: 80,
  maxMs: 600,
  minGapMs: 100,
};

export interface BlinkStats {
  blinkRate: number;
  avgBlinkDuration: number;
  longBlinkCount: number;
  anxietyIndex: number;
  /** 最近 60 秒内计入的眨眼次数 */
  sampleCount: number;
}

export class BlinkStateMachine {
  private closed = false;
  private startMs = 0;
  private lastEndMs = 0;
  private readonly events: BlinkEvent[] = [];

  constructor(private readonly params: BlinkParams = BLENDSHAPE_PARAMS) {}

  /**
   * 把连续闭合度转成布尔「是否闭眼」，带迟滞。
   * 已处于闭眼状态时要用更低的 offThreshold 才判定睁开，
   * 避免分数在阈值附近抖动导致一次眨眼被切成多次。
   */
  isClosed(score: number): boolean {
    return this.closed
      ? score > this.params.offThreshold
      : score > this.params.onThreshold;
  }

  /**
   * 推进状态机。closed 为当前帧的闭眼判定（可先经 isClosed 迟滞处理）。
   * @returns 本帧若完成一次有效眨眼，返回该事件；否则 null
   */
  update(closed: boolean, nowMs: number): BlinkEvent | null {
    if (closed) {
      if (!this.closed) {
        this.closed = true;
        this.startMs = nowMs;
      }
      return null;
    }

    if (!this.closed) return null;
    this.closed = false;

    const duration = nowMs - this.startMs;
    if (duration < this.params.minMs || duration >= this.params.maxMs) {
      // 过短视为噪声/抖帧；过长视为休息性闭眼或检测丢失，均不计入眨眼
      this.lastEndMs = nowMs;
      return null;
    }
    if (this.startMs - this.lastEndMs < this.params.minGapMs) return null;

    const event: BlinkEvent = { time: nowMs, duration };
    this.events.push(event);
    this.lastEndMs = nowMs;
    return event;
  }

  /** 强行结束当前闭眼区间（例如人脸消失），不产生眨眼事件 */
  breakClosed(nowMs: number): void {
    if (this.closed) {
      this.closed = false;
      this.lastEndMs = nowMs;
    }
  }

  reset(): void {
    this.closed = false;
    this.startMs = 0;
    this.lastEndMs = 0;
    this.events.length = 0;
  }

  get all(): readonly BlinkEvent[] {
    return this.events;
  }

  /**
   * 滚动指标。公式与原实现逐项一致，保证改造前后口径可比。
   * @param nowMs 当前时刻
   * @param sessionStartMs 会话起点（用于把眨眼数换算成次/分）
   */
  stats(nowMs: number, sessionStartMs: number): BlinkStats {
    const elapsed = (nowMs - sessionStartMs) / 1000;
    const recent = this.events.filter((e) => nowMs - e.time < 60000);

    const blinkRate =
      elapsed > 0 ? Math.round((recent.length / Math.max(elapsed, 1)) * 60) : 0;
    const avgBlinkDuration = recent.length
      ? Math.round(recent.reduce((s, e) => s + e.duration, 0) / recent.length)
      : 0;
    const longBlinkCount = recent.filter((e) => e.duration > 400).length;

    let anxietyIndex = 0;
    if (blinkRate > 25) anxietyIndex += Math.min(40, (blinkRate - 25) * 4);
    else if (blinkRate < 10 && blinkRate > 0) anxietyIndex += Math.min(30, (10 - blinkRate) * 3);
    if (longBlinkCount > 2) anxietyIndex += Math.min(30, longBlinkCount * 10);
    if (avgBlinkDuration > 350) anxietyIndex += 15;
    anxietyIndex = Math.min(100, anxietyIndex);

    return {
      blinkRate,
      avgBlinkDuration,
      longBlinkCount,
      anxietyIndex,
      sampleCount: recent.length,
    };
  }
}

/** 亮度法的动态阈值：滚动均值 − max(1.5σ, 5)。保持与原实现一致。 */
export function brightnessThreshold(history: readonly number[]): number {
  if (history.length === 0) return 0;
  const avg = history.reduce((s, v) => s + v, 0) / history.length;
  const variance = history.reduce((s, v) => s + (v - avg) ** 2, 0) / history.length;
  return avg - Math.max(Math.sqrt(variance) * 1.5, 5);
}
