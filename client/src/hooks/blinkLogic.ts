/**
 * 眨眼判定的纯逻辑（不依赖 React / DOM）
 *
 * 从 useBlinkDetection 中抽出，目的是让判定规则可被独立测试 —— 之前这段逻辑
 * 埋在 requestAnimationFrame 回调里，除了接真摄像头没有别的验证手段。
 *
 * 判定链：连续信号（blendshape 闭合度 或 亮度）→ 布尔「闭眼」→ 迟滞去抖
 *        → 时长合理性过滤 → 眨眼事件 → 滚动指标
 */

/**
 * 眨眼信号来源：
 * - `blendshape`：MediaPipe eyeBlinkLeft/Right 闭合度（主路径）
 * - `ear`：面部关键点眼睛纵横比阈值（blendshape 缺失时的降级路径）
 * - `brightness`：画面固定矩形亮度（仅独立眨眼页在 MediaPipe 完全不可用时兜底）
 */
export type BlinkMethod = 'blendshape' | 'ear' | 'brightness' | 'none';

export interface BlinkEvent {
  /** 睁眼（闭眼区间结束）时刻，ms */
  time: number;
  /** 闭眼持续时长，ms */
  duration: number;
  /**
   * 该时长是否由「亚采样插值」得到。
   *
   * 采样是离散的：闭眼区间的起止只能落在采样点上，于是时长被量化成采样间隔的整数倍
   * （30fps → 33ms，10fps → 100ms）。当调用方把连续分数一并传入时，状态机用
   * 「阈值穿越点」在两帧之间做线性插值，把量化误差从 ±一个采样间隔降到 ±半帧，
   * 从而让 `avgBlinkDuration` / `longBlinkCount` 不再被采样率绑死。
   */
  estimated?: boolean;
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
  /**
   * 是否用**亚采样插值**估计闭眼时长（默认 `false`，即保持历史行为）。
   *
   * 实测结论（`tools/blink_eval/sampling_effect.js` + 分桶诊断，二者都用真实状态机）：
   *   - 当分数在采样间隔内近似**线性**过渡时，插值把时长 MAE 降低约 25–33%；
   *   - 当过渡是**强非线性**（如 sin^0.6 型 blendshape 响应）时，在 30fps 下插值
   *     与不插值基本打平（±10ms 量级，方向还会随采样相位翻转）。
   *
   * 原因是旧写法的两个端点误差方向相反、会互相抵消：进入时刻被推后（偏短）、
   * 退出时刻被推后（偏长）。插值缩小了每个端点的误差，却打断了这个抵消。
   *
   * 由于 `avgBlinkDuration` / `longBlinkCount` 会直接给 `anxietyIndex` 加分，
   * **在没有真实摄像头数据的验证之前，这里保持关闭**；验证方法见
   * `docs/blink_signal_source_selection.md` 的步 A（自录片段 + `sampling_effect.js --csv`）。
   */
  interpolateDurations?: boolean;
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
  /**
   * 近期采样间隔的中位数（ms）。调用方（useEyeTracking）用它如实上报
   * 「本次测量的时长量化粒度」—— 10fps 采样下这个值就是 100ms，
   * 界面/证据里应当能看到，而不是让用户以为时长精度是毫秒级。
   */
  sampleIntervalMs: number;
  /** 窗口内由插值得到时长的比例 0–1（越低说明采样越粗或分数不可用） */
  interpolatedShare: number;
}

/** 采样间隔统计窗口（约 4 秒 @30fps） */
const MAX_INTERVALS = 120;

export class BlinkStateMachine {
  private closed = false;
  private startMs = 0;
  private lastEndMs = 0;
  private readonly events: BlinkEvent[] = [];
  /** 上一拍的时刻与分数（用于亚采样插值；分数不可用时保持 null） */
  private lastSampleMs = 0;
  private lastScore: number | null = null;
  /** 闭眼起点是否由插值得到 */
  private startEstimated = false;
  /** 近期采样间隔（ms），用于上报时长量化粒度 */
  private readonly intervals: number[] = [];

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
   * 两拍之间「分数穿过阈值」的时刻（线性插值）。
   *
   * @param t0/s0 上一拍的时刻与分数
   * @param t1/s1 本拍的时刻与分数
   * @param threshold 穿越判定用的阈值（进入用 onThreshold，退出用 offThreshold）
   * @returns 插值时刻；分数缺失、分数不变或穿越点不在两拍之间时返回 null
   *          （调用方退回采样点时刻，行为与旧实现完全一致）
   */
  private static interpolate(
    t0: number,
    s0: number | null,
    t1: number,
    s1: number | null,
    threshold: number,
  ): number | null {
    if (s0 === null || s1 === null || t0 <= 0 || t1 <= t0) return null;
    const ds = s1 - s0;
    if (Math.abs(ds) < 1e-6) return null;
    const ratio = (threshold - s0) / ds;
    if (!(ratio >= 0 && ratio <= 1)) return null;
    return t0 + ratio * (t1 - t0);
  }

  /**
   * 推进状态机。
   * @param closed 当前帧的闭眼判定（可先经 isClosed 迟滞处理）
   * @param nowMs  当前帧时刻
   * @param score  当前帧的连续分数（0–1）。传入后启用**亚采样插值**，
   *               把闭眼时长的量化误差从 ±一个采样间隔降到 ±半帧；
   *               不传则退化为旧行为（时长 = 采样点之差）。
   * @returns 本帧若完成一次有效眨眼，返回该事件；否则 null
   */
  update(closed: boolean, nowMs: number, score?: number | null): BlinkEvent | null {
    const useInterp = this.params.interpolateDurations === true;
    const numericScore =
      useInterp && typeof score === 'number' && Number.isFinite(score) ? score : null;
    const prevT = this.lastSampleMs;
    const prevScore = this.lastScore;

    // 采样间隔统计（只用于如实上报量化粒度，不参与判定）
    if (prevT > 0) {
      const dt = nowMs - prevT;
      if (dt > 0) {
        this.intervals.push(dt);
        if (this.intervals.length > MAX_INTERVALS) this.intervals.shift();
      }
    }
    this.lastSampleMs = nowMs;
    this.lastScore = numericScore;

    if (closed) {
      if (!this.closed) {
        this.closed = true;
        const cross = numericScore === null ? null : BlinkStateMachine.interpolate(
          prevT, prevScore, nowMs, numericScore, this.params.onThreshold,
        );
        this.startEstimated = cross !== null;
        this.startMs = cross !== null ? cross : nowMs;
      }
      return null;
    }

    if (!this.closed) return null;
    this.closed = false;

    const cross = numericScore === null ? null : BlinkStateMachine.interpolate(
      prevT, prevScore, nowMs, numericScore, this.params.offThreshold,
    );
    const endMs = cross !== null ? cross : nowMs;
    const duration = Math.max(0, endMs - this.startMs);
    const estimated = this.startEstimated && cross !== null;
    this.startEstimated = false;

    if (duration < this.params.minMs || duration >= this.params.maxMs) {
      // 过短视为噪声/抖帧；过长视为休息性闭眼或检测丢失，均不计入眨眼
      this.lastEndMs = nowMs;
      return null;
    }
    // `lastEndMs === 0` 表示「此前没有眨眼」，而不是「上一次眨眼结束于 t=0」。
    // 旧写法会把会话开头的第一次眨眼按「间隔不足」丢掉 —— 时间戳从 0 附近起步时
    // 就会命中（亚采样插值把起点提前后更容易命中）。间隔检查只对真实的上一次眨眼生效。
    if (this.lastEndMs > 0 && this.startMs - this.lastEndMs < this.params.minGapMs) return null;

    const event: BlinkEvent = { time: nowMs, duration: Math.round(duration), estimated };
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
    this.startEstimated = false;
  }

  reset(): void {
    this.closed = false;
    this.startMs = 0;
    this.lastEndMs = 0;
    this.events.length = 0;
    this.lastSampleMs = 0;
    this.lastScore = null;
    this.startEstimated = false;
    this.intervals.length = 0;
  }

  get all(): readonly BlinkEvent[] {
    return this.events;
  }

  /**
   * 滚动指标。公式与原实现逐项一致，仅修正眨眼频率的分母。
   *
   * 原实现 `blinkRate = recent.length / elapsed * 60`：`recent` 是「最近 60 秒」的
   * 眨眼数，分母却是「会话总时长」，量纲不一致 —— 会话超过 60 秒后频率会被系统性
   * 摊薄（10 分钟会话里稳定的 15 次/分会显示成 1.5 次/分）。现改为按实际统计窗口
   * `min(elapsed, 60)` 秒折算，60 秒以内与原来一致。
   *
   * @param nowMs 当前时刻
   * @param sessionStartMs 会话起点（用于把眨眼数换算成次/分）
   */
  stats(nowMs: number, sessionStartMs: number): BlinkStats {
    const elapsed = (nowMs - sessionStartMs) / 1000;
    const recent = this.events.filter((e) => nowMs - e.time < 60000);
    const windowSec = Math.min(Math.max(elapsed, 1), 60);

    const blinkRate =
      windowSec > 0 ? Math.round((recent.length / windowSec) * 60) : 0;
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
      sampleIntervalMs: this.medianIntervalMs(),
      interpolatedShare: recent.length
        ? Math.round((recent.filter((e) => e.estimated).length / recent.length) * 100) / 100
        : 0,
    };
  }

  /** 近期采样间隔中位数（ms）；样本不足时返回 0（表示未知，不猜） */
  private medianIntervalMs(): number {
    if (this.intervals.length < 2) return 0;
    const sorted = [...this.intervals].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    const median = sorted.length % 2 === 0
      ? (sorted[mid - 1] + sorted[mid]) / 2
      : sorted[mid];
    return Math.round(median);
  }
}

/** 亮度法的动态阈值：滚动均值 − max(1.5σ, 5)。保持与原实现一致。 */
export function brightnessThreshold(history: readonly number[]): number {
  if (history.length === 0) return 0;
  const avg = history.reduce((s, v) => s + v, 0) / history.length;
  const variance = history.reduce((s, v) => s + (v - avg) ** 2, 0) / history.length;
  return avg - Math.max(Math.sqrt(variance) * 1.5, 5);
}
