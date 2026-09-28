/**
 * 眨眼通道的采样与质量策略（纯逻辑，不依赖 React / DOM）
 *
 * ## 为什么需要这个文件
 *
 * 眨眼是本平台唯一「由时间分辨率决定准不准」的指标：
 *   - 真人眨眼 100–400ms，原实现以 100ms（10fps）采样 → 闭眼时长只能取到
 *     100ms 的整数倍，`avgBlinkDuration` 与 `longBlinkCount(>400ms)` 随之失真，
 *     而这两项直接给 `anxietyIndex` 加分（见 `blinkLogic.ts` 的 stats）。
 *   - 实测（`tools/blink_eval/sampling_effect.js`，用真实状态机跑）：
 *     10fps 下眨眼**事件召回**为 95.5%（ramp 信号）、**时长量化粒度** 100ms；
 *     30fps 下分别是 100% 与 33ms。
 *
 * 因此这里把「眨眼采样率」和「其余面部指标的计算率」**拆成两层**：
 *   - 眨眼通道：目标 30fps（仅取本帧已有的 blendshape/EAR，几乎零额外计算）
 *   - 指标通道：维持 10fps（AU 映射、情绪映射、React 状态更新）
 * 代价只有 MediaPipe 推理次数 ×3（GPU delegate 下约 5–10ms/帧），
 * 而收益是时长精度 3 倍、事件召回回到 100%。
 *
 * ## 为什么要自适应降级
 *
 * 低端笔记本/手机 30fps 跑不动时，硬顶只会让 `setInterval` 排队、时间戳抖动，
 * 反而让「时长」更不可信。所以给一条 30 → 15 → 10 fps 的阶梯：
 * 连续 N 次推理耗时超过间隔预算就降档，连续 M 次很轻松就尝试升档。
 * 无论降到哪一档，都会把**实际帧率**如实上报，让下游的 `signalQuality` 与
 * 「时长量化粒度」可解释 —— 不假装 30fps。
 */

/** 采样率阶梯（fps），从高到低 */
export const BLINK_RATE_LADDER = [30, 15, 10] as const;

/** 眨眼通道默认目标帧率 */
export const DEFAULT_BLINK_FPS = BLINK_RATE_LADDER[0];

/** 其余面部指标（AU/情绪/UI）的计算率，保持原实现口径 */
export const METRICS_CHANNEL_HZ = 10;

/** 推理耗时占采样间隔的比例上限：超过即视为跑不动 */
const BUSY_RATIO = 0.6;
/** 推理耗时占采样间隔的比例下限：低于它才考虑升档 */
const IDLE_RATIO = 0.3;
/** 连续多少次慢/快才换档（迟滞，避免抖动） */
export const DOWN_AFTER = 3;
export const UP_AFTER = 10;

/** fps → 采样间隔 ms */
export function intervalMsFor(fps: number): number {
  return fps > 0 ? Math.round(1000 / fps) : 0;
}

/** 指标通道每多少拍计算一次（把 30fps 的眨眼通道降成 10fps 的指标通道） */
export function metricsEveryN(blinkFps: number, metricsHz: number = METRICS_CHANNEL_HZ): number {
  if (blinkFps <= 0 || metricsHz <= 0) return 1;
  return Math.max(1, Math.round(blinkFps / metricsHz));
}

/** 本拍是否该做昂贵的指标计算 */
export function shouldComputeMetrics(tick: number, every: number): boolean {
  return every <= 1 || tick % every === 0;
}

/** 从时间戳序列估计实测帧率（需要 ≥2 个样本且跨度 > 0） */
export function measuredFps(timestamps: readonly number[], windowMs = 3000): number {
  if (!timestamps || timestamps.length < 2) return 0;
  const now = timestamps[timestamps.length - 1];
  let first = 0;
  for (let i = timestamps.length - 1; i >= 0; i--) {
    if (now - timestamps[i] > windowMs) break;
    first = i;
  }
  if (timestamps.length - first < 2) return 0;
  const span = now - timestamps[first];
  if (span <= 0) return 0;
  return Math.round(((timestamps.length - first - 1) / span) * 1000 * 10) / 10;
}

/** 时长量化粒度：不插值时，闭眼时长只能取采样间隔的整数倍 */
export function durationQuantumMs(fps: number): number {
  return fps > 0 ? Math.round(1000 / fps) : 0;
}

export interface RateDecisionInput {
  /** 当前档位 fps */
  currentFps: number;
  /** 最近一次「推理 + 信号下发」的耗时 ms */
  inferMs: number;
  /** 已连续多少次超预算 */
  consecutiveSlow: number;
  /** 已连续多少次很轻松 */
  consecutiveFast: number;
}

export interface RateDecision {
  /** 下一拍应使用的 fps */
  fps: number;
  /** 调用方需要维护的两个计数器 */
  consecutiveSlow: number;
  consecutiveFast: number;
}

/**
 * 自适应帧率决策（纯函数）。
 *
 * - 连到 `DOWN_AFTER` 次耗时 > 间隔 × 0.6 → 降一档（已是最低档则保持）
 * - 连到 `UP_AFTER` 次耗时 < 间隔 × 0.3 → 升一档（已是最高档则保持）
 * - 其余情况维持
 */
export function nextBlinkRate(input: RateDecisionInput): RateDecision {
  const { currentFps, inferMs, consecutiveSlow, consecutiveFast } = input;
  const idx = Math.max(0, BLINK_RATE_LADDER.indexOf(currentFps as (typeof BLINK_RATE_LADDER)[number]));
  const interval = intervalMsFor(currentFps);

  if (inferMs > interval * BUSY_RATIO) {
    const slow = consecutiveSlow + 1;
    if (slow >= DOWN_AFTER && idx < BLINK_RATE_LADDER.length - 1) {
      return { fps: BLINK_RATE_LADDER[idx + 1], consecutiveSlow: 0, consecutiveFast: 0 };
    }
    return { fps: currentFps, consecutiveSlow: slow, consecutiveFast: 0 };
  }

  if (inferMs < interval * IDLE_RATIO) {
    const fast = consecutiveFast + 1;
    if (fast >= UP_AFTER && idx > 0) {
      return { fps: BLINK_RATE_LADDER[idx - 1], consecutiveSlow: 0, consecutiveFast: 0 };
    }
    return { fps: currentFps, consecutiveSlow: 0, consecutiveFast: fast };
  }

  return { fps: currentFps, consecutiveSlow: 0, consecutiveFast: 0 };
}

export interface BlinkChannelQualityInput {
  /** 信号来源基准分：blendshape 0.9 / ear 0.5 */
  sourceBase: number;
  /** 实测眨眼通道帧率 */
  measuredFps: number;
  /** 目标帧率 */
  targetFps: number;
  /** 近 60 秒面部帧覆盖率 0–1（实际到帧 / 期望到帧） */
  coverage: number;
}

/**
 * 眨眼通道信号质量（0–1）。
 *
 * `帧率达成率` 用 `min(1, 实测/目标)` 而不是布尔：15fps 不是"失败"，
 * 而是"时长精度只有 67ms"—— 应该体现在分数里，而不是被当成 0 或 1。
 */
export function blinkChannelQuality(input: BlinkChannelQualityInput): number {
  const { sourceBase, measuredFps: fps, targetFps, coverage } = input;
  const rateRatio = targetFps > 0 ? Math.min(1, Math.max(0, fps / targetFps)) : 0;
  const cov = Math.min(1, Math.max(0, coverage));
  return Math.round(sourceBase * rateRatio * cov * 100) / 100;
}
