/**
 * 表情通道的**采样节律**与**展示平滑**（纯逻辑，不依赖 React / DOM）
 *
 * ## 为什么需要这个文件
 *
 * 旧实现把采样间隔写死成「至少 1 秒，且不小于上次耗时的 2 倍」：
 *
 * ```ts
 * const minGap = Math.max(1000, lastLatency * 2);
 * ```
 *
 * 在慢后端（无 WebGPU → wasm 单线程）上一次推理可能 600~1500ms，
 * 于是实际周期被抬到 **1.2~3 秒**；界面上的三条概率因此几秒才跳一次，
 * 看上去就是「一直在等结果」。而这不是模型能力问题，是节律问题：
 * 串行保护（`inFlight`）已经保证不会排队，固定 1s 下限和 2× 退避都是多余的。
 *
 * 现在改成**贴住后端能力连续推理**：
 *
 * ```
 * gap = clamp(上次耗时 × IDLE_RATIO, FLOOR, CEILING)
 * ```
 *
 * - 后端快（WebGPU，几十毫秒）→ 触到 `FLOOR`（60ms），
 *   而人脸帧本身只在**指标拍**（10fps）到来，所以实际就是**每拍都跑**；
 * - 后端慢（wasm，上千毫秒）→ 按 `1.15 × 耗时` 背靠背跑，
 *   比旧的 `2 × 耗时` 快将近一倍，且不会让 CPU 空转。
 *
 * ## 为什么要平滑，以及为什么它不是"编数据"
 *
 * 每张推理结果都带噪（人脸角度、光照、表情过渡帧）。如果三条概率原样上屏，
 * 以 5~10Hz 刷新会剧烈抖动，反而看不清趋势。这里用**时间常数明确**的指数滑动平均：
 *
 * ```
 * α = 1 - exp(-Δt / τ)        P(t) = P(t-1) + α · (raw(t) - P(t-1))
 * ```
 *
 * τ 是公开常量（`FACE_EMOTION_SMOOTH_TAU_MS`），界面上必须**如实标注**是平滑值；
 * 它只是低通滤波，不引入任何新信息，更不参与任何判定（风险等级一律取融合结果）。
 *
 * ## 失败退避
 *
 * 推理连续失败（例如会话被释放、显存不足）时必须退避，否则每一拍都会
 * 重新发起一次注定失败的推理、并刷一条日志。失败次数越多退避越长，
 * 上限 `MAX_FAILURE_BACKOFF_MS`；只要成功一次立即清零。
 */

import type { FaceEmotionLabel } from '../types/multimodal.types';
import { measuredFps } from './blinkChannel';

/** 最快节律下限（ms）。人脸帧只在指标拍（10fps）到来，所以 60ms 的实际含义是"每拍都跑" */
export const FACE_EMOTION_PACE_FLOOR_MS = 60;

/** 最慢节律上限（ms）。后端再慢也不至于把采样间隔拉到 2 秒以上 */
export const FACE_EMOTION_PACE_CEILING_MS = 2000;

/** 空闲比例：两次发起之间留出 `IDLE_RATIO - 1` 的空隙，避免把 CPU/GPU 跑满影响眨眼通道 */
export const FACE_EMOTION_PACE_IDLE_RATIO = 1.15;

/** 还没有耗时样本时的初始估计（ms）：避免第一次就按 FLOOR 猛冲 */
export const FACE_EMOTION_DEFAULT_LATENCY_MS = 250;

/** 失败退避基数（ms）：第 n 次连续失败退避 `BASE × n`，上限见下 */
export const FACE_EMOTION_FAILURE_BACKOFF_BASE_MS = 1000;

/** 失败退避上限（ms） */
export const FACE_EMOTION_MAX_FAILURE_BACKOFF_MS = 8000;

/** 展示平滑的时间常数（ms）：0.45s 让数值连续移动，又不至于明显滞后于真实表情 */
export const FACE_EMOTION_SMOOTH_TAU_MS = 450;

/** 读数有效期下限（ms）：与旧的 `FACE_EMOTION_READING_TTL_MS` 一致 */
export const FACE_EMOTION_READING_TTL_FLOOR_MS = 1500;

/** 读数有效期上限（ms） */
export const FACE_EMOTION_READING_TTL_CEILING_MS = 8000;

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/**
 * 下一次采样距上一次**发起**至少间隔多少毫秒。
 *
 * @param latencyMs 上一次推理耗时；`<= 0` 或未知时用 `FACE_EMOTION_DEFAULT_LATENCY_MS`
 * @param consecutiveFailures 连续失败次数（0 = 上一拍成功）
 */
export function faceEmotionSampleGapMs(
  latencyMs: number,
  consecutiveFailures = 0,
): number {
  const lat = Number.isFinite(latencyMs) && latencyMs > 0
    ? latencyMs
    : FACE_EMOTION_DEFAULT_LATENCY_MS;
  const base = clamp(lat * FACE_EMOTION_PACE_IDLE_RATIO,
    FACE_EMOTION_PACE_FLOOR_MS, FACE_EMOTION_PACE_CEILING_MS);
  if (consecutiveFailures <= 0) return Math.round(base);
  const backoff = Math.min(
    FACE_EMOTION_MAX_FAILURE_BACKOFF_MS,
    FACE_EMOTION_FAILURE_BACKOFF_BASE_MS * consecutiveFailures,
  );
  return Math.round(base + backoff);
}

export interface FaceEmotionSampleInput {
  /** 上一次**发起**推理的时刻（performance.now）；0 = 还没有过 */
  lastAt: number;
  /** 当前时刻（performance.now） */
  now: number;
  /** 上一次推理耗时 ms */
  latencyMs: number;
  /** 连续失败次数 */
  consecutiveFailures?: number;
}

/** 本拍是否该发起一次推理（纯函数，便于离线验证节律） */
export function shouldSampleFaceEmotion(input: FaceEmotionSampleInput): boolean {
  const { lastAt, now, latencyMs, consecutiveFailures = 0 } = input;
  if (lastAt <= 0) return true;
  return now - lastAt >= faceEmotionSampleGapMs(latencyMs, consecutiveFailures);
}

/**
 * 读数有效期（ms）：必须**长于两个采样周期**，否则慢后端下读数会在两次成功之间过期，
 * 界面就会闪成「暂无新读数」——这正是用户看到的「一直在等结果」的另一种形态。
 */
export function faceEmotionReadingTtlMs(latencyMs: number): number {
  const gap = faceEmotionSampleGapMs(latencyMs, 0);
  return clamp(gap * 2.5, FACE_EMOTION_READING_TTL_FLOOR_MS, FACE_EMOTION_READING_TTL_CEILING_MS);
}

/**
 * 指数滑动平均（时间常数 τ，单位与 `dtMs` 一致）。
 *
 * `τ` 越大越平滑、越滞后；`dtMs <= 0` 时按"不变"处理。
 * `prev` 为 `null` 时直接用 `next` 初始化（避免从 0 缓慢爬升这段假过渡）。
 */
export function smoothProbabilities(
  prev: Record<FaceEmotionLabel, number> | null,
  next: Record<FaceEmotionLabel, number>,
  dtMs: number,
  tauMs: number = FACE_EMOTION_SMOOTH_TAU_MS,
): Record<FaceEmotionLabel, number> {
  if (!prev) return { ...next };
  const dt = Number.isFinite(dtMs) && dtMs > 0 ? dtMs : 0;
  const tau = tauMs > 0 ? tauMs : 1;
  const alpha = dt > 0 ? 1 - Math.exp(-dt / tau) : 0;
  const out = {} as Record<FaceEmotionLabel, number>;
  for (const k of Object.keys(next) as FaceEmotionLabel[]) {
    out[k] = prev[k] + alpha * (next[k] - prev[k]);
  }
  // 归一化：EMA 保序但不严格保和，四舍五入/中断后可能偏离 1
  const sum = (Object.values(out) as number[]).reduce((s, v) => s + v, 0);
  if (sum > 0) for (const k of Object.keys(out) as FaceEmotionLabel[]) out[k] /= sum;
  return out;
}

/** 实测采样率（Hz）：复用眨眼通道的时间戳窗口估计，避免两处各写一份 */
export function measuredSampleHz(timestamps: readonly number[], windowMs = 4000): number {
  return measuredFps(timestamps, windowMs);
}
