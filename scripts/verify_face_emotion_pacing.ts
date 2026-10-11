/**
 * 表情通道**采样节律**的离线验证：把"改前 vs 改后"的刷新率算出来。
 *
 * 为什么需要它：用户反馈的是两件可量化的事 ——「一次识别间隔太长」和
 * 「开心几乎识别不出来」。前者完全由 `faceEmotionPacing.ts` 的节律决定，
 * 而它是不依赖 DOM 的纯逻辑，所以**可以直接在 Node 下把刷新率跑出来**，
 * 不需要开摄像头、也不需要人来"看着感觉快了"。
 *
 * 本脚本做四件事：
 *   1. 逐档推理耗时，对比旧策略 `max(1000, 耗时 × 2)` 与新策略的**实际采样周期**；
 *   2. 用真实的人脸帧节律（指标拍 10fps）驱动一遍，数出 5 秒内成功采样多少次；
 *   3. 断言读数有效期严格大于两个采样周期（否则界面会在两次成功之间闪「暂无新读数」）；
 *   4. 断言平滑是**单调收敛、和为 1、首帧等于原始值**（不制造从 0 爬升的假过渡）。
 *
 * 用法：
 *   node "server/node_modules/tsx/dist/cli.mjs" scripts/verify_face_emotion_pacing.ts
 * 退出码 1 表示任一断言失败（可直接接进 CI）。
 */
import {
  FACE_EMOTION_PACE_FLOOR_MS,
  FACE_EMOTION_READING_TTL_FLOOR_MS,
  FACE_EMOTION_SMOOTH_TAU_MS,
  faceEmotionReadingTtlMs,
  faceEmotionSampleGapMs,
  shouldSampleFaceEmotion,
  smoothProbabilities,
} from '../client/src/hooks/faceEmotionPacing';
import type { FaceEmotionLabel } from '../client/src/types/multimodal.types';

/** 人脸帧只在「指标拍」到来：眨眼通道 30fps / 指标通道 10Hz ⇒ 100ms 一拍 */
const METRICS_TICK_MS = 100;

/** 旧策略：与 `useFaceEmotion` 的历史实现逐字一致，用于对照 */
function legacyGapMs(latencyMs: number): number {
  return Math.max(1000, latencyMs * 2);
}

let failures = 0;
function check(ok: boolean, what: string, detail = ''): void {
  if (ok) {
    console.log(`  ✅ ${what}${detail ? `  ${detail}` : ''}`);
  } else {
    failures += 1;
    console.log(`  ❌ ${what}${detail ? `  ${detail}` : ''}`);
  }
}

/** 用给定的"节律函数"跑 `durationMs` 时长，返回成功采样的时刻列表 */
function simulate(durationMs: number, latencyMs: number, gapOf: (l: number) => number): number[] {
  const out: number[] = [];
  // 用 -Infinity 而不是 0 作哨兵：否则 t=0 与 t=100 两拍都会被判为"首次"，
  // 慢后端下会把结果多算一次（这是本脚本第一版的 bug）。
  let lastAt = Number.NEGATIVE_INFINITY;
  for (let t = 0; t <= durationMs; t += METRICS_TICK_MS) {
    if (t - lastAt < gapOf(latencyMs)) continue;
    out.push(t);
    lastAt = t; // 简化：忽略推理期间的时间流逝（串行保护下不会排队，量与量之间可比）
  }
  return out;
}

console.log('='.repeat(78));
console.log('表情通道采样节律验证（指标拍 100ms / 10Hz 上界）');
console.log('='.repeat(78));
console.log(`下限 FLOOR=${FACE_EMOTION_PACE_FLOOR_MS}ms  平滑 τ=${FACE_EMOTION_SMOOTH_TAU_MS}ms  ` +
  `有效期下限=${FACE_EMOTION_READING_TTL_FLOOR_MS}ms`);

// ── 1/2. 刷新率对照 ────────────────────────────────────────────────────────
console.log('\n[1] 实际采样周期与 5 秒内采样次数：旧策略 vs 新策略');
console.log(`  ${'单次推理'.padEnd(10)}${'旧周期'.padStart(9)}${'新周期'.padStart(9)}` +
  `${'旧次数/5s'.padStart(11)}${'新次数/5s'.padStart(11)}${'倍率'.padStart(8)}`);
const latencies = [30, 60, 100, 200, 400, 600, 1000, 1500, 2500];
for (const lat of latencies) {
  const oldGap = legacyGapMs(lat);
  const newGap = faceEmotionSampleGapMs(lat, 0);
  // 两边都按 100ms 拍去对齐（真实实现就是"每拍检查一次"）
  const oldAligned = simulate(5000, lat, () => oldGap).length;
  const newN = simulate(5000, lat, (l) => faceEmotionSampleGapMs(l, 0)).length;
  console.log(`  ${`${lat}ms`.padEnd(10)}${`${oldGap}ms`.padStart(9)}${`${newGap}ms`.padStart(9)}` +
    `${String(oldAligned).padStart(11)}${String(newN).padStart(11)}` +
    `${(newN / Math.max(1, oldAligned)).toFixed(2).padStart(8)}`);
  check(newGap <= oldGap, `耗时 ${lat}ms：新周期不超过旧周期`,
    `新 ${newGap}ms ≤ 旧 ${oldGap}ms`);
  check(faceEmotionSampleGapMs(lat, 0) <= 2000,
    `耗时 ${lat}ms：新周期不超过 2s 上限`);
}

console.log('\n[2] 关键档位的具体结论');
{
  // 快后端（WebGPU）：应当**每一拍都采样**（10Hz，指标拍本身的上界）
  const fastN = simulate(5000, 30, (l) => faceEmotionSampleGapMs(l, 0)).length;
  check(fastN === 51, `WebGPU(30ms)：5 秒内每拍都跑，共 ${fastN} 次（≈10 次/秒）`);

  // 慢后端（wasm 单线程 1500ms）：旧策略 3 秒 1 次，新策略约 0.6 次/秒
  const oldSlow = simulate(5000, 1500, () => legacyGapMs(1500)).length;
  const newSlow = simulate(5000, 1500, (l) => faceEmotionSampleGapMs(l, 0)).length;
  check(newSlow > oldSlow, `wasm(1500ms)：采样次数由 ${oldSlow} 提升到 ${newSlow}（5 秒内）`);
}

// ── 3. 有效期必须长于两个采样周期 ─────────────────────────────────────────
console.log('\n[3] 读数有效期 vs 采样周期（有效期必须 > 2 个周期，否则界面会闪「暂无新读数」）');
for (const lat of latencies) {
  const gap = faceEmotionSampleGapMs(lat, 0);
  const ttl = faceEmotionReadingTtlMs(lat);
  check(ttl > gap * 2, `耗时 ${lat}ms：有效期 ${ttl}ms > 2 × 周期 ${gap}ms`);
}

// ── 4. 平滑：单调收敛 / 和为 1 / 首帧不造假过渡 ────────────────────────────
console.log('\n[4] 展示平滑（EMA，τ=450ms）');
{
  const first = smoothProbabilities(null, { happy: 0.7, neutral: 0.2, sad: 0.1 }, 0);
  check(Math.abs(first.happy - 0.7) < 1e-9,
    '首帧直接等于模型原始值（不从 0 缓慢爬升）', `happy=${first.happy}`);

  let p = first;
  const seen: number[] = [];
  for (let i = 0; i < 20; i++) {
    // 目标突变到 0.1：EMA 必须单调下降并收敛，不能过冲
    p = smoothProbabilities(p, { happy: 0.1, neutral: 0.8, sad: 0.1 }, 100);
    seen.push(p.happy);
  }
  let monotone = true;
  for (let i = 1; i < seen.length; i++) if (seen[i] > seen[i - 1] + 1e-12) monotone = false;
  check(monotone, '目标下移后单调收敛，无过冲', `0.70 → ${seen[0].toFixed(3)} → ${seen[19].toFixed(3)}`);
  check(seen[19] > 0.1 && seen[19] < 0.15, 'τ=450ms 下 2 秒后基本到位但仍有残差');

  const sumsOk = [first, p].every((v) =>
    Math.abs((v.happy + v.neutral + v.sad) - 1) < 1e-9);
  check(sumsOk, '三类概率和恒为 1（EMA 后重新归一化）');

  // 时间常数必须真的起作用：Δt = τ 时 α ≈ 0.632
  const oneTau = smoothProbabilities({ happy: 0, neutral: 1, sad: 0 },
    { happy: 1, neutral: 0, sad: 0 }, FACE_EMOTION_SMOOTH_TAU_MS);
  check(Math.abs(oneTau.happy - 0.6321) < 0.01,
    `Δt=τ 时前进 63.2%（实测 ${(oneTau.happy * 100).toFixed(1)}%）`);
}

// ── 5. shouldSampleFaceEmotion 的边界 ──────────────────────────────────────
console.log('\n[5] shouldSampleFaceEmotion 边界');
check(shouldSampleFaceEmotion({ lastAt: 0, now: 12345, latencyMs: 0 }), '首次调用必定采样');
check(!shouldSampleFaceEmotion({ lastAt: 1000, now: 1000 + FACE_EMOTION_PACE_FLOOR_MS - 1, latencyMs: 30 }),
  '未到下界时不采样');
check(shouldSampleFaceEmotion({ lastAt: 1000, now: 1000 + FACE_EMOTION_PACE_FLOOR_MS, latencyMs: 30 }),
  '刚好到下界时采样');
{
  const g0 = faceEmotionSampleGapMs(300, 0);
  const g3 = faceEmotionSampleGapMs(300, 3);
  check(g3 > g0, '连续失败会退避', `${g0}ms → ${g3}ms`);
  check(faceEmotionSampleGapMs(300, 999) <= g0 + 8000 + 1,
    '失败退避有上限（8s 基数封顶）');
}

// ── 6. 端到端：同一段 5 秒，旧/新策略各能上屏多少个"不同的数字" ─────────────
console.log('\n[6] 用户观感：5 秒内三条百分比会变化多少次（= 成功采样次数）');
for (const lat of [30, 200, 600, 1500]) {
  const oldN = simulate(5000, lat, () => legacyGapMs(lat)).length;
  const newN = simulate(5000, lat, (l) => faceEmotionSampleGapMs(l, 0)).length;
  console.log(`  单次推理 ${String(lat).padStart(4)}ms：旧 ${String(oldN).padStart(2)} 次` +
    ` → 新 ${String(newN).padStart(2)} 次（约每 ${(5000 / Math.max(1, newN)).toFixed(0)}ms 变一次）`);
}

console.log('\n' + '='.repeat(78));
if (failures === 0) {
  console.log('✅ 全部断言通过：节律、有效期、平滑三项行为符合设计。');
} else {
  console.log(`❌ ${failures} 项断言失败`);
  process.exitCode = 1;
}
