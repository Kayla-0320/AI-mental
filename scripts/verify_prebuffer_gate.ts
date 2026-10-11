/**
 * 开场预缓冲**触发条件**的离线回归 —— 守住"每轮只生效一次"。
 *
 * ## 为什么专门守这一条（2026-10-11 用户："直接检查预缓冲的触发代码，
 * ## 确认它是否通过'每轮已启动播放'的标记严格限制为一次"）
 *
 * 用户这个怀疑被证实了。原判据是 `nextStartTime <= 0`，而 `nextStartTime`
 * **每次队列被抽干都会归零** —— CosyVoice3 阵发式供音的间隙里，队列必然短暂为空，
 * 于是下一块到达时又被判成"开场"、**再等一整个预缓冲**。后果不是"多等一点"，
 * 而是把要治的卡顿放大（本该在断档后立刻接上的音频又推迟 300ms）。
 *
 * 这个脚本把四个关键场景钉死，**不需要浏览器**：
 *
 *   组 1：一轮里第一块等待、后续块不等待（基本语义）
 *   组 2：⭐ 阵发间隙（队列被抽干）**不得**重复触发（原缺陷的正面回归）
 *   组 3：一轮里"提前到达的后续块"不吃掉预缓冲（清账要看队列是否起飞）
 *   组 4：新一轮（`armTurn`）重新生效一次 —— 即"每轮恰好一次"
 *
 * 用法（仓库根目录）：
 *   node "server/node_modules/tsx/dist/cli.mjs" scripts/verify_prebuffer_gate.ts
 * 退出码 1 表示任一断言失败（可直接接进 CI）。
 */
import { PrebufferGate } from '../client/src/services/prebufferGate';

let failed = 0;
function check(ok: boolean, what: string, detail = ''): void {
  if (ok) {
    console.log(`  ✅ ${what}${detail ? `  ${detail}` : ''}`);
  } else {
    failed += 1;
    console.log(`  ❌ ${what}${detail ? `  ${detail}` : ''}`);
  }
}

/**
 * 模拟播放器的排程循环：把每块音频喂进去，按 gate 的决定算"起点推后多少"。
 *
 * ⚠️ 必须按 `TtsPlayer.enqueueChunk` 的真实算术来（`nextStartTime` = 已排到的播放位置，
 * 与**到达时刻**比较），并用**绝对虚拟时钟**推进 —— 第一版这里忘了推进 `now`，
 * 于是"第二块"被算到 1.8 秒之后，测试自己报假失败。
 */
function makeScheduler(prebufferMs: number) {
  const gate = new PrebufferGate(true);
  let nextStartTime = 0; // 0 = 播放队列空
  let waits = 0;
  let activations = 0;
  return {
    gate,
    get waits() {
      return waits;
    },
    get activations() {
      return activations;
    },
    armTurn(): void {
      gate.armTurn();
    },
    /** 在**绝对时刻** `atMs` 排一块 `chunkMs` 长的音频 */
    arrival(atMs: number, chunkMs: number): void {
      const queuedAudioMs = Math.max(0, nextStartTime - atMs);
      const want = prebufferMs > 0 && gate.decide(queuedAudioMs);
      const base = want ? atMs + prebufferMs : nextStartTime > 0 ? nextStartTime : atMs;
      const startAt = Math.max(atMs, base);
      nextStartTime = startAt + chunkMs;
      if (want) {
        waits += 1;
        activations += 1;
      }
    },
  };
}

/** 真机形态的一轮：首块 ≈1.5s，之后每 ≈1.4s 来一块 ≈0.9s 音频（阵发式） */
function burstTurn(s: ReturnType<typeof makeScheduler>, startAtMs: number): void {
  s.arrival(startAtMs, 1500);
  s.arrival(startAtMs + 1400, 900);
  s.arrival(startAtMs + 2800, 900);
}

console.log('组 1：一轮里"第一块等、后续块不等"');
{
  const s = makeScheduler(300);
  s.arrival(0, 1500); // 开场第一块（真机约 1.5s 音频）
  s.arrival(200, 171); // 预缓冲窗口内到达的后续块
  s.arrival(400, 171);
  check(s.waits === 1, '一轮只生效一次', `waits=${s.waits}`);
  check(s.gate.pending === false, '排上音频后"欠账"已清（本轮不再欠）', `pending=${s.gate.pending}`);
}

console.log('组 2：⭐ 阵发间隙（队列被抽干）不得重复触发 —— 原缺陷的正面回归');
{
  const s = makeScheduler(300);
  s.arrival(0, 400); // 第一块（小）
  // t=400：这块播完，队列抽干（真机：阵间隙）
  s.arrival(400, 400);
  check(s.waits === 1, '间隙后到达的块不重复等待（旧实现会在这里再等 300ms）', `waits=${s.waits}`);
  s.arrival(800, 400); // 第二次间隙
  check(s.waits === 1, '第二次间隙同样不重复', `waits=${s.waits}`);
  s.arrival(1200, 400); // 第三次
  check(s.waits === 1, '多次间隙也不会多生效', `waits=${s.waits}`);
}

console.log('组 3：真机形态的一轮里，预缓冲只在开场发生一次');
{
  const s = makeScheduler(300);
  burstTurn(s, 0);
  check(s.waits === 1, '一整轮（3 阵、含 2 个间隙）只生效 1 次', `waits=${s.waits}`);
  // 时钟推进到整轮播完，再确认没有"补触发"
  check(s.gate.pending === false, '整轮结束后仍不欠账（要等 armTurn）');
}

console.log('组 4：每轮恰好一次（armTurn 才重新欠账）');
{
  const s = makeScheduler(300);
  burstTurn(s, 0); // 第 1 轮
  check(s.waits === 1, '第 1 轮生效 1 次', `waits=${s.waits}`);
  s.armTurn(); // 用户又说了话 ⇒ 真正开第 2 轮
  burstTurn(s, 5000);
  check(s.waits === 2, '第 2 轮再生效 1 次（每轮一次，不是整通只一次）', `waits=${s.waits}`);
  s.armTurn(); // 第 3 轮
  burstTurn(s, 12000);
  check(s.waits === 3, '第 3 轮同样恰好 1 次', `waits=${s.waits}`);
  check(s.activations === s.waits, '生效次数与等待次数一致（读数可信）', `${s.activations}/${s.waits}`);
}

console.log('组 5：一轮里第一批就来了多块（服务端流式）也不多吃预缓冲');
{
  const s = makeScheduler(300);
  s.arrival(0, 1000); // 开场块
  s.arrival(150, 900); // 还在预缓冲窗口内
  s.arrival(300, 900); // 窗口刚结束
  check(s.waits === 1, '窗口内到达的后续块不重复等待', `waits=${s.waits}`);
}

console.log('组 6：预缓冲为 0 时整个机制是空操作（旧行为可回退）');
{
  const s = makeScheduler(0);
  burstTurn(s, 0);
  s.armTurn();
  burstTurn(s, 5000);
  check(s.waits === 0, '一次都不生效', `waits=${s.waits}`);
}

console.log('');
if (failed > 0) {
  console.log(`❌ 预缓冲触发条件回归：${failed} 项失败`);
  process.exit(1);
}
console.log('✅ 预缓冲触发条件回归：全部通过（每轮严格只生效一次）');
