/**
 * 预缓冲修复的**对照实验**：旧判据 vs 新状态机，跑真机实测出来的阵发时间线。
 *
 * ## 用户要验收的四项（2026-10-11）
 *
 * > 1. 一轮回复中即使出现多次供音间隙，`TTS·预缓冲生效` 也只增加一次；
 * > 2. 多轮对话中，生效次数不超过实际产生语音的轮数；
 * > 3. 阵间隙恢复供音时，不再额外等待 300ms；
 * > 4. **缓冲见底次数、句内见底时长和实际听感都没有反弹**。
 *
 * 前三条是"修复生效"的证据；第 4 条是"没有把原来的问题换成另一个"的证据。
 * 用户特别强调："预缓冲生效次数正确，只能证明重复触发的问题受到控制，
 * 并不能单独证明卡顿已经解决。仍需要观察实际水位和听感。"
 *
 * ⇒ 所以这个脚本同时算 **生效次数** 与 **播放水位**（最低水位、见底累计），
 * 把"旧判据 / 新状态机 / 干脆关掉预缓冲"三种跑在**同一条时间线**上对比。
 *
 * 输入是 `probe_tts_supply.mjs` 实测出来的真实形态：首块 ≈1.6s 到且音频 ≈1.5s，
 * 之后每 ≈1.4s 来一块 ≈0.9s（阵发），整轮有效 RTF ≈0.6。
 *
 * 用法（仓库根目录）：
 *   node "server/node_modules/tsx/dist/cli.mjs" scripts/verify_prebuffer_ab.ts
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
 * 真机形态的一轮：`[到达时刻 ms, 该块音频时长 ms]`。
 *
 * 数值来自 `probe_tts_supply.mjs` 实测（本机）：
 *   · 首块 1.5~1.8s 到达，携带 ≈1.5s 音频（开头那一大块）；
 *   · 其后每 0.7~2.4s 来一小块，每块只有 0.2~0.4s 音频（阵发式供音）。
 *
 * ⚠️ 第一版把后续块写成 0.9s/1.4s，结果**队列从来不会抽干** —— 那样旧判据的
 * "队列空就当开场"根本不会触发，A/B 也测不出差别（测试自己写错了基数）。
 * 块必须**小**到能被播放吃掉，才复现真机的阵间隙。
 */
function burstTurn(startMs: number, bursts = 5): Array<[number, number]> {
  const out: Array<[number, number]> = [[startMs, 1500]]; // 首块：大
  for (let i = 1; i <= bursts; i += 1) {
    // 每 700ms 来一块 400ms 音频 ⇒ 供音速率 0.57 < 播放速率 1，队列必然被消耗
    out.push([startMs + 700 * i, 400]);
  }
  return out;
}

/** 判据模式 */
type Mode = 'fixed' | 'legacy' | 'off';

interface SimResult {
  /** 预缓冲生效次数（用户核对的 TTS·预缓冲生效） */
  activations: number;
  /** 实际产生语音的轮数 */
  speechTurns: number;
  /** 播放水位最低值（毫秒，负数 = 见底） */
  minLevel: number;
  /** 见底累计时长（毫秒）—— 耳朵里"卡"的总量 */
  starveTotalMs: number;
  /** 首声延迟（第一块音频真正开始播的时刻） */
  firstSoundMs: number;
  /**
   * **被预缓冲额外推迟的总时长**（毫秒）——本轮修复的直接收益。
   *
   * 每次生效都会把"本该接上"的音频往后推 `prebufferMs`；只有**第一次**（真开场）
   * 那次是有意义的，其后每一次都是白等。所以这个数越大，说明白白拖长的听觉时间越多。
   */
  extraDelayMs: number;
}

/**
 * 模拟播放器：把多轮音频块按到达时刻喂进去，算出"起点排在哪里"与水位曲线。
 *
 * 时钟与 `TtsPlayer.enqueueChunk` 一致：
 *   `nextStartTime` = 已经排到的**绝对时刻**（0 = 队列空）；
 *   一块音频的起点取 `now` 与 `已排位置 − 跨句交叠` 的较大值。
 *
 * `fixed`  —— 用新的 `PrebufferGate`（每轮 `armTurn` 一次）
 * `legacy` —— 复刻旧判据 `nextStartTime <= now`（队列一空就当开场）
 * `off`    —— 预缓冲关闭（prebuffer=0）
 */
function simulate(turns: Array<Array<[number, number]>>, prebufferMs: number, mode: Mode): SimResult {
  const gate = new PrebufferGate(true);
  const eps = 1e-6;
  let nextStartTime = 0; // 已排到的绝对时刻；0 = 队列空
  let firstArrival = -1;
  let firstSound = -1;
  let activations = 0;
  let extraDelay = 0;
  /** 排程事件 `[起点绝对时刻, 音频时长]` —— 用来算水位曲线（口径同 probe） */
  const plan: Array<[number, number]> = [];

  for (let i = 0; i < turns.length; i += 1) {
    if (mode === 'fixed') gate.armTurn();

    for (const [atMs, chunkMs] of turns[i]) {
      if (firstArrival < 0) firstArrival = atMs;
      const queueEmpty = nextStartTime <= atMs + eps;
      const scheduledEnd = nextStartTime; // want 为真时它会变，先留旧值算"本该接上的位置"
      let want = false;
      if (prebufferMs > 0) {
        if (mode === 'fixed') want = gate.decide(Math.max(0, nextStartTime - atMs));
        // 旧判据：只看"队列是不是空的" —— 阵间隙里队列空了就会被再判一次开场
        else if (mode === 'legacy') want = queueEmpty;
      }

      const base = want ? atMs + prebufferMs : queueEmpty ? atMs : nextStartTime;
      const startAt = Math.max(atMs, base);
      if (want) {
        activations += 1;
        // 第 1 次是真开场（那时本来也没音频可播）；其后每次都是白等
        if (activations > 1) extraDelay += prebufferMs;
      }
      if (firstSound < 0) firstSound = startAt;
      nextStartTime = startAt + chunkMs;
      plan.push([startAt, chunkMs]);
      void scheduledEnd;
    }
  }

  // ── 播放水位曲线（与 `probe_tts_supply.mjs` 同一模型）───────────────────
  // 水位(t) = 到 t 为止**已经排上时间轴**的音频量 − t 与首次出声之差；
  // 每个排程事件的起点处取一次（那一刻新音频造成的抬升已经发生）。
  const playStart = plan.length > 0 ? plan[0][0] : 0;
  let cum = 0;
  let minLevel = Infinity;
  let starveTotal = 0;
  let prevT = playStart;
  for (const [startAt, chunkMs] of plan) {
    // 从上一个事件到这一个事件之间，播放头在消耗
    const levelBefore = cum - (startAt - playStart);
    if (levelBefore < 0) starveTotal += Math.min(-levelBefore, startAt - prevT);
    cum += chunkMs;
    const levelAfter = cum - (startAt - playStart);
    if (levelAfter < minLevel) minLevel = levelAfter;
    prevT = startAt;
  }
  // 收尾：最后一块播完
  const endLevel = cum - (plan.length > 0 ? plan[plan.length - 1][0] + plan[plan.length - 1][1] - playStart : 0);
  if (endLevel < minLevel) minLevel = endLevel;

  return {
    activations,
    speechTurns: turns.length,
    minLevel: Math.round(minLevel),
    starveTotalMs: Math.round(starveTotal),
    firstSoundMs: Math.round(firstSound),
    extraDelayMs: Math.round(extraDelay),
  };
}

console.log('══════ 场景：3 轮对话，每轮 1 大块 + 5 小块（含多个阵间隙）══════');
const turns3 = [burstTurn(0), burstTurn(12000, 3), burstTurn(22000, 6)];

const fixed = simulate(turns3, 300, 'fixed');
const legacy = simulate(turns3, 300, 'legacy');
const off = simulate(turns3, 0, 'off');

console.log('');
console.log('指标                 | 旧判据(300ms) | 新状态机(300ms) | 关闭(0ms)');
console.log('---------------------|---------------|-----------------|----------');
const row = (label: string, pick: (r: SimResult) => string, w = 13) =>
  console.log(
    `${label.padEnd(20)} | ${pick(legacy).padStart(w)} | ${pick(fixed).padStart(15)} | ${pick(off).padStart(8)}`,
  );
row('预缓冲生效次数', (r) => String(r.activations));
row('实际产生语音轮数', (r) => String(r.speechTurns));
row('白等总时长(ms)', (r) => String(r.extraDelayMs));
row('最低水位(ms)', (r) => String(r.minLevel));
row('见底累计(ms)', (r) => String(r.starveTotalMs));
row('首声延迟(ms)', (r) => String(r.firstSoundMs));

console.log('');
console.log('验收 1/2：生效次数受控（用户核对项）');
check(
  fixed.activations === fixed.speechTurns,
  '新状态机：生效次数 == 产生语音的轮数（每轮恰好一次）',
  `${fixed.activations} / ${fixed.speechTurns}`,
);
check(
  legacy.activations > legacy.speechTurns,
  '旧判据：生效次数**超过**轮数（缺陷复现）',
  `${legacy.activations} > ${legacy.speechTurns}`,
);

console.log('');
console.log('验收 3：阵间隙恢复供音时不再额外等待');
{
  // 单轮：1 大块 + 5 小块 ⇒ 队列会在中段被吃空多次（真机形态）
  const oneTurn = [burstTurn(0)];
  const f = simulate(oneTurn, 300, 'fixed');
  const l = simulate(oneTurn, 300, 'legacy');
  check(f.activations === 1, '新状态机：一轮里的多个间隙仍只生效 1 次', `activations=${f.activations}`);
  check(l.activations > 1, '旧判据：一轮里每个间隙都再等一次（缺陷复现）', `activations=${l.activations}`);
  check(
    f.firstSoundMs === l.firstSoundMs,
    '两者首声相同（修复不影响开场那一等，只去掉后面多余的）',
    `${f.firstSoundMs}ms`,
  );
  check(
    l.extraDelayMs - f.extraDelayMs === (l.activations - 1) * 300,
    '旧判据多出来的白等 = （多触发的次数）× 300ms —— 正是用户担心的那一项',
    `多 ${l.extraDelayMs - f.extraDelayMs}ms（多触发 ${l.activations - 1} 次）`,
  );
}

console.log('');
console.log('验收 4：水位/见底没有反弹（修复不能把卡顿换回来）');
check(
  fixed.minLevel >= legacy.minLevel,
  '新状态机最低水位**不低于**旧判据（不多等 ⇒ 水位只会更好）',
  `新 ${fixed.minLevel}ms vs 旧 ${legacy.minLevel}ms`,
);
check(
  fixed.starveTotalMs <= legacy.starveTotalMs,
  '见底累计没有变差',
  `新 ${fixed.starveTotalMs}ms vs 旧 ${legacy.starveTotalMs}ms`,
);
check(
  fixed.activations === 3 && fixed.firstSoundMs === 300,
  '修复后"每轮一次"：3 轮共 3 次，且第一次仍是开场那一等',
  `n=${fixed.activations} 首声=${fixed.firstSoundMs}ms`,
);

console.log('');
console.log('补充：预缓冲本身有没有用（新状态机 vs 关闭）');
check(
  fixed.minLevel >= off.minLevel,
  '开预缓冲的最低水位不低于关闭（说明它确实在起作用）',
  `开 ${fixed.minLevel}ms vs 关 ${off.minLevel}ms`,
);
check(
  fixed.starveTotalMs <= off.starveTotalMs,
  '开预缓冲的见底累计不超过关闭',
  `开 ${fixed.starveTotalMs}ms vs 关 ${off.starveTotalMs}ms`,
);

console.log('');
if (failed > 0) {
  console.log(`❌ 预缓冲 A/B 对照：${failed} 项失败`);
  process.exit(1);
}
console.log('✅ 预缓冲 A/B 对照：全部通过');
console.log('');
console.log('⚠️ 这个脚本只能证明"触发条件"与"结构上不会更差"；');
console.log('   最终验收仍以真机为准：刷新后保持 prebuffer=300，看面板');
console.log('   `TTS·预缓冲生效 N 次（应 ≤ 轮数）`、`缓冲·见底 句内/跨句`、`缓冲·水位`，并听感确认。');
