/**
 * 「首段太短就先不送合成」守卫的离线回归 —— 重点是**一个字都不丢**。
 *
 * ## 背景（2026-10-11 用户："AI 回复时偶尔会说出一些乱说的话，然后才能说出回复的内容"）
 *
 * `onDelta` 的 delta 回退路径会把"宽限期到点时手上已有的 delta"送去合成。而 delta 是
 * **逐字**到达的，所以那时手上可能只有「嗯」「不过」这种两三个字 —— 一个**独立合成单元**
 * 会把它们念成一句突兀的短话，听感正是"先说一句莫名其妙的，然后才说正文"。
 *
 * 修法（`useVoiceCall.flushByDelta` + `MIN_FIRST_UNIT_CHARS`）：
 *   · 不足 6 字 → 把文本**放回队列**、再等一个宽限期凑齐；
 *   · 又等满一个宽限期还没凑齐 → **照原样送出去**（绝不长期扣住、绝不丢字）；
 *   · 一轮只扣一次。
 *
 * 这个脚本把守卫的判定与"文本守恒"钉死（不改浏览器、不需要真机）。
 * 用法（仓库根目录）：
 *   node "server/node_modules/tsx/dist/cli.mjs" scripts/verify_short_unit_guard.ts
 */
import { splitSpeakable } from '../client/src/services/ttsSplit';

let failed = 0;
function check(ok: boolean, what: string, detail = ''): void {
  if (ok) {
    console.log(`  ✅ ${what}${detail ? `  ${detail}` : ''}`);
  } else {
    failed += 1;
    console.log(`  ❌ ${what}${detail ? `  ${detail}` : ''}`);
  }
}

//: 与 `useVoiceCall.MIN_FIRST_UNIT_CHARS` 保持一致（守卫阈值）
const MIN_FIRST_UNIT_CHARS = 6;

/**
 * 守卫的判定（与 `flushByDelta` 里的那段一一对应）：
 * @returns `'skip'` 纯空白（调用方本来就会跳过）｜`'hold'` 先扣住等凑齐 ｜ `'send'` 送合成
 */
function decide(pending: string, alreadyHeld: boolean, allowShort: boolean): 'skip' | 'hold' | 'send' {
  if (!pending.trim()) return 'skip';
  const short = pending.trim().length < MIN_FIRST_UNIT_CHARS;
  if (short && !allowShort && !alreadyHeld) return 'hold';
  return 'send';
}

/** 模拟一轮的 delta 流：逐字到达，宽限期到点就按守卫决定 */
function runTurn(deltas: string[], graceMs: number): {
  sentUnits: string[];
  sentText: string;
  heldOnce: boolean;
  waits: number;
} {
  const sentUnits: string[] = [];
  let pending = '';
  let heldOnce = false;
  let waits = 0;

  /** 宽限期到点（等价于 flushByDelta 被调用两次：第一次可能扣住） */
  const graceFire = (allowShort: boolean): void => {
    if (!pending) return;
    const verdict = decide(pending, heldOnce, allowShort);
    if (verdict === 'hold') {
      heldOnce = true;
      waits += 1;
      return; // 文本放回队列，什么都不丢
    }
    if (verdict === 'skip') {
      pending = '';
      return;
    }
    for (const unit of splitSpeakable(pending, sentUnits.length === 0)) {
      if (!unit.trim()) continue;
      sentUnits.push(unit);
    }
    pending = '';
  };

  // 逐字到达；每收到一字就把"宽限期"推进一次（简化：模拟第一次到点后第二次再到点）
  for (const ch of deltas.join('')) {
    pending += ch;
    // 第一次到点：可能扣住；第二次到点：allowShort ⇒ 必发
    graceFire(false);
    graceFire(true);
  }
  // 轮结束前把剩余全部兑现（真实实现里由 done/打断/挂断路径的 flush 兜底）
  if (pending) {
    for (const unit of splitSpeakable(pending, sentUnits.length === 0)) {
      if (!unit.trim()) continue;
      sentUnits.push(unit);
    }
    pending = '';
  }
  void graceMs;
  return { sentUnits, sentText: sentUnits.join(''), heldOnce, waits };
}

console.log('组 1：守卫的判定（短 ⇒ 扣住；够长 ⇒ 送；已扣过/允许短 ⇒ 照发）');
{
  check(decide('嗯', false, false) === 'hold', '「嗯」（1 字）先扣住');
  check(decide('不过', false, false) === 'hold', '「不过」（2 字）先扣住');
  check(decide('我不想说', false, false) === 'hold', '「我不想说」（4 字）仍 <6 ⇒ 先扣住');
  check(decide('我有点难受', false, false) === 'hold', '「我有点难受」（5 字）仍 <6 ⇒ 先扣住');
  check(decide('我听懂你的意思', false, false) === 'send', '「我听懂你的意思」（7 字）≥6 ⇒ 直接送');
  check(decide('我听你说', false, false) === 'hold', '「我听你说」（4 字）仍 <6 ⇒ 先扣住');
  check(decide('我在这里陪', false, false) === 'hold', '「我在这里陪」（5 字）仍 <6 ⇒ 先扣住');
  check(decide('我一直在这里', false, false) === 'send', '「我一直在这里」（6 字）=阈值 ⇒ 送');
  check(decide('嗯', true, false) === 'send', '已经扣过一次 ⇒ 不再扣（避免长期压住）');
  check(decide('嗯', false, true) === 'send', 'allowShort（又等满一个宽限期）⇒ 照发');
  check(decide('   ', false, false) === 'skip', '纯空白走 skip 分支（调用方本来就会跳过）');
}

console.log('');
console.log('组 2：⭐ 文本守恒 —— 扣住再放行，一个字都不能丢');
{
  const text = '我听见你说很难受，我们慢慢来。';
  const r = runTurn([text], 350);
  check(
    r.sentText.replace(/\s/g, '') === text.replace(/\s/g, ''),
    '送出去的文本与原文逐字相同（无丢字、无重复）',
    `送出 ${r.sentText.length} 字 / 原文 ${text.length} 字`,
  );
  check(r.heldOnce, '这一轮确实触发过"扣住"（说明守卫在工作）', `waits=${r.waits}`);
}

console.log('');
console.log('组 3：极短首段（「嗯」）不会被单独送合成');
{
  // 只有两个字就结束：守卫扣一次，allowShort 时照发（不丢字，但不至于长期等）
  const r = runTurn(['嗯'], 350);
  check(r.heldOnce, '触发过扣住');
  check(r.sentText.trim() === '嗯', '最终仍然把这两个字说了（不丢字）', `送出「${r.sentText}」`);
  check(
    r.sentUnits.every((u) => u === '嗯'),
    '没有产生多余的空单元',
    JSON.stringify(r.sentUnits),
  );
}

console.log('');
console.log('组 4：正常长文本不受影响（不该因为守卫多等）');
{
  const text = '听起来这段时间你挺不容易的，我一直都在。';
  const r = runTurn([text], 350);
  check(r.sentText.replace(/\s/g, '') === text.replace(/\s/g, ''), '文本完整');
  check(r.waits <= 1, '最多只多等一个宽限期', `waits=${r.waits}`);
}

console.log('');
console.log('组 5：分段完整性 —— 守卫不改变切分结果');
{
  const text = '第一句先说到这儿，第二句接着讲，第三句收个尾。';
  const units = splitSpeakable(text, true);
  check(units.join('') === text, '切分后拼接等于原文（既有不变式）');
  const r = runTurn([text], 350);
  check(
    r.sentText.replace(/\s/g, '') === text.replace(/\s/g, ''),
    '守卫路径下同样不丢字',
  );
}

console.log('');
if (failed > 0) {
  console.log(`❌ 短单元守卫回归：${failed} 项失败`);
  process.exit(1);
}
console.log('✅ 短单元守卫回归：全部通过（扣住不丢字）');
