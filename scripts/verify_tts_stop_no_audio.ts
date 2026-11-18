/**
 * 「掐断之后不该误报"没合成出音频"」的**离线回归**（不需要浏览器、不需要真机）。
 *
 * 为什么需要它 —— 用户截图（2026-10-11 通话页）：
 *   「⚠️ 语音合成出错：这一句没有合成出音频」
 * 这一条弹出时用户正在听 AI 说话，字幕上还标着「安全审计已改稿」。查下去发现
 * 它是**客户端自己制造的假错误**：
 *
 *   1. 一轮回复是**一次把所有句都 `say` 出去**的（`speakUnit` 逐句发），
 *      服务端串行合成、边合成边推 PCM；
 *   2. 收到 `revise`（或用户打断）时 `TtsChannel.stopFrom()` 先 `player.stopAll()`
 *      —— 播放器的 `open` 表被清空；
 *   3. 那些**已经 `start` 过、`end` 还没到**的句子（`seq < minSeq`，不在取消范围内）
 *      稍后走 `end` 分支：`endSegment()` 查不到记录 → 返回 0；
 *   4. 旧代码把 `0` 一律当成"合成成功但没音频" → `onError('这一句没有合成出音频')`。
 *
 * 于是**每次改稿/打断都会弹一条假错误**。修法：`stopFrom`/`stopAll` 在 `stopAll()`
 * 之前把播放器里还挂着的 seq 登记给会话层（`markPlayerStopped`），`end` 分支据此
 * 把"我们主动丢的"计入 `stoppedWithoutAudio` 读数，**只有真的没合成出音频才报错**。
 *
 * 用法（仓库根目录）：
 *   node "server/node_modules/tsx/dist/cli.mjs" scripts/verify_tts_stop_no_audio.ts
 * 退出码 1 表示任一条断言失败（可直接接 CI）。
 */
import { TtsStreamSession } from '../client/src/services/ttsStream';
import type { TtsPlayer } from '../client/src/services/ttsPlayer';

let failed = 0;
function check(ok: boolean, what: string, detail = ''): void {
  if (ok) {
    console.log(`  [OK]   ${what}${detail ? `  ${detail}` : ''}`);
  } else {
    failed += 1;
    console.log(`  [FAIL] ${what}${detail ? `  ${detail}` : ''}`);
  }
}

/**
 * 播放器替身：只实现会话层真正用到的那几个方法。
 *
 * `endSegment` 的语义与真播放器一致（`ttsPlayer.ts`）：
 *   · `open` 里没有这个 seq（被 `stopAll()` 清了，或压根没排上）→ 返回 **0**；
 *   · 有 → 返回累计块数。
 * 这正是"掐断后返回 0"的成因，替身必须如实复现，否则测不出这个 bug。
 */
class FakePlayer {
  open = new Set<number>();
  chunks = new Map<number, number>();
  enqueued: number[] = [];

  /** 模拟"这一句的音频已经排上播放队列" */
  scheduleAudio(seq: number, chunkCount = 3): void {
    this.open.add(seq);
    this.chunks.set(seq, chunkCount);
  }

  openSeqs(): number[] {
    return [...this.open];
  }

  endSegment(seq: number): number {
    if (!this.open.has(seq)) return 0;
    this.open.delete(seq);
    return this.chunks.get(seq) ?? 0;
  }

  enqueueChunk(seq: number): boolean {
    this.enqueued.push(seq);
    return true;
  }

  enqueue(seq: number): boolean {
    this.enqueued.push(seq);
    return true;
  }

  stopAll(): void {
    this.open.clear();
  }
}

/** 造一个会话：回调只记事件，不碰网络/AudioContext */
function makeSession(): {
  session: TtsStreamSession;
  player: FakePlayer;
  errors: { message: string; seq?: number }[];
} {
  const errors: { message: string; seq?: number }[] = [];
  const player = new FakePlayer();
  const cb = {
    onError: (message: string, seq?: number) => errors.push({ message, seq }),
  };
  const session = new TtsStreamSession(
    cb as never,
    player as unknown as TtsPlayer,
  );
  return { session, player, errors };
}

/** 直接喂一条服务端 JSON 帧（`onMessage` 是 private，靠 as any 触达） */
function feed(session: TtsStreamSession, frame: Record<string, unknown>): void {
  (session as unknown as { onMessage: (ev: { data: string }) => void }).onMessage({
    data: JSON.stringify(frame),
  });
}

console.log('组 1：改稿/打断掐断后，被掐断的句子**不得**误报"没合成出音频"');
{
  const { session, player, errors } = makeSession();
  // 一轮两句都发出去了
  session.say(1, '第一句。');
  session.say(2, '第二句。');
  // 两句的音频都已排上播放队列（服务端串行合成，第 1 句先出声）
  player.scheduleAudio(1);
  player.scheduleAudio(2);
  feed(session, { type: 'start', seq: 1, sample_rate: 24000, spoken_text: '第一句。' });
  feed(session, { type: 'start', seq: 2, sample_rate: 24000, spoken_text: '第二句。' });

  // ── 复现 TtsChannel.stopFrom(2)：先取在途句号 → stopAll → 登记 → cancelFrom ──
  const inFlight = player.openSeqs();
  player.stopAll();
  session.markPlayerStopped(inFlight);
  session.cancelFrom(2);

  // 服务端随后照常把两帧都发完（第 2 句是 cancelled，第 1 句是我们主动丢的 end）
  feed(session, { type: 'cancelled', seq: 2 });
  feed(session, {
    type: 'end', seq: 1, spoken_text: '第一句。', sample_rate: 24000,
    audio_ms: 1200, synth_ms: 300, rtf: 0.2,
  });

  check(errors.length === 0, '掐断后没有弹"没合成出音频"', `errors=${JSON.stringify(errors)}`);
  check(
    session.diagnostics.stoppedWithoutAudio === 1,
    '被掐断的句子计入"掐断未播"读数',
    `stoppedWithoutAudio=${session.diagnostics.stoppedWithoutAudio}`,
  );
}

console.log('组 2：真的没合成出音频时**必须**如实报错（别把上面那条修成"一律不报"）');
{
  const { session, player, errors } = makeSession();
  session.say(3, '第三句。');
  // 服务端 start 到了，但一个 PCM 块都没有（空文本 / 全被 OOV 吞掉）
  feed(session, { type: 'start', seq: 3, sample_rate: 24000, spoken_text: '第三句。' });
  feed(session, {
    type: 'end', seq: 3, spoken_text: '第三句。', sample_rate: 24000,
    audio_ms: 0, synth_ms: 10, rtf: 0,
  });

  check(
    errors.length === 1 && errors[0].message === '这一句没有合成出音频',
    '空音频仍然如实上报',
    `errors=${JSON.stringify(errors)}`,
  );
  check(
    session.diagnostics.stoppedWithoutAudio === 0,
    '这一条不算"掐断未播"',
    `stoppedWithoutAudio=${session.diagnostics.stoppedWithoutAudio}`,
  );
  check(player.enqueued.length === 0, '没有假装播过（未入队任何音频）');
}

console.log('组 3：水位线之下的 `end` 也不报错（取消范围内的句子）');
{
  const { session, errors } = makeSession();
  session.say(4, '第四句。');
  feed(session, { type: 'start', seq: 4, sample_rate: 24000, spoken_text: '第四句。' });
  session.cancelFrom(4);
  // 服务端两种收尾都可能到：先 end 再 cancelled（或反过来）
  feed(session, {
    type: 'end', seq: 4, spoken_text: '', sample_rate: 24000,
    audio_ms: 0, synth_ms: 0, rtf: 0,
  });
  check(errors.length === 0, '取消范围内的空 end 不报错', `errors=${JSON.stringify(errors)}`);
}

if (failed > 0) {
  console.log(`\n[FAIL] 共 ${failed} 条断言失败`);
  process.exit(1);
}
console.log('\n[OK] 全部断言通过：掐断不再误报，真空音频仍然可见');
