/**
 * 通话过程记录（"每轮都像文字对话那样"）的**离线回归** —— 不需要浏览器、不需要真机。
 *
 * ## 为什么需要它
 *
 * 用户报告："现在 AI 电话看不到完整的对话过程，我需要把每轮对话过程都像文字对话那样弄出来。"
 * 这件事的难点全在**编排**上，不在画面上：
 *
 *   · AI 的一轮回复分多次到达（`speak` 计划 / 流式 delta 逐句 / 字幕随音频逐段揭示 /
 *     审计改稿整体替换），这些必须是**同一条**消息的增量，写成新的一条就会出现
 *     "同一句话在记录里出现四遍"；
 *   · 用户插话发生在上一轮还在飞的时候，它必须是**新的一轮**，不能被并进上一轮；
 *   · 打断 / 改稿 / 危机会让"屏幕上有的字"多于"耳朵里听到的字"，
 *     "这一截没念出来"这个事实必须落在**那一条**消息上；
 *   · 已说出的内容**只增不删**（断网重连、打断、改稿都不许清空）。
 *
 * 上面四条都是纯函数判据，所以用 `services/callTranscript.ts` 做离线回归守住它们。
 *
 * 用法（仓库根目录）：
 *   node "server/node_modules/tsx/dist/cli.mjs" scripts/verify_call_transcript.ts
 * 退出码 1 表示任一断言失败（可直接接进 CI）。
 */
import {
  appendAssistantDelta,
  appendUserMessage,
  createTranscript,
  groupTranscript,
  nextTranscriptId,
  type CallTranscriptEntry,
} from '../client/src/services/callTranscript';

let failed = 0;
function check(ok: boolean, what: string, detail = ''): void {
  if (ok) {
    console.log(`  ✅ ${what}${detail ? `  ${detail}` : ''}`);
  } else {
    failed += 1;
    console.log(`  ❌ ${what}${detail ? `  ${detail}` : ''}`);
  }
}

/** 与 hook 内一致的最小写入器：id 计数器在外部持有 */
function writer() {
  let counter = 0;
  let turn = 0;
  let list: CallTranscriptEntry[] = createTranscript();
  return {
    get list() {
      return list;
    },
    get turn() {
      return turn;
    },
    user(text: string, ts: number): void {
      turn += 1;
      counter += 1;
      list = appendUserMessage(list, text, turn, ts, nextTranscriptId(counter));
    },
    /** AI 一侧的到达：`replace` 为整轮替换，`delta` 为增量 */
    ai(
      text: string,
      ts: number,
      mode: 'delta' | 'replace' = 'delta',
      flags: { revised?: boolean; unspoken?: boolean } = {},
    ): void {
      counter += 1;
      list = appendAssistantDelta(list, text, turn, ts, nextTranscriptId(counter), mode, flags);
    },
    /** 模拟 hook 里的 `turnId` 未被赋值时的退化口径 */
    aiWithoutTurn(text: string, ts: number, fallbackTurn: number): void {
      counter += 1;
      list = appendAssistantDelta(list, text, fallbackTurn, ts, nextTranscriptId(counter), 'delta');
    },
  };
}

const T0 = 1_700_000_000_000;

console.log('组 1：一次提问 + 一次回答 = 一条用户消息 + 一条 AI 消息');
{
  const w = writer();
  w.user('我这次数学没考好。', T0);
  w.ai('听起来', T0 + 800);
  w.ai('你为这次考试', T0 + 1200);
  w.ai('挺难受的。', T0 + 1600);

  check(w.list.length === 2, '流式三次到达仍然是**两条**消息（不是四条）', `条数=${w.list.length}`);
  check(w.list[0].speaker === 'user' && w.list[1].speaker === 'ai', '讲话人顺序对', w.list.map((m) => m.speaker).join(','));
  check(
    w.list[1].text === '听起来你为这次考试挺难受的。',
    'AI 文本增量拼接完整',
    JSON.stringify(w.list[1].text),
  );
  check(w.list[0].turn === 1 && w.list[1].turn === 1, '一次提问与它的回答同属第 1 轮');
  check(w.list[0].tsMs === T0, '用户消息保留定稿时刻');
  check(w.list[1].tsMs === T0 + 800, 'AI 消息保留首次到达时刻（不被后续增量改写）', String(w.list[1].tsMs - T0));
}

console.log('组 2：`speak` 计划 / 字幕逐段揭示是**整轮替换**，不是新增');
{
  const w = writer();
  w.user('我不知道该怎么说。', T0);
  w.ai('没关系，', T0 + 900, 'delta');
  w.ai('没关系，我们慢慢来。', T0 + 1400, 'replace'); // 定稿计划：全文
  w.ai('没关系，我们慢慢来。', T0 + 1800, 'replace'); // 字幕随音频再次揭示全文

  check(w.list.length === 2, '四次到达（1 delta + 2 次全文）仍是两条消息', `条数=${w.list.length}`);
  check(w.list[1].text === '没关系，我们慢慢来。', '最终文本是全文，没有把前文粘两遍', JSON.stringify(w.list[1].text));
}

console.log('组 3：审计改稿 = 同一条 AI 消息被**整体替换**，并留下"已改稿"事实位');
{
  const w = writer();
  w.user('我想死。', T0);
  w.ai('你怎么了', T0 + 600, 'delta');
  w.ai('我很在意你刚才说的。', T0 + 1100, 'replace', { revised: true });

  check(w.list.length === 2, '改稿不会变成两条 AI 消息', `条数=${w.list.length}`);
  check(w.list[1].text === '我很在意你刚才说的。', '记录里是改稿后的文本（屏幕上什么样，记录里就什么样）');
  check(w.list[1].revised === true, '"已改稿"事实位挂在那一条 AI 消息上');
}

console.log('组 4：事实位只增不减（改稿之后又来增量，不能把标记抹掉）');
{
  const w = writer();
  w.user('嗯。', T0);
  w.ai('我听见了。', T0 + 500, 'replace', { revised: true });
  w.ai('我一直在。', T0 + 900, 'delta');
  check(w.list[1].revised === true, '后续增量不会丢掉"已改稿"', JSON.stringify(w.list[1]));
  check(w.list[1].unspoken === undefined, '没有未朗读时不乱标');
  w.ai('（打断后的一次整轮兑现）', T0 + 1200, 'replace', { unspoken: true });
  check(w.list[1].unspoken === true, '打断/危机停播后补上"未朗读"（且仍是同一条）', `条数=${w.list.length}`);
}

console.log('组 5：插话 —— 上一轮还在飞时用户又开口，必须是**新的一轮**');
{
  const w = writer();
  w.user('我很难受。', T0);
  w.ai('我在听，', T0 + 700, 'delta');
  w.user('因为考试。', T0 + 900); // 插话：排队，等本轮 done 后作为下一轮发
  w.ai('原来是考试。', T0 + 1500, 'replace', { revised: false });

  const groups = groupTranscript(w.list);
  check(groups.length === 2, '两轮：插话没有被并进上一轮', `组数=${groups.length}`);
  check(groups[0].messages.length === 2 && groups[1].messages.length === 2, '每轮都是一问一答');
  check(groups[1].messages[0].speaker === 'user', '第二轮以用户发言开头');
  check(
    groups[0].messages.map((m) => m.speaker).join(',') === 'user,ai',
    '第一轮的顺序是 用户 → AI',
    groups[0].messages.map((m) => m.speaker).join(','),
  );
  check(w.turn === 2, '轮次号随用户开口递增', String(w.turn));
}

console.log('组 6：已说出的内容只增不删（长通话里最早的几句仍在）');
{
  const w = writer();
  const said: string[] = [];
  for (let i = 1; i <= 8; i += 1) {
    const u = `用户第 ${i} 句`;
    const a = `小安第 ${i} 句`;
    w.user(u, T0 + i * 10_000);
    w.ai(a, T0 + i * 10_000 + 900, 'replace');
    said.push(u, a);
  }
  check(w.list.length === 16, '8 轮 = 16 条消息，一条不丢', `条数=${w.list.length}`);
  check(
    w.list.map((m) => m.text).join('|') === said.join('|'),
    '整通电话的文本按序完整保留',
  );
  const groups = groupTranscript(w.list);
  check(groups.length === 8, '按轮分组 = 8 组', `组数=${groups.length}`);
  check(
    groups.every((g) => g.messages.length === 2),
    '每组两条（第 N 轮的用户第 N 句 + 小安第 N 句）',
  );
  check(
    w.list[0].text === '用户第 1 句',
    '最早的几句仍在列表里（不会被"只显示最近 N 条"截掉）',
  );
}

console.log('组 7：时间戳单调 —— 页面上的"通话第几秒"才有意义');
{
  const w = writer();
  w.user('第一句', T0);
  w.ai('第一条回复', T0 + 500);
  w.user('第二句', T0 + 5000);
  w.ai('第二条回复', T0 + 5500);
  const times = w.list.map((m) => m.tsMs);
  check(
    times.every((t, i) => i === 0 || t >= times[i - 1]),
    '消息时刻单调不减',
    times.map((t) => t - T0).join(','),
  );
  check(times[3] - times[0] === 5500, '第 2 轮落在通话第 5.5 秒（与顶栏计时同一基准）');
}

console.log('组 8：退化路径 —— 空文本 / 未归属轮次时不给记录添乱');
{
  const w = writer();
  w.user('   ', T0); // 空定稿：不追加（hook 在更早的地方已经 trim 过，这里再守一层）
  check(w.list.length === 0, '空文本不产生消息');
  w.user('真的说了', T0 + 1000);
  check(w.list.length === 1, '只有真说了才产生消息');
  // 轮次号为 0（异常路径：AI 先于任何用户消息到达）时也不能与"第 1 轮"串号
  w.aiWithoutTurn('异常到达', T0 + 1200, 0);
  const groups = groupTranscript(w.list);
  check(groups.length === 2, '轮次 0 单独成组（不与第 1 轮合并）', `组数=${groups.length}`);
}

console.log('');
if (failed > 0) {
  console.log(`❌ 通话过程记录回归：${failed} 项失败`);
  process.exit(1);
}
console.log('✅ 通话过程记录回归：全部通过');
