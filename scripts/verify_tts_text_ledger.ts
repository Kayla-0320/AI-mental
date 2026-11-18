/**
 * 「文本 ↔ 语音」对账链的**离线回归**（不需要浏览器、不需要真机）。
 *
 * 为什么需要它：用户报告"AI 的回复和语音内容完全不匹配，而且语音还输出不全"。
 * 这两句话分别落在**两个独立层面**上，靠听感分不开：
 *
 *   · 文本层面：计划 / 分段 / 过滤 / 双重合成 → 用 {@link splitForSpeech}、
 *     {@link planJoinMatchesText}、{@link unitsToCancelForPlan} 的纯函数回归守住；
 *   · 播放层面：空洞（欠载）→ 由 `TtsPlayer.underrun` 在真机的 `?debug=1` 上量。
 *
 * 这个脚本守的是第一类。四个测试组与用户给的回归顺序一一对应：
 *   1. 固定短句       —— 单次 say、分段、号码过滤是否正确
 *   2. 长回复         —— 分段的完整性（不丢字、不重复）
 *   3. 人为打断       —— 定稿删除尾段时，多余音频能否被精确撤回
 *   4. 多轮连续对话   —— A/B/C/D 四份文本的对账与"预期例外"分类
 *
 * 用法（仓库根目录）：
 *   node "server/node_modules/tsx/dist/cli.mjs" scripts/verify_tts_text_ledger.ts
 * 退出码 1 表示任一断言失败（可直接接进 CI）。
 */
import {
  MAX_TTS_UNIT_CHARS,
  splitForSpeech,
  splitSpeakable,
  pack,
  planJoinMatchesText,
  unitsToCancelForPlan,
  normalizeForCompare,
  planReviseSpeech,
  sliceByNormalizedIndex,
} from '../client/src/services/ttsSplit';
import { buildVoiceTextLedger } from '../client/src/services/voiceLedger';

let failed = 0;
function check(ok: boolean, what: string, detail = ''): void {
  if (ok) {
    console.log(`  ✅ ${what}${detail ? `  ${detail}` : ''}`);
  } else {
    failed += 1;
    console.log(`  ❌ ${what}${detail ? `  ${detail}` : ''}`);
  }
}

/** 切分是否丢字：**有序片段**按序拼接必须等于原文（号码在中间也不能错位） */
function lossless(unit: string): boolean {
  const { fragments } = splitForSpeech(unit);
  return normalizeForCompare(fragments.map((f) => f.text).join(''))
    === normalizeForCompare(unit);
}

console.log('组 1：固定短句 —— 分段 / 号码过滤 / 完整性');
{
  const plain = '听起来今天在学校挺不好受的。';
  const r1 = splitForSpeech(plain);
  check(r1.spoken.length === 1 && r1.spoken[0] === plain, '短句原样送去合成', JSON.stringify(r1.spoken));
  check(r1.skipped.length === 0, '短句没有预期例外');

  // 旧实现：整段命中号码正则 → 整句都不出声（"语音输出不全"的真凶之一）
  const withPhone = '有需要可以打 400 161 9995 找值班老师，我一直都在。';
  const r2 = splitForSpeech(withPhone);
  const phoneSkipped = r2.skipped.filter((s) => s.reason === 'phone');
  check(phoneSkipped.length === 1, '号码被单独摘出来', JSON.stringify(phoneSkipped));
  check(
    r2.spoken.join('').includes('找值班老师') && r2.spoken.join('').includes('我一直都在'),
    '号码前后文照常送合成（不再整句消音）',
    JSON.stringify(r2.spoken),
  );
  check(lossless(withPhone), '号码句切分不丢字（片段按序拼接 == 原文）');
  check(r2.fragments.length === 3 && r2.fragments[1].reason === 'phone',
    '片段顺序保留（号码仍在中间位置）', JSON.stringify(r2.fragments.map((f) => f.reason ?? 'spoken')));

  const punctuationOnly = '「」——……';
  const r3 = splitForSpeech(punctuationOnly);
  check(r3.spoken.length === 0 && r3.skipped.every((s) => s.reason === 'unspeakable'),
    '纯标点不送合成（记为预期例外）');
  check(lossless(punctuationOnly), '纯标点切分不丢字');
}

console.log('\n组 2：长回复 —— 分段完整性（不丢字、不重复）');
{
  const long = [
    '听起来这段时间你一直在硬撑，白天上课、晚上还要写作业，',
    '连喘口气的时间都没有。',
    '你刚才说的那个"一想到明天就心慌"，很可能就是身体在提醒你：',
    '你已经累过头了。要不要我们先从今晚能少做的一件事开始？',
  ].join('');
  const units = splitSpeakable(long);
  check(units.every((u) => u.length <= MAX_TTS_UNIT_CHARS), '每段都不超过上限',
    `max=${Math.max(...units.map((u) => u.length))} limit=${MAX_TTS_UNIT_CHARS}`);
  check(normalizeForCompare(units.join('')) === normalizeForCompare(long),
    '分段拼接等于原文（不丢字、不重复）', `${units.length} 段`);
  check(lossless(long), '逐段过号码/标点过滤后仍不丢字');

  // 单段超长、且没有标点的硬切路径
  const noPunct = '啊'.repeat(MAX_TTS_UNIT_CHARS * 2 + 7);
  check(normalizeForCompare(pack(noPunct, MAX_TTS_UNIT_CHARS).join('')) === noPunct,
    '无标点超长句硬切不丢字');
}

console.log('\n组 3：人为打断 —— 定稿删除尾段时能否精确撤回');
{
  // delta 已经送出去三段（含一句多余的追问），定稿只保留前两句
  const said = [
    { seq: 11, text: '听起来今天挺不好受的。' },
    { seq: 12, text: '老师批评你的时候，' },
    { seq: 13, text: '哪句话最扎心？' },
    { seq: 14, text: '你当时是什么感觉？' },
  ];
  const plan = '听起来今天挺不好受的。老师批评你的时候，哪句话最扎心？';
  const from = unitsToCancelForPlan(said, plan);
  check(from === 14, '只撤回超出的第 4 段（前面的保留）', `from=${from}`);

  // 已发文本还短于定稿 ⇒ 没有多余的，不许误撤
  const shorter = [{ seq: 1, text: '听起来今天挺不好受的。' }];
  check(unitsToCancelForPlan(shorter, plan) === null, '音频落后于文本时不撤回');

  // 跨边界的那一段：保留（宁可多念半句，也不吞字）
  const straddle = [
    { seq: 1, text: '听起来今天挺不好受的。' },
    { seq: 2, text: '老师批评你的时候，哪句话最扎心？你当时是什么感觉？' },
  ];
  check(unitsToCancelForPlan(straddle, plan) === null, '跨边界段保留，不吞字');

  // 定稿是改写（不是删尾）⇒ 无法对齐，交给 revise 处理
  const rewrite = [{ seq: 1, text: '我完全理解你的感受。' }];
  check(unitsToCancelForPlan(rewrite, '听起来今天挺不好受的。') === null, '改写不对齐时返回 null');
}

console.log('\n组 4：A/B/C/D 对账 —— 一致时不误报，不一致时指出第一层');
{
  const text = '我听你这么说挺高兴的。之后需要我，随时都在。';
  const units = splitSpeakable(text);
  const saidUnits = units.map((t, i) => ({ seq: i + 1, text: t }));

  const clean = buildVoiceTextLedger({
    turnId: 1,
    source: 'speak-plan',
    shown: text,
    planText: text,
    planSegments: units,
    saidUnits,
    spokenUnits: saidUnits,
    skipped: [],
    notes: [],
    unspokenTail: null,
  });
  check(clean.exceptions.length === 0, '四份文本一致时零误报', JSON.stringify(clean.exceptions));

  // 带空白的文本：服务端 segments 会 strip，必须按空白归一化后判为一致
  const spaced = '嗯， 我在。今天想从哪儿说起？';
  check(planJoinMatchesText(spaced, ['嗯，', '我在。', '今天想从哪儿说起？']),
    'segments 丢空白不算违约（否则会白拒正常计划）');
  check(!planJoinMatchesText(text, ['完全不同的句子。']), '真不一致必须判违约');

  // C ≠ D：发出去 3 段、只出声 1 段 → 必须点名"串单或音频缺失"
  // （用显式三段构造：上面那句只有 1 段，拿来测这个会假通过）
  const threeUnits = [
    { seq: 21, text: '我听你这么说挺高兴的。' },
    { seq: 22, text: '这段时间你确实做了不少努力。' },
    { seq: 23, text: '之后需要我，随时都在。' },
  ];
  const threeText = threeUnits.map((u) => u.text).join('');
  const short = buildVoiceTextLedger({
    turnId: 2,
    source: 'speak-plan',
    shown: threeText,
    planText: threeText,
    planSegments: threeUnits.map((u) => u.text),
    saidUnits: threeUnits,
    spokenUnits: threeUnits.slice(0, 1),
    skipped: [],
    notes: [],
    unspokenTail: null,
  });
  check(short.exceptions.some((e) => e.startsWith('C≠D')), 'C≠D 被点名为第一优先修复点',
    JSON.stringify(short.exceptions));

  // 计划迟到（delta 先发）→ 记一条，但不是"S≠D"那种串单
  const late = buildVoiceTextLedger({
    turnId: 3,
    source: 'delta-fallback',
    shown: text,
    planText: text,
    planSegments: units,
    saidUnits,
    spokenUnits: saidUnits,
    skipped: [],
    notes: ['speak 计划迟到（当前来源=delta-fallback），未再入队'],
    unspokenTail: null,
  });
  check(late.exceptions.some((e) => e.includes('计划迟到')), '计划迟到被如实记录');
  check(!late.exceptions.some((e) => e.startsWith('C≠D')), '迟到不等于串单（不误报）');

  // 号码是预期例外，不能混进"串单"
  const withPhone = buildVoiceTextLedger({
    turnId: 4,
    source: 'speak-plan',
    shown: '可以打 400 161 9995 找老师。',
    planText: '可以打 400 161 9995 找老师。',
    planSegments: ['可以打 400 161 9995 找老师。'],
    saidUnits: [{ seq: 1, text: '可以打 找老师。' }],
    spokenUnits: [{ seq: 1, text: '可以打 找老师。' }],
    skipped: [{ text: '400 161 9995', reason: 'phone' }],
    notes: [],
    unspokenTail: null,
  });
  check(withPhone.exceptions.some((e) => e.startsWith('预期例外')),
    '号码屏蔽被标成预期例外（不是串单）', JSON.stringify(withPhone.exceptions));
}

console.log('组 5：审计改稿（revise）—— 前缀对齐，只补念真正变了的部分');
{
  // 原稿三段，改稿把第三段换了说法（前两段一字未动）
  const said = [
    { seq: 1, text: '考得不如预期，' },
    { seq: 2, text: '心里一下子沉下来了。' },
    { seq: 3, text: '你想说说是哪一科吗？' },
  ];

  // ① 尾句换说法 → 保留前两段，从第三段撤回，只补念新尾句
  const planA = planReviseSpeech(
    said,
    '考得不如预期，心里一下子沉下来了。你愿意说说最让你难受的是哪一部分吗？',
  );
  check(planA.keepSeqs.join(',') === '1,2', '已发出且仍是新稿前缀的段被保留', JSON.stringify(planA));
  check(planA.cancelFrom === 3, '从跨过分歧点的那一段撤回', `cancelFrom=${planA.cancelFrom}`);
  check(
    planA.tail === '你愿意说说最让你难受的是哪一部分吗？',
    '补念文本 = 新稿去掉已保留前缀',
    JSON.stringify(planA.tail),
  );
  check(planA.keptChars === 17, '如实记录保留了多少字', `keptChars=${planA.keptChars}`);

  // ② 真机最常见的回归点：定稿只是把 "  \n" 抹掉，内容一字未变
  //    → 归一化后相等 ⇒ **不撤回、不补念**（早先版本会全撤 + 整段重念，首声被推后一个周期）
  const wsSaid = [
    { seq: 1, text: '嗯…闹别扭了，心里堵着，' },
    { seq: 2, text: '连话都不想说——这感觉挺真实的。' },
    { seq: 3, text: '  \n' },
    { seq: 4, text: '你愿意说说，当时发生了什么吗？' },
  ];
  const planWs = planReviseSpeech(
    wsSaid,
    '嗯…闹别扭了，心里堵着，连话都不想说——这感觉挺真实的。  \n你愿意说说，当时发生了什么吗？',
  );
  check(
    planWs.cancelFrom === null && planWs.tail === '',
    '空白差异 ⇒ 不撤回、不补念（什么都不做）',
    JSON.stringify(planWs),
  );

  // ③ 新稿只是删掉尾巴 → 保留前缀、从被删的那段撤回、无需补念
  const planC = planReviseSpeech(said, '考得不如预期，心里一下子沉下来了。');
  check(planC.cancelFrom === 3, '新稿已覆盖的部分之后的段进撤回名单',
    `cancelFrom=${planC.cancelFrom}`);
  check(planC.tail === '', '新稿已被已发内容覆盖 ⇒ 无需补念', JSON.stringify(planC.tail));

  // ④ 改稿把第二段换了说法 → 第一段保留，从第二段起重念
  const planD = planReviseSpeech(said, '考得不如预期，心里堵得慌。我在这儿听你说。');
  check(planD.keepSeqs.join(',') === '1', '分歧点之前的段保留', JSON.stringify(planD.keepSeqs));
  check(planD.cancelFrom === 2, '从分歧的那一段撤回');
  check(planD.tail === '心里堵得慌。我在这儿听你说。', '补念从分歧点开始', JSON.stringify(planD.tail));

  // ⑤ 带空格的英文：归一化坐标 → 原文坐标不能吞掉空格
  //    "ok let us talk" 的前 5 个非空白字符是 "oklet"（对应原文 "ok let"），
  //    所以剩余部分从 't' 后面那个空格开始 —— 一个字符都不能多切
  check(
    sliceByNormalizedIndex('ok let us talk', 5) === ' us talk',
    '按归一化下标切原文（保留空格）',
    JSON.stringify(sliceByNormalizedIndex('ok let us talk', 5)),
  );
  check(sliceByNormalizedIndex('ok let us talk', 0) === 'ok let us talk', 'n=0 ⇒ 原文原样');
  check(sliceByNormalizedIndex('你好。', 99) === '', 'n 超出 ⇒ 空串（无需补念）');

  // ⑥ 改稿轮的对账：A≠B 必须标成**预期例外**，而不是"对账失败"
  const revised = buildVoiceTextLedger({
    turnId: 5,
    source: 'speak-plan',
    shown: '考得不如预期，心里一下子沉下来了。你愿意说说最让你难受的是哪一部分吗？',
    planText: '考得不如预期，心里一下子沉下来了。你想说说是哪一科吗？',
    planSegments: said.map((u) => u.text),
    saidUnits: [...said, { seq: 4, text: '你愿意说说最让你难受的是哪一部分吗？' }],
    spokenUnits: [...said, { seq: 4, text: '你愿意说说最让你难受的是哪一部分吗？' }],
    skipped: [],
    notes: ['改稿：前 17 字沿用已播出的说法，不重念'],
    unspokenTail: null,
    sawRevise: true,
  });
  check(
    revised.exceptions.some((e) => e.startsWith('预期例外') && e.includes('改稿')),
    '改稿轮的 A≠B 标成预期例外',
    JSON.stringify(revised.exceptions),
  );
  check(
    !revised.exceptions.some((e) => e.startsWith('A≠')),
    '改稿轮不再报"对账失败"式的 A≠B',
    JSON.stringify(revised.exceptions),
  );
  check(
    !revised.exceptions.some((e) => e.startsWith('B≠C')),
    '改稿轮不再报"对账失败"式的 B≠C（补念尾段本来就不在原计划里）',
    JSON.stringify(revised.exceptions),
  );
}

console.log(`\n${failed === 0 ? '全部通过' : `${failed} 项失败`}`);
process.exit(failed === 0 ? 0 : 1);
