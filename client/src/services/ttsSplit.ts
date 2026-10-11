/**
 * AI 回复文本 → "送合成的批次"（施工单 §S4.4）
 *
 * 从 `hooks/useVoiceCall.ts` 抽出来的**纯函数**模块：那边是 1200 行的 React hook，
 * 切分逻辑埋在里面既测不了也复用不了。这里没有任何浏览器/React 依赖，
 * 可以直接在 Node 里用真实文本跑回归（见 `_verify_client_split.mjs`）。
 */

/**
 * 单次送合成的最大字数。
 *
 * ⚠️ 旧值 **36 是 VITS / Qwen3-TTS 时代**定的（当时的理由是"80 字 CUDA OOM"），
 * 对 CosyVoice3 **已经不成立**：合成跑在独立进程 8002 上，硬上限是
 * `algorithm/tts/engine.py:68` 的 `MAX_TEXT_CHARS = 200`
 * （`cosyvoice_engine.py:54`、`cosyvoice_service/server.py:195` 三处对齐）。
 *
 * 为什么必须调大（2026-10-05）：
 * **每多切一段 = 多一次独立的合成请求**，代价有三条，都不小：
 *
 *   1. **同一句回复会被采样两次**。CosyVoice 的 LLM 采样没有定种子，实测同一
 *      文本重复生成的频谱平坦度摆 **70%**、高频占比摆 **81%**、峰值（音量）摆
 *      **6 dB** —— 切成两段，等于让 AI 在一句话中间换一次音色/音量。
 *   2. **多塞一个模型自己的逗号停顿**。实测 37 字被切成 22 + 15 两段后，
 *      接缝处出现约 300 ms 的低电平停顿，哪怕那句话本来不需要在这儿停。
 *   3. 极端情况下 {@link pack} 会对**没有标点的超长小句**按字数硬切，切口落在词
 *      中间，两端都带非零波形 —— 那才会真的响一声（播放端已加 12 ms 交叉淡化兜底）。
 *
 * 取 100 而不是直接顶到 200：留一半余量给标点/术语替换后的膨胀，
 * 同时让超长回复仍能分成 2 段流水线推进（第 1 段先出声，第 2 段边合成边等）。
 */
export const MAX_TTS_UNIT_CHARS = 100;

/**
 * 按逗号贪心打包到 `limit` 字以内。
 *
 * @param text 待切分文本
 * @param limit 单块最大字数
 * @returns 切分结果；`text` 不超过 `limit` 时原样返回单元素数组
 */
export function pack(text: string, limit: number): string[] {
  if (text.length <= limit) return [text];
  const parts = text.split(/(?<=[，,、：:])/).filter((s) => s.trim());
  const out: string[] = [];
  let buf = '';
  for (const p of parts) {
    if (p.length > limit) {
      // 单个小句本身就超长（无逗号的长句）——只能按字数硬切
      if (buf) { out.push(buf); buf = ''; }
      out.push(...(p.match(new RegExp(`.{1,${limit}}`, 'gs')) ?? [p]));
      continue;
    }
    if (buf && buf.length + p.length > limit) { out.push(buf); buf = p; }
    else buf += p;
  }
  if (buf) out.push(buf);
  return out;
}

/**
 * 把一段 AI 文本切成"送合成的批次"。
 *
 * 安全闸门已经按句切过，所以正常情况**不用切**；这里只处理"一段太长"这一件事。
 * `isFirst` 参数已废弃（保留只为兼容调用点签名）：
 * 早先为降延迟对首段做 4 字硬切，实测把完整语义单位劈碎，用户听到
 * "句子念不完整"，且段数翻倍后每段各等一次合成，间隙反而更多。
 * **韵律完整性 > 首字延迟**，所以首段不再特殊处理。
 *
 * @param unit 一段 AI 文本
 * @param isFirst 已废弃，忽略
 * @returns 送合成的文本批次
 */
export function splitSpeakable(unit: string, isFirst = false): string[] {
  void isFirst;
  return pack(unit, MAX_TTS_UNIT_CHARS);
}

// ─────────────────────────────────────────────────────────────────────────────
// 「文本 ↔ 语音」对账链（2026-10-11）
//
// 用户报告："AI 的回复和语音内容完全不匹配，而且语音还输出不全"。
// 这两句话指向两条独立的链路，靠猜分不开 —— 所以这里先把**可对账的四份文本**
// 定义成数据，再让每一层把实际发生的事实填进去：
//
//   A 页面最终显示的文本        `useVoiceCall` 的 aiLines
//   B speak 计划（text/segments）SSE `speak` 事件
//   C 实际发出的 say 文本（按序）  TTS 请求侧
//   D 实际出声的 spoken_text（按序）TTS 回包 `start` + 播放器 `onStart`
//
// 四者逐层比较，**第一处不一致就是第一优先修复点**：
//   A≠B → 字幕/revise 的替换逻辑；B≠C → 分段、过滤、重复入队；
//   C≠D → seq 串单/回包错配；D 对但听着断了 → 播放队列与调度。
//
// 本文件只放**纯函数**（不碰 DOM/网络），所以上面这条链可以在 Node 里直接跑回归
// （见 `scripts/verify_tts_text_ledger.ts`）。
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 「含号码特征」的正则**源**（`PHONE_LIKE` 与全局匹配版共用一份，避免漂移）。
 *
 * 为什么必须落成代码而不是靠自觉：实测（`docs/echo_calibration.md` §4.3）
 * `400 161 9995` 被念成「四百六十一九九九九五」—— TTS 把它当成一个 11 位数读。
 * 危机场景下念错热线号码的后果不可接受，所以号码**只以文字卡片呈现**。
 */
export const PHONE_LIKE_SOURCE = String.raw`\d[\d\s\-—–()（）]{5,}\d|\d{7,}`;

/** 单次判定用（**不带 g**：带 g 的 RegExp 有 `lastIndex` 副作用，`test()` 会时真时假） */
export const PHONE_LIKE = new RegExp(PHONE_LIKE_SOURCE);

/** 全局查找用（把号码片段从句子里**摘出来**，而不是把整句消音） */
export const PHONE_LIKE_GLOBAL = new RegExp(PHONE_LIKE_SOURCE, 'g');

/** 汉字 / 字母 / 数字都算"能发音"；纯标点送合成只会得到空白音频 */
export const SPEAKABLE_RE = /[\p{Script=Han}\p{L}\p{N}]/u;

/** 被**有意**跳过、不送合成的片段（对账时算"预期例外"，不是故障） */
export interface SkippedSpeechFragment {
  text: string;
  /** `phone` = 号码（只上屏不朗读）｜`unspeakable` = 纯标点/符号｜`blank` = 纯空白 */
  reason: 'phone' | 'unspeakable' | 'blank';
}

/** 一段文本切分后的**有序**片段：送合成 or 有意跳过（两者拼起来恰好等于原文） */
export interface SpeechFragment {
  text: string;
  /** 是否送合成 */
  spoken: boolean;
  /** 不送合成的原因；`spoken: true` 时没有这个字段 */
  reason?: SkippedSpeechFragment['reason'];
}

/** 一段文本切分后的结果 */
export interface SpeechSplit {
  /** 有序片段（对账/回归用这个：按序拼接**必须**等于原文） */
  fragments: SpeechFragment[];
  /** 送合成的片段（顺序即出声顺序） */
  spoken: string[];
  /** 有意跳过的片段 */
  skipped: SkippedSpeechFragment[];
}

/**
 * 把一段 AI 文本切成"送合成的批次"，并把**号码片段单独摘出来**。
 *
 * ## 与旧行为的区别（这就是"语音输出不全"的一个真凶）
 *
 * 旧实现是在**整段（最多 100 字，按逗号打包）**上测 `PHONE_LIKE`，命中就 `return`
 * —— 一句里只要出现号码，**整句都不出声**，而字幕是完整的。回复里带号码虽然少见，
 * 但一旦出现就是"字幕有一整句、语音一个字节都没有"。
 *
 * 现在只屏蔽号码本身，前后文照常合成；被跳过的片段连**原因**一起回报，
 * 让对账链能把它标成"预期例外"，而不是当成串单。
 *
 * @param unit 一段 AI 文本
 * @param limit 单次送合成的最大字数（默认 {@link MAX_TTS_UNIT_CHARS}）
 * @returns 送合成片段与跳过片段；两者按序拼接**等于** `unit`
 */
export function splitForSpeech(unit: string, limit: number = MAX_TTS_UNIT_CHARS): SpeechSplit {
  const fragments: SpeechFragment[] = [];
  const spoken: string[] = [];
  const skipped: SkippedSpeechFragment[] = [];
  if (!unit) return { fragments, spoken, skipped };

  const push = (text: string, reason: SkippedSpeechFragment['reason'] | null): void => {
    if (!text) return;
    if (reason === null) {
      fragments.push({ text, spoken: true });
      spoken.push(text);
      return;
    }
    fragments.push({ text, spoken: false, reason });
    skipped.push({ text, reason });
  };

  // ① 先按**位置**切出"号码区"与"可念区"（顺序必须保留：号码在句中时，
  //    前面的文字必须先于号码被记录，否则片段拼不回原文）
  PHONE_LIKE_GLOBAL.lastIndex = 0;
  const regions: { text: string; phone: boolean }[] = [];
  let cursor = 0;
  for (const m of unit.matchAll(PHONE_LIKE_GLOBAL)) {
    const at = m.index ?? 0;
    if (at > cursor) regions.push({ text: unit.slice(cursor, at), phone: false });
    regions.push({ text: m[0], phone: true });
    cursor = at + m[0].length;
  }
  if (cursor < unit.length) regions.push({ text: unit.slice(cursor), phone: false });

  // ② 按序处理每个区：号码整段跳过；可念区按字数打包，纯空白/纯标点也记为预期例外
  for (const region of regions) {
    if (region.phone) {
      push(region.text, 'phone');
      continue;
    }
    for (const piece of pack(region.text, limit)) {
      if (!piece) continue;
      if (!piece.trim()) {
        push(piece, 'blank');
        continue;
      }
      if (!SPEAKABLE_RE.test(piece)) {
        push(piece, 'unspeakable');
        continue;
      }
      push(piece, null);
    }
  }
  return { fragments, spoken, skipped };
}

/** 比对用归一化：去掉空白（服务端的 `segments` 会 strip，客户端切分也会丢空白） */
export function normalizeForCompare(text: string): string {
  return (text || '').replace(/\s+/g, '');
}

/**
 * 计划自洽性：`segments` 按序拼接是否等于 `text`（**按空白归一化后**比较）。
 *
 * ⚠️ 必须归一化：服务端 `_split_speak_units` 会 `strip()` 每个片段、丢弃纯空白片段，
 * 所以带空格/换行的回复（英文、号码分组）`join('')` 本来就不等于原文 ——
 * 那不是协议违约，直接判违约会把**正常计划**拒掉、退回 delta 路径。
 */
export function planJoinMatchesText(text: string, segments: readonly string[]): boolean {
  return normalizeForCompare(segments.join('')) === normalizeForCompare(text);
}

/** 公共前缀长度（按归一化后的字符比较，返回归一化坐标系里的长度） */
export function commonPrefixLength(a: string, b: string): number {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a[i] === b[i]) i += 1;
  return i;
}

/**
 * 定稿文本到达时，**已按 delta 送出去的那几段里**哪一段起要撤回（返回该段 seq）。
 *
 * 场景：服务端会删掉尾部整句（"一轮只问一个"、长度上限）。delta 已经先发出去合成了，
 * 定稿文本比它短 —— 那就把**越界的那几段**撤掉（`stopFrom`），让"听到的"回到定稿文本。
 *
 * 取舍：
 *   · 跨越边界的那一段**保留**（宁可多念半句，也不要吞字 —— 与"内容不藏"一致）；
 *   · 无法对齐（不是前缀关系，即定稿是**改写**而非删尾）返回 `null`，交给 revise 处理。
 *
 * @param units 已发出的合成单元（**按发出顺序**）
 * @param planText 定稿文本
 * @returns 要撤回的第一段 seq；无需撤回或无法对齐时为 null
 */
export function unitsToCancelForPlan(
  units: readonly { seq: number; text: string }[],
  planText: string,
): number | null {
  const plan = normalizeForCompare(planText);
  if (!plan || units.length === 0) return null;
  let acc = '';
  for (let i = 0; i < units.length; i += 1) {
    acc += normalizeForCompare(units[i].text);
    if (!plan.startsWith(acc)) {
      // 第 i 段跨过了定稿边界：留着它，从**下一段**开始撤
      return i + 1 < units.length ? units[i + 1].seq : null;
    }
    if (acc.length >= plan.length) {
      // 第 i 段正好覆盖到定稿末尾：后面几段全是多余的
      return i + 1 < units.length ? units[i + 1].seq : null;
    }
  }
  // 已发文本还没说到定稿末尾（音频落后于文本）：没有多余的，什么都不用撤
  return null;
}

/**
 * 按**归一化坐标**切出剩余部分：返回 `text` 里"前 `n` 个非空白字符"之后的原文。
 *
 * 为什么需要"归一化坐标 → 原文坐标"这一步：前缀对齐是在 `normalizeForCompare`
 * （去掉空白）之后算的，而**要送去合成的必须是原文**（空格影响英文断词、也影响
 * 字幕定位）。所以按下标走一遍原文、数够 `n` 个非空白字符即可。
 *
 * @param text 原文
 * @param n 已经覆盖的非空白字符数（≥ 原文的非空白总数时返回空串）
 * @returns 剩余部分的原文
 */
export function sliceByNormalizedIndex(text: string, n: number): string {
  if (n <= 0) return text;
  let seen = 0;
  for (let i = 0; i < text.length; i += 1) {
    if (!/\s/.test(text[i])) {
      seen += 1;
      if (seen === n) return text.slice(i + 1);
    }
  }
  return '';
}

/** 改稿后"该保留哪几段、从哪一段起撤回、还要补念什么" */
export interface ReviseSpeechPlan {
  /** 不必重念的句号（**已完整播完**、且拼接后仍是改稿文本的前缀） */
  keepSeqs: number[];
  /** 从这一段起撤回（含）；没有要撤的段时为 null */
  cancelFrom: number | null;
  /** 需要补念的文本（改稿文本去掉"已保留前缀"之后的原文）；空串 = 无需补念 */
  tail: string;
  /** 保留前缀在顺序拼接里的长度（字符数，供台账如实记录） */
  keptChars: number;
}

/**
 * 审计改稿（`revise`）之后**怎么把新稿念出来**。
 *
 * 用户口径（2026-10-11）："**每一次输出的内容都需要语音合成**" —— 屏幕上有的字，
 * 耳朵里必须有；但又不能整段重念（用户会把开头听第二遍）。所以做**前缀对齐**：
 *
 * 1. 从第一段起，只要这一段接上去仍是新稿的前缀，就保留（它是新稿的一部分，
 *    **不必重念** —— 哪怕它还没播完，它念的内容也是对的）；
 * 2. 遇到第一段"跨过分歧点"的（它的文本不在新稿的这个位置上），**从它开始撤回**；
 * 3. 补念 = 新稿去掉已保留前缀的剩余部分。
 *
 * ⚠️ 这里**不需要**"是否已经播完"这个条件（2026-10-11 修正）：早先版本要求
 * "保留段必须是已完整播完的段"，而真机上 `revise` 紧跟在最后一个 delta 后面到达
 * （实测 1.3 s，此时**一句都还没播完**），于是每一轮都退化成"全撤 + 整段重念"，
 * 首声被推后一整个合成周期 —— 用户反馈的"延迟比以前高"就是它。
 * 按前缀判断后：内容没变的一轮 `cancelFrom = null` 且 `tail = ''`，**什么都不做**。
 *
 * 边界：
 *   · 空白差异（`"  \n"` 这类，定稿只是抹空白）→ 归一化后相等 → 前两条件全满足
 *     → 不撤回、不补念（这是最常见的一轮，也是回归点）；
 *   · 新稿只是删掉尾巴 → 保留前缀，从被删的那段撤回，无需补念；
 *   · 新稿更短且已被覆盖 → `tail` 为空。
 *
 * @param saidUnits 本轮**实际发出**的合成段（按发出顺序，C 端点）
 * @param revisedText 审计改稿后的最终文本
 * @returns 撤回点 + 保留段 + 待补念文本
 */
export function planReviseSpeech(
  saidUnits: readonly { seq: number; text: string }[],
  revisedText: string,
): ReviseSpeechPlan {
  const revised = normalizeForCompare(revisedText);
  let covered = '';
  const keepSeqs: number[] = [];
  let cancelFrom: number | null = null;

  for (const unit of saidUnits) {
    const piece = normalizeForCompare(unit.text);
    // 纯空白段（`"  \n"` 这种）：既没有音频、也不占归一化坐标，跳过即可 ——
    // 不能让它把游标停住，否则后面本来能保留的段会被当成"分歧点"（回归测试组 5 ②）。
    if (!piece) continue;
    if (revised.startsWith(covered + piece)) {
      covered += piece;
      keepSeqs.push(unit.seq);
      continue;
    }
    // 这一段跨过了分歧点：从它开始撤回（它的后半句可能没听到，重念比吞字安全）
    cancelFrom = unit.seq;
    break;
  }

  return {
    keepSeqs,
    cancelFrom,
    tail: sliceByNormalizedIndex(revisedText, covered.length),
    keptChars: covered.length,
  };
}
