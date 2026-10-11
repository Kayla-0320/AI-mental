/**
 * 对话总结（元宝纪要风格）的提示词与解析 —— 纯函数，不碰网络与数据库。
 *
 * ## 为什么单独一个文件
 *
 * 提示词里带着本项目的**安全红线**（AGENTS.md：禁诊断性表述、禁药物建议），
 * 而"把模型返回的 JSON 兜成可渲染的结构"这段解析逻辑又必须能离线单测 ——
 * 模型完全可能返回带 ``` 围栏、缺字段、甚至整段非 JSON 的内容。
 * 两者都与"怎么调千问"无关，所以从 `ai.service.ts` 里抽出来。
 *
 * ## ⚠️ 隐私与语气约束（写在提示词最前面，比格式要求优先）
 *
 * - 总结是给**用户自己**看的，不是给医生看的：只复述用户说过的事与感受，
 *   不做任何判断、不下任何结论性的标签（例如"抑郁""焦虑症"）。
 * - 不建议药物，也不给医疗建议；只在末尾按需提示"可随时找信任的人或专业机构"。
 * - 危机内容不在此处做判断（判断由算法层 `audit_rules.yaml` 统一负责），
 *   这里只做纪要到文本的转换。
 */

/** 对模型可见的一段对话（服务层从 Message 表投影而来）。 */
export interface TranscriptTurn {
  role: 'user' | 'assistant';
  content: string;
  /** 该轮的情绪标注（仅当 `sentiment` 里带 `primaryEmotion` 时才有值） */
  emotion?: string;
}

/** 总结的最终结构（也是存进 `Conversation.summary` 的 JSON 结构）。 */
export interface ConversationSummaryFields {
  /** 一段话总览（元宝纪要的"会议概述"位置） */
  overview: string;
  /** 谈到的关键点 */
  keyPoints: string[];
  /** 情绪变化轨迹（一句话，非诊断性） */
  emotionalTrend: string;
  /** 对用户有帮助的点（用户自己用过的办法、被认可的资源等） */
  helpfulSuggestions: string[];
  /** 可以接着聊的方向 */
  followUps: string[];
}

/** 落库的总结记录（`Conversation.summary` 的值就是这个结构的 JSON）。 */
export interface ConversationSummaryRecord extends ConversationSummaryFields {
  /** 生成时间（ISO 字符串） */
  generatedAt: string;
  /** 生成时覆盖的消息条数 */
  messageCount: number;
  /** 生成模型（便于事后审计"这条结要是哪个模型写的"） */
  model: string;
}

/** 与 `TranscriptTurn[]` 拼出来的纯文本一起使用的系统提示词。 */
export const SUMMARY_SYSTEM_PROMPT = `你是"小安"（一个面向青少年的 AI 心理陪伴助手）的纪要做好帮手，负责把一段用户与小安的对话整理成**给用户自己看的纪要**，风格接近办公软件的"智能纪要"：客观、清楚、有条理。

铁律（优先级高于格式要求）：
1. 只复述对话里**确实出现过**的内容与感受，不得推测、不得补全、不得编造细节。
2. 严禁任何诊断性表述：不得出现"抑郁""焦虑症""确诊""患有""症状""病人"等词，也不得给出心理疾病判断。
3. 严禁药物相关内容：不得出现任何药物名称、剂量或用药建议。
4. 不评判、不说教、不给医疗建议；语气平静、尊重、对被总结的人友好。
5. 如果这段对话里出现自伤/自杀等高风险内容，不要在纪要里复述细节、也不要分析，
   只在 overview 末尾平淡提一句"这段内容更适合和信任的成年人或专业机构当面聊"。
6. 用第二人称或无人称叙述（"你提到…""谈到…"），不要用"患者""来访者"。

只输出一个 JSON 对象，不要 markdown 围栏、不要多余解释。

⚠️ 引号纪律（很容易踩，请务必遵守）：JSON 的字符串里**不得出现未转义的双引号**。
要引用用户的原话时，一律改用中文引号「」或『』，例如写「不知道」而不是 "不知道"。
下面每个字符串值里都只放纯文本，不放 JSON、不放换行符。

{
  "overview": "2-3 句话概述这段对话聊了什么、用户当时的处境（不超过 120 字）",
  "keyPoints": ["谈话里出现的具体事情或困扰，3-5 条，每条一句话"],
  "emotionalTrend": "情绪上的变化过程，一句话（例如：从刚开口时的紧绷，到后面慢慢放松）",
  "helpfulSuggestions": ["对话里对用户可能有帮助的点，2-4 条；没有就给空数组"],
  "followUps": ["下次可以接着聊的方向，1-3 条，用疑问或话题形式"]
}`;

/**
 * 把对话投影拼成提示词里那段"对话原文"。
 *
 * @param turns 按时间正序的对话
 * @returns 形如 `我：…\n小安：…` 的纯文本；情绪标注只在确实有值时才出现
 */
export function buildTranscriptText(turns: TranscriptTurn[]): string {
  return turns
    .map((turn) => {
      const speaker = turn.role === 'user' ? '我' : '小安';
      const emotion = turn.emotion ? `（当时情绪：${turn.emotion}）` : '';
      return `${speaker}${emotion}：${turn.content.trim()}`;
    })
    .join('\n');
}

/**
 * 从模型输出里抠出 JSON。
 *
 * 模型即使被明确要求"只输出 JSON"，实践中仍会带 ```json 围栏、前后寒暄，
 * 甚至**双重编码**（整个 JSON 又被当作字符串塞进某个字段）。因此这里按
 * 围栏 → 花括号配平扫描 → 原文 四级降级尝试，全失败返回 null
 * 由调用方决定是否把原文当 overview 用。
 *
 * 为什么用"配平扫描"而不是 `indexOf('{') … lastIndexOf('}')`：
 * 模型在 JSON 前后写寒暄时，寒暄里完全可能出现花括号（"这里用 {…} 表示"），
 * 那样首尾一夹就切出**非法 JSON**，于是整次结构化解析静默降级成一段纯文本，
 * 用户看到的是"一坨 JSON 文本"当纪要（现场实测出现过）。
 * 配平扫描会跳过字符串内部的花括号，从第一个真正的顶层对象边界取值。
 */
export function extractJsonObject(raw: string): Record<string, unknown> | null {
  const attempts: string[] = [];

  const fenced = raw.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenced?.[1]) attempts.push(fenced[1].trim());

  const balanced = sliceFirstJsonObject(raw);
  if (balanced) attempts.push(balanced);

  attempts.push(raw.trim());

  for (const candidate of attempts) {
    try {
      const parsed = JSON.parse(candidate);
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    } catch {
      /* 换下一种切法 */
    }
  }

  return null;
}

/**
 * 从文本里切出第一个**配平**的顶层 JSON 对象。
 *
 * 逐字符扫描，跟踪"是否在字符串内"与转义状态 —— 字符串里的 `{` `}` `\`
 * 都不参与配平计算。取深度归零那一刻作为结束位置。
 *
 * @param text 任意文本（可能前后有寒暄）
 * @returns 切出的对象子串，找不到返回 null
 */
function sliceFirstJsonObject(text: string): string | null {
  const start = text.indexOf('{');
  if (start < 0) return null;

  let depth = 0;
  let inString = false;
  let escaped = false;

  for (let i = start; i < text.length; i += 1) {
    const ch = text[i];

    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (ch === '\\') {
        escaped = true;
      } else if (ch === '"') {
        inString = false;
      }
      continue;
    }

    if (ch === '"') {
      inString = true;
    } else if (ch === '{') {
      depth += 1;
    } else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return text.slice(start, i + 1);
    }
  }

  return null;
}

/** 把任意值收敛成"去空白后的字符串数组"，并丢掉空项与重复项。 */
function toStringList(value: unknown, max: number): string[] {
  const source = Array.isArray(value) ? value : value === undefined || value === null ? [] : [value];
  const out: string[] = [];
  for (const item of source) {
    const text = typeof item === 'string' ? item.trim() : '';
    if (!text) continue;
    if (out.includes(text)) continue;
    out.push(text);
    if (out.length >= max) break;
  }
  return out;
}

/** 把任意值收敛成"去空白后的字符串"。 */
function toStringValue(value: unknown): string {
  if (typeof value === 'string') return value.trim();
  if (typeof value === 'number' && Number.isFinite(value)) return String(value);
  return '';
}

/**
 * 把模型返回的 JSON 规范化成可渲染、可落库的结构。
 *
 * 关键取舍：**宁可字段为空，也不填占位文案**。前端见到空数组就整段不渲染，
 * 而"暂无"这种占位会让人以为总结真的分析过、只是没结论。
 *
 * @param raw 模型原始输出
 * @returns 规范化后的字段；`overview` 为空时用原文兜底（避免整次调用白花）
 */
export function normalizeSummaryFields(raw: string): ConversationSummaryFields {
  return normalizeObject(extractJsonObject(raw), raw);
}

/**
 * 把"任意来源"的对象规范化成五段纪要。
 *
 * @param parsed 严格解析成功的对象；为 null 表示需要走容错抽取
 * @param raw 模型原始输出（容错抽取与最终兜底都用它）
 */
function normalizeObject(
  parsed: Record<string, unknown> | null,
  raw: string,
): ConversationSummaryFields {
  // 严格解析能过就用严格结果（最可信）；过不了走容错抽取。
  // 现场实测：模型会在某个要点里写未转义的双引号（`"…出现"不知道"…"`），
  // 这种输出**必然**让 JSON.parse 失败 —— 但内容其实是好的。
  // 只靠 JSON.parse 会让这种"内容好、语法坏"的回复整段降级成纯文本，很亏。
  const obj = parsed
    ? unwrapJsonStringFields(parsed)
    : extractFieldsTolerantly(raw);

  const overview = cleanText(toStringValue(obj.overview));
  const keyPoints = toStringList(obj.keyPoints, 8).map(cleanText);
  const emotionalTrend = cleanText(toStringValue(obj.emotionalTrend));
  const helpfulSuggestions = toStringList(obj.helpfulSuggestions, 6).map(cleanText);
  const followUps = toStringList(obj.followUps, 4).map(cleanText);

  // 连容错抽取都拿不到东西时，把原文当作概述。
  const nothingParsed =
    !overview &&
    keyPoints.length === 0 &&
    !emotionalTrend &&
    helpfulSuggestions.length === 0 &&
    followUps.length === 0;

  if (nothingParsed) {
    // 兜底必须把围栏语言标记一并去掉，否则用户会在纪要里看到 ```json 字样
    const fallback = raw
      .replace(/```[a-z]*/gi, '')
      .replace(/```/g, '')
      .trim();
    return {
      overview: fallback.slice(0, 500),
      keyPoints: [],
      emotionalTrend: '',
      helpfulSuggestions: [],
      followUps: [],
    };
  }

  return { overview, keyPoints, emotionalTrend, helpfulSuggestions, followUps };
}

/**
 * 修复"双重编码"的模型输出。
 *
 * 现场实测：模型返回的 `overview` 字段值不是概述文本，而是**一整段 JSON 字符串**
 * （`{"overview":"…","keyPoints":[…]}`）。此时外层解析成功、但每个字段都取不到，
 * 结果是结构化的五段纪要全部丢失，用户看到一坨 JSON 文本。
 *
 * 三层尝试，逐层放宽：
 *   1. 字段值本身是**合法** JSON 对象 → 直接换用内层对象（最干净）；
 *   2. 字段值是**语法坏掉**的 JSON 文本（例如内层也有漏转义的引号）
 *      → 对内层文本跑容错抽取；
 *   3. 都不行 → 原样返回，交给调用方按普通文本处理。
 *
 * 判据带 `overview`/`keyPoints` 关键字，避免把"用户真的引用过一段 JSON"
 * 这种正常内容误判成编码问题。
 *
 * @param obj 外层解析结果
 * @returns 还原后的对象（无需还原时原样返回）
 */
function unwrapJsonStringFields(obj: Record<string, unknown>): Record<string, unknown> {
  const candidates: unknown[] = [obj.overview, obj.keyPoints, obj.emotionalTrend];

  for (const candidate of candidates) {
    if (typeof candidate !== 'string' || !candidate.trim().startsWith('{')) continue;

    const inner = extractJsonObject(candidate);
    if (inner && isSummaryShaped(inner)) return inner;

    const salvaged = extractFieldsTolerantly(candidate);
    if (isSummaryShaped(salvaged)) return salvaged;
  }

  return obj;
}

/**
 * 清洗单个字段文本。
 *
 * 所有动作只针对**结构性残留**，正文本身不会被改写：
 *
 * 1. 拼接残留。内层 JSON 坏在一个要点里漏转义的引号时，按配对关系切出来的
 *    `overview` 会把后面的 `",\n  "keyPoints": [` 一路带上（甚至因漏转义而残缺成
 *    `: [`）。正文里正常不会出现 `":` 或 `",\s*[` 这种写法，看到就截断。
 * 2. 收尾残留。字段名正好出现在自身值的末尾（`emotionalTrend` 值后面跟着
 *    `: [],\n  "followUps": [`）时同样截断。
 * 3. 解开残余转义：`\"` → `"`、`\n` → 换行。放在最后做，转义符不可能再把正文切坏。
 *
 * @param text 原始字段文本
 * @returns 可直接渲染的干净文本
 */
function cleanText(text: string): string {
  // 结构性尾巴的三种形态，取最先出现的一处截断：
  //   `": [` / `": "`  —— 漏转义导致字段名被吃掉后剩下的分隔符
  //   `",` + 空白 + `[` 或 `{` —— 完整保留的"下一个字段值"开头
  //   `",` + 空白 + `"`  —— 完整保留的"下一个字段名"开头
  const structural = [':\\s*[\\["]', '",\\s*[\\[{"]'];
  let cut = text.length;
  for (const pattern of structural) {
    const match = text.match(new RegExp(pattern));
    if (match?.index !== undefined && match.index < cut) cut = match.index;
  }

  return text
    .slice(0, cut)
    .replace(/[\s",:]+$/, '')
    .replace(/\\"/g, '"')
    .replace(/\\n/g, '\n')
    .trim();
}

/** 这个对象看起来是不是"五段纪要"结构（而不是用户随手引用的一段 JSON） */
function isSummaryShaped(obj: Record<string, unknown>): boolean {  const hasOverview = typeof obj.overview === 'string' && obj.overview.trim().length > 0;
  const hasList =
    (Array.isArray(obj.keyPoints) && obj.keyPoints.length > 0) ||
    (Array.isArray(obj.followUps) && obj.followUps.length > 0) ||
    (typeof obj.emotionalTrend === 'string' && obj.emotionalTrend.trim().length > 0);
  return hasOverview || hasList;
}

/**
 * 容错抽取：不要求整段是合法 JSON，按字段名逐个把值抠出来。
 *
 * 为什么必须要有这一层：模型（qwen-turbo 实测）会把用户原话里的双引号直接
 * 写进字符串而不转义，于是整份输出语法非法。此时**内容其实是完整可用的**，
 * 只因为一个字符就让五段纪要退化成"一坨 JSON 文本"在界面上显示，代价太大。
 *
 * 做法是极小的手写扫描器，而不是正则一把梭：字符串值里可能出现中文引号与
 * 英文引号混排、也可能被下一行的下一个字段名截断，需要"遇到下一个 `"字段名":`
 * 就收尾"的判定才能切干净。
 *
 * @param text 模型原始输出（可能带围栏、寒暄、语法错误）
 * @returns 抽到的字段（拿不到就是空对象，由调用方决定兜底）
 */
function extractFieldsTolerantly(text: string): Record<string, unknown> {
  // 去围栏：`[a-z]*` 而不是 `.*`，否则 `{"overview": true}` 里的 `` `t` 会被
  // 当成围栏语言标记吃掉一个字符（现场踩过）。
  const cleaned = text.replace(/```[a-z]*/gi, '').replace(/```/g, '');
  const result: Record<string, unknown> = {};

  // 数组先抽：数组元素里出现未转义引号时会把"下一个字段名"包进去，
  // 先占好位置，字符串字段的边界判定才不会被污染。
  for (const key of ARRAY_FIELDS) {
    const values = extractArrayField(cleaned, key);
    if (values) result[key] = values;
  }

  for (const [key, value] of Object.entries(findStringFieldSpans(cleaned))) {
    result[key] = value;
  }

  return result;
}

/** 数组型字段名（顺序无关，仅用于遍历） */
const ARRAY_FIELDS = ['keyPoints', 'helpfulSuggestions', 'followUps'] as const;

/** 字符串型字段名 */
const STRING_FIELDS = ['overview', 'emotionalTrend'] as const;

/**
 * 找出字符串型字段的取值。
 *
 * ⚠️ 只认**数组之外**的那一处出现。原因：双重编码时 `overview` 的值本身是一段
 * JSON 文本，里面**也**含 `"overview"`；数组元素里同样可能混进 `"emotionalTrend"`
 * 这样的字样。就近取第一个会把子字段的匹配误当成外层字段，切出来的值就错了。
 *
 * 每个值的结束位置取"下一个字段值的起点"（或文本末尾），
 * 这样 `overview` 的值即使被后面的字段名污染，也能在字段名之前收住。
 */
function findStringFieldSpans(text: string): Record<string, string> {
  const arraySpans = ARRAY_FIELDS
    .map((key) => findArraySpan(text, key))
    .filter((span): span is { start: number; end: number } => span !== null);

  const hits: Array<{ key: string; valueStart: number }> = [];

  for (const key of STRING_FIELDS) {
    const marker = new RegExp(`"${key}"\\s*:\\s*"`, 'g');
    let match: RegExpExecArray | null;
    while ((match = marker.exec(text)) !== null) {
      const at = match.index;
      const insideArray = arraySpans.some((span) => at > span.start && at < span.end);
      if (insideArray) continue; // 数组内的同名字样不是外层字段
      hits.push({ key, valueStart: at + match[0].length });
      break; // 同名只取第一次有效出现
    }
  }

  hits.sort((a, b) => a.valueStart - b.valueStart);

  const out: Record<string, string> = {};
  for (let i = 0; i < hits.length; i += 1) {
    const { key, valueStart } = hits[i];
    const rawEnd = i + 1 < hits.length ? hits[i + 1].valueStart : text.length;
    out[key] = cleanExtractedString(text.slice(valueStart, rawEnd), i + 1 < hits.length);
  }
  return out;
}

/**
 * 定位某个数组字段的 `[ … ]` 跨度（用于把"数组内部"排除在字符串字段匹配之外）。
 *
 * @returns 起止下标（`[` 与 `]` 的位置）；字段不存在返回 null
 */
function findArraySpan(text: string, key: string): { start: number; end: number } | null {
  const marker = new RegExp(`"${key}"\\s*:\\s*\\[`);
  const match = marker.exec(text);
  if (!match) return null;

  const start = match.index + match[0].length - 1; // 指向 `[`
  let depth = 0;
  let inString = false;
  let escaped = false;

  for (let i = start; i < text.length; i += 1) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === '\\') escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === '[') depth += 1;
    else if (ch === ']') {
      depth -= 1;
      if (depth === 0) return { start, end: i };
    }
  }

  return { start, end: text.length };
}

/**
 * 从切出的片段里得到"干净的一行/一段文本"。
 *
 * 切片的右边界是下一个字段的开头（形如 `",\n  "keyPoints": [`）或文本末尾。
 *
 * ⚠️ 不还原 `\"` / `\n` 这类转义：这一步处理的本来就是**语法已坏**的输出，
 * 此时 `"` 的配对关系不可信，去猜哪个 `\"` 是转义、哪个是漏转义只会把正文切坏。
 * 多留几个反斜杠字符，比把用户的话改掉要好得多 —— 只在切到文本末尾
 * （说明这个值后面没有别的字段，尾部一定带着结构性的收尾符号）时才清理多余引号。
 *
 * @param segment 原始切片
 * @param truncated 右侧是否被"下一个字段"截断（截断时不追加任何猜测性清理）
 */
function cleanExtractedString(segment: string, truncated: boolean): string {
  let value = segment.replace(/^[\s"']+/, '');

  if (!truncated) {
    // 未被截断：末尾的 `"` `,` `}` `]` 与空白都属于 JSON 结构，去掉
    value = value.replace(/[\s"',\]}]+$/, '');
  }

  return value.replace(/\\n/g, '\n').trim();
}

/**
 * 抽取数组型字段（如 keyPoints）。找不到该字段返回 null（与"找到但是空数组"区分开）。
 *
 * 元素边界按"`",` 分隔符 + 收尾的 `"]`"来判定：模型漏转义引号时，
 * 元素内部会出现多余的 `"`，所以**不能**简单地按引号成对切分，
 * 而是以结构性分隔符为准。
 */
function extractArrayField(text: string, key: string): string[] | null {
  const marker = new RegExp(`"${key}"\\s*:\\s*\\[`);
  const match = marker.exec(text);
  if (!match) return null;

  const start = match.index + match[0].length;
  // 找数组结尾的 `]`：按方括号配平，并跳过字符串内部（字符串判定对
  // "元素里的裸引号"不敏感，但配平 `[`/`]` 只需知道是否在字符串里即可，
  // 裸引号最多让配平状态提前切换，而本结构不会出现嵌套数组元素）
  let depth = 1;
  let inString = false;
  let escaped = false;
  let end = text.length;
  for (let i = start; i < text.length; i += 1) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === '\\') escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === '[') depth += 1;
    else if (ch === ']') {
      depth -= 1;
      if (depth === 0) {
        end = i;
        break;
      }
    }
  }

  const body = text.slice(start, end).trim();
  if (!body) return [];

  return body
    .split(/",\s*"/)
    .map((piece) =>
      piece
        .replace(/^[\s"'\[]+/, '')
        .replace(/[\s",\]]+$/, '')
        .replace(/\\"/g, '"')
        .replace(/\\n/g, '\n')
        .trim(),
    )
    .filter(Boolean);
}

/**
 * 修复"双重编码"的模型输出。
 *
 * 现场实测：模型返回的 `overview` 字段值不是概述文本，而是**一整段 JSON 字符串**
 * （`{"overview":"…","keyPoints":[…]}`）。此时外层解析成功、但每个字段都取不到，
 * 结果是结构化的五段纪要全部丢失，用户看到一坨 JSON 文本。
 *
 * 这里做一次针对性还原：字段值本身能解析成对象、且带本结构的关键字段时，
 * 就用内层对象替代外层。判据带 `overview`/`keyPoints` 关键字，
 * 避免把"用户真的引用过一段 JSON"这种正常内容误当编码问题。
 *
 * @param obj 外层解析结果
 * @returns 还原后的对象（无需还原时原样返回）
 */
function unwrapDoubleEncoded(obj: Record<string, unknown>): Record<string, unknown> {
  const candidates = [obj.overview, obj.keyPoints, obj.emotionalTrend];
  for (const candidate of candidates) {
    if (typeof candidate !== 'string' || !candidate.trim().startsWith('{')) continue;
    const inner = extractJsonObject(candidate);
    if (!inner) continue;
    if (!('overview' in inner) && !('keyPoints' in inner)) continue;
    return inner;
  }
  return obj;
}
