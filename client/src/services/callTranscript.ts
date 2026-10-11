/**
 * 通话过程记录（**每一轮完整可读**）—— 纯函数，可离线回归。
 *
 * ## 为什么要有这个文件（用户口径，2026-10-11）
 *
 * 用户："现在 AI 电话看不到完整的对话过程，我需要把每轮对话过程都像文字对话那样弄出来。"
 *
 * 电话页原来只有两块**扁平的字符串数组**（`userLines` / `aiLines`），它们回答的问题是
 * "现在屏幕上该显示什么字"，而不是"这一通电话聊了什么"：
 *
 *   · 没有讲话人 —— 只有靠左右对齐去猜哪句是谁说的；
 *   · 没有时刻 —— 用户读下文时不知道这句话是第几秒说的；
 *   · **两两交错**：`userLines` 是"用户定稿句"、`aiLines` 是"AI 回复句"，两个数组各自
 *     追加，谁先谁后是**两个数组的写入顺序**，不是一个时间轴；
 *   · `aiLines` 的最后一条会被**反复整体改写**（流式 delta 用 `[...prev.slice(0,-1), full]`
 *     替换尾部），"这一段是新的还是刚才那段的延续"在数组里看不出来；
 *   · 定稿句与"被改稿 / 没念出来"的事实不在一处，对不上账。
 *
 * 所以这里把它归一成一条**时间轴上的消息流**：一条消息 = 谁 + 什么时刻 + 属于第几轮 + 说了什么，
 * 页面上就可以像文字对话一样**按轮**铺开、按时间读。
 *
 * ## 安全边界（一个字都没变）
 *
 * 这里**只做编排，不做内容加工**：不改字、不删字、不截断、不排序到别处。
 * 文本仍来自 `useVoiceCall` 里原来那些 `setUserLines` / `setAiLines` 的同一个字符串
 * （安全审计改稿后的文本原样进来），因此"字幕最终等于模型全文"这条既有不变式不受影响。
 * 另外，这里**不涉及任何**风险等级、融合权重、画像分数的读写 —— 通话记录只是旁路呈现。
 *
 * ## 时间口径
 *
 * 消息自带 `tsMs`（`Date.now()` 墙钟）。页面上要显示的是**相对通话开始**的 `mm:ss`
 * （与顶栏计时同一来源），所以模型里不放"格式化好的字符串"，避免两个时间口径各写一遍。
 *
 * ## 发言人的两种来源
 *
 * 1. **用户**：每一条定稿句落进时间轴（`appendUserMessage`）；
 * 2. **AI**：一轮的回复会分多次到达（`speak` 计划 / 流式 delta 逐句 / 改稿替换），
 *    它们是**同一条消息的增量**，所以走 `appendAssistantDelta`：
 *    同一轮的第二次到达就替换尾部，而不是当成新的一条 —— 这与 hook 里
 *    `setAiLines` 的既有语义严格一致。
 */

/** 一条通话消息的讲话人 */
export type CallSpeaker = 'user' | 'ai';

/** 时间轴上的一条消息 */
export interface CallTranscriptEntry {
  /** 稳定唯一 id：渲染 key 与"替换尾部"的判据 */
  id: string;
  speaker: CallSpeaker;
  /** 全文（`speaker === 'ai'` 时可能被本轮的后续增量整体改写） */
  text: string;
  /**
   * 归属轮次。AI 的一条回复与其**前一条**用户消息同号；用户消息的号 = 它是第几次
   * 用户开口（从 1 开始）。页面按这个号分轮展示。
   */
  turn: number;
  /** 写入时刻（墙钟毫秒）—— 与通话开始时刻相减即"本轮第几秒" */
  tsMs: number;
  /** 这一轮由安全审计改稿过（文本已整体替换；改稿后不再出声，见 useVoiceCall） */
  revised?: boolean;
  /** 这一条里有一段**没有被念出来**（打断 / 改稿 / 危机 / 挂断时停播） */
  unspoken?: boolean;
}

/** 按轮分组的投影（页面按这个渲染；`turn` 升序） */
export interface CallTranscriptGroup {
  /** 轮次号（从 1 开始）；`0` 表示"还没归属到任何一轮"的消息（理论上只有异常路径会出现） */
  turn: number;
  /** 组内第一条消息的时刻 —— 页面上"第 N 轮 00:12"用它的相对时间 */
  tsMs: number;
  messages: CallTranscriptEntry[];
}

/** 新建一个空时间轴 */
export function createTranscript(): CallTranscriptEntry[] {
  return [];
}

/** 下一条消息的 id（调用方持有计数器：纯函数不依赖随机数/时间，便于回归） */
export function nextTranscriptId(counter: number): string {
  return `m${counter}`;
}

/**
 * 用户说完一句（服务端定稿）→ 追加一条用户消息。
 *
 * @param list 现有时间轴（**返回新数组**，不改原数组）
 * @param text 定稿文本（调用方已 trim；空串不追加）
 * @param turn 这是用户第几次开口（从 1 开始）
 * @param tsMs 定稿时刻（墙钟毫秒）
 * @param id 消息 id（见 {@link nextTranscriptId}）
 */
export function appendUserMessage(
  list: readonly CallTranscriptEntry[],
  text: string,
  turn: number,
  tsMs: number,
  id: string,
): CallTranscriptEntry[] {
  if (!text.trim()) return list as CallTranscriptEntry[];
  const entry: CallTranscriptEntry = { id, speaker: 'user', text, turn, tsMs };
  return [...list, entry];
}

/**
 * AI 一侧的文本到达（`speak` 计划 / 流式 delta / 审计改稿）。
 *
 * 同一条回复分多次到达：**同一轮**的第二次到达替换尾部那条 AI 消息的文本
 * （与 hook 里 `setAiLines` 的既有语义一致），而不是新增一条。
 * 不同轮（或尾部那条是用户消息）则新增一条。
 *
 * @param list 现有时间轴
 * @param text 本次到达的文本。`mode='delta'` 时是**增量**（追加），
 *             `mode='replace'` 时是**整轮替换后的全文**（改稿 / 计划 / 纯文字路径）
 * @param turn 本轮轮次号（见 {@link CallTranscriptEntry.turn}）
 * @param tsMs 到达时刻（墙钟毫秒）
 * @param id 需要新增时使用的消息 id
 * @param mode 增量追加还是整体替换（默认 `delta`）
 * @param flags 附加事实：`revised` 改稿、`unspoken` 有未朗读段落
 */
export function appendAssistantDelta(
  list: readonly CallTranscriptEntry[],
  text: string,
  turn: number,
  tsMs: number,
  id: string,
  mode: 'delta' | 'replace' = 'delta',
  flags: { revised?: boolean; unspoken?: boolean } = {},
): CallTranscriptEntry[] {
  if (!text) return list as CallTranscriptEntry[];

  const tail = list[list.length - 1];
  const canMerge = tail !== undefined && tail.speaker === 'ai' && tail.turn === turn;

  if (canMerge) {
    const merged: CallTranscriptEntry = {
      ...tail,
      text: mode === 'delta' ? tail.text + text : text,
      // 事实位只增不减：这一轮被改稿过 / 有没念出来的段落，后续增量不能把它抹掉
      revised: tail.revised === true || flags.revised === true ? true : undefined,
      unspoken: tail.unspoken === true || flags.unspoken === true ? true : undefined,
    };
    return [...list.slice(0, -1), merged];
  }

  const entry: CallTranscriptEntry = {
    id,
    speaker: 'ai',
    text,
    turn,
    tsMs,
    ...(flags.revised ? { revised: true } : {}),
    ...(flags.unspoken ? { unspoken: true } : {}),
  };
  return [...list, entry];
}

/**
 * 把时间轴按轮分组（保持原顺序，`turn` 不再重排）。
 *
 * 为什么分组而不是直接平铺：用户要的是"**每轮**对话过程"。分组后页面可以在每轮之间
 * 画一条分隔线并标出轮次与时刻，读起来与文字对话的记录一致；同时保留每条消息自己的
 * 时刻，"轮内两条消息差了几秒"也读得出来。
 */
export function groupTranscript(
  list: readonly CallTranscriptEntry[],
): CallTranscriptGroup[] {
  const groups: CallTranscriptGroup[] = [];
  for (const m of list) {
    const last = groups[groups.length - 1];
    if (last && last.turn === m.turn) last.messages.push(m);
    else groups.push({ turn: m.turn, tsMs: m.tsMs, messages: [m] });
  }
  return groups;
}

/**
 * 归一化一行记录文本：折叠空白，供**渲染期**判断"这条有没有内容"（不改变原文）。
 */
export function transcriptText(list: readonly CallTranscriptEntry[]): string {
  return list.map((m) => m.text).join('\n');
}
