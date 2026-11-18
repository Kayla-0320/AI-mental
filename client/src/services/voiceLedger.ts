/**
 * 「文本 ↔ 语音」对账链（**纯函数模块，可直接在 Node 里跑回归**）
 *
 * 用户报告（2026-10-11）："AI 的回复和语音内容完全不匹配，而且语音还输出不全"。
 * 这两句指向两条独立链路，靠猜分不开 —— 所以先把**可对账的四份文本**定义成数据，
 * 再让每一层把实际发生的事实填进去：
 *
 * ```
 *   A shown   页面最终显示的文本
 *   B plan    SSE `speak` 的 text / segments
 *   C said    实际发出的每个 say（按序拼接）
 *   D spoken  实际出声的 spoken_text（按播放顺序拼接）
 * ```
 *
 * 逐层比较，**第一处不一致就是第一优先修复点**：
 *
 * | 比较 | 说明什么 |
 * |---|---|
 * | A ≠ B | 字幕 / revise 的替换逻辑（改稿把已播出的说法悄悄换掉） |
 * | B ≠ C | 分段、过滤（号码/标点）、重复入队、"计划 + delta 双路合成" |
 * | C ≠ D | seq 串单 / 回包错配（音频串到了别的片段） |
 * | D 对但听着断 | 播放队列与调度（看 `TtsUnderrunStats` 的空洞读数） |
 *
 * ⚠️ 有意跳过的片段（号码只上屏、纯标点不合成）与术语规范化**不算故障**，
 * 它们以"预期例外"单列 —— 否则对账链会天天误报，最后没人看。
 *
 * 判据与阈值设计见 `docs/voice_call_text_voice_ledger.md`。
 * 本模块不依赖 React / DOM / 网络，回归脚本：`scripts/verify_tts_text_ledger.ts`。
 */

import { normalizeForCompare, planJoinMatchesText, type SkippedSpeechFragment } from './ttsSplit';

/** 单轮合成的来源（同一轮只允许一个，见 `SPEAK_PLAN_GRACE_MS` 的注释） */
export type SpeakPlanMode = 'pending' | 'speak-plan' | 'delta-fallback';

/** 一个已发出的合成单元（C 端点的元素） */
export interface SpokenUnitRecord {
  seq: number;
  text: string;
}

/** 对账链的输入（由调用方从它自己的累计量里填） */
export interface VoiceTextLedgerInput {
  turnId: number;
  source: SpeakPlanMode;
  /** A：页面最终显示的文本 */
  shown: string;
  /** B：`speak` 计划；没收到为 null */
  planText: string | null;
  planSegments: string[] | null;
  /** C：实际发出的每个 say（按序） */
  saidUnits: readonly SpokenUnitRecord[];
  /** D：实际出声的 spoken_text（按播放顺序） */
  spokenUnits: readonly SpokenUnitRecord[];
  /** 有意跳过的片段（号码 / 纯标点 / 空白） */
  skipped: readonly SkippedSpeechFragment[];
  /** 本轮的处理笔记（计划迟到、按定稿文本撤回尾段……） */
  notes: readonly string[];
  /** 未被朗读的尾部（打断/改稿/危机/挂断时非空） */
  unspokenTail: string | null;
  /**
   * 本轮是否收到过 `revise`（审计改稿）。
   *
   * 只为把 A≠B 标成**预期**：改稿后屏幕上是新稿、而语音来源是本轮原先的定稿，
   * 两者本来就不同 —— 不标出来，每一轮改稿都会被当成"对账失败"。
   * （2026-10-11 起新稿**会被补念**，所以这里记的是"说法不同"，不是"没念"。）
   */
  sawRevise?: boolean;
}

/** 一轮的对账记录 */
export interface VoiceTextLedger extends Omit<VoiceTextLedgerInput, 'saidUnits' | 'spokenUnits' | 'skipped' | 'notes'> {
  said: string;
  spoken: string;
  saidUnits: SpokenUnitRecord[];
  spokenUnits: SpokenUnitRecord[];
  skipped: SkippedSpeechFragment[];
  notes: string[];
  /** 人能读的差异清单（空数组 = 四份文本一致） */
  exceptions: string[];
}

/**
 * 把一轮的事实整理成对账记录。
 *
 * 差异清单的**顺序即优先级**：第一行就是"第一处不一致"，照它去改对应的那一层。
 *
 * @param input 本轮四份文本与笔记
 * @returns 对账记录（含 `exceptions`）
 */
export function buildVoiceTextLedger(input: VoiceTextLedgerInput): VoiceTextLedger {
  const said = input.saidUnits.map((u) => u.text).join('');
  const spoken = input.spokenUnits.map((u) => u.text).join('');
  const exceptions: string[] = [];

  const nShown = normalizeForCompare(input.shown);
  const nSaid = normalizeForCompare(said);
  const nSpoken = normalizeForCompare(spoken);
  const nPlan = input.planText === null ? null : normalizeForCompare(input.planText);

  // ① 计划自相矛盾：`segments` 拼接对不上 `text`（服务端契约，按空白归一化后比较）
  if (input.planText !== null && input.planSegments !== null
      && !planJoinMatchesText(input.planText, input.planSegments)) {
    exceptions.push('B 自相矛盾：speak.segments 拼接 ≠ speak.text');
  }

  // ② 本轮根本没有计划（纯文字端 / 非流式回退 / 计划违规被拒收）
  if (input.source === 'pending') {
    exceptions.push('本轮没有可用的 speak 计划：音频按 delta 回退切分，与定稿文本可能不同源');
  } else if (input.source === 'delta-fallback') {
    exceptions.push('计划迟到：音频由 delta 驱动（已按定稿文本撤回多余尾段，见 notes）');
  }

  // ③ B ≠ C：计划说要念的 vs 实际发出去的（号码/标点是预期例外，单独列）
  if (nPlan !== null && nSaid !== nPlan) {
    if (input.sawRevise) {
      // 改稿轮：C 里多了"补念的新稿尾段"，而 B 是改稿前的计划 —— 必然不等，且不是故障。
      exceptions.push(
        `预期例外：B≠C 系审计改稿（补念的尾段不在原计划里；发出 ${nSaid.length} 字 vs 计划 ${nPlan.length} 字）`,
      );
    } else {
      exceptions.push(
        `B≠C：计划 ${nPlan.length} 字，实际发出 ${nSaid.length} 字` +
          (input.skipped.length ? `（有意跳过 ${input.skipped.length} 段）` : ''),
      );
    }
  }
  const phoneCount = input.skipped.filter((s) => s.reason === 'phone').length;
  if (phoneCount > 0) exceptions.push(`预期例外：${phoneCount} 段号码只上屏不朗读`);

  // ④ C ≠ D：发出去的 vs 真正出声的 —— 这一条非零就是**串单或音频缺失**
  if (nSpoken !== nSaid) {
    exceptions.push(`C≠D：发出 ${nSaid.length} 字，实际出声 ${nSpoken.length} 字（seq 串单或音频缺失）`);
  }

  // ⑤ A ≠ (B 优先，否则 C)：页面显示的与语音来源
  const ref = nPlan ?? nSaid;
  if (nShown !== ref) {
    if (input.sawRevise) {
      // 改稿轮：屏幕上是新稿（A），语音来源是改稿前的定稿（B）—— 必然不等。
      // 新稿的未播部分已由 `planReviseSpeech` 补念（见 notes），所以这不是缺音。
      exceptions.push('预期例外：A≠B 系审计改稿（新稿未播部分已补念，前段沿用已播出的说法）');
    } else {
      exceptions.push(`A≠${nPlan !== null ? 'B' : 'C'}：页面显示 ${nShown.length} 字，语音来源 ${ref.length} 字`);
    }
  }

  // ⑥ 未朗读的尾段（打断/改稿/危机/挂断）：不是故障，但必须让用户与排查者都知道
  if (input.unspokenTail) {
    exceptions.push(`本轮有 ${input.unspokenTail.length} 字未朗读（被打断/改稿/危机，页面已标注）`);
  }

  return {
    ...input,
    said,
    spoken,
    // 复制一份：台账是"快照"，不能随后续追加而变
    saidUnits: [...input.saidUnits],
    spokenUnits: [...input.spokenUnits],
    skipped: [...input.skipped],
    notes: [...input.notes],
    exceptions,
  };
}
