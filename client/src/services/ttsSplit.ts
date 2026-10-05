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
