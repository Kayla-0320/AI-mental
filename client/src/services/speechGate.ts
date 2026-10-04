/**
 * 解码打转判定 —— 丢掉"声学模型在没有真实语音时退化出来的假文本"
 *
 * 为什么需要它：
 *   声学模型不是"有没有人说话"的判别器。把静音或底噪喂进去，它照样会解码出一串字，
 *   而且因为声学上下文为空，输出会退化成重复的单字串，典型形态是
 *   "好的的的的。我的我的没的。"（见 `_probe_asr_halluc.py` 的实测输出：
 *   纯数字静音 6s → final「我们我我的时呢。」；-60dBFS 白噪 6s → final「好的的的的好了。」）。
 *
 *   聊天页的语音录入会把定稿**直接追加进输入框**，于是"打开语音、没人说话，
 *   输入框自己冒出一串假话"就成立了。
 *
 * ⚠️ 这里只保留了**文本侧的兜底**。原先同文件里的 `SpeechGate`（逐块 RMS 的音频
 *   活动门）已删除：它按"说话停顿就关门"工作，而聊天页语音输入改成了整段录入
 *   （停顿不切断、由用户点结束），那道门会把句间停顿从音频里抠掉，反而破坏上下文。
 *   现在由 `streamingAsr.ts` 只负责掐掉开头那段静音，其余音频原样上行。
 *
 * 定位：**兜底**，不是主要防线。
 * 所以刻意把阈值卡在 4 —— 只丢掉几乎不可能是人话的串，宁可漏杀不可错杀。
 */
export function looksLikeDecoderLoop(text: string, maxRepeat = 4): boolean {
  const t = text.replace(/[\s，。、！？,.!?~…—]/g, '');
  if (t.length < maxRepeat) return false;
  let run = 1;
  for (let i = 1; i < t.length; i++) {
    if (t[i] === t[i - 1]) {
      run += 1;
      if (run >= maxRepeat) return true;
    } else {
      run = 1;
    }
  }
  return false;
}
