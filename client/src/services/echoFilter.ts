/**
 * 文本级自回声过滤 —— S5「方案 D」
 *
 * ## 它解决的是什么问题
 *
 * 通话时麦克风是常开的，而 AI 的声音从扬声器出来会回到麦克风里被 ASR 识别。
 * AEC（`mediaStreamManager` 的默认约束已开）会先拿掉一大部分，但它**不总是生效**：
 * 蓝牙耳机、切后台、某些虚拟声卡都会让它失效，而 `getSettings()` 只是"浏览器说它开着"。
 * 所以还需要一道不依赖任何声学实现的判据。
 *
 * 半双工（AI 说话时丢弃定稿）也挡不干净，它漏掉的正是最危险的那一段：
 *
 * ```
 *   AI 说完最后一个字  ──►  播放队列排空（phase → listening）
 *                          │
 *                          └─ 0.45–1.2 s 后：ASR 端点定稿"刚才听到的 AI 的话"
 *                             ⇒ 此时 phase 已经是 listening，会被当成**用户说的一句**
 *                             ⇒ 落库一条假的用户消息 + 触发一轮莫名其妙的回复
 * ```
 *
 * ⚠️ 这就是"AI 自己跟自己聊起来"的成因。它的窗口只有 1 秒左右，靠加长半双工
 * 挡不干净（端点判定要等尾随静音），必须靠**内容比对**。
 *
 * ## 判据
 *
 * 我们**完全知道自己在说什么**（`TtsPlayer` 的播放日志），所以拿 ASR 的文本与
 * "最近正在播的 TTS 文本"比：先看互相包含，再算**字符二元组重叠率**。
 *
 * 阈值**不是拍的**：`docs/echo_calibration.md` 里那份 40 组 `(spoken, heard)`
 * 配对（10 句 AI 回复 × 3 种声学耦合质量 + 10 句用户话）是标定它的依据，
 * 复跑命令在文末。改阈值前请先看那份数据。
 *
 * ## 为什么单独一个文件
 *
 * 它是**纯函数 + 一个不碰 DOM 的小状态机** —— 可以在 Node 里直接 `import` 出来
 * 逐条断言（`_exp_echo_decide.js` 就是这么算标定表的），
 * 不必为了测一个字符串判据去搭一整套语音环境。
 *
 * 复现标定：
 * ```powershell
 * & 'D:\Develop\AIC\.venv\Scripts\python.exe' D:\Develop\AIC\_exp_echo_coupling.py
 * node D:\Develop\AIC\_exp_echo_decide.js
 * ```
 */

/** 归一化：去掉空白与标点，只留下能参与比对的字符 */
export function normalizeForEcho(text: string): string {
  return (text || '').replace(/[\s\p{P}\p{S}]/gu, '');
}

/** 字符二元组集合（中文按单字、英文按字母，够用） */
function bigrams(s: string): Set<string> {
  const out = new Set<string>();
  for (let i = 0; i + 1 < s.length; i++) out.add(s.slice(i, i + 2));
  return out;
}

/** 判定结果；带 `reason` / `overlap` 便于调试与后续重新标定 */
export interface EchoVerdict {
  echo: boolean;
  reason: string;
  /** 命中的那句 AI 原文（若有） */
  matched?: string;
  /** 最佳二元组重叠率 0–1（判据的原始分数，不是四舍五入过的） */
  overlap: number;
}

/** 与某句 AI 原文的相似度 */
export interface EchoScore {
  /** 最佳重叠率 0–1 */
  overlap: number;
  /** 是否互相包含（ASR 把 AI 的整句/半句听了回来） */
  containment: boolean;
  /** 取得该分数的那句 AI 原文 */
  matched: string;
}

/**
 * 低于这个长度不判回声（判不准时宁可放行用户的话）。
 *
 * 标定依据：30 组真回声里，归一化后的**最短**"听到的话"是 10 字
 * （`docs/echo_calibration.md` 表 A 的 ai05「你好呀，今天想说点什么？」）。
 * 而"嗯""好的""对，没错"这类真实短插话不该被匹配 —— 它们太短，
 * 二元组太少，随便两句都能撞上，假阳性极高。
 * 取 4 是给"只听到半句"的中间结果留余量（partial 可能只有几个字）。
 */
export const ECHO_MIN_LEN = 4;

/**
 * 二元组重叠率阈值 —— **实测标定值**，不要凭感觉改。
 *
 * 依据（`docs/echo_calibration.md`，40 组真实配对，seed=42）：
 *
 * | 量 | 实测 |
 * |---|---|
 * | 真回声（30 组）**最低**重叠 | **41.7%**（ai10/clean，号码被念糊那一句） |
 * | 普通负样本（8 句）**最高**重叠 | **37.5%**（u02「嗯，最近考试压力特别大」） |
 *
 * 阈值取在两者之间 → **0.40**。此时真回声 30/30 全中；负样本只剩 1 例误判，
 * 且那一例是**故意构造的困难负样本**（u09「对，我最近睡眠不好，感觉特别烦」——
 * 用户把 AI 刚说的话原样重复了一遍，83.3%）。文本级判据无法区分
 * "回声"与"用户复述"，这是本方案已知的边界，靠 2 秒时间窗把它压到最小
 * （见 {@link ECHO_TAIL_MS}）。
 *
 * ⚠️ 41.7% 那个下限来自"号码被念糊"：`400 161 9995` 被念成
 * 「四百六十一九九九九五」，数字部分完全对不上。而施工单 §S6.3 本来就
 * **禁止把号码特征片段送去合成**，所以这个最坏情况在生产里根本不会出现；
 * 这里仍按它标定，是刻意留出的余量。
 *
 * ⚠️ 也**不要**沿用第一版的 0.6：那会漏掉 3/30 条真回声（命中率降到 90%），
 * 漏掉的正是"AI 自己的话被当成用户消息落库"的那条路径。
 */
export const ECHO_OVERLAP_THRESHOLD = 0.4;

/**
 * 播完之后还要"记得"多久（毫秒）。
 *
 * 依据是端点判定延迟：AI 说完最后一个字到 ASR 定稿之间要等 0.8s 尾随静音
 * （服务端 rule2）加一次往返，实测 0.45–1.2 s。取 2 s 留一倍余量。
 *
 * ⚠️ **不能像 S4 那样用 8 秒**：窗口开得越久，误杀越多 ——
 * 用户过一会儿自己说一句"我最近睡眠不好"（正是 AI 刚说过的词）是**真的在说话**，
 * 困难负样本 u09/u10 量的就是这个（见 `docs/echo_calibration.md` 表 C）。
 */
export const ECHO_TAIL_MS = 2000;

/**
 * 算"这段听到的话"与 AI 最近说过的哪一句最像。
 *
 * 判据顺序（与最终决策一致，抽出来是为了让标定脚本能扫阈值）：
 *   1. 互相包含 ⇒ 直接满分（`overlap = 1`）
 *   2. 否则取字符二元组重叠率的最大值
 *
 * @param heard ASR 出来的文本
 * @param spokenTexts AI 说过的文本（必须用服务端回传的 `spoken_text`）
 */
export function scoreAgainstSpoken(heard: string, spokenTexts: string[]): EchoScore {
  const h = normalizeForEcho(heard);
  const candidates = (spokenTexts || [])
    .map((s) => ({ raw: s, norm: normalizeForEcho(s) }))
    .filter((c) => c.norm.length >= ECHO_MIN_LEN);

  if (h.length === 0 || candidates.length === 0) {
    return { overlap: 0, containment: false, matched: '' };
  }

  for (const c of candidates) {
    if (c.norm.includes(h) || h.includes(c.norm)) {
      return { overlap: 1, containment: true, matched: c.raw };
    }
  }

  const hb = bigrams(h);
  if (hb.size === 0) return { overlap: 0, containment: false, matched: '' };
  let best = 0;
  let bestRaw = '';
  for (const c of candidates) {
    const cb = bigrams(c.norm);
    if (cb.size === 0) continue;
    let hit = 0;
    for (const g of hb) if (cb.has(g)) hit += 1;
    const ratio = hit / hb.size;
    if (ratio > best) {
      best = ratio;
      bestRaw = c.raw;
    }
  }
  return { overlap: best, containment: false, matched: bestRaw };
}

/**
 * 这段"听到的话"是不是 AI 自己刚说过的？（纯函数版）
 *
 * @param heard ASR 出来的文本
 * @param spokenTexts AI 说过的文本
 * @param opts.threshold 覆盖 {@link ECHO_OVERLAP_THRESHOLD}（标定用）
 */
export function looksLikeSelfEcho(
  heard: string,
  spokenTexts: string[],
  opts: { threshold?: number; minLen?: number } = {},
): EchoVerdict {
  const threshold = opts.threshold ?? ECHO_OVERLAP_THRESHOLD;
  const minLen = opts.minLen ?? ECHO_MIN_LEN;
  const h = normalizeForEcho(heard);
  if (h.length < minLen) {
    return { echo: false, overlap: 0, reason: `太短（${h.length} < ${minLen}），无法判断，放行` };
  }
  if (!spokenTexts || spokenTexts.length === 0) {
    return { echo: false, overlap: 0, reason: 'AI 最近没有说过话，无从比对' };
  }

  const score = scoreAgainstSpoken(heard, spokenTexts);
  if (score.containment) {
    return {
      echo: true,
      overlap: 1,
      reason: '听到的内容与 AI 原句互相包含',
      matched: score.matched,
    };
  }
  if (score.overlap >= threshold) {
    return {
      echo: true,
      overlap: score.overlap,
      reason: `与 AI 原句二元组重叠 ${(score.overlap * 100).toFixed(0)}% ≥ ${(threshold * 100).toFixed(0)}%`,
      matched: score.matched,
    };
  }
  return {
    echo: false,
    overlap: score.overlap,
    reason: `最大重叠 ${(score.overlap * 100).toFixed(0)}% < ${(threshold * 100).toFixed(0)}%`,
  };
}

/** 一条"AI 说过的话" */
interface SpokenRecord {
  text: string;
  /** 开始出声的墙钟时刻（`performance.now()` 域） */
  from: number;
  /** 说完的时刻；`null` 表示还在播 */
  until: number | null;
}

/** `EchoFilter` 的构造参数（都为标定/测试留了口子） */
export interface EchoFilterOptions {
  threshold?: number;
  minLen?: number;
  /** 播完之后还记得多久 */
  tailMs?: number;
  /** 同时记住多少句（防长通话里无限增长） */
  maxRecords?: number;
}

/**
 * 通话期使用的回声过滤器（施工单 §S5.2 的接口）。
 *
 * 与纯函数版的区别只有一点：**它带时间**。
 * "AI 说过什么"必须配"什么时候说的"，否则一个 8 秒的回溯窗口会把
 * 用户十分钟里所有跟 AI 撞词的话都判成回声。
 *
 * 典型接线（`useVoiceCall`）：
 * ```ts
 * onStart: (seq, text) => echoFilter.note(text, performance.now())
 * onEnd:   (seq, text) => echoFilter.retire(text, performance.now())
 * onDrained: () => echoFilter.retireAll(performance.now())   // stopAll 掐断的也要退场
 * ```
 *
 * ⚠️ `stopAll()` **不会**触发播放器的 `onEnd`（`onended` 被置空以避免重入），
 * 所以"被打断的那半句"只能靠 `retireAll` 退场 —— 它的 `note` 已经在 `onStart` 记过了。
 */
export class EchoFilter {
  private readonly threshold: number;
  private readonly minLen: number;
  private readonly tailMs: number;
  private readonly maxRecords: number;
  private records: SpokenRecord[] = [];

  constructor(opts: EchoFilterOptions = {}) {
    this.threshold = opts.threshold ?? ECHO_OVERLAP_THRESHOLD;
    this.minLen = opts.minLen ?? ECHO_MIN_LEN;
    this.tailMs = opts.tailMs ?? ECHO_TAIL_MS;
    this.maxRecords = opts.maxRecords ?? 24;
  }

  /** AI 开始说一句 —— 登记它，从这里开始它可能回到麦克风里 */
  note(spokenText: string, at: number): void {
    const text = spokenText || '';
    if (normalizeForEcho(text).length < this.minLen) return;
    // 同一句被重复登记（例如重连后重播）时不新增，只保留最早的一次
    const existing = this.records.find((r) => r.text === text && r.until === null);
    if (existing) {
      existing.from = Math.min(existing.from, at);
      return;
    }
    this.records.push({ text, from: at, until: null });
    if (this.records.length > this.maxRecords) {
      this.records.splice(0, this.records.length - this.maxRecords);
    }
  }

  /** AI 把这一句说完了；从此刻起再记 {@link ECHO_TAIL_MS} 毫秒 */
  retire(spokenText: string, at: number): void {
    const r = this.records.find((x) => x.text === spokenText && x.until === null);
    if (r) r.until = at;
  }

  /** 全部退场（`stopAll()` 掐断时用：被打断的那半句不会走 `onEnd`） */
  retireAll(at: number): void {
    for (const r of this.records) if (r.until === null) r.until = at;
  }

  /** 当前仍在"可能回到麦克风"窗口内的文本（供诊断面板展示） */
  active(now: number): string[] {
    this.prune(now);
    return this.records.map((r) => r.text);
  }

  /** ASR 文本是否为 AI 自己的回声 */
  isEcho(asrText: string, now: number): boolean {
    return this.judge(asrText, now).echo;
  }

  /** 带理由的判定（`?debug=1` 与闸门读它） */
  judge(asrText: string, now: number): EchoVerdict {
    this.prune(now);
    if (this.records.length === 0) {
      return { echo: false, overlap: 0, reason: '窗口内 AI 没有说话' };
    }
    return looksLikeSelfEcho(
      asrText,
      this.records.map((r) => r.text),
      { threshold: this.threshold, minLen: this.minLen },
    );
  }

  /** 清空（挂断时调用） */
  clear(): void {
    this.records = [];
  }

  /** 丢掉已经过期的记录 */
  private prune(now: number): void {
    this.records = this.records.filter((r) => {
      const end = r.until ?? now;
      return now - end <= this.tailMs;
    });
  }
}
