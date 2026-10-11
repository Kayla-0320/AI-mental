/**
 * 播放缓冲区日志 —— 回答用户那个关键问题：**卡在句子中间，还是卡在两句之间**。
 *
 * ## 为什么要单独一件（2026-10-11 用户："现在可以正常出声了，但是中途还有些小卡顿"）
 *
 * 两条候选病因的**修复方向完全不同**，而它们听感相似：
 *
 * | 现象 | 病因 | 该动哪里 |
 * |---|---|---|
 * | 同一句话**中间**偶尔停一下 | 供音速度跟不上播放（合成 RTF > 1，缓冲被吃空） | 预缓冲 / 推理速度 |
 * | 每句话**之间**有明显停顿 | 下一段太晚才开始供音（片段调度） | 提前发下一句的合成请求 |
 *
 * 判据只有一个：**卡顿那一刻缓冲区是不是见底**。所以这里做两件事：
 *   1. `onChunk`：每块入库时记一行（时刻 / 该块音频时长 / **入库后的缓冲剩余** / 是否同句）；
 *   2. `onUnderrun`：真的见底时，把**最近若干行**一起打出来 —— 那几行就是"卡顿的原因"。
 *
 * ## 口径（与 `ttsPlayer` 的空洞统计对齐）
 *
 * * `sameSeq = true` 的断档 ⇒ **句内供给不足**（"一句话中间卡"）；
 * * `sameSeq = false` 的断档 ⇒ **跨句接缝**（"每句话之间卡"）。
 *
 * ⚠️ 这里**只记录、不决策**：不改排程、不改缓冲策略、不改任何音频参数。
 * 它是"先加日志再决定改哪一层"那一步的产物。
 */

/** 一块 PCM 入库时的快照 */
export interface ChunkTrace {
  /** 入库时刻（墙钟毫秒，`performance.now()`） */
  atMs: number;
  seq: number;
  /** 这一块的音频时长（毫秒） */
  chunkMs: number;
  /** 入库**之后**播放器里还剩多少音频（毫秒）—— 缓冲区水位 */
  bufferMs: number;
  /** 与上一块是否同一句（true = 句内供给；false = 跨句接缝） */
  sameSeq: boolean;
  /** 这一块进来时是否发生了"见底"（空洞） */
  starved: boolean;
  /** 见底时长（毫秒）；没见底为 0 */
  starveMs: number;
}

/** 环形日志：只留最近 N 行（长通话里不涨内存） */
const MAX_TRACES = 240;

/** 见底时的现场快照 */
export interface StarveEvent {
  atMs: number;
  seq: number;
  /** 见底时长（毫秒） */
  starveMs: number;
  /** 是否句内（true = "一句话中间卡"，false = "句子之间卡"） */
  sameSeq: boolean;
}

export class TtsBufferLog {
  private traces: ChunkTrace[] = [];
  private totalChunks = 0;
  /** 句内见底次数（"一句话中间卡"） */
  private withinSentenceStarves = 0;
  /** 跨句见底次数（"两句之间卡"） */
  private gapStarves = 0;
  /** 累计见底时长 */
  private starveTotalMs = 0;
  private events: StarveEvent[] = [];

  /** 记一块入库 */
  onChunk(t: ChunkTrace): void {
    this.totalChunks += 1;
    if (t.starved) {
      this.starveTotalMs += t.starveMs;
      if (t.sameSeq) this.withinSentenceStarves += 1;
      else this.gapStarves += 1;
      this.events.push({
        atMs: t.atMs,
        seq: t.seq,
        starveMs: t.starveMs,
        sameSeq: t.sameSeq,
      });
      if (this.events.length > 60) this.events.shift();
    }
    this.traces.push(t);
    if (this.traces.length > MAX_TRACES) this.traces.shift();
  }

  /** 当前缓冲区水位（毫秒） */
  get bufferMs(): number {
    const last = this.traces[this.traces.length - 1];
    return last ? last.bufferMs : 0;
  }

  /** 面板/日志用的汇总 */
  summary(): {
    chunks: number;
    withinSentenceStarves: number;
    gapStarves: number;
    starveTotalMs: number;
    bufferMs: number;
  } {
    return {
      chunks: this.totalChunks,
      withinSentenceStarves: this.withinSentenceStarves,
      gapStarves: this.gapStarves,
      starveTotalMs: Math.round(this.starveTotalMs),
      bufferMs: Math.round(this.bufferMs),
    };
  }

  /**
   * 见底现场：**最近 12 块**的入库明细 + 每条见底事件。
   *
   * 这就是用户要的"卡顿那一刻的日志"：把 `bufferMs` 一路看下来，
   * 是在句内被吃干（`sameSeq=true` 星标），还是在两句之间空着等（`sameSeq=false`）。
   */
  dump(recentChunks = 12): string {
    const s = this.summary();
    const lines: string[] = [
      `[播放缓冲] 共入库 ${s.chunks} 块 · 见底 ${s.withinSentenceStarves + s.gapStarves} 次`
      + `（句内 ${s.withinSentenceStarves} / 跨句 ${s.gapStarves}）· 累计 ${s.starveTotalMs}ms`
      + ` · 当前水位 ${s.bufferMs}ms`,
    ];
    if (this.events.length > 0) {
      lines.push('  见底事件（近 %d 条）:'.replace('%d', String(Math.min(this.events.length, 20))));
      for (const e of this.events.slice(-20)) {
        lines.push(
          `    ⚠️ t=${Math.round(e.atMs)}ms seq=${e.seq} 断 ${Math.round(e.starveMs)}ms`
          + ` [${e.sameSeq ? '句内供给不足' : '跨句接缝'}]`,
        );
      }
    }
    const recent = this.traces.slice(-recentChunks);
    if (recent.length > 0) {
      lines.push('  最近入库的块（时刻 / 块长 / 入库后水位 / 同句）:');
      for (const t of recent) {
        lines.push(
          `    t=${Math.round(t.atMs)} seq=${t.seq} 块=${Math.round(t.chunkMs)}ms`
          + ` 水位=${Math.round(t.bufferMs)}ms ${t.sameSeq ? '同句' : '跨句'}`
          + (t.starved ? ` ⚠️断${Math.round(t.starveMs)}ms` : ''),
        );
      }
    }
    return lines.join('\n');
  }

  /** 会话重置（新通话） */
  reset(): void {
    this.traces = [];
    this.events = [];
    this.totalChunks = 0;
    this.withinSentenceStarves = 0;
    this.gapStarves = 0;
    this.starveTotalMs = 0;
  }
}
