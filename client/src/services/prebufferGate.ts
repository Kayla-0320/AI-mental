/**
 * 播放队列的**开场判定** —— 预缓冲该不该在这块音频上生效。
 *
 * ## 为什么单独抽出来（2026-10-11 用户指出："直接检查预缓冲的触发代码，
 * ## 确认它是否通过'每轮已启动播放'的标记严格限制为一次"）
 *
 * 用户这个怀疑是对的，而且**原实现确实错了**。原来的判据是：
 *
 * ```ts
 * const queueWasEmpty = this.nextStartTime <= 0;   // ← 太松
 * ```
 *
 * 而 `nextStartTime` 在**每次队列被抽干时都会归零**（句播完、`stopAll` 都写 0）。
 * 于是 CosyVoice3 阵发式供音的间隙里：队列空了（`nextStartTime = 0`）→ 下一块
 * 到达 → 又被判成"开场" → **再推后一整个预缓冲**。本该在断档后立刻接上的音频
 * 变成"再等 300ms"，**它把要治的卡顿又放大了一次**。
 *
 * ## 正确口径：一次通话里**每轮只许生效一次**
 *
 * 状态机只有两个状态，三个事件：
 *
 * ```
 *      armTurn()（新一轮开始）        consume(还有音频在飞?)
 *   ┌──────────────────────┐   ┌──────────────────────────┐
 *   │ owed = true          │──▶│ 有音频在飞 ⇒ owed = false │   ← 用掉了，本轮不再给
 *   │ （欠一次预缓冲）      │   │ 队列真空 ⇒ 保持 owed=true │   ← 还没用掉，留给下一块
 *   └──────────────────────┘   └──────────────────────────┘
 * ```
 *
 * ⚠️ 两个容易写错的地方，都在这里被明确固定：
 *
 * 1. **不能按"队列空不空"决定是否重新欠账** —— 那正是原 bug（阵发间隙也会归零）。
 *    只有 `armTurn()`（`useVoiceCall.runTurn` 真正开一轮）才重新欠账；
 * 2. **不能按"这块音频"清账，要按"队列是否真起飞了"清账** ——
 *    第一块常常是先到的那一小块（服务端流式的第一段），紧接着还有几块在路上；
 *    若第一块就把 `owed` 清掉，而它在预缓冲窗口内**一块都没再来**，
 *    真正开播时缓冲里就只有那一小块，预缓冲白给。
 *
 * 纯状态机、不碰 DOM/WebAudio，所以可以离线回归（见 `scripts/verify_prebuffer_gate.ts`）。
 */
export class PrebufferGate {
  /** 是否还欠一次预缓冲（只由 `armTurn` 置真） */
  private owed: boolean;

  /**
   * @param owedAtStart 是否"开局就欠一次"。
   *   播放器默认 `true`：第一轮之前没有任何 `armTurn` 调用，也要吃到预缓冲。
   */
  constructor(owedAtStart = true) {
    this.owed = owedAtStart;
  }

  /** 本轮是否还欠一次预缓冲（面板/日志读数用） */
  get pending(): boolean {
    return this.owed;
  }

  /**
   * 新一轮开始 —— **唯一**能让预缓冲重新生效的地方。
   *
   * 调用点：`useVoiceCall.runTurn`（用户说完、真正开一轮的那一刻）。
   * 绝不能挂在"队列播空"上（见类注释第 1 条）。
   */
  armTurn(): void {
    this.owed = true;
  }

  /**
   * 排一块音频时问它：这块要不要**等预缓冲**？
   *
   * @param queuedAudioMs 此刻**已经排在播放时间轴上、还没播完**的音频时长（毫秒）
   * @returns `true` = 这块是开场，调用方应把起点推后预缓冲那么多；
   *          `false` = 照原样紧接排程（不等待）
   */
  decide(queuedAudioMs: number): boolean {
    if (!this.owed) return false;
    // 队列里已经有音频在飞 ⇒ 这不是"开场那一下"，而是同一轮的后续块：
    // 预缓冲已经"用掉"了（用掉的标志就是有音频起飞），此后本轮不再生效。
    if (queuedAudioMs > 0) {
      this.owed = false;
      return false;
    }
    // 队列真空 ⇒ 这才是开场。**刻意不在这里清账**：
    // 等积攒到下一块时若已有音频在飞，那时再清（见类注释第 2 条）。
    return true;
  }

  /** 复位（挂断 / 释放） */
  reset(): void {
    this.owed = true;
  }
}
