/**
 * 每轮只允许一次开场预缓冲。
 * 首块排程时立即消费资格；后续供音即使晚到、队列已空，也不能重新等待。
 * 只有真正开始下一轮时调用 armTurn()，已有音频排队时沿用当前时间轴。
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
    // 第一块排程时就消费本轮资格。即使下一块在首块耗尽之后才到，
    // 也不能再次等待预缓冲；已有音频时则直接沿用当前播放时间轴。
    this.owed = false;
    return queuedAudioMs <= 0;
  }

  /** 复位（挂断 / 释放） */
  reset(): void {
    this.owed = true;
  }
}
