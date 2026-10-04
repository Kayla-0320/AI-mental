/**
 * 媒体设备统一管理 —— 摄像头 / 麦克风的**唯一拥有者**。
 *
 * 为什么需要它（背景）：
 * 改造前仓库里有 5 个各自 `getUserMedia` 的调用点，且每个 hook 都是
 * `streamRef.current = stream` 这种**覆盖写** —— 开新流之前不停旧流。
 * 结果是"再开一次"永远等于"漏一份"：旧的轨道仍然 live（浏览器麦克风指示灯亮着），
 * 但已无任何引用可达，`stop()` 永远够不到它。
 *
 * 在聊天/试用场景里这个问题靠"用户不会乱点"勉强掩盖得住；但视频通话里
 * "再开一次"是**自动事件** —— 切前后摄像头、断线重连、组件重挂载（StrictMode）、
 * 手机来电打断、切后台再回来。每一次都是一次泄漏。
 *
 * 因此这个模块做三件事：
 * 1. **幂等**：同一个 `(kind, consumer)` 重复 `acquire` 拿到的是**同一份**流，
 *    底层 `getUserMedia` 只调用一次（含并发调用，用 in-flight promise 去重）。
 * 2. **引用计数**：一份流可以被多个消费者共用；只有最后一个 `release` 之后才真的
 *    `stop()` 轨道。避免"通话页和感知 hook 各开一路、互相抢同一个设备"。
 * 3. **可诊断**：`holders(kind)` 能回答"现在谁在用摄像头"；`onEnded` 能把
 *    "设备被别的 App 抢走 / 被拔掉"这类**流级失败**通知出去，
 *    而不是像以前那样让 UI 一直显示"检测中"而实际零数据。
 *
 * 本模块不依赖 React，也不碰任何 DOM 元素 —— 它只负责流的生命周期。
 */

/** 设备种类。 */
export type MediaKind = 'camera' | 'microphone';

/**
 * 消费者身份。一个消费者可以同时持有摄像头和麦克风，
 * 但**释放要显式带上 kind** —— 否则面部检测停止时会误停掉麦克风。
 */
export type MediaConsumer =
  | 'perception-facial'
  | 'perception-voice'
  | 'call'
  | 'call-video'
  | 'counselor-room'
  | 'blink';

/** 设备不可用的原因 —— 直接用于界面上给用户看的提示。 */
export type MediaUnavailableReason =
  | 'permission-denied'
  | 'no-device'
  | 'device-busy'
  | 'unsupported'
  | 'unknown';

/** `acquire` 失败时抛出的错误，带机器可判断的 `reason`。 */
export class MediaUnavailableError extends Error {
  readonly reason: MediaUnavailableReason;
  readonly kind: MediaKind;

  constructor(kind: MediaKind, reason: MediaUnavailableReason, message: string) {
    super(message);
    this.name = 'MediaUnavailableError';
    this.kind = kind;
    this.reason = reason;
  }
}

/** 流级失败事件（轨道 ended），供"摄像头已断开"这类提示使用。 */
export interface MediaEndedEvent {
  kind: MediaKind;
  /** 事件发生时还持有该流的所有消费者（用于判断是否影响当前界面）。 */
  consumers: MediaConsumer[];
  /** 原始 track 的 label，便于排查是哪台设备。 */
  trackLabel: string;
}

type EndedListener = (event: MediaEndedEvent) => void;

interface StreamEntry {
  stream: MediaStream;
  /** 引用计数：非空即表示该流仍被需要，不能 stop。 */
  consumers: Set<MediaConsumer>;
  constraints?: MediaStreamConstraints;
  /** 已挂上的 track ended 监听，用来在条目销毁时摘掉。 */
  detach: () => void;
}

/**
 * 默认采集约束。
 *
 * ## 麦克风为什么开 AEC / NS / AGC（施工单 S5.3 改的就是这一行）
 *
 * 通话时麦克风是常开的，而 AI 的声音从扬声器出来会回到麦克风里。没有回声消除
 * （AEC）时，ASR 会把 AI 自己的话识别成"用户在说话"，于是
 *
 *   · 打断逻辑会在 AI 每说完一句时都被自己的声音触发；
 *   · 定稿还会被当成用户的一轮输入落库 —— 表现是"AI 自己跟自己聊起来"
 *     （实测配对数据见 `docs/echo_calibration.md`）。
 *
 * 文本级过滤（`echoFilter.ts`）是**第二道**保险，它靠"我们知道自己说了什么"，
 * 但那是事后判据；能不让回声进模型就别让它进。
 *
 * ## 为什么用 `ideal` 而不是硬约束
 *
 * `{ echoCancellation: true }` 是**硬**约束：某些虚拟声卡 / 采集卡 /
 * 蓝牙设备的驱动根本不声明这项能力，硬约束会直接 `OverconstrainedError`，
 * 结果是"为了降回声，把麦克风整个弄不能用了" —— 通话彻底废掉。
 * `ideal` 是"尽量满足"，拿不到就退化成裸音频。
 *
 * ⚠️ 所以**绝不能假设 AEC 生效了**：通话页要读 `track.getSettings()` 的实际值
 * 公示出来（`voice_call_plan.md` §6.1）。
 */
const DEFAULT_CONSTRAINTS: Record<MediaKind, MediaStreamConstraints> = {
  camera: { video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } } },
  microphone: {
    audio: {
      echoCancellation: { ideal: true },
      noiseSuppression: { ideal: true },
      autoGainControl: { ideal: true },
    },
  },
};

const entries = new Map<MediaKind, StreamEntry>();
/** 并发 `acquire` 去重：同一个 kind 的 getUserMedia 同时只会有一个在飞。 */
const inFlight = new Map<MediaKind, Promise<MediaStream>>();
/**
 * 流级监听器**独立于条目存在** —— 允许调用方在 `acquire` 之前就订阅，
 * 否则"第一次打开设备时就立刻被占用"的事件会丢。
 */
const endedListeners = new Map<MediaKind, Set<EndedListener>>();

/** 把 `getUserMedia` 的 DOMException 翻译成可展示的中文原因。 */
export function classifyMediaError(err: unknown, kind: MediaKind): MediaUnavailableError {
  const name = (err as { name?: string } | null)?.name ?? '';
  const device = kind === 'camera' ? '摄像头' : '麦克风';
  switch (name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return new MediaUnavailableError(kind, 'permission-denied', `${device}不可用（权限被拒绝）`);
    case 'NotFoundError':
    case 'OverconstrainedError':
      return new MediaUnavailableError(kind, 'no-device', `${device}不可用（没有找到设备）`);
    case 'NotReadableError':
    case 'AbortError':
      return new MediaUnavailableError(kind, 'device-busy', `${device}不可用（被其他程序占用）`);
    case 'TypeError':
      return new MediaUnavailableError(kind, 'unsupported', `${device}不可用（浏览器不支持）`);
    default:
      return new MediaUnavailableError(
        kind,
        'unknown',
        `${device}启动失败${name ? `（${name}）` : ''}`,
      );
  }
}

function attachEndedListener(kind: MediaKind, entry: StreamEntry): () => void {
  const handler = (event: Event) => {
    const track = event.target as MediaStreamTrack | null;
    const payload: MediaEndedEvent = {
      kind,
      consumers: [...entry.consumers],
      trackLabel: track?.label ?? '',
    };
    // 设备消失了就注销登记，下一次 acquire 会重新申请。
    // 继续留着一条 ended 的流只会让后续 release 变成空操作，
    // 并让界面一直以为"还在跑"。
    destroyEntry(kind, entry);
    endedListeners.get(kind)?.forEach((listener) => {
      try {
        listener(payload);
      } catch (err) {
        console.error('[媒体管理] onEnded 监听器抛错:', err);
      }
    });
  };

  const tracks = entry.stream.getTracks();
  tracks.forEach((track) => track.addEventListener('ended', handler));
  return () => tracks.forEach((track) => track.removeEventListener('ended', handler));
}

/** 摘监听 + 清引用 + 取消登记（**不** stop 轨道，供 ended 自身路径使用）。 */
function destroyEntry(kind: MediaKind, entry: StreamEntry): void {
  entry.detach();
  entry.consumers.clear();
  if (entries.get(kind) === entry) entries.delete(kind);
}

/** 摘监听 + 清引用 + 取消登记 + stop 轨道。 */
function stopEntry(kind: MediaKind, entry: StreamEntry): void {
  entry.detach();
  entry.consumers.clear();
  entry.stream.getTracks().forEach((track) => track.stop());
  if (entries.get(kind) === entry) entries.delete(kind);
}

function releaseImpl(kind: MediaKind, consumer: MediaConsumer): void {
  const entry = entries.get(kind);
  if (!entry) return;
  entry.consumers.delete(consumer);
  if (entry.consumers.size === 0) stopEntry(kind, entry);
}

/**
 * 借一路媒体流。
 *
 * 幂等语义（这是本模块存在的理由）：
 * - 同一个 `(kind, consumer)` 重复调用 → 返回**同一份**流，不重新打开设备。
 * - 不同 consumer 借同一个 kind → 共用**同一份**流，引用计数 +1。
 * - 并发调用 → 共用在飞的那个 `getUserMedia`，不会打开两次。
 *
 * 失败时抛 `MediaUnavailableError`（带 `reason`），**不吞异常** ——
 * 调用方必须能区分"开了"和"没开成"。
 */
async function acquireImpl(
  kind: MediaKind,
  consumer: MediaConsumer,
  constraints?: MediaStreamConstraints,
): Promise<MediaStream> {
  // 有限次重试：等待在飞请求期间，那一份流可能已被别处释放，
  // 此时不能把未登记的流交出去（那正是要消灭的泄漏模式）。
  for (let attempt = 0; attempt < 3; attempt++) {
    const existing = entries.get(kind);
    if (existing) {
      existing.consumers.add(consumer);
      return existing.stream;
    }

    const pending = inFlight.get(kind);
    if (pending) {
      // 失败由发起方抛出；这里只等它落地，然后回到循环顶部重新判断 entries。
      await pending.catch(() => undefined);
      continue;
    }

    if (!globalThis.navigator?.mediaDevices?.getUserMedia) {
      throw new MediaUnavailableError(
        kind,
        'unsupported',
        kind === 'camera' ? '摄像头不可用（浏览器不支持）' : '麦克风不可用（浏览器不支持）',
      );
    }

    const request = navigator.mediaDevices.getUserMedia(
      constraints ?? DEFAULT_CONSTRAINTS[kind],
    );
    inFlight.set(kind, request);
    try {
      const stream = await request;
      const entry: StreamEntry = {
        stream,
        consumers: new Set([consumer]),
        constraints: constraints ?? DEFAULT_CONSTRAINTS[kind],
        detach: () => {},
      };
      entry.detach = attachEndedListener(kind, entry);
      entries.set(kind, entry);
      return stream;
    } catch (err) {
      throw classifyMediaError(err, kind);
    } finally {
      inFlight.delete(kind);
    }
  }

  throw new MediaUnavailableError(
    kind,
    'unknown',
    kind === 'camera' ? '摄像头启动失败（设备竞争，请重试）' : '麦克风启动失败（设备竞争，请重试）',
  );
}

export const mediaStreamManager = {
  acquire: acquireImpl,

  /**
   * 归还一路媒体流。
   *
   * 必须显式给 `kind` —— 面部检测停止时不应该顺手把麦克风也停掉。
   * 引用计数归零才真的 `stop()` 轨道。
   */
  release: releaseImpl,

  /** 归还该消费者持有的**所有**设备（挂断、组件卸载时用）。 */
  releaseAll(consumer: MediaConsumer): void {
    releaseImpl('camera', consumer);
    releaseImpl('microphone', consumer);
  },

  /** 现在谁在用这个设备。用于调试面板，也用于通话页判断"摄像头已被占用"。 */
  holders(kind: MediaKind): MediaConsumer[] {
    return [...(entries.get(kind)?.consumers ?? [])];
  },

  /** 该设备当前是否有一份活着的流。 */
  has(kind: MediaKind): boolean {
    return entries.has(kind);
  },

  /** 取当前流（**不**增加引用计数），没有则返回 null。 */
  getStream(kind: MediaKind): MediaStream | null {
    return entries.get(kind)?.stream ?? null;
  },

  /**
   * 取一份**只读的轨道视图**，供第二个消费者复用同一台设备而不重复打开。
   * 返回的是原轨道，因此持有方**不应**调用 `stop()` —— 释放请走 `release`。
   */
  borrowTracks(kind: MediaKind, consumer: MediaConsumer): MediaStreamTrack[] {
    const entry = entries.get(kind);
    if (!entry) return [];
    entry.consumers.add(consumer);
    return entry.stream.getTracks();
  },

  /**
   * 订阅流级失败（设备被抢走 / 被拔掉 / 权限被改）。返回取消订阅函数。
   *
   * 与 `acquire` 的顺序无关：可以先订阅再打开设备。
   */
  onEnded(kind: MediaKind, listener: EndedListener): () => void {
    let set = endedListeners.get(kind);
    if (!set) {
      set = new Set();
      endedListeners.set(kind, set);
    }
    set.add(listener);
    return () => {
      set?.delete(listener);
    };
  },

  /** 仅供测试/热重载：强制释放所有设备并清空监听器。 */
  resetForTest(): void {
    entries.forEach((entry, kind) => stopEntry(kind, entry));
    entries.clear();
    inFlight.clear();
    endedListeners.clear();
  },
};

export type MediaStreamManager = typeof mediaStreamManager;
