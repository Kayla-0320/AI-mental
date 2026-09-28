/**
 * 文本流式对话（SSE）客户端
 *
 * 为什么不直接用 `EventSource`：
 *   `EventSource` 只支持 GET，既无法携带 `Authorization` 头，也无法传请求体，
 *   而本接口是 POST + Bearer。因此这里用 `fetch` 拿到 `response.body`
 *   （一个 `ReadableStream`），再手工按 SSE 规范分帧。
 *
 * 事件语义（与 `algorithm/shared/dataclasses.py` 的 `ChatStreamEventType`、
 * `server/src/services/algorithm-bridge.ts` 三处同步）：
 *
 *   user   → 服务端落库后的用户消息（替换本地乐观占位气泡）
 *   meta   → 风险等级 / 对话模式 / 情绪概率；**可能不出现**（见下）
 *   delta  → 追加到当前回复气泡
 *   revise → **整体替换**当前回复气泡（安全审计改稿，或走了降级路径）
 *   done   → 本轮结束，`payload` 与非流式接口的 data 同构，另有 text
 *   error  → 失败，`payload.failed` 为真时按「可重试失败态」渲染
 *
 * ⚠️ `delta` 与 `revise` 必须分开处理。本项目的安全审计是整段回复级的，无法在
 * 吐字之前拿到结论，所以采用「先说、后校」：审计一旦否掉已生成内容就发 `revise`。
 * 把改稿当增量拼上去会得到「不安全原文 + 安全改稿」，是最坏结果。
 *
 * ⚠️ `meta` **不保证出现**：算法服务不可用时服务端会回退到非流式路径，此时只有
 * `user` → `revise` → `done`。调用方不得依赖 `meta` 来创建气泡。
 */

export type ChatStreamEventName = 'user' | 'meta' | 'speak' | 'delta' | 'revise' | 'done' | 'error';

/** `meta` 事件的载荷 */
export interface ChatStreamMetaPayload {
  risk_level: string;
  dialogue_mode: string;
  emotion_probs: number[];
  /** 本轮是否允许增量上屏；high / crisis 恒为 false */
  streaming: boolean;
}

/**
 * `speak` 事件的载荷 —— **定稿文本的合成计划**，在第一个 `delta` 之前下发。
 *
 * 为什么需要它：本项目的定稿规则（一轮只问一个、去套话、压长度）必须在
 * **音频合成之前**生效。否则会出现"规则后置"的听感事故 —— 实测（2026-09-27）
 * 一轮里模型先说 62 字（两句 + 一个追问），前几段已经送去合成了，定稿才把
 * 尾部整句按"只能问一个"删掉，用户听到的是**一句话念到一半被硬掐断**。
 *
 * 消费端按 `segments` 合成、也按 `segments` 上屏，"听到的"="看到的"。
 * ⚠️ 与服务端约定：`segments.join('')` 必须等于 `text`；不满足时消费端应
 * **忽略这个事件**、退回按 delta 自行切分（`chatStream.ts` 会先校验再回调）。
 * 纯文字端可以完全不处理它。
 */
export interface ChatStreamSpeakPayload {
  /** 定稿后的回复全文 */
  text: string;
  /** 按句切好的送合成单位（顺序即出声顺序） */
  segments: string[];
}

/** `error` 事件的载荷 */
export interface ChatStreamErrorPayload {
  failed?: boolean;
  errorCode?: string;
  errorMessage?: string;
}

/** 回调集合；全部可选，未提供的分支静默忽略 */
export interface ChatStreamCallbacks {
  /** 服务端落库后的用户消息（含真实 id） */
  onUser?: (userMessage: Record<string, unknown>) => void;
  onMeta?: (meta: ChatStreamMetaPayload) => void;
  /**
   * 定稿文本的合成计划（见 {@link ChatStreamSpeakPayload}）。
   *
   * ⚠️ 只在服务端自检 `segments.join('') === text` 通过时才会触发；不一致时
   * 这里会**静默跳过**并留一条 console 线索，调用方应保持"按 delta 自行切分"
   * 的既有行为作为兜底。
   */
  onSpeak?: (plan: ChatStreamSpeakPayload) => void;
  /** 增量文本，调用方应**追加** */

  onDelta?: (text: string) => void;
  /** 权威改稿，调用方应**整体替换** */
  onRevise?: (text: string) => void;
  /** 本轮最终载荷（含 aiMessage / isCrisis / riskLevel 等） */
  onDone?: (payload: Record<string, unknown>) => void;
  onError?: (payload: ChatStreamErrorPayload) => void;
}

/** `sendChatMessageStream` 抛出的错误：带上 HTTP 状态供调用方决定是否回退 */
export class ChatStreamTransportError extends Error {
  constructor(
    message: string,
    readonly status: number,
    /** 服务端是否已经确认收到并落库了用户消息 */
    readonly userPersisted: boolean,
  ) {
    super(message);
    this.name = 'ChatStreamTransportError';
  }
}

/** 把一次请求抽成可取消的句柄，便于组件卸载时中止 */
export interface ChatStreamHandle {
  /** 中止本次流式请求（不抛异常，回调不再触发） */
  abort: () => void;
  /** 请求结束（无论成功失败）后 resolve */
  completed: Promise<void>;
}

/**
 * 解析一个 SSE 帧为 `[事件名, 载荷]`。
 *
 * 无法识别的帧返回 `null` 由调用方跳过：流式链路上单帧解析失败不该让整轮对话崩掉，
 * 真正的兜底是随后的 `done` 或走非流式回退。
 */
function parseSseFrame(frame: string): [ChatStreamEventName, any] | null {
  let eventName = 'message';
  const dataLines: string[] = [];

  for (const line of frame.split('\n')) {
    if (line.startsWith(':')) continue; // 注释 / 心跳
    if (line.startsWith('event:')) {
      eventName = line.slice(6).trim();
    } else if (line.startsWith('data:')) {
      // SSE 规范：`data:` 后的第一个空格是分隔符，不属于数据
      dataLines.push(line.slice(5).replace(/^ /, ''));
    }
  }

  if (dataLines.length === 0) return null;

  try {
    return [eventName as ChatStreamEventName, JSON.parse(dataLines.join('\n'))];
  } catch {
    return null;
  }
}

/**
 * 发送一条消息并**流式**接收 AI 回复。
 *
 * 与 `consultationApi.sendMessage` 走同一份服务端逻辑，区别只是回复边生成边下发。
 * 本函数**不抛业务错误**：失败通过 `onError` 回调 + 返回值表达，只有传输层
 * 失败（非 2xx / 网络异常 / 无响应体）才抛 `ChatStreamTransportError`，
 * 让调用方决定是否回退到非流式接口。
 *
 * @param conversationId 会话 ID
 * @param content 用户消息文本
 * @param callbacks 事件回调
 * @param contentType 消息类型，默认 `text`
 * @returns 可取消句柄；`completed` 在请求彻底结束时 resolve
 */
export function sendChatMessageStream(
  conversationId: string,
  content: string,
  callbacks: ChatStreamCallbacks = {},
  contentType = 'text',
): ChatStreamHandle {
  const controller = new AbortController();
  /** 服务端是否已回传落库后的用户消息 —— 决定了失败时能不能安全重发 */
  let userPersisted = false;

  const completed = (async () => {
    let response: Response;
    try {
      const token = localStorage.getItem('accessToken');
      response = await fetch(
        `/api/consultation/conversations/${conversationId}/messages/stream`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'text/event-stream',
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
          },
          body: JSON.stringify({ content, contentType }),
          signal: controller.signal,
        },
      );
    } catch (err) {
      if (controller.signal.aborted) return;
      throw new ChatStreamTransportError(
        err instanceof Error ? err.message : '网络异常',
        0,
        userPersisted,
      );
    }

    if (!response.ok) {
      throw new ChatStreamTransportError(
        `流式接口返回 ${response.status}`,
        response.status,
        userPersisted,
      );
    }
    if (!response.body) {
      throw new ChatStreamTransportError('响应没有可读流', response.status, userPersisted);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    /** 是否已经收到终结事件；用于识别「流被截断」并告知调用方 */
    let settled = false;

    const dispatch = (eventName: ChatStreamEventName, payload: any) => {
      switch (eventName) {
        case 'user':
          userPersisted = true;
          callbacks.onUser?.(payload.userMessage ?? payload);
          break;
        case 'meta':
          callbacks.onMeta?.(payload as ChatStreamMetaPayload);
          break;
        case 'speak': {
          // 契约自检：`segments` 必须正好拼出 `text`。对不上就当作没收到这个事件 ——
          // 宁可退回"按 delta 自行切分"，也不能按一份与全文不符的计划合成/上屏。
          const speak = payload as Partial<ChatStreamSpeakPayload>;
          const segs = Array.isArray(speak.segments) ? speak.segments.map(String) : [];
          const full = String(speak.text ?? '');
          if (!segs.length || segs.join('') !== full) {
            console.warn('[chatStream] speak 计划与全文不一致，已忽略:', speak);
            break;
          }
          callbacks.onSpeak?.({ text: full, segments: segs });
          break;
        }
        case 'delta':
          callbacks.onDelta?.(String(payload.text ?? ''));
          break;
        case 'revise':
          callbacks.onRevise?.(String(payload.text ?? ''));
          break;
        case 'done':
          settled = true;
          callbacks.onDone?.(payload);
          break;
        case 'error':
          settled = true;
          callbacks.onError?.(payload as ChatStreamErrorPayload);
          break;
        default:
          // 未知事件：算法层新增了事件名但这里还没跟上。忽略而不是崩掉，
          // 并在控制台留下线索。
          console.warn('[chatStream] 未知事件类型，已忽略:', eventName, payload);
      }
    };

    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // 算法层用 "\n\n" 分帧，反向代理可能改写成 "\r\n\r\n"，统一归一化
        buffer = buffer.replace(/\r\n/g, '\n');
        let sep = buffer.indexOf('\n\n');
        while (sep !== -1) {
          const frame = buffer.slice(0, sep);
          buffer = buffer.slice(sep + 2);
          const parsed = parseSseFrame(frame);
          if (parsed) dispatch(parsed[0], parsed[1]);
          sep = buffer.indexOf('\n\n');
        }
      }

      // 连接结束但没收到 done/error：说明流被中途掐断（网关超时、服务重启）。
      // 如实告知调用方，让它走非流式回退拿一份完整回复，而不是留半句话在屏幕上。
      if (!settled && !controller.signal.aborted) {
        callbacks.onError?.({
          failed: true,
          errorCode: 'STREAM_TRUNCATED',
          errorMessage: '回复生成中断，请重试',
        });
      }
    } finally {
      reader.cancel().catch(() => undefined);
    }
  })();

  return {
    abort: () => controller.abort(),
    completed,
  };
}
