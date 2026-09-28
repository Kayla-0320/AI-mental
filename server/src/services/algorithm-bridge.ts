/**
 * Algorithm Bridge Service —— Server ↔ Algorithm 桥接层
 *
 * 将 Node.js 后端与 Python 算法后端 (FastAPI) 连接起来。
 * 所有 AI 咨询请求经过此桥接层调用算法模块：
 *   - 对话状态机 (DialogEngine)
 *   - 安全审计中间件 (AuditedLLMMiddleware)
 *   - 多任务风险评估 (MultiTaskPredictor)
 *   - 危机升级 (Escalation)
 *
 * 当 Python 算法服务不可用时，自动降级到原有 LLM 直调模式。
 */
import { config } from '../config';

// ============================================================
// 类型定义（对应 Python algorithm/shared/dataclasses.py）
// ============================================================

export interface EmotionResult {
  text_emotion_probs: number[];
  audio_risk_prob: number | null;
  confidence: number;
  timestamp: number;
  evidence: string[];
}

export interface RiskAssessment {
  phq9_estimated: [number, number];
  gad7_estimated: [number, number];
  risk_level: 'low' | 'medium' | 'high' | 'crisis';
  confidence: number;
  evidence: { source: string; description: string; weight: number }[];
}

export interface DialogAction {
  action_type: string;
  params: Record<string, unknown>;
  state: string;
  requires_escalation: boolean;
  forbidden_patterns: string[];
}

export interface AuditVerdict {
  passed: boolean;
  results: { axis: string; passed: boolean; reason: string; suggested_action: string }[];
  final_action: string;
  rewritten_content: string;
}

export interface SmartChatRequest {
  user_id: string;
  message: string;
  conversation_history: { role: string; content: string }[];
  session_id?: string;
}

/**
 * C2D2 情景检索结果元信息（对应 Python SceneRetriever.build_context 的产物）。
 * 只暴露"命中几个情境"，不暴露具体情境文本与扭曲标签。
 */
export interface SceneRetrievalMeta {
  scenes: number;
  injected: boolean;
}

/**
 * 认知扭曲 8 分类结果（对应 Python perception.cognitive.distortion_classifier）。
 *
 * ⚠️ 这是**内部审计字段，不是诊断结论**。官方 test acc 0.68–0.69，
 * 逐类 F1 从 0.87（非扭曲）到 0.37（算命）不等。
 * **不得直接呈现给患者**（AGENTS.md 禁止诊断性语言）；仅供咨询师端与审计链路使用。
 */
export interface CognitiveDistortionMeta {
  label: number;
  label_zh: string;
  confidence: number;
  is_distorted: boolean;
  margin: number;
}

export interface SmartChatResponse {
  reply: string;
  dialog_state: string;
  dialogue_mode?: string;  // 对话模式：EMPATHY（共情）/ SOCRATIC（引导）
  action_type: string;
  risk_level: string;
  audit_passed: boolean;
  requires_escalation: boolean;
  emotion_probs: number[];
  evidence: string[];
  fallback: boolean;  // 是否使用了降级模式
  scene_retrieval?: SceneRetrievalMeta | null;
  cognitive_distortion?: CognitiveDistortionMeta | null;
}

// ============================================================
// 文本流式对话（SSE）契约
// ============================================================

/**
 * 流式对话事件名。
 *
 * ⚠️ 这组取值同时出现在三处，改动必须三处同步（见
 * `algorithm/shared/dataclasses.py` 的 `ChatStreamEventType`）：
 *   1. Python 产出端 `algorithm/api/intervention.py`（/smart-chat/stream）
 *   2. 本文件（原样透传，不做语义解释）
 *   3. 浏览器消费端 `client/src/services/index.ts`
 *
 * `delta` 是**追加**语义，`revise` 是**整体替换**语义。两者必须分开：
 * 本项目的安全审计是整段回复级的，无法在吐字之前拿到结论，因此采用
 * 「先说、后校」——审计一旦否掉已生成内容就发 `revise`。把改稿当增量拼上去
 * 会得到「不安全原文 + 安全改稿」，是最坏结果。
 *
 * `speak` 是**合成计划**（定稿全文 + 按句切好的片段），只在第一个 `delta` 之前
 * 出现一次，由语音端消费以保证"听到的 = 看到的"；纯文字端忽略即可。本文件只做
 * 透传，不解释语义。
 */
export type ChatStreamEventName = 'meta' | 'speak' | 'delta' | 'revise' | 'done' | 'error';

/** 算法层推来的一次流式事件 */
export interface ChatStreamChunk {
  event: ChatStreamEventName;
  data: Record<string, unknown>;
}

/** 流式对话的开场元信息（此时还没有任何回复文本） */
export interface ChatStreamMeta {
  risk_level: string;
  dialogue_mode: string;
  emotion_probs: number[];
  /**
   * 本轮是否允许增量上屏。high / crisis 恒为 false：
   * 升级流程必须原子下发，逐字吐出的安全确认与热线会看起来像普通闲聊。
   */
  streaming: boolean;
}

/** 流式对话的终结载荷 —— 字段与非流式 /smart-chat 同构，另加 text / latency_ms */
export interface ChatStreamDone extends Omit<SmartChatResponse, 'reply'> {
  text: string;
  streaming?: boolean;
  latency_ms?: number;
}

/** 本地语音识别（sherpa-onnx Paraformer）请求 */
export interface AsrTranscribeRequest {
  /** 16bit PCM WAV 的 base64 编码 */
  audio_base64: string;
  /** 采样率，非 16000 时由识别器内部重采样 */
  sample_rate?: number;
}

/** 本地语音识别结果 */
export interface AsrTranscribeResponse {
  text: string;
  engine: string;
  duration_ms: number;
  latency_ms: number;
  /** 非空表示本次识别失败（引擎不可用 / 音频无法解析） */
  error: string;
}

// ============================================================
// 配置
// ============================================================

function getAlgorithmBase(): string {
  return process.env.ALGORITHM_API_URL || 'http://127.0.0.1:8001';
}
const REQUEST_TIMEOUT = 15000; // 15 秒超时（算法服务应快速响应）
// ASR 要上传音频（长句可达数百 KB）+ CPU 推理，给更宽的预算
const ASR_TIMEOUT = 60000;

/** 允许下发给浏览器的流式事件白名单（防止上游新增事件被无声透传） */
const CHAT_STREAM_EVENTS: readonly ChatStreamEventName[] = [
  'meta',
  'speak',
  'delta',
  'revise',
  'done',
  'error',
];

/**
 * 解析一个 SSE 帧为 `{ event, data }`。
 *
 * 刻意**不**用 `EventSource`：它只支持 GET，无法带 Authorization 头，
 * 而本接口是 POST + Bearer。因此在 fetch 的 body 流上自己分帧。
 *
 * 容错策略：无法识别的帧一律**跳过并告警**，不抛异常。流式链路上任何一帧
 * 解析失败都不该让整轮对话崩掉 —— 真正的兜底是随后的 `done` 事件或调用方的
 * 非流式回退。
 *
 * @param frame 一个完整 SSE 帧（不含分隔用的空行）
 * @returns 解析结果；该帧无数据、事件名未知或 JSON 非法时返回 `null`
 */
function parseSseFrame(frame: string): ChatStreamChunk | null {
  let eventName = 'message';
  const dataLines: string[] = [];

  for (const line of frame.split('\n')) {
    if (line.startsWith(':')) continue; // 注释 / 心跳
    if (line.startsWith('event:')) {
      eventName = line.slice(6).trim();
    } else if (line.startsWith('data:')) {
      // SSE 规范：data: 后的第一个空格是分隔符，不属于数据
      dataLines.push(line.slice(5).replace(/^ /, ''));
    }
  }

  if (dataLines.length === 0) return null;

  if (!CHAT_STREAM_EVENTS.includes(eventName as ChatStreamEventName)) {
    console.warn('[AlgorithmBridge] 未知的流式事件类型，已忽略:', eventName);
    return null;
  }

  try {
    return {
      event: eventName as ChatStreamEventName,
      data: JSON.parse(dataLines.join('\n')) as Record<string, unknown>,
    };
  } catch (err) {
    console.warn('[AlgorithmBridge] 流式事件载荷 JSON 解析失败，已忽略:', err);
    return null;
  }
}

// ============================================================
// 桥接服务
// ============================================================

class AlgorithmBridgeService {
  private available = false;
  private lastCheckTime = 0;
  private checkInterval = 10000; // 10 秒检查一次

  /**
   * 检查 Python 算法服务是否可用
   */
  async isAvailable(): Promise<boolean> {
    const now = Date.now();
    if (now - this.lastCheckTime < this.checkInterval) {
      return this.available;
    }

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      const response = await fetch(`${getAlgorithmBase()}/health`, {
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      this.available = response.ok;
    } catch {
      // 健康检查失败，但保留上次已知状态（避免闪断）
      // 只在从未成功过时才标记为不可用
      if (!this.available) {
        this.available = false;
      }
    }

    this.lastCheckTime = now;
    return this.available;
  }

  /**
   * 智能对话 —— 经过状态机 + 审计中间件的完整闭环
   *
   * 流程：
   *   1. 文本情感感知 → EmotionResult
   *   2. 多任务风险评估 → RiskAssessment
   *   3. 对话状态机 → DialogAction
   *   4. LLM 生成回复
   *   5. 安全审计 → AuditVerdict
   *   6. 返回安全回复
   *
   * 降级：算法服务不可用时，退回原有 LLM 直调模式
   */
  async smartChat(request: SmartChatRequest): Promise<SmartChatResponse> {
    // 检查算法服务可用性
    const available = await this.isAvailable();

    if (!available) {
      // 降级模式：直接返回标记
      return {
        reply: '',
        dialog_state: 'INIT',
        action_type: 'OPEN_QUESTION',
        risk_level: 'low',
        audit_passed: true,
        requires_escalation: false,
        emotion_probs: [0.2, 0.2, 0.2, 0.2, 0.2],
        evidence: ['算法服务不可用，降级模式'],
        fallback: true,
      };
    }

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);

      const response = await fetch(`${getAlgorithmBase()}/api/v1/intervention/smart-chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`算法服务返回错误: ${response.status}`);
      }

      const data = (await response.json()) as Record<string, unknown>;
      return {
        reply: (data.reply as string) || '',
        dialog_state: (data.dialog_state as string) || 'INIT',
        // 此前漏掉了这一行：接口在第 60 行声明了 dialogue_mode，
        // 但返回映射没有赋值，导致 consultation.service 里
        // `smartResult.dialogue_mode || 'EMPATHY'` 恒为 'EMPATHY'，
        // 前端永远看不到真实的对话模式（SOCRATIC 也被显示成共情）。
        dialogue_mode: (data.dialogue_mode as string) || undefined,
        action_type: (data.action_type as string) || 'OPEN_QUESTION',
        risk_level: (data.risk_level as string) || 'low',
        audit_passed: (data.audit_passed as boolean) ?? true,
        requires_escalation: (data.requires_escalation as boolean) ?? false,
        emotion_probs: (data.emotion_probs as number[]) || [0.2, 0.2, 0.2, 0.2, 0.2],
        evidence: (data.evidence as string[]) || [],
        fallback: false,
        scene_retrieval: (data.scene_retrieval as SceneRetrievalMeta | null) ?? null,
        cognitive_distortion: (data.cognitive_distortion as CognitiveDistortionMeta | null) ?? null,
      };
    } catch (error) {
      console.error('[AlgorithmBridge] smartChat 失败，降级:', error);
      return {
        reply: '',
        dialog_state: 'INIT',
        action_type: 'OPEN_QUESTION',
        risk_level: 'low',
        audit_passed: true,
        requires_escalation: false,
        emotion_probs: [0.2, 0.2, 0.2, 0.2, 0.2],
        evidence: ['算法服务调用失败，降级模式'],
        fallback: true,
      };
    }
  }

  /**
   * 智能对话的**流式**版本 —— 逐句上屏，审计结论仍由算法层给出。
   *
   * 只做传输与解析，**不解释事件语义、不落库**：落库由
   * `consultation.service.sendMessageStream` 在收到 `done` 后完成。
   * 之所以保持「哑管道」，是为了让「什么算安全」这件事有唯一权威源
   * （算法层的 SafetyLoop），Node 侧不参与判断。
   *
   * 与非流式的 `smartChat` 不同，这里**不**先查 `isAvailable()`：
   * 探活是一次额外的往返，而流式路径本身就靠 `fetch` 的失败来降级
   * （调用方捕获异常后退回非流式路径）。先探活只会让首字更慢。
   *
   * @param request 与非流式端点同构的请求体
   * @param signal 用于在浏览器断开时中止上游调用（避免继续烧 token）
   * @yields 解析后的 SSE 事件；顺序保证 `meta → delta* → [revise] → done`
   * @throws 算法服务不可达 / 返回非 2xx / 响应无 body 时抛出，由调用方降级
   */
  async *smartChatStream(
    request: SmartChatRequest,
    signal?: AbortSignal,
  ): AsyncGenerator<ChatStreamChunk> {
    const response = await fetch(`${getAlgorithmBase()}/api/v1/intervention/smart-chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
      body: JSON.stringify(request),
      signal,
    });

    if (!response.ok) {
      throw new Error(`算法流式服务返回错误: ${response.status}`);
    }
    if (!response.body) {
      throw new Error('算法流式服务未返回响应体');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // SSE 以空行分帧。算法层用 "\n\n"，但反向代理可能改写成 "\r\n\r\n"，
        // 因此统一归一化后再切。
        buffer = buffer.replace(/\r\n/g, '\n');
        let sep = buffer.indexOf('\n\n');
        while (sep !== -1) {
          const frame = buffer.slice(0, sep);
          buffer = buffer.slice(sep + 2);
          const parsed = parseSseFrame(frame);
          if (parsed) yield parsed;
          sep = buffer.indexOf('\n\n');
        }
      }
    } finally {
      // 无论正常结束、抛异常还是被 abort，都要释放底层连接。
      reader.cancel().catch(() => undefined);
    }
  }

  /**
   * 多任务风险评估
   */
  async assessRisk(user_id: string, text: string): Promise<RiskAssessment | null> {
    if (!(await this.isAvailable())) return null;

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);

      const response = await fetch(`${getAlgorithmBase()}/api/v1/assessment/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id, text }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) return null;
      return (await response.json()) as RiskAssessment;
    } catch {
      return null;
    }
  }

  /**
   * 数字表型提取
   */
  async getPhenotype(user_id: string, time_window_days = 14): Promise<Record<string, unknown> | null> {
    if (!(await this.isAvailable())) return null;

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);

      const response = await fetch(
        `${getAlgorithmBase()}/api/v1/profile/phenotype?user_id=${user_id}&time_window_days=${time_window_days}`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          signal: controller.signal,
        }
      );

      clearTimeout(timeoutId);

      if (!response.ok) return null;
      return (await response.json()) as Record<string, unknown>;
    } catch {
      return null;
    }
  }

  /**
   * 基线偏离度（同龄对比）
   */
  async getBaselineDeviation(user_id: string): Promise<Record<string, unknown> | null> {
    if (!(await this.isAvailable())) return null;

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);

      const response = await fetch(
        `${getAlgorithmBase()}/api/v1/profile/phenotype/${user_id}/deviation`,
        { signal: controller.signal }
      );

      clearTimeout(timeoutId);

      if (!response.ok) return null;
      return (await response.json()) as Record<string, unknown>;
    } catch {
      return null;
    }
  }

  /**
   * 共病模式分析
   */
  async analyzeComorbidity(
    user_id: string,
    depression_prob: number,
    anxiety_prob: number,
    sleep_prob: number,
  ): Promise<Record<string, unknown> | null> {
    if (!(await this.isAvailable())) return null;

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);

      const response = await fetch(`${getAlgorithmBase()}/api/v1/assessment/comorbidity/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id,
          depression_prob,
          anxiety_prob,
          sleep_prob,
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) return null;
      return (await response.json()) as Record<string, unknown>;
    } catch {
      return null;
    }
  }

  /**
   * 危机升级
   */
  async escalateCrisis(alert: {
    user_id: string;
    risk_score: number;
    trigger_evidence: string[];
  }): Promise<Record<string, unknown> | null> {
    if (!(await this.isAvailable())) return null;

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);

      const response = await fetch(`${getAlgorithmBase()}/api/v1/emergency/escalate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(alert),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) return null;
      return (await response.json()) as Record<string, unknown>;
    } catch {
      return null;
    }
  }

  /**
   * 多模态情绪感知分析（文本 + 可选的行为特征）。
   *
   * ⚠️ **刻意不接收 `facialFeatures`。**
   * 这里原来有一个 `facialFeatures?` 形参，而唯一的调用点传的是写死的
   * `{ brightness: 128, contrast: 50 }` 冒充"从帧提取的面部特征" ——
   * 服务端的情绪映射会据此推出一份**假的"快乐"模态**，进入融合后把焦虑分往下拉。
   *
   * 真实的面部分析在浏览器里做（MediaPipe Face Mesh），结果不走这条路。
   * 服务端拿不到真实人脸特征，就不应该留有这个"可以塞常量进去"的槽位：
   * 删掉形参比留一条注释更能防止它被重新填回假数据。
   */
  async analyzePerception(text: string, behaviorFeatures?: Record<string, unknown>): Promise<EmotionResult | null> {
    if (!(await this.isAvailable())) return null;

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);

      const body: Record<string, unknown> = { text };
      if (behaviorFeatures) body.behavior_features = behaviorFeatures;

      const response = await fetch(`${getAlgorithmBase()}/api/v1/perception/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) return null;
      return (await response.json()) as EmotionResult;
    } catch {
      return null;
    }
  }

  /**
   * 多模态感知上传：接收前端传来的帧(base64)和音频(base64 WAV)，
   * 保存到临时文件后调用 Python 感知 API
   */
  async analyzeMultimodalUpload(
    text: string,
    frameBase64?: string,
    audioBase64?: string,
  ): Promise<EmotionResult | null> {
    if (!(await this.isAvailable())) return null;

    const fs = await import('fs');
    const path = await import('path');
    const os = await import('os');

    const tmpDir = path.join(os.tmpdir(), 'mental-perception');
    if (!fs.existsSync(tmpDir)) fs.mkdirSync(tmpDir, { recursive: true });

    let wavPath: string | undefined;

    // 保存音频文件
    if (audioBase64) {
      const audioBuf = Buffer.from(audioBase64, 'base64');
      wavPath = path.join(tmpDir, `audio_${Date.now()}.wav`);
      fs.writeFileSync(wavPath, audioBuf);
    }

    // ⚠️ 这里**故意不发送** `facial_features`。
    //
    // 改造前这段代码把 `frameBase64` 解码后又丢掉，然后发送一个写死的
    // `{ brightness: 128, contrast: 50 }` 冒充"从帧提取的面部特征"。
    // 服务端的情绪映射会拿这两个常数推出一份**假的"快乐"模态**，
    // 而它会进入融合、把焦虑分往下拉 —— 用一个常量污染风险评估。
    //
    // 真实的面部分析在浏览器里做（MediaPipe Face Mesh），结果不经这条路径；
    // 服务端拿不到真实人脸特征，就不该声称自己拿到了。

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);

      const body: Record<string, unknown> = { text };
      if (wavPath) body.wav_path = wavPath;

      const response = await fetch(`${getAlgorithmBase()}/api/v1/perception/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      // 清理临时文件
      if (wavPath && fs.existsSync(wavPath)) {
        try { fs.unlinkSync(wavPath); } catch {}
      }

      if (!response.ok) return null;
      return (await response.json()) as EmotionResult;
    } catch {
      if (wavPath && fs.existsSync(wavPath)) {
        try { fs.unlinkSync(wavPath); } catch {}
      }
      return null;
    }
  }

  // ============================================================
  // 个人基线同步
  // ============================================================

  /**
   * 同步个人基线到算法后端
   *
   * @param user_id 用户 ID
   * @param baseline_data 基线数据（增量观测值）
   * @returns 同步后的完整基线，失败返回 null
   */
  async syncBaseline(user_id: string, baseline_data: Record<string, unknown>): Promise<Record<string, unknown> | null> {
    if (!(await this.isAvailable())) return null;

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);

      const response = await fetch(`${getAlgorithmBase()}/api/v1/assessment/baseline/sync`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id, baseline_data }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) return null;
      return (await response.json()) as Record<string, unknown>;
    } catch (error) {
      console.error('[AlgorithmBridge] syncBaseline 失败:', error);
      return null;
    }
  }

  /**
   * 获取个人基线偏离度
   *
   * @param user_id 用户 ID
   * @returns 基线数据，失败返回 null
   */
  async getBaseline(user_id: string): Promise<Record<string, unknown> | null> {
    if (!(await this.isAvailable())) return null;

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);

      const response = await fetch(`${getAlgorithmBase()}/api/v1/assessment/baseline/${user_id}`, {
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) return null;
      return (await response.json()) as Record<string, unknown>;
    } catch (error) {
      console.error('[AlgorithmBridge] getBaseline 失败:', error);
      return null;
    }
  }

  // ============================================================
  // 本地语音识别（替代浏览器 Web Speech API）
  // ============================================================

  /**
   * 把一段音频转成文字 —— 转发到本机算法服务的 sherpa-onnx Paraformer。
   *
   * 音频只在「浏览器 → Node → 本机 Python 服务」之间流转，不经过任何外部云服务。
   * 这替代了原先的 `webkitSpeechRecognition`：后者在 Edge 上走微软 Azure、
   * 在 Chrome 上走谷歌云，会把未成年人音频送出设备。
   *
   * 注意：与其它桥接方法不同，这里**不**吞掉错误 —— 引擎不可用时返回带
   * `error` 字段的对象而非 null，让前端能如实告知用户，而不是静默无声。
   *
   * @param payload 音频（base64 WAV）与采样率
   * @returns 识别结果；网络层失败时返回 null
   */
  async transcribeAudio(payload: AsrTranscribeRequest): Promise<AsrTranscribeResponse | null> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), ASR_TIMEOUT);

      const response = await fetch(`${getAlgorithmBase()}/api/v1/asr/transcribe`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          audio_base64: payload.audio_base64,
          sample_rate: payload.sample_rate ?? 16000,
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      // 400/413 是请求本身的问题，把 detail 透传给上层便于定位
      if (!response.ok) {
        let detail = `算法服务返回 ${response.status}`;
        try {
          const body = (await response.json()) as { detail?: string };
          if (body?.detail) detail = body.detail;
        } catch {
          /* 保持默认 detail */
        }
        return { text: '', engine: 'unavailable', duration_ms: 0, latency_ms: 0, error: detail };
      }

      return (await response.json()) as AsrTranscribeResponse;
    } catch (error) {
      console.error('[AlgorithmBridge] transcribeAudio 失败:', error);
      return null;
    }
  }
}

export default new AlgorithmBridgeService();
