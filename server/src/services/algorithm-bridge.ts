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

export interface SmartChatResponse {
  reply: string;
  dialog_state: string;
  action_type: string;
  risk_level: string;
  audit_passed: boolean;
  requires_escalation: boolean;
  emotion_probs: number[];
  evidence: string[];
  fallback: boolean;  // 是否使用了降级模式
}

// ============================================================
// 配置
// ============================================================

const ALGORITHM_API_BASE = process.env.ALGORITHM_API_URL || 'http://localhost:8000';
const REQUEST_TIMEOUT = 15000; // 15 秒超时（算法服务应快速响应）

// ============================================================
// 桥接服务
// ============================================================

class AlgorithmBridgeService {
  private available = false;
  private lastCheckTime = 0;
  private checkInterval = 30000; // 30 秒检查一次

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
      const response = await fetch(`${ALGORITHM_API_BASE}/health`, {
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      this.available = response.ok;
    } catch {
      this.available = false;
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

      const response = await fetch(`${ALGORITHM_API_BASE}/api/smart-chat`, {
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
        action_type: (data.action_type as string) || 'OPEN_QUESTION',
        risk_level: (data.risk_level as string) || 'low',
        audit_passed: (data.audit_passed as boolean) ?? true,
        requires_escalation: (data.requires_escalation as boolean) ?? false,
        emotion_probs: (data.emotion_probs as number[]) || [0.2, 0.2, 0.2, 0.2, 0.2],
        evidence: (data.evidence as string[]) || [],
        fallback: false,
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
   * 多任务风险评估
   */
  async assessRisk(user_id: string, text: string): Promise<RiskAssessment | null> {
    if (!(await this.isAvailable())) return null;

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);

      const response = await fetch(`${ALGORITHM_API_BASE}/api/assessment/predict`, {
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
        `${ALGORITHM_API_BASE}/api/phenotype/${user_id}?time_window_days=${time_window_days}`,
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
   * 基线偏离度（同龄对比）
   */
  async getBaselineDeviation(user_id: string): Promise<Record<string, unknown> | null> {
    if (!(await this.isAvailable())) return null;

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);

      const response = await fetch(
        `${ALGORITHM_API_BASE}/api/phenotype/${user_id}/baseline-deviation`,
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

      const response = await fetch(`${ALGORITHM_API_BASE}/api/comorbidity/analyze`, {
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

      const response = await fetch(`${ALGORITHM_API_BASE}/emergency/escalate`, {
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
   * 多模态情绪感知分析（文本 + 可选的面部/行为/语音）
   */
  async analyzePerception(text: string, facialFeatures?: Record<string, unknown>, behaviorFeatures?: Record<string, unknown>): Promise<EmotionResult | null> {
    if (!(await this.isAvailable())) return null;

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);

      const body: Record<string, unknown> = { text };
      if (facialFeatures) body.facial_features = facialFeatures;
      if (behaviorFeatures) body.behavior_features = behaviorFeatures;

      const response = await fetch(`${ALGORITHM_API_BASE}/api/v1/perception/analyze`, {
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
    let facialFeatures: Record<string, unknown> | undefined;

    // 保存音频文件
    if (audioBase64) {
      const audioBuf = Buffer.from(audioBase64, 'base64');
      wavPath = path.join(tmpDir, `audio_${Date.now()}.wav`);
      fs.writeFileSync(wavPath, audioBuf);
    }

    // 从帧提取面部特征（简化版：使用图像亮度/对比度作为代理特征）
    if (frameBase64) {
      try {
        const imgBuf = Buffer.from(frameBase64, 'base64');
        // 简单面部特征代理：计算图像平均亮度和对比度
        // 真实场景应使用 face-api.js 或 OpenCV
        facialFeatures = {
          brightness: 128, // 默认值，实际应从图像计算
          contrast: 50,
        };
      } catch {
        // 帧处理失败，忽略
      }
    }

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);

      const body: Record<string, unknown> = { text };
      if (wavPath) body.wav_path = wavPath;
      if (facialFeatures) body.facial_features = facialFeatures;

      const response = await fetch(`${ALGORITHM_API_BASE}/api/v1/perception/analyze`, {
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
}

export default new AlgorithmBridgeService();
