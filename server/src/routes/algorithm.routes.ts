/**
 * Algorithm Bridge Routes —— 算法桥接 API 路由
 *
 * 提供前端直接调用算法模块的 REST 端点：
 *   - POST /api/algorithm/smart-chat     智能对话（状态机+审计）
 *   - POST /api/algorithm/assess         多任务风险评估
 *   - GET  /api/algorithm/phenotype/:id  数字表型
 *   - GET  /api/algorithm/phenotype/:id/baseline  同龄对比
 *   - POST /api/algorithm/comorbidity    共病分析
 *   - POST /api/algorithm/escalate       危机升级
 */
import { Router } from 'express';
import { authenticate, AuthRequest } from '../middlewares/auth';
import algorithmBridge from '../services/algorithm-bridge';

const router = Router();

// 所有路由需要认证
router.use(authenticate);

// 智能对话
router.post('/smart-chat', async (req: AuthRequest, res) => {
  try {
    const { message, conversation_history, session_id } = req.body;
    const result = await algorithmBridge.smartChat({
      user_id: req.userId!,
      message,
      conversation_history: conversation_history || [],
      session_id,
    });
    res.json({ code: 0, data: result });
  } catch (error: any) {
    res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
  }
});

// 多任务风险评估
router.post('/assess', async (req: AuthRequest, res) => {
  try {
    const { text } = req.body;
    const result = await algorithmBridge.assessRisk(req.userId!, text);
    if (!result) {
      return res.json({ code: 0, data: null, message: '算法服务不可用' });
    }
    res.json({ code: 0, data: result });
  } catch (error: any) {
    res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
  }
});

// 数字表型
router.get('/phenotype/:userId', async (req: AuthRequest, res) => {
  try {
    const time_window_days = parseInt(req.query.time_window_days as string) || 14;
    const result = await algorithmBridge.getPhenotype(req.params.userId as string, time_window_days);
    if (!result) {
      return res.json({ code: 0, data: null, message: '算法服务不可用' });
    }
    res.json({ code: 0, data: result });
  } catch (error: any) {
    res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
  }
});

// 同龄对比（基线偏离度）
router.get('/phenotype/:userId/baseline', async (req: AuthRequest, res) => {
  try {
    const result = await algorithmBridge.getBaselineDeviation(req.params.userId as string);
    if (!result) {
      return res.json({ code: 0, data: null, message: '算法服务不可用' });
    }
    res.json({ code: 0, data: result });
  } catch (error: any) {
    res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
  }
});

// 同龄对比（别名路由，兼容 /deviation 路径）
router.get('/phenotype/:userId/deviation', async (req: AuthRequest, res) => {
  try {
    const result = await algorithmBridge.getBaselineDeviation(req.params.userId as string);
    if (!result) {
      return res.json({ code: 0, data: null, message: '算法服务不可用' });
    }
    res.json({ code: 0, data: result });
  } catch (error: any) {
    res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
  }
});

// 多任务风险评估（别名路由，兼容 /assessment/predict 路径）
router.post('/assessment/predict', async (req: AuthRequest, res) => {
  try {
    const { text } = req.body;
    const result = await algorithmBridge.assessRisk(req.userId!, text);
    if (!result) {
      return res.json({ code: 0, data: null, message: '算法服务不可用' });
    }
    res.json({ code: 0, data: result });
  } catch (error: any) {
    res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
  }
});

// 共病分析
router.post('/comorbidity', async (req: AuthRequest, res) => {
  try {
    const { depression_prob, anxiety_prob, sleep_prob } = req.body;
    const result = await algorithmBridge.analyzeComorbidity(
      req.userId!,
      depression_prob,
      anxiety_prob,
      sleep_prob,
    );
    if (!result) {
      return res.json({ code: 0, data: null, message: '算法服务不可用' });
    }
    res.json({ code: 0, data: result });
  } catch (error: any) {
    res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
  }
});

// 危机升级
router.post('/escalate', async (req: AuthRequest, res) => {
  try {
    const { risk_score, trigger_evidence } = req.body;
    const result = await algorithmBridge.escalateCrisis({
      user_id: req.userId!,
      risk_score,
      trigger_evidence: trigger_evidence || [],
    });
    if (!result) {
      return res.json({ code: 0, data: null, message: '算法服务不可用' });
    }
    res.json({ code: 0, data: result });
  } catch (error: any) {
    res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
  }
});

// 多模态情绪感知分析
router.post('/perception', async (req: AuthRequest, res) => {
  try {
    const { text } = req.body;
    if (!text) {
      return res.status(400).json({ code: 400, message: '请提供待分析的文本' });
    }
    const result = await algorithmBridge.analyzePerception(text);
    if (!result) {
      return res.json({ code: 0, data: null, message: '算法服务不可用' });
    }
    res.json({ code: 0, data: result });
  } catch (error: any) {
    res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
  }
});

// 多模态感知上传（帧+音频+文本）
router.post('/perception/upload', async (req: AuthRequest, res) => {
  try {
    const { text, frame_base64, audio_base64 } = req.body;
    if (!text) {
      return res.status(400).json({ code: 400, message: '请提供待分析的文本' });
    }
    const result = await algorithmBridge.analyzeMultimodalUpload(text, frame_base64, audio_base64);
    if (!result) {
      return res.json({ code: 0, data: null, message: '算法服务不可用' });
    }
    // 存储到数据库供咨询师查看
    try {
      const prisma = (await import('../config/database')).default;
      const patientProfile = await prisma.patientProfile.findUnique({ where: { userId: req.userId! } });
      if (patientProfile) {
        await prisma.psychologicalProfile.create({
          data: {
            patientProfileId: patientProfile.id,
            anxiety: Math.round((result.text_emotion_probs[2] || 0) * 100),
            depression: Math.round((result.text_emotion_probs[1] || 0) * 100),
            stress: Math.round((result.audio_risk_prob || 0) * 100),
            // ⚠️ 睡眠与社交无法从多模态推理得出，这里不再填 50 这种"看起来像真值"的占位分。
            // 0 = 未采集；睡眠真实数据只来自用户手动记录（SleepRecord）。
            sleepQuality: 0,
            socialActivity: 0,
            emotionalStability: Math.round((1 - (result.text_emotion_probs[2] || 0)) * 100),
            overallScore: Math.round((1 - (result.text_emotion_probs[1] + result.text_emotion_probs[2]) / 2) * 100),
            riskLevel: (result.text_emotion_probs[2] || 0) > 0.5 ? 'MEDIUM' : 'LOW',
            report: result.evidence.join('；'),
            assessedBy: 'AI_MULTIMODAL',
            metadata: JSON.stringify({
              emotion_probs: result.text_emotion_probs,
              audio_risk: result.audio_risk_prob,
              confidence: result.confidence,
              evidence: result.evidence,
            }),
          },
        });
      }
    } catch {}
    res.json({ code: 0, data: result });
  } catch (error: any) {
    res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
  }
});

// ============================================================
// 个人基线同步
// ============================================================

/**
 * POST /api/algorithm/baseline/sync
 * 同步个人基线到算法后端
 *
 * Body: { baseline_data: { heartRate?, breathingRate?, ... } }
 */
router.post('/baseline/sync', async (req: AuthRequest, res) => {
  try {
    const { baseline_data } = req.body;
    if (!baseline_data || typeof baseline_data !== 'object') {
      return res.status(400).json({ code: 400, message: '缺少 baseline_data' });
    }
    const result = await algorithmBridge.syncBaseline(req.userId!, baseline_data);
    if (!result) {
      return res.json({ code: 0, data: null, message: '算法服务不可用' });
    }
    res.json({ code: 0, data: result });
  } catch (error: any) {
    res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
  }
});

/**
 * GET /api/algorithm/baseline/:userId
 * 获取个人基线数据
 */
router.get('/baseline/:userId', async (req: AuthRequest, res) => {
  try {
    const available = await algorithmBridge.isAvailable();
    if (!available) {
      return res.json({ code: 0, data: null, message: '算法服务不可用' });
    }
    const result = await algorithmBridge.getBaseline(req.params.userId as string);
    if (!result) {
      return res.json({ code: 0, data: null, message: '该用户暂无基线数据' });
    }
    res.json({ code: 0, data: result });
  } catch (error: any) {
    res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
  }
});

// ============================================================
// 本地语音识别（sherpa-onnx Paraformer）—— 替代浏览器 Web Speech API
// ============================================================

/**
 * POST /api/algorithm/asr/transcribe
 *
 * 上传一段 16bit PCM WAV（base64），返回本地识别文本。
 *
 * Body: { audio_base64: string, sample_rate?: number }
 *
 * 与 `/perception/upload` 的区别：那条路做的是「情感感知」，本条只做「转文字」。
 * 识别完全在本机完成 —— 音频不离开设备，也不依赖外网；这替代了原先的
 * `webkitSpeechRecognition`（Edge 走微软 Azure、Chrome 走谷歌云）。
 */
router.post('/asr/transcribe', async (req: AuthRequest, res) => {
  try {
    const { audio_base64, sample_rate } = req.body || {};
    if (typeof audio_base64 !== 'string' || audio_base64.length === 0) {
      return res.status(400).json({ code: 400, message: '缺少 audio_base64' });
    }
    const result = await algorithmBridge.transcribeAudio({
      audio_base64,
      sample_rate: typeof sample_rate === 'number' ? sample_rate : 16000,
    });
    if (!result) {
      return res.json({ code: 0, data: null, message: '算法服务不可用' });
    }
    res.json({ code: 0, data: result });
  } catch (error: any) {
    res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
  }
});

export default router;
