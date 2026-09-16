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
    const result = await algorithmBridge.getPhenotype(req.params.userId, time_window_days);
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
    const result = await algorithmBridge.getBaselineDeviation(req.params.userId);
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

export default router;
