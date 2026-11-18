import { Router, Response, NextFunction } from 'express';
import { authenticate, AuthRequest } from '../middlewares/auth';
import privacyService, { ALLOWED_RETENTION_DAYS, type EraseScope } from '../services/privacy.service';

/**
 * 数据控制中心接口。
 *
 * 全部要求登录，且 userId 一律取 `req.userId`（不接受请求体传入的 userId ——
 * 否则任何人都能导出或删除他人的数据）。
 *
 * 对应前端：`client/src/pages/patient/Settings.tsx` 的「隐私与数据控制」卡片。
 * 此前那张卡片上的开关与按钮**没有任何后端**：开关是组件内的 useState（刷新即回默认），
 * 「导出我的数据」「删除所有数据」两个按钮连 onClick 都没有。
 */
const router = Router();

router.use(authenticate);

/** 授权状态：目录 + 当前状态 */
router.get('/consent', async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const state = await privacyService.getConsentState(req.userId!);
    res.json({ code: 0, data: state });
  } catch (e) { next(e); }
});

/** 修改授权：按用途逐项授权/撤销 */
router.put('/consent', async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const items = req.body?.items;
    if (!Array.isArray(items) || items.length === 0) {
      return res.status(400).json({ code: 400, message: 'items 不能为空' });
    }
    for (const item of items) {
      if (typeof item?.dataType !== 'string' || typeof item?.purpose !== 'string' || typeof item?.granted !== 'boolean') {
        return res.status(400).json({ code: 400, message: 'items 元素需要 { dataType, purpose, granted }' });
      }
    }
    const state = await privacyService.setConsent(req.userId!, items);
    res.json({ code: 0, data: state });
  } catch (e) { next(e); }
});

/** 数据保留期限 */
router.get('/retention', async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const days = await privacyService.getRetention(req.userId!);
    res.json({ code: 0, data: { retentionDays: days, allowed: ALLOWED_RETENTION_DAYS } });
  } catch (e) { next(e); }
});

router.put('/retention', async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const days = Number(req.body?.retentionDays);
    if (!Number.isInteger(days)) {
      return res.status(400).json({ code: 400, message: 'retentionDays 必须是整数' });
    }
    const saved = await privacyService.setRetention(req.userId!, days);
    res.json({ code: 0, data: { retentionDays: saved } });
  } catch (e) { next(e); }
});

/** 数据概览：平台到底保存了什么 */
router.get('/summary', async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const summary = await privacyService.dataSummary(req.userId!);
    res.json({ code: 0, data: summary });
  } catch (e) { next(e); }
});

/**
 * 导出：返回完整 JSON，由前端组装成文件下载。
 * 不在这里直接 `res.attachment` 是因为前端 axios 拦截器统一取 `response.data`，
 * 走 JSON 通道可以复用既有的鉴权与错误处理，不必为下载单独开一条链路。
 */
router.post('/export', async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const payload = await privacyService.buildExport(req.userId!);
    res.json({ code: 0, data: payload });
  } catch (e) { next(e); }
});

/**
 * 删除数据。
 *
 * 必须显式传 `confirm`（与前端二次确认输入一致），避免误触导致不可恢复的删除。
 * `scope`：content = 清空个人内容保留账号；account = 删除整个账号。
 */
const CONFIRM_PHRASE = '删除我的数据';

router.post('/data/erase', async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const scope = (req.body?.scope || 'content') as EraseScope;
    const confirm = req.body?.confirm;

    if (scope !== 'content' && scope !== 'account') {
      return res.status(400).json({ code: 400, message: 'scope 只能是 content 或 account' });
    }
    if (confirm !== CONFIRM_PHRASE) {
      return res.status(400).json({ code: 400, message: `请输入「${CONFIRM_PHRASE}」以确认删除` });
    }

    const result = await privacyService.erase(req.userId!, scope);
    res.json({ code: 0, data: result });
  } catch (e) { next(e); }
});

export default router;
