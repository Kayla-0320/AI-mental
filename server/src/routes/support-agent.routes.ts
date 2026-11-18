import { Router, Response, NextFunction } from 'express';
import { authenticate, authorize, AuthRequest } from '../middlewares/auth';
import supportAgentService, { SupportAgentService } from '../services/support-agent.service';

/** 为本人持续陪伴开放严格鉴权的状态与命令接口；不接受请求体中的用户 ID。 */
export function createSupportAgentRouter(service: SupportAgentService = supportAgentService): Router {
  const router = Router();
  router.use(authenticate, authorize('PATIENT'));
  router.get('/', async (req: AuthRequest, res: Response, next: NextFunction): Promise<void> => {
    try { res.json({ code: 0, data: await service.overview(req.userId!) }); } catch (error) { next(error); }
  });
  router.post('/commands', async (req: AuthRequest, res: Response, next: NextFunction): Promise<void> => {
    try { res.json({ code: 0, data: await service.command(req.userId!, req.body) }); } catch (error) { next(error); }
  });
  router.post('/model-reference', async (req: AuthRequest, res: Response, next: NextFunction): Promise<void> => {
    try {
      if (!req.body || typeof req.body !== 'object' || Array.isArray(req.body) || Object.keys(req.body).length) {
        res.status(400).json({ code: 400, message: '模型参考只读取本人已确认且来源有效的片段' }); return;
      }
      res.json({ code: 0, data: await service.refreshModelReference(req.userId!) });
    } catch (error) { next(error); }
  });
  return router;
}

export default createSupportAgentRouter();
