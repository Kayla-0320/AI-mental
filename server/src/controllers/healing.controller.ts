import { Response } from 'express';
import { AuthRequest } from '../middlewares/auth';
import healingService from '../services/healing.service';

class HealingController {
  async createSession(req: AuthRequest, res: Response) {
    try {
      const { type, title, duration } = req.body;
      const result = await healingService.createSession(req.userId!, type, title, duration);
      res.status(201).json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async completeSession(req: AuthRequest, res: Response) {
    try {
      const { sessionId } = req.params;
      const result = await healingService.completeSession(sessionId, req.userId!, req.body);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async getSessions(req: AuthRequest, res: Response) {
    try {
      const type = req.query.type as string;
      const page = parseInt(req.query.page as string) || 1;
      const pageSize = parseInt(req.query.pageSize as string) || 20;
      const result = await healingService.getSessions(req.userId!, type, page, pageSize);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async getStats(req: AuthRequest, res: Response) {
    try {
      const result = await healingService.getStats(req.userId!);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }
}

export default new HealingController();
