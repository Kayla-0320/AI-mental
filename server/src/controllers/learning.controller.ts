import { Response } from 'express';
import { AuthRequest } from '../middlewares/auth';
import learningService from '../services/learning.service';

class LearningController {
  async getContents(req: AuthRequest, res: Response) {
    try {
      const filters = {
        category: req.query.category as string,
        type: req.query.type as string,
        level: req.query.level as string,
        page: parseInt(req.query.page as string) || 1,
        pageSize: parseInt(req.query.pageSize as string) || 12,
      };
      const result = await learningService.getContents(filters);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async getContentDetail(req: AuthRequest, res: Response) {
    try {
      const result = await learningService.getContentDetail(req.params.contentId as string);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async getUserRecords(req: AuthRequest, res: Response) {
    try {
      const result = await learningService.getUserRecords(req.userId!);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async updateProgress(req: AuthRequest, res: Response) {
    try {
      const { contentId } = req.params;
      const { progress, notes } = req.body;
      const result = await learningService.updateProgress(req.userId!, contentId as string, progress, notes);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async getRecommendations(req: AuthRequest, res: Response) {
    try {
      const result = await learningService.getRecommendations(req.userId!);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async getStats(req: AuthRequest, res: Response) {
    try {
      const result = await learningService.getStats(req.userId!);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }
}

export default new LearningController();
