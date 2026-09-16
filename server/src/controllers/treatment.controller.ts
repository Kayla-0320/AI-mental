import { Response } from 'express';
import { AuthRequest } from '../middlewares/auth';
import treatmentService from '../services/treatment.service';

class TreatmentController {
  async createPlan(req: AuthRequest, res: Response) {
    try {
      const { title, description } = req.body;
      const result = await treatmentService.createPlan(req.userId!, title, description);
      res.status(201).json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async getPlans(req: AuthRequest, res: Response) {
    try {
      const result = await treatmentService.getPlans(req.userId!);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async getPlanDetail(req: AuthRequest, res: Response) {
    try {
      const { planId } = req.params;
      const result = await treatmentService.getPlanDetail(planId, req.userId!);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async updatePlanStatus(req: AuthRequest, res: Response) {
    try {
      const { planId } = req.params;
      const { status } = req.body;
      const result = await treatmentService.updatePlanStatus(planId, req.userId!, status);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async addTask(req: AuthRequest, res: Response) {
    try {
      const { planId } = req.params;
      const result = await treatmentService.addTask(planId, req.userId!, req.body);
      res.status(201).json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async updateTaskStatus(req: AuthRequest, res: Response) {
    try {
      const { taskId } = req.params;
      const { status, notes } = req.body;
      const result = await treatmentService.updateTaskStatus(taskId, req.userId!, status, notes);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }
}

export default new TreatmentController();
