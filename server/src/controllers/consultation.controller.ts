import { Response } from 'express';
import { AuthRequest } from '../middlewares/auth';
import consultationService from '../services/consultation.service';

class ConsultationController {
  async createConversation(req: AuthRequest, res: Response) {
    try {
      const result = await consultationService.createConversation(req.userId!, req.body.title);
      res.status(201).json({ code: 0, message: '创建成功', data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async getConversations(req: AuthRequest, res: Response) {
    try {
      const page = parseInt(req.query.page as string) || 1;
      const pageSize = parseInt(req.query.pageSize as string) || 20;
      const result = await consultationService.getConversations(req.userId!, page, pageSize);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async getMessages(req: AuthRequest, res: Response) {
    try {
      const { conversationId } = req.params;
      const page = parseInt(req.query.page as string) || 1;
      const pageSize = parseInt(req.query.pageSize as string) || 50;
      const result = await consultationService.getConversationMessages(conversationId, req.userId!, page, pageSize);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async sendMessage(req: AuthRequest, res: Response) {
    try {
      const { conversationId } = req.params;
      const { content, contentType } = req.body;
      const result = await consultationService.sendMessage(conversationId, req.userId!, content, contentType);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async deleteConversation(req: AuthRequest, res: Response) {
    try {
      const { conversationId } = req.params;
      await consultationService.deleteConversation(conversationId, req.userId!);
      res.json({ code: 0, message: '删除成功' });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async sendSocraticMessage(req: AuthRequest, res: Response) {
    try {
      const { conversationId } = req.params;
      const { content } = req.body;
      const result = await consultationService.sendSocraticMessage(conversationId, req.userId!, content);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }
}

export default new ConsultationController();
