import { Response } from 'express';
import { AuthRequest } from '../middlewares/auth';
import expertService from '../services/expert.service';

class ExpertController {
  async getConsultants(req: AuthRequest, res: Response) {
    try {
      const filters = {
        specialty: req.query.specialty as string,
        minRating: req.query.minRating ? parseFloat(req.query.minRating as string) : undefined,
        page: parseInt(req.query.page as string) || 1,
        pageSize: parseInt(req.query.pageSize as string) || 10,
      };
      const result = await expertService.getConsultants(filters);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async getConsultantDetail(req: AuthRequest, res: Response) {
    try {
      const result = await expertService.getConsultantDetail(req.params.consultantId);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async createBooking(req: AuthRequest, res: Response) {
    try {
      const { consultantId, scheduledAt, notes, type } = req.body;
      const result = await expertService.createBooking(req.userId!, consultantId, new Date(scheduledAt), notes, type);
      res.status(201).json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async getBookings(req: AuthRequest, res: Response) {
    try {
      const result = await expertService.getBookings(req.userId!, req.userRole!);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async getBookingById(req: AuthRequest, res: Response) {
    try {
      const result = await expertService.getBookingById(req.params.bookingId);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async updateBookingStatus(req: AuthRequest, res: Response) {
    try {
      const { bookingId } = req.params;
      const { status } = req.body;
      const result = await expertService.updateBookingStatus(bookingId, status, req.userId!);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async submitFeedback(req: AuthRequest, res: Response) {
    try {
      const { bookingId } = req.params;
      const { rating, feedback } = req.body;
      const result = await expertService.submitFeedback(bookingId, req.userId!, rating, feedback);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async sendMessage(req: AuthRequest, res: Response) {
    try {
      const { bookingId } = req.params;
      const { content, contentType } = req.body;
      const result = await expertService.sendMessage(bookingId, req.userId!, content, contentType);
      res.status(201).json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async getMessages(req: AuthRequest, res: Response) {
    try {
      const { bookingId } = req.params;
      const page = parseInt(req.query.page as string) || 1;
      const result = await expertService.getMessages(bookingId, req.userId!, page);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }
}

export default new ExpertController();
