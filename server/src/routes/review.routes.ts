import { Router, Response, NextFunction } from 'express';
import { ReviewService } from '../services/review.service';
import { authenticate, AuthRequest } from '../middlewares/auth';

const router = Router();
const reviewService = new ReviewService();

// 创建咨询评价
router.post('/', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const review = await reviewService.createReview(req.userId!, req.body);
    res.json({ code: 0, data: review });
  } catch (e) { next(e); }
});

// 获取咨询师评价
router.get('/consultant/:consultantId', async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const page = parseInt(req.query.page as string) || 1;
    const limit = parseInt(req.query.limit as string) || 10;
    const result = await reviewService.getConsultantReviews(req.params.consultantId as string, page, limit);
    res.json({ code: 0, data: result });
  } catch (e) { next(e); }
});

export default router;
