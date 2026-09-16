import { Router } from 'express';
import learningController from '../controllers/learning.controller';
import { authenticate } from '../middlewares/auth';

const router = Router();

router.use(authenticate);

router.get('/contents', learningController.getContents);
router.get('/contents/:contentId', learningController.getContentDetail);
router.get('/records', learningController.getUserRecords);
router.put('/contents/:contentId/progress', learningController.updateProgress);
router.get('/recommendations', learningController.getRecommendations);
router.get('/stats', learningController.getStats);

export default router;
