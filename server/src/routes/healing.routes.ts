import { Router } from 'express';
import healingController from '../controllers/healing.controller';
import { authenticate } from '../middlewares/auth';

const router = Router();

router.use(authenticate);

router.post('/sessions', healingController.createSession);
router.put('/sessions/:sessionId/complete', healingController.completeSession);
router.get('/sessions', healingController.getSessions);
router.get('/stats', healingController.getStats);

export default router;
