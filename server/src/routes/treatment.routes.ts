import { Router } from 'express';
import treatmentController from '../controllers/treatment.controller';
import { authenticate } from '../middlewares/auth';

const router = Router();

router.use(authenticate);

router.post('/plans', treatmentController.createPlan);
router.get('/plans', treatmentController.getPlans);
router.get('/plans/:planId', treatmentController.getPlanDetail);
router.put('/plans/:planId/status', treatmentController.updatePlanStatus);
router.post('/plans/:planId/tasks', treatmentController.addTask);
router.put('/tasks/:taskId/status', treatmentController.updateTaskStatus);

export default router;
