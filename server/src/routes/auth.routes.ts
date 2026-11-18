import { Router } from 'express';
import authController from '../controllers/auth.controller';
import { authenticate, authorize } from '../middlewares/auth';

const router = Router();

router.post('/register', authController.register);
router.post('/login', authController.login);
router.post('/refresh-token', authController.refreshToken);
router.get('/profile', authenticate, authController.getProfile);

// 管理端：启用/禁用用户（AdminUsers 页面的「禁用/启用」按钮）
router.put('/users/:userId/status', authenticate, authorize('ADMIN'), authController.updateUserStatus);

export default router;
