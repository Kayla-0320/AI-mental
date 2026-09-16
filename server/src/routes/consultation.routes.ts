import { Router } from 'express';
import consultationController from '../controllers/consultation.controller';
import { authenticate } from '../middlewares/auth';

const router = Router();

router.use(authenticate);

router.post('/conversations', consultationController.createConversation);
router.get('/conversations', consultationController.getConversations);
router.get('/conversations/:conversationId/messages', consultationController.getMessages);
router.post('/conversations/:conversationId/messages', consultationController.sendMessage);
router.delete('/conversations/:conversationId', consultationController.deleteConversation);
router.post('/conversations/:conversationId/socratic', consultationController.sendSocraticMessage);

export default router;
