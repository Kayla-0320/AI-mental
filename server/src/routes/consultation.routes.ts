import { Router } from 'express';
import consultationController from '../controllers/consultation.controller';
import { authenticate } from '../middlewares/auth';

const router = Router();

router.use(authenticate);

router.post('/conversations', consultationController.createConversation);
router.get('/conversations', consultationController.getConversations);
// 批量删除（集合级）：必须**排在** /conversations/:conversationId 之前。
// 虽然两个方法的路径字面量不同不会真正冲突，但放在一起读时顺序一目了然，
// 也避免以后有人把 :conversationId 那条改成通配时把批量请求吃掉。
router.delete('/conversations', consultationController.deleteConversations);
router.get('/conversations/:conversationId/messages', consultationController.getMessages);
router.post('/conversations/:conversationId/messages', consultationController.sendMessage);
// 流式版本（SSE）：同一份请求体，响应体是 text/event-stream。
// 与上面的非流式端点共享服务层逻辑，只是回复边生成边下发。
router.post('/conversations/:conversationId/messages/stream', consultationController.sendMessageStream);
router.delete('/conversations/:conversationId', consultationController.deleteConversation);
router.post('/conversations/:conversationId/socratic', consultationController.sendSocraticMessage);

export default router;
