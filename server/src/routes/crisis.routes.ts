import { Router, Response, NextFunction } from 'express';
import { CrisisService } from '../services/crisis.service';
import { authenticate, AuthRequest } from '../middlewares/auth';

const router = Router();
const crisisService = new CrisisService();

// 获取心理援助热线
router.get('/hotlines', (req, res) => {
  res.json({ code: 0, data: crisisService.getHotlines() });
});

// 检测文本危机（用于 AI 对话中）
router.post('/detect', authenticate, (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const { text } = req.body;
    const result = crisisService.detectCrisis(text);
    res.json({ code: 0, data: result });
  } catch (e) { next(e); }
});

// 获取紧急联系人
router.get('/contacts', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const contacts = await crisisService.getEmergencyContacts(req.userId!);
    res.json({ code: 0, data: contacts });
  } catch (e) { next(e); }
});

// 添加紧急联系人
router.post('/contacts', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const contact = await crisisService.addContact(req.userId!, req.body);
    res.json({ code: 0, data: contact });
  } catch (e) { next(e); }
});

// 更新紧急联系人
router.put('/contacts/:id', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const contact = await crisisService.updateContact(req.params.id, req.userId!, req.body);
    res.json({ code: 0, data: contact });
  } catch (e) { next(e); }
});

// 删除紧急联系人
router.delete('/contacts/:id', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    await crisisService.deleteContact(req.params.id, req.userId!);
    res.json({ code: 0, message: '删除成功' });
  } catch (e) { next(e); }
});

export default router;
