import { Router, Response, NextFunction } from 'express';
import { CommunityService } from '../services/community.service';
import { authenticate, AuthRequest } from '../middlewares/auth';

const router = Router();
const communityService = new CommunityService();

// 获取帖子列表
router.get('/posts', async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const category = req.query.category as string;
    const page = parseInt(req.query.page as string) || 1;
    const limit = parseInt(req.query.limit as string) || 10;
    const result = await communityService.getPosts(category, page, limit);
    res.json({ code: 0, data: result });
  } catch (e) { next(e); }
});

// 创建帖子
router.post('/posts', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const post = await communityService.createPost(req.userId!, req.body);
    res.json({ code: 0, data: post });
  } catch (e) { next(e); }
});

// 获取帖子详情
router.get('/posts/:id', async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const post = await communityService.getPost(req.params.id);
    res.json({ code: 0, data: post });
  } catch (e) { next(e); }
});

// 点赞
router.post('/posts/:id/like', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const result = await communityService.toggleLike(req.params.id, req.userId!);
    res.json({ code: 0, data: result });
  } catch (e) { next(e); }
});

// 发表评论
router.post('/posts/:id/comments', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const comment = await communityService.addComment(req.params.id, req.userId!, req.body.content, req.body.isAnonymous);
    res.json({ code: 0, data: comment });
  } catch (e) { next(e); }
});

// 管理员：获取待审核内容
router.get('/pending', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    if (req.userRole! !== 'ADMIN') return res.status(403).json({ code: 403, message: '无权限' });
    const result = await communityService.getPendingContent();
    res.json({ code: 0, data: result });
  } catch (e) { next(e); }
});

// 管理员：审核帖子
router.put('/posts/:id/approve', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    if (req.userRole! !== 'ADMIN') return res.status(403).json({ code: 403, message: '无权限' });
    const post = await communityService.approvePost(req.params.id, req.body.approved);
    res.json({ code: 0, data: post });
  } catch (e) { next(e); }
});

// 管理员：情绪天气仪表盘
router.get('/emotion-dashboard', authenticate, async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    if (req.userRole! !== 'ADMIN') return res.status(403).json({ code: 403, message: '无权限' });
    const userId = req.query.userId as string | undefined;
    const days = parseInt(req.query.days as string) || 7;
    const result = await communityService.getEmotionDashboard(userId, days);
    res.json({ code: 0, data: result });
  } catch (e) { next(e); }
});

export default router;
