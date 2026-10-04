import prisma from '../config/database';
import { AppError } from '../middlewares/errorHandler';
import contentModerationService, { ModerationResult } from './content-moderation.service';

export class CommunityService {
  // 创建帖子（集成温柔守护者审核）
  async createPost(userId: string, data: { title: string; content: string; category: string; tags?: string[]; isAnonymous?: boolean }) {
    // 先对标题+内容做审核
    const fullContent = `${data.title}\n${data.content}`;
    const moderation = await contentModerationService.moderate(fullContent, userId);

    // 根据审核等级决定帖子状态
    const isApproved = moderation.level === 'warm_publish';
    const post = await prisma.communityPost.create({
      data: {
        userId,
        title: data.title,
        content: data.content,
        category: data.category,
        tags: data.tags ? JSON.stringify(data.tags) : null,
        isAnonymous: data.isAnonymous ?? true,
        isApproved,
        moderationStatus: moderation.level,
        moderationReason: moderation.reasons.length > 0 ? moderation.reasons.join('；') : null,
        emotionTag: moderation.emotionTag,
        emotionScore: moderation.emotionScore,
        crisisDetected: moderation.crisisDetected,
      },
    });

    // 危机守护后续处理
    let empathyPostId: string | undefined;
    if (moderation.level === 'crisis_guard') {
      const result = await contentModerationService.handleCrisisGuard(userId, fullContent, moderation);
      empathyPostId = result.empathyPostId;
      if (empathyPostId) {
        await prisma.communityPost.update({
          where: { id: post.id },
          data: { empathyPostId },
        });
      }
    }

    return {
      post,
      moderation: {
        level: moderation.level,
        userMessage: moderation.userMessage,
        crisisHotline: moderation.crisisHotline,
        empathyPostId,
      },
    };
  }

  // 获取帖子列表（只展示已审核通过的）
  async getPosts(category?: string, page: number = 1, limit: number = 10) {
    const where: any = { isApproved: true };
    if (category) where.category = category;

    const [posts, total] = await Promise.all([
      prisma.communityPost.findMany({
        where,
        include: {
          user: { select: { id: true, nickname: true, avatar: true } },
          _count: { select: { comments: true, postLikes: true } },
        },
        orderBy: { createdAt: 'desc' },
        skip: (page - 1) * limit,
        take: limit,
      }),
      prisma.communityPost.count({ where }),
    ]);

    return {
      posts: posts.map((p: any) => ({
        ...p,
        commentCount: p._count.comments,
        likeCount: p._count.postLikes,
      })),
      total,
      page,
      limit,
    };
  }

  // 获取帖子详情
  async getPost(postId: string) {
    const post = await prisma.communityPost.findUnique({
      where: { id: postId },
      include: {
        user: { select: { id: true, nickname: true, avatar: true } },
        comments: {
          where: { isApproved: true },
          include: { user: { select: { id: true, nickname: true, avatar: true } } },
          orderBy: { createdAt: 'asc' },
        },
        _count: { select: { comments: true, postLikes: true } },
      },
    });
    if (!post) throw new AppError('帖子不存在', 404);

    // 增加浏览量
    await prisma.communityPost.update({
      where: { id: postId },
      data: { viewCount: { increment: 1 } },
    });

    return {
      ...post,
      commentCount: post._count.comments,
      likeCount: post._count.postLikes,
    };
  }

  // 点赞/取消点赞
  async toggleLike(postId: string, userId: string) {
    const existing = await prisma.communityLike.findUnique({
      where: { postId_userId: { postId, userId } },
    });

    if (existing) {
      await prisma.communityLike.delete({ where: { postId_userId: { postId, userId } } });
      await prisma.communityPost.update({ where: { id: postId }, data: { likeCount: { decrement: 1 } } });
      return { liked: false };
    } else {
      await prisma.communityLike.create({ data: { postId, userId } });
      await prisma.communityPost.update({ where: { id: postId }, data: { likeCount: { increment: 1 } } });
      return { liked: true };
    }
  }

  // 发表评论（集成温柔守护者审核）
  async addComment(postId: string, userId: string, content: string, isAnonymous: boolean = true) {
    const post = await prisma.communityPost.findUnique({ where: { id: postId } });
    if (!post) throw new AppError('帖子不存在', 404);

    // 对评论内容做审核
    const moderation = await contentModerationService.moderate(content, userId);

    const isApproved = moderation.level === 'warm_publish';
    const comment = await prisma.communityComment.create({
      data: {
        postId,
        userId,
        content,
        isAnonymous,
        isApproved,
        moderationStatus: moderation.level,
        moderationReason: moderation.reasons.length > 0 ? moderation.reasons.join('；') : null,
        emotionTag: moderation.emotionTag,
        emotionScore: moderation.emotionScore,
        crisisDetected: moderation.crisisDetected,
      },
    });

    await prisma.communityPost.update({
      where: { id: postId },
      data: { commentCount: { increment: 1 } },
    });

    // 危机守护：评论中的危机信号也触发通知
    if (moderation.level === 'crisis_guard') {
      await contentModerationService.handleCrisisGuard(userId, content, moderation);
    }

    return {
      comment,
      moderation: {
        level: moderation.level,
        userMessage: moderation.userMessage,
        crisisHotline: moderation.crisisHotline,
      },
    };
  }

  // 审核帖子（管理员）
  async approvePost(postId: string, approved: boolean) {
    return prisma.communityPost.update({
      where: { id: postId },
      data: { isApproved: approved },
    });
  }

  // 审核评论（管理员）
  async approveComment(commentId: string, approved: boolean) {
    return prisma.communityComment.update({
      where: { id: commentId },
      data: { isApproved: approved },
    });
  }

  // 获取待审核内容（管理员）—— 包含情绪天气信息
  async getPendingContent() {
    const [posts, comments] = await Promise.all([
      prisma.communityPost.findMany({
        where: { isApproved: false },
        include: { user: { select: { id: true, nickname: true } } },
        orderBy: { createdAt: 'desc' },
      }),
      prisma.communityComment.findMany({
        where: { isApproved: false },
        include: {
          user: { select: { id: true, nickname: true } },
          post: { select: { id: true, title: true } },
        },
        orderBy: { createdAt: 'desc' },
      }),
    ]);
    return { posts, comments };
  }

  // 情绪天气仪表盘（管理员）—— 追踪用户情绪趋势
  async getEmotionDashboard(userId?: string, days: number = 7) {
    const since = new Date();
    since.setDate(since.getDate() - days);

    const where: any = { createdAt: { gte: since } };
    if (userId) where.userId = userId;

    const posts = await prisma.communityPost.findMany({
      where,
      select: {
        userId: true,
        emotionTag: true,
        emotionScore: true,
        crisisDetected: true,
        moderationStatus: true,
        createdAt: true,
      },
      orderBy: { createdAt: 'asc' },
    });

    // 按用户分组统计
    const userMap = new Map<string, {
      posts: typeof posts;
      avgScore: number;
      latestWeather: string;
      crisisCount: number;
    }>();

    for (const post of posts) {
      const existing = userMap.get(post.userId) || {
        posts: [],
        avgScore: 0,
        latestWeather: '',
        crisisCount: 0,
      };
      existing.posts.push(post);
      if (post.crisisDetected) existing.crisisCount++;
      existing.latestWeather = post.emotionTag || '';
      existing.avgScore = existing.posts.reduce((sum: number, p: any) => sum + p.emotionScore, 0) / existing.posts.length;
      userMap.set(post.userId, existing);
    }

    // 转为数组并排序（最需关注的用户排前面）
    const dashboard = Array.from(userMap.entries())
      .map(([uid, data]) => ({
        userId: uid,
        postCount: data.posts.length,
        avgEmotionScore: Math.round(data.avgScore),
        latestWeather: data.latestWeather,
        crisisCount: data.crisisCount,
        recentPosts: data.posts.slice(-5),
      }))
      .sort((a, b) => a.avgEmotionScore - b.avgEmotionScore);

    return {
      period: `${days}天`,
      totalPosts: posts.length,
      crisisPosts: posts.filter((p: any) => p.crisisDetected).length,
      users: dashboard,
    };
  }
}
