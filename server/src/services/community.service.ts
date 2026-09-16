import prisma from '../config/database';
import { AppError } from '../middlewares/errorHandler';

export class CommunityService {
  // 创建帖子
  async createPost(userId: string, data: { title: string; content: string; category: string; tags?: string[]; isAnonymous?: boolean }) {
    const post = await prisma.communityPost.create({
      data: {
        userId,
        title: data.title,
        content: data.content,
        category: data.category,
        tags: data.tags ? JSON.stringify(data.tags) : null,
        isAnonymous: data.isAnonymous ?? true,
        isApproved: false, // 需要审核
      },
    });
    return post;
  }

  // 获取帖子列表
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

  // 发表评论
  async addComment(postId: string, userId: string, content: string, isAnonymous: boolean = true) {
    const post = await prisma.communityPost.findUnique({ where: { id: postId } });
    if (!post) throw new AppError('帖子不存在', 404);

    const comment = await prisma.communityComment.create({
      data: { postId, userId, content, isAnonymous, isApproved: false },
    });

    await prisma.communityPost.update({
      where: { id: postId },
      data: { commentCount: { increment: 1 } },
    });

    return comment;
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

  // 获取待审核内容（管理员）
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
}
