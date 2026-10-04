/**
 * 温柔守护者 —— 社区内容自动审核服务
 *
 * 三级审核机制：
 *   Level 1: 温暖发布（低风险 → 自动通过）
 *   Level 2: 陪伴等待（中风险 → 柔性暂存，等待人工复核）
 *   Level 3: 紧急守护（危机 → 拦截 + 关怀介入 + 匿名共鸣帖）
 */
import aiService from './ai.service';
import prisma from '../config/database';

// ============================================================
// 类型定义
// ============================================================

export type ModerationLevel = 'warm_publish' | 'gentle_wait' | 'crisis_guard';

export interface ModerationResult {
  level: ModerationLevel;
  emotionTag: string;           // 情绪标签（晴/多云/雨/暴风雨）
  emotionScore: number;         // -100 ~ 100
  crisisDetected: boolean;
  reasons: string[];            // 判定原因（管理员可见）
  // 用户看到的温暖提示语
  userMessage: string;
  // 危机守护时的共鸣帖内容（仅 crisis_guard 时有值）
  empathyPostContent?: string;
  // 危机热线提示
  crisisHotline?: string;
}

// ============================================================
// 关键词库
// ============================================================

/** 危机信号关键词 —— 最高优先级 */
const CRISIS_KEYWORDS = [
  '自杀', '不想活', '去死', '结束生命', '自残', '割腕', '跳楼',
  '没有意义', '活着没意思', '想消失', '消失掉', '世界没有我',
  '解脱', '了结', '安眠药', '写遗书', '告别这个世界',
  '不想醒来', '活着好累', '死了算了', '没人会在意我死了',
];

/** 有害内容关键词 —— 广告/攻击/违规 */
const HARMFUL_KEYWORDS = [
  // 广告引流
  '加微信', '加我微信', '私聊我', '兼职赚钱', '日赚', '刷单',
  '免费领', '扫码', '点击链接', '复制口令',
  // 人身攻击
  '傻逼', '脑残', '废物', '去死吧', '你妈', '操你',
  // 药物滥用引导
  '买药', '卖药', '处方药交易',
];

/** 中风险情绪关键词 —— 需要关注但不拦截 */
const CONCERN_KEYWORDS = [
  '失眠', '睡不着', '焦虑', '恐慌', '害怕', '恐惧',
  '崩溃', '撑不住', '受不了', '好痛苦', '很绝望',
  '被欺负', '霸凌', '校园暴力', '家暴', '虐待',
  '孤独', '没人理解', '没有人关心', '被抛弃',
];

// ============================================================
// 情绪天气映射
// ============================================================

function getEmotionWeather(score: number): string {
  if (score >= 40) return '☀️ 晴';
  if (score >= 10) return '⛅ 多云';
  if (score >= -20) return '🌧️ 雨';
  return '⛈️ 暴风雨';
}

// ============================================================
// 核心审核逻辑
// ============================================================

class ContentModerationService {

  /**
   * 审核社区内容（帖子或评论）
   */
  async moderate(content: string, userId: string): Promise<ModerationResult> {
    const reasons: string[] = [];

    // ---- Step 1: 关键词快速扫描 ----
    const crisisHits = CRISIS_KEYWORDS.filter(kw => content.includes(kw));
    const harmfulHits = HARMFUL_KEYWORDS.filter(kw => content.includes(kw));
    const concernHits = CONCERN_KEYWORDS.filter(kw => content.includes(kw));

    // ---- Step 2: 危机信号检测（关键词命中 → 直接升级） ----
    if (crisisHits.length > 0) {
      reasons.push(`检测到危机关键词：${crisisHits.join('、')}`);

      // 同时用 AI 做二次确认（非阻塞，但记录结果）
      let aiConfirmed = true;
      try {
        const aiResult = await aiService.detectCrisis(content);
        aiConfirmed = aiResult.isCrisis;
      } catch {
        // AI 不可用时，关键词命中即视为确认
      }

      if (aiConfirmed) {
        return {
          level: 'crisis_guard',
          emotionTag: '⛈️ 暴风雨',
          emotionScore: -80,
          crisisDetected: true,
          reasons,
          userMessage: '我注意到你现在可能正在经历很痛苦的事情。你愿意说出来，这已经很勇敢了。现在有一个更了解你的人想和你聊聊，可以吗？',
          empathyPostContent: await this.generateEmpathyPost(content),
          crisisHotline: '24小时心理援助热线：400-161-9995 | 紧急电话：120',
        };
      }
    }

    // ---- Step 3: 有害内容检测（广告/攻击） ----
    if (harmfulHits.length > 0) {
      reasons.push(`检测到不当内容关键词：${harmfulHits.join('、')}`);

      // 有害内容不直接拒绝，而是温柔地引导修改
      return {
        level: 'gentle_wait',
        emotionTag: getEmotionWeather(-10),
        emotionScore: -10,
        crisisDetected: false,
        reasons,
        userMessage: '感谢你的分享。我们注意到内容中可能包含一些不太适合公开的信息，能不能稍微调整一下措辞呢？我们相信你想表达的一定是很有价值的内容。',
      };
    }

    // ---- Step 4: AI 情绪分析（判断中风险） ----
    let emotionScore = 0;
    let primaryEmotion = '平静';
    let urgency = 0;

    try {
      const emotionResult = await aiService.analyzeEmotionDeep(content);
      emotionScore = emotionResult.sentimentScore;
      primaryEmotion = emotionResult.primaryEmotion;
      urgency = emotionResult.urgency;
    } catch {
      // AI 不可用时，用关键词数量估算
      emotionScore = -(concernHits.length * 15);
      primaryEmotion = concernHits.length > 0 ? '焦虑' : '平静';
      urgency = concernHits.length > 2 ? 60 : 20;
    }

    // ---- Step 5: 综合判定 ----
    const weather = getEmotionWeather(emotionScore);

    // 中风险：多个负面关键词 + AI 判定高紧迫度
    if (concernHits.length >= 2 || (concernHits.length >= 1 && urgency >= 60)) {
      reasons.push(`关注关键词：${concernHits.join('、')}`);
      reasons.push(`AI 情绪判定：${primaryEmotion}，紧迫度 ${urgency}`);

      return {
        level: 'gentle_wait',
        emotionTag: weather,
        emotionScore,
        crisisDetected: false,
        reasons,
        userMessage: '你的心事我们已经收到了。有时候重要的话值得被好好对待，我们正在认真查看，很快就会出现在社区里。在这之前，想先和你说：你说的每一个字都很重要。',
      };
    }

    // ---- Step 6: 低风险 → 温暖发布 ----
    if (concernHits.length === 1) {
      reasons.push(`轻微关注词：${concernHits[0]}`);
    }

    return {
      level: 'warm_publish',
      emotionTag: weather,
      emotionScore,
      crisisDetected: false,
      reasons,
      userMessage: '你的分享已发布，感谢你愿意说出来。',
    };
  }

  /**
   * 生成匿名共鸣帖（替代危机原帖）
   * 保留表达权，但去除可识别信息和具体自伤细节
   */
  private async generateEmpathyPost(originalContent: string): Promise<string> {
    try {
      const response = await aiService.chat(
        [{
          role: 'user',
          content: `以下是一位青少年写的一段话，ta正在经历痛苦。请将其转化为一篇匿名的、去标识化的"感受帖"，用于在社区发布。

要求：
1. 保留核心情感（痛苦、挣扎、但还有希望的微光）
2. 去除所有可识别个人信息
3. 去除具体的自伤/自杀方法描述
4. 让读到的人产生共鸣："原来不只是我这样"
5. 结尾带一句温暖的鼓励
6. 100字以内

原文：${originalContent}`,
        }],
        { temperature: 0.7 },
      );
      return response.content || '有人今天很难过，但ta还在坚持。如果你也有类似的感受，你不是一个人。';
    } catch {
      return '有人今天很难过，但ta还在坚持。如果你也有类似的感受，你不是一个人。';
    }
  }

  /**
   * 危机守护后续处理：
   * 1. 创建通知给管理员
   * 2. 发布匿名共鸣帖
   * 3. 记录危机事件
   */
  async handleCrisisGuard(
    userId: string,
    originalContent: string,
    moderationResult: ModerationResult,
  ): Promise<{ empathyPostId?: string }> {
    // 1. 通知所有管理员
    const admins = await prisma.user.findMany({
      where: { role: 'ADMIN' },
      select: { id: true },
    });

    if (admins.length > 0) {
      await prisma.notification.createMany({
        data: admins.map((admin: { id: string }) => ({
          userId: admin.id,
          type: 'crisis_alert',
          title: '⚠️ 社区危机信号预警',
          content: `用户 ${userId.slice(0, 8)}... 的帖子检测到危机信号。\n原因：${moderationResult.reasons.join('；')}\n请及时介入。`,
          link: `/admin/community/pending`,
        })),
      });
    }

    // 2. 发布匿名共鸣帖
    if (moderationResult.empathyPostContent) {
      const systemUser = await this.getOrCreateSystemUser();
      const empathyPost = await prisma.communityPost.create({
        data: {
          userId: systemUser.id,
          title: '有人和你一样',
          content: moderationResult.empathyPostContent,
          category: 'encouragement',
          isAnonymous: true,
          isApproved: true, // 共鸣帖直接发布
          moderationStatus: 'warm_publish',
          emotionTag: moderationResult.empathyPostContent ? '🌤️ 微光' : '⛈️ 暴风雨',
          emotionScore: -30,
          crisisDetected: false, // 共鸣帖本身不含危机
          moderationReason: '由危机原帖转化而来的匿名共鸣帖',
        },
      });
      return { empathyPostId: empathyPost.id };
    }

    return {};
  }

  /**
   * 获取或创建系统用户（用于发布匿名共鸣帖）
   */
  private async getOrCreateSystemUser() {
    let systemUser = await prisma.user.findFirst({
      where: { role: 'ADMIN', nickname: '温柔守护者' },
    });

    if (!systemUser) {
      systemUser = await prisma.user.create({
        data: {
          email: 'guardian@system.local',
          password: 'system_generated_hash',
          phone: '00000000000',
          nickname: '温柔守护者',
          role: 'ADMIN',
        },
      });
    }

    return systemUser;
  }
}

export default new ContentModerationService();
