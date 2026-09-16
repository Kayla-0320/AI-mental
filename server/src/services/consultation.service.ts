import prisma from '../config/database';
import aiService from './ai.service';
import algorithmBridge from './algorithm-bridge';
import { AppError } from '../middlewares/errorHandler';

class ConsultationService {
  // 创建新对话
  async createConversation(userId: string, title?: string) {
    return prisma.conversation.create({
      data: {
        userId,
        title: title || '新对话',
      },
    });
  }

  // 获取用户对话列表
  async getConversations(userId: string, page: number = 1, pageSize: number = 20) {
    const [conversations, total] = await Promise.all([
      prisma.conversation.findMany({
        where: { userId, isActive: true },
        orderBy: { updatedAt: 'desc' },
        skip: (page - 1) * pageSize,
        take: pageSize,
        include: {
          _count: { select: { messages: true } },
          messages: {
            take: 1,
            orderBy: { createdAt: 'desc' },
            select: { content: true, createdAt: true },
          },
        },
      }),
      prisma.conversation.count({ where: { userId, isActive: true } }),
    ]);

    return { conversations, total, page, pageSize };
  }

  // 获取对话详情及消息
  async getConversationMessages(conversationId: string, userId: string, page: number = 1, pageSize: number = 50) {
    const conversation = await prisma.conversation.findFirst({
      where: { id: conversationId, userId },
    });

    if (!conversation) {
      throw new AppError('对话不存在', 404);
    }

    const messages = await prisma.message.findMany({
      where: { conversationId },
      orderBy: { createdAt: 'asc' },
      skip: (page - 1) * pageSize,
      take: pageSize,
    });

    return { conversation, messages, total: messages.length };
  }

  // 发送消息并获取 AI 回复
  async sendMessage(conversationId: string, userId: string, content: string, contentType: string = 'text') {
    const conversation = await prisma.conversation.findFirst({
      where: { id: conversationId, userId },
    });

    if (!conversation) {
      throw new AppError('对话不存在', 404);
    }

    // 保存用户消息
    const userMessage = await prisma.message.create({
      data: {
        conversationId,
        userId,
        role: 'user',
        content,
        contentType,
      },
    });

    // 危机检测
    const crisisResult = await aiService.detectCrisis(content);
    if (crisisResult.isCrisis) {
      const crisisMessage = await prisma.message.create({
        data: {
          conversationId,
          role: 'assistant',
          content: crisisResult.suggestion + '\n\n请记住，你并不孤单。如果你正在经历困难时刻，请拨打24小时心理援助热线：\n- 全国：400-161-9995\n- 北京：010-82951332\n- 生命热线：400-821-1215',
          sentiment: 'crisis',
        },
      });

      // 更新用户风险等级
      await prisma.patientProfile.updateMany({
        where: { userId },
        data: { riskLevel: 'CRISIS' },
      });

      return {
        userMessage,
        aiMessage: crisisMessage,
        isCrisis: true,
      };
    }

    // 获取历史对话用于上下文
    const historyMessages = await prisma.message.findMany({
      where: { conversationId },
      orderBy: { createdAt: 'asc' },
      take: 20,
    });

    const conversationHistory = historyMessages.map((m: any) => ({
      role: (m.role === 'user' ? 'user' : 'assistant') as 'user' | 'assistant',
      content: m.content,
    }));

    // 获取患者信息
    const patientProfile = await prisma.patientProfile.findUnique({
      where: { userId },
    });

    // 调用 AI 服务（优先使用算法桥接层：状态机 + 安全审计）
    try {
      // 尝试通过算法桥接层（对话状态机 + 安全审计中间件）
      const smartResult = await algorithmBridge.smartChat({
        user_id: userId,
        message: content,
        conversation_history: conversationHistory,
        session_id: conversationId,
      });

      let aiContent: string;
      if (smartResult.fallback || !smartResult.reply) {
        // 降级：算法服务不可用，使用原有 LLM 直调
        const aiResponse = await aiService.counselingChat(content, conversationHistory, {
          riskLevel: patientProfile?.riskLevel,
        });
        aiContent = aiResponse.content;
      } else {
        aiContent = smartResult.reply;
      }

      const aiMessage = await prisma.message.create({
        data: {
          conversationId,
          role: 'assistant',
          content: aiContent,
          tokenUsage: 0,
          sentiment: JSON.stringify({
            dialog_state: smartResult.dialog_state,
            action_type: smartResult.action_type,
            risk_level: smartResult.risk_level,
            audit_passed: smartResult.audit_passed,
            emotion_probs: smartResult.emotion_probs,
            fallback: smartResult.fallback,
          }),
        },
      });

      // 更新对话信息
      await prisma.conversation.update({
        where: { id: conversationId },
        data: {
          messageCount: { increment: 2 },
          updatedAt: new Date(),
          ...(conversation.messageCount === 0 ? { title: content.slice(0, 30) + '...' } : {}),
        },
      });

      // 异步进行深度情绪分析 + 动态画像更新
      this.analyzeAndUpdateProfile(userId, content, aiMessage.id);

      return { userMessage, aiMessage, isCrisis: false };
    } catch (error) {
      // AI 服务失败时返回友好提示
      const fallbackMessage = await prisma.message.create({
        data: {
          conversationId,
          role: 'assistant',
          content: '抱歉，我暂时无法回应。但我在这里陪着你，请稍后再试。如果你急需倾诉，可以拨打心理援助热线：400-161-9995',
        },
      });

      return { userMessage, aiMessage: fallbackMessage, isCrisis: false };
    }
  }

  // 发送苏格拉底式反问消息
  async sendSocraticMessage(conversationId: string, userId: string, content: string) {
    const conversation = await prisma.conversation.findFirst({
      where: { id: conversationId, userId },
    });

    if (!conversation) throw new AppError('对话不存在', 404);

    // 保存用户消息
    const userMessage = await prisma.message.create({
      data: { conversationId, userId, role: 'user', content, contentType: 'text' },
    });

    // 获取历史对话
    const historyMessages = await prisma.message.findMany({
      where: { conversationId },
      orderBy: { createdAt: 'asc' },
      take: 20,
    });

    const conversationHistory = historyMessages.map((m: any) => ({
      role: (m.role === 'user' ? 'user' : 'assistant') as 'user' | 'assistant',
      content: m.content,
    }));

    try {
      const aiResponse = await aiService.socraticChat(content, conversationHistory);

      const aiMessage = await prisma.message.create({
        data: {
          conversationId,
          role: 'assistant',
          content: aiResponse.content,
          tokenUsage: aiResponse.usage?.total_tokens,
        },
      });

      await prisma.conversation.update({
        where: { id: conversationId },
        data: {
          messageCount: { increment: 2 },
          updatedAt: new Date(),
          ...(conversation.messageCount === 0 ? { title: content.slice(0, 30) + '...' } : {}),
        },
      });

      return { userMessage, aiMessage };
    } catch {
      const fallbackMessage = await prisma.message.create({
        data: {
          conversationId,
          role: 'assistant',
          content: '嗯，我听到了。能再多说一点吗？你刚才提到的那个感觉，是什么时候开始出现的？',
        },
      });
      return { userMessage, aiMessage: fallbackMessage };
    }
  }
  async deleteConversation(conversationId: string, userId: string) {
    const conversation = await prisma.conversation.findFirst({
      where: { id: conversationId, userId },
    });

    if (!conversation) {
      throw new AppError('对话不存在', 404);
    }

    await prisma.conversation.update({
      where: { id: conversationId },
      data: { isActive: false },
    });
  }

  // 异步深度情绪分析 + 动态画像更新
  private async analyzeAndUpdateProfile(userId: string, userText: string, aiMessageId: string) {
    try {
      // 1. 对当前消息做深度情绪分析
      const emotion = await aiService.analyzeEmotionDeep(userText);

      // 2. 更新 AI 消息的情绪标注
      await prisma.message.update({
        where: { id: aiMessageId },
        data: { sentiment: JSON.stringify(emotion) },
      });

      // 3. 收集今天的所有文字记录
      const today = new Date();
      today.setHours(0, 0, 0, 0);

      const todayMessages = await prisma.message.findMany({
        where: {
          userId,
          role: 'user',
          createdAt: { gte: today },
        },
        orderBy: { createdAt: 'asc' },
      });

      const todayMoods = await prisma.moodRecord.findMany({
        where: { userId, recordedAt: { gte: today } },
      });

      const texts = [
        ...todayMessages.map((m: any) => m.content),
        ...todayMoods.filter((m: any) => m.note).map((m: any) => `[心情:${m.mood}] ${m.note}`),
      ];

      if (texts.length === 0) return;

      // 4. 动态生成今日心理画像
      const dynamicProfile = await aiService.generateDynamicProfile(texts);

      // 5. 保存今日动态画像
      const patientProfile = await prisma.patientProfile.findUnique({ where: { userId } });
      if (patientProfile) {
        await prisma.psychologicalProfile.create({
          data: {
            patientProfileId: patientProfile.id,
            anxiety: dynamicProfile.anxiety,
            depression: dynamicProfile.depression,
            stress: dynamicProfile.stress,
            sleepQuality: dynamicProfile.sleepQuality,
            socialActivity: dynamicProfile.socialActivity,
            emotionalStability: dynamicProfile.emotionalStability,
            overallScore: dynamicProfile.overallScore,
            riskLevel: dynamicProfile.riskLevel,
            report: dynamicProfile.emotionalSummary,
            assessedBy: 'AI_DYNAMIC',
            metadata: JSON.stringify({
              dominantEmotions: dynamicProfile.dominantEmotions,
              emotionalTrend: dynamicProfile.emotionalTrend,
              source: 'daily_text_analysis',
            }),
          },
        });

        // 6. 更新患者风险等级
        await prisma.patientProfile.update({
          where: { userId },
          data: { riskLevel: dynamicProfile.riskLevel },
        });
      }
    } catch {
      // 静默失败，不影响主流程
    }
  }
}

export default new ConsultationService();
