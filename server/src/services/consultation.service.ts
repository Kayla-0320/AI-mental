import prisma from '../config/database';
import aiService from './ai.service';
import algorithmBridge, {
  SmartChatResponse,
  ChatStreamEventName,
  ChatStreamDone,
} from './algorithm-bridge';
import { AppError } from '../middlewares/errorHandler';

/**
 * 服务端下发给浏览器的流式事件 = 算法层事件 + 服务端自有事件。
 *
 * 刻意与 `ChatStreamEventName` 分开：那个是**算法层契约**（Python / Node /
 * 浏览器三端同步，见 algorithm-bridge.ts 的注释），`user` 只在 Node 与浏览器
 * 之间有含义（回传落库后的用户消息，替换本地乐观占位气泡）。把它塞进算法层
 * 契约会污染三端同步的那份约定。
 */
export type ConsultationStreamEvent = ChatStreamEventName | 'user';

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

      // 危机判定与落库已移入 persistAssistantReply（两条路径共用）。
      // 热线补全必须在这里做完再落库，且是幂等的：算法层危机资源卡本身已带热线。
      const rawReply = smartResult.reply || '';
      const aiContent =
        smartResult.risk_level === 'crisis'
          ? this.appendCrisisHotlines(rawReply)
          : rawReply;

      const persisted = await this.persistAssistantReply(
        conversationId,
        userId,
        content,
        conversation,
        smartResult,
        aiContent,
      );

      return { userMessage, ...persisted };
    } catch (error) {
      // AI 服务失败：必须留下可诊断的日志。
      // 这里曾静默吞掉异常并写入一条「抱歉，我暂时无法回应…」的 assistant 消息，
      // 导致两个后果：(1) 故障现场无任何日志，事后无法定位；
      // (2) 该伪装回复被当作真实对话上下文喂回模型，永久污染后续回复。
      const err = error instanceof Error ? error : new Error(String(error));
      console.error('[Consultation] AI 回复生成失败', {
        conversationId,
        userId,
        messagePreview: content.slice(0, 60),
        name: err.name,
        message: err.message,
        stack: err.stack,
      });

      // 显式告知调用方失败：不写入任何 assistant 消息，不伪造「已回复」状态。
      // 前端据 failed 字段渲染可重试的失败态，而不是把它当成一条 AI 回复。
      return {
        userMessage,
        aiMessage: null,
        isCrisis: false,
        riskLevel: 'low',
        failed: true,
        errorMessage: '抱歉，我暂时无法回应。但我在这里陪着你，请稍后再试。如果你急需倾诉，可以拨打心理援助热线：400-161-9995',
        errorCode: 'AI_UNAVAILABLE',
      };
    }
  }

  /**
   * 危机回复的热线补全（幂等）。
   *
   * 算法层 SafetyLoop 的危机资源卡（`audit_rules.yaml` 的 `crisis_resource_card`）
   * **已经包含三条热线**，因此仅在文本里**尚未出现**热线时才追加 —— 无条件拼接
   * 会让同一条热线在一屏内出现两遍（改造前实测确实重复）。
   *
   * @param reply 待补全的回复文本
   * @returns 含热线的回复文本
   */
  private appendCrisisHotlines(reply: string): string {
    if (reply.includes('400-161-9995')) return reply;
    return (
      reply +
      '\n\n请记住，你并不孤单。如果你正在经历困难时刻，请拨打24小时心理援助热线：' +
      '\n- 全国：400-161-9995\n- 北京：010-82951332\n- 生命热线：400-821-1215'
    );
  }

  /**
   * 持久化 AI 回复 —— 非流式（sendMessage）与流式（sendMessageStream）两条路径共用。
   *
   * 抽出来的理由与之一致：危机判定后的副作用（升级风险等级、写危机记录、落 AI 消息、
   * 更新会话计数、触发深度情绪分析）如果两条路径各写一份，必然随时间漂移，而这里
   * 任何一处漂移都意味着「某条路径的危机没有被记录」。
   *
   * @param conversationId 会话 ID
   * @param userId 用户 ID
   * @param userContent 用户本轮原文（用于危机记录与画像分析）
   * @param conversation 会话记录（需要 messageCount 判断是否首轮、用于生成标题）
   * @param smartResult 算法层给出的审计结论
   * @param aiContent 最终落库的回复文本（已含危机热线补全）
   */
  private async persistAssistantReply(
    conversationId: string,
    userId: string,
    userContent: string,
    conversation: { messageCount: number },
    smartResult: SmartChatResponse,
    aiContent: string,
  ) {
    const isCrisis = smartResult.risk_level === 'crisis';
    const requiresEscalation = smartResult.requires_escalation === true;

    if (isCrisis) {
      // 热线补全放在调用方完成（见 sendMessageStream），这里只负责副作用。
      // 更新用户风险等级
      await prisma.patientProfile.updateMany({
        where: { userId },
        data: { riskLevel: 'CRISIS' },
      });

      // 持久化危机记录到管理端
      await this.createCrisisRecord(userId, userContent, {
        isCrisis: true,
        matchedKeywords: [],
        suggestion: smartResult.reply,
      });
    }

    const aiMessage = await prisma.message.create({
      data: {
        conversationId,
        role: 'assistant',
        content: aiContent,
        tokenUsage: 0,
        sentiment: JSON.stringify({
          dialog_state: smartResult.dialog_state,
          dialogue_mode: smartResult.dialogue_mode || 'EMPATHY',
          action_type: smartResult.action_type,
          risk_level: smartResult.risk_level,
          audit_passed: smartResult.audit_passed,
          emotion_probs: smartResult.emotion_probs,
          fallback: smartResult.fallback,
          // C2D2 两根轴的结果：仅供咨询师端与审计链路使用，
          // 不得直接呈现给患者（见 algorithm-bridge.ts 的类型注释）。
          scene_retrieval: smartResult.scene_retrieval || null,
          cognitive_distortion: smartResult.cognitive_distortion || null,
        }),
      },
    });

    // 更新对话信息
    await prisma.conversation.update({
      where: { id: conversationId },
      data: {
        messageCount: { increment: 2 },
        updatedAt: new Date(),
        ...(conversation.messageCount === 0
          ? { title: userContent.slice(0, 30) + '...' }
          : {}),
      },
    });

    // 异步进行深度情绪分析 + 动态画像更新
    this.analyzeAndUpdateProfile(userId, userContent, aiMessage.id);

    return {
      aiMessage,
      isCrisis,
      riskLevel: smartResult.risk_level || 'low',
      requiresEscalation,
      dialogueMode: smartResult.dialogue_mode || 'EMPATHY',
      dialogState: smartResult.dialog_state || 'INIT',
      // 结构化透出：前端据 dialogueMode 显示非诊断性的模式标签。
      // cognitiveDistortion 不建议在患者端渲染。
      sceneRetrieval: smartResult.scene_retrieval || null,
      cognitiveDistortion: smartResult.cognitive_distortion || null,
    };
  }

  /**
   * 流式发送消息并获取 AI 回复 —— 逐句上屏，安全审计仍是最终权威。
   *
   * 与非流式 `sendMessage` 的差别只有「文本怎么拿到」：
   *   - 生成阶段：边收边通过 `emit` 推 `delta`（追加）；算法层若整体改稿则推 `revise`；
   *   - 其余（用户消息落库、危机处理、AI 消息落库、画像更新、返回结构）完全共用
   *     {@link persistAssistantReply}。
   *
   * 失败处理：流式链路任何异常都**不**直接失败，而是回退到既有非流式路径
   * （`algorithmBridge.smartChat` → `aiService.counselingChat`），并用 `revise`
   * 把已上屏的残缺内容整体替换掉。只有回退也失败时才发 `error`。
   *
   * @param emit 事件发射器；实现方负责写 SSE 帧（见 consultation.controller）
   * @param signal 浏览器断开时中止上游调用，避免继续消耗 token
   */
  async sendMessageStream(
    conversationId: string,
    userId: string,
    content: string,
    contentType: string,
    emit: (event: ConsultationStreamEvent, data: Record<string, unknown>) => void,
    signal?: AbortSignal,
  ): Promise<void> {
    const conversation = await prisma.conversation.findFirst({
      where: { id: conversationId, userId },
    });

    if (!conversation) {
      throw new AppError('对话不存在', 404);
    }

    // 先落库用户消息：与 sendMessage 一致。立刻回传带真实 id 的记录，
    // 前端用它替换本地的乐观占位气泡。
    const userMessage = await prisma.message.create({
      data: { conversationId, userId, role: 'user', content, contentType },
    });
    emit('user', { userMessage });

    const historyMessages = await prisma.message.findMany({
      where: { conversationId },
      orderBy: { createdAt: 'asc' },
      take: 20,
    });
    const conversationHistory = historyMessages.map((m: any) => ({
      role: (m.role === 'user' ? 'user' : 'assistant') as 'user' | 'assistant',
      content: m.content,
    }));

    const patientProfile = await prisma.patientProfile.findUnique({ where: { userId } });

    // ---- 1. 流式生成 ----
    let smartResult: SmartChatResponse | null = null;
    let streamedText = '';

    try {
      for await (const chunk of algorithmBridge.smartChatStream(
        {
          user_id: userId,
          message: content,
          conversation_history: conversationHistory,
          session_id: conversationId,
        },
        signal,
      )) {
        if (chunk.event === 'done') {
          const done = chunk.data as unknown as ChatStreamDone;
          smartResult = { ...done, reply: String(done.text ?? '') };
          break;
        }
        if (chunk.event === 'error') {
          throw new Error(String(chunk.data.error ?? '算法层流式对话失败'));
        }
        if (chunk.event === 'delta') {
          streamedText += String(chunk.data.text ?? '');
        } else if (chunk.event === 'revise') {
          // 整体替换：算法层的安全审计否掉了已生成内容
          streamedText = String(chunk.data.text ?? '');
        }
        emit(chunk.event, chunk.data);
      }
    } catch (error) {
      // 流式中断（网络抖动 / 算法服务崩 / 被 abort）：交给下面的回退路径。
      // 注意这里**不用** console.error：回退是预期内的降级，不是故障。
      console.warn('[Consultation] 流式生成中断，回退非流式路径:', error);
    }

    // ---- 2. 回退：流式没拿到 done ----
    if (!smartResult) {
      try {
        const fallback = await algorithmBridge.smartChat({
          user_id: userId,
          message: content,
          conversation_history: conversationHistory,
          session_id: conversationId,
        });

        if (fallback.fallback || !fallback.reply) {
          const aiResponse = await aiService.counselingChat(content, conversationHistory, {
            riskLevel: patientProfile?.riskLevel,
          });
          smartResult = { ...fallback, reply: aiResponse.content, fallback: false };
        } else {
          smartResult = fallback;
        }
      } catch (error) {
        // 回退也失败：与 sendMessage 的失败语义保持一致 ——
        // 不写任何 assistant 消息，不伪造「已回复」，让前端渲染可重试的失败态。
        const err = error instanceof Error ? error : new Error(String(error));
        console.error('[Consultation] 流式对话生成失败', {
          conversationId,
          userId,
          messagePreview: content.slice(0, 60),
          name: err.name,
          message: err.message,
          stack: err.stack,
        });
        emit('error', {
          failed: true,
          errorCode: 'AI_UNAVAILABLE',
          errorMessage:
            '抱歉，我暂时无法回应。但我在这里陪着你，请稍后再试。如果你急需倾诉，可以拨打心理援助热线：400-161-9995',
          userMessage,
        });
        return;
      }
    }

    // ---- 3. 定稿：危机热线补全（幂等）----
    const isCrisis = smartResult.risk_level === 'crisis';
    let aiContent = smartResult.reply || '';
    if (isCrisis) {
      aiContent = this.appendCrisisHotlines(aiContent);
    }
    // 算法层理论上不会返回空文本；真为空时宁可报错也不要写一条空气泡进历史。
    if (!aiContent.trim()) {
      emit('error', {
        failed: true,
        errorCode: 'EMPTY_REPLY',
        errorMessage: '抱歉，我暂时无法回应，请稍后再试。',
        userMessage,
      });
      return;
    }

    // ---- 4. 与已上屏内容比对，不一致就整体替换 ----
    // 两种情形会走到这里：(1) 走了回退路径，上屏的是半句残缺内容；
    // (2) 危机时补了热线。前端必须把 revise/done.text 当作权威文本。
    if (aiContent.trim() !== streamedText.trim()) {
      emit('revise', { text: aiContent });
    }

    // ---- 5. 落库并收尾 ----
    const persisted = await this.persistAssistantReply(
      conversationId,
      userId,
      content,
      conversation,
      smartResult,
      aiContent,
    );

    emit('done', { text: aiContent, ...persisted });
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

  //  创建危机记录（使用 raw SQL，因 CrisisRecord 模型刚添加）
  private async createCrisisRecord(userId: string, sourceText: string, crisisResult: any) {
    try {
      const keywords = crisisResult.matchedKeywords || [];
      const severity = keywords.length >= 2 ? 'CRITICAL' : 'HIGH';
      const trigger = `聊天关键词检测：${keywords.join('、')}`;

      await prisma.$executeRawUnsafe(`
        INSERT INTO crisis_records (id, "userId", severity, status, trigger, "triggerType", "sourceText", description, "detectedAt", "createdAt", "updatedAt")
        VALUES (gen_random_uuid()::text, $1, $2, 'OPEN', $3, 'chat_keyword', $4, $5, NOW(), NOW(), NOW())
      `, userId, severity, trigger, sourceText, `用户在AI聊天中发送包含危机关键词的消息：${keywords.join('、')}`);
    } catch (err) {
      console.error('[CrisisRecord] 创建失败:', err);
    }
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

      // 睡眠不可由文字推算：只能取用户真实手动记录，无记录则为 0（未采集）
      const sleepStart = new Date();
      sleepStart.setDate(sleepStart.getDate() - 30);
      const sleepRecords = await prisma.sleepRecord.findMany({
        where: { userId, recordDate: { gte: sleepStart } },
        orderBy: { recordDate: 'desc' },
        take: 30,
      });
      const realSleepScore = sleepRecords.length > 0
        ? Math.max(0, Math.min(100, Math.round(
            (sleepRecords.reduce((s: number, r: any) => s + (r.quality || 0), 0) / sleepRecords.length) * 10,
          )))
        : 0;

      // 5. 保存今日动态画像
      const patientProfile = await prisma.patientProfile.findUnique({ where: { userId } });
      if (patientProfile) {
        await prisma.psychologicalProfile.create({
          data: {
            patientProfileId: patientProfile.id,
            anxiety: dynamicProfile.anxiety,
            depression: dynamicProfile.depression,
            stress: dynamicProfile.stress,
            sleepQuality: realSleepScore,
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
