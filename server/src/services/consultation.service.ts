import prisma from '../config/database';
import aiService from './ai.service';
import algorithmBridge, {
  SmartChatResponse,
  ChatStreamEventName,
  ChatStreamDone,
} from './algorithm-bridge';
import { AppError } from '../middlewares/errorHandler';
import crisisService from './crisis.service';
import type { CrisisSeverity } from './crisis.service';
import {
  buildTranscriptText,
  normalizeSummaryFields,
  type ConversationSummaryRecord,
  type TranscriptTurn,
} from './summary.prompt';

/**
 * 服务端下发给浏览器的流式事件 = 算法层事件 + 服务端自有事件。
 *
 * 刻意与 `ChatStreamEventName` 分开：那个是**算法层契约**（Python / Node /
 * 浏览器三端同步，见 algorithm-bridge.ts 的注释），`user` 只在 Node 与浏览器
 * 之间有含义（回传落库后的用户消息，替换本地乐观占位气泡）。把它塞进算法层
 * 契约会污染三端同步的那份约定。
 */
export type ConsultationStreamEvent = ChatStreamEventName | 'user';

/**
 * 总结用的原文上限（字符）。
 *
 * 取 12000：qwen 系列上下文足够放下，而一段几百轮的陪伴对话也远超这个量 ——
 * 超长时**保留头尾**（开头是用户主动说起的事，结尾是当前状态），
 * 掐中间，而不是简单截断尾部。单纯 tail 截断会丢掉"这段对话是怎么开始的"，
 * 而纪要的第一句恰恰最需要它。
 */
const SUMMARY_TRANSCRIPT_LIMIT = 12000;

/** 总结最多纳入的消息条数（按时间正序取前 N 条；超出靠上面的字符串上限兜住） */
const SUMMARY_MESSAGE_LIMIT = 400;

/**
 * 把 `Message` 行投影成总结用的对话轮次。
 *
 * @param rows Prisma 查出的消息（需含 role/content/sentiment）
 * @returns 正序对话轮次；`system` 角色与空内容被丢弃
 */
function toTranscriptTurns(
  rows: Array<{ role: string; content: string; sentiment: string | null }>,
): TranscriptTurn[] {
  const turns: TranscriptTurn[] = [];

  for (const row of rows) {
    const content = (row.content || '').trim();
    if (!content) continue;
    if (row.role !== 'user' && row.role !== 'assistant') continue;
    // 连续重复内容只留一条：通话链路里同一句定稿可能被重复落库，
    // 重复项会让纪要把同一件事写两遍。
    const previous = turns[turns.length - 1];
    if (previous && previous.role === row.role && previous.content === content) continue;

    turns.push({
      role: row.role,
      content,
      emotion: row.role === 'assistant' ? extractPrimaryEmotion(row.sentiment) : undefined,
    });
  }

  return turns;
}

/**
 * 从 `Message.sentiment` 里取情绪主标签。
 *
 * ⚠️ `sentiment` 这一列有**两种写入方**，必须能同时容忍：
 *   - 聊天链路：对话状态机的元数据（dialogue_mode / risk_level / …）
 *   - 深度分析：`{ primaryEmotion, intensity, … }`（见 aiService.analyzeEmotionDeep）
 * 只有后者带 `primaryEmotion`；解析不出来就返回 undefined（不显示），
 * 绝不为了"显示得像纪要"而编一个情绪词。
 *
 * @param sentiment 原始 JSON 字符串
 * @returns 情绪词，或 undefined
 */
function extractPrimaryEmotion(sentiment: string | null): string | undefined {
  if (!sentiment) return undefined;
  try {
    const parsed = JSON.parse(sentiment);
    const emotion = parsed?.primaryEmotion;
    return typeof emotion === 'string' && emotion.trim() ? emotion.trim() : undefined;
  } catch {
    return undefined;
  }
}

/**
 * 超长时保留头尾、掐中间，并在拼接处留下明确标记。
 *
 * 标记必须显眼（而不是悄悄省略）：否则模型会把前后两段当成连续对话，
 * 产出"用户在上一句还平静、下一句突然崩溃"这种被截断制造出来的假转折。
 */
function truncateTranscript(text: string, limit: number = SUMMARY_TRANSCRIPT_LIMIT): string {
  if (text.length <= limit) return text;
  const headSize = Math.floor(limit * 0.6);
  const tailSize = limit - headSize;
  return (
    text.slice(0, headSize) +
    '\n……（中间部分因篇幅省略，未纳入本次纪要）……\n' +
    text.slice(text.length - tailSize)
  );
}

/**
 * 把库里存的 `Conversation.summary` 解析成结构。
 *
 * 必须是**宽容**解析，原因不只是"这一列可能被别的路径写过纯文本"：
 * 改造当天就实测到了更糟的情形 —— 模型输出语法非法（要点里漏转义引号），
 * 结构化字段全空、整段 JSON 被当成 overview 存了进去。那种记录**已经在库里**，
 * 所以读取路径也要过一遍规范化，让坏记录在界面上自动恢复成正常纪要，
 * 而不是要求用户先点一次"重新生成"。
 *
 * @param raw 库里的原始文本
 * @returns 总结结构；空值返回 null
 */
function parseStoredSummary(raw: string | null): ConversationSummaryRecord | null {
  if (!raw || !raw.trim()) return null;

  let meta: { generatedAt: string; messageCount: number; model: string } = {
    generatedAt: '',
    messageCount: 0,
    model: '',
  };
  let fieldsSource = raw;

  try {
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      meta = {
        generatedAt: typeof parsed.generatedAt === 'string' ? parsed.generatedAt : '',
        messageCount: typeof parsed.messageCount === 'number' ? parsed.messageCount : 0,
        model: typeof parsed.model === 'string' ? parsed.model : '',
      };
      fieldsSource = raw;
    }
  } catch {
    /* 不是合法 JSON（历史纯文本）→ 下面按纯文本兜底 */
  }

  // 无论严格解析成功与否，都过一遍规范化：它同时承担"双重编码还原"
  // 与"语法坏掉的输出里把字段抠回来"两件事。
  const fields = normalizeSummaryFields(fieldsSource);

  const hasAnyField =
    fields.overview ||
    fields.keyPoints.length > 0 ||
    fields.emotionalTrend ||
    fields.helpfulSuggestions.length > 0 ||
    fields.followUps.length > 0;

  if (!hasAnyField) return null;

  return { ...fields, ...meta };
}


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

  /**
   * 读取已保存的对话总结（**只读、不触发生成**）。
   *
   * 与 {@link generateConversationSummary} 分成两个端点，而不是"GET 顺手生成"：
   * 打开历史记录时若每次都跑一次模型，用户随手划过的每个会话都会烧一次 token，
   * 而且列表操作会变得又慢又贵。生成必须是用户显式点的动作。
   *
   * @param conversationId 会话 ID
   * @param userId 当前用户（越权会话一律 404，不泄露"存在但不是你的"）
   * @returns 总结结构；未生成过时为 null
   */
  async getConversationSummary(conversationId: string, userId: string) {
    const conversation = await prisma.conversation.findFirst({
      where: { id: conversationId, userId, isActive: true },
      select: { id: true, title: true, summary: true, messageCount: true, updatedAt: true },
    });

    if (!conversation) {
      throw new AppError('对话不存在', 404);
    }

    const summary = parseStoredSummary(conversation.summary);

    return {
      conversationId: conversation.id,
      title: conversation.title,
      messageCount: conversation.messageCount,
      /** 已生成过 = 非 null；前端据此决定是"查看总结"还是"生成总结" */
      summary,
    };
  }

  /**
   * 生成（或重新生成）对话总结并落库。
   *
   * 设计取舍：
   *   - **默认幂等**：已经生成过就直接返回旧结果，不重复调用模型。用户反复点
   *     "查看总结"不该反复烧 token；要刷新必须显式传 `force`（界面上是"重新生成"）。
   *   - **只取前 N 条消息**并按时间正序：纪要要对应用户记得的那段对话，
   *     而不是被最近几轮淹没。
   *   - 生成失败**不吞异常**（见 aiService.generateConversationSummary 的注释）；
   *     同时不写库，下次点还能重试。
   *
   * @param conversationId 会话 ID
   * @param userId 当前用户
   * @param force 为 true 时忽略已有总结、重新生成
   */
  async generateConversationSummary(conversationId: string, userId: string, force: boolean = false) {
    const conversation = await prisma.conversation.findFirst({
      where: { id: conversationId, userId, isActive: true },
      select: { id: true, title: true, summary: true },
    });

    if (!conversation) {
      throw new AppError('对话不存在', 404);
    }

    const existing = parseStoredSummary(conversation.summary);
    if (existing && !force) {
      return {
        conversationId: conversation.id,
        title: conversation.title,
        summary: existing,
        /** 是否本次新生成（前端据此提示"已更新"还是"沿用上次"） */
        generated: false,
      };
    }

    const [rows, total] = await Promise.all([
      prisma.message.findMany({
        where: { conversationId },
        orderBy: { createdAt: 'asc' },
        take: SUMMARY_MESSAGE_LIMIT,
        select: { role: true, content: true, sentiment: true },
      }),
      prisma.message.count({ where: { conversationId } }),
    ]);

    const turns = toTranscriptTurns(rows);

    // 空对话没有可总结的内容。这里明确报 400 而不是生成一句"无内容"，
    // 否则用户会在历史里看到一堆内容为空的总结，还以为是功能坏了。
    if (turns.length === 0) {
      throw new AppError('这段对话还没有内容，聊几句之后再来生成总结吧', 400);
    }

    const transcript = truncateTranscript(buildTranscriptText(turns));
    const { fields, model } = await aiService.generateConversationSummary(conversation.title, transcript);

    const record: ConversationSummaryRecord = {
      ...fields,
      generatedAt: new Date().toISOString(),
      messageCount: total,
      model,
    };

    await prisma.conversation.update({
      where: { id: conversationId },
      data: {
        summary: JSON.stringify(record),
        // 刻意**不**把 updatedAt 往前推：它是"对话内容何时更新"，
        // 生成纪要并没有产生新的对话内容；让它在历史列表里把会话顶到最上面
        // 会让人以为刚聊过天。
      },
    });

    return {
      conversationId: conversation.id,
      title: conversation.title,
      summary: record,
      generated: true,
    };
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

    // 获取最近 30 条对话用于上下文（降序取最新，再反转为时间正序）
    const historyMessages = await prisma.message.findMany({
      where: { conversationId },
      orderBy: { createdAt: 'desc' },
      take: 30,
    }).then(msgs => msgs.reverse());

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
    // 高危不必等到 crisis：算法层明确要求升级时必须留痕。
    // 否则 AGENTS.md 的红线「risk_level 为 high 或 crisis 时必须触发 escalation」
    // 在服务端等于没发生过（此前 requires_escalation 只被塞进返回值，无人消费）。
    const shouldEscalateToHuman = isCrisis || (smartResult.risk_level === 'high' && requiresEscalation);

    if (shouldEscalateToHuman) {
      // 热线补全放在调用方完成（见 sendMessageStream），这里只负责副作用。

      // 更新用户风险等级
      await prisma.patientProfile.updateMany({
        where: { userId },
        data: { riskLevel: isCrisis ? 'CRISIS' : 'HIGH' },
      });

      // 落危机记录 + 通知真人（管理员/咨询师）
      await this.raiseCrisisForUser(userId, userContent, smartResult, isCrisis);
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
          // TAMER（RQI / MIS / CUSUM）：随消息落库，刷新后仍可用于
          // 关系阶段指示器与 CUSUM 轨迹图（Profile 页按消息 sentiment 重建轨迹）。
          tamer_result: smartResult.tamer_result || null,
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
    voiceFeatures?: Record<string, unknown>,
    facialFeatures?: Record<string, unknown>,
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
      data: {
        conversationId, userId, role: 'user', content, contentType,
        // 声学特征存入消息 metadata，供后续风险回顾与趋势分析
        ...((voiceFeatures || facialFeatures) ? {
          metadata: JSON.stringify({
            ...(voiceFeatures ? { voiceFeatures } : {}),
            // 面部实时状态同样只留痕：它是"那一轮 AI 看到过什么"的证据，
            // **不是**画像分数、也不参与风险判定（见 profile.controller 的 facial 段落）。
            ...(facialFeatures ? { facialFeatures } : {}),
          }),
        } : {}),
      },
    });
    emit('user', { userMessage });

    // 获取最近 30 条对话用于上下文（降序取最新，再反转为时间正序）
    const historyMessages = await prisma.message.findMany({
      where: { conversationId },
      orderBy: { createdAt: 'desc' },
      take: 30,
    }).then(msgs => msgs.reverse());
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
          content_type: contentType,
          voice_features: voiceFeatures,
          facial_features: facialFeatures,
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
          content_type: contentType,
          voice_features: voiceFeatures,
          facial_features: facialFeatures,
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

    // TAMER（RQI / MIS / CUSUM）随终结事件下发，前端 Chat.tsx 读 `tamerResult`。
    // ⚠️ 算法层的原始 `done`（第 622 行的透传）用的是 snake_case `tamer_result`，
    // 但**那个不是最终载荷** —— 这里才是（前端以最后一次 done 为准）。
    // 两个名字都带上，避免消费端因取值口径不同而静默拿到 undefined。
    emit('done', {
      text: aiContent,
      ...persisted,
      tamerResult: smartResult.tamer_result ?? null,
      tamer_result: smartResult.tamer_result ?? null,
    });
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

    // 获取最近 30 条对话用于上下文（降序取最新，再反转为时间正序）
    const historyMessages = await prisma.message.findMany({
      where: { conversationId },
      orderBy: { createdAt: 'desc' },
      take: 30,
    }).then(msgs => msgs.reverse());

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

  /**
   * 批量删除对话（列表页「全选并删除」用）。
   *
   * 与单条删除同一条语义：**软删除**（isActive = false），消息记录保留。
   * 因此这里也复用 update 语义，而不是 delete —— 两者混用会让
   * 「删掉的对话还能不能被审计/回溯」这个问题变得没法回答。
   *
   * ⚠️ 必须是**真事务**，不能图省事写成"先 updateMany 再比 count"：
   * updateMany 是单条 SQL，写完就落库了；等它返回再发现 count 对不上、
   * 抛 404 时，那些合法 id 已经被删掉了 —— 用户看到"删除失败"，
   * 刷新却发现少了几条（本次实测确实如此：5 条里被删掉 1 条）。
   * 这里用一段交互式事务把「先查后改」包起来，任一步抛出都整体回滚。
   *
   * 传入的 id 里只要有**不属于当前用户或已被删除**的，就整体拒绝：
   * 静默跳过会让前端以为「全删掉了」，而列表里还留着几条，用户只能反复点。
   */
  async deleteConversations(conversationIds: string[], userId: string) {
    // 去重：前端可能因为重复点击把同一个 id 发两遍
    const uniqueIds = Array.from(new Set(conversationIds));

    if (uniqueIds.length === 0) {
      throw new AppError('请选择要删除的对话', 400);
    }

    const deleted = await prisma.$transaction(async (tx) => {
      // 条件里同时带 userId，越权 id 会自然落空，不会误删别人的会话
      const owned = await tx.conversation.findMany({
        where: { id: { in: uniqueIds }, userId, isActive: true },
        select: { id: true },
      });

      if (owned.length !== uniqueIds.length) {
        throw new AppError('部分对话不存在或已被删除，请刷新后重试', 404);
      }

      const { count } = await tx.conversation.updateMany({
        where: { id: { in: uniqueIds }, userId, isActive: true },
        data: { isActive: false },
      });

      return count;
    });

    return { deleted };
  }

  /**
   * 为一次高危对话落危机记录，并通知真人（管理员 + 咨询师）。
   *
   * 修复前的三个问题（都在这里收口）：
   *   1. 传进去的 `matchedKeywords` 写死为空数组 → trigger 恒为空串、severity 恒为 HIGH，
   *      管理端的「危急案例」计数因此永远是 0；
   *   2. 只有 `risk_level === 'crisis'` 才落记录，算法层明确要求升级的 `high` 被丢掉；
   *   3. 落库之后不通知任何人 —— 「检测到了」和「有人知道」之间是断的。
   *
   * ⚠️ 本方法吞掉自身异常：通知失败不能让患者收不到那条热线回复。
   *    但绝不静默——失败必须留日志，否则「危机已处置」就是假的。
   */
  private async raiseCrisisForUser(
    userId: string,
    userContent: string,
    smartResult: SmartChatResponse,
    isCrisis: boolean,
  ): Promise<void> {
    try {
      const { matchedKeywords } = crisisService.detectCrisis(userContent);

      // crisis 或命中两个以上关键词 = 危急；其余高危 = HIGH
      const severity: CrisisSeverity =
        isCrisis || matchedKeywords.length >= 2 ? 'CRITICAL' : 'HIGH';

      const trigger = matchedKeywords.length > 0
        ? `对话危机关键词：${matchedKeywords.join('、')}`
        : `对话风险分级：${smartResult.risk_level}（算法证据：${(smartResult.evidence?.[0] || '未提供').slice(0, 80)}）`;

      const description =
        `用户在 AI 对话中触发${isCrisis ? '危机信号' : '高风险升级'}，` +
        `算法风险等级 ${smartResult.risk_level}，安全审计${smartResult.audit_passed ? '通过' : '未通过'}。`;

      const result = await crisisService.raise(userId, {
        severity,
        trigger,
        triggerType: 'chat_keyword',
        sourceText: userContent.slice(0, 2000),
        description,
      });

      console.log(
        `[Crisis] 用户 ${userId} ${result.created ? '新建' : '合并'}危机记录 ` +
        `${result.recordId}（${severity}），已通知值班人员`,
      );
    } catch (err) {
      console.error('[Crisis] 危机记录/通知失败:', err);
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
        // ⚠️ 这条行也会成为「最新画像」，被 /profile/psychological 直接读出。
        // 修复前 metadata 里写的是 source: 'daily_text_analysis'（**字符串**），
        // 而心理画像页的「评估依据」按对象读 dataSources.selfReport —— 字符串会让那块退化成
        // "还没有量表自评结果"，看起来像 20 题自查没被采纳。
        // 现在统一写对象口径（与 updateProfile / getLatestProfile / emotional-state 一致）。
        const latestSelfCheck = await prisma.assessment.findFirst({
          where: { userId, type: 'MOOD20' },
          orderBy: { createdAt: 'desc' },
        });
        const moodRecordCount = await prisma.moodRecord.count({ where: { userId } });
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
              source: {
                selfReport: latestSelfCheck
                  ? {
                      type: 'MOOD20',
                      assessmentId: latestSelfCheck.id,
                      totalScore: latestSelfCheck.totalScore,
                      level: latestSelfCheck.level,
                      completedAt: latestSelfCheck.createdAt,
                      weight: 0.6,
                      // 本行分数来自当日文字分析、尚未按量表融合，标注清楚避免误读
                      appliedToScores: false,
                    }
                  : null,
                moodRecords: moodRecordCount,
                sleepRecords: realSleepScore > 0 ? 1 : 0,
                note: '本行由当日文字分析生成：数值来自对话/情绪文本；焦虑/抑郁/压力的量表融合口径见 /profile/update 生成的画像行。',
              },
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
