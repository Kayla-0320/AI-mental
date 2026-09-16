import prisma from '../config/database';
import aiService from './ai.service';
import { AppError } from '../middlewares/errorHandler';

class ProfileService {
  // 获取用户心理画像
  async getProfile(patientProfileId: string) {
    const profiles = await prisma.psychologicalProfile.findMany({
      where: { patientProfileId },
      orderBy: { createdAt: 'desc' },
      take: 10,
    });

    return profiles;
  }

  // 获取最新心理画像
  async getLatestProfile(patientProfileId: string) {
    const profile = await prisma.psychologicalProfile.findFirst({
      where: { patientProfileId },
      orderBy: { createdAt: 'desc' },
    });

    return profile;
  }

  // 基于对话和测评更新心理画像
  async updateProfile(userId: string) {
    const patientProfile = await prisma.patientProfile.findUnique({
      where: { userId },
    });

    if (!patientProfile) {
      throw new AppError('患者档案不存在', 404);
    }

    // 获取最近的对话摘要
    const recentConversations = await prisma.conversation.findMany({
      where: { userId },
      orderBy: { updatedAt: 'desc' },
      take: 5,
      include: {
        messages: {
          where: { role: 'user' },
          orderBy: { createdAt: 'desc' },
          take: 10,
        },
      },
    });

    const conversationSummaries = recentConversations.map(c =>
      c.messages.map(m => m.content).join(' ')
    );

    // 获取最近的测评结果
    const recentAssessments = await prisma.assessment.findMany({
      where: { userId },
      orderBy: { createdAt: 'desc' },
      take: 3,
    });

    // 获取最近的心情记录
    const recentMoods = await prisma.moodRecord.findMany({
      where: { userId },
      orderBy: { recordedAt: 'desc' },
      take: 14,
    });

    // 计算情绪维度
    const moodScores = this.calculateMoodScores(recentMoods);

    // 调用 AI 生成画像报告
    let report = '';
    try {
      report = await aiService.generateProfileReport(
        conversationSummaries,
        { assessments: recentAssessments, moods: recentMoods }
      );
    } catch {
      report = 'AI画像生成暂时不可用，请稍后重试。';
    }

    // 创建新的心理画像记录
    const newProfile = await prisma.psychologicalProfile.create({
      data: {
        patientProfileId: patientProfile.id,
        anxiety: moodScores.anxiety,
        depression: moodScores.depression,
        stress: moodScores.stress,
        sleepQuality: moodScores.sleepQuality,
        socialActivity: moodScores.socialActivity,
        emotionalStability: moodScores.emotionalStability,
        overallScore: moodScores.overall,
        riskLevel: this.calculateRiskLevel(moodScores),
        report,
        assessedBy: 'AI',
      },
    });

    // 更新患者风险等级
    await prisma.patientProfile.update({
      where: { userId },
      data: { riskLevel: newProfile.riskLevel },
    });

    return newProfile;
  }

  // 获取情绪趋势数据
  async getMoodTrend(userId: string, days: number = 30) {
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);

    const moodRecords = await prisma.moodRecord.findMany({
      where: {
        userId,
        recordedAt: { gte: startDate },
      },
      orderBy: { recordedAt: 'asc' },
    });

    const profiles = await prisma.psychologicalProfile.findMany({
      where: {
        patientProfile: { userId },
      },
      orderBy: { createdAt: 'asc' },
      take: 30,
    });

    return { moodRecords, profiles };
  }

  // 获取心理测评列表
  async getAssessments(userId: string) {
    return prisma.assessment.findMany({
      where: { userId },
      orderBy: { createdAt: 'desc' },
    });
  }

  // 提交心理测评
  async submitAssessment(userId: string, type: string, title: string, answers: any) {
    const result = this.calculateAssessmentResult(type, answers);

    return prisma.assessment.create({
      data: {
        userId,
        type,
        title,
        answers: JSON.stringify(answers),
        result: JSON.stringify(result),
        totalScore: result.totalScore,
        level: result.level,
        suggestion: result.suggestion,
      },
    });
  }

  // 记录心情
  async recordMood(userId: string, mood: string, score: number, note?: string, tags?: string[]) {
    return prisma.moodRecord.create({
      data: {
        userId,
        mood,
        score,
        note,
        tags: tags ? JSON.stringify(tags) : undefined,
      },
    });
  }

  // 获取实时情绪状态（基于今天的所有文字）
  async getTodayEmotionalState(userId: string) {
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    // 收集今天的对话消息
    const todayMessages = await prisma.message.findMany({
      where: {
        userId,
        role: 'user',
        createdAt: { gte: today },
      },
      orderBy: { createdAt: 'asc' },
    });

    // 收集今天的心情记录
    const todayMoods = await prisma.moodRecord.findMany({
      where: { userId, recordedAt: { gte: today } },
      orderBy: { recordedAt: 'asc' },
    });

    // 获取最新的动态画像
    const patientProfile = await prisma.patientProfile.findUnique({ where: { userId } });
    const latestDynamicProfile = patientProfile
      ? await prisma.psychologicalProfile.findFirst({
          where: { patientProfileId: patientProfile.id, assessedBy: 'AI_DYNAMIC' },
          orderBy: { createdAt: 'desc' },
        })
      : null;

    const texts = [
      ...todayMessages.map((m: any) => m.content),
      ...todayMoods.filter((m: any) => m.note).map((m: any) => `[心情:${m.mood}] ${m.note}`),
    ];

    // 如果有动态画像，检查是否需要重新分析
    if (latestDynamicProfile) {
      const profileAge = Date.now() - new Date(latestDynamicProfile.createdAt).getTime();
      const isStale = profileAge > 5 * 60 * 1000; // 超过 5 分钟视为过期
      const hasNewMessages = todayMessages.length > 0;

      // 如果画像过期且有新消息，重新分析
      if (isStale && hasNewMessages) {
        const dynamicProfile = await aiService.generateDynamicProfile(texts);
        // 保存新画像
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
              }),
            },
          });
        }
        return {
          hasData: true,
          messageCount: todayMessages.length,
          moodCount: todayMoods.length,
          profile: dynamicProfile,
          updatedAt: new Date(),
        };
      }

      const meta = latestDynamicProfile.metadata ? JSON.parse(latestDynamicProfile.metadata) : {};
      return {
        hasData: texts.length > 0,
        messageCount: todayMessages.length,
        moodCount: todayMoods.length,
        profile: {
          anxiety: latestDynamicProfile.anxiety,
          depression: latestDynamicProfile.depression,
          stress: latestDynamicProfile.stress,
          sleepQuality: latestDynamicProfile.sleepQuality,
          socialActivity: latestDynamicProfile.socialActivity,
          emotionalStability: latestDynamicProfile.emotionalStability,
          overallScore: latestDynamicProfile.overallScore,
          riskLevel: latestDynamicProfile.riskLevel,
          emotionalSummary: latestDynamicProfile.report,
          dominantEmotions: meta.dominantEmotions || [],
          emotionalTrend: meta.emotionalTrend || '平稳',
        },
        updatedAt: latestDynamicProfile.createdAt,
      };
    }

    // 如果没有动态画像但有文字，实时分析
    if (texts.length > 0) {
      const dynamicProfile = await aiService.generateDynamicProfile(texts);
      return {
        hasData: true,
        messageCount: todayMessages.length,
        moodCount: todayMoods.length,
        profile: dynamicProfile,
        updatedAt: new Date(),
      };
    }

    return { hasData: false, messageCount: 0, moodCount: 0, profile: null, updatedAt: null };
  }

  // 生成应对策略
  async getCopingStrategies(userId: string) {
    // 获取最新画像
    const patientProfile = await prisma.patientProfile.findUnique({ where: { userId } });
    if (!patientProfile) throw new AppError('患者档案不存在', 404);

    const latestProfile = await prisma.psychologicalProfile.findFirst({
      where: { patientProfileId: patientProfile.id },
      orderBy: { createdAt: 'desc' },
    });

    if (!latestProfile) {
      return {
        immediateActions: ['深呼吸放松 5 分钟', '喝一杯温水', '到窗边看看远处'],
        shortTermStrategies: ['尝试一次冥想练习', '记录今日心情', '与信任的人聊聊'],
        longTermSuggestions: ['建立规律作息', '培养运动习惯', '学习情绪管理技巧'],
        recommendedModules: ['冥想引导', '呼吸练习', '情绪日记'],
        encouragingMessage: '每一个小小的改变都值得肯定，你正在变得更好。',
      };
    }

    const meta = latestProfile.metadata ? JSON.parse(latestProfile.metadata) : {};

    const strategies = await aiService.generateCopingStrategies({
      primaryEmotion: meta.dominantEmotions?.[0] || '平静',
      dominantEmotions: meta.dominantEmotions || [],
      anxiety: latestProfile.anxiety,
      depression: latestProfile.depression,
      stress: latestProfile.stress,
      emotionalSummary: latestProfile.report || '',
    });

    return strategies;
  }

  // 获取心情记录
  async getMoodRecords(userId: string, startDate?: Date, endDate?: Date) {
    const where: any = { userId };
    if (startDate || endDate) {
      where.recordedAt = {};
      if (startDate) where.recordedAt.gte = startDate;
      if (endDate) where.recordedAt.lte = endDate;
    }

    return prisma.moodRecord.findMany({
      where,
      orderBy: { recordedAt: 'desc' },
    });
  }

  // 计算心情分数
  private calculateMoodScores(moodRecords: any[]) {
    if (moodRecords.length === 0) {
      return { anxiety: 50, depression: 50, stress: 50, sleepQuality: 50, socialActivity: 50, emotionalStability: 50, overall: 50 };
    }

    const avgScore = moodRecords.reduce((sum, r) => sum + r.score, 0) / moodRecords.length;
    const moodMap: Record<string, number> = {
      happy: 90, calm: 75, excited: 80, sad: 30, anxious: 25, angry: 20, tired: 35,
    };

    const moodValues = moodRecords.map(r => moodMap[r.mood] || 50);
    const avgMood = moodValues.reduce((a, b) => a + b, 0) / moodValues.length;

    return {
      anxiety: Math.max(0, Math.min(100, 100 - avgMood)),
      depression: Math.max(0, Math.min(100, 100 - avgScore * 10)),
      stress: Math.max(0, Math.min(100, 100 - avgMood * 0.8)),
      sleepQuality: Math.max(0, Math.min(100, avgScore * 10)),
      socialActivity: Math.max(0, Math.min(100, avgMood)),
      emotionalStability: Math.max(0, Math.min(100, avgMood)),
      overall: Math.round(avgMood),
    };
  }

  // 计算风险等级
  private calculateRiskLevel(scores: any): string {
    if (scores.depression > 80 || scores.anxiety > 80) return 'HIGH';
    if (scores.depression > 60 || scores.anxiety > 60) return 'MEDIUM';
    return 'LOW';
  }

  // 计算测评结果
  private calculateAssessmentResult(type: string, answers: any): any {
    const scores = Object.values(answers) as number[];
    const totalScore = scores.reduce((sum: number, s: number) => sum + (s || 0), 0);

    let level = 'normal';
    let suggestion = '';

    switch (type) {
      case 'PHQ9':
        if (totalScore >= 20) { level = 'severe'; suggestion = '重度抑郁倾向，强烈建议寻求专业心理咨询。'; }
        else if (totalScore >= 15) { level = 'moderate'; suggestion = '中重度抑郁倾向，建议预约心理咨询师进行详细评估。'; }
        else if (totalScore >= 10) { level = 'moderate'; suggestion = '中度抑郁倾向，建议关注情绪变化，尝试平台疗愈功能。'; }
        else if (totalScore >= 5) { level = 'mild'; suggestion = '轻度抑郁倾向，建议保持规律作息，使用平台疗愈室进行自我调节。'; }
        else { suggestion = '目前状态良好，继续保持积极的生活方式。'; }
        break;
      case 'GAD7':
        if (totalScore >= 15) { level = 'severe'; suggestion = '重度焦虑倾向，建议尽快寻求专业帮助。'; }
        else if (totalScore >= 10) { level = 'moderate'; suggestion = '中度焦虑倾向，建议尝试正念冥想和呼吸练习。'; }
        else if (totalScore >= 5) { level = 'mild'; suggestion = '轻度焦虑倾向，建议进行放松训练和规律运动。'; }
        else { suggestion = '焦虑水平正常，继续保持良好的心态。'; }
        break;
      default:
        if (totalScore > 70) { level = 'severe'; suggestion = '评估得分偏高，建议寻求专业咨询。'; }
        else if (totalScore > 50) { level = 'moderate'; suggestion = '评估得分中等，建议关注相关方面。'; }
        else if (totalScore > 30) { level = 'mild'; suggestion = '评估得分偏低，建议持续自我观察。'; }
        else { suggestion = '评估结果正常，请继续保持。'; }
    }

    return { totalScore, level, suggestion };
  }

  // 获取来访者最新焦虑数据（咨询师用）
  async getPatientAnxiety(patientUserId: string) {
    // 获取最近的焦虑检测记录
    const recentAnxiety = await prisma.moodRecord.findFirst({
      where: {
        userId: patientUserId,
        note: { contains: '焦虑检测' },
      },
      orderBy: { recordedAt: 'desc' },
    });

    if (!recentAnxiety) return null;

    const tags = recentAnxiety.tags ? JSON.parse(recentAnxiety.tags) : {};
    return {
      anxietyIndex: tags.anxietyIndex || 0,
      level: tags.level || 'low',
      keyboardAnxiety: tags.keyboard || 0,
      blinkAnxiety: tags.blink || 0,
      context: recentAnxiety.note,
      recordedAt: recentAnxiety.recordedAt,
    };
  }

  // 获取患者咨询室完整数据（咨询师用）
  async getPatientConsultationData(patientUserId: string) {
    // 获取患者基本信息
    const patientProfile = await prisma.patientProfile.findUnique({
      where: { userId: patientUserId },
      include: { user: { select: { id: true, nickname: true, email: true } } },
    });

    if (!patientProfile) {
      throw new AppError('患者档案不存在', 404);
    }

    // 获取最新心理画像
    const latestProfile = await prisma.psychologicalProfile.findFirst({
      where: { patientProfileId: patientProfile.id },
      orderBy: { createdAt: 'desc' },
    });

    // 获取最新焦虑数据
    const recentAnxiety = await prisma.moodRecord.findFirst({
      where: {
        userId: patientUserId,
        note: { contains: '焦虑检测' },
      },
      orderBy: { recordedAt: 'desc' },
    });

    // 获取最近心情记录
    const recentMoods = await prisma.moodRecord.findMany({
      where: { userId: patientUserId },
      orderBy: { recordedAt: 'desc' },
      take: 10,
    });

    // 获取最近对话
    const recentConversations = await prisma.conversation.findMany({
      where: { userId: patientUserId },
      orderBy: { updatedAt: 'desc' },
      take: 5,
      select: {
        id: true,
        title: true,
        messageCount: true,
        updatedAt: true,
      },
    });

    const anxietyData = recentAnxiety?.tags ? JSON.parse(recentAnxiety.tags) : {};
    const profileMeta = latestProfile?.metadata ? JSON.parse(latestProfile.metadata) : {};

    return {
      patient: {
        id: patientProfile.id,
        userId: patientProfile.userId,
        nickname: patientProfile.user?.nickname,
        email: patientProfile.user?.email,
        riskLevel: patientProfile.riskLevel,
      },
      profile: latestProfile ? {
        anxiety: latestProfile.anxiety,
        depression: latestProfile.depression,
        stress: latestProfile.stress,
        sleepQuality: latestProfile.sleepQuality,
        socialActivity: latestProfile.socialActivity,
        emotionalStability: latestProfile.emotionalStability,
        overallScore: latestProfile.overallScore,
        riskLevel: latestProfile.riskLevel,
        report: latestProfile.report,
        dominantEmotions: profileMeta.dominantEmotions || [],
        emotionalTrend: profileMeta.emotionalTrend || '平稳',
        updatedAt: latestProfile.createdAt,
      } : null,
      anxiety: {
        anxietyIndex: anxietyData.anxietyIndex || 0,
        level: anxietyData.level || 'low',
        keyboardAnxiety: anxietyData.keyboard || 0,
        blinkAnxiety: anxietyData.blink || 0,
        context: recentAnxiety?.note || '',
        recordedAt: recentAnxiety?.recordedAt || null,
      },
      recentMoods: recentMoods.map((m: any) => ({
        mood: m.mood,
        score: m.score,
        note: m.note,
        recordedAt: m.recordedAt,
      })),
      recentConversations,
    };
  }
}

export default new ProfileService();
