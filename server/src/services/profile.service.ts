import prisma from '../config/database';
import aiService from './ai.service';
import achievementService from './achievement.service';
import { AppError } from '../middlewares/errorHandler';

/**
 * 首页「心理状态自查 · 20 题」(MOOD20) 各维度的自评结果。
 * 每题 0-3 分、每维 4 题，因此单项满分 12；同时给出换算成 0-100 的分数，
 * 以及该次自查的日期（用于判断"是否近期"）。
 */
interface SelfReportScores {
  mood: number;        // 0-100：情绪低落/兴趣减退
  anxiety: number;     // 0-100：紧张担忧
  stress: number;      // 0-100：压力与应对
  sleep: number;       // 0-100：睡眠困扰（**不参与画像**，仅用于前端说明）
  social: number;      // 0-100：人际支持不足
  overall: number;     // 0-100：总分换算
  totalScore: number;  // 0-60 原始总分
  level: string;       // normal / mild / moderate / severe
  completedAt: Date;
}

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

  // 获取最新心理画像（前端心理画像页走的就是这条：/profile/psychological）
  async getLatestProfile(patientProfileId: string) {
    const profile = await prisma.psychologicalProfile.findFirst({
      where: { patientProfileId },
      orderBy: { createdAt: 'desc' },
    });

    if (!profile) return profile;

    // ⚠️ 必须在这里补「评估依据」：前端进页面读的是这条记录，
    // 只让 updateProfile 带这些字段的话，刷新后「评估依据」会退回"还没有量表自评"，
    // 用户会以为量表结果根本没被采纳。
    const owner = await prisma.patientProfile.findUnique({
      where: { id: patientProfileId },
      select: { userId: true },
    });
    if (!owner) return profile;

    const view = await this.buildDataSourceView(owner.userId, profile);
    return { ...profile, ...view };
  }

  /**
   * 给画像记录补上「评估依据」字段，供前端展示"分数是怎么算出来的"。
   *
   * ⚠️ 必须同时用于 `getLatestProfile` 和 `updateProfile` 两条路径：
   * 前端进页面读的是 getLatestProfile（直接返回库里的历史行），只有 update 带这些字段的话，
   * 「评估依据」在刷新后就会变成"还没有量表自评结果"——看起来像量表数据没被采纳。
   */
  private async buildDataSourceView(userId: string, profile: any | null) {
    const latestSelfCheck = await prisma.assessment.findFirst({
      where: { userId, type: 'MOOD20' },
      orderBy: { createdAt: 'desc' },
    });
    const selfReport = this.calculateSelfReportScores(latestSelfCheck);

    let moodParts: any = {};
    try {
      const meta = profile?.metadata ? JSON.parse(profile.metadata) : {};
      if (meta?.moodParts) moodParts = meta.moodParts;
    } catch {
      moodParts = {};
    }

    return {
      dataSources: {
        selfReport: selfReport
          ? {
              type: 'MOOD20',
              assessmentId: latestSelfCheck!.id,
              totalScore: selfReport.totalScore,
              level: selfReport.level,
              completedAt: selfReport.completedAt,
              weight: 0.6,
            }
          : null,
        moodRecords: moodParts?.moodRecords ?? 0,
        // 有手动睡眠记录时该维度不是 0；具体条数在画像页另有睡眠卡片展示
        sleepRecords: profile?.sleepQuality ? 1 : 0,
        note: '焦虑/抑郁/压力 = 量表自评 60% + 心情打卡 40%；社交/情绪稳定来自心情打卡；'
          + '睡眠只来自手动睡眠记录，量表睡眠项仅作说明、不参与评分。',
      },
      selfReport,
      moodParts,
    };
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

    // 获取最近的一次 20 题自查（MOOD20）
    const latestSelfCheck = await prisma.assessment.findFirst({
      where: { userId, type: 'MOOD20' },
      orderBy: { createdAt: 'desc' },
    });

    // 获取最近的心情记录
    const recentMoods = await prisma.moodRecord.findMany({
      where: { userId },
      orderBy: { recordedAt: 'desc' },
      take: 14,
    });

    // 计算情绪维度（量表自评与心情打卡按权重融合，见 calculateMoodScores）
    const selfReport = this.calculateSelfReportScores(latestSelfCheck);
    const moodScores = this.calculateMoodScores(recentMoods, selfReport);

    // 睡眠维度只能来自用户真实记录（无夜间传感器，禁止由情绪分推算）
    const realSleepScore = await this.calculateRealSleepQuality(userId);

    // 画像的数据来源：落库到 metadata，并用同一个 helper 回给前端，
    // 保证「进页面读到的」(getProfile) 与「刚算完的」(updateProfile) 口径完全一致
    const dataSources = {
      moodRecords: recentMoods.length,
      selfReport: selfReport
        ? {
            type: 'MOOD20',
            assessmentId: latestSelfCheck!.id,
            totalScore: selfReport.totalScore,
            level: selfReport.level,
            completedAt: selfReport.completedAt,
            weight: 0.6,
          }
        : null,
      sleepRecords: realSleepScore === null ? 0 : 1,
      note: '焦虑/抑郁/压力 = 量表自评 60% + 心情打卡 40%；社交/情绪稳定来自心情打卡；'
        + '睡眠只来自手动睡眠记录，量表睡眠项仅作说明、不参与评分。',
    };

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
        // 0 表示"用户尚未记录睡眠"，前端据此显示"未采集"而不是把 0 当成睡眠质量
        sleepQuality: realSleepScore ?? 0,
        socialActivity: moodScores.socialActivity,
        emotionalStability: moodScores.emotionalStability,
        overallScore: moodScores.overall,
        riskLevel: this.calculateRiskLevel(moodScores),
        report,
        assessedBy: 'AI',
        // 落库留痕：这次画像用了哪些数据、各占什么权重（moodParts 供 getProfile 复用展示）
        metadata: JSON.stringify({
          source: dataSources,
          moodParts: {
            anxiety: Math.round(moodScores.moodAnxiety ?? 0),
            depression: Math.round(moodScores.moodDepression ?? 0),
            stress: Math.round(moodScores.moodStress ?? 0),
            socialActivity: Math.round(moodScores.moodSocialActivity ?? 0),
            emotionalStability: Math.round(moodScores.moodEmotionalStability ?? 0),
            moodRecords: recentMoods.length,
          },
        }),
      },
    });

    // 更新患者风险等级
    await prisma.patientProfile.update({
      where: { userId },
      data: { riskLevel: newProfile.riskLevel },
    });

    // 与 getProfile 同一口径补齐「评估依据」，前端两条路径拿到的字段一致
    const view = await this.buildDataSourceView(userId, newProfile);
    return {
      ...newProfile,
      ...view,
      // 本次刚算出来的自评分（与 view.selfReport 同源，保留显式字段便于前端直接用）
      selfReport: selfReport ?? view.selfReport,
    };
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

    const assessment = await prisma.assessment.create({
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

    // 测评徽章在此解锁：旧实现没有任何测评侧的触发点（spec G1）
    const newlyUnlocked = await achievementService.evaluateAchievements(userId);

    return { ...assessment, newlyUnlocked };
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
        // 睡眠不参与 AI 文字分析：单独取真实记录（无记录则记 0 = 未采集）
        const realSleepScore = await this.calculateRealSleepQuality(userId);
        // 保存新画像
        if (patientProfile) {
          // ⚠️ 这条 AI_DYNAMIC 行也会成为「最新画像」，被 /profile/psychological 直接读出。
          // 若它的分数只来自 AI 文本分析、metadata 里又没有 source/moodParts，
          // 心理画像页的「评估依据」就会退化成"自评 X → 画像 Y（没有打卡对照、打卡 0 条）"，
          // 看起来像量表与打卡都没生效。所以这里统一按同一套口径重算并写入同一份元数据。
          const recentForScores = await prisma.moodRecord.findMany({
            where: { userId },
            orderBy: { recordedAt: 'desc' },
            take: 14,
          });
          const latestSelfCheckForScores = await prisma.assessment.findFirst({
            where: { userId, type: 'MOOD20' },
            orderBy: { createdAt: 'desc' },
          });
          const selfReportForScores = this.calculateSelfReportScores(latestSelfCheckForScores);
          // 有量表自评时按同一套融合口径落库，否则沿用 AI 文本分析的分数（保持既有行为）
          const fused = this.calculateMoodScores(recentForScores, selfReportForScores);
          const scores = selfReportForScores
            ? {
                anxiety: fused.anxiety,
                depression: fused.depression,
                stress: fused.stress,
                socialActivity: fused.socialActivity,
                emotionalStability: fused.emotionalStability,
                overallScore: fused.overall,
              }
            : {
                anxiety: dynamicProfile.anxiety,
                depression: dynamicProfile.depression,
                stress: dynamicProfile.stress,
                socialActivity: dynamicProfile.socialActivity,
                emotionalStability: dynamicProfile.emotionalStability,
                overallScore: dynamicProfile.overallScore,
              };

          const created = await prisma.psychologicalProfile.create({
            data: {
              patientProfileId: patientProfile.id,
              anxiety: scores.anxiety,
              depression: scores.depression,
              stress: scores.stress,
              sleepQuality: realSleepScore ?? 0,
              socialActivity: scores.socialActivity,
              emotionalStability: scores.emotionalStability,
              overallScore: scores.overallScore,
              riskLevel: dynamicProfile.riskLevel,
              report: dynamicProfile.emotionalSummary,
              assessedBy: 'AI_DYNAMIC',
              metadata: JSON.stringify({
                dominantEmotions: dynamicProfile.dominantEmotions,
                emotionalTrend: dynamicProfile.emotionalTrend,
                // 保留 AI 文本分析的原始读数，便于事后核对口径
                aiTextScores: {
                  anxiety: dynamicProfile.anxiety,
                  depression: dynamicProfile.depression,
                  stress: dynamicProfile.stress,
                  socialActivity: dynamicProfile.socialActivity,
                  emotionalStability: dynamicProfile.emotionalStability,
                  overallScore: dynamicProfile.overallScore,
                },
                // source + moodParts 与 updateProfile/getLatestProfile 同口径。
                // ⚠️ 必须直接用上面算出来的 fused（尤其是未融合时它带着原始打卡口径），
                //    不能靠 buildDataSourceView 去库里读这条刚建的 metadata —— 那时还没写进去。
                source: {
                  selfReport: selfReportForScores && latestSelfCheckForScores
                    ? {
                        type: 'MOOD20',
                        assessmentId: latestSelfCheckForScores.id,
                        totalScore: selfReportForScores.totalScore,
                        level: selfReportForScores.level,
                        completedAt: selfReportForScores.completedAt,
                        weight: 0.6,
                      }
                    : null,
                  moodRecords: recentForScores.length,
                  sleepRecords: realSleepScore === null ? 0 : 1,
                  note: '焦虑/抑郁/压力 = 量表自评 60% + 心情打卡 40%；社交/情绪稳定来自心情打卡；'
                    + '睡眠只来自手动睡眠记录，量表睡眠项仅作说明、不参与评分。',
                },
                moodParts: {
                  anxiety: Math.round(fused.moodAnxiety),
                  depression: Math.round(fused.moodDepression),
                  stress: Math.round(fused.moodStress),
                  socialActivity: Math.round(fused.moodSocialActivity),
                  emotionalStability: Math.round(fused.moodEmotionalStability),
                  moodRecords: recentForScores.length,
                },
              }),
            },
          });
        }
        return {
          hasData: true,
          messageCount: todayMessages.length,
          moodCount: todayMoods.length,
          profile: { ...dynamicProfile, sleepQuality: realSleepScore ?? 0, sleepQualitySource: realSleepScore === null ? 'none' : 'manual_record' },
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
          sleepQualitySource: latestDynamicProfile.sleepQuality > 0 ? 'manual_record' : 'none',
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
      const realSleepScore = await this.calculateRealSleepQuality(userId);
      return {
        hasData: true,
        messageCount: todayMessages.length,
        moodCount: todayMoods.length,
        profile: { ...dynamicProfile, sleepQuality: realSleepScore ?? 0, sleepQualitySource: realSleepScore === null ? 'none' : 'manual_record' },
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
  private calculateMoodScores(moodRecords: any[], selfReport: SelfReportScores | null = null) {
    // ⚠️ 这里不再返回 sleepQuality：
    // 由心情自评分数反推"睡眠质量"是编造数据，睡眠只能来自用户手动记录。
    const base = moodRecords.length === 0
      ? { anxiety: 50, depression: 50, stress: 50, socialActivity: 50, emotionalStability: 50, overall: 50 }
      : (() => {
          const avgScore = moodRecords.reduce((sum, r) => sum + r.score, 0) / moodRecords.length;
          const moodMap: Record<string, number> = {
            happy: 90, calm: 75, excited: 80, sad: 30, anxious: 25, angry: 20, tired: 35,
          };

          const moodValues = moodRecords.map(r => moodMap[r.mood] || 50);
          const avgMood = moodValues.reduce((a, b) => a + b, 0) / moodValues.length;

          // 这三个维度直接落库（DB 字段是 Int）：先取整，
          // 避免"有无量表自评"两条路径分别落 78 和 78.4 造成数值抖动
          return {
            anxiety: Math.round(Math.max(0, Math.min(100, 100 - avgMood))),
            depression: Math.round(Math.max(0, Math.min(100, 100 - avgScore * 10))),
            stress: Math.round(Math.max(0, Math.min(100, 100 - avgMood * 0.8))),
            // 这两项同样直接落库：先取整，避免存 37.5 而各页显示 37/38 不一致
            socialActivity: Math.round(Math.max(0, Math.min(100, avgMood))),
            emotionalStability: Math.round(Math.max(0, Math.min(100, avgMood))),
            overall: Math.round(avgMood),
          };
        })();

    // 没有量表自评（或数据不可解析）→ 完全沿用原来的心情打卡口径，行为不变
    if (!selfReport) {
      return {
        ...base,
        // 与落库字段一致取整：避免"回给前端的原始口径"带小数、和存储值对不上
        moodAnxiety: Math.round(base.anxiety),
        moodDepression: Math.round(base.depression),
        moodStress: Math.round(base.stress),
        moodSocialActivity: Math.round(base.socialActivity),
        moodEmotionalStability: Math.round(base.emotionalStability),
      };
    }

    // 量表自评权重：20 题自评比单日心情打卡更能代表近两周状态，但不是唯一依据
    const w = 0.6;
    const blend = (a: number, b: number) => Math.round(Math.max(0, Math.min(100, a * w + b * (1 - w))));

    // 只用维度明确对应的三项做融合：
    //   抑郁/兴趣 ← 自评 mood；焦虑 ← 自评 anxiety；压力 ← 自评 stress
    // 「社交」「情绪稳定」没有与 20 题量表一一对应的自评维度（量表第 5 部分测的是
    // "人际支持不足"，取反不等于"社交活跃度"），继续由心情打卡给出，不硬凑；
    // 睡眠仍然只认手动睡眠记录。
    const anxiety = blend(selfReport.anxiety, base.anxiety);
    const depression = blend(selfReport.mood, base.depression);
    const stress = blend(selfReport.stress, base.stress);

    return {
      anxiety,
      depression,
      stress,
      socialActivity: base.socialActivity,
      emotionalStability: base.emotionalStability,
      // 综合分与前端雷达图口径一致：六个维度等权平均（睡眠未采集时按 50 中性计）
      overall: Math.round((anxiety + depression + stress + base.socialActivity + base.emotionalStability + 50) / 6),
      // 融合前的原始口径，仅供前端展示对照（取整，与落库字段同口径）
      moodAnxiety: Math.round(base.anxiety),
      moodDepression: Math.round(base.depression),
      moodStress: Math.round(base.stress),
      moodSocialActivity: Math.round(base.socialActivity),
      moodEmotionalStability: Math.round(base.emotionalStability),
    };
  }

  /**
   * 解析最近一次 20 题自查（MOOD20）的答案，得到各维度自评分。
   *
   * 维度划分必须与客户端 `client/src/components/HomeQuestionnaire.tsx` 的
   * `QUESTIONS` 一致（每题 0-3 分、每维 4 题，满分 12）：
   *   0-3   情绪与兴趣 → mood
   *   4-7   紧张与担忧 → anxiety
   *   8-11  睡眠与精力 → sleep
   *   12-15 压力与应对 → stress
   *   16-19 人际与支持 → social
   * 客户端改题目顺序/维度时必须同步改这里，否则分数会串到别的维度。
   *
   * 拿不到答案或格式不对时返回 null，调用方回退到纯心情打卡口径（不猜、不编）。
   */
  private calculateSelfReportScores(assessment: any | null): SelfReportScores | null {
    if (!assessment) return null;

    let answers: any;
    try {
      answers = JSON.parse(assessment.answers || '{}');
    } catch {
      return null;
    }
    if (!answers || typeof answers !== 'object') return null;

    // 每题 0-3 分；缺失或非数字按 0 计（与量表提交时的 0 分语义一致）
    const scores: number[] = [];
    for (let i = 0; i < 20; i++) {
      const raw = Number(answers[i]);
      scores.push(Number.isFinite(raw) ? Math.max(0, Math.min(3, raw)) : 0);
    }
    const sum = (from: number, to: number) =>
      scores.slice(from, to).reduce((a, b) => a + b, 0);
    // 12 分制 → 0-100
    const to100 = (raw: number) => Math.round((raw / 12) * 100);

    const totalScore = scores.reduce((a, b) => a + b, 0);
    return {
      mood: to100(sum(0, 4)),
      anxiety: to100(sum(4, 8)),
      sleep: to100(sum(8, 12)),
      stress: to100(sum(12, 16)),
      social: to100(sum(16, 20)),
      overall: to100(totalScore / 5),
      totalScore,
      level: assessment.level || 'normal',
      completedAt: assessment.createdAt,
    };
  }

  /**
   * 由用户真实睡眠记录换算 0-100 的睡眠质量分。
   * 没有任何记录时返回 null —— 表示"未采集"，绝不用其它信号代替。
   */
  private async calculateRealSleepQuality(userId: string): Promise<number | null> {
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - 30);
    const records = await prisma.sleepRecord.findMany({
      where: { userId, recordDate: { gte: startDate } },
      orderBy: { recordDate: 'desc' },
      take: 30,
    });
    if (records.length === 0) return null;
    // 主观质量 1-10 → 0-100
    const avgQuality = records.reduce((s: number, r: any) => s + (r.quality || 0), 0) / records.length;
    return Math.max(0, Math.min(100, Math.round(avgQuality * 10)));
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
      // 20 题心理状态自查（首页「心理状态自查」板块，满分 60）。
      // ⚠️ 分档必须与 client/src/components/HomeQuestionnaire.tsx 的 SCORE_BANDS 同值同步：
      //    同一份答案在结果页与「心理画像」历史记录里不允许出现两种结论。
      //    分档按「近两周困扰持续时间」取宽档（12 分起算轻度），避免把偶发情绪误报成需要干预的状态。
      // ⚠️ 此处仅新增计分分支，未改动 PHQ9/GAD7 的分档与任何升级（escalation）条件。
      case 'MOOD20':
        if (totalScore >= 30) {
          level = 'severe';
          suggestion = '近两周的困扰程度偏高（仅供参考，不是诊断）。建议尽快预约一次面对面咨询，让专业的人陪你一起梳理。';
        } else if (totalScore >= 20) {
          level = 'moderate';
          suggestion = '有不少困扰正在影响你。建议先做几次疗愈室练习，并考虑和公益咨询师聊一聊。';
        } else if (totalScore >= 12) {
          level = 'mild';
          suggestion = '最近状态有些起伏，多数人都会遇到。保持规律作息，试试疗愈室的呼吸练习。';
        } else {
          // 显式写 normal：不依赖函数开头的初始值，以后改初始化也不会静默影响这一档
          level = 'normal';
          suggestion = '最近整体状态比较平稳，继续保持现在的作息和节奏就好。';
        }
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
    // 获取患者基本信息，没有档案则自动创建
    let patientProfile = await prisma.patientProfile.findUnique({
      where: { userId: patientUserId },
      include: { user: { select: { id: true, nickname: true, email: true } } },
    });

    if (!patientProfile) {
      // 尝试从 User 表获取基本信息
      const user = await prisma.user.findUnique({
        where: { id: patientUserId },
        select: { id: true, nickname: true, email: true },
      });
      if (!user) throw new AppError('患者不存在', 404);
      // 返回基础数据（不强制要求 patientProfile）
      return {
        patient: { id: null, userId: user.id, nickname: user.nickname, email: user.email, riskLevel: 'LOW' },
        profile: null,
        anxiety: { anxietyIndex: 0, level: 'low', keyboardAnxiety: 0, blinkAnxiety: 0, context: '', recordedAt: null },
        recentMoods: [],
        recentConversations: [],
      };
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
        // 0 表示患者未手动记录过睡眠（平台无法自动检测）
        sleepQuality: latestProfile.sleepQuality,
        sleepQualitySource: latestProfile.sleepQuality > 0 ? 'manual_record' : 'none',
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
