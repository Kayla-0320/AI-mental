import prisma from '../config/database';
import { AppError } from '../middlewares/errorHandler';

/**
 * 数据授权（同意管理）+ 数据控制中心。
 *
 * 设计原则（对应平台对用户的三个承诺）：
 *   1. **按用途分别授权**：粒度是 (dataType, purpose)，不是一个「同意全部」开关。
 *   2. **可查、可导出、可删除**：导出与删除都走真实数据库操作，返回可核对的回执。
 *   3. **删除不等于隐藏**：删除是 `DELETE`，不是把按钮藏起来或加个 `isDeleted` 标记。
 *
 * ⚠️ 与 `algorithm/privacy/consent/` 的关系：
 *    那里的数据结构与危机例外条款是权威口径，但它的存储是**进程内 dict**
 *    （`_consent_store = {}`，重启即丢），且没有任何调用方。本文件把同一套口径
 *    落到数据库，成为唯一的授权真相来源；算法侧如需读取，应通过这里的接口，
 *    而不是各自维护一份。
 */

/** 授权目录项：决定前端「数据控制面板」里能看到哪些开关 */
export interface ConsentCatalogItem {
  dataType: string;
  purpose: string;
  label: string;
  description: string;
  group: string;
  /** 默认是否授权（用户首次进入时的初始状态） */
  defaultGranted: boolean;
  /** 服务必需项：不可关闭，但必须写明为什么 */
  required?: boolean;
  /** 不可关闭时的依据说明 */
  requiredReason?: string;
}

/**
 * 授权目录。
 *
 * 默认只开「服务运行所必需」的三项；感知类（语音/面部/行为/睡眠）默认关闭 ——
 * 用户主动打开才采集，而不是默认打开再让用户去关。
 */
export const CONSENT_CATALOG: ConsentCatalogItem[] = [
  {
    dataType: 'text',
    purpose: 'emotion_analysis',
    label: '文字对话的情绪分析',
    description: '分析你发送的文字，用于情绪状态跟踪与回复风格适配。',
    group: '基础服务',
    defaultGranted: true,
  },
  {
    dataType: 'assessment',
    purpose: 'risk_assessment',
    label: '量表结果用于风险评估',
    description: 'PHQ-9 / GAD-7 等自评结果用于识别需要支持的时段。量表是自评筛查，不是诊断。',
    group: '基础服务',
    defaultGranted: true,
  },
  {
    dataType: 'risk_assessment',
    purpose: 'crisis_intervention',
    label: '危机情况下的必要干预',
    description:
      '当你出现明确的自伤/自杀风险信号时，平台需要通知值班人员并提供必要的风险信息。' +
      '这一项不可关闭——依据《个人信息保护法》第十三条第四款「紧急情况下为保护自然人的' +
      '生命健康和财产安全所必需」。我们只在必要时使用，并记录使用痕迹。',
    group: '基础服务',
    defaultGranted: true,
    required: true,
    requiredReason: '生命健康保护的法律例外条款，不依赖同意',
  },
  {
    dataType: 'emotion_result',
    purpose: 'consultant_view',
    label: '咨询师查看情绪与风险评估',
    description: '在你预约或接受真人咨询时，让咨询师看到情绪趋势与风险等级，便于接续沟通。',
    group: '共享范围',
    defaultGranted: true,
  },
  {
    dataType: 'voice',
    purpose: 'emotion_analysis',
    label: '语音声学特征分析',
    description:
      '从你的语音中提取音高、能量等特征用于情绪分析。开启后，语音会以音频形式发送到平台服务端做识别' +
      '（不是纯本地处理）。关闭后语音通话与语音输入不可用。',
    group: '感知数据',
    defaultGranted: false,
  },
  {
    dataType: 'facial',
    purpose: 'emotion_analysis',
    label: '摄像头表情分析',
    description: '使用摄像头在本机提取面部关键点用于表情分析，原始画面不落库。关闭后表情感知不可用。',
    group: '感知数据',
    defaultGranted: false,
  },
  {
    dataType: 'behavior',
    purpose: 'emotion_analysis',
    label: '键盘节奏与输入行为',
    description: '使用打字节奏等行为信号辅助判断状态。仅记录统计特征，不记录你输入的内容。',
    group: '感知数据',
    defaultGranted: false,
  },
  {
    dataType: 'sleep',
    purpose: 'emotion_analysis',
    label: '睡眠记录分析',
    description: '使用你手动记录的睡眠数据参与状态分析。平台没有夜间传感器，睡眠只能来自你的记录。',
    group: '感知数据',
    defaultGranted: false,
  },
  {
    dataType: 'emotion_result',
    purpose: 'research',
    label: '参与心理健康研究',
    description: '允许将去标识化后的数据用于心理健康研究。可以随时关闭。',
    group: '其他用途',
    defaultGranted: false,
  },
  {
    dataType: 'emotion_result',
    purpose: 'federated_learning',
    label: '参与联邦学习',
    description: '在不离开本机的前提下参与模型改进。可以随时关闭。',
    group: '其他用途',
    defaultGranted: false,
  },
];

export type EraseScope = 'content' | 'account';

/** 允许用户设置的保留期限（天），与前端选项保持一致 */
export const ALLOWED_RETENTION_DAYS = [30, 90, 180, 365];

export class PrivacyService {
  // ============================================================
  // 授权
  // ============================================================

  /** 首次访问时补齐默认授权项（幂等：已存在的行不覆盖） */
  async ensureDefaults(userId: string): Promise<void> {
    const existing = await prisma.consentRecord.findMany({
      where: { userId },
      select: { dataType: true, purpose: true },
    });
    const have = new Set(existing.map((r) => `${r.dataType}::${r.purpose}`));

    const missing = CONSENT_CATALOG.filter(
      (item) => !have.has(`${item.dataType}::${item.purpose}`),
    );
    if (missing.length === 0) return;

    await prisma.consentRecord.createMany({
      data: missing.map((item) => ({
        userId,
        dataType: item.dataType,
        purpose: item.purpose,
        granted: item.defaultGranted,
        grantedAt: item.defaultGranted ? new Date() : null,
      })),
      skipDuplicates: true,
    });
  }

  /** 读取完整授权状态：目录 + 当前状态 */
  async getConsentState(userId: string) {
    await this.ensureDefaults(userId);

    const rows = await prisma.consentRecord.findMany({ where: { userId } });
    const byKey = new Map(rows.map((r) => [`${r.dataType}::${r.purpose}`, r]));

    const items = CONSENT_CATALOG.map((item) => {
      const row = byKey.get(`${item.dataType}::${item.purpose}`);
      return {
        dataType: item.dataType,
        purpose: item.purpose,
        label: item.label,
        description: item.description,
        group: item.group,
        required: item.required === true,
        requiredReason: item.requiredReason || null,
        // 必需项永远显示为已授权，且前端不可切换
        granted: item.required ? true : (row?.granted ?? item.defaultGranted),
        grantedAt: item.required ? null : (row?.grantedAt ?? null),
        revokedAt: item.required ? null : (row?.revokedAt ?? null),
      };
    });

    const granted = items.filter((i) => i.granted);
    return {
      items,
      summary: {
        total: items.length,
        granted: granted.length,
        // 「感知数据」专指语音/面部/行为/睡眠四类
        sensingGranted: granted.filter((i) => i.group === '感知数据').length,
        sensingTotal: items.filter((i) => i.group === '感知数据').length,
        hasSensingConsent: granted.some((i) => i.group === '感知数据'),
      },
    };
  }

  /** 写入授权变更。必需项忽略（不能关） */
  async setConsent(
    userId: string,
    updates: { dataType: string; purpose: string; granted: boolean }[],
  ) {
    await this.ensureDefaults(userId);
    const now = new Date();

    for (const update of updates) {
      const catalogItem = CONSENT_CATALOG.find(
        (c) => c.dataType === update.dataType && c.purpose === update.purpose,
      );
      if (!catalogItem) {
        throw new AppError(`未知的授权项：${update.dataType}/${update.purpose}`, 400);
      }
      if (catalogItem.required) continue; // 必需项不允许被关闭

      await prisma.consentRecord.upsert({
        where: {
          userId_dataType_purpose: {
            userId,
            dataType: update.dataType,
            purpose: update.purpose,
          },
        },
        create: {
          userId,
          dataType: update.dataType,
          purpose: update.purpose,
          granted: update.granted,
          grantedAt: update.granted ? now : null,
          revokedAt: update.granted ? null : now,
        },
        update: {
          granted: update.granted,
          // 撤销不删行：保留撤销时间才能回答「什么时候撤的」
          grantedAt: update.granted ? now : undefined,
          revokedAt: update.granted ? null : now,
        },
      });
    }

    return this.getConsentState(userId);
  }

  /**
   * 判断某项授权是否有效。
   * 采集侧（服务端）在写入感知数据前应调用它，而不是假设前端已经拦住了。
   */
  async hasConsent(userId: string, dataType: string, purpose: string): Promise<boolean> {
    const item = CONSENT_CATALOG.find(
      (c) => c.dataType === dataType && c.purpose === purpose,
    );
    if (item?.required) return true;

    const row = await prisma.consentRecord.findUnique({
      where: { userId_dataType_purpose: { userId, dataType, purpose } },
    });
    if (!row) return item?.defaultGranted === true;
    return row.granted;
  }

  // ============================================================
  // 保留期限
  // ============================================================

  async getRetention(userId: string): Promise<number> {
    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: { dataRetentionDays: true },
    });
    return user?.dataRetentionDays ?? 90;
  }

  async setRetention(userId: string, days: number): Promise<number> {
    if (!ALLOWED_RETENTION_DAYS.includes(days)) {
      throw new AppError(`保留期限只能是 ${ALLOWED_RETENTION_DAYS.join('/')} 天`, 400);
    }
    await prisma.user.update({ where: { id: userId }, data: { dataRetentionDays: days } });
    return days;
  }

  // ============================================================
  // 数据概览（「你们到底存了我什么」）
  // ============================================================

  async dataSummary(userId: string) {
    const [
      conversations, messages, assessments, moodRecords, moodCheckIns,
      sleepRecords, healingSessions, treatmentPlans, communityPosts,
      crisisRecords, consents,
    ] = await Promise.all([
      prisma.conversation.count({ where: { userId } }),
      prisma.message.count({ where: { userId } }),
      prisma.assessment.count({ where: { userId } }),
      prisma.moodRecord.count({ where: { userId } }),
      prisma.moodCheckIn.count({ where: { userId } }),
      prisma.sleepRecord.count({ where: { userId } }),
      prisma.healingSession.count({ where: { userId } }),
      prisma.treatmentPlan.count({ where: { userId } }),
      prisma.communityPost.count({ where: { userId } }),
      prisma.crisisRecord.count({ where: { userId } }),
      prisma.consentRecord.count({ where: { userId } }),
    ]);

    return {
      retentionDays: await this.getRetention(userId),
      items: [
        { key: 'conversations', label: 'AI 对话', count: conversations },
        { key: 'messages', label: '对话消息', count: messages },
        { key: 'assessments', label: '量表测评记录', count: assessments },
        { key: 'moodRecords', label: '心情记录', count: moodRecords },
        { key: 'moodCheckIns', label: '每日打卡', count: moodCheckIns },
        { key: 'sleepRecords', label: '睡眠记录', count: sleepRecords },
        { key: 'healingSessions', label: '疗愈练习记录', count: healingSessions },
        { key: 'treatmentPlans', label: '成长/治疗计划', count: treatmentPlans },
        { key: 'communityPosts', label: '社区发帖', count: communityPosts },
        { key: 'crisisRecords', label: '危机干预记录', count: crisisRecords },
        { key: 'consents', label: '授权记录', count: consents },
      ],
      notes: {
        crisisRecords: '危机干预记录出于安全与复盘需要保存，不随「保留期限」自动清理。',
        rawMedia: '平台不保存原始音视频文件；语音在服务端转写后只保留文本与声学特征统计量。',
      },
    };
  }

  // ============================================================
  // 导出
  // ============================================================

  async buildExport(userId: string) {
    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: {
        id: true, email: true, phone: true, nickname: true, role: true,
        createdAt: true, dataRetentionDays: true,
      },
    });
    if (!user) throw new AppError('用户不存在', 404);

    const withUser = { where: { userId } };
    const [
      patientProfile, conversations, messages, assessments, moodRecords,
      moodCheckIns, sleepRecords, healingSessions, treatmentPlans,
      communityPosts, communityComments, emergencyContacts, consents,
      healthReports, crisisRecords, notifications,
    ] = await Promise.all([
      prisma.patientProfile.findUnique({ where: { userId } }),
      prisma.conversation.findMany(withUser),
      prisma.message.findMany(withUser),
      prisma.assessment.findMany(withUser),
      prisma.moodRecord.findMany(withUser),
      prisma.moodCheckIn.findMany(withUser),
      prisma.sleepRecord.findMany(withUser),
      prisma.healingSession.findMany(withUser),
      prisma.treatmentPlan.findMany({ where: { userId }, include: { tasks: true } }),
      prisma.communityPost.findMany(withUser),
      prisma.communityComment.findMany(withUser),
      prisma.emergencyContact.findMany(withUser),
      prisma.consentRecord.findMany(withUser),
      prisma.healthReport.findMany(withUser),
      prisma.crisisRecord.findMany(withUser),
      prisma.notification.findMany(withUser),
    ]);

    // 心理画像挂在 PatientProfile 下（没有 userId 字段）
    const psychologicalProfiles = patientProfile
      ? await prisma.psychologicalProfile.findMany({
          where: { patientProfileId: patientProfile.id },
        })
      : [];

    return {
      exportedAt: new Date().toISOString(),
      exportVersion: '1.0',
      notice:
        '这是平台当前为你的账号保存的全部个人数据。' +
        '不含登录凭据（密码哈希不导出），也不含原始音视频文件（平台不保存原始音视频）。',
      account: user,
      consent: consents,
      profile: patientProfile,
      psychologicalProfiles,
      conversations,
      messages,
      assessments,
      mood: { records: moodRecords, checkIns: moodCheckIns },
      sleep: sleepRecords,
      healing: healingSessions,
      growth: { treatmentPlans },
      community: { posts: communityPosts, comments: communityComments },
      emergencyContacts,
      healthReports,
      crisisRecords,
      notifications,
    };
  }

  // ============================================================
  // 删除
  // ============================================================

  /**
   * 删除用户数据。
   *
   * - `scope: 'content'` 清空全部个人内容，保留账号（用户可继续使用）
   * - `scope: 'account'` 删除整个账号（数据库级联删除全部关联记录）
   *
   * ⚠️ 这不是软删除：用的是 `delete` / `deleteMany`，删完查不到、导不出。
   * ⚠️ 危机干预记录在 `content` 范围内也会被删除（用户有删除权），
   *    但会先返回删除前的计数回执，便于核对。
   */
  async erase(userId: string, scope: EraseScope) {
    const summaryBefore = await this.dataSummary(userId);
    const deleted: Record<string, number> = {};

    if (scope === 'account') {
      await prisma.user.delete({ where: { id: userId } });
      for (const item of summaryBefore.items) deleted[item.key] = item.count;
      return {
        scope,
        deleted,
        accountDeleted: true,
        message: '账号及其全部关联数据已从数据库中删除。',
      };
    }

    // scope === 'content'：按外键顺序清空内容，保留账号与授权记录
    await prisma.$transaction(async (tx) => {
      deleted.messages = (await tx.message.deleteMany({ where: { userId } })).count;
      deleted.conversations = (await tx.conversation.deleteMany({ where: { userId } })).count;
      deleted.assessments = (await tx.assessment.deleteMany({ where: { userId } })).count;
      deleted.moodRecords = (await tx.moodRecord.deleteMany({ where: { userId } })).count;
      deleted.moodCheckIns = (await tx.moodCheckIn.deleteMany({ where: { userId } })).count;
      deleted.sleepRecords = (await tx.sleepRecord.deleteMany({ where: { userId } })).count;
      deleted.healingSessions = (await tx.healingSession.deleteMany({ where: { userId } })).count;
      deleted.healthReports = (await tx.healthReport.deleteMany({ where: { userId } })).count;
      deleted.communityComments = (await tx.communityComment.deleteMany({ where: { userId } })).count;
      deleted.communityPosts = (await tx.communityPost.deleteMany({ where: { userId } })).count;
      deleted.notifications = (await tx.notification.deleteMany({ where: { userId } })).count;

      // 计划与任务：先删任务（无 userId 字段，靠 planId 关联）
      const plans = await tx.treatmentPlan.findMany({
        where: { userId },
        select: { id: true },
      });
      if (plans.length > 0) {
        deleted.treatmentTasks = (await tx.treatmentTask.deleteMany({
          where: { planId: { in: plans.map((p: { id: string }) => p.id) } },
        })).count;
      }
      deleted.treatmentPlans = (await tx.treatmentPlan.deleteMany({ where: { userId } })).count;
      deleted.crisisRecords = (await tx.crisisRecord.deleteMany({ where: { userId } })).count;
      // 心理画像没有 userId，只能经 PatientProfile 定位
      const patientProfile = await tx.patientProfile.findUnique({
        where: { userId },
        select: { id: true },
      });
      if (patientProfile) {
        deleted.psychologicalProfiles = (await tx.psychologicalProfile.deleteMany({
          where: { patientProfileId: patientProfile.id },
        })).count;
      }
      // 注意：不删除 UserFeedback —— 它记录的是用户主动提交的投诉/建议，属于用户与平台之间的
      // 服务记录，且用户可能正在等待处理，删除会让其反馈凭空消失。导出里同样不含它。
    });

    return {
      scope,
      deleted,
      accountDeleted: false,
      message: '个人内容数据已从数据库中删除；账号与授权记录保留。',
    };
  }

  // ============================================================
  // 保留期限清理（由 app.ts 的定时任务调用）
  // ============================================================

  /**
   * 按每个用户的 `dataRetentionDays` 硬删除过期记录。
   *
   * 清理范围：对话消息、心情/打卡/睡眠/疗愈记录、通知。
   * **不清理**：量表测评、危机干预记录、授权记录 —— 这些属于临床与安全留痕。
   */
  async purgeExpired(now: Date = new Date()) {
    const users = await prisma.user.findMany({ select: { id: true, dataRetentionDays: true } });
    const totals: Record<string, number> = {};
    let touched = 0;

    for (const user of users) {
      const days = user.dataRetentionDays || 90;
      const cutoff = new Date(now.getTime() - days * 24 * 60 * 60 * 1000);
      const where = { userId: user.id, createdAt: { lt: cutoff } };

      const [messages, moods, checkIns, sleep, healing, notifications] = await Promise.all([
        prisma.message.deleteMany({ where }),
        prisma.moodRecord.deleteMany({ where: { userId: user.id, createdAt: { lt: cutoff } } }),
        prisma.moodCheckIn.deleteMany({ where: { userId: user.id, createdAt: { lt: cutoff } } }),
        prisma.sleepRecord.deleteMany({ where: { userId: user.id, createdAt: { lt: cutoff } } }),
        prisma.healingSession.deleteMany({ where: { userId: user.id, createdAt: { lt: cutoff } } }),
        prisma.notification.deleteMany({ where }),
      ]);

      const n = messages.count + moods.count + checkIns.count + sleep.count + healing.count + notifications.count;
      if (n > 0) {
        touched += 1;
        totals.messages = (totals.messages || 0) + messages.count;
        totals.moodRecords = (totals.moodRecords || 0) + moods.count;
        totals.moodCheckIns = (totals.moodCheckIns || 0) + checkIns.count;
        totals.sleepRecords = (totals.sleepRecords || 0) + sleep.count;
        totals.healingSessions = (totals.healingSessions || 0) + healing.count;
        totals.notifications = (totals.notifications || 0) + notifications.count;
      }
    }

    return { usersTouched: touched, deleted: totals };
  }
}

const privacyService = new PrivacyService();
export default privacyService;
