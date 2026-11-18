import prisma from '../config/database';
import { AppError } from '../middlewares/errorHandler';

/** 危机记录严重度：与 schema 注释一致 */
export type CrisisSeverity = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';

export interface RaiseCrisisInput {
  severity: CrisisSeverity;
  trigger: string;
  triggerType: 'chat_keyword' | 'assessment' | 'community' | 'manual';
  sourceText?: string;
  description?: string;
}

export interface RaiseCrisisResult {
  recordId: string;
  created: boolean;
  severityUpgraded: boolean;
}

const SEVERITY_RANK: Record<string, number> = { LOW: 0, MEDIUM: 1, HIGH: 2, CRITICAL: 3 };

export class CrisisService {
  /**
   * 危机记录「无人接收」的超时阈值（15 分钟）。
   * 超过这个时长仍是 OPEN，说明没有任何真人接手，需要升级提醒。
   */
  readonly ACK_TIMEOUT_MS = 15 * 60 * 1000;

  /** 同一用户的危机在多久内视为同一次事件（避免长对话里每轮都插一条） */
  readonly DEDUPE_WINDOW_MS = 6 * 60 * 60 * 1000;

  // 检测文本中的危机关键词
  private crisisKeywords = [
    '自杀', '自残', '想死', '不想活', '结束生命', '活不下去',
    '死了算了', '没有意义', '世界没有我会更好', '解脱',
    '伤害自己', '割腕', '跳楼', '吃药', '上吊',
  ];

  detectCrisis(text: string): { isCrisis: boolean; matchedKeywords: string[] } {
    const matched = this.crisisKeywords.filter(k => text.includes(k));
    return { isCrisis: matched.length > 0, matchedKeywords: matched };
  }

  /**
   * 由命中的关键词推断严重度。
   *
   * 单一关键词 = HIGH；命中两个及以上 = CRITICAL。
   * ⚠️ 这是关键词规则，不是临床分级；仅供「先处理谁」排序使用。
   */
  severityFromKeywords(keywords: string[]): CrisisSeverity {
    return keywords.length >= 2 ? 'CRITICAL' : 'HIGH';
  }

  /**
   * 落一条危机记录，并通知真人。
   *
   * 三个必须一起成立，否则「危机已处置」是假的：
   *   1. 记录本身带真实的触发原因与严重度（不是空字符串）
   *   2. 有真人收到通知（不是只写一行数据库）
   *   3. 同一用户的连续危机合并成一次事件（否则管理员会被淹没，真实信号被淹没在噪声里）
   *
   * @returns 记录 ID 与本次是否为新建 / 是否发生了严重度升级
   */
  async raise(userId: string, input: RaiseCrisisInput): Promise<RaiseCrisisResult> {
    const now = new Date();
    const windowStart = new Date(now.getTime() - this.DEDUPE_WINDOW_MS);

    // 1. 去重：同一用户最近一次尚未结束的危机，视为同一次事件
    const existing = await prisma.crisisRecord.findFirst({
      where: {
        userId,
        status: { in: ['OPEN', 'IN_PROGRESS'] },
        detectedAt: { gte: windowStart },
      },
      orderBy: { detectedAt: 'desc' },
    });

    if (existing) {
      const upgraded =
        (SEVERITY_RANK[input.severity] ?? 0) > (SEVERITY_RANK[existing.severity] ?? 0);

      await prisma.crisisRecord.update({
        where: { id: existing.id },
        data: {
          severity: upgraded ? input.severity : existing.severity,
          // 把新的触发证据追加进去，而不是丢掉——复盘时要看得到完整过程
          sourceText: input.sourceText || existing.sourceText,
          description: input.description
            ? `${existing.description ? existing.description + ' / ' : ''}${input.description}`
            : existing.description,
          updatedAt: now,
        },
      });

      // 只有严重度升级才再次打扰真人，避免刷屏
      if (upgraded) {
        await this.notifyStaff(existing.id, userId, input.severity, input.trigger, true);
      }
      return { recordId: existing.id, created: false, severityUpgraded: upgraded };
    }

    // 2. 新建
    const record = await prisma.crisisRecord.create({
      data: {
        userId,
        severity: input.severity,
        status: 'OPEN',
        trigger: input.trigger,
        triggerType: input.triggerType,
        sourceText: input.sourceText || null,
        description: input.description || null,
      },
    });

    // 3. 通知真人（管理员 + 咨询师）
    await this.notifyStaff(record.id, userId, input.severity, input.trigger, false);

    return { recordId: record.id, created: true, severityUpgraded: true };
  }

  /**
   * 通知所有管理员与咨询师。
   *
   * 这是「检测到了」到「有人知道」之间唯一的一步。
   * 通知失败不能阻断危机记录落库，但必须留下错误日志——静默失败等于没通知。
   */
  async notifyStaff(
    recordId: string,
    patientId: string,
    severity: CrisisSeverity,
    trigger: string,
    isEscalation: boolean,
  ): Promise<number> {
    try {
      const staff = await prisma.user.findMany({
        where: { role: { in: ['ADMIN', 'CONSULTANT'] }, isActive: true },
        select: { id: true, role: true },
      });
      if (staff.length === 0) {
        console.error('[CrisisNotify] 没有可通知的管理员或咨询师，危机无人接收:', recordId);
        return 0;
      }

      const badge = severity === 'CRITICAL' ? '🚨 危急' : '⚠️ 高风险';
      const prefix = isEscalation ? '【升级】' : '';

      await prisma.notification.createMany({
        data: staff.map((s: { id: string; role: string }) => ({
          userId: s.id,
          type: 'crisis_alert',
          title: `${prefix}${badge} 危机信号待接收`,
          content:
            `一位来访者的对话触发危机信号（${severity}）。\n` +
            `触发原因：${trigger}\n` +
            `记录 ID：${recordId.slice(-8)}\n` +
            '请在 15 分钟内确认接收；超时未接收将再次提醒。',
          link: s.role === 'ADMIN' ? '/admin/crisis' : '/consultant/crisis',
        })),
      });
      return staff.length;
    } catch (err) {
      console.error('[CrisisNotify] 通知失败:', err);
      return 0;
    }
  }

  /**
   * 超时兜底：把「仍无人接收」的危机再次推给管理员。
   *
   * 由 app.ts 的定时任务调用（见 startCrisisWatchdog）。
   * 幂等依靠记录的 metadata.overdueNotifiedAt，重复执行不会重复打扰。
   *
   * @returns 本轮被升级提醒的记录数
   */
  async escalateOverdue(now: Date = new Date()): Promise<number> {
    const deadline = new Date(now.getTime() - this.ACK_TIMEOUT_MS);

    const overdue = await prisma.crisisRecord.findMany({
      where: { status: 'OPEN', detectedAt: { lt: deadline } },
      orderBy: { detectedAt: 'asc' },
      take: 50,
    });

    let escalated = 0;
    for (const record of overdue) {
      let meta: Record<string, unknown> = {};
      try {
        meta = record.metadata ? JSON.parse(record.metadata) : {};
      } catch {
        meta = {};
      }
      if (meta.overdueNotifiedAt) continue; // 已经升级提醒过

      const admins = await prisma.user.findMany({
        where: { role: 'ADMIN', isActive: true },
        select: { id: true },
      });
      if (admins.length === 0) continue;

      const minutes = Math.round((now.getTime() - new Date(record.detectedAt).getTime()) / 60000);
      await prisma.notification.createMany({
        data: admins.map((a: { id: string }) => ({
          userId: a.id,
          type: 'crisis_alert',
          title: `🔴 危机超时未接收（已 ${minutes} 分钟）`,
          content:
            `记录 ${record.id.slice(-8)}（${record.severity}）自发现起已 ${minutes} 分钟无人确认接收。\n` +
            `触发原因：${record.trigger}\n` +
            '请立即介入或改派给在线咨询师。',
          link: '/admin/crisis',
        })),
      });

      await prisma.crisisRecord.update({
        where: { id: record.id },
        data: {
          metadata: JSON.stringify({ ...meta, overdueNotifiedAt: now.toISOString() }),
          updatedAt: now,
        },
      });
      escalated += 1;
    }

    return escalated;
  }

  // 心理援助热线
  getHotlines() {
    return [
      { name: '北京心理危机研究与干预中心', phone: '010-82951332', hours: '24小时' },
      { name: '全国希望24小时热线', phone: '400-161-9995', hours: '24小时' },
      { name: '生命热线', phone: '400-821-1215', hours: '24小时' },
      { name: '青少年心理咨询热线', phone: '12355', hours: '24小时' },
      { name: '全国心理援助热线', phone: '400-680-6101', hours: '24小时' },
    ];
  }

  // 获取用户紧急联系人
  async getEmergencyContacts(userId: string) {
    return prisma.emergencyContact.findMany({
      where: { userId },
      orderBy: { isPrimary: 'desc' },
    });
  }

  // 添加紧急联系人
  async addContact(userId: string, data: { name: string; relation: string; phone: string; isPrimary?: boolean }) {
    if (data.isPrimary) {
      await prisma.emergencyContact.updateMany({
        where: { userId, isPrimary: true },
        data: { isPrimary: false },
      });
    }
    return prisma.emergencyContact.create({ data: { ...data, userId } });
  }

  // 更新紧急联系人
  async updateContact(contactId: string, userId: string, data: { name?: string; relation?: string; phone?: string; isPrimary?: boolean }) {
    const contact = await prisma.emergencyContact.findFirst({ where: { id: contactId, userId } });
    if (!contact) throw new AppError('联系人不存在', 404);
    if (data.isPrimary) {
      await prisma.emergencyContact.updateMany({
        where: { userId, isPrimary: true, id: { not: contactId } },
        data: { isPrimary: false },
      });
    }
    return prisma.emergencyContact.update({ where: { id: contactId }, data });
  }

  // 删除紧急联系人
  async deleteContact(contactId: string, userId: string) {
    const contact = await prisma.emergencyContact.findFirst({ where: { id: contactId, userId } });
    if (!contact) throw new AppError('联系人不存在', 404);
    await prisma.emergencyContact.delete({ where: { id: contactId } });
    return { success: true };
  }
}

/** 全局单例：危机记录落库 + 真人通知的唯一入口 */
const crisisService = new CrisisService();
export default crisisService;

