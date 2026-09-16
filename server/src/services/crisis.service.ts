import prisma from '../config/database';
import { AppError } from '../middlewares/errorHandler';

export class CrisisService {
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
