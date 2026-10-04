import prisma from '../config/database';
import aiService from './ai.service';
import { AppError } from '../middlewares/errorHandler';

class TreatmentService {
  // 创建治疗规划
  async createPlan(userId: string, title: string, description?: string) {
    // 获取患者心理画像
    const patientProfile = await prisma.patientProfile.findUnique({
      where: { userId },
      include: {
        psychologicalProfiles: {
          orderBy: { createdAt: 'desc' },
          take: 1,
        },
      },
    });

    // 获取咨询历史
    const conversations = await prisma.conversation.findMany({
      where: { userId },
      orderBy: { updatedAt: 'desc' },
      take: 5,
      select: { title: true, messageCount: true },
    });

    let goals = '[]';
    let phases = '[]';

    // AI 生成治疗方案
    try {
      const aiResult = await aiService.generateTreatmentPlan(
        patientProfile?.psychologicalProfiles[0] || {},
        { conversations }
      );
    
      try {
        // 提取 JSON（处理 markdown 代码块）
        let jsonStr = aiResult;
        const jsonMatch = aiResult.match(/```json\s*([\s\S]*?)```/);
        if (jsonMatch) {
          jsonStr = jsonMatch[1].trim();
        } else {
          // 尝试找到第一个 { 和最后一个 }
          const start = aiResult.indexOf('{');
          const end = aiResult.lastIndexOf('}');
          if (start !== -1 && end !== -1) {
            jsonStr = aiResult.substring(start, end + 1);
          }
        }
    
        const parsed = JSON.parse(jsonStr);
        // 如果返回的是完整方案对象
        if (parsed.goals || parsed.phases || parsed.dailyTasks) {
          goals = JSON.stringify({
            goals: parsed.goals || { shortTerm: [], longTerm: [] },
            phases: parsed.phases || [],
            dailyTasks: parsed.dailyTasks || [],
            assessmentNodes: parsed.assessmentNodes || [],
          });
          phases = JSON.stringify(parsed.phases || []);
        } else {
          goals = JSON.stringify([{ title: '情绪稳定', description: aiResult }]);
        }
      } catch {
        goals = JSON.stringify([{ title: '情绪稳定', description: aiResult }]);
      }
    } catch {
      goals = JSON.stringify([
        { title: '情绪调节', description: '学习情绪管理技巧' },
        { title: '认知重建', description: '识别和改变消极思维模式' },
      ]);
      phases = JSON.stringify([
        { name: '初始评估', duration: '1-2 周', tasks: ['心理测评', '建立咨询关系'] },
        { name: '深入探索', duration: '3-4 周', tasks: ['探索问题根源', '情绪表达'] },
        { name: '技能训练', duration: '4-6 周', tasks: ['认知重构', '行为激活'] },
        { name: '巩固维持', duration: '2-4 周', tasks: ['回顾进展', '预防复发'] },
      ]);
    }

    return prisma.treatmentPlan.create({
      data: {
        userId,
        title,
        description,
        goals,
        phases,
        status: 'DRAFT',
      },
    });
  }

  // 获取用户治疗规划
  async getPlans(userId: string) {
    return prisma.treatmentPlan.findMany({
      where: { userId },
      orderBy: { updatedAt: 'desc' },
      include: {
        tasks: {
          orderBy: { createdAt: 'asc' },
        },
      },
    });
  }

  // 获取规划详情
  async getPlanDetail(planId: string, userId: string) {
    const plan = await prisma.treatmentPlan.findFirst({
      where: { id: planId, userId },
      include: {
        tasks: {
          orderBy: { createdAt: 'asc' },
        },
      },
    });

    if (!plan) {
      throw new AppError('治疗规划不存在', 404);
    }

    return plan;
  }

  // 更新规划状态
  async updatePlanStatus(planId: string, userId: string, status: string) {
    const plan = await prisma.treatmentPlan.findFirst({
      where: { id: planId, userId },
    });

    if (!plan) {
      throw new AppError('治疗规划不存在', 404);
    }

    const updateData: any = { status };
    if (status === 'ACTIVE' && !plan.startDate) {
      updateData.startDate = new Date();
    }
    if (status === 'COMPLETED') {
      updateData.endDate = new Date();
      updateData.progress = 100;
    }

    return prisma.treatmentPlan.update({
      where: { id: planId },
      data: updateData,
    });
  }

  // 添加治疗任务
  async addTask(planId: string, userId: string, data: {
    title: string;
    description?: string;
    type: string;
    frequency: string;
    scheduledAt?: Date;
  }) {
    const plan = await prisma.treatmentPlan.findFirst({
      where: { id: planId, userId },
    });

    if (!plan) {
      throw new AppError('治疗规划不存在', 404);
    }

    return prisma.treatmentTask.create({
      data: {
        planId,
        ...data,
      },
    });
  }

  // 更新任务状态
  async updateTaskStatus(taskId: string, userId: string, status: string, notes?: string) {
    const task = await prisma.treatmentTask.findFirst({
      where: { id: taskId, plan: { userId } },
    });

    if (!task) {
      throw new AppError('任务不存在', 404);
    }

    const updateData: any = { status };
    if (status === 'COMPLETED') updateData.completedAt = new Date();
    if (notes) updateData.notes = notes;

    await prisma.treatmentTask.update({
      where: { id: taskId },
      data: updateData,
    });

    // 更新规划进度
    await this.updatePlanProgress(task.planId);

    return prisma.treatmentTask.findUnique({ where: { id: taskId } });
  }

  // 更新规划进度
  private async updatePlanProgress(planId: string) {
    const tasks = await prisma.treatmentTask.findMany({
      where: { planId },
    });

    const completed = tasks.filter(t => t.status === 'COMPLETED').length;
    const progress = tasks.length > 0 ? Math.round((completed / tasks.length) * 100) : 0;

    await prisma.treatmentPlan.update({
      where: { id: planId },
      data: { progress },
    });
  }
}

export default new TreatmentService();
