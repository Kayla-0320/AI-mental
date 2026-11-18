import { Router } from 'express';
import expertController from '../controllers/expert.controller';
import { authenticate, authorize } from '../middlewares/auth';
import prisma from '../config/database';

/**
 * 管理端专用接口（仅 ADMIN）。
 *
 * 为什么单独开一组而不是把管理端字段混进 /api/expert/consultants：
 *   那条是**面向患者**的公开列表，where 写死了 `isAvailable && user.isActive`，
 *   待审核/已拒绝的咨询师不会出现，管理端也就无从审核。
 *   这里给管理员返回全量并附推导出的 status。
 *
 * 前端调用位置：
 *   client/src/pages/admin/AdminConsultants.tsx
 */
const router = Router();

router.use(authenticate, authorize('ADMIN'));

// 咨询师列表（?status=PENDING|APPROVED|REJECTED|SUSPENDED，不传=全部）
router.get('/consultants', expertController.getConsultantsForAdmin);

// 审核：通过 / 拒绝 / 停用 / 撤回待审
router.put('/consultants/:consultantId/status', expertController.updateConsultantStatus);

/**
 * 分组覆盖与风险分布（真实数据）。
 *
 * ⚠️ 这里刻意**不**返回 accuracy / recall / 公平性判定：
 *    那需要「模型预测 + 金标准标签」配对，而平台目前没有任何标注真值来源。
 *    此前管理端那张表是用写死的 0.82/0.85/... 冒充性能指标，并配上
 *    「所有分组公平性指标均在可接受范围内」的结论 —— 属于不可验证的断言。
 *    现在改为只报可核对的口径：每组多少人、其中高风险多少、多少人有测评记录。
 */
router.get('/group-stats', async (_req, res, next) => {
  try {
    const [profiles, assessedRows, totalPatients] = await Promise.all([
      prisma.patientProfile.findMany({
        select: { userId: true, gender: true, birthDate: true, riskLevel: true },
      }),
      prisma.assessment.findMany({ select: { userId: true }, distinct: ['userId'] }),
      prisma.user.count({ where: { role: 'PATIENT' } }),
    ]);

    const assessedSet = new Set(assessedRows.map((a: { userId: string }) => a.userId));

    const ageBandOf = (birthDate: Date | null): string => {
      if (!birthDate) return '未填写';
      const age = Math.floor((Date.now() - new Date(birthDate).getTime()) / (365.25 * 24 * 3600 * 1000));
      if (age < 12) return '12岁以下';
      if (age <= 15) return '12-15岁';
      if (age <= 18) return '16-18岁';
      if (age <= 25) return '19-25岁';
      return '26岁以上';
    };

    const genderBandOf = (gender: string | null): string => {
      if (!gender) return '未填写';
      if (gender.includes('男') || /^m(ale)?$/i.test(gender)) return '男性';
      if (gender.includes('女') || /^f(emale)?$/i.test(gender)) return '女性';
      return '其他/未填写';
    };

    type Bucket = { users: number; highRisk: number; assessed: number };
    const aggregate = (
      bandOf: (p: { gender: string | null; birthDate: Date | null }) => string,
    ): { group: string; users: number; highRisk: number; highRiskRate: number; assessed: number; assessedRate: number }[] => {
      const buckets = new Map<string, Bucket>();
      for (const p of profiles) {
        const key = bandOf(p);
        const b = buckets.get(key) || { users: 0, highRisk: 0, assessed: 0 };
        b.users += 1;
        if (p.riskLevel === 'HIGH' || p.riskLevel === 'CRISIS') b.highRisk += 1;
        if (assessedSet.has(p.userId)) b.assessed += 1;
        buckets.set(key, b);
      }
      return [...buckets.entries()]
        .map(([group, b]) => ({
          group,
          users: b.users,
          highRisk: b.highRisk,
          highRiskRate: b.users ? Number((b.highRisk / b.users).toFixed(4)) : 0,
          assessed: b.assessed,
          assessedRate: b.users ? Number((b.assessed / b.users).toFixed(4)) : 0,
        }))
        .sort((a, b) => b.users - a.users);
    };

    res.json({
      code: 0,
      data: {
        coverage: {
          totalPatients,
          withProfile: profiles.length,
          withAssessment: assessedSet.size,
        },
        byAge: aggregate((p) => ageBandOf(p.birthDate)),
        byGender: aggregate((p) => genderBandOf(p.gender)),
        // 明确写清「哪些指标没有」以及为什么没有，避免读者把它当成性能评测
        unavailableNote:
          '分组准确率/召回率与公平性判定需要「模型预测 + 金标准标签」配对数据，' +
          '本平台未采集标注真值，因此不予展示，也不做「公平性达标」结论。',
      },
    });
  } catch (e) { next(e); }
});

export default router;
