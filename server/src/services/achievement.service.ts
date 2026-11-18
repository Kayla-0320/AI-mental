import prisma from '../config/database';
import { computeCounters, GrowthCounters } from './growth.counters';

/**
 * 徽章引擎。
 *
 * 修掉的两个历史问题：
 *  - 旧实现只在「心情打卡」里查 `category:'checkin'` 的徽章，疗愈/测评徽章永不触发；
 *  - 旧实现把 `streak` 条件写成 `totalCheckIns >= days`，断续打卡也能拿「连续 7 天」。
 * 现在所有条件统一在这里求值，且连续天数走 growth.counters 的唯一口径。
 */

interface AchievementCondition {
  type: 'total_checkins' | 'streak' | 'total_sessions' | 'healing_minutes' | 'total_assessments' | 'active_days';
  count?: number;
  days?: number;
  minutes?: number;
}

export interface BadgeView {
  id: string;
  name: string;
  description: string;
  icon: string;
  category: string;
  unlocked: boolean;
  unlockedAt: Date | null;
  progress: { current: number; target: number; percent: number };
}

export interface BadgeBoard {
  total: number;
  unlocked: number;
  percent: number;
  badges: BadgeView[];
}

type AchievementRow = {
  id: string;
  name: string;
  description: string;
  icon: string;
  category: string;
  condition: string;
};

/** 默认徽章。原有 7 枚 id 保持不变（不清库、不破坏已解锁记录），新增 5 枚拉开阶段。 */
const DEFAULT_ACHIEVEMENTS: Array<{
  id: string;
  name: string;
  description: string;
  icon: string;
  category: string;
  condition: string;
}> = [
  // —— 打卡 ——
  { id: 'ach_checkin_first', name: '初次打卡', description: '完成第一次心情打卡', icon: '🌱', category: 'checkin', condition: JSON.stringify({ type: 'total_checkins', count: 1 }) },
  { id: 'ach_checkin_streak_3', name: '小小坚持', description: '连续打卡 3 天', icon: '✨', category: 'checkin', condition: JSON.stringify({ type: 'streak', days: 3 }) },
  { id: 'ach_checkin_week', name: '坚持一周', description: '连续打卡 7 天', icon: '🔥', category: 'checkin', condition: JSON.stringify({ type: 'streak', days: 7 }) },
  { id: 'ach_checkin_total_7', name: '七日之约', description: '累计打卡 7 天', icon: '📅', category: 'checkin', condition: JSON.stringify({ type: 'total_checkins', count: 7 }) },
  { id: 'ach_checkin_month', name: '月度达人', description: '连续打卡 30 天', icon: '🏅', category: 'checkin', condition: JSON.stringify({ type: 'streak', days: 30 }) },
  // —— 疗愈 ——
  { id: 'ach_healing_first', name: '疗愈新手', description: '完成第一次疗愈练习', icon: '🧘', category: 'healing', condition: JSON.stringify({ type: 'total_sessions', count: 1 }) },
  { id: 'ach_healing_minutes_60', name: '静心一小时', description: '疗愈累计 60 分钟', icon: '⏳', category: 'healing', condition: JSON.stringify({ type: 'healing_minutes', minutes: 60 }) },
  { id: 'ach_healing_master', name: '疗愈大师', description: '完成 10 次疗愈练习', icon: '🌟', category: 'healing', condition: JSON.stringify({ type: 'total_sessions', count: 10 }) },
  // —— 探索 ——
  { id: 'ach_assessment_first', name: '测评先锋', description: '完成第一次心理测评', icon: '📋', category: 'assessment', condition: JSON.stringify({ type: 'total_assessments', count: 1 }) },
  { id: 'ach_assessment_three', name: '三次探索', description: '完成 3 次心理测评', icon: '🔍', category: 'assessment', condition: JSON.stringify({ type: 'total_assessments', count: 3 }) },
  { id: 'ach_assessment_explorer', name: '自我探索者', description: '完成 5 次心理测评', icon: '🧭', category: 'assessment', condition: JSON.stringify({ type: 'total_assessments', count: 5 }) },
  // —— 里程碑 ——
  { id: 'ach_active_days_30', name: '月度陪伴', description: '30 天里每天都有记录或练习', icon: '🗓️', category: 'milestone', condition: JSON.stringify({ type: 'active_days', days: 30 }) },
];

/** 求值单条条件，返回「当前值 / 目标值」；条件类型不认识时返回 null（不显示假进度）。 */
export function evaluateCondition(
  raw: string,
  counters: GrowthCounters,
): { current: number; target: number } | null {
  let condition: AchievementCondition;
  try {
    condition = JSON.parse(raw);
  } catch {
    return null;
  }

  switch (condition.type) {
    case 'total_checkins':
      return { current: counters.totalCheckIns, target: condition.count ?? 1 };
    case 'streak':
      return { current: counters.streakDays, target: condition.days ?? 1 };
    case 'total_sessions':
      return { current: counters.healingSessions, target: condition.count ?? 1 };
    case 'healing_minutes':
      return { current: counters.healingMinutes, target: condition.minutes ?? 1 };
    case 'total_assessments':
      return { current: counters.assessments, target: condition.count ?? 1 };
    case 'active_days':
      return { current: counters.activeDays30, target: condition.days ?? 1 };
    default:
      return null;
  }
}

function isUnlocked(progress: { current: number; target: number } | null): boolean {
  return !!progress && progress.current >= progress.target;
}

export class AchievementService {
  // 获取所有成就
  async getAllAchievements() {
    return prisma.achievement.findMany({ orderBy: [{ category: 'asc' }, { id: 'asc' }] });
  }

  // 获取用户已解锁成就
  async getUserAchievements(userId: string) {
    return prisma.userAchievement.findMany({
      where: { userId },
      include: { achievement: true },
      orderBy: { unlockedAt: 'desc' },
    });
  }

  /**
   * 徽章墙视图：全部徽章 + 是否解锁 + 进度。
   * 已有 counters 时直接传入，避免一次请求里重复聚合。
   */
  async getBadgeBoard(userId: string, counters?: GrowthCounters): Promise<BadgeBoard> {
    const [all, owned, effective] = await Promise.all([
      this.getAllAchievements(),
      prisma.userAchievement.findMany({ where: { userId }, select: { achievementId: true, unlockedAt: true } }),
      counters ? Promise.resolve(counters) : computeCounters(userId),
    ]);

    const ownedMap = new Map(owned.map((o) => [o.achievementId, o.unlockedAt]));

    const badges: BadgeView[] = (all as AchievementRow[]).map((a) => {
      const progress = evaluateCondition(a.condition, effective);
      // 已解锁的徽章即使当前连击断了也保持点亮，进度按目标封顶
      const unlocked = ownedMap.has(a.id);
      const target = progress?.target ?? 1;
      const current = progress?.current ?? 0;
      return {
        id: a.id,
        name: a.name,
        description: a.description,
        icon: a.icon,
        category: a.category,
        unlocked,
        unlockedAt: ownedMap.get(a.id) ?? null,
        progress: {
          current,
          target,
          percent: Math.min(100, Math.round((current / target) * 100)),
        },
      };
    });

    const total = badges.length;
    const unlocked = badges.filter((b) => b.unlocked).length;
    return { total, unlocked, percent: total > 0 ? Math.round((unlocked / total) * 100) : 0, badges };
  }

  /**
   * 检查并解锁成就（幂等）。打卡 / 疗愈完成 / 测评提交三处调用。
   * 返回本次新解锁的徽章，供接口回执与前端提示使用。
   */
  async evaluateAchievements(userId: string, counters?: GrowthCounters) {
    const effective = counters ?? (await computeCounters(userId));
    const all = (await this.getAllAchievements()) as AchievementRow[];

    const owned = await prisma.userAchievement.findMany({
      where: { userId },
      select: { achievementId: true },
    });
    const ownedIds = new Set(owned.map((o) => o.achievementId));

    const newlyUnlocked: Array<{ id: string; name: string; icon: string }> = [];

    for (const achievement of all) {
      if (ownedIds.has(achievement.id)) continue;
      const progress = evaluateCondition(achievement.condition, effective);
      if (!isUnlocked(progress)) continue;

      await prisma.userAchievement.upsert({
        where: { userId_achievementId: { userId, achievementId: achievement.id } },
        update: {},
        create: { userId, achievementId: achievement.id },
      });
      newlyUnlocked.push({ id: achievement.id, name: achievement.name, icon: achievement.icon });
    }

    return newlyUnlocked;
  }

  /**
   * 兼容旧接口形状（GET /api/extra/achievements）。
   * `achievements` 保持「已解锁记录」不变，另附 badges 带进度的完整徽章墙。
   */
  async getAchievementProgress(userId: string) {
    const counters = await computeCounters(userId);
    const [all, userAchievements, board] = await Promise.all([
      prisma.achievement.count(),
      this.getUserAchievements(userId),
      this.getBadgeBoard(userId, counters),
    ]);

    const unlocked = board.unlocked;

    return {
      total: all,
      unlocked,
      // 徽章表为空时不再出现 NaN（旧实现 unlocked/all 直接相除）
      progress: all > 0 ? Math.round((unlocked / all) * 100) : 0,
      stats: {
        totalCheckIns: counters.totalCheckIns,
        totalHealingSessions: counters.healingSessions,
        totalAssessments: counters.assessments,
      },
      achievements: userAchievements,
      badges: board.badges,
    };
  }

  /** 初始化默认成就（幂等：按固定 id upsert）。 */
  async initDefaultAchievements() {
    for (const { id, ...item } of DEFAULT_ACHIEVEMENTS) {
      await prisma.achievement.upsert({
        where: { id },
        update: item,
        create: { id, ...item },
      });
    }
    return DEFAULT_ACHIEVEMENTS.length;
  }
}

export default new AchievementService();
