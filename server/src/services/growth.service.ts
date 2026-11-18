import achievementService from './achievement.service';
import {
  computeCounters,
  computeHeatmap,
  computeMoodSeries,
  extractKeywords,
  startOfLocalDay,
  GrowthCounters,
} from './growth.counters';
import prisma from '../config/database';

/**
 * 「我的成长」指标板块聚合服务。
 *
 * 前端只做渲染：指标值、口径说明、空态、下一步动作全部在这里算好返回，
 * 避免再出现「4 个请求各自 catch(() => null)，失败和真是 0 分不清」的老问题。
 */

const LEVEL_NAMES = ['萌芽', '新芽', '花苞', '初绽', '盛放', '向阳', '绽放', '璀璨', '星光', '彩虹'];
const LEVEL_ICONS = ['🌱', '🌿', '🍀', '🌷', '🌸', '🌻', '🌼', '💫', '⭐', '🌈'];
/** Lv.n 所需累计成长值（Lv.1 起点为 0） */
const LEVEL_THRESHOLDS = [0, 50, 140, 280, 480, 760, 1140, 1620, 2220, 3000];

const clamp = (n: number, min: number, max: number) => Math.max(min, Math.min(max, n));
const round1 = (n: number) => Math.round(n * 10) / 10;

export interface MetricCard {
  key: string;
  group: 'persistence' | 'mood' | 'investment' | 'exploration';
  label: string;
  /** 已格式化的展示值；无数据时为 null */
  value: string | null;
  display: string;
  unit: string;
  /** ok=有数据 / empty=未采集 / building=样本不足 */
  status: 'ok' | 'empty' | 'building';
  hint: string;
  source: string;
  current: number | null;
  target: number | null;
  percent: number | null;
}

export interface GrowthOverview {
  index: {
    score: number;
    grade: string;
    dimensions: Array<{ key: string; label: string; score: number | null; weight: number; hint: string }>;
  };
  level: {
    level: number;
    name: string;
    icon: string;
    xp: number;
    currentLevelXp: number;
    nextLevelXp: number;
    progress: number;
    isMax: boolean;
  };
  summary: string;
  counters: GrowthCounters;
  metrics: MetricCard[];
  keywords: string[];
  trend: Awaited<ReturnType<typeof computeMoodSeries>>;
  heatmap: Awaited<ReturnType<typeof computeHeatmap>>;
  badges: Awaited<ReturnType<typeof achievementService.getBadgeBoard>>;
  newlyUnlocked: Array<{ id: string; name: string; icon: string }>;
  nextActions: Array<{ key: string; label: string; desc: string; path: string; icon: string }>;
}

/** 成长值：由行为量累加，等级不再依赖「徽章数 / 3」那种封顶算法。 */
function computeXp(c: GrowthCounters): number {
  return (
    c.totalCheckIns * 10 +
    c.healingSessions * 8 +
    c.assessments * 15 +
    Math.min(c.streakDays, 30) * 2 +
    c.activeDays30 * 2
  );
}

function computeLevel(xp: number) {
  let level = 1;
  for (let i = 0; i < LEVEL_THRESHOLDS.length; i += 1) {
    if (xp >= LEVEL_THRESHOLDS[i]) level = i + 1;
  }
  const isMax = level >= LEVEL_THRESHOLDS.length;
  const currentLevelXp = LEVEL_THRESHOLDS[level - 1];
  const nextLevelXp = isMax ? currentLevelXp : LEVEL_THRESHOLDS[level];
  const span = nextLevelXp - currentLevelXp;
  const progress = isMax ? 100 : Math.round(((xp - currentLevelXp) / span) * 100);
  return {
    level,
    name: LEVEL_NAMES[level - 1],
    icon: LEVEL_ICONS[level - 1],
    xp,
    currentLevelXp,
    nextLevelXp,
    progress: clamp(progress, 0, 100),
    isMax,
  };
}

/** 四个维度分（0-100）。null = 该维度样本不足，不参与指数加权。 */
function computeDimensions(c: GrowthCounters) {
  const persistence = Math.round(
    c.checkInRate30 * 0.6 + Math.min(c.streakDays / 21, 1) * 100 * 0.4,
  );

  // 心情分：均分归一化（1-10 → 0-100）与稳定度六四开；只有稳定度时（无近14天均分）返回 null
  let mood: number | null = null;
  if (c.avgMood14 !== null) {
    const normalized = ((c.avgMood14 - 1) / 9) * 100;
    mood = Math.round(c.moodStability7 === null ? normalized : normalized * 0.6 + c.moodStability7 * 0.4);
  }

  const investment = Math.round(
    Math.min(c.healingSessions / 20, 1) * 50 + Math.min(c.healingMinutes / 180, 1) * 50,
  );
  const exploration = Math.round(Math.min(c.assessments / 5, 1) * 100);

  return [
    {
      key: 'persistence',
      label: '坚持',
      score: persistence,
      weight: 0.35,
      hint: '近 30 天打卡率 × 0.6 + 连续打卡（21 天封顶）× 0.4',
    },
    {
      key: 'mood',
      label: '心情',
      score: mood,
      weight: 0.3,
      hint: '近 14 天心情均分（1-10 换算 0-100）× 0.6 + 情绪稳定度 × 0.4',
    },
    {
      key: 'investment',
      label: '投入',
      score: investment,
      weight: 0.2,
      hint: '疗愈次数（20 次封顶）× 50 + 疗愈时长（180 分钟封顶）× 50',
    },
    {
      key: 'exploration',
      label: '探索',
      score: exploration,
      weight: 0.15,
      hint: '测评次数（5 次封顶）× 100',
    },
  ];
}

function gradeOf(score: number): string {
  if (score >= 80) return '状态很好';
  if (score >= 60) return '稳步积累';
  if (score >= 35) return '正在起步';
  if (score > 0) return '刚刚开始';
  return '等你来记录';
}

/** 把计数器翻译成前端可直接渲染的指标卡。 */
function buildMetrics(c: GrowthCounters, todayCheckedIn: boolean): MetricCard[] {
  const cards: MetricCard[] = [
    {
      key: 'streakDays',
      group: 'persistence',
      label: '连续打卡',
      value: `${c.streakDays}`,
      display: c.streakDays > 0 ? `${c.streakDays} 天` : '今天还没打卡',
      unit: '天',
      // 0 天不能标「已记录」：那会让人以为签到已经生效了（截图里就是这个误会）
      status: c.streakDays > 0 ? 'ok' : 'empty',
      hint: todayCheckedIn
        ? '含今天，从今天起往前数不断档'
        : c.streakDays > 0
          ? '今天还没打卡；不补的话连击会在明天断掉'
          : '今天还没打卡，打完今天就重新开始计数',
      source: 'mood_checkins.checkInDate',
      current: c.streakDays,
      target: Math.max(c.streakDays, 7),
      percent: Math.min(100, Math.round((c.streakDays / Math.max(c.streakDays, 7)) * 100)),
    },
    {
      key: 'checkInRate7',
      group: 'persistence',
      label: '近 7 天打卡率',
      value: `${c.checkInRate7}`,
      display: `${c.checkInRate7}%`,
      unit: '%',
      status: c.checkInRate7 > 0 ? 'ok' : 'empty',
      hint: '近 7 个自然日里打过卡的天数 ÷ 7',
      source: 'mood_checkins.checkInDate',
      current: c.checkInRate7,
      target: 100,
      percent: c.checkInRate7,
    },
    {
      key: 'activeDays30',
      group: 'persistence',
      label: '近 30 天活跃',
      value: `${c.activeDays30}`,
      display: `${c.activeDays30} / 30 天`,
      unit: '天',
      status: 'ok',
      hint: '有打卡或完成过一次练习的天数（同一天只算一次）',
      source: 'mood_checkins ∪ healing_sessions',
      current: c.activeDays30,
      target: 30,
      percent: Math.round((c.activeDays30 / 30) * 100),
    },
    {
      key: 'totalCheckIns',
      group: 'persistence',
      label: '累计打卡',
      value: `${c.totalCheckIns}`,
      display: `${c.totalCheckIns} 天`,
      unit: '天',
      status: 'ok',
      hint: '按自然日去重的打卡总天数',
      source: 'mood_checkins',
      current: c.totalCheckIns,
      target: null,
      percent: null,
    },
    {
      key: 'avgMood7',
      group: 'mood',
      label: '近 7 天心情',
      value: c.avgMood7 === null ? null : `${c.avgMood7}`,
      display: c.avgMood7 === null ? '待采集' : `${c.avgMood7} / 10`,
      unit: '/10',
      status: c.avgMood7 === null ? 'empty' : 'ok',
      hint: '近 7 天打卡自评分的平均（1-10，不是量表结论）',
      source: 'mood_checkins.score',
      current: c.avgMood7,
      target: 10,
      percent: c.avgMood7 === null ? null : Math.round((c.avgMood7 / 10) * 100),
    },
    {
      key: 'moodStability7',
      group: 'mood',
      label: '情绪稳定度',
      value: c.moodStability7 === null ? null : `${c.moodStability7}`,
      display: c.moodStability7 === null ? '样本不足' : `${c.moodStability7} / 100`,
      unit: '/100',
      status: c.moodStability7 === null ? 'building' : 'ok',
      hint: '近 7 天自评分的波动越小分越高（至少需要 2 天记录）',
      source: 'mood_checkins.score 标准差',
      current: c.moodStability7,
      target: 100,
      percent: c.moodStability7,
    },
    {
      key: 'moodDelta',
      group: 'mood',
      label: '近 7 天变化',
      value: c.moodDelta === null ? null : `${c.moodDelta > 0 ? '+' : ''}${c.moodDelta}`,
      display:
        c.moodDelta === null
          ? '数据不足'
          : `${c.moodDelta > 0 ? '+' : ''}${c.moodDelta} 分`,
      unit: '分',
      status: c.moodDelta === null ? 'building' : 'ok',
      hint: '近 7 天均分 − 前 7 天均分；需要前后各至少 1 天记录',
      source: 'mood_checkins.score',
      current: c.moodDelta,
      target: null,
      percent: null,
    },
    {
      key: 'healingSessions',
      group: 'investment',
      label: '完成练习',
      value: `${c.healingSessions}`,
      display: `${c.healingSessions} 次 · ${c.healingMinutes} 分钟`,
      unit: '次',
      status: 'ok',
      hint: '疗愈空间里完成的冥想 / 呼吸 / 小游戏次数与累计时长',
      source: 'healing_sessions(completed=true)',
      current: c.healingSessions,
      target: 10,
      percent: Math.min(100, Math.round((c.healingSessions / 10) * 100)),
    },
    {
      key: 'assessments',
      group: 'exploration',
      label: '量表测评',
      value: `${c.assessments}`,
      display: c.lastAssessment
        ? `${c.assessments} 次 · 最近「${c.lastAssessment.title}」`
        : `${c.assessments} 次`,
      unit: '次',
      status: c.assessments === 0 ? 'empty' : 'ok',
      hint: '提交过的心理量表次数；结果只反映当时状态，不做诊断',
      source: 'assessments',
      current: c.assessments,
      target: 3,
      percent: Math.min(100, Math.round((c.assessments / 3) * 100)),
    },
  ];

  return cards;
}

function buildSummary(c: GrowthCounters, todayCheckedIn: boolean, indexScore: number): string {
  if (c.totalCheckIns === 0) return '先记录一次今天的心情，这里就会开始长东西。';
  if (!todayCheckedIn && c.streakDays > 0) return `已连续 ${c.streakDays} 天，今天补上一笔就能接着算。`;
  if (!todayCheckedIn) return `已经记录了 ${c.totalCheckIns} 天，今天再写一笔就重新连起来。`;
  if (c.streakDays >= 7) return `已经连续打卡 ${c.streakDays} 天，这份坚持很了不起。`;
  if (c.streakDays >= 3) return `连续 ${c.streakDays} 天，习惯正在长出来。`;
  return `今天已打卡，累计 ${c.totalCheckIns} 天，成长指数 ${indexScore} 分。`;
}

function buildNextActions(c: GrowthCounters, todayCheckedIn: boolean) {
  const actions: Array<{ key: string; label: string; desc: string; path: string; icon: string }> = [];
  if (!todayCheckedIn) {
    actions.push({ key: 'checkin', label: '记录今天的心情', desc: '打卡后连续天数会继续累加', path: '/', icon: '📅' });
  }
  if (c.healingSessions === 0) {
    actions.push({ key: 'healing', label: '去疗愈空间放松一下', desc: '完成一次练习可解锁「疗愈新手」', path: '/healing', icon: '🌿' });
  }
  if (c.assessments === 0) {
    actions.push({ key: 'assessment', label: '做一次心情检测', desc: '了解当下的状态，解锁「测评先锋」', path: '/profile', icon: '🔍' });
  }
  if (actions.length === 0) {
    actions.push({ key: 'chat', label: '和 AI 聊聊最近', desc: '把感受说出来，也是一种整理', path: '/chat', icon: '💬' });
  }
  return actions;
}

export class GrowthService {
  /** 汇总成长指标板块（GET /api/extra/growth） */
  async getOverview(userId: string, days = 14): Promise<GrowthOverview> {
    const counters = await computeCounters(userId);

    // 打开页面时顺手补一次解锁：历史遗留的疗愈/测评记录也能点亮徽章
    const newlyUnlocked = await achievementService.evaluateAchievements(userId, counters);

    const fourteenDaysAgo = startOfLocalDay(new Date());
    fourteenDaysAgo.setDate(fourteenDaysAgo.getDate() - 13);

    const [badges, trend, heatmap, moodRows, todayRow] = await Promise.all([
      achievementService.getBadgeBoard(userId, counters),
      computeMoodSeries(userId, Math.min(Math.max(days, 7), 30)),
      computeHeatmap(userId, 30),
      prisma.moodCheckIn.findMany({
        where: { userId, checkInDate: { gte: fourteenDaysAgo } },
        select: { mood: true, note: true, tags: true },
        orderBy: { checkInDate: 'desc' },
      }),
      prisma.moodCheckIn.findFirst({
        where: { userId, checkInDate: { gte: startOfLocalDay(new Date()) } },
        select: { id: true },
      }),
    ]);

    const dimensions = computeDimensions(counters);
    const available = dimensions.filter((d) => d.score !== null);
    const weightSum = available.reduce((s, d) => s + d.weight, 0);
    const indexScore = weightSum > 0
      ? Math.round(available.reduce((s, d) => s + (d.score as number) * d.weight, 0) / weightSum)
      : 0;

    const todayCheckedIn = !!todayRow;

    return {
      index: { score: indexScore, grade: gradeOf(indexScore), dimensions },
      level: computeLevel(computeXp(counters)),
      summary: buildSummary(counters, todayCheckedIn, indexScore),
      counters,
      metrics: buildMetrics(counters, todayCheckedIn),
      keywords: extractKeywords(moodRows),
      trend,
      heatmap,
      badges,
      newlyUnlocked,
      nextActions: buildNextActions(counters, todayCheckedIn),
    };
  }
}

export default new GrowthService();
