import prisma from '../config/database';

/**
 * 「我的成长」指标的唯一口径层。
 *
 * 为什么单独抽一个文件：连续天数、打卡率这类口径一旦有两份实现，
 * 就会重演历史 bug —— 徽章判定用「累计打卡数」冒充「连续天数」
 * （旧 mood.service.ts:92）。徽章引擎与成长指数都从这里取数。
 */

const DAY_MS = 24 * 60 * 60 * 1000;

/** 本地日零点。指标一律按本地日归并（跨时区/夏令时都不在该项目范围内）。 */
export function startOfLocalDay(input: Date | string | number): Date {
  const d = new Date(input);
  d.setHours(0, 0, 0, 0);
  return d;
}

/** 本地日 key：YYYY-MM-DD */
export function dayKey(input: Date | string | number): string {
  const d = startOfLocalDay(input);
  const m = `${d.getMonth() + 1}`.padStart(2, '0');
  const day = `${d.getDate()}`.padStart(2, '0');
  return `${d.getFullYear()}-${m}-${day}`;
}

/** n 天前的本地日 key（n=0 即今天） */
export function dayKeyAgo(n: number): string {
  const d = startOfLocalDay(new Date());
  d.setDate(d.getDate() - n);
  return dayKey(d);
}

export interface AssessmentBrief {
  type: string;
  title: string;
  totalScore: number;
  level: string;
  createdAt: Date;
}

export interface GrowthCounters {
  /** 累计打卡天数 */
  totalCheckIns: number;
  /** 当前连续打卡天数（今天未打卡时从昨天起算，不因此归零） */
  streakDays: number;
  /** 历史最长连续打卡天数 */
  longestStreak: number;
  /** 近 7 天打卡率 0-100 */
  checkInRate7: number;
  /** 近 30 天打卡率 0-100 */
  checkInRate30: number;
  /** 近 30 天活跃天数（打卡 ∪ 完成疗愈，按本地日去重） */
  activeDays30: number;
  /** 近 7 天心情均分 1-10，无记录为 null */
  avgMood7: number | null;
  /** 近 14 天心情均分 1-10 */
  avgMood14: number | null;
  /** 前 7 天（第 8-14 天）心情均分，用于算变化 */
  avgMoodPrev7: number | null;
  /** 情绪稳定度 0-100（近 7 天，样本 <2 为 null） */
  moodStability7: number | null;
  /** 近 7 天相对前 7 天的均分变化，缺一侧为 null */
  moodDelta: number | null;
  /** 完成的疗愈/小游戏次数 */
  healingSessions: number;
  /** 疗愈累计时长（分钟，四舍五入） */
  healingMinutes: number;
  /** 测评次数 */
  assessments: number;
  /** 最近一次测评，无则 null */
  lastAssessment: AssessmentBrief | null;
}

const round1 = (n: number) => Math.round(n * 10) / 10;
const clamp = (n: number, min: number, max: number) => Math.max(min, Math.min(max, n));

/**
 * 由「本地日 key 集合」计算当前连续天数。
 * 今天没打卡不算断（早上打开页面不该看到归零），从昨天起算。
 */
export function computeStreak(dayKeys: Set<string>): number {
  let streak = 0;
  let cursor = 0;
  if (!dayKeys.has(dayKeyAgo(0))) cursor = 1;
  // 上限 400 天，避免脏数据导致死循环
  while (cursor < 400 && dayKeys.has(dayKeyAgo(cursor))) {
    streak += 1;
    cursor += 1;
  }
  return streak;
}

/** 由升序去重的本地日 key 列表计算最长连续天数。 */
export function computeLongestStreak(sortedKeys: string[]): number {
  if (sortedKeys.length === 0) return 0;
  let longest = 1;
  let run = 1;
  for (let i = 1; i < sortedKeys.length; i += 1) {
    const prev = new Date(`${sortedKeys[i - 1]}T00:00:00`).getTime();
    const curr = new Date(`${sortedKeys[i]}T00:00:00`).getTime();
    if (Math.round((curr - prev) / DAY_MS) === 1) {
      run += 1;
    } else {
      run = 1;
    }
    longest = Math.max(longest, run);
  }
  return longest;
}

/** 总体标准差（不是样本标准差：这里的分数是「这段时间的全部记录」而非抽样）。 */
export function stdev(values: number[]): number {
  if (values.length < 2) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

/** 近 14 天记录里的真实高频词（取用户自己的原词，不做临床解读）。 */
export function extractKeywords(
  records: Array<{ mood: string; note?: string | null; tags?: string | null }>,
  limit = 6,
): string[] {
  const stop = new Set([
    '今天', '感觉', '有点', '还是', '就是', '什么', '自己', '一个', '没有', '但是',
    '因为', '所以', '可能', '觉得', '现在', '已经', '这个', '我们', '他们',
  ]);
  const counter = new Map<string, number>();
  const push = (word: string | null | undefined) => {
    if (!word) return;
    const w = word.trim();
    if (w.length < 2 || w.length > 6) return;
    if (stop.has(w)) return;
    counter.set(w, (counter.get(w) || 0) + 1);
  };

  for (const r of records) {
    push(r.mood);
    if (r.tags) {
      try {
        const parsed = JSON.parse(r.tags);
        if (Array.isArray(parsed)) parsed.forEach((t: any) => typeof t === 'string' && push(t));
      } catch {
        /* tags 不是 JSON 就忽略，指标不该因为脏字段整体失败 */
      }
    }
    if (r.note) {
      // 备注里挖原词：emoji/标点切分，只保留中文或字母数字串
      r.note
        .split(/[^\u4e00-\u9fa5A-Za-z0-9]+/)
        .filter((t) => t.length >= 2 && t.length <= 6 && /[\u4e00-\u9fa5A-Za-z]/.test(t))
        .forEach(push);
    }
  }

  return Array.from(counter.entries())
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, limit)
    .map(([w]) => w);
}

function meanOrNull(values: number[]): number | null {
  if (values.length === 0) return null;
  return round1(values.reduce((a, b) => a + b, 0) / values.length);
}

/**
 * 一次取齐全部原始数据并算出计数器。
 *
 * 只查 3 张表：mood_checkins / healing_sessions / assessments。
 * 打卡表一次全量取（只选必要列），因为连续天数需要完整历史；
 * 单用户记录量级在千条以内，代价可接受。
 */
export async function computeCounters(userId: string): Promise<GrowthCounters> {
  const [checkIns, healings, assessments] = await Promise.all([
    prisma.moodCheckIn.findMany({
      where: { userId },
      select: { checkInDate: true, mood: true, score: true, note: true, tags: true },
      orderBy: { checkInDate: 'desc' },
      take: 2000,
    }),
    prisma.healingSession.findMany({
      where: { userId, completed: true },
      select: { duration: true, createdAt: true },
      orderBy: { createdAt: 'desc' },
      take: 2000,
    }),
    prisma.assessment.findMany({
      where: { userId },
      select: { type: true, title: true, totalScore: true, level: true, createdAt: true },
      orderBy: { createdAt: 'desc' },
      take: 500,
    }),
  ]);

  const checkInKeys = new Set(checkIns.map((c) => dayKey(c.checkInDate)));
  const sortedKeys = Array.from(checkInKeys).sort();

  // 近 N 天窗口（按本地日 key 比较，避免时区偏移把边界算错）
  const windowKeys = (days: number) =>
    new Set(Array.from({ length: days }, (_, i) => dayKeyAgo(i)));

  const last7Keys = windowKeys(7);
  const last14Keys = windowKeys(14);
  const prev7Keys = new Set(Array.from({ length: 7 }, (_, i) => dayKeyAgo(i + 7)));
  const last30Keys = windowKeys(30);

  const rate = (window: Set<string>) => {
    let hit = 0;
    window.forEach((k) => { if (checkInKeys.has(k)) hit += 1; });
    return Math.round((hit / window.size) * 100);
  };

  const scoresIn = (window: Set<string>) =>
    checkIns.filter((c) => window.has(dayKey(c.checkInDate))).map((c) => c.score);

  const scores7 = scoresIn(last7Keys);
  const scores14 = scoresIn(last14Keys);
  const scoresPrev7 = scoresIn(prev7Keys);

  const avgMood7 = meanOrNull(scores7);
  const avgMoodPrev7 = meanOrNull(scoresPrev7);
  const stability = scores7.length >= 2
    ? Math.round(clamp(100 - (stdev(scores7) / 4.5) * 100, 0, 100))
    : null;

  // 活跃日 = 打卡日 ∪ 完成疗愈日
  const activeKeys = new Set<string>();
  checkIns.forEach((c) => activeKeys.add(dayKey(c.checkInDate)));
  healings.forEach((h) => activeKeys.add(dayKey(h.createdAt)));
  let activeDays30 = 0;
  last30Keys.forEach((k) => { if (activeKeys.has(k)) activeDays30 += 1; });

  const recent14 = checkIns.filter((c) => last14Keys.has(dayKey(c.checkInDate)));

  return {
    totalCheckIns: checkInKeys.size,
    streakDays: computeStreak(checkInKeys),
    longestStreak: computeLongestStreak(sortedKeys),
    checkInRate7: rate(last7Keys),
    checkInRate30: rate(last30Keys),
    activeDays30,
    avgMood7,
    avgMood14: meanOrNull(scores14),
    avgMoodPrev7,
    moodStability7: stability,
    moodDelta: avgMood7 !== null && avgMoodPrev7 !== null ? round1(avgMood7 - avgMoodPrev7) : null,
    healingSessions: healings.length,
    healingMinutes: Math.round(healings.reduce((sum, h) => sum + (h.duration || 0), 0) / 60),
    assessments: assessments.length,
    lastAssessment: assessments[0]
      ? {
          type: assessments[0].type,
          title: assessments[0].title,
          totalScore: assessments[0].totalScore,
          level: assessments[0].level,
          createdAt: assessments[0].createdAt,
        }
      : null,
  };
}

/** 近 N 天的心情序列（按本地日聚合，同日多条取均值），供折线图使用。 */
export async function computeMoodSeries(userId: string, days = 14) {
  const from = startOfLocalDay(new Date());
  from.setDate(from.getDate() - (days - 1));

  const records = await prisma.moodCheckIn.findMany({
    where: { userId, checkInDate: { gte: from } },
    select: { checkInDate: true, score: true, mood: true },
    orderBy: { checkInDate: 'asc' },
  });

  const byDay = new Map<string, number[]>();
  const moodByDay = new Map<string, string>();
  records.forEach((r) => {
    const k = dayKey(r.checkInDate);
    byDay.set(k, [...(byDay.get(k) || []), r.score]);
    moodByDay.set(k, r.mood);
  });

  return Array.from({ length: days }, (_, i) => {
    const offset = days - 1 - i;
    const k = dayKeyAgo(offset);
    const scores = byDay.get(k);
    return {
      dayKey: k,
      label: k.slice(5),
      score: scores ? round1(scores.reduce((a, b) => a + b, 0) / scores.length) : null,
      mood: moodByDay.get(k) || null,
    };
  });
}

/** 近 N 天打卡热力图：0 未打卡，1 打卡且心情偏低，2 打卡且心情中上。 */
export async function computeHeatmap(userId: string, days = 30) {
  const from = startOfLocalDay(new Date());
  from.setDate(from.getDate() - (days - 1));
  const records = await prisma.moodCheckIn.findMany({
    where: { userId, checkInDate: { gte: from } },
    select: { checkInDate: true, score: true },
  });
  const map = new Map<string, number>();
  records.forEach((r) => map.set(dayKey(r.checkInDate), r.score));
  return Array.from({ length: days }, (_, i) => {
    const k = dayKeyAgo(days - 1 - i);
    const score = map.get(k);
    return {
      dayKey: k,
      label: k.slice(5),
      level: score === undefined ? 0 : score >= 6 ? 2 : 1,
      score: score ?? null,
    };
  });
}
