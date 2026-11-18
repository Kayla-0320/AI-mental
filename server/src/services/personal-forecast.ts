/** 冻结EMA配方用于本人自评参考；项目量表上的效果尚待验证。 */
export interface PersonalForecast {
  status: 'experimental' | 'abstain'; reason?: string; targetDate?: string;
  score?: number; observedDays: number; method: 'ema_0.4'; scale: 'project_mood_1_10';
  validatedForProject: false;
}
export type MoodObservation = { score: number; recordedAt: Date };
const formatter = new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Shanghai', year: 'numeric', month: '2-digit', day: '2-digit' });
const dayOf = (date: Date): number => {
  const parts = formatter.formatToParts(date);
  const value = (type: string): number => Number(parts.find(part => part.type === type)?.value);
  return Date.UTC(value('year'), value('month') - 1, value('day')) / 86400000;
};

/** 按实际日历聚合，拒绝未来/缺测/冷启动；不把模型概率当本人自评。 */
export function forecastSelfReports(rows: MoodObservation[], now: Date): PersonalForecast {
  const base = { observedDays: 0, method: 'ema_0.4', scale: 'project_mood_1_10', validatedForProject: false } as const;
  const today = dayOf(now);
  const groups = new Map<number, number[]>();
  for (const row of rows) {
    if (!Number.isFinite(row.recordedAt.getTime()) || row.recordedAt > now || !Number.isFinite(row.score) || row.score < 1 || row.score > 10) continue;
    const day = dayOf(row.recordedAt);
    if (day <= today - 14) continue;
    groups.set(day, [...(groups.get(day) ?? []), row.score]);
  }
  const days = [...groups.keys()].sort((a, b) => a - b);
  if (days.at(-1) !== today) return { ...base, status: 'abstain', reason: 'latest_day_missing', observedDays: days.length };
  let start = 0;
  for (let i = 1; i < days.length; i++) if (days[i] - days[i - 1] > 3) start = i;
  const history = days.slice(start).map(day => groups.get(day)!.reduce((sum, score) => sum + score, 0) / groups.get(day)!.length);
  if (history.length < 3) return { ...base, status: 'abstain', reason: 'insufficient_history', observedDays: history.length };
  let prediction = history[0];
  for (const value of history.slice(1)) prediction = .6 * prediction + .4 * value;
  return { ...base, status: 'experimental', observedDays: history.length,
    targetDate: new Date((today + 1) * 86400000).toISOString().slice(0, 10), score: Number(prediction.toFixed(2)) };
}
