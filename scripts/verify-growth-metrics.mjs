/**
 * 成长指标交叉校验：接口值 vs 独立重算值。
 *
 * 为什么要这个脚本：成长页的历史问题之一就是「指标口径有两份实现，谁也不知道哪个对」
 * （徽章用累计数冒充连续天数）。这里用**另一条代码路径**直接从数据库取原始行重算，
 * 与 `GET /api/extra/growth` 返回的每个数字逐项比对。
 *
 * 用法：
 *   node scripts/verify-growth-metrics.mjs                                  # 默认校验演示账号
 *   node scripts/verify-growth-metrics.mjs --email=patient@mental.com      # 校验指定账号
 *   node scripts/verify-growth-metrics.mjs --base=http://127.0.0.1:3000
 *
 * 退出码：0 = 全部一致；1 = 有指标不一致。
 */

import { createRequire } from 'module';
import path from 'path';
import { fileURLToPath } from 'url';

const require = createRequire(import.meta.url);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(__dirname, '..');

const dotenv = require(path.join(REPO, 'server', 'node_modules', 'dotenv'));
dotenv.config({ path: path.join(REPO, 'server', '.env') });
const { PrismaClient } = require(path.join(REPO, 'server', 'node_modules', '@prisma/client'));

const prisma = new PrismaClient();

const args = process.argv.slice(2);
const argOf = (n) => {
  const hit = args.find((a) => a.startsWith(`--${n}=`));
  return hit ? hit.split('=').slice(1).join('=') : null;
};
const email = argOf('email') || 'growth-demo@mental.com';
const password = argOf('password') || '123456';
const base = argOf('base') || 'http://127.0.0.1:3000';

// ── 独立口径（刻意不复用 server/src/services/growth.counters.ts，避免同错同源）──
const key = (d) => {
  const x = new Date(d);
  return `${x.getFullYear()}-${String(x.getMonth() + 1).padStart(2, '0')}-${String(x.getDate()).padStart(2, '0')}`;
};
const keyAgo = (n) => {
  const d = new Date();
  d.setHours(0, 0, 0, 0);
  d.setDate(d.getDate() - n);
  return key(d);
};
const mean = (a) => (a.length ? Math.round((a.reduce((x, y) => x + y, 0) / a.length) * 10) / 10 : null);
const sd = (a) => {
  if (a.length < 2) return null;
  const m = a.reduce((x, y) => x + y, 0) / a.length;
  return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / a.length);
};

const results = [];
function check(name, apiValue, expected) {
  const ok = JSON.stringify(apiValue) === JSON.stringify(expected);
  results.push({ name, ok, apiValue, expected });
  console.log(`${ok ? '  ✓' : '  ✗'} ${name}: 接口=${JSON.stringify(apiValue)} 重算=${JSON.stringify(expected)}`);
}

async function main() {
  const loginRes = await fetch(`${base}/api/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });
  const login = await loginRes.json();
  if (login.code !== 0) throw new Error(`登录失败: ${JSON.stringify(login)}`);
  const token = login.data.accessToken;
  const userId = login.data.user.id;

  const growthRes = await fetch(`${base}/api/extra/growth?days=14`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  const growth = await growthRes.json();
  if (growth.code !== 0) throw new Error(`成长接口失败: ${JSON.stringify(growth)}`);
  const api = growth.data;

  // ── 原始数据 ──
  const checkIns = await prisma.moodCheckIn.findMany({
    where: { userId },
    select: { checkInDate: true, score: true },
  });
  const healings = await prisma.healingSession.findMany({
    where: { userId, completed: true },
    select: { duration: true, createdAt: true },
  });
  const assessmentCount = await prisma.assessment.count({ where: { userId } });

  const checkDays = new Set(checkIns.map((c) => key(c.checkInDate)));
  const sortedDays = [...checkDays].sort();
  const activeDays = new Set([...checkDays, ...healings.map((h) => key(h.createdAt))]);

  const window = (from, to) => new Set(Array.from({ length: to - from + 1 }, (_, i) => keyAgo(from + i)));
  const inWindow = (days) => checkIns.filter((c) => days.has(key(c.checkInDate))).map((c) => c.score);

  const last7 = window(0, 6);
  const last14 = window(0, 13);
  const prev7 = window(7, 13);
  const last30 = window(0, 29);

  let streak = 0;
  for (let i = checkDays.has(keyAgo(0)) ? 0 : 1; checkDays.has(keyAgo(i)) && i < 400; i += 1) streak += 1;

  let longest = 0;
  let run = 0;
  sortedDays.forEach((d, idx) => {
    if (idx === 0) { run = 1; } else {
      const gap = (new Date(`${d}T00:00:00`) - new Date(`${sortedDays[idx - 1]}T00:00:00`)) / 86400000;
      run = gap === 1 ? run + 1 : 1;
    }
    longest = Math.max(longest, run);
  });

  const hit = (set) => { let n = 0; set.forEach((k) => { if (checkDays.has(k)) n += 1; }); return n; };
  const scores7 = inWindow(last7);
  const scores14 = inWindow(last14);
  const scoresPrev7 = inWindow(prev7);
  const stability = scores7.length >= 2
    ? Math.round(Math.max(0, Math.min(100, 100 - (sd(scores7) / 4.5) * 100)))
    : null;

  console.log(`\n用户：${email}（${userId}）`);
  console.log('── 与接口逐项比对 ──');
  check('累计打卡天数 totalCheckIns', api.counters.totalCheckIns, checkDays.size);
  check('当前连续天数 streakDays', api.counters.streakDays, streak);
  check('最长连续天数 longestStreak', api.counters.longestStreak, longest);
  check('近7天打卡率 checkInRate7', api.counters.checkInRate7, Math.round((hit(last7) / 7) * 100));
  check('近30天打卡率 checkInRate30', api.counters.checkInRate30, Math.round((hit(last30) / 30) * 100));
  check('近30天活跃 activeDays30', api.counters.activeDays30,
    [...last30].filter((k) => activeDays.has(k)).length);
  check('近7天心情均分 avgMood7', api.counters.avgMood7, mean(scores7));
  check('近14天心情均分 avgMood14', api.counters.avgMood14, mean(scores14));
  check('前7天心情均分 avgMoodPrev7', api.counters.avgMoodPrev7, mean(scoresPrev7));
  check('情绪稳定度 moodStability7', api.counters.moodStability7, stability);
  check('近7天变化 moodDelta', api.counters.moodDelta,
    mean(scores7) !== null && mean(scoresPrev7) !== null
      ? Math.round((mean(scores7) - mean(scoresPrev7)) * 10) / 10
      : null);
  check('完成练习次数 healingSessions', api.counters.healingSessions, healings.length);
  check('练习累计分钟 healingMinutes', api.counters.healingMinutes,
    Math.round(healings.reduce((s, h) => s + (h.duration || 0), 0) / 60));
  check('测评次数 assessments', api.counters.assessments, assessmentCount);

  // ── 结构性断言：指标卡 / 徽章 / 空值语义 ──
  console.log('── 结构断言 ──');
  const struct = [];
  const assert = (name, cond) => { struct.push({ name, ok: !!cond }); console.log(`${cond ? '  ✓' : '  ✗'} ${name}`); };

  assert('指标卡 9 张且全部带口径与来源',
    api.metrics.length === 9 && api.metrics.every((m) => m.hint && m.source && m.group));
  assert('无数据的指标状态为 empty/building 而不是伪装成 0',
    api.metrics.filter((m) => m.value === null).every((m) => m.status !== 'ok'));
  assert('成长指数在 0-100 之间', api.index.score >= 0 && api.index.score <= 100);
  assert('等级在 1-10 且名称非空', api.level.level >= 1 && api.level.level <= 10 && !!api.level.name);
  assert('等级进度的分母合法（未满级时 nextLevelXp > currentLevelXp）',
    api.level.isMax || api.level.nextLevelXp > api.level.currentLevelXp);
  assert('徽章总数 >= 12 且每枚都有 current/target',
    api.badges.total >= 12 && api.badges.badges.every((b) => typeof b.progress.current === 'number' && b.progress.target > 0));
  assert('已解锁徽章数与列表一致',
    api.badges.unlocked === api.badges.badges.filter((b) => b.unlocked).length);
  assert('趋势为 14 个点、每点含本地日与分数', api.trend.length === 14 && api.trend.every((t) => t.dayKey && 'score' in t));
  assert('热力图为 30 天', api.heatmap.length === 30);
  assert('下一步动作非空且带路径', api.nextActions.length > 0 && api.nextActions.every((a) => a.path));

  const failed = [...results, ...struct].filter((r) => !r.ok);
  console.log(`\n结果：${results.length + struct.length - failed.length}/${results.length + struct.length} 项通过`);
  if (failed.length) {
    console.log('不一致项：');
    failed.forEach((f) => console.log(`  - ${f.name}`));
    process.exitCode = 1;
  } else {
    console.log('全部一致 ✅');
  }
}

main()
  .catch((e) => {
    console.error('校验失败：', e);
    process.exitCode = 1;
  })
  .finally(() => prisma.$disconnect());
