/**
 * 「我的成长」指标板块演示数据。
 *
 * 目的：把成长页从「一片 0」变成可以看到真实指标的样子，同时**不动真实账号的数据**。
 * 默认只写一个独立演示账号 `growth-demo@mental.com`（密码 123456）；
 * 想给自己已有的账号灌数据：`node scripts/seed-growth-demo.mjs --email=patient@mental.com`。
 *
 * 特性：
 *  - 幂等：同一天已有打卡就跳过；疗愈/测评按 title 标记去重，重复执行不会翻倍。
 *  - 可回滚：`--clean` 删除演示账号及其全部数据（级联）。
 *  - 确定性：随机数固定种子 42，多次执行结果一致（符合 AGENTS.md 种子约定）。
 *
 * 用法：
 *   node scripts/seed-growth-demo.mjs                 # 建/补演示账号
 *   node scripts/seed-growth-demo.mjs --email=x@y.com # 写入指定账号
 *   node scripts/seed-growth-demo.mjs --clean         # 删除演示账号
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
const bcrypt = require(path.join(REPO, 'server', 'node_modules', 'bcryptjs'));

const prisma = new PrismaClient();

const DEMO_EMAIL = 'growth-demo@mental.com';
const DEMO_PASSWORD = '123456';
const DEMO_NICKNAME = '成长演示';
const HEALING_MARK = '成长演示';
const ASSESSMENT_MARK = '[演示]';

const args = process.argv.slice(2);
const argOf = (name) => {
  const hit = args.find((a) => a.startsWith(`--${name}=`));
  return hit ? hit.split('=').slice(1).join('=') : null;
};
const email = argOf('email') || DEMO_EMAIL;
const shouldClean = args.includes('--clean');

// —— 固定种子 42 的确定性伪随机 ——
let seed = 42;
const rnd = () => {
  seed = (seed * 1103515245 + 12345) % 2147483648;
  return seed / 2147483648;
};
const pick = (arr) => arr[Math.floor(rnd() * arr.length)];

const dayStart = (offset) => {
  const d = new Date();
  d.setHours(0, 0, 0, 0);
  d.setDate(d.getDate() - offset);
  return d;
};

const MOODS = [
  { mood: '超好', score: 9, emoji: '😄', tags: ['有成就感', '被理解'] },
  { mood: '不错', score: 7, emoji: '😊', tags: ['朋友', '运动'] },
  { mood: '一般', score: 5, emoji: '😐', tags: ['作业多'] },
  { mood: '低落', score: 3, emoji: '😔', tags: ['考试', '没睡好'] },
];
const NOTES = [
  '今天和同桌把话说明白了，心里松快很多',
  '跑完步回来出了一身汗，脑子清醒了',
  '作业有点多，写完感觉还行',
  '有点想家，给妈妈打了电话就好一些',
  '晚自习效率一般，明天早点睡',
  '被老师表扬了一句，开心了一整天',
];
const HEALING_PLAN = [
  { offset: 12, type: 'game:flick', title: '甩点挑战', duration: 30 },
  { offset: 10, type: 'meditation', title: '10 分钟呼吸练习', duration: 900 },
  { offset: 9, type: 'mindfulness', title: '正念静坐', duration: 900 },
  { offset: 8, type: 'breathing', title: '4-7-8 呼吸', duration: 300 },
  { offset: 6, type: 'game:reaction-light', title: '反应灯', duration: 45 },
  { offset: 5, type: 'music', title: '睡前放松音乐', duration: 600 },
  { offset: 3, type: 'meditation', title: '身体扫描', duration: 900 },
  { offset: 2, type: 'game:flick', title: '甩点挑战', duration: 30 },
  { offset: 0, type: 'breathing', title: '考前一分钟呼吸', duration: 120 },
];
const ASSESSMENT_PLAN = [
  { offset: 14, type: 'PHQ9', title: '情绪状态自评（PHQ-9）', totalScore: 4, level: 'normal', suggestion: '目前状态良好，继续保持积极的生活方式。' },
  { offset: 7, type: 'GAD7', title: '紧张程度自评（GAD-7）', totalScore: 6, level: 'mild', suggestion: '轻度焦虑倾向，建议进行放松训练和规律运动。' },
  { offset: 3, type: 'PHQ9', title: '情绪状态自评（PHQ-9）', totalScore: 3, level: 'normal', suggestion: '目前状态良好，继续保持积极的生活方式。' },
];
/** 第 7、13 天故意留空缺：让「连续天数 / 打卡率」看起来是真实曲线而不是人造满格 */
const CHECK_IN_GAPS = new Set([7, 13, 17]);

async function cleanup() {
  const user = await prisma.user.findUnique({ where: { email } });
  if (!user) return console.log(`[clean] ${email} 不存在，无需清理`);
  await prisma.user.delete({ where: { email } }); // 关联数据全部 onDelete: Cascade
  console.log(`[clean] 已删除 ${email} 及其全部数据`);
}

async function ensureUser() {
  const existing = await prisma.user.findUnique({ where: { email } });
  if (existing) return existing;
  const password = await bcrypt.hash(DEMO_PASSWORD, 12);
  const user = await prisma.user.create({
    data: {
      email,
      password,
      nickname: DEMO_NICKNAME,
      role: 'PATIENT',
      patientProfile: { create: {} },
    },
  });
  console.log(`[user] 已创建演示账号 ${email} / ${DEMO_PASSWORD}`);
  return user;
}

async function seedCheckIns(userId, profileId) {
  let created = 0;
  for (let offset = 0; offset <= 20; offset += 1) {
    if (CHECK_IN_GAPS.has(offset)) continue;
    const date = dayStart(offset);
    const next = new Date(date);
    next.setDate(next.getDate() + 1);

    const exists = await prisma.moodCheckIn.findFirst({
      where: { userId, checkInDate: { gte: date, lt: next } },
      select: { id: true },
    });
    if (exists) continue;

    // 越近的日期分数略高：能真实体现「近 7 天变化」这个指标
    const base = offset <= 6 ? 7 : offset <= 13 ? 6 : 5;
    const jitter = rnd() < 0.3 ? -1 : rnd() < 0.25 ? 1 : 0;
    const score = Math.max(1, Math.min(10, base + jitter));
    const palette = MOODS.reduce((best, m) => (Math.abs(m.score - score) < Math.abs(best.score - score) ? m : best), MOODS[0]);
    const note = `${palette.emoji} ${palette.mood}｜${pick(NOTES)}`;

    await prisma.moodCheckIn.create({
      data: {
        userId,
        patientProfileId: profileId,
        mood: palette.mood,
        score,
        energy: Math.max(1, Math.min(10, score + (rnd() < 0.5 ? 1 : -1))),
        sleepHours: 5.5 + Math.round(rnd() * 3 * 10) / 10,
        note,
        tags: JSON.stringify(palette.tags),
        checkInDate: date,
        createdAt: date,
      },
    });
    created += 1;
  }
  return created;
}

async function seedHealing(userId) {
  let created = 0;
  for (const plan of HEALING_PLAN) {
    const at = dayStart(plan.offset);
    at.setHours(20, 30, 0, 0);
    const next = new Date(at);
    next.setDate(next.getDate() + 1);

    // 幂等键 = 同一天 + 同名练习（同一天重复跑脚本不会翻倍，不同天同名练习各自保留）
    const exists = await prisma.healingSession.findFirst({
      where: { userId, title: plan.title, createdAt: { gte: at, lt: next } },
      select: { id: true },
    });
    if (exists) continue;

    await prisma.healingSession.create({
      data: {
        userId,
        type: plan.type,
        title: plan.title,
        duration: plan.duration,
        completed: true,
        moodBefore: '一般',
        moodAfter: '不错',
        metadata: JSON.stringify({ seed: HEALING_MARK }),
        createdAt: at,
      },
    });
    created += 1;
  }
  return created;
}

async function seedAssessments(userId) {
  let created = 0;
  for (const plan of ASSESSMENT_PLAN) {
    const at = dayStart(plan.offset);
    at.setHours(19, 0, 0, 0);
    const next = new Date(at);
    next.setDate(next.getDate() + 1);
    const title = `${ASSESSMENT_MARK}${plan.title}`;

    const exists = await prisma.assessment.findFirst({
      where: { userId, title, createdAt: { gte: at, lt: next } },
      select: { id: true },
    });
    if (exists) continue;

    await prisma.assessment.create({
      data: {
        userId,
        type: plan.type,
        title,
        answers: JSON.stringify({ 1: 1, 2: 0, 3: 1 }),
        result: JSON.stringify({ totalScore: plan.totalScore, level: plan.level, suggestion: plan.suggestion }),
        totalScore: plan.totalScore,
        level: plan.level,
        suggestion: plan.suggestion,
        completedAt: at,
        createdAt: at,
      },
    });
    created += 1;
  }
  return created;
}

async function main() {
  if (shouldClean) {
    await cleanup();
    return;
  }

  const user = await ensureUser();
  const profile = await prisma.patientProfile.findUnique({ where: { userId: user.id } });

  const [checkIns, healings, assessments] = [
    await seedCheckIns(user.id, profile?.id ?? null),
    await seedHealing(user.id),
    await seedAssessments(user.id),
  ];

  const counts = {
    打卡: await prisma.moodCheckIn.count({ where: { userId: user.id } }),
    疗愈: await prisma.healingSession.count({ where: { userId: user.id, completed: true } }),
    测评: await prisma.assessment.count({ where: { userId: user.id } }),
  };

  console.log(`[seed] 新增：打卡 ${checkIns} 天 · 疗愈 ${healings} 次 · 测评 ${assessments} 次`);
  console.log(`[seed] 当前总量：打卡 ${counts.打卡} 天 · 疗愈 ${counts.疗愈} 次 · 测评 ${counts.测评} 次`);
  console.log(`[seed] 账号：${email} / ${DEMO_PASSWORD}`);
  console.log('[seed] 打开 http://localhost:5173/growth 查看指标板块');
}

main()
  .catch((e) => {
    console.error('[seed] 失败：', e);
    process.exitCode = 1;
  })
  .finally(() => prisma.$disconnect());
