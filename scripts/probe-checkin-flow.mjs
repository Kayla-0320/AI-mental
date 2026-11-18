/**
 * 签到（每日心情打卡）流程探针 —— 在真实浏览器里把整条链路点一遍。
 *
 * 验的是这些曾经坏过的地方：
 *   1. 打卡卡渐变是否真的生效（全局 `!important` 曾把卡片压成白底 → 白字隐形）；
 *   2. 首次打卡后是否切成「今天已经打过卡啦」并显示所选心情（旧代码读错响应结构，恒判不出来）；
 *   3. 点错表情能不能当天更正（PUT /mood/checkin，仍是每天一条）；
 *   4. 「昨晚睡了多久」这一步是否真的落库、质量是否是 1-10 而不是 800；
 *   5. 打卡顺带解锁的徽章是否提示；
 *   6. **打卡之后成长页的数字有没有跟着更新**（「连续签到数据不能正常更新」的原始抱怨）。
 *
 * 用法：
 *   node scripts/probe-checkin-flow.mjs                                   # 临时账号，跑完自动清理
 *   node scripts/probe-checkin-flow.mjs --email=patient@mental.com        # 真实账号（不会注册/删除）
 * 退出码：0 = 全部通过；1 = 有失败项。
 */

import { createRequire } from 'module';
import path from 'path';
import fs from 'fs';
import { REPO, login, launchBrowser, injectAuth, goto, sleep } from './lib/cdp.mjs';

const require = createRequire(import.meta.url);
const dotenv = require(path.join(REPO, 'server', 'node_modules', 'dotenv'));
dotenv.config({ path: path.join(REPO, 'server', '.env') });
const { PrismaClient } = require(path.join(REPO, 'server', 'node_modules', '@prisma/client'));
const prisma = new PrismaClient();

const args = process.argv.slice(2);
const argOf = (n) => {
  const hit = args.find((a) => a.startsWith(`--${n}=`));
  return hit ? hit.split('=').slice(1).join('=') : null;
};

const API = 'http://127.0.0.1:3000';
const APP = 'http://localhost:5173';
const PORT = 9341;
const OUT = path.join(REPO, 'docs', 'screenshots', '29_checkin_flow_fixed.png');
const OUT_GROWTH = path.join(REPO, 'docs', 'screenshots', '30_growth_after_checkin.png');

const CHECKS = [];
const check = (name, ok, detail) => {
  CHECKS.push({ name, ok: !!ok });
  console.log(`${ok ? '  ✅' : '  ❌'} ${name}${detail === undefined ? '' : `  ${JSON.stringify(detail)}`}`);
};

const todayKey = () => {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
};
const authed = (token) => ({ Authorization: `Bearer ${token}` });
const apiGet = async (url, token) => (await fetch(url, { headers: authed(token) })).json();

/** 读打卡卡上的心情标签文本（已打卡/更正后才有） */
const readTag = (cdp) => cdp.eval(`(() => { const t = document.querySelector('.ant-card-body .ant-tag'); return t ? t.textContent.trim() : ''; })()`);

/**
 * 轮询标签直到包含期望心情。
 * 不能等「今天已经打过卡」——在「今天已打卡」的账号上这个条件点击前就成立了，
 * 会立刻返回旧文本，把上一次的选择误判成本次选择。
 */
async function waitTag(cdp, label, timeoutMs = 12000) {
  const start = Date.now();
  let last = '';
  while (Date.now() - start < timeoutMs) {
    last = await readTag(cdp).catch(() => '');
    if (last.includes(label)) return last;
    await sleep(250);
  }
  return last;
}

async function main() {
  const emailArg = argOf('email');
  const email = emailArg || `signin-probe-${Date.now()}@mental.com`;
  const password = argOf('password') || '123456';
  const isTemp = !emailArg;

  if (isTemp) {
    const reg = await (await fetch(`${API}/api/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, nickname: '签到探针' }),
    })).json();
    if (reg.code !== 0) throw new Error(`注册失败：${JSON.stringify(reg)}`);
  }
  const auth = await login(API, email, password);
  const token = auth.accessToken;

  // 开工前的真实状态：今天是否已打卡 + 成长计数
  const before = (await apiGet(`${API}/api/extra/growth`, token)).data;
  const already = !!(await apiGet(`${API}/api/mood/checkin/today`, token)).data?.checkedIn;
  console.log(`账号 ${email}｜今天${already ? '已' : '未'}打卡｜累计 ${before.counters.totalCheckIns} 天，连续 ${before.counters.streakDays} 天`);

  const { cdp, close } = await launchBrowser({ port: PORT });
  try {
    await injectAuth(cdp, auth);
    await goto(cdp, `${APP}/`);

    // ── 1. 打卡卡状态 ──
    let text = await cdp.waitForText((t) => t.includes('今天心情怎么样') || t.includes('今天已经打过卡'), { timeoutMs: 20000 });
    check('首页加载出打卡卡', text.includes('今天心情怎么样') || text.includes('今天已经打过卡'));
    if (!already) check('未打卡时显示引导文案', text.includes('今天心情怎么样'));
    if (already) check('已打卡时显示「今天已经打过卡啦」', text.includes('今天已经打过卡'));
    check('打卡状态与接口一致', text.includes('今天已经打过卡') === already, { already });

    const emojiButtons = await cdp.eval(`[...document.querySelectorAll('button')].filter(b => /^[\\u{1F300}-\\u{1FAFF}\\u2600-\\u27BF]$/u.test((b.textContent||'').trim())).length`);
    check('5 个心情表情都可点', emojiButtons >= 5, emojiButtons);

    const skin = await cdp.eval(`(() => {
      const card = [...document.querySelectorAll('.ant-card')].find((c) => (c.innerText||'').includes('今天心情怎么样') || (c.innerText||'').includes('已经打过卡'));
      if (!card) return { found: false };
      const body = card.querySelector('.ant-card-body');
      return { found: true, bodyBg: body ? getComputedStyle(body).backgroundImage : getComputedStyle(card).backgroundImage };
    })()`);
    check('打卡卡铺上了紫色渐变底（白字不再白底隐形）', /linear-gradient/.test(skin.bodyBg || ''), skin);

    // ── 2. 点一个表情（已打卡则是更正）──
    await cdp.eval(`[...document.querySelectorAll('button')].find(b => (b.textContent||'').trim() === '😐').click(), 1`);
    const tag2 = await waitTag(cdp, '一般');
    text = await cdp.eval('document.body.innerText');
    check('打卡后切换成「今天已经打过卡啦」', text.includes('今天已经打过卡'));
    check('显示刚选的心情「一般」', tag2.includes('一般'), tag2);
    if (!already && !before.badges.badges.some((b) => b.unlocked)) {
      check('首次打卡提示解锁徽章（初次打卡）', text.includes('解锁新徽章') && text.includes('初次打卡'));
    } else {
      console.log('  ⏭️  该账号此前已有徽章 / 今天已打卡，「首次解锁提示」不适用');
    }
    check('出现「昨晚睡了多久」这一步或已结束', text.includes('昨晚睡了多久') || already);

    const tagStyle = await cdp.eval(`(() => {
      const tag = document.querySelector('.ant-tag');
      if (!tag) return null;
      const body = tag.closest('.ant-card-body');
      return {
        color: getComputedStyle(tag).color,
        text: (tag.textContent||'').trim(),
        onGradient: body ? /linear-gradient/.test(getComputedStyle(body).backgroundImage) : false,
      };
    })()`);
    check('心情标签是白字且落在渐变底上', !!tagStyle && tagStyle.onGradient, tagStyle);

    // ── 3. 当天更正：改成一个与当前不同的心情，确保真的发生了一次更正 ──
    const currentTag = await readTag(cdp);
    const reviseTo = currentTag.includes('很难过')
      ? { emoji: '😐', label: '一般' }
      : { emoji: '😢', label: '很难过' };
    await cdp.eval(`[...document.querySelectorAll('button')].find(b => (b.textContent||'').trim() === ${JSON.stringify(reviseTo.emoji)}).click(), 1`);
    // 允许重试：请求进行中按钮是禁用态（antd 会拦掉点击），第一次点击可能正好落在窗口里
    let tag3 = await waitTag(cdp, reviseTo.label, 5000);
    for (let attempt = 0; attempt < 2 && !tag3.includes(reviseTo.label); attempt += 1) {
      await sleep(700);
      await cdp.eval(`[...document.querySelectorAll('button')].find(b => (b.textContent||'').trim() === ${JSON.stringify(reviseTo.emoji)}).click(), 1`);
      tag3 = await waitTag(cdp, reviseTo.label, 5000);
    }
    check('点错可以当天更正', tag3.includes(reviseTo.label), { from: currentTag, to: tag3 });

    const history = await apiGet(`${API}/api/mood/checkin/history?days=1`, token);
    const rows = Array.isArray(history.data) ? history.data : (history.data?.checkIns || []);
    const todayRows = rows.filter((r) => String(r.checkInDate).slice(0, 10) === todayKey());
    check('更正后今天仍然只有一条打卡记录', todayRows.length === 1, { rows: todayRows.length });
    check('记录内容是最新一次选择', todayRows[0]?.mood === reviseTo.label,
      { mood: todayRows[0]?.mood, expect: reviseTo.label });

    // ── 4. 睡眠步骤真的落库、量纲是 1-10 ──
    // 真实账号上跑时可用 --skip-sleep=1 跳过，避免给别人的账号伪造一条睡眠记录
    const sleepBefore = (await apiGet(`${API}/api/extra/sleep?days=30`, token)).data?.length || 0;
    const skipSleep = args.includes('--skip-sleep=1');
    const clickedSleep = skipSleep ? false : await cdp.eval(`(() => {
      const btn = [...document.querySelectorAll('button')].find(b => (b.textContent||'').trim() === '8h');
      if (!btn) return false;
      btn.click();
      return true;
    })()`);
    await sleep(skipSleep ? 0 : 2000);
    if (skipSleep) {
      console.log('  ⏭️  已按要求跳过睡眠步骤（不改动该账号的睡眠数据）');
    } else if (clickedSleep) {
      const sleepRows = (await apiGet(`${API}/api/extra/sleep?days=30`, token)).data || [];
      const newest = sleepRows[0];
      check('睡眠记录真的写进库了（曾经全表 0 行）', sleepRows.length === sleepBefore + 1, { before: sleepBefore, after: sleepRows.length });
      check('睡眠时长与选择一致（8 小时）', Math.abs((newest?.duration ?? 0) - 8) < 0.01, { duration: newest?.duration });
      check('睡眠质量按 1-10 存（不是 800 分）', newest?.quality >= 1 && newest?.quality <= 10, { quality: newest?.quality });
    } else {
      check('睡眠步骤可点（已打卡后仍可记录）', false, '没找到 8h 按钮');
    }

    let shot = await cdp.screenshot(OUT);
    console.log(`\n📸 首页截图：${shot.file}（${shot.width}×${shot.height}）`);

    // ── 5. 关键：成长页数字是否跟着更新 ──
    const after = (await apiGet(`${API}/api/extra/growth`, token)).data;
    check('接口侧：本次签到已计入累计打卡',
      after.counters.totalCheckIns === (already ? before.counters.totalCheckIns : before.counters.totalCheckIns + 1),
      { before: before.counters.totalCheckIns, after: after.counters.totalCheckIns, already });
    check('接口侧：连续天数 ≥ 1（今天有记录）', after.counters.streakDays >= 1, { streak: after.counters.streakDays });

    await goto(cdp, `${APP}/growth`);
    await cdp.waitForText((t) => t.includes('指标板块'), { timeoutMs: 20000 });

    /** 从指标卡里读数（排除顶部含「成长指数」的大卡，避免匹配到摘要句子） */
    const cardNumber = (label) => cdp.eval(`(() => {
      const cards = [...document.querySelectorAll('.ant-card')].filter((c) => !(c.innerText||'').includes('成长指数'));
      const card = cards.find((c) => (c.innerText||'').includes(${JSON.stringify(label)}));
      if (!card) return null;
      const m = (card.innerText||'').match(/(\\d+)\\s*天/);
      return m ? Number(m[1]) : null;
    })()`);

    const domStreak = await cardNumber('连续打卡');
    const domTotal = await cardNumber('累计打卡');
    check('成长页「连续打卡」= 接口值（签到后立刻更新）', domStreak === after.counters.streakDays,
      { dom: domStreak, api: after.counters.streakDays });
    check('成长页「累计打卡」= 接口值', domTotal === after.counters.totalCheckIns,
      { dom: domTotal, api: after.counters.totalCheckIns });

    const growthText = await cdp.eval('document.body.innerText');
    check('连续打卡 0 天时不再挂绿色「已记录」标签',
      after.counters.streakDays > 0 || !/连续打卡[\\s\\S]{0,30}?已记录/.test(growthText));

    shot = await cdp.screenshot(OUT_GROWTH);
    console.log(`📸 成长页截图：${shot.file}（${shot.width}×${shot.height}）`);

    const antdWarnings = cdp.consoleErrors.filter((m) => /Warning: \[antd:/.test(m));
    const realErrors = cdp.consoleErrors.filter((m) => !/Warning: \[antd:/.test(m));
    check('整条流程无 console 报错', realErrors.length === 0, realErrors.slice(0, 3));
    if (antdWarnings.length) {
      console.log(`  ℹ️  antd 既有提示 ${antdWarnings.length} 条（不计失败）：${antdWarnings[0].slice(0, 80)}`);
    }

    const failed = CHECKS.filter((c) => !c.ok);
    console.log(`\n结果：${CHECKS.length - failed.length}/${CHECKS.length} 项通过`);
    if (failed.length) {
      console.log('失败项：' + failed.map((f) => f.name).join(' / '));
      process.exitCode = 1;
    }
  } finally {
    await close();
    if (isTemp) {
      await prisma.user.delete({ where: { email } }).catch(() => {});
      console.log(`🧹 已清理探针账号 ${email}`);
    } else {
      console.log(`ℹ️  真实账号 ${email} 未做清理（本次签到/睡眠记录保留）`);
    }
    if (!fs.existsSync(OUT)) console.log('（没有生成截图：流程提前失败）');
  }
}

main()
  .catch((e) => { console.error('探针失败：', e); process.exitCode = 1; })
  .finally(() => prisma.$disconnect());
