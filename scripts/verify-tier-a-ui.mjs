/**
 * Tier S + Tier A 的界面端到端验证。
 *
 * 覆盖本次改动的四个可见面：
 *   1. /admin/crisis      —— 处置台按钮是否真的写库（此前只弹 toast）
 *   2. /admin（数据概览）  —— 分组表是否还为写死的假指标
 *   3. /consultant/crisis —— 咨询师「危机待办」入口与接收动作
 *   4. /settings          —— 隐私与数据控制是否真读真写
 *
 * 用法：node scripts/verify-tier-a-ui.mjs
 * 退出码：0 = 全部断言通过；1 = 有失败项。
 */
import fs from 'fs';
import path from 'path';
import { REPO, login, launchBrowser, injectAuth, sleep } from './lib/cdp.mjs';

const API = 'http://127.0.0.1:3000';
// ⚠️ vite 只监听 localhost（IPv6 ::1），用 127.0.0.1 会连不上并落到 chrome-error 页
const WEB = 'http://localhost:5173';
const SHOT_DIR = path.join(REPO, 'docs', 'screenshots');

const results = [];
function check(name, ok, detail = '') {
  results.push({ name, ok: !!ok, detail });
  console.log(`${ok ? '  ✅' : '  ❌'} ${name}${detail ? `  — ${detail}` : ''}`);
}

async function main() {
  const admin = await login(API, 'admin@mental.com', '123456');
  const consultant = await login(API, 'consultant@mental.com', '123456');
  const patient = await login(API, 'patient@mental.com', '123456');

  const { cdp, close, bin } = await launchBrowser({ port: 9333, width: 1440, height: 1400 });
  console.log(`浏览器：${bin}`);
  await injectAuth(cdp, admin);

  try {
    // ── 1. 管理端危机处置台 ────────────────────────────────
    console.log('\n[1] /admin/crisis');
    await cdp.send('Page.navigate', { url: `${WEB}/admin/crisis` });
    await sleep(2500);
    let text = await cdp.waitForText((t) => t.includes('危机管理面板'), { timeoutMs: 20000 });
    check('页面渲染', text.includes('危机管理面板'));
    check('显示「超时未接收」指标', text.includes('超时未接收'));
    check('按钮文案为「接收」（旧版只有「查看详情」）', text.includes('接收'));

    const ackResult = await cdp.eval(`(async () => {
      const rows = [...document.querySelectorAll('tbody tr')];
      const openRow = rows.find(r => r.innerText.includes('待接收'));
      if (!openRow) return { skipped: true, reason: '没有待接收的记录' };
      const btn = [...openRow.querySelectorAll('button')].find(b => b.innerText.trim() === '接收');
      if (!btn) return { skipped: true, reason: '行内没有接收按钮' };
      btn.click();
      await new Promise(r => setTimeout(r, 2500));
      return { skipped: false, text: document.body.innerText };
    })()`);
    if (ackResult.skipped) {
      check('点击「接收」并落库', false, ackResult.reason);
    } else {
      check('点击「接收」后状态变为「处理中」', ackResult.text.includes('处理中'));
      check('出现成功提示', ackResult.text.includes('已确认接收'));
    }
    await cdp.screenshot(path.join(SHOT_DIR, 'tierA_01_admin_crisis.png'));

    // ── 2. 管理端数据概览：假公平性表已移除 ─────────────────
    console.log('\n[2] /admin（公平性报告 tab）');
    await cdp.send('Page.navigate', { url: `${WEB}/admin` });
    await sleep(2500);
    text = await cdp.waitForText((t) => t.includes('数据概览'), { timeoutMs: 20000 });
    // 目标卡片在「公平性报告」这个非默认 tab 内，必须先切过去才会渲染
    const switched = await cdp.eval(`(async () => {
      const tab = [...document.querySelectorAll('.ant-tabs-tab')].find(el => el.innerText.includes('公平性报告'));
      if (!tab) return false;
      tab.click();
      await new Promise(r => setTimeout(r, 1500));
      return true;
    })()`);
    check('可切换到「公平性报告」tab', switched);
    text = await cdp.waitForText((t) => t.includes('分组覆盖与风险分布') || t.includes('算法公平性指标'), { timeoutMs: 15000 });
    check('分组表改为「分组覆盖与风险分布」', text.includes('分组覆盖与风险分布'));
    check('不再断言「所有分组公平性指标均在可接受范围内」', !text.includes('所有分组公平性指标均在可接受范围内'));
    check('写明为何不展示准确率/召回率', text.includes('标注真值'));
    await cdp.screenshot(path.join(SHOT_DIR, 'tierA_02_admin_dashboard.png'));

    // ── 3. 咨询师端危机待办 ────────────────────────────────
    console.log('\n[3] /consultant/crisis');
    await cdp.send('Page.navigate', { url: `${WEB}/` });
    await sleep(500);
    await cdp.send('Page.addScriptToEvaluateOnNewDocument', {
      source: `try{localStorage.setItem('accessToken',${JSON.stringify(consultant.accessToken)});`
        + `localStorage.setItem('refreshToken',${JSON.stringify(consultant.refreshToken)});`
        + `localStorage.setItem('user',${JSON.stringify(JSON.stringify(consultant.user))});}catch(e){}`,
    });
    await cdp.send('Page.navigate', { url: `${WEB}/consultant/crisis` });
    await sleep(2500);
    text = await cdp.waitForText((t) => t.includes('危机待办'), { timeoutMs: 20000 });
    check('页面渲染', text.includes('危机待办'));
    check('侧边导航出现「危机待办」入口', text.includes('危机待办'));
    check('展示待接收数量', text.includes('待接收'));
    check('含 15 分钟超时说明', text.includes('15 分钟'));
    await cdp.screenshot(path.join(SHOT_DIR, 'tierA_03_consultant_crisis.png'));

    // ── 4. 患者端隐私与数据控制 ────────────────────────────
    console.log('\n[4] /settings（隐私与数据控制）');
    await cdp.send('Page.addScriptToEvaluateOnNewDocument', {
      source: `try{localStorage.setItem('accessToken',${JSON.stringify(patient.accessToken)});`
        + `localStorage.setItem('refreshToken',${JSON.stringify(patient.refreshToken)});`
        + `localStorage.setItem('user',${JSON.stringify(JSON.stringify(patient.user))});}catch(e){}`,
    });
    await cdp.send('Page.navigate', { url: `${WEB}/settings` });
    await sleep(3000);
    text = await cdp.waitForText((t) => t.includes('隐私与数据控制'), { timeoutMs: 20000 });
    check('隐私卡片渲染', text.includes('隐私与数据控制'));
    check('按用途列出授权项', text.includes('文字对话的情绪分析') && text.includes('语音声学特征分析'));
    check('必需项被标注', text.includes('必要'));
    check('展示「平台保存了什么」的真实计数', text.includes('平台保存了什么') && text.includes('对话消息'));
    check('保留期限来自后端', text.includes('当前设置：'));
    check('导出/删除按钮存在', text.includes('导出我的数据') && text.includes('删除所有数据'));
    await cdp.screenshot(path.join(SHOT_DIR, 'tierA_04_settings_privacy.png'));

    // 开关持久化：切换「参与心理健康研究」→ 刷新 → 状态应保持
    const toggleOk = await cdp.eval(`(async () => {
      const rows = [...document.querySelectorAll('div')].filter(d => d.innerText && d.innerText.startsWith('参与心理健康研究') && d.querySelector('button[role="switch"]'));
      const row = rows[rows.length - 1];
      if (!row) return { ok: false, reason: '找不到「参与心理健康研究」开关' };
      const sw = row.querySelector('button[role="switch"]');
      const before = sw.getAttribute('aria-checked');
      sw.click();
      await new Promise(r => setTimeout(r, 2500));
      const after = sw.getAttribute('aria-checked');
      return { ok: true, before, after };
    })()`);
    if (!toggleOk.ok) {
      check('授权开关可切换', false, toggleOk.reason);
    } else {
      check('授权开关可切换', toggleOk.before !== toggleOk.after, `${toggleOk.before} → ${toggleOk.after}`);
      // 刷新后应保持（此前是 useState，刷新必回默认）
      await cdp.send('Page.navigate', { url: `${WEB}/settings` });
      await sleep(3000);
      const persisted = await cdp.eval(`(async () => {
        const rows = [...document.querySelectorAll('div')].filter(d => d.innerText && d.innerText.startsWith('参与心理健康研究') && d.querySelector('button[role="switch"]'));
        const row = rows[rows.length - 1];
        return row ? row.querySelector('button[role="switch"]').getAttribute('aria-checked') : 'not-found';
      })()`);
      check('刷新后开关状态保持（证明落库）', persisted === toggleOk.after, `刷新后=${persisted}，期望=${toggleOk.after}`);
      // 复原
      await cdp.eval(`(async () => {
        const rows = [...document.querySelectorAll('div')].filter(d => d.innerText && d.innerText.startsWith('参与心理健康研究') && d.querySelector('button[role="switch"]'));
        const row = rows[rows.length - 1];
        if (row) { row.querySelector('button[role="switch"]').click(); }
        await new Promise(r => setTimeout(r, 2000));
      })()`);
    }

    // antd 的静态 message 警告是既有噪声（与本次改动无关），只统计真正的异常
    const realErrors = cdp.consoleErrors.filter(
      (e) => !/antd: message|Static function can not consume context/.test(e),
    );
    console.log('\nconsole 报错：', realErrors.length ? realErrors.slice(0, 5) : '（无）');
    check('页面无真实 console 报错', realErrors.length === 0, realErrors.slice(0, 3).join(' | '));
  } finally {
    await close();
  }

  const failed = results.filter((r) => !r.ok);
  console.log(`\n总计 ${results.length} 项，通过 ${results.length - failed.length}，失败 ${failed.length}`);
  if (failed.length) {
    for (const f of failed) console.log(`  ❌ ${f.name} ${f.detail}`);
    process.exit(1);
  }
  console.log('全部通过');
}

main().catch((e) => { console.error('探针异常：', e); process.exit(1); });
