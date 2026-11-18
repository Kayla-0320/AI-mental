/**
 * 「我的成长」指标板块页面探针。
 *
 * 做四件事：
 *   1. 用真实接口登录演示账号，把 token 注进 localStorage（页面加载前）；
 *   2. 打开 /growth，等页面把指标渲染出来；
 *   3. 断言页面上真的出现了指标板块的关键内容，并收集 console 报错；
 *   4. 整页截图，落盘到 docs/screenshots/。
 *
 * 用法：
 *   node scripts/probe-growth-page.mjs
 *   node scripts/probe-growth-page.mjs --email=patient@mental.com --out=docs/screenshots/xx.png
 * 退出码：0 = 全部断言通过；1 = 有失败项。
 */

import fs from 'fs';
import path from 'path';
import { REPO, login as apiLogin, launchBrowser, injectAuth, sleep } from './lib/cdp.mjs';

const args = process.argv.slice(2);
const argOf = (n) => {
  const hit = args.find((a) => a.startsWith(`--${n}=`));
  return hit ? hit.split('=').slice(1).join('=') : null;
};
const email = argOf('email') || 'growth-demo@mental.com';
const password = argOf('password') || '123456';
const APP = argOf('app') || 'http://localhost:5173';
const API = argOf('api') || 'http://127.0.0.1:3000';
const outPath = path.resolve(REPO, argOf('out') || 'docs/screenshots/26_growth_metrics.png');
const PORT = 9339;

const EDGE_CANDIDATES = [
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
];

class Cdp {
  constructor(ws) {
    this.ws = ws;
    this.seq = 0;
    this.pending = new Map();
    this.consoleErrors = [];
    ws.on('message', (raw) => {
      let msg;
      try { msg = JSON.parse(raw.toString()); } catch { return; }
      if (msg.method === 'Runtime.consoleAPICalled' && msg.params.type === 'error') {
        this.consoleErrors.push((msg.params.args || []).map((a) => a.value || a.description || '').join(' '));
      }
      if (msg.method === 'Runtime.exceptionThrown') {
        this.consoleErrors.push(msg.params?.exceptionDetails?.exception?.description || 'exception');
      }
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (msg.error) reject(new Error(`${msg.error.message}`));
        else resolve(msg.result);
      }
    });
  }
  send(method, params = {}) {
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
      setTimeout(() => {
        if (this.pending.has(id)) { this.pending.delete(id); reject(new Error(`CDP timeout: ${method}`)); }
      }, 60000);
    });
  }
  async eval(expression) {
    const r = await this.send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true, userGesture: true });
    if (r.exceptionDetails) throw new Error(`页面内异常: ${JSON.stringify(r.exceptionDetails).slice(0, 400)}`);
    return r.result.value;
  }
}

const CHECKS = [];
const check = (name, ok, detail) => {
  CHECKS.push({ name, ok: !!ok });
  console.log(`${ok ? '  ✅' : '  ❌'} ${name}${detail === undefined ? '' : `  ${JSON.stringify(detail)}`}`);
};

async function main() {
  // ── 1. 登录拿 token ──
  const auth = await apiLogin(API, email, password);
  const accessToken = auth.accessToken;

  const { cdp, close } = await launchBrowser({ port: PORT });

  try {
    // 注入登录态（详见 scripts/lib/cdp.mjs 的说明）
    await injectAuth(cdp, auth);

    const loaded = new Promise((resolve) => {
      const onMsg = (raw) => {
        try {
          const m = JSON.parse(raw.toString());
          if (m.method === 'Page.loadEventFired') { cdp.ws.off('message', onMsg); resolve(); }
        } catch { /* ignore */ }
      };
      cdp.ws.on('message', onMsg);
    });

    await cdp.send('Page.navigate', { url: `${APP}/growth` });
    await Promise.race([loaded, sleep(15000)]);

    // 等指标渲染出来（接口 + recharts 需要一点时间）
    let text = '';
    for (let i = 0; i < 40; i += 1) {
      await sleep(400);
      text = await cdp.eval('document.body.innerText').catch(() => '');
      if (text.includes('成长指数') && text.includes('徽章墙')) break;
    }

    const url = await cdp.eval('location.pathname');
    check('停在 /growth（未被踢回登录页）', url === '/growth', url);
    check('渲染出成长指数', text.includes('成长指数'));
    check('渲染出等级', /Lv\.\d/.test(text), (text.match(/Lv\.\d+\s*\S*/) || [])[0]);
    check('渲染出指标板块', text.includes('指标板块'));
    check('四个指标分组齐全', ['坚持', '心情', '投入', '探索'].every((g) => text.includes(g)));
    check('渲染出徽章墙与解锁计数', text.includes('徽章墙') && /\d+\s*\/\s*\d+\s*已解锁/.test(text),
      (text.match(/\d+\s*\/\s*\d+\s*已解锁/) || [])[0]);
    check('渲染出下一步建议', text.includes('下一步可以做'));
    check('心情轨迹与热力图区块存在', text.includes('近 14 天心情轨迹') && text.includes('近 30 天打卡'));

    const cards = await cdp.eval(`[...document.querySelectorAll('.cloud-card')].length`);
    check('指标卡已挂载（cloud-card 数量 > 8）', cards > 8, cards);

    const chartPath = await cdp.eval(`document.querySelectorAll('.recharts-line-curve').length`);
    // 没有近 14 天记录时应当显示空态引导，而不是画一条空线或一条贴底的假线
    const hasEmptyGuide = text.includes('还没有心情记录');
    check('心情区要么画出折线、要么给出空态引导（不会是空白）', chartPath >= 1 || hasEmptyGuide,
      { lines: chartPath, emptyGuide: hasEmptyGuide });

    const bodyHeight = await cdp.eval('document.body.scrollHeight');
    check('页面有实际内容高度', bodyHeight > 1200, bodyHeight);

    // 顶部指数卡是白字，必须真的铺上了渐变底 —— 否则就是「白字白底」的隐形内容。
    // （innerText 断言查不出这种问题：文字在 DOM 里，只是看不见。）
    const skin = await cdp.eval(`(() => {
      const cards = [...document.querySelectorAll('.ant-card')];
      const idx = cards.find((c) => (c.innerText || '').includes('成长指数'));
      if (!idx) return { found: false };
      const body = idx.querySelector('.ant-card-body');
      return {
        found: true,
        cardBgImage: getComputedStyle(idx).backgroundImage,
        cardBgColor: getComputedStyle(idx).backgroundColor,
        bodyBgImage: body ? getComputedStyle(body).backgroundImage : '',
        bodyBgColor: body ? getComputedStyle(body).backgroundColor : '',
      };
    })()`);
    const hasGradient = /linear-gradient/.test(skin.cardBgImage || '') || /linear-gradient/.test(skin.bodyBgImage || '');
    check('指数卡铺上了渐变底（白字不会白底隐形）', hasGradient, skin);

    check('页面无 console 报错', cdp.consoleErrors.length === 0, cdp.consoleErrors.slice(0, 3));

    // ── 整页截图 ──
    const metrics = await cdp.send('Page.getLayoutMetrics');
    const width = Math.ceil(metrics.cssContentSize.width);
    const height = Math.min(Math.ceil(metrics.cssContentSize.height), 6000);
    const shot = await cdp.send('Page.captureScreenshot', {
      format: 'png',
      captureBeyondViewport: true,
      clip: { x: 0, y: 0, width, height, scale: 1 },
    });
    fs.mkdirSync(path.dirname(outPath), { recursive: true });
    fs.writeFileSync(outPath, Buffer.from(shot.data, 'base64'));
    console.log(`\n📸 截图：${outPath}（${width}×${height}）`);
    console.log(`   账号：${email}`);

    // ── 可选：顺带验证首页打卡状态（G6：旧实现恒显示「未打卡 / 连续 0 天」）──
    // 必须放在截图之后，否则会截到首页而不是成长页。
    if (args.includes('--home=1')) {
      const todayRes = await fetch(`${API}/api/mood/checkin/today`, { headers: { Authorization: `Bearer ${accessToken}` } });
      const checkedIn = !!(await todayRes.json())?.data?.checkedIn;

      await cdp.send('Page.navigate', { url: `${APP}/` });
      // 按接口报的真实状态去等，别用「首屏文案出现了就以为加载完了」：
      // 首屏先渲染「今天心情怎么样」，checkTodayCheckIn 是异步的，
      // 抢在它回来之前读文本会把「已打卡」误判成「未打卡」（探针自己踩过一次）。
      let homeText = checkedIn
        ? await cdp.waitForText((t) => t.includes('今天已经打过卡'), { timeoutMs: 20000 })
        : await cdp.waitForText((t) => t.includes('今天心情怎么样'), { timeoutMs: 20000 });
      if (!checkedIn) {
        await sleep(1500); // 再等一拍，确保 checkTodayCheckIn 已经给出否定结论
        homeText = await cdp.eval('document.body.innerText').catch(() => homeText);
      }
      check('首页能正确显示「今天已打卡」',
        checkedIn ? homeText.includes('今天已经打过卡') : !homeText.includes('今天已经打过卡'),
        { checkedIn });

      const growth = (await (await fetch(`${API}/api/extra/growth`, { headers: { Authorization: `Bearer ${accessToken}` } })).json()).data;
      const streakOnHome = Number((homeText.match(/连续\s*(\d+)\s*天/) || [])[1] || 0);
      check('首页连续天数与成长接口同一口径', streakOnHome === growth.counters.streakDays,
        { home: streakOnHome, api: growth.counters.streakDays });

      const homeShotPath = path.join(path.dirname(outPath), '28_home_checkin_state.png');
      const homeMetrics = await cdp.send('Page.getLayoutMetrics');
      const homeShot = await cdp.send('Page.captureScreenshot', {
        format: 'png',
        captureBeyondViewport: true,
        clip: {
          x: 0, y: 0,
          width: Math.ceil(homeMetrics.cssContentSize.width),
          height: Math.min(Math.ceil(homeMetrics.cssContentSize.height), 4000),
          scale: 1,
        },
      });
      fs.writeFileSync(homeShotPath, Buffer.from(homeShot.data, 'base64'));
      console.log(`📸 首页截图：${homeShotPath}`);
    }

    const failed = CHECKS.filter((c) => !c.ok);
    console.log(`\n结果：${CHECKS.length - failed.length}/${CHECKS.length} 项通过`);
    if (failed.length) process.exitCode = 1;
  } finally {
    await close();
  }
}

main().catch((e) => {
  console.error('探针失败：', e);
  process.exitCode = 1;
});
