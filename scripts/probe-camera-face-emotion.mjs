/**
 * 摄像头探针：多模态开关打开后，摄像头到底有没有真的起来。
 *
 * 为什么需要它：用户报「摄像头在前端显示打不开了」。这类问题有两种完全不同的病因，
 * 必须分开，否则会去改错的地方：
 *   A. **真回归**：代码/模型改动让取流链路挂了 —— 全新加载的页面也起不来；
 *   B. **热更新残留**：`useFacialAnalysis.start()` 用 `startingRef`/`isActiveRef` 做幂等，
 *      Vite HMR 把 hook 重挂载时若上一次 `start()` 的 promise 被孤立，
 *      这两个 ref 就永久卡住 → 此后点开关**没有任何反应**，但刷新页面即恢复。
 * 本探针用**全新 profile + 全新页面**跑一遍，就能判定属于哪一种。
 *
 * 用假摄像头（`--use-fake-device-for-media-stream`）跑，不碰真实设备。
 *
 * 用法
 *   node scripts/probe-camera-face-emotion.mjs
 *   ... --keep        # 不清账号（默认注册的临时账号会删掉）
 */
import { createRequire } from 'module';
import { login, launchBrowser, injectAuth, goto, sleep } from './lib/cdp.mjs';

const require = createRequire(import.meta.url);
const API = 'http://127.0.0.1:3000';
const APP = 'http://localhost:5173';
const PORT = 9333;

function argOf(flag) {
  const hit = process.argv.find((a) => a.startsWith(`${flag}=`));
  return hit ? hit.split('=').slice(1).join('=') : undefined;
}

async function main() {
  const email = argOf('--email') || `camera-probe-${Date.now()}@mental.com`;
  const password = argOf('--password') || '123456';
  const isTemp = !argOf('--email');

  await fetch(`${API}/api/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password, nickname: '摄像头探针' }),
  }).catch(() => {});
  const auth = await login(API, email, password);
  console.log(`账号 ${email}`);

  const { cdp, close, bin } = await launchBrowser({
    port: PORT, width: 1440, height: 1000,
    extraArgs: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'],
  });
  console.log(`浏览器 ${bin}`);

  try {
    await injectAuth(cdp, auth);
    await goto(cdp, `${APP}/`, { waitMs: 2500 });

    // 0) 页面本身渲染出来了吗（渲染崩溃会表现为"打不开"）
    const bodyLen = await cdp.eval('document.body.innerText.length').catch(() => 0);
    console.log(`\n[页面] body 文本长度 = ${bodyLen}`);

    // 1) 设备与权限：先看"这台浏览器里 getUserMedia 本身能不能用"
    const devProbe = await cdp.eval(`(async () => {
      try {
        const s = await navigator.mediaDevices.getUserMedia({ video: true });
        const t = s.getVideoTracks()[0];
        const info = { ok: true, label: t && t.label, state: t && t.readyState,
                       settings: t && t.getSettings ? t.getSettings() : null };
        s.getTracks().forEach((x) => x.stop());
        return JSON.stringify(info);
      } catch (e) { return JSON.stringify({ ok: false, name: e.name, message: e.message }); }
    })()`).catch((e) => `eval 失败: ${e}`);
    console.log(`[设备] 直接 getUserMedia = ${devProbe}`);

    // 2) 打开 Popover，再点里面的开关。
    //    ⚠️ 外层触发按钮的 aria-label 是「多模态感知：已关闭」，**也含「多模态」**，
    //    直接用模糊匹配会点到触发器上（只弹窗、不启动摄像头）—— 第一版探针就栽在这，
    //    表现为"点了但 console 里没有 [多模态] 开启摄像头/麦克风..."。
    const trigger = await cdp.eval(`(() => {
      const el = document.querySelector('[data-quick="multimodal-open"]');
      if (!el) return 'TRIGGER_NOT_FOUND';
      el.click();
      return 'OPENED';
    })()`);
    console.log(`[开关] 打开 Popover = ${trigger}`);
    await sleep(800);

    const clicked = await cdp.eval(`(() => {
      const el = document.querySelector('[aria-label="多模态感知开关"]');
      if (!el) return 'SWITCH_NOT_FOUND';
      el.click();
      return 'CLICKED';
    })()`);
    console.log(`[开关] 点击结果 = ${clicked}`);
    if (clicked !== 'CLICKED') {
      console.log('❗ 没找到开关 —— 见上面「打开 Popover」的结果');
    }

    // 3) 等链路起来（模型 15 MB + ORT wasm 首次加载 + MediaPipe 初始化）
    const waitSec = Number(argOf('--wait') || 60);
    for (let i = 0; i < Math.ceil(waitSec / 2.5); i += 1) {
      await sleep(2500);
      if (i % 4 === 3) console.log(`  等待 ${Math.round((i + 1) * 2.5)}s`);
    }

    // 3a) 表情通道**自身的状态**（工具栏收起态那句话就是它的状态机）
    const faceState = await cdp.eval(`(() => {
      const t = document.body.innerText;
      const m = ['模型加载中…','模型不可用','等待画面…','识别中…','画面中没有人脸','未开启摄像头'].filter((k) => t.includes(k));
      return JSON.stringify(m);
    })()`).catch((e) => `eval 失败: ${e}`);
    console.log(`[表情通道状态] ${faceState}`);

    // 3b) 开关最终状态（aria-label 会跟着变）
    const ariaNow = await cdp.eval(
      `document.querySelector('[data-quick="multimodal-open"]')?.getAttribute('aria-label') ?? '(找不到开关)'`,
    ).catch(() => '?');
    console.log(`\n[开关状态] ${ariaNow}`);

    // 3c) 浏览器到底有没有去取那个模型（直接看资源计时，不靠猜）
    const res = await cdp.eval(`(() => JSON.stringify(
      performance.getEntriesByType('resource')
        .filter((e) => /models\\/|emotion|\\.onnx/.test(e.name))
        .map((e) => ({ f: e.name.split('/').pop(), bytes: e.transferSize, ms: Math.round(e.duration) }))
    ))()`).catch((e) => `eval 失败: ${e}`);
    console.log(`[网络] 模型资源 = ${res}`);

    // 4) 摄像头有没有真的在跑：DOM 里的 video / 流 / 轨道
    const videoState = await cdp.eval(`(() => {
      const vs = [...document.querySelectorAll('video')];
      return JSON.stringify({
        count: vs.length,
        videos: vs.map((v) => ({
          w: v.videoWidth, h: v.videoHeight, readyState: v.readyState,
          paused: v.paused, hasStream: !!v.srcObject,
          display: getComputedStyle(v).display, visible: v.offsetParent !== null,
          tracks: v.srcObject ? v.srcObject.getVideoTracks().map((t) => t.readyState + ':' + t.label) : [],
        })),
      });
    })()`).catch((e) => `eval 失败: ${e}`);
    console.log(`\n[画面] ${videoState}`);

    // 5) 左下角工具栏文案（表情通道状态就写在这里）
    const dockText = await cdp.eval(`(() => {
      const hits = [...document.querySelectorAll('div')]
        .filter((d) => /面部表情|其他表情|眨眼|感知中|暂无新读数/.test(d.textContent || ''))
        .map((d) => (d.textContent || '').trim())
        .filter((t) => t.length > 0);
      // 最短的那个最接近工具栏本体（外层容器会把整页文本都包进去）
      hits.sort((a, b) => a.length - b.length);
      return JSON.stringify(hits.slice(0, 2));
    })()`).catch((e) => `eval 失败: ${e}`);
    console.log(`[工具栏] ${dockText}`);

    // 6) console：链路关键日志 + 错误
    console.log('\n[console 关键日志]');
    cdp.dumpConsole('多模态|面部分析|表情模型|摄像头|getUserMedia|onnx|ORT');
    console.log('\n[console 错误]');
    if (cdp.consoleErrors.length === 0) console.log('  （无）');
    else cdp.consoleErrors.slice(0, 12).forEach((e) => console.log(`  ❌ ${String(e).slice(0, 300)}`));
    console.log(`\n[console 全部尾部（最后 25 条，未过滤）]`);
    cdp.consoleAll.slice(-25).forEach((m) => console.log(`  [${m.type}] ${m.text.slice(0, 240)}`));

    await cdp.screenshot('D:/Develop/AIC/_verify_camera/probe.png').catch(() => {});

    // 7) **浏览器里**能不能真的把新模型跑起来（与页面状态无关的独立验证）。
    //    为什么单独测：应用里是先试 WebGPU、失败再退 wasm；如果 WebGPU 建会话**卡住**
    //    （不是抛错），表情通道就会永远停在「识别中…」。
    console.log('\n[浏览器内模型自检] 用与线上相同的 ORT 版本 + 相同预处理跑一次 224px 前向…');
    const ortCheck = await cdp.eval(`(async () => {
      const out = { steps: [] };
      try {
        const ort = await import('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.mjs');
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
        out.steps.push('ORT ' + (ort.env?.versions?.web || '?') + ' 已加载');
        let sess = null, provider = null;
        const t0 = performance.now();
        try {
          sess = await ort.InferenceSession.create('/models/face_emotion_8cls_hsemotion.onnx', { executionProviders: ['webgpu'] });
          provider = 'webgpu';
        } catch (e) {
          out.steps.push('webgpu 建会话失败（会退 wasm）：' + (e && e.message ? e.message.slice(0,120) : e));
          sess = await ort.InferenceSession.create('/models/face_emotion_8cls_hsemotion.onnx', { executionProviders: ['wasm'] });
          provider = 'wasm';
        }
        out.loadMs = Math.round(performance.now() - t0);
        out.provider = provider;
        out.inputs = sess.inputNames;
        const data = new Float32Array(3 * 224 * 224).fill(0.4);
        const t1 = performance.now();
        const r = await sess.run({ [sess.inputNames[0]]: new ort.Tensor('float32', data, [1, 3, 224, 224]) });
        out.inferMs = Math.round(performance.now() - t1);
        const logits = r[Object.keys(r)[0]].data;
        out.outLen = logits.length;
        out.outHead = Array.from(logits).slice(0, 4).map((v) => Math.round(v * 1000) / 1000);
        sess.release && sess.release();
        out.ok = out.outLen === 8;
      } catch (e) { out.ok = false; out.err = String(e).slice(0, 300); }
      return JSON.stringify(out);
    })()`).catch((e) => `eval 失败: ${e}`);
    console.log(`[浏览器内模型自检] ${ortCheck}`);
    console.log('\n截图（如果成功）：D:/Develop/AIC/_verify_camera/probe.png');
  } finally {
    await close();
    if (isTemp && !argOf('--keep')) {
      await fetch(`${API}/api/probe/cleanup?email=${encodeURIComponent(email)}`).catch(() => {});
      console.log(`ℹ️  临时账号 ${email} 需手工清理（没有清理接口时请忽略）`);
    }
  }
}

main().catch((e) => { console.error('[探针失败]', e); process.exit(1); });
