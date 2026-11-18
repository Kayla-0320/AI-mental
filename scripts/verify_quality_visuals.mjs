/** 真实质量组件的证据口径与窄屏验证；界面记录是自编夹具，不写业务数据。 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { launchBrowser, goto, sleep, REPO } from './lib/cdp.mjs';

const origin = process.env.QUALITY_VITE || 'http://localhost:5173';
const output = path.join(REPO, 'output/quality-startup');
const clean = { available: true, observed_modalities: ['text', 'visual'], expert_names: ['text', 'visual'],
  expert_weights: [.6, .4], warnings: [], knowledge_matches: [], quality: { visual: {
    quality_score: 1, reason_codes: ['frame_timing_unverified', 'target_face_presence_unverified'], metrics: {
      frame_count: 3, luminance_mean: 118.3, contrast_mean: 42.8, laplacian_variance_128px: 180.2,
      dark_frame_fraction: 0, overexposed_frame_fraction: 0, low_contrast_frame_fraction: 0, blur_proxy_fraction: 0,
      quality_mos_worst: 3.8,
    },
  } } };
const poor = { ...clean, observed_modalities: ['text'], expert_names: ['text'], expert_weights: [1], warnings: ['visual_quality_fallback_to_text'], quality: { visual: {
  quality_score: .25, reason_codes: ['underexposure_proxy', 'blur_proxy'], metrics: {
    ...clean.quality.visual.metrics, luminance_mean: 8.7, dark_frame_fraction: 1, blur_proxy_fraction: 2 / 3,
  },
} } };
const browser = await launchBrowser({ port: 9357, width: 1100, height: 1000, extraArgs: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'] });
try {
  const { cdp } = browser;
  await cdp.send('Fetch.enable', { patterns: [{ urlPattern: '*/__quality_visuals' }] });
  cdp.ws.on('message', async raw => {
    const event = JSON.parse(raw.toString());
    if (event.method !== 'Fetch.requestPaused') return;
    await cdp.send('Fetch.fulfillRequest', { requestId: event.params.requestId, responseCode: 200,
      responseHeaders: [{ name: 'Content-Type', value: 'text/html' }], body: Buffer.from(`<html><head><meta name="viewport" content="width=device-width,initial-scale=1"></head><body style="margin:0;background:#f5f2f8">
      <div id="root"></div><script type="module">import RefreshRuntime from '/@react-refresh';RefreshRuntime.injectIntoGlobalHook(window);
      window.$RefreshReg$=()=>{};window.$RefreshSig$=()=>type=>type;window.__vite_plugin_react_preamble_installed__=true;</script></body></html>`).toString('base64') });
  });
  await goto(cdp, `${origin}/__quality_visuals`);
  await cdp.eval(`(async () => {
    const React = (await import('/node_modules/.vite/deps/react.js')).default;
    const ReactDOM = (await import('/node_modules/.vite/deps/react-dom_client.js')).default;
    const Component = (await import('/src/components/QualityEvidence.tsx')).default;
    window.__root = ReactDOM.createRoot(document.getElementById('root'));
    window.__show = cases => window.__root.render(React.createElement('div', {
      style: { display:'flex', flexWrap:'wrap', gap:20, padding:20, alignItems:'start' }
    }, ...cases.map((props, i) => React.createElement('div', { key:i, style:{width:300,maxWidth:'100%',padding:16,background:'#fff',borderRadius:18,boxSizing:'border-box'} },
      React.createElement(Component, props)))));
  })()`);
  const show = async (reference, waiting = false) => {
    await cdp.eval(`window.__show([{ reference:${JSON.stringify(reference)}, waiting:${waiting} }])`);
    await sleep(150);
    return cdp.eval(`document.querySelector('[data-testid="quality-evidence"]').innerText`);
  };
  let text = await show(clean);
  assert.ok(text.includes('已采用画面') && text.includes('图文参考'));
  assert.ok(await cdp.eval(`document.querySelector('[aria-label="曝光：100% 的采样帧未触发该项问题"]') !== null`));
  assert.equal(await cdp.eval(`document.querySelectorAll('.quality-visual-frame-icons .is-recorded').length`), 3);
  text = await show(poor);
  assert.ok(text.includes('已退回文字') && text.includes('文字参考') && text.includes('画面过暗'));
  assert.ok(await cdp.eval(`document.querySelector('[aria-label="曝光：0% 的采样帧未触发该项问题"]') !== null`));
  assert.ok(text.includes('33%'), '三帧有两帧模糊时，未触发模糊的帧占比应为33%');
  const processText = () => cdp.eval('document.querySelector(".quality-process-stage").innerText');
  const chooseStep = async index => {
    await cdp.eval(`document.querySelectorAll('.quality-process-steps button')[${index}].click()`);
    await sleep(100);
  };
  assert.ok((await processText()).includes('实际收到 3 张'));
  await chooseStep(1);
  assert.ok((await processText()).includes('画面过暗') && (await processText()).includes('66.7%'));
  await chooseStep(2);
  assert.ok((await processText()).includes('停止参考') && (await processText()).includes('0.25'));
  fs.mkdirSync(output, { recursive: true });
  await cdp.screenshot(path.join(output, 'quality-correction-gate.png'));
  await chooseStep(3);
  assert.ok((await processText()).includes('已退回文字参考') && (await processText()).includes('100%'));
  assert.equal(await cdp.eval('document.querySelectorAll(".quality-process-weights>div").length'), 1);
  await chooseStep(0);
  await cdp.eval(`[...document.querySelectorAll('.quality-process-controls button')].find(b=>b.textContent.includes('播放过程')).click()`);
  await sleep(2000);
  assert.equal(await cdp.eval(`document.querySelector('.quality-process-steps [aria-current="step"]').innerText.includes('检查问题')`), true);
  await cdp.eval(`[...document.querySelectorAll('.quality-process-controls button')].find(b=>b.textContent.includes('暂停')).click()`);
  await sleep(2000);
  assert.equal(await cdp.eval(`document.querySelector('.quality-process-steps [aria-current="step"]').innerText.includes('检查问题')`), true);
  text = await show({ ...poor, warnings: ['quality_model_fallback_to_text'] });
  assert.ok(text.includes('已退回文字') && !text.includes('已采用画面'), '画质模型退回也必须展示纠偏');
  const shadow = structuredClone(clean);
  shadow.quality.visual.metrics.quality_mos_worst = 1.4;
  text = await show(shadow);
  assert.ok(text.includes('已采用画面') && !text.includes('已退回文字'), '辅助MOS低分不能自行变更本轮决策');
  text = await show({ ...clean, available: false, observed_modalities: ['text'] });
  assert.ok(text.includes('画面暂未采用') && !text.includes('已采用画面'));
  text = await show(null);
  assert.ok(text.includes('等待画面记录') && !text.includes('0%') && !text.includes('100%'));
  text = await show({ ...clean, observed_modalities: ['text'], quality: {} });
  assert.ok(text.includes('本轮仅用文字') && text.includes('文字参考') && !text.includes('图文参考'));
  text = await show(clean, true);
  assert.ok(text.includes('正在等这一轮') && !text.includes('已采用画面') && !text.includes('100%'));
  const partial = structuredClone(clean);
  partial.quality.visual.metrics = { frame_count: 3, luminance_mean: 90 };
  await show(partial);
  assert.equal(await cdp.eval(`document.querySelectorAll('.quality-visual-track.is-empty').length`), 3, '缺失分项不能当作全通过');
  await cdp.eval(`window.__show([{reference:${JSON.stringify(clean)}},{reference:${JSON.stringify(poor)}},{reference:null}])`);
  await sleep(400);
  fs.mkdirSync(output, { recursive: true });
  await cdp.screenshot(path.join(output, 'quality-visual-states.png'));
  await cdp.send('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await show(poor);
  assert.ok(await cdp.eval('document.documentElement.scrollWidth <= innerWidth + 1'));
  await cdp.eval(`document.querySelector('.quality-visual-details').open = true`);
  assert.ok((await cdp.eval('document.body.innerText')).includes('平均亮度'));
  await cdp.screenshot(path.join(output, 'quality-visual-mobile.png'));
  // 实时预览消费同一份假摄像头流；人脸标志与采样数在这里是明确的界面夹具。
  await cdp.eval(`(async () => {
    const React = (await import('/node_modules/.vite/deps/react.js')).default;
    const Live = (await import('/src/components/LiveQualityPreview.tsx')).default;
    window.__manager = (await import('/src/services/mediaStreamManager.ts')).mediaStreamManager;
    window.__count = 0;
    window.__renderLive = flags => window.__root.render(React.createElement('div', {style:{padding:20,width:320}},
      React.createElement(Live, {state:{enabled:false,detecting:false,facePresent:false,reason:'',getSampleCount:()=>window.__count,...flags},active:true})));
    window.__renderLive({});
  })()`);
  await sleep(400);
  assert.ok((await cdp.eval('document.body.innerText')).includes('摄像头未开启'));
  assert.equal(await cdp.eval('window.__manager.has("camera")'), false, '预览不能自行申请摄像头');
  await cdp.eval(`(async () => {
    window.__stream = await window.__manager.acquire('camera','quality-live-test',{video:true});
    window.__renderLive({enabled:true});
  })()`);
  await sleep(600);
  assert.ok((await cdp.eval('document.body.innerText')).includes('检测器准备中'));
  assert.equal(await cdp.eval('document.querySelector("video").srcObject === window.__stream'), true);
  await cdp.eval('window.__renderLive({enabled:true,detecting:true})');
  await sleep(350);
  assert.ok((await cdp.eval('document.body.innerText')).includes('未检测到人脸'));
  assert.equal(await cdp.eval('document.querySelector("video").videoWidth > 0'), true);
  for (const count of [1, 2, 3]) {
    await cdp.eval(`window.__count=${count};window.__renderLive({enabled:true,detecting:true,facePresent:true})`);
    await sleep(350);
    assert.equal(await cdp.eval('document.querySelectorAll(".quality-live-sampling .is-ready").length'), count);
  }
  assert.ok((await cdp.eval('document.body.innerText')).includes('画面已就绪'));
  await cdp.screenshot(path.join(output,'quality-live-preview.png'));
  await cdp.eval('window.__root.unmount()');
  assert.equal(await cdp.eval('window.__stream.getVideoTracks()[0].readyState'), 'live', '卸载预览不能关闭共享流');
  await cdp.eval('window.__manager.release("camera","quality-live-test")');
  assert.equal(await cdp.eval('window.__stream.getVideoTracks()[0].readyState'), 'ended');
  const errors = cdp.consoleErrors.filter(message => !message.startsWith('[vite] failed to connect'));
  assert.equal(errors.length, 0, JSON.stringify(errors));
  const report = { status: 'passed', scope: 'real React component; synthetic quality records; no business writes',
    cases: ['adopted', 'rule_fallback', 'model_fallback', 'shadow_mos', 'unavailable', 'no_record', 'text_only', 'pending_turn', 'partial_metrics'],
    process_replay: true, pause_and_step_selection: true, recorded_gating_and_final_weights: true,
    live_preview: true, sampling_progress: true, shared_stream_preserved: true, mobile_overflow: false, consoleErrors: errors };
  fs.writeFileSync(path.join(output, 'visual-verification.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
} finally { await browser.close(); }
process.exit(0);
