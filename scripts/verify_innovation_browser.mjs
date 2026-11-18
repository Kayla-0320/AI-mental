/** 真实React组件及浏览器canvas探针，接口状态使用明确的自编夹具。 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { launchBrowser, goto, REPO } from './lib/cdp.mjs';

const origin = process.env.INNOVATION_VITE || 'http://127.0.0.1:5187';
const output = path.join(REPO, 'output/innovation-integration');
const overview = {
  state: { schema: 'support-agent.v1', revision: 1, enabled: true, preferredTool: 'none', actions: [],
    background: { enabled: false, notifyInApp: false, intervalHours: 24, quietStart: 22, quietEnd: 8, nextRunAt: null, events: [] } },
  trend: { observations: 3, latestScore: 8, mean: 6.67, change: 3, stale: false, label: '最近自评记录，仅描述过去变化',
    points: [{ id: '1', score: 5, at: '2026-11-15T12:00:00+08:00' }, { id: '2', score: 7, at: '2026-11-16T12:00:00+08:00' }, { id: '3', score: 8, at: '2026-11-17T12:00:00+08:00' }],
    forecast: { status: 'experimental', score: 6.68, observedDays: 3, targetDate: '2026-11-18' } },
  suggestionEligible: false, context: { status: 'unavailable', evidence: [], revoked: 0, goal: 'unspecified' }, modelReference: null,
};
const browser = await launchBrowser({ port: 9347, width: 1100, height: 1000 });
try {
  const cdp = browser.cdp;
  await cdp.send('Fetch.enable', { patterns: [{ urlPattern: '*/__innovation_check' }, { urlPattern: '*/api/*' }] });
  cdp.ws.on('message', async raw => {
    const event = JSON.parse(raw.toString());
    if (event.method !== 'Fetch.requestPaused') return;
    const { requestId, request } = event.params;
    const html = request.url.endsWith('/__innovation_check');
    const body = html ? `<html><body style="margin:24px;background:#fafafa"><div id="root"></div><script type="module">import RefreshRuntime from '/@react-refresh';RefreshRuntime.injectIntoGlobalHook(window);window.$RefreshReg$=()=>{};window.$RefreshSig$=()=>type=>type;window.__vite_plugin_react_preamble_installed__=true;</script></body></html>` : JSON.stringify({ code: 0, data: overview });
    await cdp.send('Fetch.fulfillRequest', { requestId, responseCode: 200, responseHeaders: [{ name: 'Content-Type', value: html ? 'text/html' : 'application/json' }], body: Buffer.from(body).toString('base64') });
  });
  await goto(cdp, `${origin}/__innovation_check`, { waitMs: 1000 });
  await cdp.eval(`(async()=>{
    const React=(await import('/node_modules/.vite/deps/react.js')).default;
    const ReactDOM=(await import('/node_modules/.vite/deps/react-dom_client.js')).default;
    const Agent=(await import('/src/components/PersonalSupportAgent.tsx')).default;
    const Status=(await import('/src/components/InnovationStatus.tsx')).default;
    window.__root=ReactDOM.createRoot(document.getElementById('root'));
    window.__root.render(React.createElement('div',null,React.createElement(Agent),React.createElement(Status,{sentiment:JSON.stringify({innovation:{available:true,observed_modalities:['text'],warnings:['visual_quality_fallback_to_text']}})})));
  })()`);
  const text = await cdp.waitForText(value => value.includes('6.68/10') && value.includes('已退回文字专家'), { timeoutMs: 20000 });
  assert.ok(text.includes('这是试用预测') && text.includes('2026-11-18') && text.includes('已退回文字专家'));
  const capture = await cdp.eval(`(async()=>{
    const {VisualFrameBuffer}=await import('/src/services/visualFrameBuffer.ts');
    const source=document.createElement('canvas');source.width=320;source.height=240;
    const ctx=source.getContext('2d');ctx.fillStyle='#67b4a4';ctx.fillRect(0,0,320,240);
    const stream=source.captureStream(5);const video=document.createElement('video');video.muted=true;video.srcObject=stream;await video.play();
    const buffer=new VisualFrameBuffer();const now=Date.now();
    buffer.capture(video,now-2500);buffer.capture(video,now-1400);buffer.capture(video,now-300);
    const frames=buffer.snapshot(now);const valid=frames.length===3&&frames.every(f=>f.jpeg.startsWith('data:image/jpeg;base64,'));
    const copy=buffer.snapshot(now);copy[0].jpeg='changed';const isolated=buffer.snapshot(now)[0].jpeg!=='changed';
    const expired=buffer.snapshot(now+12000).length===0;buffer.clear();const cleared=buffer.snapshot(now).length===0;
    stream.getTracks().forEach(t=>t.stop());return {valid,isolated,expired,cleared};
  })()`);
  assert.deepEqual(capture, { valid: true, isolated: true, expired: true, cleared: true });
  fs.mkdirSync(output, { recursive: true });
  await cdp.screenshot(path.join(output, 'support-forecast-browser.png'));
  // 自定义文档夹具没有完整Vite页面握手；单独保留HMR传输提示，仍拒绝所有应用异常。
  const developmentWarnings = cdp.consoleErrors.filter(message => message.startsWith('[vite] failed to connect to websocket.'));
  const appErrors = cdp.consoleErrors.filter(message => !developmentWarnings.includes(message));
  const report = { status: 'passed', scope: 'actual React UI and canvas/video frame capture; synthetic interface fixture', capture, appErrors, developmentWarnings };
  assert.equal(appErrors.length, 0, JSON.stringify(appErrors));
  fs.writeFileSync(path.join(output, 'browser-verification.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
} finally { await browser.close(); }
