/** 假摄像头上的真实面部管线；测量冷启动、再次开启与取消，不写业务数据。 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { launchBrowser, goto, sleep, REPO } from './lib/cdp.mjs';

const baseline = process.argv.includes('--baseline');
const origin = process.env.QUALITY_VITE || 'http://localhost:5173';
const output = path.join(REPO, 'output/quality-startup');
const baselineHook = path.join(REPO, 'client/src/hooks/.quality-startup-baseline.ts');
if (baseline) fs.writeFileSync(baselineHook, execFileSync('git', ['show', 'HEAD:client/src/hooks/useFacialAnalysis.ts'], { cwd: REPO }), { flag: 'wx' });
const browser = await launchBrowser({ port: 9356, extraArgs: [
  '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
  '--autoplay-policy=no-user-gesture-required',
] });
try {
  const { cdp } = browser;
  const requests = [];
  await cdp.send('Network.enable');
  cdp.ws.on('message', async raw => {
    const event = JSON.parse(raw.toString());
    if (event.method === 'Network.requestWillBeSent') requests.push(event.params.request.url);
    if (event.method !== 'Fetch.requestPaused') return;
    await cdp.send('Fetch.fulfillRequest', { requestId: event.params.requestId, responseCode: 200,
      responseHeaders: [{ name: 'Content-Type', value: 'text/html' }],
      body: Buffer.from(`<html><body><div id="root"></div><script type="module">
        import RefreshRuntime from '/@react-refresh';
        RefreshRuntime.injectIntoGlobalHook(window);
        window.$RefreshReg$=()=>{};window.$RefreshSig$=()=>type=>type;
        window.__vite_plugin_react_preamble_installed__=true;
      </script></body></html>`).toString('base64') });
  });
  await cdp.send('Fetch.enable', { patterns: [{ urlPattern: '*/__quality_startup' }] });
  await goto(cdp, `${origin}/__quality_startup`);
  await cdp.eval(`(async () => {
    const React = (await import('/node_modules/.vite/deps/react.js')).default;
    const { createRoot } = (await import('/node_modules/.vite/deps/react-dom_client.js')).default;
    const { useFacialAnalysis } = await import(${JSON.stringify(baseline ? '/src/hooks/.quality-startup-baseline.ts' : '/src/hooks/useFacialAnalysis.ts')});
    window.__root = createRoot(document.getElementById('root'));
    function Probe() {
      window.__face = useFacialAnalysis(null);
      return React.createElement('pre', null, JSON.stringify(window.__face.metrics));
    }
    window.__root.render(React.createElement(Probe));
  })()`);
  const wait = async (predicate, timeout = 20000) => {
    const until = Date.now() + timeout;
    while (Date.now() < until) {
      if (await cdp.eval(predicate)) return true;
      await sleep(100);
    }
    return false;
  };
  assert.ok(await wait('!!window.__face'));
  const start = async () => {
    await cdp.eval(`(() => {
      window.__startup = { at: performance.now(), settled: false };
      window.__face.start().then(() => {
        window.__startup.ms = performance.now() - window.__startup.at;
        window.__startup.settled = true;
      }).catch(e => { window.__startup.error = String(e); window.__startup.settled = true; });
    })()`);
    await wait('window.__startup.settled');
    return cdp.eval('({ ...window.__startup, metrics: window.__face.metrics })');
  };
  const cold = await start();
  const report = { phase: baseline ? 'before' : 'after', scope: 'real MediaPipe + fake camera; no business writes', cold,
    detectorInitialized: cdp.consoleAll.some(item => item.text.includes('MediaPipe Face Mesh 初始化成功')),
    externalRequests: requests.filter(url => /^https?:/.test(url) && new URL(url).origin !== origin) };
  if (!baseline) {
    assert.ok(cold.settled && !cold.error && cold.metrics.isDetecting, JSON.stringify(cold));
    assert.ok(await wait('window.__face.videoRef.current?.currentTime > 0'), JSON.stringify(cdp.consoleAll));
    assert.ok(cdp.consoleAll.some(item => item.text.includes('MediaPipe Face Mesh 初始化成功')), JSON.stringify(cdp.consoleAll));
    await cdp.eval('window.__face.stop()');
    assert.equal(await cdp.eval('window.__face.streamRef.current'), null);
    const count = requests.length;
    report.reopen = await start();
    assert.ok(report.reopen.settled && !report.reopen.error);
    report.reopenModelRequests = requests.slice(count).filter(url => /vision_wasm|face_landmarker\.task/.test(url));
    assert.equal(report.reopenModelRequests.length, 0, '再次开启应复用检测器');
    await cdp.eval('window.__face.stop()');
    report.externalRequests = requests.filter(url => /^https?:/.test(url) && new URL(url).origin !== origin);
    assert.equal(report.externalRequests.length, 0, '检测器不应依赖外站资源');
    // 设备等待期间关闭：迟到的 start 不能重新开启摄像头。
    await cdp.eval(`(() => {
      const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
      navigator.mediaDevices.getUserMedia = async (...args) => {
        await new Promise(resolve => setTimeout(resolve, 300));
        return original(...args);
      };
      window.__cancelled = window.__face.start();
      window.__face.stop();
    })()`);
    await sleep(700);
    await cdp.eval('window.__cancelled');
    report.cancelled = await cdp.eval('!window.__face.metrics.isDetecting && !window.__face.streamRef.current');
    assert.ok(report.cancelled);
    await cdp.eval('window.__unmountStart = window.__face.start(); window.__root.unmount()');
    await sleep(700);
    await cdp.eval('window.__unmountStart');
    report.unmountReleasedCamera = await cdp.eval(`(async () => !(await import('/src/services/mediaStreamManager.ts')).mediaStreamManager.has('camera'))()`);
    assert.ok(report.unmountReleasedCamera);
  }
  report.consoleErrors = cdp.consoleErrors.filter(message => !message.startsWith('[vite] failed to connect')
    && !message.startsWith('INFO: Created TensorFlow Lite XNNPACK delegate'));
  if (!baseline) assert.equal(report.consoleErrors.length, 0, JSON.stringify(report.consoleErrors));
  fs.mkdirSync(output, { recursive: true });
  fs.writeFileSync(path.join(output, `${report.phase}.json`), JSON.stringify(report, null, 2));
  console.log(JSON.stringify({ phase: report.phase, coldMs: report.cold.ms, reopenMs: report.reopen?.ms,
    detectorInitialized: report.detectorInitialized, externalRequests: report.externalRequests,
    cancelled: report.cancelled, unmountReleasedCamera: report.unmountReleasedCamera, consoleErrors: report.consoleErrors }, null, 2));
} catch (error) {
  console.error(JSON.stringify({ error: String(error), console: browser.cdp.consoleAll }, null, 2));
  throw error;
} finally {
  await browser.close();
  if (baseline) fs.unlinkSync(baselineHook);
}
process.exit(0);
