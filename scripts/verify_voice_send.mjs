/** 真实 Chat、音频采集与设备；ASR 文本和聊天回复为拦截夹具，零业务写入。 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { launchBrowser, injectAuth, login, goto, sleep } from './lib/cdp.mjs';

const auth = await login('http://127.0.0.1:3000', 'patient@mental.com', '123456');
const browser = await launchBrowser({ port: 9363, width: 1500, height: 1100,
  extraArgs: ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'] });
const sends = [];
let releaseReply;
let replyGate = new Promise(resolve => { releaseReply = resolve; });
try {
  const { cdp } = browser;
  await injectAuth(cdp, auth);
  await cdp.send('Page.addScriptToEvaluateOnNewDocument', { source: `(() => {
    const Original = window.WebSocket;
    window.__asrSockets = [];
    window.WebSocket = class extends EventTarget {
      static OPEN = 1; static CONNECTING = 0; static CLOSING = 2; static CLOSED = 3;
      constructor(url, protocols) {
        super();
        if (!String(url).includes('/api/v1/asr/ws')) return new Original(url, protocols);
        this.readyState = 0; this.controls = []; this.closes = 0;
        window.__asrSockets.push(this); window.__asr = this;
        setTimeout(() => { this.readyState=1; this.onopen?.({}); this.emit({type:'status',available:true,engine:'explicit-test-fixture'}); }, 10);
      }
      emit(data) { this.onmessage?.({data:JSON.stringify(data)}); }
      send(value) { if(typeof value==='string') this.controls.push(JSON.parse(value)); }
      close() { this.closes++; this.readyState=3; this.onclose?.({}); }
    };
  })()` });
  await cdp.send('Fetch.enable', { patterns: [{ urlPattern: '*/api/*' }] });
  cdp.ws.on('message', async raw => {
    const event = JSON.parse(raw.toString());
    if (event.method !== 'Fetch.requestPaused') return;
    const { requestId, request } = event.params;
    const url = new URL(request.url);
    let data = {}, body, type = 'application/json';
    if (url.pathname === '/api/consultation/conversations') data = { conversations: [{id:'voice-fixture',title:'语音发送验证'}],total:1 };
    else if (url.pathname.endsWith('/messages/stream')) {
      const payload = JSON.parse(request.postData); sends.push(payload);
      await replyGate;
      const createdAt = new Date().toISOString();
      type = 'text/event-stream';
      body = `event: user\ndata: ${JSON.stringify({userMessage:{id:'u'+sends.length,role:'user',content:payload.content,createdAt}})}\n\n`
        + `event: done\ndata: ${JSON.stringify({aiMessage:{id:'a'+sends.length,role:'assistant',content:'明确的测试回复 '+sends.length,createdAt}})}\n\n`;
    } else if (url.pathname.endsWith('/messages')) data = {messages:[],total:0};
    else if (request.method === 'GET') { await cdp.send('Fetch.continueRequest',{requestId}); return; }
    await cdp.send('Fetch.fulfillRequest', { requestId, responseCode:200,
      responseHeaders:[{name:'Content-Type',value:type}], body:Buffer.from(body ?? JSON.stringify({code:0,data})).toString('base64') });
  });
  const wait = async (predicate, label) => {
    const until = Date.now()+15000;
    while(Date.now()<until) { if(await predicate())return; await sleep(60); }
    throw new Error(label+': '+JSON.stringify(cdp.consoleAll));
  };
  await goto(cdp,'http://localhost:5173/chat/voice-fixture?panel=quality');
  await wait(()=>cdp.eval('!!document.querySelector(".chat-composer input")'),'input missing');
  await cdp.eval('document.querySelector(".chat-composer .anticon-video-camera").closest("button").click()');
  await wait(()=>cdp.eval('window.__asr?.readyState === 1'),'ASR missing');
  await cdp.eval(`(async()=>{window.__media=(await import('/src/services/mediaStreamManager.ts')).mediaStreamManager;})()`);
  await wait(()=>cdp.eval('window.__media.has("camera")'),'camera missing');
  await cdp.eval(`window.__mic=window.__media.getStream('microphone');window.__camera=window.__media.getStream('camera');`);
  const value = () => cdp.eval('document.querySelector(".chat-composer input").value');
  const emit = data => cdp.eval(`window.__asr.emit(${JSON.stringify(data)})`);
  const clickSend = () => cdp.eval(`[...document.querySelectorAll('.chat-composer button')].find(b=>b.textContent==='发送').click()`);
  await emit({type:'partial',text:'今天累'});
  await wait(async()=>await value()==='今天累','partial not in composer');
  await emit({type:'partial',text:'今天很累'});
  await wait(async()=>await value()==='今天很累','partial duplicated');
  await cdp.eval(`(()=>{const el=document.querySelector('.chat-composer input');Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(el,'我想说：今天很累');el.dispatchEvent(new Event('input',{bubbles:true}));})()`);
  await sleep(60);
  await clickSend(); await clickSend();
  await wait(()=>cdp.eval('window.__asr.controls.filter(x=>x.type==="finalize").length===1'),'send did not finalize');
  await emit({type:'final',text:'今天很累。',reason:'finalize'});
  await emit({type:'partial',text:'还有一件事'});
  await wait(()=>sends.length===1,'first message missing');
  assert.equal(sends[0].content,'我想说：今天很累。');
  await wait(async()=>await value()==='还有一件事','next draft lost');
  releaseReply(); replyGate=null;
  await wait(()=>cdp.eval('document.body.innerText.includes("明确的测试回复 1")'),'reply missing');
  await wait(()=>cdp.eval('!document.querySelector(".chat-composer button.ant-btn-loading")'),'send still busy');
  await clickSend();
  await wait(()=>cdp.eval('window.__asr.controls.filter(x=>x.type==="finalize").length===2'),'second boundary missing');
  await emit({type:'final',text:'还有另一件事。',reason:'finalize'});
  await wait(()=>sends.length===2,'second message missing');
  assert.equal(sends[1].content,'还有另一件事。');
  assert.equal(await value(),'');
  const devices=await cdp.eval(`({micSame:window.__media.getStream('microphone')===window.__mic,cameraSame:window.__media.getStream('camera')===window.__camera,
    micLive:window.__mic.getAudioTracks()[0].readyState==='live',cameraLive:window.__camera.getVideoTracks()[0].readyState==='live',closes:window.__asr.closes,sockets:window.__asrSockets.length})`);
  assert.deepEqual(devices,{micSame:true,cameraSame:true,micLive:true,cameraLive:true,closes:0,sockets:1});
  // 定稿服务超时：本轮发送可见草稿，重连识别服务但不释放设备；旧回执不能串入下一轮。
  await wait(()=>cdp.eval('document.body.innerText.includes("明确的测试回复 2")'),'second reply missing');
  await emit({type:'partial',text:'这一轮延迟了'});
  await wait(async()=>await value()==='这一轮延迟了','third draft missing');
  await cdp.eval('window.__oldAsr=window.__asr');
  await clickSend();
  await wait(()=>sends.length===3,'timeout did not send visible draft');
  assert.equal(sends[2].content,'这一轮延迟了');
  await wait(()=>cdp.eval('window.__asr!==window.__oldAsr && window.__asr.readyState===1'),'recognizer did not recover');
  await cdp.eval('window.__oldAsr.emit({type:"final",text:"旧段迟到的定稿",reason:"finalize"})');
  await emit({type:'partial',text:'新的下一轮'});
  await wait(async()=>await value()==='新的下一轮','late old final polluted next draft');
  assert.equal(await cdp.eval('window.__media.getStream("microphone")===window.__mic && window.__mic.getAudioTracks()[0].readyState==="live" && window.__media.getStream("camera")===window.__camera'), true);
  fs.mkdirSync('output/voice-send',{recursive:true});
  await cdp.screenshot('output/voice-send/continuous-voice.png');
  const report={status:'passed',scope:'real Chat + fake devices; ASR and replies explicitly mocked; no business writes',sends:sends.map(x=>x.content),devices,
    partial_replacement:true,manual_edit_preserved:true,double_submit_blocked:true,next_draft_preserved:true,timeout_recovery:true,late_final_ignored:true};
  fs.writeFileSync('output/voice-send/verification.json',JSON.stringify(report,null,2));
  console.log(JSON.stringify(report,null,2));
} finally { await browser.close(); }
process.exit(0);
