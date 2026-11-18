/** 真 MediaPipe 人脸检测和真像素质量检查；画面来自离线人像夹具，聊天写请求全部拦截。 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { launchBrowser, injectAuth, login, goto, sleep, REPO } from './lib/cdp.mjs';

const imagePath = process.env.VISUAL_FACE_FIXTURE || path.resolve(REPO,'../../../_face_sad_audit/sad_wrong.png');
const portrait = fs.readFileSync(imagePath).toString('base64');
const python = process.env.ALGORITHM_PYTHON || 'D:/Develop/AIC/.venv/Scripts/python.exe';
const auth = await login('http://127.0.0.1:3000','patient@mental.com','123456');
const browser = await launchBrowser({port:9364,width:1500,height:1100,extraArgs:['--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream']});
const turns = [], failures = [];
try {
  const { cdp } = browser;
  await injectAuth(cdp,auth);
  await cdp.send('Page.addScriptToEvaluateOnNewDocument',{source:`(() => {
    const originalMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = async constraints => {
      if (!constraints.video) return originalMedia(constraints);
      const image=new Image();image.src='data:image/png;base64,${portrait}';await image.decode();
      const canvas=document.createElement('canvas');canvas.width=640;canvas.height=480;
      const ctx=canvas.getContext('2d');window.__faceVisible=true;
      const draw=()=>{ctx.fillStyle='#ddd';ctx.fillRect(0,0,640,480);if(window.__faceVisible)ctx.drawImage(image,181,20,105,95,128,48,384,384);};
      draw();window.__faceCanvasTimer=setInterval(draw,70);return canvas.captureStream(15);
    };
    const OriginalSocket=window.WebSocket;
    window.WebSocket=class {
      static OPEN=1;
      constructor(url,protocols){if(!String(url).includes('/api/v1/asr/ws'))return new OriginalSocket(url,protocols);
        this.readyState=0;window.__asr=this;setTimeout(()=>{this.readyState=1;this.onopen?.({});this.emit({type:'status',available:true,engine:'explicit-asr-fixture'});},10);}
      emit(data){this.onmessage?.({data:JSON.stringify(data)});}
      send(data){if(typeof data!=='string')return;const control=JSON.parse(data);if(control.type==='finalize')setTimeout(()=>this.emit({type:'final',text:window.__finalText||'',reason:'finalize'}),2100);}
      close(){this.readyState=3;this.onclose?.({});}
    };
  })()`});
  await cdp.send('Fetch.enable',{patterns:[{urlPattern:'*/api/*'},{urlPattern:'*/models/face_emotion*.onnx'}]});
  cdp.ws.on('message',async raw=>{
    const event=JSON.parse(raw.toString());if(event.method!=='Fetch.requestPaused')return;
    const {requestId,request}=event.params;const url=new URL(request.url);
    try {
      if(url.pathname.includes('/models/face_emotion')){await cdp.send('Fetch.fulfillRequest',{requestId,responseCode:503,body:''});return;}
      let data={},body,type='application/json';
      if(url.pathname==='/api/consultation/conversations')data={conversations:[{id:'visual-fixture',title:'画面链路验证'}],total:1};
      else if(url.pathname.endsWith('/messages/stream')){
        const payload=JSON.parse(request.postData),frames=payload.facialFeatures?.visualFrames||[];
        let visual;
        if(frames.length)visual=JSON.parse(execFileSync(python,['-X','utf8','-c',
          'import sys,json,time;from dataclasses import asdict;from perception.innovation import decode_frames,visual_quality;print(json.dumps(asdict(visual_quality(decode_frames(json.load(sys.stdin),time.time()*1000)))))'],
          {cwd:path.join(REPO,'algorithm'),input:JSON.stringify(frames),encoding:'utf8',env:{...process.env,OMP_NUM_THREADS:'2'},timeout:20000}));
        turns.push({frames:frames.length,hasInventedProbabilities:'probabilities' in (payload.facialFeatures||{}),qualityFrameCount:visual?.metrics.frame_count,score:visual?.quality_score});
        const createdAt=new Date().toISOString();
        // 只核验真实质量链路；不冒充运行过专家预测。
        const innovation={available:false,reason:'quality_probe_only_no_expert_prediction',observed_modalities:[],expert_names:[],expert_weights:[],warnings:[],quality:visual?{visual}:{},knowledge_matches:[]};
        type='text/event-stream';body=`event: user\ndata: ${JSON.stringify({userMessage:{id:'u'+turns.length,role:'user',content:payload.content,createdAt}})}\n\nevent: done\ndata: ${JSON.stringify({aiMessage:{id:'a'+turns.length,role:'assistant',content:'仅用于验证画面链路的夹具回复 '+turns.length,createdAt,sentiment:JSON.stringify({innovation})}})}\n\n`;
      }else if(url.pathname.endsWith('/messages'))data={messages:[],total:0};
      else if(request.method==='GET'){await cdp.send('Fetch.continueRequest',{requestId});return;}
      await cdp.send('Fetch.fulfillRequest',{requestId,responseCode:200,responseHeaders:[{name:'Content-Type',value:type}],body:Buffer.from(body??JSON.stringify({code:0,data})).toString('base64')});
    }catch(error){failures.push(String(error));await cdp.send('Fetch.failRequest',{requestId,errorReason:'Failed'});}
  });
  const wait=async(predicate,label)=>{const end=Date.now()+25000;while(Date.now()<end){if(await predicate())return;await sleep(100);}throw new Error(label+JSON.stringify({failures,console:cdp.consoleAll}));};
  await goto(cdp,'http://localhost:5173/chat/visual-fixture?panel=quality');
  await wait(()=>cdp.eval('!!document.querySelector(".chat-composer input")'),'chat absent');
  await cdp.eval('document.querySelector(".chat-composer .anticon-video-camera").closest("button").click()');
  await wait(()=>cdp.eval('document.querySelectorAll(".quality-live-sampling .is-ready").length===3'),'real face never sampled three frames');
  const type=async text=>{await cdp.eval(`(()=>{const el=document.querySelector('.chat-composer input');Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(el,${JSON.stringify(text)});el.dispatchEvent(new Event('input',{bubbles:true}));})()`);await sleep(60);};
  const send=()=>cdp.eval(`[...document.querySelectorAll('.chat-composer button')].find(b=>b.textContent==='发送').click()`);
  for(let i=1;i<=2;i++){
    await type('图文采样测试 '+i);await send();
    await wait(()=>turns.length===i,'frame request absent');
    await wait(()=>cdp.eval(`document.body.innerText.includes('夹具回复 ${i}')`),'reply absent');
    assert.equal(turns[i-1].frames,3);
    assert.equal(turns[i-1].qualityFrameCount,3);
    assert.equal(turns[i-1].hasInventedProbabilities,false);
  }
  await wait(()=>cdp.eval('document.querySelector(".quality-visual-frames").innerText.includes("3 帧已检查")'),'real quality record absent in UI');
  fs.mkdirSync('output/visual-turn',{recursive:true});await cdp.screenshot('output/visual-turn/three-frames-recorded.png');
  await cdp.eval('window.__faceVisible=false');
  await wait(()=>cdp.eval('document.querySelector(".quality-live").innerText.includes("未检测到人脸")'),'face absence not detected');
  await type('无人脸时不应发送旧画面');await send();
  await wait(()=>turns.length===3,'no-face turn absent');
  assert.equal(turns[2].frames,0);
  assert.equal(failures.length,0,JSON.stringify(failures));
  const report={status:'passed',scope:'real MediaPipe + offline portrait camera + actual Python pixel quality; ASR/final replies are fixtures; no business writes',delayed_voice_commit_ms:2100,cnn_model_deliberately_unavailable:true,turns};
  fs.writeFileSync('output/visual-turn/verification.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
}catch(error){console.error(JSON.stringify({error:String(error),failures,console:browser.cdp.consoleAll},null,2));throw error;}
finally{await browser.close();}
process.exit(0);
