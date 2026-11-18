/** AI倾诉与陪伴合并验证：真实本人页面；丰富记录用明确拦截夹具，零写入。 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { launchBrowser, injectAuth, login, goto, REPO, sleep } from './lib/cdp.mjs';

const origin = 'http://localhost:5173';
const output = path.join(REPO, 'output/innovation-integration/unified-chat');
const auth = await login('http://127.0.0.1:3000', 'patient@mental.com', '123456');
const browser = await launchBrowser({ port: 9348, width: 1600, height: 1080 });
try {
  const cdp = browser.cdp;
  await cdp.send('Emulation.setDeviceMetricsOverride',{width:1600,height:1080,deviceScaleFactor:1,mobile:false});
  await injectAuth(cdp, auth);
  await goto(cdp, `${origin}/chat`);
  await cdp.waitForText(text => text.includes('陪伴与感知') && text.includes('图文理解'), { timeoutMs: 30000 });
  assert.equal(await cdp.eval(`[...document.querySelectorAll('nav[aria-label="主导航"] button')].filter(el=>el.textContent.trim()==='智能陪伴').length`), 0);
  const initial = await cdp.eval(`({width:innerWidth,composer:!!document.querySelector('.chat-composer input'),panel:!!document.querySelector('.chat-companion-aside [data-testid="chat-companion-panel"]'),text:document.body.innerText.slice(0,800)})`);
  assert.ok(initial.composer && initial.panel, JSON.stringify({initial,errors:cdp.consoleErrors}));
  assert.equal(await cdp.eval('document.documentElement.scrollWidth <= window.innerWidth + 1'), true);
  await cdp.eval(`(()=>{const input=document.querySelector('.chat-composer input');Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(input,'保留这段尚未发送的草稿');input.dispatchEvent(new Event('input',{bubbles:true}));window.__agentNode=document.querySelector('[data-testid="personal-support-agent"]')})()`);
  const open = async name => { await cdp.eval(`document.querySelector('[aria-label="打开${name}面板"]').click()`); await sleep(150); };
  for (const name of ['持续陪伴','质量纠偏','图文理解','持续陪伴']) await open(name);
  assert.equal(await cdp.eval(`document.querySelector('.chat-composer input').value`), '保留这段尚未发送的草稿');
  assert.equal(await cdp.eval(`window.__agentNode===document.querySelector('[data-testid="personal-support-agent"]')`), true);
  assert.equal(await cdp.eval('location.pathname'), '/chat');
  await cdp.eval(`(()=>{const b=[...document.querySelectorAll('.chat-companion-heading button')].find(el=>el.textContent.trim()==='收起');b.click()})()`);
  await open('持续陪伴');
  assert.equal(await cdp.eval(`window.__agentNode===document.querySelector('[data-testid="personal-support-agent"]')`), true);
  await open('图文理解');
  await cdp.screenshot(path.join(output, 'unified-chat-desktop.png'));

  const reference = { available:true,model_id:'a'.repeat(64),observed_modalities:['text'],expert_names:['text','candidate_knowledge'],expert_weights:[.7,.3],warnings:['visual_quality_fallback_to_text'],
    quality:{visual:{quality_score:.05,reason_codes:['underexposure_proxy'],metrics:{frame_count:3,luminance_mean:0,contrast_mean:0}}},
    knowledge_matches:[{id:'test-node',text:'仅属于会话A的自编测试依据',source_url:'https://github.com/allenai/comet-atomic-2020',review_status:'user_attested',relations:[]}]};
  const stamp = new Date().toISOString();
  const agent = {state:{schema:'support-agent.v1',revision:1,enabled:true,preferredTool:'none',actions:[],background:{enabled:false,notifyInApp:false,intervalHours:24,quietStart:22,quietEnd:8,nextRunAt:null,events:[]}},
    trend:{observations:3,latestScore:8,mean:6.67,change:3,stale:false,label:'本人自评记录',points:[5,7,8].map((score,i)=>({id:String(i),score,at:`2026-11-${15+i}T12:00:00+08:00`})),forecast:{status:'experimental',score:6.68,observedDays:3,targetDate:'2026-11-18'}},
    suggestionEligible:false,context:{status:'unavailable',goal:'unspecified',evidence:[],revoked:0},modelReference:null};
  let writes = 0;
  await cdp.send('Fetch.enable',{patterns:[{urlPattern:'*/api/consultation/conversations*'},{urlPattern:'*/api/support-agent*'}]});
  cdp.ws.on('message',async raw=>{
    const event=JSON.parse(raw.toString());if(event.method!=='Fetch.requestPaused')return;
    const {requestId,request}=event.params;
    if(request.method!=='GET')writes++;
    const url=new URL(request.url);let data;
    if(url.pathname==='/api/support-agent')data=agent;
    else if(url.pathname.endsWith('/messages')){
      const a=url.pathname.includes('ui-current-a');
      data={conversation:{id:a?'ui-current-a':'ui-current-b',title:a?'自编会话 A':'自编会话 B'},total:2,messages:[
        {id:'fixture-user',role:'user',content:'自编输入',createdAt:stamp},
        {id:a?'answer-a':'answer-b',role:'assistant',content:'自编界面回复',createdAt:stamp,sentiment:JSON.stringify(a?{innovation:reference}:{})}]};
    }else data={conversations:[{id:'ui-current-a',title:'自编会话 A'},{id:'ui-current-b',title:'自编会话 B'}],total:2};
    await cdp.send('Fetch.fulfillRequest',{requestId,responseCode:200,responseHeaders:[{name:'Content-Type',value:'application/json'}],body:Buffer.from(JSON.stringify({code:0,data})).toString('base64')});
  });
  await goto(cdp,`${origin}/chat/ui-current-a?panel=fusion`);
  const evidence=await cdp.waitForText(text=>text.includes('仅属于会话A的自编测试依据'),{timeoutMs:20000});
  assert.ok(evidence.includes('知识概念专家'));
  await open('质量纠偏');
  assert.ok((await cdp.waitForText(t=>t.includes('已退回文字'))).includes('画面过暗'));
  await cdp.screenshot(path.join(output, 'unified-quality-desktop.png'));
  await cdp.eval(`(()=>{const row=[...document.querySelectorAll('.chat-history .ant-list-item')].find(el=>el.textContent.includes('自编会话 B'));row.click()})()`);
  await cdp.waitForText(text=>text.includes('自编界面回复'),{timeoutMs:10000});await sleep(500);
  assert.equal(await cdp.eval(`document.querySelector('[data-testid="chat-companion-panel"]').innerText.includes('仅属于会话A')`),false);
  assert.equal(await cdp.eval('location.pathname'),'/chat/ui-current-b');

  await goto(cdp,`${origin}/companion#tracking`);
  await cdp.waitForText(t=>t.includes('6.68/10'),{timeoutMs:20000});
  assert.equal(await cdp.eval('location.pathname'),'/chat');
  assert.equal(await cdp.eval('new URLSearchParams(location.search).get("panel")'),'tracking');
  assert.ok(await cdp.eval(`!!document.querySelector('[aria-label="你的心情变化，1 到 10 分"] svg')`));
  await cdp.send('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});await sleep(400);
  assert.ok(await cdp.eval(`!!document.querySelector('.ant-drawer-open [data-testid="chat-companion-panel"]')`));
  assert.equal(await cdp.eval('document.documentElement.scrollWidth <= window.innerWidth + 1'),true);
  await cdp.screenshot(path.join(output,'unified-chat-mobile.png'),{maxHeight:1200});
  await cdp.eval(`document.querySelector('.ant-drawer-close').click()`);await sleep(250);
  assert.ok(await cdp.eval(`!!document.querySelector('.chat-composer input')`));
  await goto(cdp,`${origin}/chat/ui-current-a?panel=quality`);
  await cdp.waitForText(t=>t.includes('已退回文字'),{timeoutMs:10000});
  assert.equal(await cdp.eval('document.documentElement.scrollWidth <= window.innerWidth + 1'),true);
  await cdp.screenshot(path.join(output,'unified-quality-mobile.png'),{maxHeight:1200});
  assert.equal(writes,0,'切换面板不应生成消息或执行活动');
  const warnings=cdp.consoleErrors.filter(t=>t.startsWith('[vite] failed to connect to websocket.')||/^Warning: \[antd: (Modal|Card)\] `(destroyOnClose|bodyStyle)` is deprecated\./.test(t));
  const errors=cdp.consoleErrors.filter(t=>!warnings.includes(t));assert.equal(errors.length,0,JSON.stringify(errors));
  const report={status:'passed',single_chat_workspace:true,no_separate_navigation:true,draft_preserved:true,support_component_preserved:true,evidence_follows_selected_conversation:true,legacy_redirect:true,mobile_drawer:true,no_fixture_writes:writes===0,application_errors:errors,developmentWarnings:warnings};
  fs.mkdirSync(output,{recursive:true});fs.writeFileSync(path.join(output,'verification.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
}finally{await browser.close();}
