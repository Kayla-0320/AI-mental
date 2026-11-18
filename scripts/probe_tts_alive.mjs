/**
 * 单独验一条链路：**本机 TTS 服务现在还出不出音频**。
 *
 * 背景：用户真机反馈"我两次说我感觉好一些了都没有声音出来"，而诊断面板上
 * `改稿/未朗读 = 3/0`（三条回复都走了安全审计改稿的补念路径）、TTS 状态是
 * `cosyvoice3 · 24000Hz`、`服务端首块 1844ms · RTF 1.31` —— 也就是"服务端自述能合成"。
 * 但**自述**不等于"真的出音频"，所以这里绕开浏览器与 React，直接用协议打一次：
 *
 *   status → say(seq=1) → 二进制帧 + end(spoken_text/audio_ms/first_chunk_ms/rtf)
 *
 * 判据：
 *   · 收到 > 0 字节 PCM   ⇒ 服务端这条链路是通的，问题在客户端播放/编排；
 *   · 只回 `end` 无二进制 ⇒ 服务端"合成成功但没音频"（OOV/空音频），客户端会如实报错。
 *
 * 用法：
 *   node scripts/probe_tts_alive.mjs [文本]
 */
const WS_URL = process.env.TTS_WS_URL ?? 'ws://127.0.0.1:8001/api/v1/tts/ws';
const TEXT = process.argv[2] ?? '我感觉好一些了。刚才说的那些，我都记着呢。';

const t0 = Date.now();
const ws = new WebSocket(WS_URL);
let binaryBytes = 0;
let binaryFrames = 0;
const events = [];

const timer = setTimeout(() => {
  console.log('⏰ 15 秒内没等到 end，链路卡住');
  finish();
}, 15000);

function finish() {
  clearTimeout(timer);
  try {
    ws.close();
  } catch {
    /* 忽略 */
  }
  console.log('');
  console.log('── 汇总 ──');
  for (const e of events) console.log(' ', e);
  console.log(`  二进制帧 ${binaryFrames} 个 · 共 ${binaryBytes} 字节`);
  const ok = binaryBytes > 0;
  console.log(ok ? '✅ 服务端 TTS 出音频（链路通）' : '❌ 服务端 TTS 没有出任何音频');
  process.exit(ok ? 0 : 1);
}

ws.binaryType = 'arraybuffer';
ws.onopen = () => {
  events.push(`onopen +${Date.now() - t0}ms`);
};
ws.onerror = (e) => {
  events.push(`onerror: ${e?.message ?? e}`);
};
ws.onmessage = (ev) => {
  const dt = Date.now() - t0;
  if (typeof ev.data !== 'string') {
    binaryFrames += 1;
    binaryBytes += ev.data.byteLength ?? 0;
    if (binaryFrames <= 3 || binaryFrames % 20 === 0) {
      events.push(`PCM #${binaryFrames} +${dt}ms ${ev.data.byteLength} 字节`);
    }
    return;
  }
  let msg;
  try {
    msg = JSON.parse(ev.data);
  } catch {
    events.push(`非 JSON 文本帧 ${ev.data.slice(0, 80)}`);
    return;
  }
  events.push(`${msg.type} +${dt}ms ${JSON.stringify(msg).slice(0, 200)}`);
  if (msg.type === 'status') {
    if (!msg.available) {
      console.log('❌ 服务端自述不可用:', msg.error ?? '（无原因）');
      finish();
      return;
    }
    console.log('say →', TEXT);
    ws.send(JSON.stringify({ type: 'say', seq: 1, text: TEXT }));
    return;
  }
  if (msg.type === 'end' || msg.type === 'cancelled' || msg.type === 'error') {
    finish();
  }
};
