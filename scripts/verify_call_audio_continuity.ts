/** 运行真实播放器和传输代码，复现断档、重复预缓冲及撤回尾段的竞态。 */
import assert from 'node:assert/strict';
import { TtsPlayer } from '../client/src/services/ttsPlayer';
import { TtsChannel, TtsStreamSession } from '../client/src/services/ttsStream';
import { combineSpeechSegments, speechStartDelay, splitForSpeech } from '../client/src/services/ttsSplit';

class Source {
  buffer: { duration: number } | null = null;
  onended: (() => void) | null = null;
  at = 0;
  stopped = false;
  ended = false;
  connect(): void {}
  disconnect(): void {}
  start(at: number): void { this.at = at; }
  stop(): void { this.stopped = true; }
}
class AudioClock {
  static latest: AudioClock;
  currentTime = 0;
  state = 'running';
  sampleRate = 24000;
  destination = {};
  sources: Source[] = [];
  constructor() { AudioClock.latest = this; }
  addEventListener(): void {}
  resume(): Promise<void> { return Promise.resolve(); }
  close(): Promise<void> { return Promise.resolve(); }
  createBuffer(_channels: number, samples: number, rate: number) {
    return { duration: samples / rate, getChannelData: () => new Float32Array(samples) };
  }
  createBufferSource(): Source {
    const source = new Source();
    this.sources.push(source);
    return source;
  }
  createGain() {
    return { connect() {}, disconnect() {}, gain: { setValueAtTime() {}, linearRampToValueAtTime() {} } };
  }
  advance(seconds: number): void {
    this.currentTime = seconds;
    for (const s of this.sources) {
      if (!s.ended && !s.stopped && s.at + (s.buffer?.duration ?? 0) <= seconds) {
        s.ended = true;
        s.onended?.();
      }
    }
  }
}
let timerId = 0;
(globalThis as any).window = {
  AudioContext: AudioClock,
  setTimeout: () => ++timerId,
  clearTimeout() {},
};
(globalThis as any).WebSocket = { OPEN: 1 };
const audio = new Int16Array(2400); // 100ms
let failures = 0;
function test(name: string, run: () => void): void {
  try { run(); console.log(`PASS ${name}`); }
  catch (error) { failures++; console.error(`FAIL ${name}`, error); }
}

test('首块已耗尽后，同一轮不得再次等待300ms，句内断档必须被记录', () => {
  const player = new TtsPlayer({}, 300);
  player.enqueueChunk(1, audio, 24000, '完整的一句话。');
  const clock = AudioClock.latest;
  clock.advance(0.6);
  player.enqueueChunk(1, audio, 24000, '完整的一句话。');
  assert.equal(clock.sources[1].at, 0.6);
  assert.equal(player.prebufferActivationsCount, 1);
  assert.equal(player.underrun.holes, 1);
  assert.ok(Math.abs(player.underrun.totalHoleMs - 200) < 0.01);
});

test('上一句已自然播完，下一句迟到仍算本轮断档；新一轮留白不算', () => {
  const player = new TtsPlayer({}, 300);
  player.enqueueChunk(1, audio, 24000, '第一句。');
  player.endSegment(1);
  const clock = AudioClock.latest;
  clock.advance(0.6);
  player.enqueueChunk(2, audio, 24000, '第二句。');
  assert.equal(player.underrun.holes, 1);
  assert.equal(clock.sources[1].at, 0.6);
  player.endSegment(2);
  clock.advance(5);
  player.armPrebufferTurn();
  player.enqueueChunk(3, audio, 24000, '新一轮。');
  assert.equal(player.underrun.holes, 1);
  assert.equal(clock.sources[2].at, 5.3);
});

test('改稿只撤回尾段，前面已经排程的正确内容必须继续播放', () => {
  const channel = new TtsChannel({}, 0);
  const player = (channel as any).player as TtsPlayer;
  const stream = (channel as any).stream as TtsStreamSession;
  stream.say(1, '保留这一句。');
  stream.say(2, '撤回这一句。');
  player.enqueueChunk(1, audio, 24000, '保留这一句。');
  player.enqueueChunk(2, audio, 24000, '撤回这一句。');
  const clock = AudioClock.latest;
  channel.stopFrom(2);
  assert.equal(clock.sources[0].stopped, false);
  assert.equal(clock.sources[1].stopped, true);
  assert.deepEqual(player.openSeqs(), [1]);
  channel.stopFrom(0);
  assert.equal(clock.sources[0].stopped, true);
  assert.equal(stream.pendingUnits(), 0);
});

test('撤回多段后，迟到的最高序号start不得复活旧音频', () => {
  const player = new TtsPlayer({}, 0);
  const stream = new TtsStreamSession({}, player);
  for (let seq = 1; seq <= 3; seq++) stream.say(seq, `第${seq}段。`);
  stream.cancelFrom(2);
  const feed = (frame: object) => (stream as any).onMessage({ data: JSON.stringify(frame) });
  feed({ type: 'start', seq: 3, sample_rate: 24000, spoken_text: '第3段。' });
  assert.equal(stream.pendingUnits(), 1);
  feed({ type: 'start', seq: 1, sample_rate: 24000, spoken_text: '第1段。' });
  assert.equal(stream.pendingUnits(), 1);
});

test('撤回尾段期间，正在播放的前缀继续收PCM；整轮打断取消全部在途请求', () => {
  const player = new TtsPlayer({}, 0);
  const stream = new TtsStreamSession({}, player);
  const sent: { type: string; seq: number }[] = [];
  (stream as any).ws = { readyState: 1, send: (raw: string) => sent.push(JSON.parse(raw)) };
  const feed = (frame: object) => (stream as any).onMessage({ data: JSON.stringify(frame) });
  stream.say(1, '第一句完整保留。');
  stream.say(2, '第二句撤回。');
  stream.say(3, '第三句撤回。');
  feed({ type: 'start', seq: 1, sample_rate: 24000, spoken_text: '第一句完整保留。' });
  (stream as any).onMessage({ data: audio.buffer });
  stream.cancelFrom(2);
  (stream as any).onMessage({ data: audio.buffer });
  assert.equal(AudioClock.latest.sources.length, 2, '前缀后续的PCM不能被取消水位线误丢');
  assert.deepEqual(sent.filter((item) => item.type === 'cancel').map((item) => item.seq), [2, 3]);
  stream.say(4, '下一轮。');
  stream.cancelFrom(0);
  assert.deepEqual(sent.filter((item) => item.type === 'cancel').map((item) => item.seq), [2, 3, 1, 4]);
  assert.equal(stream.pendingUnits(), 0);
  const before = AudioClock.latest.sources.length;
  (stream as any).onMessage({ data: audio.buffer });
  assert.equal(AudioClock.latest.sources.length, before, '打断后的残留PCM不得排程');
  stream.say(5, '更新的一轮。');
  feed({ type: 'start', seq: 5, sample_rate: 24000, spoken_text: '更新的一轮。' });
  (stream as any).onMessage({ data: audio.buffer });
  assert.equal(AudioClock.latest.sources.length, before + 1, '下一轮必须能恢复出声');
});

test('定稿短句合并后文本守恒，号码过滤仍只跳过号码', () => {
  const segments = ['嗯。', '我听到你说今天有点累。', '我们慢慢聊，可以联系12345678获取帮助。'];
  const batches = combineSpeechSegments(segments);
  assert.equal(batches.length, 1);
  assert.equal(batches.join(''), segments.join(''));
  const { spoken, skipped } = splitForSpeech(batches[0]);
  assert.equal(spoken.join(''), segments.join('').replace('12345678', ''));
  assert.deepEqual(skipped, [{ text: '12345678', reason: 'phone' }]);
  const long = ['甲'.repeat(70) + '。', '乙'.repeat(70) + '。', '丙'.repeat(120) + '。'];
  const bounded = combineSpeechSegments(long);
  assert.equal(bounded.join(''), long.join(''));
  assert.ok(bounded.every((part) => part.length <= 100));
});

test('完整句子立即起声，短语和不完整句仍等待累积', () => {
  assert.equal(speechStartDelay('我听到你说今天很累。', 350), 0);
  assert.equal(speechStartDelay('他说：“我今天有点累。”', 350), 0);
  assert.equal(speechStartDelay('嗯。', 350), 350);
  assert.equal(speechStartDelay('我听到你说，', 350), 350);
});

if (failures) process.exit(1);
