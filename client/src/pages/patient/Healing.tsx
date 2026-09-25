/**
 * 疗愈空间 —— 心理小游戏合集
 *
 * 三大分类：
 *   🎯 注意力 & 专注力：舒尔特方格、色词冲突
 *   🌿 缓解压力：气球呼吸训练、冥想音频
 *   🎨 释放压力：涂鸦画板、橡皮人拉扯
 */
import { useState, useEffect, useRef, useCallback } from 'react';
import { Card, Tabs, Tag, Space, Typography, Button, Row, Col } from 'antd';
import {
  BulbOutlined, CloudOutlined, ExperimentOutlined,
} from '@ant-design/icons';

const { Title, Text } = Typography;

// ============================================================
// 1. 舒尔特方格
// ============================================================
function SchulteGrid() {
  const SIZE = 5;
  const [grid, setGrid] = useState<number[]>([]);
  const [next, setNext] = useState(1);
  const [timer, setTimer] = useState(0);
  const [running, setRunning] = useState(false);
  const [done, setDone] = useState(false);
  const [best, setBest] = useState<number | null>(null);
  const timerRef = useRef<any>(null);

  const shuffle = useCallback(() => {
    const arr = Array.from({ length: SIZE * SIZE }, (_, i) => i + 1);
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    setGrid(arr);
    setNext(1);
    setTimer(0);
    setDone(false);
    setRunning(true);
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = setInterval(() => setTimer(t => t + 0.1), 100);
  }, []);

  useEffect(() => { () => { if (timerRef.current) clearInterval(timerRef.current); }; }, []);

  const handleClick = (num: number) => {
    if (done) return;
    if (num === next) {
      if (next === SIZE * SIZE) {
        setDone(true);
        setRunning(false);
        clearInterval(timerRef.current);
        const finalTime = Math.round(timer * 10) / 10;
        if (best === null || finalTime < best) setBest(finalTime);
      } else {
        setNext(n => n + 1);
      }
    }
  };

  const cellColor = (num: number) => {
    if (!running) return '#f0f0f0';
    if (num < next) return '#b7eb8f';
    if (num === next) return '#ffe58f';
    return '#f0f0f0';
  };

  return (
    <div style={{ textAlign: 'center' }}>
      <Text type="secondary" style={{ display: 'block', marginBottom: 12, fontSize: 13 }}>
        按顺序点击 1~25，越快越好！训练视觉搜索和注意力广度。
      </Text>
      <div style={{ display: 'flex', justifyContent: 'center', gap: 40, marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 12, color: '#999', marginBottom: 4 }}>用时</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: '#722ed1' }}>{timer.toFixed(1)} 秒</div>
        </div>
        {best !== null && (
          <div>
            <div style={{ fontSize: 12, color: '#999', marginBottom: 4 }}>最佳</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: '#52c41a' }}>{best.toFixed(1)} 秒</div>
          </div>
        )}
      </div>
      <div style={{
        display: 'grid', gridTemplateColumns: `repeat(${SIZE}, 1fr)`,
        gap: 6, maxWidth: 320, margin: '0 auto 16px',
      }}>
        {grid.map((num, i) => (
          <div key={i} onClick={() => handleClick(num)} style={{
            width: 56, height: 56, display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: cellColor(num), borderRadius: 10, cursor: 'pointer',
            fontSize: 18, fontWeight: 600, color: num < next ? '#52c41a' : '#333',
            border: num === next && running ? '2px solid #faad14' : '1px solid #e8e8e8',
            transition: 'all 0.15s', userSelect: 'none',
          }}>{num}</div>
        ))}
      </div>
      <Button type="primary" onClick={shuffle} style={{ borderRadius: 20 }}>
        {done ? '再来一局' : running ? '重新开始' : '开始'}
      </Button>
      {done && <div style={{ marginTop: 12 }}><Tag color="green" style={{ fontSize: 14, padding: '4px 12px' }}>完成！用时 {timer.toFixed(1)} 秒</Tag></div>}
    </div>
  );
}

// ============================================================
// 2. 色词冲突 (Stroop)
// ============================================================
const STROOP_COLORS = [
  { name: '红', hex: '#f5222d' },
  { name: '蓝', hex: '#1890ff' },
  { name: '绿', hex: '#52c41a' },
  { name: '黄', hex: '#faad14' },
  { name: '紫', hex: '#722ed1' },
];

function StroopTest() {
  const [word, setWord] = useState('');
  const [color, setColor] = useState('');
  const [score, setScore] = useState(0);
  const [total, setTotal] = useState(0);
  const [timeLeft, setTimeLeft] = useState(30);
  const [playing, setPlaying] = useState(false);
  const [feedback, setFeedback] = useState<'correct' | 'wrong' | null>(null);
  const timerRef = useRef<any>(null);

  const newRound = useCallback(() => {
    const w = STROOP_COLORS[Math.floor(Math.random() * STROOP_COLORS.length)];
    let c = STROOP_COLORS[Math.floor(Math.random() * STROOP_COLORS.length)];
    // 70% 概率字色不一致
    if (Math.random() < 0.7) {
      while (c.name === w.name) c = STROOP_COLORS[Math.floor(Math.random() * STROOP_COLORS.length)];
    }
    setWord(w.name);
    setColor(c.hex);
    setFeedback(null);
  }, []);

  const start = () => {
    setScore(0); setTotal(0); setTimeLeft(30); setPlaying(true);
    newRound();
    timerRef.current = setInterval(() => {
      setTimeLeft(t => {
        if (t <= 0.1) { clearInterval(timerRef.current); setPlaying(false); return 0; }
        return t - 0.1;
      });
    }, 100);
  };

  const answer = (colorName: string) => {
    if (!playing) return;
    const correctColor = STROOP_COLORS.find(c => c.hex === color);
    setTotal(t => t + 1);
    if (correctColor?.name === colorName) {
      setScore(s => s + 1);
      setFeedback('correct');
    } else {
      setFeedback('wrong');
    }
    setTimeout(() => newRound(), 300);
  };

  useEffect(() => () => { if (timerRef.current) clearInterval(timerRef.current); }, []);

  return (
    <div style={{ textAlign: 'center' }}>
      <Text type="secondary" style={{ display: 'block', marginBottom: 12, fontSize: 13 }}>
        选择字的<strong>颜色</strong>（不是字义）！训练抑制控制和认知灵活性。
      </Text>
      <div style={{ display: 'flex', justifyContent: 'center', gap: 40, marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 12, color: '#999', marginBottom: 4 }}>得分</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: '#722ed1' }}>{score}</div>
        </div>
        <div>
          <div style={{ fontSize: 12, color: '#999', marginBottom: 4 }}>剩余</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: timeLeft < 10 ? '#f5222d' : '#1890ff' }}>{timeLeft.toFixed(1)} 秒</div>
        </div>
      </div>
      {!playing && total === 0 && (
        <Button type="primary" onClick={start} style={{ borderRadius: 20, marginBottom: 16 }}>开始挑战（30秒）</Button>
      )}
      {playing && (
        <>
          <div style={{ fontSize: 52, fontWeight: 800, color, marginBottom: 24, height: 70, lineHeight: '70px' }}>{word}</div>
          <Space wrap size={12} style={{ justifyContent: 'center' }}>
            {STROOP_COLORS.map(c => (
              <Button key={c.name} onClick={() => answer(c.name)}
                style={{ width: 72, height: 44, borderRadius: 22, fontSize: 16, fontWeight: 600, color: '#fff', background: c.hex, border: 'none' }}>
                {c.name}
              </Button>
            ))}
          </Space>
          {feedback && (
            <div style={{ marginTop: 12, fontSize: 18, color: feedback === 'correct' ? '#52c41a' : '#f5222d', fontWeight: 600 }}>
              {feedback === 'correct' ? '✓ 正确！' : '✗ 看颜色不是字义！'}
            </div>
          )}
        </>
      )}
      {!playing && total > 0 && (
        <div>
          <Tag color={score / total > 0.7 ? 'green' : score / total > 0.4 ? 'orange' : 'red'}
            style={{ fontSize: 16, padding: '6px 16px' }}>
            正确率 {Math.round(score / total * 100)}%（{score}/{total}）
          </Tag>
          <div style={{ marginTop: 12 }}><Button onClick={start} style={{ borderRadius: 20 }}>再来一局</Button></div>
        </div>
      )}
    </div>
  );
}

// ============================================================
// 3. 气球呼吸训练
// ============================================================
function BalloonBreathing() {
  const [phase, setPhase] = useState<'idle' | 'inhale' | 'hold' | 'exhale'>('idle');
  const [scale, setScale] = useState(1);
  const [cycles, setCycles] = useState(0);
  const [timer, setTimer] = useState(0);
  const animRef = useRef<any>(null);
  const phaseRef = useRef<'idle' | 'inhale' | 'hold' | 'exhale'>('idle');

  const INHALE_SEC = 4, HOLD_SEC = 4, EXHALE_SEC = 6;

  const startBreathing = () => {
    setPhase('inhale');
    phaseRef.current = 'inhale';
    setTimer(0);
    setCycles(0);
    setScale(1);
    runCycle();
  };

  const runCycle = () => {
    // 吸气 4s
    phaseRef.current = 'inhale';
    setPhase('inhale');
    let t = 0;
    const inhaleInterval = setInterval(() => {
      t += 0.05;
      setTimer(t);
      setScale(1 + (2.2 - 1) * Math.min(t / INHALE_SEC, 1));
      if (t >= INHALE_SEC) {
        clearInterval(inhaleInterval);
        // 屏息 4s
        phaseRef.current = 'hold';
        setPhase('hold');
        let h = 0;
        const holdInterval = setInterval(() => {
          h += 0.05;
          setTimer(h);
          if (h >= HOLD_SEC) {
            clearInterval(holdInterval);
            // 呼气 6s
            phaseRef.current = 'exhale';
            setPhase('exhale');
            let e = 0;
            const exhaleInterval = setInterval(() => {
              e += 0.05;
              setTimer(e);
              setScale(2.2 - (2.2 - 1) * Math.min(e / EXHALE_SEC, 1));
              if (e >= EXHALE_SEC) {
                clearInterval(exhaleInterval);
                setCycles(c => c + 1);
                // 继续下一轮
                runCycle();
              }
            }, 50);
            animRef.current = exhaleInterval;
          }
        }, 50);
        animRef.current = holdInterval;
      }
    }, 50);
    animRef.current = inhaleInterval;
  };

  const stop = () => {
    if (animRef.current) clearInterval(animRef.current);
    setPhase('idle');
    phaseRef.current = 'idle';
    setScale(1);
  };

  useEffect(() => () => { if (animRef.current) clearInterval(animRef.current); }, []);

  const phaseText = { idle: '点击开始', inhale: '吸气...', hold: '屏住...', exhale: '呼气...' }[phase];
  const phaseColor = { idle: '#bfbfbf', inhale: '#69c0ff', hold: '#ffc069', exhale: '#95de64' }[phase];

  return (
    <div style={{ textAlign: 'center' }}>
      <Text type="secondary" style={{ display: 'block', marginBottom: 12, fontSize: 13 }}>
        跟随气球节奏呼吸：吸气 4 秒 → 屏息 4 秒 → 呼气 6 秒。缓解焦虑，激活副交感神经。
      </Text>
      <div style={{ position: 'relative', height: 280, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{
          width: 100, height: 120, borderRadius: '50% 50% 50% 50% / 60% 60% 40% 40%',
          background: `radial-gradient(ellipse at 35% 35%, #ffd6e7, #ff85a2)`,
          transform: `scale(${scale})`,
          transition: 'transform 0.05s linear',
          boxShadow: `0 0 ${20 * scale}px rgba(255,133,162,0.4)`,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <div style={{ fontSize: 28 }}>🎈</div>
        </div>
        {/* 气球线 */}
        <div style={{
          position: 'absolute', bottom: 40, width: 2, height: 40 * scale,
          background: '#ccc', left: '50%', transform: 'translateX(-50%)',
        }} />
      </div>
      <div style={{ fontSize: 22, fontWeight: 700, color: phaseColor, marginBottom: 8 }}>{phaseText}</div>
      <div style={{ marginBottom: 16 }}>
        <Tag color="purple" style={{ fontSize: 14 }}>已完成 {cycles} 轮</Tag>
      </div>
      <Button type={phase === 'idle' ? 'primary' : 'default'} onClick={phase === 'idle' ? startBreathing : stop}
        style={{ borderRadius: 20 }}>
        {phase === 'idle' ? '开始呼吸' : '停止'}
      </Button>
    </div>
  );
}

// ============================================================
// 4. 冥想音频 (Web Audio API 实时合成)
// ============================================================
const MEDITATION_TRACKS = [
  { title: '森林清晨', emoji: '\u{1F332}', desc: '鸟鸣与微风', duration: '5:00', color: '#52c41a', type: 'forest' as const },
  { title: '海浪轻拍', emoji: '\u{1F30A}', desc: '潮汐白噪音', duration: '5:00', color: '#1890ff', type: 'ocean' as const },
  { title: '雨打窗棂', emoji: '\u{1F327}\u{FE0F}', desc: '淅沥雨声', duration: '5:00', color: '#722ed1', type: 'rain' as const },
  { title: '篝火噼啪', emoji: '\u{1F525}', desc: '温暖篝火声', duration: '5:00', color: '#fa8c16', type: 'fire' as const },
  { title: '溪流潺潺', emoji: '\u{1F4A7}', desc: '山间清泉', duration: '5:00', color: '#13c2c2', type: 'stream' as const },
  { title: '星空静谧', emoji: '\u{1F319}', desc: '深夜环境音', duration: '5:00', color: '#2f54eb', type: 'night' as const },
];

type SoundType = 'forest' | 'ocean' | 'rain' | 'fire' | 'stream' | 'night';

/** 创建噪声缓冲区 */
function createNoiseBuffer(ctx: AudioContext, type: 'white' | 'pink' | 'brown'): AudioBuffer {
  const length = ctx.sampleRate * 2;
  const buffer = ctx.createBuffer(1, length, ctx.sampleRate);
  const data = buffer.getChannelData(0);
  let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
  for (let i = 0; i < length; i++) {
    const white = Math.random() * 2 - 1;
    if (type === 'white') {
      data[i] = white;
    } else if (type === 'pink') {
      b0 = 0.99886 * b0 + white * 0.0555179;
      b1 = 0.99332 * b1 + white * 0.0750759;
      b2 = 0.96900 * b2 + white * 0.1538520;
      b3 = 0.86650 * b3 + white * 0.3104856;
      b4 = 0.55000 * b4 + white * 0.5329522;
      b5 = -0.7616 * b5 - white * 0.0168980;
      data[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362) * 0.11;
      b6 = white * 0.115926;
    } else {
      data[i] = (b0 = (b0 + (0.02 * white)) / 1.02);
    }
  }
  return buffer;
}

function MeditationAudio() {
  const [playing, setPlaying] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const timerRef = useRef<any>(null);
  const audioRef = useRef<{ ctx: AudioContext; nodes: AudioNode[]; intervals: number[] } | null>(null);
  const DURATION = 300;

  const stopSound = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.intervals.forEach(id => clearInterval(id));
      audioRef.current.nodes.forEach(n => { try { n.disconnect(); } catch {} });
      audioRef.current.ctx.close();
      audioRef.current = null;
    }
  }, []);

  const startSound = useCallback((type: SoundType) => {
    const ctx = new AudioContext();
    const master = ctx.createGain();
    master.gain.value = 0.3;
    master.connect(ctx.destination);
    const nodes: AudioNode[] = [master];
    const intervals: number[] = [];

    if (type === 'ocean') {
      const noise = ctx.createBufferSource();
      noise.buffer = createNoiseBuffer(ctx, 'brown');
      noise.loop = true;
      const filter = ctx.createBiquadFilter();
      filter.type = 'lowpass';
      filter.frequency.value = 400;
      const lfo = ctx.createOscillator();
      lfo.frequency.value = 0.08;
      const lfoGain = ctx.createGain();
      lfoGain.gain.value = 0.15;
      lfo.connect(lfoGain);
      lfoGain.connect(master.gain);
      lfo.start();
      noise.connect(filter);
      filter.connect(master);
      noise.start();
      nodes.push(noise, filter, lfo, lfoGain);
    } else if (type === 'rain') {
      const noise = ctx.createBufferSource();
      noise.buffer = createNoiseBuffer(ctx, 'pink');
      noise.loop = true;
      const bp = ctx.createBiquadFilter();
      bp.type = 'bandpass';
      bp.frequency.value = 3000;
      bp.Q.value = 0.5;
      const hp = ctx.createBiquadFilter();
      hp.type = 'highpass';
      hp.frequency.value = 1000;
      noise.connect(bp);
      bp.connect(hp);
      hp.connect(master);
      noise.start();
      nodes.push(noise, bp, hp);
    } else if (type === 'fire') {
      const noise = ctx.createBufferSource();
      noise.buffer = createNoiseBuffer(ctx, 'brown');
      noise.loop = true;
      const bp = ctx.createBiquadFilter();
      bp.type = 'bandpass';
      bp.frequency.value = 800;
      bp.Q.value = 1;
      const crackleGain = ctx.createGain();
      crackleGain.gain.value = 0.5;
      const cid = window.setInterval(() => {
        if (Math.random() < 0.3 && ctx.state === 'running') {
          crackleGain.gain.setValueAtTime(0.8, ctx.currentTime);
          crackleGain.gain.exponentialRampToValueAtTime(0.3, ctx.currentTime + 0.1);
        }
      }, 200);
      intervals.push(cid);
      noise.connect(bp);
      bp.connect(crackleGain);
      crackleGain.connect(master);
      noise.start();
      nodes.push(noise, bp, crackleGain);
    } else if (type === 'stream') {
      const noise = ctx.createBufferSource();
      noise.buffer = createNoiseBuffer(ctx, 'white');
      noise.loop = true;
      const hp = ctx.createBiquadFilter();
      hp.type = 'highpass';
      hp.frequency.value = 2000;
      const lfo = ctx.createOscillator();
      lfo.frequency.value = 6;
      const lfoGain = ctx.createGain();
      lfoGain.gain.value = 0.1;
      lfo.connect(lfoGain);
      lfoGain.connect(master.gain);
      lfo.start();
      noise.connect(hp);
      hp.connect(master);
      noise.start();
      nodes.push(noise, hp, lfo, lfoGain);
    } else if (type === 'forest') {
      const noise = ctx.createBufferSource();
      noise.buffer = createNoiseBuffer(ctx, 'pink');
      noise.loop = true;
      const lp = ctx.createBiquadFilter();
      lp.type = 'lowpass';
      lp.frequency.value = 2000;
      noise.connect(lp);
      lp.connect(master);
      noise.start();
      nodes.push(noise, lp);
      const bid = window.setInterval(() => {
        if (Math.random() < 0.4 && ctx.state === 'running') {
          const osc = ctx.createOscillator();
          const bg = ctx.createGain();
          osc.frequency.value = 2000 + Math.random() * 3000;
          osc.type = 'sine';
          bg.gain.setValueAtTime(0, ctx.currentTime);
          bg.gain.linearRampToValueAtTime(0.06, ctx.currentTime + 0.05);
          bg.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.3);
          osc.connect(bg);
          bg.connect(ctx.destination);
          osc.start();
          osc.stop(ctx.currentTime + 0.3);
        }
      }, 2000);
      intervals.push(bid);
    } else if (type === 'night') {
      const noise = ctx.createBufferSource();
      noise.buffer = createNoiseBuffer(ctx, 'brown');
      noise.loop = true;
      const lp = ctx.createBiquadFilter();
      lp.type = 'lowpass';
      lp.frequency.value = 200;
      const g = ctx.createGain();
      g.gain.value = 0.15;
      noise.connect(lp);
      lp.connect(g);
      g.connect(master);
      noise.start();
      nodes.push(noise, lp, g);
      const drone = ctx.createOscillator();
      drone.frequency.value = 80;
      drone.type = 'sine';
      const dg = ctx.createGain();
      dg.gain.value = 0.04;
      drone.connect(dg);
      dg.connect(master);
      drone.start();
      nodes.push(drone, dg);
    }

    audioRef.current = { ctx, nodes, intervals };
  }, []);

  const play = useCallback((title: string, type: SoundType) => {
    if (playing === title) {
      stopSound();
      setPlaying(null);
      setProgress(0);
      if (timerRef.current) clearInterval(timerRef.current);
      return;
    }
    stopSound();
    setPlaying(title);
    setProgress(0);
    startSound(type);
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = setInterval(() => {
      setProgress(p => {
        if (p >= DURATION) {
          clearInterval(timerRef.current);
          stopSound();
          setPlaying(null);
          return 0;
        }
        return p + 1;
      });
    }, 1000);
  }, [playing, stopSound, startSound]);

  useEffect(() => () => {
    if (timerRef.current) clearInterval(timerRef.current);
    stopSound();
  }, [stopSound]);

  return (
    <div>
      <Text type="secondary" style={{ display: 'block', marginBottom: 16, fontSize: 13, textAlign: 'center' }}>
        选一个自然声音，闭上眼睛，让声音带你进入平静。
      </Text>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12 }}>
        {MEDITATION_TRACKS.map(track => {
          const isActive = playing === track.title;
          const pct = isActive ? (progress / DURATION) * 100 : 0;
          return (
            <Card key={track.title} hoverable onClick={() => play(track.title, track.type)}
              bodyStyle={{ padding: '16px 12px', textAlign: 'center' }}
              style={{
                borderRadius: 16, border: isActive ? `2px solid ${track.color}` : '1px solid #f0f0f0',
                background: isActive ? `${track.color}08` : '#fff',
                transition: 'all 0.3s',
              }}>
              <div style={{ fontSize: 36, marginBottom: 8 }}>{track.emoji}</div>
              <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 4 }}>{track.title}</div>
              <div style={{ fontSize: 12, color: '#999', marginBottom: 8 }}>{track.desc}</div>
              {isActive && (
                <div>
                  <div style={{ height: 4, background: '#f0f0f0', borderRadius: 2, overflow: 'hidden', marginBottom: 6 }}>
                    <div style={{ width: `${pct}%`, height: '100%', background: track.color, borderRadius: 2, transition: 'width 1s' }} />
                  </div>
                  <Text style={{ fontSize: 11, color: track.color }}>
                    {Math.floor(progress / 60)}:{String(progress % 60).padStart(2, '0')} / 5:00
                  </Text>
                </div>
              )}
              {!isActive && <Tag style={{ fontSize: 11 }}>{track.duration}</Tag>}
            </Card>
          );
        })}
      </div>
      <div style={{ textAlign: 'center', marginTop: 16, padding: 12, background: '#f6ffed', borderRadius: 12 }}>
        <Text style={{ fontSize: 12, color: '#52c41a' }}>
          {'\u{1F3A7}'} 声音由 Web Audio API 实时合成，建议佩戴耳机体验更佳。
        </Text>
      </div>
    </div>
  );
}

// ============================================================
// 5. 涂鸦画板
// ============================================================
function DoodleCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawingRef = useRef(false);
  const lastPosRef = useRef<{ x: number; y: number } | null>(null);
  const [color, setColor] = useState('#722ed1');
  const [brushSize, setBrushSize] = useState(4);
  const colors = ['#722ed1', '#f5222d', '#fa8c16', '#52c41a', '#1890ff', '#eb2f96', '#333333', '#ffffff'];

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.fillStyle = '#fafafa';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }, []);

  const getPos = (e: React.MouseEvent | React.TouchEvent) => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
    const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
    return {
      x: (clientX - rect.left) * (canvas.width / rect.width),
      y: (clientY - rect.top) * (canvas.height / rect.height),
    };
  };

  const startDraw = (e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault();
    drawingRef.current = true;
    lastPosRef.current = getPos(e);
  };

  const draw = (e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault();
    if (!drawingRef.current || !lastPosRef.current) return;
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext('2d')!;
    const pos = getPos(e);
    ctx.beginPath();
    ctx.moveTo(lastPosRef.current.x, lastPosRef.current.y);
    ctx.lineTo(pos.x, pos.y);
    ctx.strokeStyle = color;
    ctx.lineWidth = brushSize;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.stroke();
    lastPosRef.current = pos;
  };

  const stopDraw = () => {
    drawingRef.current = false;
    lastPosRef.current = null;
  };

  const clear = () => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = '#fafafa';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  };

  return (
    <div style={{ textAlign: 'center' }}>
      <Text type="secondary" style={{ display: 'block', marginBottom: 12, fontSize: 13 }}>
        随意涂鸦，把情绪画出来。没有对错，画完可以清空重来。
      </Text>
      <canvas ref={canvasRef} width={400} height={300}
        onMouseDown={startDraw} onMouseMove={draw} onMouseUp={stopDraw} onMouseLeave={stopDraw}
        onTouchStart={startDraw} onTouchMove={draw} onTouchEnd={stopDraw}
        style={{
          width: '100%', maxWidth: 400, height: 'auto', borderRadius: 16,
          border: '2px solid #f0f0f0', cursor: 'crosshair', touchAction: 'none',
          background: '#fafafa',
        }}
      />
      <div style={{ marginTop: 12, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 12, flexWrap: 'wrap' }}>
        {colors.map(c => (
          <div key={c} onClick={() => setColor(c)} style={{
            width: 28, height: 28, borderRadius: '50%', background: c, cursor: 'pointer',
            border: color === c ? '3px solid #333' : '2px solid #ddd',
            transition: 'all 0.2s',
          }} />
        ))}
        <span style={{ fontSize: 12, color: '#999', marginLeft: 8 }}>粗细：</span>
        {[2, 4, 8, 14].map(s => (
          <div key={s} onClick={() => setBrushSize(s)} style={{
            width: 28, height: 28, borderRadius: '50%', background: '#f0f0f0',
            display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer',
            border: brushSize === s ? '2px solid #722ed1' : '1px solid #ddd',
          }}>
            <div style={{ width: s, height: s, borderRadius: '50%', background: '#333' }} />
          </div>
        ))}
        <Button size="small" onClick={clear} style={{ borderRadius: 14 }}>清空</Button>
      </div>
    </div>
  );
}

// ============================================================
// 6. 橡皮人拉扯
// ============================================================
function RubberPerson() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState(false);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [stretch, setStretch] = useState({ x: 1, y: 1 });
  const [pullCount, setPullCount] = useState(0);
  const animRef = useRef<any>(null);
  const dragStartRef = useRef<{ x: number; y: number } | null>(null);

  const handleDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setDragging(true);
    dragStartRef.current = { x: e.clientX, y: e.clientY };
    if (animRef.current) cancelAnimationFrame(animRef.current);
  };

  useEffect(() => {
    if (!dragging) return;
    const handleMove = (e: MouseEvent) => {
      if (!dragStartRef.current) return;
      const dx = e.clientX - dragStartRef.current.x;
      const dy = e.clientY - dragStartRef.current.y;
      const maxStretch = 3;
      const sx = Math.max(0.3, Math.min(maxStretch, 1 + dx / 100));
      const sy = Math.max(0.3, Math.min(maxStretch, 1 + dy / 100));
      setStretch({ x: sx, y: sy });
      setOffset({ x: dx * 0.3, y: dy * 0.3 });
    };
    const handleUp = () => {
      setDragging(false);
      dragStartRef.current = null;
      setPullCount(c => c + 1);
      // 弹性回弹动画
      const startTime = Date.now();
      const startStretch = { ...stretch };
      const animate = () => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / 600, 1);
        // 弹性缓动
        const ease = 1 - Math.pow(1 - progress, 3) * Math.cos(progress * Math.PI * 2);
        const s = {
          x: startStretch.x + (1 - startStretch.x) * ease,
          y: startStretch.y + (1 - startStretch.y) * ease,
        };
        setStretch(s);
        setOffset({ x: offset.x * (1 - ease), y: offset.y * (1 - ease) });
        if (progress < 1) {
          animRef.current = requestAnimationFrame(animate);
        } else {
          setStretch({ x: 1, y: 1 });
          setOffset({ x: 0, y: 0 });
        }
      };
      animRef.current = requestAnimationFrame(animate);
    };
    window.addEventListener('mousemove', handleMove);
    window.addEventListener('mouseup', handleUp);
    return () => {
      window.removeEventListener('mousemove', handleMove);
      window.removeEventListener('mouseup', handleUp);
    };
  }, [dragging]);

  useEffect(() => () => { if (animRef.current) cancelAnimationFrame(animRef.current); }, []);

  return (
    <div style={{ textAlign: 'center' }}>
      <Text type="secondary" style={{ display: 'block', marginBottom: 12, fontSize: 13 }}>
        抓住小人用力拖拽！松手后会弹性回弹。拉扯释放压力，像捏解压玩具一样。
      </Text>
      <div ref={containerRef} style={{
        height: 280, display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: '#fafafa', borderRadius: 16, border: '2px dashed #e8e8e8',
        cursor: dragging ? 'grabbing' : 'grab', userSelect: 'none', overflow: 'hidden',
      }}
        onMouseDown={handleDown}>
        <div style={{
          transform: `translate(${offset.x}px, ${offset.y}px) scale(${stretch.x}, ${stretch.y})`,
          transition: dragging ? 'none' : 'transform 0.1s',
          fontSize: 80, filter: `hue-rotate(${pullCount * 30}deg)`,
        }}>          {'\u{1F9D1}'}
        </div>
      </div>
      <div style={{ marginTop: 12 }}>
        <Tag color="orange" style={{ fontSize: 14 }}>已拉扯 {pullCount} 次</Tag>
        {pullCount > 5 && <Tag color="green" style={{ fontSize: 12, marginLeft: 8 }}>压力释放中... 💨</Tag>}
        {pullCount > 15 && <Tag color="purple" style={{ fontSize: 12, marginLeft: 8 }}>解压大师！🏆</Tag>}
      </div>
    </div>
  );
}

// ============================================================
// 主页面
// ============================================================
const gameCategories = [
  {
    key: 'focus',
    label: (
      <span><BulbOutlined style={{ marginRight: 6, color: '#722ed1' }} />注意力 & 专注力</span>
    ),
    children: (
      <Row gutter={24}>
        <Col xs={24} lg={12}>
          <Card title="🔢 舒尔特方格" className="cloud-card" bodyStyle={{ padding: 24 }}>
            <SchulteGrid />
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="🎨 色词冲突" className="cloud-card" bodyStyle={{ padding: 24 }}>
            <StroopTest />
          </Card>
        </Col>
      </Row>
    ),
  },
  {
    key: 'relax',
    label: (
      <span><CloudOutlined style={{ marginRight: 6, color: '#52c41a' }} />缓解压力</span>
    ),
    children: (
      <Row gutter={24}>
        <Col xs={24} lg={12}>
          <Card title=" 气球呼吸训练" className="cloud-card" bodyStyle={{ padding: 24 }}>
            <BalloonBreathing />
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title=" 冥想自然音" className="cloud-card" bodyStyle={{ padding: 24 }}>
            <MeditationAudio />
          </Card>
        </Col>
      </Row>
    ),
  },
  {
    key: 'release',
    label: (
      <span>🎨 释放压力</span>
    ),
    children: (
      <Row gutter={24}>
        <Col xs={24} lg={12}>
          <Card title="🖌️ 涂鸦画板" className="cloud-card" bodyStyle={{ padding: 24 }}>
            <DoodleCanvas />
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="🧸 橡皮人拉扯" className="cloud-card" bodyStyle={{ padding: 24 }}>
            <RubberPerson />
          </Card>
        </Col>
      </Row>
    ),
  },
];

export default function Healing() {
  return (
    <div>
      <div style={{ marginBottom: 24, textAlign: 'center' }}>
        <Title level={3} style={{ marginBottom: 4 }}>
          <ExperimentOutlined style={{ color: '#ff8fab', marginRight: 8 }} />
          心理小游戏
        </Title>
        <Text type="secondary">选一个喜欢的游戏，在玩耍中照顾自己的心情 🌸</Text>
      </div>

      <Tabs items={gameCategories} defaultActiveKey="focus"
        tabBarStyle={{ marginBottom: 20 }}
        size="large"
      />
    </div>
  );
}
