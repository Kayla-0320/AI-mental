import { useEffect, useRef, useState } from 'react';
import { CameraOutlined } from '@ant-design/icons';
import { Progress } from 'antd';
import { mediaStreamManager } from '../services/mediaStreamManager';

export interface LiveQualityState {
  enabled: boolean; detecting: boolean; facePresent: boolean; reason: string;
  getSampleCount: () => number;
}

/** 只预览用户已开启的共享流；不申请设备、不保存画面、不自行给质量模型打分。 */
export default function LiveQualityPreview({ state, active }: { state: LiveQualityState; active: boolean }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const current = useRef(state);
  current.current = state;
  const [live, setLive] = useState({ connected: false, playing: false, count: 0, brightness: null as number | null, paused: false });
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !active) return;
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 16;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    let disposed = false;
    const tick = (): void => {
      const stream = mediaStreamManager.getStream('camera');
      const connected = !!stream?.getVideoTracks().some(track => track.readyState === 'live');
      if (video.srcObject !== stream) {
        video.srcObject = connected ? stream : null;
        if (connected) void video.play().catch(() => undefined);
      }
      const paused = document.hidden;
      const playing = connected && !paused && video.readyState >= 2 && video.videoWidth > 0;
      let brightness: number | null = null;
      if (playing && ctx) {
        try {
          ctx.drawImage(video, 0, 0, 16, 16);
          const pixels = ctx.getImageData(0, 0, 16, 16).data;
          let total = 0;
          for (let i = 0; i < pixels.length; i += 4) total += .2126 * pixels[i] + .7152 * pixels[i + 1] + .0722 * pixels[i + 2];
          brightness = Math.round(total / (pixels.length / 4));
        } catch { /* 未就绪时保留未知值。 */ }
      }
      if (!disposed) setLive({ connected, playing, brightness, paused, count: connected && !paused ? current.current.getSampleCount() : 0 });
    };
    tick();
    const timer = window.setInterval(tick, 300);
    return () => { disposed = true; window.clearInterval(timer); video.pause(); video.srcObject = null; };
  }, [active]);
  const status = live.paused ? '已暂停' : !live.connected ? state.enabled ? '摄像头未连接' : '摄像头未开启'
    : !live.playing ? '等待视频画面' : !state.detecting ? '检测器准备中' : !state.facePresent ? '未检测到人脸'
      : live.count < 3 ? `正在采样 ${live.count}/3` : '画面已就绪';
  const hint = live.paused ? '回到页面后继续采样。' : !live.connected ? state.reason || '用聊天框左下角的摄像头按钮开启。'
    : !state.detecting ? '画面已经接通，正在准备人脸检测。'
      : !state.facePresent ? '请面向镜头，让脸完整出现在画面内。'
        : live.count < 3 ? '正在收集这一轮可用的三张缩略帧。' : '已采集 3 帧。现在发一句话，回复后会更新下方质量检查结果。';
  return <section className="quality-live" aria-label="实时摄像头与采样状态" data-testid="quality-live">
    <div className="quality-visual-heading"><strong>实时画面</strong><span className="quality-visual-badge" role="status">{status}</span></div>
    <div className="quality-live-preview">
      <video ref={videoRef} muted playsInline aria-label="本机摄像头实时预览" style={{ visibility: live.playing ? 'visible' : 'hidden' }} />
      {!live.playing && <div className="quality-live-placeholder"><CameraOutlined /><span>{status}</span></div>}
      {live.playing && <span className="quality-live-chip">本机预览</span>}
    </div>
    <div className="quality-live-sampling" aria-label={`已采样 ${live.count} / 3 帧`}>
      {[0, 1, 2].map(index => <span key={index} className={live.count > index ? 'is-ready' : ''} />)}
      <small>{live.count} / 3 帧</small>
    </div>
    <div className="quality-live-brightness"><span>实时亮度</span><Progress percent={live.brightness === null ? 0 : live.brightness / 255 * 100} showInfo={false} size="small" strokeColor="#a89bc0" /><small>{live.brightness === null ? '—' : `${live.brightness}/255`}</small></div>
    <p>{hint}</p>
  </section>;
}
