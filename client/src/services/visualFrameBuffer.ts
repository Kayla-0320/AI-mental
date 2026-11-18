export interface VisualFrame { sampledAt: number; jpeg: string }

/** 只保留内存中的最近三张缩略帧，聊天与通话共用相同契约。 */
export class VisualFrameBuffer {
  private frames: VisualFrame[] = [];
  private canvas: HTMLCanvasElement | null = null;

  clear(): void { this.frames = []; }

  capture(video: HTMLVideoElement, now = Date.now()): void {
    if (document.hidden || video.readyState < 2 || !video.videoWidth || !video.videoHeight) { this.clear(); return; }
    if (this.frames.length && now - this.frames[this.frames.length - 1].sampledAt < 1000) return;
    this.canvas ??= document.createElement('canvas');
    this.canvas.width = 224; this.canvas.height = 224;
    try {
      const ctx = this.canvas.getContext('2d');
      if (!ctx) { this.clear(); return; }
      const side = Math.min(video.videoWidth, video.videoHeight);
      ctx.drawImage(video, (video.videoWidth - side) / 2, (video.videoHeight - side) / 2, side, side, 0, 0, 224, 224);
      this.frames = [...this.frames, { sampledAt: now, jpeg: this.canvas.toDataURL('image/jpeg', .75) }].slice(-3);
    } catch { this.clear(); }
  }

  count(now = Date.now()): number {
    if (document.hidden) { this.clear(); return 0; }
    this.frames = this.frames.filter(frame => now >= frame.sampledAt && now - frame.sampledAt <= 10000);
    return this.frames.length;
  }

  snapshot(now = Date.now()): VisualFrame[] {
    return this.count(now) === 3 ? this.frames.map(frame => ({ ...frame })) : [];
  }
}
