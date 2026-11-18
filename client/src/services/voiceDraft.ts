export type VoiceDraftEvent = { type: 'partial' | 'final'; text: string };

/** 一段可编辑的语音草稿；识别修订只替换仍由识别器拥有的文本。 */
class Draft {
  text = '';
  private start = 0;
  private recognized = '';
  private owned = '';
  private edited = false;

  edit(text: string): void {
    if (text === this.text) return;
    // 用户一旦编辑，保护已有内容；后续只追加识别出的新尾部。
    this.text = text;
    this.edited = !!this.recognized;
    this.owned = '';
  }

  recognize(text: string, final: boolean): void {
    if (!this.recognized) this.start = this.text.length;
    if (this.edited) {
      if (text.startsWith(this.recognized)) this.text += text.slice(this.recognized.length);
    } else {
      this.text = this.text.slice(0, this.start) + text + this.text.slice(this.start + this.owned.length);
      this.owned = text;
    }
    this.recognized = text;
    if (final) { this.recognized = ''; this.owned = ''; this.start = this.text.length; this.edited = false; }
  }
}

/** 点击发送建立段边界；旧段定稿归本轮，之后的语音进入下一份草稿。 */
export class VoiceDraft {
  private draft = new Draft();
  private pending: Draft | null = null;
  get text(): string { return this.draft.text; }
  edit(text: string): void { this.draft.edit(text); }
  receive(event: VoiceDraftEvent): void {
    (this.pending ?? this.draft).recognize(event.text, event.type === 'final');
    if (event.type === 'final') this.pending = null;
  }
  beginSend(): { text: () => string; finish: () => void } {
    const submitted = this.draft;
    this.pending = submitted;
    this.draft = new Draft();
    return { text: () => submitted.text, finish: () => { if (this.pending === submitted) this.pending = null; } };
  }
}
