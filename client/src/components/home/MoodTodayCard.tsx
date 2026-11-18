/**
 * 首页「今日心情」卡片 —— 纯展示组件（所有请求都由父页面的 controller 驱动）。
 *
 * 为什么不在这里保存任何状态：
 *   打卡 / 更正 / 写备注 / 记睡眠都要落到同一个后端会话（今天已打卡时点别的表情走 PUT 更正），
 *   状态留在父页面才不会出现"卡片以为改了、服务器其实没改"。这里只把 controller 的字段映射成界面。
 *
 * 关于 `MOOD_OPTIONS`：
 *   父页面调打卡接口时要用 label（后端存中文标签）和 score（1-10 分），所以这份列表必须 export，
 *   全站只保留这一份，避免"表情改了但分数没改"的错位。
 */
import { useState } from 'react';
import './mood-today.css';

export interface MoodOption { emoji: string; label: string; score: number; }
export const MOOD_OPTIONS: MoodOption[] = [
  { emoji: '😄', label: '超好', score: 9 },
  { emoji: '😊', label: '不错', score: 7 },
  { emoji: '😐', label: '一般', score: 5 },
  { emoji: '😔', label: '低落', score: 3 },
  { emoji: '😢', label: '很难过', score: 1 },
];

export interface MoodTodayController {
  checkedIn: boolean;
  selectedMood: number | null;   // index into MOOD_OPTIONS
  checking: boolean;             // a check-in request is in flight
  streak: number;
  select: (index: number) => void;
  note: string;
  onNoteChange: (v: string) => void;
  saveNote: () => void;
  noteSaving: boolean;
  noteSaved: boolean;            // true right after a successful note save
  noteMaxLength: number;         // 60
  sleepStep: boolean;            // show the optional "昨晚睡了多久" step
  sleepHours: number | null;
  sleepLoading: boolean;
  recordSleep: (hours: number) => void;
  dismissSleep: () => void;
}

/** 睡眠选项：3-10 小时覆盖绝大多数情况，不做滚动选择器（首页不该出现三级交互） */
const SLEEP_HOURS = [4, 5, 6, 7, 8, 9, 10];

/**
 * 打卡后的一句回应，按心情松紧取用，措辞保持陪伴感、不下诊断。
 * 键是 MOOD_OPTIONS 的 score，避免与具体下标耦合。
 */
function moodFeedback(score: number): string {
  if (score >= 9) return '太好了，把这份好心情记住呀。';
  if (score >= 7) return '心情不错的时候，也可以记一件让你开心的小事。';
  if (score >= 5) return '平平淡淡也算数，愿意来看一眼自己就很好。';
  if (score >= 3) return '有点低落也没关系，先在这里放一会儿。';
  return '难过就难过一会儿吧，你已经很努力了。';
}

export default function MoodTodayCard({ controller }: { controller: MoodTodayController }) {
  const {
    checkedIn, selectedMood, checking, streak, select,
    note, onNoteChange, saveNote, noteSaving, noteSaved, noteMaxLength,
    sleepStep, sleepHours, sleepLoading, recordSleep, dismissSleep,
  } = controller;

  /**
   * 正在请求中的那个表情下标。
   * controller.select() 是"发射后不管"的，父页面不会回传 pending 下标，所以在点击那一刻自己记下来，
   * 请求结束（checking 转 false）后这个值自然失效，不用额外的 effect 去清理。
   */
  const [pendingIndex, setPendingIndex] = useState<number | null>(null);

  const selectedOption = selectedMood === null ? null : MOOD_OPTIONS[selectedMood] ?? null;
  const noteRemaining = Math.max(0, noteMaxLength - note.length);

  const handleMoodClick = (index: number) => {
    if (checking) return;
    setPendingIndex(index);
    select(index);
  };

  return (
    <section className="gh-card gh-card--raise gh-mood" aria-labelledby="gh-mood-title">
      <span className="gh-mood__blob" aria-hidden="true" />
      {/* 右上角的星点：原创内联 SVG，纯装饰 */}
      <svg className="gh-mood__spark" viewBox="0 0 64 64" fill="none" aria-hidden="true" focusable="false">
        <path d="M32 6l4.6 14.4L51 25l-14.4 4.6L32 44l-4.6-14.4L13 25l14.4-4.6L32 6Z" fill="url(#ghMoodSpark)" />
        <path d="M50 40l2.2 6.8L59 49l-6.8 2.2L50 58l-2.2-6.8L41 49l6.8-2.2L50 40Z" fill="url(#ghMoodSpark)" opacity="0.7" />
        <defs>
          <linearGradient id="ghMoodSpark" x1="14" y1="8" x2="56" y2="56" gradientUnits="userSpaceOnUse">
            <stop stopColor="#ffc78a" />
            <stop offset="1" stopColor="#cbbcec" />
          </linearGradient>
        </defs>
      </svg>

      <header className="gh-mood__head">
        <h3 className="gh-mood__title" id="gh-mood-title">今日心情</h3>
        <p className="gh-mood__sub">记录此刻的心情，给情绪一个出口</p>
      </header>

      <p className="gh-mood__hint">
        {checkedIn ? '今天已经打过卡啦，想改就点别的表情' : '今天心情怎么样？点一点记录一下'}
      </p>

      <div className="gh-mood__row" role="group" aria-label="选择今天的心情">
        {MOOD_OPTIONS.map((option, index) => {
          const isSelected = selectedMood === index;
          const isPending = checking && pendingIndex === index;
          return (
            <button
              key={option.label}
              type="button"
              className={`gh-mood__opt${isSelected ? ' gh-mood__opt--on' : ''}${isPending ? ' gh-mood__opt--pending' : ''}`}
              aria-pressed={isSelected}
              aria-label={option.label}
              disabled={checking}
              onClick={() => handleMoodClick(index)}
            >
              <span className="gh-mood__opt-emoji" aria-hidden="true">{option.emoji}</span>
              <span className="gh-mood__opt-label">{option.label}</span>
            </button>
          );
        })}
      </div>

      {/* 打卡后的暖色反馈条：表情 + 连续天数 + 一句陪伴的话 */}
      {checkedIn && selectedOption && (
        <div className="gh-mood__feedback gh-reveal">
          <span className="gh-mood__feedback-face" aria-hidden="true">{selectedOption.emoji}</span>
          <div className="gh-mood__feedback-body">
            <p className="gh-mood__feedback-top">
              <span className="gh-mood__feedback-label">今天记的是「{selectedOption.label}」</span>
              {streak > 0 && <span className="gh-chip gh-mood__streak">🔥 连续 {streak} 天</span>}
            </p>
            <p className="gh-mood__feedback-text">
              {streak > 0 ? moodFeedback(selectedOption.score) : '明天再来，就能连起来了'}
            </p>
          </div>
        </div>
      )}

      {/* 一句话备注：单行输入 + 字数 + 圆形发送键 */}
      <div className="gh-mood__note">
        <label className="gh-sr-only" htmlFor="gh-mood-note-input">
          写下一句话备注（最多 {noteMaxLength} 字，还可写 {noteRemaining} 字）
        </label>
        <div className="gh-mood__note-field">
          <input
            id="gh-mood-note-input"
            className="gh-mood__note-input"
            type="text"
            value={note}
            maxLength={noteMaxLength}
            placeholder="写下此刻的心情吧…"
            disabled={selectedMood === null}
            onChange={(e) => onNoteChange(e.target.value)}
            onKeyDown={(e) => {
              // 回车即保存，等价于点发送键（输入框是单行，回车没有别的语义）
              if (e.key === 'Enter' && selectedMood !== null && !noteSaving) {
                e.preventDefault();
                saveNote();
              }
            }}
          />
          <span className={`gh-mood__counter${noteRemaining <= 10 ? ' gh-mood__counter--low' : ''}`} aria-hidden="true">
            {note.length}/{noteMaxLength}
          </span>
          <button
            type="button"
            className={`gh-mood__send${noteSaved ? ' gh-mood__send--done' : ''}`}
            onClick={saveNote}
            disabled={selectedMood === null || noteSaving}
            aria-label={noteSaving ? '正在保存备注' : '保存备注'}
          >
            {noteSaving ? (
              <span className="gh-mood__send-spin" aria-hidden="true" />
            ) : noteSaved ? (
              <span aria-hidden="true">✿</span>
            ) : (
              <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" focusable="false" className="gh-mood__send-icon">
                <path d="M3.6 12 20.4 3.6 15 20.4l-3.6-6-7.8-2.4Z" fill="currentColor" />
              </svg>
            )}
          </button>
        </div>
        <div className="gh-mood__note-foot">
          {noteSaved ? (
            <span className="gh-mood__saved gh-reveal">已记下 ✿</span>
          ) : selectedMood === null ? (
            <span className="gh-mood__note-tip">先选一个心情，就能写下此刻的心情啦</span>
          ) : (
            <span className="gh-mood__note-tip">备注会保存在今天的打卡里</span>
          )}
        </div>
      </div>

      {/* 可选步骤：昨晚睡了多久（点「先不记」或记完自动收起，由父页面控制） */}
      {sleepStep && (
        <div className="gh-mood__sleep gh-reveal">
          <p className="gh-mood__sleep-title">🌙 昨晚睡了多久？</p>
          <div className="gh-mood__sleep-pills">
            {SLEEP_HOURS.map((h) => {
              const isChosen = sleepHours === h;
              // 请求进行中：所有 pill 一起禁用，避免连点产生第二条睡眠记录
              return (
                <button
                  key={h}
                  type="button"
                  className={`gh-mood__pill${isChosen ? ' gh-mood__pill--on' : ''}`}
                  disabled={sleepLoading}
                  aria-pressed={isChosen}
                  aria-label={`昨晚睡了 ${h} 小时`}
                  onClick={() => recordSleep(h)}
                >
                  {sleepLoading && isChosen ? <span className="gh-mood__pill-spin" aria-hidden="true" /> : `${h}h`}
                </button>
              );
            })}
          </div>
          <button type="button" className="gh-mood__skip" onClick={dismissSleep}>先不记</button>
        </div>
      )}
    </section>
  );
}
