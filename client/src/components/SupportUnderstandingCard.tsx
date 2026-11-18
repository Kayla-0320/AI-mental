import { useEffect, useRef, useState, type ReactNode } from 'react';
import { Alert, Input } from 'antd';
import { MessageOutlined } from '@ant-design/icons';
import { consultationApi } from '../services';
import './support-echo.css';

export interface SupportStateView {
  schema_version: 'support.v1';
  session_id: string;
  revision: number;
  goal: 'unspecified' | 'listen' | 'understand' | 'act';
  declined: boolean;
  items: Array<{
    id: string; slot: string; quote: string; source_id: string;
    status: 'proposed' | 'confirmed' | 'corrected' | 'rejected'; origin: string;
  }>;
}

export function supportStateFromSentiment(raw?: string): SupportStateView | null {
  try {
    const state = JSON.parse(raw || '{}').support_state;
    return state?.schema_version === 'support.v1' && typeof state.session_id === 'string'
      && Number.isInteger(state.revision) && state.revision >= 0 && typeof state.declined === 'boolean'
      && ['unspecified', 'listen', 'understand', 'act'].includes(state.goal)
      && Array.isArray(state.items) && state.items.length <= 4 && state.items.every((item: any) =>
        typeof item?.id === 'string' && typeof item?.quote === 'string' && typeof item?.source_id === 'string'
        && ['proposed', 'confirmed', 'corrected', 'rejected'].includes(item.status)
        && ['user_message', 'user_correction'].includes(item.origin)) ? state : null;
  } catch { return null; }
}

const labels: Record<string, string> = { event: '发生的事', thought: '你心里的想法', feeling: '你的感受', concern: '你心里在意的' };

const GOAL_OPTIONS: Array<[Exclude<SupportStateView['goal'], 'unspecified'>, string]> = [
  ['listen', '先听我说'],
  ['understand', '一起理清'],
  ['act', '一起想办法'],
];

const GOAL_LABELS: Record<string, string> = { listen: '先听你说', understand: '一起理清', act: '一起想办法' };

/**
 * 「我听到的」回声条。
 *
 * 旧版是一张卡：每条原话配「待你确认 / 是这个意思 / 改一下 / 不是这个意思」，
 * 再让用户选支持目标。那等于在情绪最脆弱的时候让用户给 AI 的理解做验收 ——
 * 把「被听见」变成了「填表单」，还把每轮对话都变成一次追问。
 *
 * 现在默认只陈述「我听到你说的是……」，不索取任何确认；用户点某一句原话，
 * 这一句就地变成可编辑的输入框，改完保存即可。确认从「必须完成的动作」
 * 变成「不必发生的事」，修正从「逐条按钮」变成「点哪句改哪句」。
 * 陪伴方式只在开头轻轻问一次，之后保留一行可改的状态，不再反复追问。
 */
export default function SupportUnderstandingCard({ conversationId, messageId, state, onSaved, onBusyChange, showGoalOffer = true }: {
  conversationId: string; messageId: string; state: SupportStateView;
  onSaved: (state: SupportStateView) => void; onBusyChange?: (busy: boolean) => void;
  /** 只在对话开头展示「这次，你想我怎么陪你」，避免每轮都变成选择题。 */
  showGoalOffer?: boolean;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [editing, setEditing] = useState<string | null>(null);
  const [correction, setCorrection] = useState('');
  const [goalOpen, setGoalOpen] = useState(false);
  const rootRef = useRef<HTMLElement | null>(null);

  // 输入框由 autoFocus 接管焦点；整块用 nearest 轻推一下，避免被输入区挡住。
  useEffect(() => {
    if (!editing) return;
    const frame = window.requestAnimationFrame(() => {
      rootRef.current?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [editing]);

  const submit = async (action: string, itemId = '', text = '', goal = '') => {
    if (busy) return;
    setBusy(true); onBusyChange?.(true); setError('');
    try {
      const res = await consultationApi.submitSupportFeedback(conversationId, messageId, {
        feedback_id: crypto.randomUUID(), expected_revision: state.revision,
        action, item_id: itemId, text, ...(goal ? { goal } : {}),
      }) as any;
      const updated = supportStateFromSentiment(JSON.stringify({ support_state: res.data?.supportState }));
      if (!updated || updated.session_id !== conversationId) throw new Error('反馈未保存，请刷新后重试');
      onSaved(updated);
      if (action === 'set_goal') setGoalOpen(false);
      else { setEditing(null); setCorrection(''); }
    } catch (err: any) {
      setError(err?.message ? `刚才那句话我没记上。${err.message}` : '刚才那句话我没记上，你再试一次好吗？');
    }
    finally { setBusy(false); onBusyChange?.(false); }
  };

  if (state.session_id !== conversationId) return null;

  const items = state.items.filter(item => item.status !== 'rejected');
  const hasContent = items.length > 0;
  const activeItem = editing ? state.items.find(item => item.id === editing && item.status !== 'rejected') ?? null : null;
  const offerGoal = !state.declined && showGoalOffer && state.goal === 'unspecified';
  const hasGoalSummary = !state.declined && state.goal !== 'unspecified';
  if (!hasContent && !state.declined && !offerGoal && !hasGoalSummary) return null;

  const startEdit = (item: SupportStateView['items'][number]): void => {
    if (busy) return;
    setEditing(item.id); setCorrection(item.quote);
  };
  const cancelEdit = (): void => { setEditing(null); setCorrection(''); };
  const jumpToSource = (item: SupportStateView['items'][number]): void => {
    document.querySelector(`[data-msg-id="${CSS.escape(item.source_id)}"]`)
      ?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  };

  const goalPicker = (lead: string, trailing?: ReactNode) => (
    <div className="support-echo-goal">
      <span className="support-echo-goal-label">{lead}</span>
      <div className="support-echo-chips" role="group" aria-label="本次支持目标">
        {GOAL_OPTIONS.map(([value, text]) => <button key={value} type="button" className="support-echo-chip"
          aria-pressed={state.goal === value} disabled={busy} onClick={() => submit('set_goal', '', '', value)}>
          {text}
        </button>)}
      </div>
      {trailing}
    </div>
  );

  const goalRow = offerGoal ? goalPicker('这次，你想我怎么陪你？')
    : hasGoalSummary ? (goalOpen
      ? goalPicker('这次，你想我怎么陪你？', <button type="button" className="support-echo-action is-quiet" disabled={busy}
        onClick={() => setGoalOpen(false)}>就这个</button>)
      : <div className="support-echo-goal">
        <span className="support-echo-goal-label">现在我陪你：{GOAL_LABELS[state.goal] || ''}</span>
        <button type="button" className="support-echo-action is-quiet" disabled={busy}
          onClick={() => setGoalOpen(true)}>换个方式</button>
      </div>)
    : null;

  return <section ref={rootRef} className="support-echo" data-testid="support-understanding" aria-label="我听到的你" aria-busy={busy}>
    {state.declined ? <div className="support-echo-head">
      <span className="support-echo-icon" aria-hidden><MessageOutlined /></span>
      <p className="support-echo-text">好，这次先不记了。你想说的时候，我一直在。
        <button type="button" className="support-echo-link" disabled={busy} onClick={() => submit('resume')}>还是记一下吧</button>
      </p>
    </div> : <>
      {hasContent && <div className="support-echo-head">
        <span className="support-echo-icon" aria-hidden><MessageOutlined /></span>
        {activeItem ? <div className="support-echo-edit" data-evidence-slot={activeItem.slot}>
          <div className="support-echo-edit-label">我可能没听准 · {labels[activeItem.slot] || '你说的这句'}</div>
          <Input.TextArea aria-label="修正我的意思" autoFocus value={correction}
            onChange={e => setCorrection(e.target.value)} maxLength={240} showCount
            autoSize={{ minRows: 2, maxRows: 5 }} disabled={busy} />
          <div className="support-echo-edit-actions">
            <button type="button" className="support-echo-save" disabled={!correction.trim() || correction.trim() === activeItem.quote || busy}
              onClick={() => submit('correct', activeItem.id, correction)}>{busy ? '正在改…' : '改好了'}</button>
            <button type="button" className="support-echo-action is-quiet" disabled={busy} onClick={cancelEdit}>先不改了</button>
            <button type="button" className="support-echo-action" disabled={busy}
              onClick={() => submit('reject', activeItem.id)}>别记这句</button>
            {activeItem.origin === 'user_message' && <button type="button" className="support-echo-action is-quiet"
              disabled={busy} onClick={() => jumpToSource(activeItem)}>看看原话</button>}
          </div>
          <div className="support-echo-foot">
            <span className="support-echo-note">我记这些，只是想更懂你一点。</span>
            <button type="button" className="support-echo-action is-quiet" disabled={busy}
              onClick={() => submit('decline')}>这次先不记了</button>
          </div>
        </div> : <p className="support-echo-quotes">
          <span className="support-echo-label">我听到你说的是</span>
          {items.map((item, index) => <span key={item.id}>
            {index > 0 && <span className="support-echo-sep" aria-hidden> · </span>}
            <button type="button" data-evidence-slot={item.slot}
              className={item.status === 'corrected' ? 'support-echo-quote-button is-mine' : 'support-echo-quote-button'}
              title="点一下，改这句" aria-label={`修改这句：${item.quote}`} disabled={busy}
              onClick={() => startEdit(item)}>“{item.quote}”</button>
          </span>)}
        </p>}
        {!activeItem && <button type="button" className="support-echo-link" disabled={busy}
          onClick={() => startEdit(items[0])}>我理解对了吗</button>}
      </div>}
      {goalRow}
    </>}
    {error && <Alert role="alert" type="error" showIcon message={error} className="support-echo-error" />}
    <span className="support-echo-live" aria-live="polite">{busy ? '我在记…' : ''}</span>
  </section>;
}
