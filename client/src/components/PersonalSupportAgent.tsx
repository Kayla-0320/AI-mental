import { useCallback, useEffect, useRef, useState } from 'react';
import { Alert, Button, Card, Select, Space, Switch, Tag, Typography } from 'antd';
import api from '../services/api';
import { Line, LineChart, ResponsiveContainer, Tooltip as ChartTooltip, XAxis, YAxis } from 'recharts';

type Tool = 'breathing' | 'music';
interface SupportAction {
  id: string; tool: Tool; title: string; resourceId: string; durationSeconds: number; reason: string;
  status: 'offered' | 'accepted' | 'running' | 'completed' | 'failed' | 'cancelled' | 'declined' | 'expired';
  expiresAt: string; origin?: 'page' | 'background'; executionToken?: string; feedback?: 'helpful' | 'neutral' | 'unhelpful';
}
interface Overview {
  state: { schema: 'support-agent.v1'; revision: number; enabled: boolean; preferredTool: 'none' | Tool; actions: SupportAction[];
    background: { enabled: boolean; notifyInApp: boolean; intervalHours: 6 | 12 | 24; quietStart: number; quietEnd: number; nextRunAt: string | null; events: Array<{at: string; result: string}> } };
  trend: { forecast?: { status: 'experimental' | 'abstain'; reason?: string; targetDate?: string; score?: number; observedDays: number }; observations: number; latestScore: number | null; mean: number | null; change: number | null; stale: boolean;
    label: string; points: Array<{ id: string; score: number; at: string }> };
  suggestionEligible: boolean;
  context: { goal: 'unspecified' | 'listen' | 'understand' | 'act'; status: string; evidence: Array<{ id: string; quote: string; origin: string }>; revoked: number };
  modelReference: { available: boolean; task_id: string; probabilities: number[]; limitations: string[] } | null;
}
type CommandBody = { op: string; requestId: string; expectedRevision: number; [key: string]: unknown };
type ResponseData = { data: Overview };
const statusText: Record<SupportAction['status'], string> = {
  offered: '等你说要不要', accepted: '你答应了，还没开始', running: '正在进行', completed: '完成了', failed: '这次没成功', cancelled: '已经停下', declined: '这次先不用', expired: '已经过去了',
};
// 与疗愈空间「雨打窗棂」相同的固定音源；服务端仅返回资源 ID，不接受任意 URL。
const RAIN_URL = 'https://assets.mixkit.co/active_storage/sfx/2394/2394-preview.mp3';

/** 本人自评追踪与可撤回的站内支持活动；关闭时不读取历史、不自动播放。 */
export default function PersonalSupportAgent({ embedded = false, refreshKey = '' }: { embedded?: boolean; refreshKey?: string } = {}): JSX.Element {
  const [overview, setOverview] = useState<Overview | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [progress, setProgress] = useState<number | null>(null);
  const [phase, setPhase] = useState('');
  const [modelMessage, setModelMessage] = useState('');
  const current = useRef<Overview | null>(null);
  const pending = useRef<CommandBody | null>(null);
  const mounted = useRef(true);
  const active = useRef<{ id: string; stopped: boolean; stop: () => void } | null>(null);
  const inFlight = useRef(false);
  const suggestedRevision = useRef(-1);
  const previousRefreshKey = useRef(refreshKey);

  const adopt = useCallback((value: Overview): void => {
    current.current = value;
    const running = active.current;
    const action = running && value.state.actions.find(item => item.id === running.id);
    if (running && (!value.state.enabled || !action || !['accepted', 'running'].includes(action.status))) {
      running.stopped = true; running.stop(); active.current = null;
      if (mounted.current) { setProgress(null); setPhase(''); }
    }
    if (mounted.current) setOverview(value);
  }, []);
  const refresh = useCallback(async (): Promise<void> => {
    const revisionBefore = current.current?.state.revision;
    const result = await api.get<unknown, ResponseData>('/support-agent');
    // GET 期间若本页已有更新，旧快照不能覆盖刚确认的命令结果。
    if (current.current?.state.revision === revisionBefore || result.data.state.revision >= (current.current?.state.revision ?? 0)) adopt(result.data);
  }, [adopt]);
  const transmit = useCallback(async (body: CommandBody): Promise<Overview> => {
    if (inFlight.current) throw new Error('我还在处理上一步，你稍等一下好吗？');
    inFlight.current = true;
    if (mounted.current) { setBusy(true); setError(''); }
    pending.current = body;
    try {
      const result = await api.post<unknown, ResponseData>('/support-agent/commands', body);
      pending.current = null; adopt(result.data); return result.data;
    } catch (failure: unknown) {
      const detail = failure as { status?: number; message?: string };
      if (detail.status && detail.status < 500) pending.current = null;
      if (mounted.current) setError(detail.message || '刚才的记录还没确认存好，你可以再试一次；活动随时可以停。');
      if (detail.status === 409) await refresh().catch(() => undefined);
      throw failure;
    } finally { inFlight.current = false; if (mounted.current) setBusy(false); }
  }, [adopt, refresh]);
  const send = useCallback(async (op: string, extra: Record<string, unknown> = {}): Promise<Overview> => {
    if (pending.current) throw new Error('上一次还没存好，先再试一次好吗？');
    return transmit({ op, ...extra, requestId: crypto.randomUUID(), expectedRevision: current.current?.state.revision ?? 0 });
  }, [transmit]);

  useEffect(() => {
    // 仅在本人开启、近期有自评且服务器允许时主动展示建议；提议不会启动工具。
    const revision = overview?.state.revision;
    if (!overview?.suggestionEligible || overview.state.background.enabled || busy || pending.current || revision === undefined || suggestedRevision.current === revision) return;
    suggestedRevision.current = revision;
    void send('propose').catch(() => undefined);
  }, [overview, busy, send]);

  useEffect(() => {
    mounted.current = true;
    refresh().catch(() => setError('我这边暂时没接上，你过一会儿再看看好吗？'));
    return (): void => {
      mounted.current = false;
      const running = active.current;
      running?.stop();
      // 离开页面立即停止本地工具；服务器回执尽力发送，断网时不能冒称已经保存。
      if (running && !running.stopped) running.stopped = true;
      if (running && current.current?.state.enabled && !pending.current) {
        const body = { op: 'cancel', actionId: running.id, requestId: crypto.randomUUID(), expectedRevision: current.current.state.revision };
        const token = localStorage.getItem('accessToken');
        void fetch('/api/support-agent/commands', { method: 'POST', keepalive: true,
          headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}) }, body: JSON.stringify(body) }).catch(() => undefined);
      }
      active.current = null;
    };
  }, [refresh]);

  // 对话完成或本人修正后刷新依据，组件不重挂，进行中的活动保持连续。
  useEffect(() => {
    if (previousRefreshKey.current === refreshKey) return;
    previousRefreshKey.current = refreshKey;
    void refresh().catch(() => setError('我这边暂时没接上，你过一会儿再看看好吗？'));
  }, [refreshKey, refresh]);

  useEffect(() => {
    if (!active.current && !overview?.state.background.enabled) return;
    // 其他窗口撤回授权/取消时，当前本地工具在下一次轮询停止；不在后台自动执行工具。
    const timer = window.setInterval(() => { if (!inFlight.current) void refresh().catch(() => undefined); }, active.current ? 2000 : 15000);
    return (): void => window.clearInterval(timer);
  }, [overview?.state.revision, overview?.state.background.enabled, refresh]);

  const stopAndCancel = useCallback(async (actionId: string): Promise<void> => {
    if (active.current) { active.current.stopped = true; active.current.stop(); active.current = null; }
    setProgress(null); setPhase('');
    try {
      if (pending.current) await transmit(pending.current);
      await send('cancel', { actionId });
    } catch { /* 保存失败由 transmit 呈现；工具已停止。 */ }
  }, [send, transmit]);

  const start = useCallback(async (action: SupportAction): Promise<void> => {
    if (active.current || busy || pending.current) return;
    let audio: HTMLAudioElement | null = null;
    let frame = 0;
    let timeout = 0;
    let launchTimeout = 0;
    const running = { id: action.id, stopped: false, stop: (): void => {
      cancelAnimationFrame(frame); window.clearTimeout(timeout); window.clearTimeout(launchTimeout);
      if (audio) { audio.pause(); audio.removeAttribute('src'); audio.load(); }
    } };
    active.current = running;
    setProgress(0); setPhase('正在开始');
    try {
      if (action.tool === 'music') {
        if (action.resourceId !== 'rain_mixkit_2394') throw new Error('不支持的音源');
        audio = new Audio(RAIN_URL); audio.loop = true; audio.volume = 0.35;
        // 真正尝试播放后才记录 started；播放策略或网络失败会写失败回执。
        await Promise.race([audio.play(), new Promise<never>((_resolve, reject) => {
          launchTimeout = window.setTimeout(() => reject(new Error('播放启动超时')), 12000);
        })]);
        window.clearTimeout(launchTimeout);
      } else if (action.resourceId !== 'balloon_4_4_6_v1') throw new Error('不支持的练习');
      if (running.stopped || !mounted.current) return;
      await send('started', { actionId: action.id, executionToken: action.executionToken });
      if (running.stopped || !mounted.current) return;
      const began = performance.now();
      const tick = (): void => {
        if (running.stopped || !mounted.current) return;
        const elapsed = Math.min(action.durationSeconds, (performance.now() - began) / 1000);
        setProgress(elapsed);
        if (action.tool === 'breathing') {
          const cycle = elapsed % 14;
          setPhase(cycle < 4 ? '轻轻吸气' : cycle < 8 ? '舒适地停一停，不必勉强' : '慢慢呼气');
        } else setPhase('雨声在响');
        frame = requestAnimationFrame(tick);
      };
      frame = requestAnimationFrame(tick);
      timeout = window.setTimeout(() => {
        if (running.stopped || !mounted.current) return;
        running.stopped = true; running.stop(); active.current = null; setProgress(null); setPhase('');
        void send('complete', { actionId: action.id, executionToken: action.executionToken }).catch(() => undefined);
      }, action.durationSeconds * 1000 + 150);
      const playbackFailed = (): void => {
        if (running.stopped) return;
        running.stopped = true; running.stop(); active.current = null; setProgress(null);
        void send('fail', { actionId: action.id, executionToken: action.executionToken, reason: 'playback_failed' }).catch(() => undefined);
        setError('声音断了一下，我先停下了。');
      };
      if (audio) { audio.onerror = playbackFailed; audio.onwaiting = playbackFailed; audio.onstalled = playbackFailed; }
    } catch {
      const alreadyStopped = running.stopped;
      running.stopped = true; running.stop(); active.current = null; setProgress(null); setPhase('');
      if (alreadyStopped) return;
      if (!pending.current && mounted.current) {
        await send('fail', { actionId: action.id, executionToken: action.executionToken, reason: action.tool === 'music' ? 'playback_failed' : 'tool_failed' }).catch(() => undefined);
        setError('这次没能开始，我也没有把它算作完成。你可以换一个试试。');
      }
    }
  }, [busy, send]);

  useEffect(() => {
    const onHide = (): void => {
      if (document.hidden && active.current) void stopAndCancel(active.current.id);
    };
    document.addEventListener('visibilitychange', onHide);
    return (): void => document.removeEventListener('visibilitychange', onHide);
  }, [stopAndCancel]);

  const run = (op: string, extra: Record<string, unknown> = {}): void => { void send(op, extra).catch(() => undefined); };
  const latest = overview?.state.actions[overview.state.actions.length - 1];
  const isLocal = !!latest && active.current?.id === latest.id;
  const enabled = overview?.state.enabled ?? false;
  const background = overview?.state.background;
  const configureBackground = (extra: Record<string, unknown>): void => {
    if (!background) return;
    run('background_settings', { enabled: background.enabled, notifyInApp: background.notifyInApp, intervalHours: background.intervalHours,
      quietStart: background.quietStart, quietEnd: background.quietEnd, ...extra });
  };
  return (
    <Card data-testid="personal-support-agent" title={embedded ? undefined : '我会记得你'} variant={embedded ? 'borderless' : 'outlined'} style={{ marginBottom: embedded ? 0 : 24, borderRadius: embedded ? 0 : 20, background: embedded ? 'transparent' : undefined }}>
      <Space direction="vertical" size="middle" style={{ width: '100%' }}>
        <Typography.Paragraph style={{ marginBottom: 0 }}>
          如果你愿意，我会记着你最近的心情，也会在合适的时候给你一个小建议——比如一起呼吸一下，或者听一会儿雨声。开不开、做不做，都由你决定。
        </Typography.Paragraph>
        {error && <Alert type="warning" showIcon message={error} action={pending.current ? <Button size="small" disabled={busy}
          onClick={() => { if (pending.current) void transmit(pending.current).catch(() => undefined); }}>重试保存</Button> : <Button size="small" onClick={() => { void refresh().catch(() => undefined); }}>刷新</Button>} />}
        {!overview ? <Typography.Text type="secondary">我正在看看你之前的选择…</Typography.Text> : (
          <>
            <Space wrap>
              <Button type={enabled ? 'default' : 'primary'} disabled={busy || !!pending.current} onClick={() => {
                if (active.current) { active.current.stopped = true; active.current.stop(); active.current = null; setProgress(null); }
                run('configure', { enabled: !enabled, preferredTool: overview.state.preferredTool });
              }}>{enabled ? '关闭持续陪伴' : '开启持续陪伴'}</Button>
              {enabled && <Select aria-label="支持活动偏好" value={overview.state.preferredTool} disabled={busy || !!pending.current} style={{ width: 160 }}
                onChange={(preferredTool) => run('configure', { enabled: true, preferredTool })}
                options={[{ value: 'none', label: '没有特别偏好' }, { value: 'breathing', label: '偏好呼吸练习' }, { value: 'music', label: '偏好自然音' }]} />}
            </Space>
            {enabled && <>
              {background && <div style={{ border: '1px solid #eee', padding: 12, borderRadius: 12 }}>
                <Space wrap>
                  <Switch aria-label="后台持续建议" checked={background.enabled} disabled={busy || !!pending.current}
                    onChange={(checked) => configureBackground({ enabled: checked })} />
                  <Typography.Text>我不在页面时，也把想给你的小建议留在这里</Typography.Text>
                  <Select aria-label="后台建议频率" value={background.intervalHours} disabled={busy || !!pending.current} style={{ width: 130 }}
                    options={[{ value: 6, label: '每 6 小时' }, { value: 12, label: '每 12 小时' }, { value: 24, label: '每 24 小时' }]}
                    onChange={(intervalHours) => configureBackground({ intervalHours })} />
                </Space>
                <div style={{ marginTop: 8 }}><Typography.Text type="secondary">小建议会安静地留在这里，等你回来再决定要不要。不会自动播放，也不会发到站外。</Typography.Text></div>
                <Space wrap style={{ marginTop: 8 }}>
                  <Switch aria-label="站内建议提醒" checked={background.notifyInApp} disabled={!background.enabled || busy || !!pending.current}
                    onChange={(checked) => configureBackground({ notifyInApp: checked })} />
                  <Typography.Text>有新建议时，在站内提醒我一声</Typography.Text>
                </Space>
                <Space wrap style={{ marginTop: 8 }}>
                  <Typography.Text>不打扰你的时间（北京时间）</Typography.Text>
                  <Select aria-label="安静时段开始" value={background.quietStart} disabled={busy || !!pending.current} style={{ width: 90 }}
                    options={Array.from({ length: 24 }, (_, hour) => ({ value: hour, label: `${hour}:00` }))} onChange={(quietStart) => configureBackground({ quietStart })} />
                  <Typography.Text>至</Typography.Text>
                  <Select aria-label="安静时段结束" value={background.quietEnd} disabled={busy || !!pending.current} style={{ width: 90 }}
                    options={Array.from({ length: 24 }, (_, hour) => ({ value: hour, label: `${hour}:00` }))} onChange={(quietEnd) => configureBackground({ quietEnd })} />
                </Space>
                {background.enabled && <div><Typography.Text type="secondary">下一次：{background.nextRunAt ? new Date(background.nextRunAt).toLocaleString() : '等待安排'}。如果你没有新的记录，或者正好在不打扰你的时间，我就先不打扰。</Typography.Text></div>}
              </div>}
              {overview.context && (overview.context.status === 'current' || overview.context.status === 'declined') && <div>
                <Typography.Text strong>你告诉过我的事</Typography.Text>
                <div><Typography.Text type="secondary">这些来自我们最近的聊天。你改过、撤回或删掉的，我就不会再拿来用。</Typography.Text></div>
                {overview.context.evidence.map(item => <Typography.Paragraph key={item.id} style={{ marginTop: 8, marginBottom: 8 }}>“{item.quote}”</Typography.Paragraph>)}
                {overview.context.goal !== 'unspecified' && <Tag>{({ listen: '现在希望先被倾听', understand: '现在希望一起理清', act: '现在希望一起想办法' })[overview.context.goal]}</Tag>}
                {['listen', 'understand'].includes(overview.context.goal) && <div><Typography.Text type="secondary">你现在更想被听着，或者想先理清，我就不急着推荐活动。</Typography.Text></div>}
                {overview.context.evidence.length > 0 && <Button size="small" disabled={busy || !!pending.current} onClick={() => {
                  setBusy(true); setModelMessage('');
                  void api.post<unknown, ResponseData>('/support-agent/model-reference', {}).then(result => {
                    adopt(result.data); if (!result.data.modelReference?.available) setModelMessage('这次我没能给出参考，还是以你说的为准。');
                  }).catch(() => setModelMessage('这次没取到参考，还是以你说的为准。')).finally(() => setBusy(false));
                }}>看看这段文字读起来是什么感觉</Button>}
                {overview.modelReference?.available && <div style={{ marginTop: 8 }}><Typography.Text type="secondary">这段话读起来偏{['负向', '中性', '正向'][overview.modelReference.probabilities.indexOf(Math.max(...overview.modelReference.probabilities))]}。它只是文字给人的感觉，不一定是你现在的心情——你觉得呢？</Typography.Text></div>}
                {modelMessage && <div><Typography.Text type="secondary">{modelMessage}</Typography.Text></div>}
              </div>}
              {overview.context?.status === 'existing_escalation' && <Alert type="info" message="请继续使用当前支持与人工协助流程，这里暂不推送普通活动。" />}
              <div>
                <Typography.Text>{overview.trend.label}</Typography.Text>
                <div style={{ marginTop: 6, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  {overview.trend.points.map(point => <Tag key={point.id}>{new Date(point.at).toLocaleDateString()} · {point.score}/10</Tag>)}
                </div>
                {overview.trend.mean !== null && <Typography.Text type="secondary">最近 {overview.trend.observations} 次自评平均 {overview.trend.mean}/10，从第一次到现在 {overview.trend.change! > 0 ? '+' : ''}{overview.trend.change} 分。这只是你过去写下的感受，不代表现在。</Typography.Text>}
                {overview.trend.points.length >= 2 && <div role="img" aria-label="你的心情变化，1 到 10 分" style={{ height: 180, marginTop: 16 }}>
                  <ResponsiveContainer width="100%" height="100%"><LineChart data={overview.trend.points.map(point => ({
                    label: new Date(point.at).toLocaleString('zh-CN', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' }), score: point.score,
                  }))} margin={{ top: 8, right: 16, bottom: 4, left: -20 }}>
                    <XAxis dataKey="label" tick={{ fontSize: 11 }} minTickGap={35} /><YAxis domain={[1, 10]} ticks={[1, 5, 10]} tick={{ fontSize: 11 }} />
                    <ChartTooltip formatter={(value: number) => [`${value}/10`, '你当时的自评']} /><Line dataKey="score" stroke="#9b81ba" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
                  </LineChart></ResponsiveContainer>
                </div>}
                {overview.trend.forecast?.status === 'experimental' && <div style={{ marginTop: 8 }}><Typography.Text type="secondary">我试着猜一下 {overview.trend.forecast.targetDate} 的心情：{overview.trend.forecast.score}/10（基于你 {overview.trend.forecast.observedDays} 天的自评）。这只是个试试看的推测，还没在每个人身上验证过；你当天的感受更算数。</Typography.Text></div>}
                {overview.trend.forecast?.status === 'abstain' && overview.trend.observations > 0 && <div><Typography.Text type="secondary">{overview.trend.forecast.reason === 'latest_day_missing' ? '今天记一下心情，我就能看得更清楚一点。' : '多记几天，我才能看出一点变化；中间断了也没关系，我们重新开始。'}</Typography.Text></div>}
                {overview.trend.observations === 0 && <div><a href="/">去记一下今天的心情</a></div>}
              </div>
              {overview.suggestionEligible && <Button disabled={busy || !!pending.current} onClick={() => run('propose')}>看看有什么适合我的小建议</Button>}
              {latest && <div style={{ padding: 16, borderRadius: 12, background: '#faf7ff' }}>
                <Space wrap><Typography.Text strong>{latest.title}</Typography.Text><Tag>{statusText[latest.status]}</Tag>{latest.origin === 'background' && <Tag>我在后台记下的</Tag>}</Space>
                <Typography.Paragraph style={{ marginTop: 8 }}>{latest.reason}</Typography.Paragraph>
                {progress !== null && <div role="status" aria-live="polite" style={{ textAlign: 'center', padding: 16 }}>
                  <div aria-hidden style={{ fontSize: 44, transform: latest.tool === 'breathing' ? `scale(${phase === '轻轻吸气' ? 1.15 : 1})` : undefined, transition: 'transform 3s ease' }}>{latest.tool === 'breathing' ? '🎈' : '🌧️'}</div>
                  <div>{phase} · {Math.max(0, Math.ceil(latest.durationSeconds - progress))} 秒</div>
                  {latest.tool === 'breathing' && <Typography.Text type="secondary">按你舒服的节奏来；不舒服就停下，没关系的。</Typography.Text>}
                </div>}
                <Space wrap>
                  {latest.status === 'offered' && <>
                    <Button type="primary" disabled={busy || !!pending.current} onClick={() => run('accept', { actionId: latest.id })}>接受建议</Button>
                    <Button disabled={busy || !!pending.current} onClick={() => run('decline', { actionId: latest.id })}>这次先不用</Button>
                  </>}
                  {latest.status === 'accepted' && !isLocal && <Button type="primary" disabled={busy || !!pending.current} onClick={() => { void start(latest); }}>{latest.tool === 'music' ? '开始播放' : '开始练习'}</Button>}
                  {['accepted', 'running', 'offered'].includes(latest.status) && <Button danger onClick={() => { void stopAndCancel(latest.id); }}>停止并取消</Button>}
                  {latest.status === 'completed' && <>
                    <Typography.Text>刚才那个，对你有一点帮助吗？</Typography.Text>
                    {(['helpful', 'neutral', 'unhelpful'] as const).map((value, index) => <Button key={value} type={latest.feedback === value ? 'primary' : 'default'} disabled={busy || !!pending.current}
                      onClick={() => run('feedback', { actionId: latest.id, value })}>{['有一点', '一般', '没有'][index]}</Button>)}
                  </>}
                  {latest.status === 'running' && !isLocal && <Typography.Text type="secondary">这个页面没有正在进行的活动；刷新不会自动重来，你可以把这条建议先取消。</Typography.Text>}
                </Space>
              </div>}
            </>}
          </>
        )}
      </Space>
    </Card>
  );
}
