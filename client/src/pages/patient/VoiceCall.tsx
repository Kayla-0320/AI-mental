/**
 * AI 陪伴通话 —— 整屏页面
 *
 * ## 为什么是整屏、且刻意不进 `PatientLayout`
 *
 * 通话的视觉隐喻是"电话"：全屏、没有侧边栏、没有导航。挂上侧边栏就变回"网页"了。
 * 因此路由与 `/login` 同级（见 `App.tsx`），而不是挂在 `/` 下面。
 *
 * ## 页面负责什么 / 不负责什么
 *
 * 这个文件**只负责看和点**：布局、状态词、字幕、按钮、计时、跳转。
 * 轮次编排（ASR → 定稿 → SSE → TTS → 打断）全在 `useVoiceCall` 里；
 * 通话期视觉（存在感 / 光线 / 只给自己看的读数）全在 `useCallVision` 里 ——
 * 那两个 hook 才是需要写测试的地方。
 *
 * ## 为什么"开始通话"必须点一下（不是偷懒，是两个硬约束的交点）
 *
 * 1. `AudioContext` 必须在**用户手势**里 `resume()`，否则自动播放策略会让第一句话没声音，
 *    而且事后无法补救。
 * 2. 麦克风若在 effect 里申请，会撞上 React 18 StrictMode 的"假卸载"：先 cleanup
 *    （`release`）再重新 effect（`acquire`），而 `mediaStreamManager` 的引用计数是
 *    **按消费者名去重的 Set** —— 同一个名字记不了两次，于是后一次拿到的可能是
 *    **已经被 stop 掉的轨道**（麦看起来是开的、界面什么都不报，实际全是静音）。
 *    放进点击回调就只会申请一次，从根上绕开这个竞态。
 *
 * 摄像头同理：它也必须由点击打开（`useCallVision.start()` 会 `getUserMedia`）。
 *
 * ## `?debug=1`
 *
 * 打开一个**只显示真实测量值**的诊断面板（轮次耗时、ASR/TTS 引擎、回声判定、
 * 打断计数、麦克风**实际生效**的 AEC 值、视觉读数）。它不是演示数据 ——
 * 里面每个数字都是这一通电话里实测出来的，用来给 S4/S5/S6/S7 的闸门取数。
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';
import { Alert, Button, Card, Space, Tag, Tooltip, Typography } from 'antd';
import {
  ArrowLeftOutlined,
  AudioMutedOutlined,
  AudioOutlined,
  PhoneFilled,
  VideoCameraOutlined,
} from '@ant-design/icons';
import { consultationApi } from '../../services';
import { mediaStreamManager } from '../../services/mediaStreamManager';
import { useVoiceCall, type CallPhase } from '../../hooks/useVoiceCall';
import { useCallVision } from '../../hooks/useCallVision';
import CrisisResourceCard from '../../components/CrisisResourceCard';

const { Text, Title } = Typography;

const PHASE_LABEL: Record<CallPhase, string> = {
  idle: '准备就绪',
  connecting: '正在接通…',
  listening: '正在聆听',
  thinking: '正在思考…',
  speaking: '正在回应…',
  crisis: '已切换为文字陪伴',
  ended: '通话已结束',
  error: '需要注意',
};

/** 状态点的颜色：绿=在听，紫=在想/在说，橙=危机，红=异常。 */
const PHASE_DOT: Record<CallPhase, string> = {
  idle: '#a5b4fc',
  connecting: '#a5b4fc',
  listening: '#52c41a',
  thinking: '#8b5cf6',
  speaking: '#6366f1',
  crisis: '#fa8c16',
  ended: '#bfbfbf',
  error: '#ff4d4f',
};

/**
 * 头像律动。**纯装饰** —— 说话时快而明显、聆听时慢而轻。
 * 真正的信息在状态词与字幕里，所以读屏用户不会漏掉任何东西。
 */
const AVATAR_ANIMATION: Record<CallPhase, string> = {
  idle: 'call-breathe 4s ease-in-out infinite',
  connecting: 'call-breathe 2.4s ease-in-out infinite',
  listening: 'call-breathe 4s ease-in-out infinite',
  thinking: 'call-breathe 1.6s ease-in-out infinite',
  speaking: 'call-pulse 0.9s ease-in-out infinite',
  crisis: 'none',
  ended: 'none',
  error: 'none',
};

/** `mm:ss` */
function formatDuration(totalSeconds: number): string {
  const safe = Math.max(0, Math.floor(totalSeconds));
  return `${String(Math.floor(safe / 60)).padStart(2, '0')}:${String(safe % 60).padStart(2, '0')}`;
}

/** 一行的诊断读数 */
function DebugRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: 'flex', gap: 8, fontSize: 12 }}>
      <span style={{ color: '#8c8aa7', minWidth: 104 }}>{label}</span>
      <span data-debug={label} style={{ color: '#3f3d56', fontVariantNumeric: 'tabular-nums' }}>
        {value}
      </span>
    </div>
  );
}

/** 视觉小窗里那句"人话"（每一句都能对上一条实测状态，没有形容词撑场面） */
function presenceText(v: {
  paused: boolean;
  unavailableReason: string;
  frameFresh: boolean;
  present: boolean;
}): string {
  if (v.paused) return '采集已暂停（页面在后台）';
  if (v.unavailableReason) return v.unavailableReason;
  if (!v.frameFresh) return '画面已停住，读数暂空';
  return v.present ? '你在画面里' : '暂时看不到你';
}

/** 光线分档 → 一句人话 */
function lightText(level: 'ok' | 'dim' | 'dark' | 'unknown'): string {
  switch (level) {
    case 'ok':
      return '光线正常';
    case 'dim':
      return '光线偏暗';
    case 'dark':
      return '光线太暗，看不清';
    default:
      return '光线 —';
  }
}

export default function VoiceCall() {
  const { conversationId } = useParams<{ conversationId?: string }>();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const debugEnabled = searchParams.get('debug') === '1';

  const [conversationError, setConversationError] = useState<string | null>(null);
  const [retryToken, setRetryToken] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [muted, setMuted] = useState(false);
  const [hangingUp, setHangingUp] = useState(false);
  const [crisisCardDismissed, setCrisisCardDismissed] = useState(false);
  const [visionError, setVisionError] = useState<string | null>(null);
  const [visionBusy, setVisionBusy] = useState(false);
  /**
   * 网络断了（施工单 §S9.3 走查项："断网 5 秒 → 提示网络不稳定，正在重连…，已说出的字幕不丢"）。
   *
   * 为什么用浏览器的 `offline` / `online` 事件而不是等 WebSocket 报错：
   * 断网到 WS 真正超时之间有十几秒的静默期，用户那段时间看到的是一个"没反应"的界面。
   * `navigator.onLine` 是立刻的、并且是**真话**（它反映的就是系统网络状态）。
   *
   * ⚠️ 字幕**不清空**：已说出的内容留着，重连回来接着往上加。
   */
  const [networkDown, setNetworkDown] = useState(
    typeof navigator !== 'undefined' ? navigator.onLine === false : false,
  );

  useEffect(() => {
    const goOffline = (): void => setNetworkDown(true);
    const goOnline = (): void => setNetworkDown(false);
    window.addEventListener('offline', goOffline);
    window.addEventListener('online', goOnline);
    return () => {
      window.removeEventListener('offline', goOffline);
      window.removeEventListener('online', goOnline);
    };
  }, []);

  const bootstrapRef = useRef<Promise<string> | null>(null);
  const startedAtRef = useRef(0);

  const call = useVoiceCall({ conversationId });
  const { phase, userLines, partialText, aiLines, revised, notice, voiceMetrics, ttsStatus, turn } = call;
  /**
   * 通话期视觉。
   *
   * ⚠️ 它只是"把读数摆出来"：**不进风险、不上报、不写库**（施工单 §S7 / F7）。
   * 硬验收：开/关摄像头两次，落库的 `riskLevel` 完全一致。
   */
  const vision = useCallVision({ ageGroup: null });

  // ── 会话生命周期：没有 `:conversationId` 就建一个，成功后 replace 到带 id 的地址 ──
  //
  // 用 ref 缓存这个 promise：StrictMode 会把 effect 跑两遍，
  // 若直接在 effect 里 await 建会话，就会建出**两个**空会话 ——
  // 一个挂着当前通话、另一个永远没人用，却留在用户的对话历史列表里。
  useEffect(() => {
    if (conversationId) {
      // 已经有 id 了，缓存作废（也顺带避免"退回 /call 又被送回旧会话"）
      bootstrapRef.current = null;
      return;
    }
    if (!bootstrapRef.current) {
      setConversationError(null);
      bootstrapRef.current = consultationApi.createConversation().then((res) => {
        const id = (res as { data?: { id?: string } })?.data?.id;
        if (!id) throw new Error('服务端没有返回会话 id');
        return id;
      });
    }

    let alive = true;
    bootstrapRef.current
      .then((id) => {
        if (alive) navigate(`/call/${id}`, { replace: true });
      })
      .catch((err: unknown) => {
        if (!alive) return;
        bootstrapRef.current = null; // 允许重试
        setConversationError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      alive = false;
    };
  }, [conversationId, navigate, retryToken]);

  // 通话计时：从"接通"开始（`idle` 一旦离开就开始）
  const active = phase !== 'idle';
  useEffect(() => {
    if (!active) return;
    if (startedAtRef.current === 0) startedAtRef.current = Date.now();
    const id = window.setInterval(() => {
      setElapsed(Math.floor((Date.now() - startedAtRef.current) / 1000));
    }, 1000);
    return () => window.clearInterval(id);
  }, [active]);

  /**
   * 静音 = 关掉**共享的那一条**麦克风轨道。
   *
   * 通话页自己不持设备引用（ASR 的 `'perception-voice'` 就是那一份），
   * 所以这里按名字取当下正在用的流，而不是自己再 `acquire` 一次 ——
   * 同一份流被两个消费者各持一次会给释放带来歧义（施工单 §S4 取舍 5）。
   *
   * **静音不等于挂断**：会话与安全监测继续；只是没有音频就没有文本，
   * 此时不做任何风险推断。
   */
  const handleToggleMute = useCallback(() => {
    const stream = mediaStreamManager.getStream('microphone');
    if (!stream) return;
    const next = !muted;
    stream.getAudioTracks().forEach((track) => {
      track.enabled = !next;
    });
    setMuted(next);
  }, [muted]);

  /** 挂断：冲刷最后一段定稿 → 等本轮落库 → 回文字会话。 */
  const handleHangup = useCallback(async () => {
    if (hangingUp) return;
    setHangingUp(true);
    try {
      // 摄像头先关（它不参与落库，先还设备）
      try {
        vision.stop();
      } catch {
        /* 忽略 */
      }
      await call.endCall();
    } finally {
      // 用 replace：避免用户按浏览器后退又回到通话页（会重新申请麦克风）
      navigate(conversationId ? `/chat/${conversationId}` : '/chat', { replace: true });
    }
  }, [call, conversationId, hangingUp, navigate, vision]);

  const handleStartCall = useCallback(() => {
    if (conversationError) return;
    void call.startCall();
  }, [call, conversationError]);

  /**
   * 摄像头开关。
   *
   * **失败绝不能让通话中断**（`video_call_build_plan.md` §3.6）：
   * 拒权限 / 被抢走 / 设备忙 → 明确提示 + 语音通话照常，且**绝不显示"已开启"**。
   */
  const handleToggleCamera = useCallback(async () => {
    if (visionBusy) return;
    setVisionBusy(true);
    setVisionError(null);
    try {
      if (vision.enabled || vision.detecting) {
        vision.stop();
      } else {
        await vision.start();
      }
    } catch (err) {
      setVisionError(
        err instanceof Error ? `摄像头不可用：${err.message}（已改为纯语音通话）` : '摄像头不可用（已改为纯语音通话）',
      );
    } finally {
      setVisionBusy(false);
    }
  }, [vision, visionBusy]);

  const handleRetryConversation = useCallback(() => setRetryToken((t) => t + 1), []);

  const handleCrisisGoChat = useCallback(() => {
    void handleHangup();
  }, [handleHangup]);

  // "接通中"也要留着按钮（用 loading 态显示进度）—— 否则一点下去按钮就消失，
  // 用户会以为点漏了而反复点。
  const showStartButton =
    !conversationError && (phase === 'idle' || phase === 'connecting' || phase === 'error');

  const cameraOn = vision.enabled || vision.detecting;
  const showCrisisCard = phase === 'crisis' && !crisisCardDismissed;

  return (
    <div
      data-page="voice-call"
      style={{
        position: 'fixed',
        inset: 0,
        display: 'flex',
        flexDirection: 'column',
        padding: '18px 26px 22px',
        boxSizing: 'border-box',
        background: 'linear-gradient(165deg, #fdf4ff 0%, #eef2ff 46%, #e6f4ff 100%)',
        overflow: 'hidden',
      }}
    >
      <style>{`
        @keyframes call-breathe { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.045); } }
        @keyframes call-pulse {
          0%, 100% { transform: scale(1); box-shadow: 0 16px 38px rgba(129,140,248,0.38), 0 0 0 0 rgba(99,102,241,0.30); }
          50% { transform: scale(1.06); box-shadow: 0 16px 38px rgba(129,140,248,0.38), 0 0 0 20px rgba(99,102,241,0); }
        }
        @keyframes call-blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
      `}</style>

      {/* ── 顶栏 ─────────────────────────────────────────────── */}
      <header
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 16,
          flexShrink: 0,
        }}
      >
        <Button
          type="text"
          icon={<ArrowLeftOutlined />}
          loading={hangingUp}
          onClick={() => void handleHangup()}
          style={{ color: '#5a4a6a' }}
        >
          返回
        </Button>
        <Space size={8} wrap>
          {phase !== 'idle' && (
            <Tag
              style={{
                fontSize: 13,
                fontVariantNumeric: 'tabular-nums',
                background: '#fff',
                border: 'none',
                color: '#5a4a6a',
                padding: '4px 12px',
                borderRadius: 20,
              }}
            >
              {formatDuration(elapsed)}
            </Tag>
          )}
          <Tooltip title="语音识别与语音合成都在本机完成，通话音频不上传服务器">
            <Tag style={{ fontSize: 12, background: '#ffffffb0', border: 'none', color: '#7c7a99', borderRadius: 20 }}>
              本机识别 · 音频不出设备
            </Tag>
          </Tooltip>
          {cameraOn && (
            <Tooltip title="画面在本机浏览器内分析，只用于「你在不在画面里」和光线提示；不参与风险评估、不上报">
              <Tag
                data-tag="vision"
                style={{ fontSize: 12, background: '#f0fdf4', border: 'none', color: '#3f6212', borderRadius: 20 }}
              >
                画面本机分析 · 不进风险
              </Tag>
            </Tooltip>
          )}
        </Space>
      </header>

      {/* ── 提示条 ───────────────────────────────────────────── */}
      {(conversationError || notice || voiceMetrics.asrError || visionError || networkDown) && (
        <div style={{ maxWidth: 620, width: '100%', margin: '10px auto 0', flexShrink: 0 }}>
          {networkDown && (
            <Alert
              data-alert="network"
              type="warning"
              showIcon
              style={{ borderRadius: 12, marginBottom: 8 }}
              message="网络不稳定，正在重连…"
              description="已经说出来的内容不会丢，恢复后接着聊。"
            />
          )}
          {conversationError && (
            <Alert
              type="error"
              showIcon
              style={{ borderRadius: 12 }}
              message="无法建立通话会话"
              description={conversationError}
              action={
                <Button size="small" onClick={handleRetryConversation}>
                  重试
                </Button>
              }
            />
          )}
          {notice && (
            <Alert
              type={phase === 'crisis' ? 'warning' : 'info'}
              showIcon
              closable
              onClose={call.dismissNotice}
              style={{ borderRadius: 12, marginTop: conversationError ? 8 : 0 }}
              message={notice}
            />
          )}
          {voiceMetrics.asrError && (
            <Alert
              type="warning"
              showIcon
              style={{ borderRadius: 12, marginTop: 8 }}
              message={`语音识别不可用：${voiceMetrics.asrError}`}
            />
          )}
          {visionError && (
            <Alert
              type="warning"
              showIcon
              closable
              onClose={() => setVisionError(null)}
              style={{ borderRadius: 12, marginTop: 8 }}
              message={visionError}
            />
          )}
        </div>
      )}

      {/* ── 舞台：头像 + 状态 + 字幕 ──────────────────────────── */}
      <main
        style={{
          flex: 1,
          minHeight: 0,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 12,
          padding: '10px 0',
        }}
      >
        <div
          style={{
            width: 118,
            height: 118,
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'linear-gradient(140deg, #a5b4fc 0%, #818cf8 48%, #f0abfc 100%)',
            boxShadow: '0 16px 38px rgba(129,140,248,0.38)',
            animation: AVATAR_ANIMATION[phase],
            flexShrink: 0,
          }}
        >
          <PhoneFilled style={{ fontSize: 38, color: '#fff' }} />
        </div>

        <Title level={4} style={{ margin: 0, color: '#4a4a6a' }}>
          小安 · 心理陪伴
        </Title>

        <Space size={8} align="center">
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              background: PHASE_DOT[phase],
              display: 'inline-block',
              animation: phase === 'listening' ? 'call-blink 2s ease-in-out infinite' : undefined,
            }}
          />
          <Text data-phase={phase} style={{ fontSize: 15, color: '#5a4a6a' }}>
            {PHASE_LABEL[phase]}
          </Text>
        </Space>

        {showStartButton && (
          <Button
            type="primary"
            size="large"
            icon={<PhoneFilled />}
            loading={phase === 'connecting'}
            disabled={phase === 'connecting'}
            onClick={handleStartCall}
            style={{ marginTop: 6, borderRadius: 24, padding: '0 34px', height: 46 }}
          >
            {phase === 'error' ? '重试' : '开始通话'}
          </Button>
        )}

        {/* 字幕区：用户说的话 + AI 的回应 */}
        {(userLines.length > 0 || partialText || aiLines.length > 0) && (
          <div
            style={{
              width: 'min(720px, 100%)',
              maxHeight: '38vh',
              overflowY: 'auto',
              display: 'flex',
              flexDirection: 'column',
              gap: 8,
              padding: '4px 2px',
            }}
          >
            {userLines.map((line, index) => (
              <div
                key={`u-${index}-${line}`}
                data-caption="user"
                style={{
                  alignSelf: 'flex-end',
                  maxWidth: '86%',
                  padding: '9px 14px',
                  borderRadius: '14px 14px 4px 14px',
                  background: '#6366f1',
                  color: '#fff',
                  lineHeight: 1.65,
                  fontSize: 15,
                  whiteSpace: 'pre-wrap',
                }}
              >
                {line}
              </div>
            ))}

            {/* 中间结果：边说边出字。刻意用浅色 + 斜体，与"已定稿"明确区分 */}
            {partialText && (
              <div
                data-caption="partial"
                style={{
                  alignSelf: 'flex-end',
                  maxWidth: '86%',
                  padding: '9px 14px',
                  borderRadius: '14px 14px 4px 14px',
                  background: '#c7d2fe',
                  color: '#3730a3',
                  lineHeight: 1.65,
                  fontSize: 15,
                  fontStyle: 'italic',
                }}
              >
                {partialText}
              </div>
            )}

            {aiLines.map((line, index) => (
              <div
                key={`a-${index}-${line}`}
                data-caption="ai"
                style={{
                  alignSelf: 'flex-start',
                  maxWidth: '88%',
                  padding: '10px 14px',
                  borderRadius: '14px 14px 14px 4px',
                  background: '#fff',
                  boxShadow: '0 2px 10px rgba(99,102,241,0.10)',
                  color: '#3f3d56',
                  lineHeight: 1.7,
                  fontSize: 15,
                  whiteSpace: 'pre-wrap',
                }}
              >
                <Text type="secondary" style={{ fontSize: 11, display: 'block', marginBottom: 2 }}>
                  小安{revised ? '（安全审计已改稿）' : ''}
                </Text>
                {line}
              </div>
            ))}
          </div>
        )}

        {phase === 'listening' && userLines.length === 0 && !partialText && (
          <Text type="secondary" style={{ fontSize: 13, textAlign: 'center' }}>
            直接说话就行，你说的话会在这里逐字出现
          </Text>
        )}

        {call.aiSpeaking && (
          <Text data-hint="barge-in" type="secondary" style={{ fontSize: 12, textAlign: 'center' }}>
            随时可以打断 —— 直接开口说话，小安会立刻停下来听你说
          </Text>
        )}

        {call.forwardingPaused && (
          <Text data-hint="half-duplex-window" type="secondary" style={{ fontSize: 11, textAlign: 'center' }}>
            正在出声，先让 AEC 收敛 0.4 秒
          </Text>
        )}

        {phase === 'crisis' && (
          <Text style={{ fontSize: 13, color: '#a8600a', textAlign: 'center', maxWidth: 520 }}>
            这一轮不再用语音，避免听到不合适的内容。你也可以回到文字对话继续。
          </Text>
        )}
      </main>

      {/* ── 视觉小窗（只在开了摄像头时出现；**不进风险、不上报**）──────── */}
      {cameraOn && (
        <div
          data-panel="vision"
          style={{
            position: 'fixed',
            right: 22,
            bottom: 92,
            width: 176,
            borderRadius: 14,
            background: '#fff',
            boxShadow: '0 10px 30px rgba(99,102,241,0.20)',
            padding: 10,
            zIndex: 15,
          }}
        >
          <div
            ref={(el) => vision.attachPreview(el)}
            data-vision-preview
            style={{
              width: '100%',
              height: 118,
              borderRadius: 10,
              overflow: 'hidden',
              background: '#111827',
              marginBottom: 8,
            }}
          />
          <Space direction="vertical" size={2} style={{ width: '100%' }}>
            <Text data-vision="presence" style={{ fontSize: 12, color: '#3f3d56' }}>
              {presenceText(vision)}
            </Text>
            <Text data-vision="light" style={{ fontSize: 12, color: '#8c8aa7' }}>
              {lightText(vision.lightLevel)}
              {vision.brightness === null ? '' : `（${vision.brightness}/255）`}
            </Text>
            <Text data-vision="head" type="secondary" style={{ fontSize: 11, color: '#a6a3bb' }}>
              头部 {vision.headPitchDeg === null ? '—' : `${vision.headPitchDeg}°`} · 低头
              {Math.round(vision.headDownRatio * 100)}%
            </Text>
            <Text type="secondary" style={{ fontSize: 11, color: '#a6a3bb' }}>
              只给你自己看 · 不进风险
            </Text>
          </Space>
        </div>
      )}

      {/* ── 控制栏 ───────────────────────────────────────────── */}
      <footer
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 14,
          flexWrap: 'wrap',
          flexShrink: 0,
        }}
      >
        <Button
          size="large"
          shape="round"
          icon={muted ? <AudioMutedOutlined /> : <AudioOutlined />}
          type={muted ? 'primary' : 'default'}
          danger={muted}
          disabled={phase === 'idle' || phase === 'connecting' || phase === 'ended'}
          onClick={handleToggleMute}
          style={{ height: 46, padding: '0 22px' }}
        >
          {muted ? '取消静音' : '静音'}
        </Button>

        <Tooltip
          title={
            cameraOn
              ? '关闭摄像头：视觉读数停止，通话继续（视觉从不参与风险判定）'
              : '打开摄像头：只做"你在不在画面里"、光线提示和给你自己看的小读数'
          }
        >
          <Button
            size="large"
            shape="round"
            data-action="toggle-camera"
            icon={<VideoCameraOutlined />}
            type={cameraOn ? 'primary' : 'default'}
            loading={visionBusy}
            disabled={phase === 'idle' || phase === 'connecting' || phase === 'ended'}
            onClick={() => void handleToggleCamera()}
            style={{ height: 46, padding: '0 22px' }}
          >
            {cameraOn ? '关闭摄像头' : '摄像头'}
          </Button>
        </Tooltip>

        <Button
          size="large"
          shape="round"
          type="primary"
          danger
          loading={hangingUp}
          icon={<PhoneFilled style={{ transform: 'rotate(135deg)' }} />}
          onClick={() => void handleHangup()}
          style={{ height: 46, padding: '0 26px' }}
        >
          挂断
        </Button>
      </footer>

      {/* ── 危机卡片（S6）：复用文字页那份文案与号码，**电话号码只上屏、绝不朗读** ── */}
      {showCrisisCard && (
        <div
          data-overlay="crisis"
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(24,20,40,0.55)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: 20,
            zIndex: 40,
          }}
        >
          <Card style={{ width: 'min(440px, 100%)', borderRadius: 16, maxHeight: '92vh', overflowY: 'auto' }}>
            <CrisisResourceCard
              compact
              context="语音已暂停，我们改用文字继续陪着你。"
              onPrimary={handleCrisisGoChat}
              onDismiss={() => setCrisisCardDismissed(true)}
            />
          </Card>
        </div>
      )}

      {/* ── `?debug=1`：只显示**实测**读数，不是演示数据 ─────────────── */}
      {debugEnabled && (
        <Card
          size="small"
          title="诊断（实测读数）"
          style={{
            position: 'fixed',
            right: 20,
            bottom: 20,
            width: 340,
            zIndex: 20,
            borderRadius: 14,
            boxShadow: '0 10px 30px rgba(99,102,241,0.18)',
            maxHeight: '72vh',
            overflowY: 'auto',
          }}
        >
          <Space direction="vertical" size={4} style={{ width: '100%' }}>
            <DebugRow label="状态" value={phase} />
            <DebugRow
              label="ASR"
              value={
                voiceMetrics.asrEngine
                  ? `${voiceMetrics.asrEngine}${voiceMetrics.asrError ? ` · ${voiceMetrics.asrError}` : ''}`
                  : '未启动'
              }
            />
            <DebugRow
              label="TTS"
              value={
                ttsStatus
                  ? `${ttsStatus.engine ?? '-'} · ${ttsStatus.sampleRate}Hz · ${ttsStatus.numSpeakers}音色`
                  : '未连接'
              }
            />
            <DebugRow
              label="麦克风实际"
              value={
                call.micSettings
                  ? `AEC=${call.micSettings.echoCancellation ? '开' : '关'} NS=${
                      call.micSettings.noiseSuppression ? '开' : '关'
                    } AGC=${call.micSettings.autoGainControl ? '开' : '关'} ${call.micSettings.sampleRate}Hz`
                  : '未启动'
              }
            />
            <DebugRow label="送帧暂停" value={call.forwardingPaused ? '是（半双工窗口）' : '否'} />
            <DebugRow label="定稿句数" value={String(userLines.length)} />
            <DebugRow label="被丢弃定稿" value={String(call.discardedFinals)} />
            <DebugRow
              label="判为回声丢弃"
              value={`${call.droppedEchoes}${call.lastEchoReason ? ` · ${call.lastEchoReason}` : ''}`}
            />
            <DebugRow label="打断次数" value={String(call.bargeIns)} />
            <DebugRow label="并发轮次峰值" value={String(call.maxConcurrentTurns)} />
            <DebugRow label="跳过号码片段" value={String(call.skippedPhoneUnits)} />
            <DebugRow label="复述用户未登记" value={String(call.parrotSkips)} />
            <DebugRow label="纯标点未合成" value={String(call.skippedUnspeakableUnits)} />
            <DebugRow label="服务端判风险" value={call.lastRisk ?? '—'} />
            {turn ? (
              <>
                <DebugRow label="首段合成" value={turn.firstPcmReadyMs === null ? '—' : `${turn.firstPcmReadyMs} ms`} />
                <DebugRow label="首段出声" value={turn.firstSoundMs === null ? '—' : `${turn.firstSoundMs} ms`} />
                <DebugRow label="首个 delta" value={turn.firstDeltaMs === null ? '—' : `${turn.firstDeltaMs} ms`} />
                <DebugRow
                  label="插话→静音"
                  value={turn.interruptStopMs === null ? '—' : `${turn.interruptStopMs} ms`}
                />
                <DebugRow
                  label="插话时刻"
                  value={turn.interruptMs === null ? '—' : `本轮第 ${turn.interruptMs} ms`}
                />
                <DebugRow label="整轮" value={turn.doneMs === null ? '—' : `${turn.doneMs} ms`} />
                <DebugRow label="合成段数" value={String(turn.units)} />
                <DebugRow
                  label="首 delta 时通道"
                  value={
                    turn.ttsPresentAtFirstDelta === null
                      ? '—'
                      : `${turn.ttsPresentAtFirstDelta ? '已就绪' : '未就绪'} / 允许出声=${String(turn.voiceAllowedAtFirstDelta)}`
                  }
                />
                <DebugRow label="首 delta 原文" value={turn.firstDeltaText || '—'} />
                <DebugRow
                  label="meta.streaming"
                  value={turn.streaming === null ? '无 meta（非流式回退）' : String(turn.streaming)}
                />
                <DebugRow label="被改稿" value={String(turn.revised)} />
              </>
            ) : (
              <DebugRow label="轮次" value="尚未发生" />
            )}
            <DebugRow label="视觉" value={cameraOn ? '已开启' : '未开启'} />
            {cameraOn && (
              <>
                <DebugRow label="采样帧率" value={`${vision.fps} Hz`} />
                <DebugRow
                  label="画面新鲜"
                  value={vision.frameFresh ? '是' : '否（读数不计）'}
                />
                <DebugRow label="在画面里" value={vision.present ? '是' : '否'} />
                <DebugRow
                  label="亮度"
                  value={vision.brightness === null ? '—' : `${vision.brightness}/255`}
                />
                <DebugRow
                  label="头部俯仰"
                  value={vision.headPitchDeg === null ? '—' : `${vision.headPitchDeg}°`}
                />
                <DebugRow
                  label="低头占比"
                  value={`${Math.round(vision.headDownRatio * 100)}%（>25°且>3s）`}
                />
              </>
            )}
          </Space>
        </Card>
      )}
    </div>
  );
}
