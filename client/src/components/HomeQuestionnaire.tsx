import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Button, Card, Modal, Progress, Result, Space, Tag, Typography, message,
} from 'antd';
import {
  ArrowLeftOutlined, ArrowRightOutlined, CheckCircleFilled,
  HeartFilled, HeartOutlined, MoonOutlined, SmileOutlined, ThunderboltOutlined,
} from '@ant-design/icons';
import type { ReactNode } from 'react';
import { profileApi } from '../services';

const { Title, Text, Paragraph } = Typography;

/* ============================================================================
 * 20 题心理状态自查（跨维度情绪状态自查，约 5 分钟）
 *
 * ⚠️ 计分与后端 server/src/services/profile.service.ts 的 `case 'MOOD20'`
 *    **必须保持一致**：同一份答案，前端结果页与「心理画像」历史记录里的
 *    总分/等级不允许出现两种结论。改分档时两边一起改。
 *
 * ⚠️ 安全红线（见仓库 AGENTS.md）：
 *   · 题目与反馈文案一律非诊断性 —— 只说「自查结果 / 建议」，不说「你有 X 症」；
 *   · 不给任何药物名称、剂量或用药建议；
 *   · 第 16 题「不想活下去」命中时弹出关怀弹窗 + 24 小时热线，但**不阻断**作答。
 * ========================================================================== */

/** 题型：四选一（0-3 分） */
const OPTIONS: Array<{ value: number; label: string }> = [
  { value: 0, label: '完全没有' },
  { value: 1, label: '有几天' },
  { value: 2, label: '一半以上时间' },
  { value: 3, label: '几乎每天' },
];

/**
 * 计分区间（满分 60）。
 *
 * 分档按「近两周困扰持续时间」取宽档：12 分（平均每题 0.6 分）才算轻度，
 * 避免把偶发情绪误报成需要干预的状态 —— 误报会直接推高用户焦虑。
 * ⚠️ 与后端 `case 'MOOD20'` 同值同步。
 */
const SCORE_BANDS: Array<{ min: number; level: string; label: string; hint: string }> = [
  { min: 30, level: 'severe', label: '困扰较明显', hint: '近两周的困扰程度偏高。建议预约一次面对面咨询，让专业的人陪你一起梳理。' },
  { min: 20, level: 'moderate', label: '有些困扰', hint: '有不少困扰正在影响你。建议先做几次疗愈室练习，并考虑和公益咨询师聊一聊。' },
  { min: 12, level: 'mild', label: '轻微波动', hint: '最近状态有些起伏，多数人都会遇到。保持规律作息，试试疗愈室的呼吸练习。' },
  { min: 0, level: 'normal', label: '状态平稳', hint: '最近整体状态比较平稳，继续保持现在的作息和节奏就好。' },
];

const TOTAL_SCORE_MAX = 60;

type SectionKey = 'mood' | 'anxiety' | 'sleep' | 'stress' | 'social';

const SECTIONS: Record<SectionKey, { name: string; icon: ReactNode; color: string }> = {
  mood: { name: '情绪与兴趣', icon: <SmileOutlined />, color: '#6366f1' },
  anxiety: { name: '紧张与担忧', icon: <ThunderboltOutlined />, color: '#ec4899' },
  sleep: { name: '睡眠与精力', icon: <MoonOutlined />, color: '#fa8c16' },
  stress: { name: '压力与应对', icon: <HeartOutlined />, color: '#52c41a' },
  social: { name: '人际与支持', icon: <HeartFilled />, color: '#13c2c2' },
};

interface Question {
  /** 题干（近两周频率） */
  text: string;
  /** 所属维度 */
  section: SectionKey;
  /**
   * 选中非 0 分时需要弹出关怀提示（自伤相关条目）。
   * ⚠️ 危机相关逻辑：真实产品需接入完整危机干预流程，此处仅做可见提示 + 热线。
   */
  safety?: boolean;
}

const QUESTIONS: Question[] = [
  { text: '做事情提不起劲，也找不到兴趣', section: 'mood' },
  { text: '心情低落、提不起精神，觉得没什么盼头', section: 'mood' },
  { text: '容易自责，觉得自己不够好、让别人失望', section: 'mood' },
  { text: '就算遇到开心的事，也很难真正高兴起来', section: 'mood' },
  { text: '心里紧张、发慌，很难放松下来', section: 'anxiety' },
  { text: '担心的事情很多，停不下来', section: 'anxiety' },
  { text: '控制不住地往最坏的方向去想', section: 'anxiety' },
  { text: '身体紧绷：心跳快、手心出汗、坐不住', section: 'anxiety' },
  { text: '入睡困难，或者半夜容易醒来', section: 'sleep' },
  { text: '睡够了时间，白天还是没精神', section: 'sleep' },
  { text: '白天容易累，做事没什么力气', section: 'sleep' },
  { text: '作息不规律，白天晚上颠倒', section: 'sleep' },
  { text: '学习或考试带来的压力让我喘不过气', section: 'stress' },
  { text: '很难专心，注意力容易被拉走', section: 'stress' },
  { text: '压力大的时候，不知道该怎么让自己好受一点', section: 'stress' },
  { text: '会有「不想活下去」这类念头', section: 'stress', safety: true },
  { text: '不想和同学、朋友说话，只想一个人待着', section: 'social' },
  { text: '在班里或家里，觉得自己是孤单的', section: 'social' },
  { text: '和父母沟通让我感到疲惫或委屈', section: 'social' },
  { text: '遇到困难时，身边没有可以说心里话的人', section: 'social' },
];

const TOTAL_QUESTIONS = QUESTIONS.length;

const SAFETY_HOTLINES = [
  { name: '全国 24 小时心理援助热线', number: '400-161-9995' },
  { name: '生命热线', number: '400-821-1215' },
];

/** 取分档（入参必须是已确定的总分） */
function bandOf(totalScore: number) {
  return SCORE_BANDS.find((b) => totalScore >= b.min) || SCORE_BANDS[SCORE_BANDS.length - 1];
}

/**
 * 把答案整理成规范形态：**恰好 20 个键（0..19）**、每题 0-3 的整数。
 * 提交前统一走这里，保证发给后端的永远是完整结构（缺答按 0 分语义补齐），
 * 避免"看着答完了、服务端却收到空/缺键"的情况。
 */
function normAnswers(src: Record<number, number>): Record<number, number> {
  const out: Record<number, number> = {};
  for (let i = 0; i < TOTAL_QUESTIONS; i++) {
    const raw = Number(src?.[i]);
    out[i] = Number.isFinite(raw) ? Math.max(0, Math.min(3, Math.round(raw))) : 0;
  }
  return out;
}

/** 4 级选项按钮：点击即选中，选中态走站内统一的「浅粉底 + 粉边 + 粉字」 */
function OptionButton({
  label, value, selected, accent, onClick,
}: { label: string; value: number; selected: boolean; accent: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={selected}
      style={{
        cursor: 'pointer',
        textAlign: 'left',
        padding: '12px 14px',
        borderRadius: 14,
        border: selected ? '2px solid #ff8fab' : '2px solid rgba(255,182,193,0.35)',
        background: selected ? '#fff0f3' : '#fffdfe',
        color: selected ? '#e2578a' : '#5a4a6a',
        fontWeight: selected ? 600 : 500,
        fontSize: 14,
        lineHeight: 1.4,
        boxShadow: selected ? '0 4px 14px rgba(255,143,171,0.22)' : 'none',
        transition: 'all 0.2s cubic-bezier(0.34, 1.56, 0.64, 1)',
      }}
      onMouseEnter={(e) => { if (!selected) e.currentTarget.style.transform = 'translateY(-2px)'; }}
      onMouseLeave={(e) => { e.currentTarget.style.transform = 'translateY(0)'; }}
    >
      <span style={{ color: selected ? accent : '#b0a0c0', marginRight: 6, fontSize: 12 }}>{value}</span>
      {label}
    </button>
  );
}

export default function HomeQuestionnaire() {
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [current, setCurrent] = useState(0);
  const [answers, setAnswers] = useState<Record<number, number>>({});
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<{ totalScore: number; level: string; suggestion: string; computedLocally?: boolean } | null>(null);
  /** 已答完的近一次结果（首页摘要用） */
  const [lastResult, setLastResult] = useState<{ totalScore: number; level: string; date: string } | null>(null);
  const [safetyOpen, setSafetyOpen] = useState(false);
  const safetyShownRef = useRef(false);

  const answeredCount = Object.keys(answers).length;
  const isLast = current === TOTAL_QUESTIONS - 1;
  const activeQuestion = QUESTIONS[current];
  const activeSection = SECTIONS[activeQuestion.section];

  /**
   * 拉最近一次 MOOD20 测评，用于首页摘要。
   * 后端按 createdAt 倒序返回，取第一条 MOOD20 即可；失败就保持"未测过"，
   * 不编造分数（与心理画像页的口径一致：拿不到数据就不显示结论）。
   */
  const loadLastResult = useCallback(async () => {
    try {
      const res: any = await profileApi.getAssessments();
      const list: any[] = Array.isArray(res?.data) ? res.data : [];
      const latest = list.find((a) => a?.type === 'MOOD20');
      if (!latest) return;
      const score = Number(latest.totalScore ?? 0);
      setLastResult({
        totalScore: score,
        level: bandOf(score).label,
        date: new Date(latest.createdAt).toLocaleDateString('zh-CN'),
      });
    } catch {
      // 静默：摘要只是锦上添花，接口失败不应干扰首页其它内容
    }
  }, []);

  useEffect(() => { loadLastResult(); }, [loadLastResult]);

  // 打开时回到上次进度（未提交过则从第 1 题开始）
  const handleOpen = () => {
    if (result) {
      // 已看过结果：再点一次表示「重新测一次」
      setResult(null);
      setAnswers({});
      setCurrent(0);
    }
    setOpen(true);
  };

  const handleAnswer = (value: number) => {
    setAnswers((prev) => ({ ...prev, [current]: value }));
    // ⚠️ 自伤条目选中非 0 分：先给关怀，不阻断作答（每次进入问卷只弹一次）
    if (activeQuestion.safety && value > 0 && !safetyShownRef.current) {
      safetyShownRef.current = true;
      setSafetyOpen(true);
    }
  };

  const handleNext = () => {
    if (answers[current] === undefined) {
      message.warning('先选一个最接近的答案再往下~');
      return;
    }
    if (!isLast) setCurrent((c) => Math.min(c + 1, TOTAL_QUESTIONS - 1));
  };

  const handleSubmit = async () => {
    if (answeredCount < TOTAL_QUESTIONS) {
      message.warning(`还有 ${TOTAL_QUESTIONS - answeredCount} 题没答完`);
      return;
    }
    setSubmitting(true);
    try {
      // 提交时先收起"关怀弹窗"：它与结果弹窗同为 Modal，叠在一起会把结果盖住，
      // 用户会以为"答完了没结果"。危机资源改在结果页里直接展示（见下方"现在就有人可以聊"）。
      setSafetyOpen(false);
      // 提交前规范化：固定 20 个键，缺答按 0 分补齐（与服务端一致）
      const payloadAnswers = normAnswers(answers);
      const localTotal = Object.values(payloadAnswers).reduce((a, b) => a + b, 0);
      const res: any = await profileApi.submitAssessment({
        type: 'MOOD20',
        title: '近两周心理状态自查（20 题）',
        answers: payloadAnswers,
      });
      const data = res?.data ?? res;
      const serverTotal = Number(data?.totalScore);
      const hasServerTotal = Number.isFinite(serverTotal);
      // ⚠️ 结果页不能"答完了却显示 0 分"：后端没给分就用本地和兜底（并在页面上标注来源）
      const shownTotal = hasServerTotal ? serverTotal : localTotal;
      const shownBand = bandOf(shownTotal);
      setResult({
        totalScore: shownTotal,
        level: String(data?.level ?? shownBand.level),
        suggestion: String(data?.suggestion || '') || shownBand.hint,
        computedLocally: !hasServerTotal,
      });
      setLastResult({
        totalScore: shownTotal,
        level: bandOf(shownTotal).label,
        date: new Date().toLocaleDateString('zh-CN'),
      });
      loadLastResult();
      // 结果落库后回到首页时，摘要与「心理画像」的历史记录都应同步
    } catch (e: any) {
      // 不静默：提交失败必须让用户看见，否则会以为"已经记下了"
      message.error(e?.message || '提交失败，请稍后重试');
    } finally {
      setSubmitting(false);
    }
  };

  const handleClose = () => {
    setOpen(false);
    safetyShownRef.current = false;
  };

  const handleRestart = () => {
    setResult(null);
    setAnswers({});
    setCurrent(0);
    safetyShownRef.current = false;
  };

  /** 维度小分：结果页用来告诉用户"哪一块拖后腿"，比只给总分更有用 */
  const sectionScores = (() => {
    const acc: Record<string, { score: number; max: number }> = {};
    QUESTIONS.forEach((q, i) => {
      if (!acc[q.section]) acc[q.section] = { score: 0, max: 0 };
      acc[q.section].score += answers[i] || 0;
      acc[q.section].max += 3;
    });
    return acc;
  })();

  /**
   * 展示用总分：以后端返回为准（那是落库、也是心理画像采用的那一份）。
   * ⚠️ 只有在后端**没给**分数时才回退到本地计算，并明确标注"本地计算"，
   * 避免出现"答完了却显示总分 0"这种看起来像没生效的结果。
   * 若两边都有值但不一致，以服务端为准（落库数据才是画像用的），并在控制台留痕便于排查。
   */
  const finalScore = (() => {
    if (!result) return null;
    const localTotal = Object.values(normAnswers(answers)).reduce((a, b) => a + b, 0);
    const serverTotal = Number(result.totalScore);
    const serverValid = Number.isFinite(serverTotal);
    if (serverValid && serverTotal !== localTotal) {
      // 不是错误路径，但值得留痕：出现即说明前后端计分不一致
      console.warn('[问卷] 服务端总分与本地计算不一致，以服务端为准', { serverTotal, localTotal });
    }
    return {
      total: serverValid ? serverTotal : localTotal,
      source: serverValid ? 'server' : 'local',
    };
  })();

  // 分档必须跟着**展示用的总分**走，否则兜底路径下等级和分数会对不上
  const band = finalScore ? bandOf(finalScore.total) : null;

  /** 结果页的逐题作答明细：让用户能核对自己选了哪一项 */
  const answeredRows = QUESTIONS.map((q, i) => {
    const value = answers[i];
    const section = SECTIONS[q.section];
    return {
      index: i,
      text: q.text,
      sectionName: section.name,
      sectionColor: section.color,
      label: value === undefined ? '未作答' : (OPTIONS.find((o) => o.value === value)?.label || '—'),
      value: typeof value === 'number' ? value : null,
    };
  });

  /** 自伤相关条目选了非 0 分：结果页必须继续给到热线，而不是只弹一次就没了 */
  const safetyFlag = QUESTIONS.some((q, i) => q.safety && typeof answers[i] === 'number' && (answers[i] as number) > 0);

  useEffect(() => {
    // 结果页/首页摘要用得到：避免长问卷刷屏时把用户弹窗状态带出去
    if (!open) setSafetyOpen(false);
  }, [open]);

  return (
    <>
      {/* 组件内动画：只在本题干卡片上用，不污染全局样式 */}
      <style>{`
        @keyframes homeQ20FadeIn {
          from { opacity: 0; transform: translateX(12px); }
          to   { opacity: 1; transform: translateX(0); }
        }
        .home-q20-fade { animation: homeQ20FadeIn 0.28s ease-out; }
        /* 站内主题给所有 primary 按钮加了浅粉渐变 !important，白字压在上面只有约 1.4:1，
           行内样式盖不住，只能用带 !important 的类把底色压深 */
        .home-q20-cta.ant-btn-primary {
          background: #e2578a !important;
          border-color: #e2578a !important;
          color: #fff !important;
        }
        .home-q20-cta.ant-btn-primary:hover {
          background: #d24a7c !important;
          border-color: #d24a7c !important;
        }
        @media (max-width: 767px) {
          .home-q20-actions { align-items: flex-start !important; width: 100%; }
          .home-q20-banner { padding: 16px !important; }
        }
      `}</style>

      {/* ===== 首页板块：与新版首页设计系统同一套卡片语言 =====
           这里刻意不用 antd Card：旧主题把 `.ant-card` 全局压成了粉色云朵渐变，
           在首页里会和 `styles/garden.css` 的玻璃卡片打架。 */}
      <div className="gh-card gh-q20">
        <div className="gh-q20__main">
          <span className="gh-q20__icon" aria-hidden="true">📋</span>
          <div className="gh-q20__text">
            <h3 className="gh-q20__title">心理状态自查 · 20 题</h3>
            <p className="gh-q20__sub">约 5 分钟，覆盖情绪、紧张、睡眠、压力与支持五个方面</p>
            {lastResult ? (
              <p className="gh-q20__meta gh-q20__meta--done">
                <CheckCircleFilled style={{ color: '#3f9e74' }} /> {lastResult.date} 已完成 · {lastResult.totalScore}/{TOTAL_SCORE_MAX} 分 · {lastResult.level}
              </p>
            ) : (
              <p className="gh-q20__meta">结果仅供自我参考，不构成医学诊断</p>
            )}
          </div>
        </div>

        <div className="gh-q20__actions home-q20-actions">
          <button type="button" className="gh-btn gh-btn--primary" onClick={handleOpen}>
            {result || lastResult ? '再测一次' : '开始自查'}
            <ArrowRightOutlined />
          </button>
          <button type="button" className="gh-btn gh-btn--ghost" onClick={() => navigate('/profile')}>
            心理画像
          </button>
        </div>
      </div>

      {/* ===== 作答弹窗 ===== */}
      <Modal
        open={open}
        onCancel={handleClose}
        footer={null}
        width={640}
        centered
        destroyOnHidden
        title={null}
        styles={{ body: { padding: 0 } }}
      >
        {result && band && finalScore ? (
          /* ---------- 结果页 ---------- */
          <div style={{ padding: '8px 4px 4px', maxHeight: '76vh', overflowY: 'auto' }}>
            <Result
              status={band.level === 'normal' ? 'success' : band.level === 'mild' ? 'warning' : 'error'}
              title="自查完成"
              subTitle={`总分 ${finalScore.total} / ${TOTAL_SCORE_MAX} · ${band.label}`}
              extra={finalScore.source === 'local' ? (
                <Tag color="orange" style={{ borderRadius: 10 }}>
                  分数由本机按你的选择算出（服务端未返回分数）
                </Tag>
              ) : null}
            />
            <div style={{ padding: '0 16px 8px' }}>
              <div style={{
                padding: '12px 16px', borderRadius: 12, marginBottom: 16,
                background: '#fff7fb', border: '1px solid rgba(255,182,193,0.4)',
                fontSize: 14, lineHeight: 1.8, color: '#5a4a6a',
              }}>
                {result.suggestion || band.hint}
              </div>

              {/* ⚠️ 第 16 题（自伤念头）选了非 0 分：结果页继续给热线，并可直接找 AI 聊 */}
              {safetyFlag && (
                <div style={{
                  padding: '12px 14px', borderRadius: 12, marginBottom: 16,
                  background: '#fff5f5', border: '1px solid #ffccc7',
                }}>
                  <Text strong style={{ fontSize: 13, color: '#cf1322', display: 'block', marginBottom: 6 }}>
                    💛 现在就有人可以聊
                  </Text>
                  <Text style={{ fontSize: 12, color: '#8a7a9a', display: 'block', marginBottom: 8 }}>
                    你在第 16 题选了「{OPTIONS.find((o) => o.value === answers[15])?.label || '有几天'}」。
                    这样的念头往往说明你已经扛了很久，不是你的错，也不用一个人面对。
                  </Text>
                  {SAFETY_HOTLINES.map((h) => (
                    <div key={h.number} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13 }}>
                      <Text style={{ color: '#5a4a6a' }}>{h.name}</Text>
                      <Text strong style={{ color: '#ff4d4f' }}>{h.number}</Text>
                    </div>
                  ))}
                  <Button
                    size="small" danger
                    style={{ marginTop: 10, borderRadius: 12 }}
                    onClick={() => { handleClose(); navigate('/chat'); }}
                  >
                    <HeartOutlined /> 去找 AI 聊聊
                  </Button>
                </div>
              )}

              {/* 维度小分 */}
              <Text strong style={{ fontSize: 13, color: '#8a7a9a' }}>各方面情况</Text>
              <div style={{ marginTop: 8, marginBottom: 16 }}>
                {(Object.keys(SECTIONS) as SectionKey[]).map((key) => {
                  const s = sectionScores[key];
                  if (!s) return null;
                  const percent = Math.round((s.score / s.max) * 100);
                  return (
                    <div key={key} style={{ marginBottom: 8 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                        <Text style={{ color: SECTIONS[key].color }}>
                          {SECTIONS[key].icon} {SECTIONS[key].name}
                        </Text>
                        <Text type="secondary">{s.score}/{s.max}</Text>
                      </div>
                      <Progress
                        percent={percent}
                        showInfo={false}
                        size="small"
                        strokeColor={SECTIONS[key].color}
                      />
                    </div>
                  );
                })}
              </div>

              {/* 逐题作答明细：让用户能核对"我每题选了什么、怎么加出这个分的" */}
              <details style={{ marginBottom: 16 }}>
                <summary style={{
                  cursor: 'pointer', fontSize: 13, fontWeight: 600,
                  color: '#6366f1', padding: '6px 0', listStyle: 'none',
                }}>
                  查看 20 题作答明细 ▾
                </summary>
                <div style={{
                  marginTop: 8, border: '1px solid rgba(255,182,193,0.35)',
                  borderRadius: 10, overflow: 'hidden',
                }}>
                  {answeredRows.map((q, i) => (
                    <div
                      key={q.index}
                      style={{
                        display: 'flex', gap: 8, alignItems: 'flex-start', justifyContent: 'space-between',
                        padding: '8px 10px', fontSize: 12, lineHeight: 1.6,
                        background: i % 2 === 0 ? '#fffdfe' : '#fff8fa',
                      }}
                    >
                      <div style={{ minWidth: 0 }}>
                        <span style={{ color: '#b0a0c0', marginRight: 4 }}>{q.index + 1}.</span>
                        <span style={{ color: '#5a4a6a' }}>{q.text}</span>
                        <span style={{ color: q.sectionColor, marginLeft: 6, fontSize: 11 }}>
                          [{q.sectionName}]
                        </span>
                      </div>
                      <div style={{
                        flex: '0 0 auto', color: q.value ? '#e2578a' : '#b0a0c0',
                        fontWeight: q.value ? 600 : 400, whiteSpace: 'nowrap',
                      }}>
                        {q.label}{q.value ? ` · ${q.value}分` : ''}
                      </div>
                    </div>
                  ))}
                  <div style={{
                    padding: '8px 10px', fontSize: 12, background: '#f6ffed',
                    color: '#389e0d', borderTop: '1px solid rgba(255,182,193,0.35)',
                  }}>
                    合计 {finalScore.total} 分（满分 {TOTAL_SCORE_MAX}），已保存并会参与你的心理画像。
                  </div>
                </div>
              </details>

              <div style={{
                padding: '10px 14px', borderRadius: 10, marginBottom: 16,
                background: '#fffbe6', border: '1px solid #ffe58f', fontSize: 12, color: '#ad6800',
              }}>
                本结果仅供自我参考，<b>不构成医学诊断</b>。若负面感受持续两周以上，或影响到吃饭、睡觉、上学，
                建议让家长、老师知道，并预约一次公益咨询。
              </div>

              <Space direction="vertical" style={{ width: '100%' }} size="small">
                <Button
                  type="primary" block
                  onClick={() => { handleClose(); navigate('/experts'); }}
                  style={{ borderRadius: 12 }}
                >
                  预约公益咨询师
                </Button>
                <Button block onClick={() => { handleClose(); navigate('/healing'); }} style={{ borderRadius: 12 }}>
                  去疗愈空间放松一下
                </Button>
                <Button type="link" block onClick={handleRestart}>重新测一次</Button>
              </Space>
              <Text type="secondary" style={{ fontSize: 11, display: 'block', marginTop: 8, textAlign: 'center' }}>
                结果已保存到「心理画像 → 标准化量表测评 → 历史记录」
              </Text>
            </div>
          </div>
        ) : (
          /* ---------- 作答页 ---------- */
          <div style={{ padding: '20px 20px 16px' }}>
            {/* 顶部：进度 + 维度 */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
              <Space size={6}>
                <Tag color="magenta" style={{ borderRadius: 10, fontSize: 12 }}>
                  {activeSection.icon} {activeSection.name}
                </Tag>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  第 {current + 1} / {TOTAL_QUESTIONS} 题
                </Text>
              </Space>
              <Text type="secondary" style={{ fontSize: 12 }}>已答 {answeredCount} 题</Text>
            </div>
            <Progress
              percent={Math.round((answeredCount / TOTAL_QUESTIONS) * 100)}
              showInfo={false}
              strokeColor={{ from: '#6366f1', to: '#ff8fab' }}
              style={{ marginBottom: 18 }}
            />

            {/* 题干 */}
            <div key={current} className="home-q20-fade" style={{ minHeight: 132 }}>
              <Text style={{ fontSize: 12, color: '#b0a0c0', display: 'block', marginBottom: 6 }}>
                近两周，下面这种情况出现的频率是？
              </Text>
              <Title level={4} style={{ fontSize: 19, lineHeight: 1.6, color: '#4a3a5a', margin: 0 }}>
                {current + 1}. {activeQuestion.text}
              </Title>

              <div style={{
                display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
                gap: 10, marginTop: 18,
              }}>
                {OPTIONS.map((o) => (
                  <OptionButton
                    key={o.value}
                    label={o.label}
                    value={o.value}
                    selected={answers[current] === o.value}
                    accent={activeSection.color}
                    onClick={() => handleAnswer(o.value)}
                  />
                ))}
              </div>
            </div>

            {/* 底部导航 */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: 16 }}>
              <Button
                icon={<ArrowLeftOutlined />}
                disabled={current === 0}
                onClick={() => setCurrent((c) => Math.max(0, c - 1))}
                style={{ borderRadius: 12 }}
              >
                上一题
              </Button>
              {isLast ? (
                <Button
                  type="primary"
                  size="large"
                  loading={submitting}
                  disabled={answeredCount < TOTAL_QUESTIONS}
                  onClick={handleSubmit}
                  style={{ borderRadius: 12, minWidth: 140 }}
                >
                  {answeredCount < TOTAL_QUESTIONS ? `还有 ${TOTAL_QUESTIONS - answeredCount} 题` : '提交自查'}
                </Button>
              ) : (
                <Button
                  type="primary"
                  onClick={handleNext}
                  disabled={answers[current] === undefined}
                  style={{ borderRadius: 12, minWidth: 120 }}
                >
                  下一题 <ArrowRightOutlined />
                </Button>
              )}
            </div>

            <Text type="secondary" style={{ fontSize: 11, display: 'block', marginTop: 12, textAlign: 'center' }}>
              你的答案会加密保存，仅用于生成你自己的状态参考；可随时要求删除。
            </Text>
          </div>
        )}
      </Modal>

      {/* ⚠️ 自伤条目关怀弹窗：仅提示 + 热线，不阻断作答 */}
      <Modal
        open={safetyOpen}
        onCancel={() => setSafetyOpen(false)}
        footer={null}
        width={480}
        centered
        title={null}
      >
        <div style={{ textAlign: 'center', padding: '16px 0' }}>
          <div style={{ fontSize: 44, marginBottom: 12 }}>💛</div>
          <Title level={4} style={{ marginBottom: 8, color: '#5a4a6a' }}>谢谢你愿意说出来</Title>
          <Paragraph style={{ fontSize: 14, color: '#8a7a9a' }}>
            看到你选了这一项，我们有点担心你现在的感受。这样的念头常常说明你已经扛了很久，
            它不是你的错，也不是你一个人要面对的。
          </Paragraph>
          <Card style={{ borderRadius: 12, marginBottom: 16, background: '#fff5f5', border: '1px solid #ffccc7' }}>
            {SAFETY_HOTLINES.map((h) => (
              <div key={h.number} style={{ marginBottom: 8 }}>
                <Text strong>{h.name}</Text>
                <Title level={3} style={{ color: '#ff4d4f', margin: '4px 0' }}>{h.number}</Title>
              </div>
            ))}
          </Card>
          <Space direction="vertical" style={{ width: '100%' }} size="small">
            <Button
              type="primary" block size="large"
              onClick={() => { setSafetyOpen(false); handleClose(); navigate('/chat'); }}
              style={{ borderRadius: 12 }}
            >
              <HeartOutlined /> 去找 AI 聊聊
            </Button>
            <Button block size="large" onClick={() => setSafetyOpen(false)} style={{ borderRadius: 12 }}>
              我暂时还好，继续作答
            </Button>
          </Space>
        </div>
      </Modal>
    </>
  );
}
