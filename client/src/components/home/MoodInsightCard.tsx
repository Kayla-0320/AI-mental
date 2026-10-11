/**
 * 心理状态与数据洞察卡（首页 · 纯展示组件）
 *
 * 数据来源：`GrowthOverview`（GET /api/extra/growth），见 types/garden.ts。
 * 本组件**只做展示**：不发请求、不派生任何心理结论。
 *
 * ⚠️ 安全红线（产品硬约束）：
 *   只用后端已经算好的 `counters` / `trend` 字段说话。前端**不推断**情绪好坏、
 *   不下任何"焦虑/抑郁"之类的判断；样本不够时明确说"记录还不够"，
 *   而不是用 1 个点画一条看起来很确定的曲线。
 *
 * ⚠️ 为什么不用 antd 的 Card/Button：
 *   styles/healing-theme.css 里有 `.ant-card { background: 粉色渐变 !important }`
 *   一类全局规则（旧页面仍在依赖，不能删），用 antd 组件要跟它打 !important 的架。
 *   新首页统一改用 garden.css 的 `.gh-*` 基础类 + 普通 HTML 元素。
 */

import { useId } from 'react';
import type { GrowthOverview, MoodTrendPoint } from '../../types/garden';
import './mood-insight.css';

/** 少于这个打卡天数就不画曲线，改走友好空态 */
const MIN_SCORED_DAYS = 3;

/** 手绘 SVG 曲线的内边距与刻度范围（score 固定 1-10） */
const PAD_L = 12;
const PAD_R = 12;
const PAD_T = 14;
const PAD_B = 10;
const SCALE_MIN = 1;
const SCALE_MAX = 10;

interface Pt {
  x: number;
  y: number;
}

interface Sparkline {
  /** viewBox 宽度（与点数成比例，让桌面/移动端点位间距都舒展） */
  width: number;
  height: number;
  /** 连续有分的点连成的线段集合；null 的日子天然断开，不连线也不当 0 */
  segments: Pt[][];
  /** 所有真实点（用于画圆点、判断"今天"） */
  dots: Pt[];
  /** 所有真实点的原始坐标 */
  view: Pt;
}

/**
 * 把 trend 序列转成折线几何。
 *
 * 关键点：`score === null` 的日子必须表现为**断口**——既不能连成直线穿过它，
 * 也不能当成 0 分去画（0 分会被读成"那天心情极差"，是纯粹的臆造）。
 */
function buildSparkline(trend: MoodTrendPoint[], viewW: number, viewH: number): Sparkline {
  const innerW = viewW - PAD_L - PAD_R;
  const innerH = viewH - PAD_T - PAD_B;
  const n = trend.length;
  // n === 1 时没有"间距"概念，直接放中间，避免除零
  const step = n > 1 ? innerW / (n - 1) : 0;
  const point = (index: number, score: number): Pt => ({
    x: n > 1 ? PAD_L + index * step : PAD_L + innerW / 2,
    y: PAD_T + (1 - (score - SCALE_MIN) / (SCALE_MAX - SCALE_MIN)) * innerH,
  });

  const segments: Pt[][] = [];
  const dots: Pt[] = [];
  let run: Pt[] = [];

  trend.forEach((day, i) => {
    if (typeof day.score === 'number') {
      const p = point(i, day.score);
      dots.push(p);
      run.push(p);
    } else if (run.length > 0) {
      // 遇到空档：把之前累积的连续段收尾
      segments.push(run);
      run = [];
    }
  });
  if (run.length > 0) segments.push(run);

  return { width: viewW, height: viewH, segments, dots, view: { x: viewW, y: viewH } };
}

/** 折线路径（直线段：心情分数本身是日均值，用平滑曲线反而会暗示不存在的中间值） */
function linePath(seg: Pt[]): string {
  return seg.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ');
}

/** 面积路径：沿折线走一遍，再贴底边闭合 */
function areaPath(seg: Pt[], bottom: number): string {
  if (seg.length < 2) return '';
  const first = seg[0];
  const last = seg[seg.length - 1];
  return `${linePath(seg)} L${last.x.toFixed(1)} ${bottom} L${first.x.toFixed(1)} ${bottom} Z`;
}

/** 统一保留一位小数（后端 avgMood7 / moodDelta 也是一位小数） */
function fmt1(n: number): string {
  return n.toFixed(1);
}

/** 情绪稳定度分档：三档都对应"能看见的状态"，避开"差/糟糕"这类评判词 */
function stabilityTier(v: number): { cls: string; word: string } {
  if (v >= 70) return { cls: 'is-high', word: '比较平稳' };
  if (v >= 40) return { cls: 'is-mid', word: '有些起伏' };
  return { cls: 'is-low', word: '起伏较多' };
}

/** 较上周：向上/向下都用中性温度的说法，不做"变好/变差"的判断 */
function deltaMeta(v: number): { cls: string; arrow: string; text: string } {
  if (v > 0) return { cls: 'is-up', arrow: '↑', text: `比上周高 ${fmt1(v)} 分` };
  if (v < 0) return { cls: 'is-down', arrow: '↓', text: `比上周低 ${fmt1(Math.abs(v))} 分` };
  return { cls: 'is-flat', arrow: '→', text: '和上周差不多' };
}

export default function MoodInsightCard({
  data,
  loading,
  onOpenProfile,
}: {
  data: GrowthOverview | null;
  loading: boolean;
  onOpenProfile: () => void;
}) {
  // SVG 里的渐变 id 必须全局唯一：同一页面渲染两张卡时 id 冲突会让填充引用到错误节点
  const uid = useId().replace(/:/g, '');
  const gradId = `gh-insight-grad-${uid}`;

  // 后端偶尔可能多给几天，这里只取最后 7 天
  const trend: MoodTrendPoint[] = (data?.trend ?? []).slice(-7);
  // 空态判定的"打卡天数"由前端自算：counters 里的窗口口径与 trend 未必一致
  const scoredCount = trend.filter((d) => typeof d.score === 'number').length;
  const hasEnough = scoredCount >= MIN_SCORED_DAYS;

  const counters = data?.counters ?? null;
  const keywords = data?.keywords ?? [];

  const busy = loading || !data;

  const chart = buildSparkline(trend, Math.max(240, trend.length * 56), 104);
  const baseline = chart.height - PAD_B;

  const ariaPoints = trend
    .filter((d) => typeof d.score === 'number')
    .map((d) => `${d.label} 为 ${fmt1(d.score as number)} 分`);
  const ariaLabel =
    ariaPoints.length > 0 ? `近 7 天心情曲线，${ariaPoints.join('，')}` : '近 7 天心情曲线，暂无记录';

  // 末点 = 今天：加粗圆点并带出当天的心情词
  const lastScoredIndex = (() => {
    for (let i = trend.length - 1; i >= 0; i -= 1) {
      if (typeof trend[i].score === 'number') return i;
    }
    return -1;
  })();

  // 最外层是普通 div 而不是 `.gh-sec`：这张卡现在和「20 题自查」并排在一行两栏里
  // （见 Home.tsx 的 `.gh-sec.gh-cols2`）。`.gh-sec` 带 max-width 与左右内边距，
  // 套在半栏里会让卡片再缩一半、并叠出多余留白，宽度统一交给 grid。
  return (
    <div className="gh-insight" aria-labelledby={`gh-insight-title-${uid}`}>
      <div className="gh-card gh-card--sky gh-insight__box gh-reveal" style={{ '--gh-delay': '60ms' } as React.CSSProperties}>
        <header className="gh-insight__head">
          <div>
            <h2 className="gh-insight__title" id={`gh-insight-title-${uid}`}>
              心理状态
            </h2>
            <p className="gh-insight__sub">近 7 天的记录，慢慢就能看见自己的变化</p>
          </div>
          <button type="button" className="gh-btn gh-btn--ghost gh-btn--sm gh-insight__profile" onClick={onOpenProfile}>
            心理画像 <span aria-hidden="true">→</span>
          </button>
        </header>

        {busy ? (
          /* 加载中：保持与真实卡片一致的骨架，避免数据到达时整块跳动 */
          <div className="gh-insight__body" aria-busy="true">
            <div className="gh-skeleton gh-insight__sk-chart" />
            <div className="gh-insight__cols">
              {[0, 1, 2].map((i) => (
                <div className="gh-insight__sk-tile" key={i}>
                  <span className="gh-skeleton gh-insight__sk-line gh-insight__sk-line--sm" />
                  <span className="gh-skeleton gh-insight__sk-line gh-insight__sk-line--lg" />
                </div>
              ))}
            </div>
          </div>
        ) : !hasEnough ? (
          /* 记录不足：给一句说明 + 一句鼓励，而不是画一条只有一两个点的"曲线" */
          <div className="gh-insight__empty">
            <div className="gh-empty gh-insight__empty-box">
              <p className="gh-insight__empty-main">再记录几天，就能看到属于你的变化</p>
              <p className="gh-insight__empty-sub">
                {/* 明确写「近 7 天」：成长卡上的「累计打卡 N 天」是全量口径，
                    两处数字不一样时，标清统计范围才不会让人以为是数据错了 */}
                近 7 天已记录 {scoredCount} 天，攒够 {MIN_SCORED_DAYS} 天就能画出你的心情曲线啦。
              </p>
            </div>
          </div>
        ) : (
          <div className="gh-insight__body">
            <div className="gh-insight__chart">
              <svg
                className="gh-insight__svg"
                viewBox={`0 0 ${chart.width} ${chart.height}`}
                preserveAspectRatio="none"
                role="img"
                aria-label={ariaLabel}
              >
                <defs>
                  <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="var(--gh-sky-1)" stopOpacity="0.38" />
                    <stop offset="60%" stopColor="var(--gh-mint-2)" stopOpacity="0.16" />
                    <stop offset="100%" stopColor="var(--gh-mint-2)" stopOpacity="0" />
                  </linearGradient>
                </defs>

                {/* 三条很淡的参考线，只作为视觉标尺（1 / 5.5 / 10） */}
                {[0, 0.5, 1].map((r) => (
                  <line
                    key={r}
                    className="gh-insight__grid"
                    x1={PAD_L}
                    x2={chart.width - PAD_R}
                    y1={PAD_T + r * (baseline - PAD_T)}
                    y2={PAD_T + r * (baseline - PAD_T)}
                  />
                ))}

                {chart.segments.map((seg, i) => (
                  <g key={`seg-${i}`}>
                    {seg.length > 1 && <path className="gh-insight__area" d={areaPath(seg, baseline)} fill={`url(#${gradId})`} />}
                    <path className="gh-insight__line" d={linePath(seg)} />
                  </g>
                ))}

                {chart.dots.map((p, i) => (
                  <circle className="gh-insight__dot" key={`dot-${i}`} cx={p.x} cy={p.y} r={4} />
                ))}

                {lastScoredIndex >= 0 && chart.dots.length > 0 && (
                  <circle
                    className="gh-insight__dot gh-insight__dot--today"
                    cx={chart.dots[chart.dots.length - 1].x}
                    cy={chart.dots[chart.dots.length - 1].y}
                    r={7}
                  />
                )}
              </svg>

              <div className="gh-insight__axis" aria-hidden="true">
                {trend.map((d, i) => (
                  <span
                    className={`gh-insight__tick${i === trend.length - 1 ? ' is-today' : ''}${typeof d.score === 'number' ? '' : ' is-blank'}`}
                    key={d.dayKey || d.label || i}
                  >
                    {d.label}
                  </span>
                ))}
              </div>

              <p className="gh-insight__today">
                {lastScoredIndex >= 0 && trend[lastScoredIndex].mood
                  ? `今天：${trend[lastScoredIndex].mood}`
                  : '今天还没有打卡记录'}
              </p>
            </div>

            <div className="gh-insight__cols">
              <div className="gh-insight__tile">
                <span className="gh-insight__tile-label">情绪稳定度</span>
                {counters && counters.moodStability7 !== null ? (
                  <>
                    <span className="gh-insight__tile-value">
                      {counters.moodStability7}
                      <em className="gh-insight__tile-unit">/100</em>
                    </span>
                    <span className={`gh-insight__bar ${stabilityTier(counters.moodStability7).cls}`}>
                      <i style={{ width: `${Math.max(0, Math.min(100, counters.moodStability7))}%` }} />
                    </span>
                    <span className="gh-insight__tile-hint">{stabilityTier(counters.moodStability7).word}</span>
                  </>
                ) : (
                  <span className="gh-insight__tile-hint">记录还不够，暂时算不出来</span>
                )}
              </div>

              <div className="gh-insight__tile">
                <span className="gh-insight__tile-label">近 7 天均值</span>
                {counters && counters.avgMood7 !== null ? (
                  <>
                    <span className="gh-insight__tile-value">
                      {fmt1(counters.avgMood7)}
                      <em className="gh-insight__tile-unit">分</em>
                    </span>
                    <span className="gh-insight__tile-hint">基于你自己的打分平均</span>
                  </>
                ) : (
                  <span className="gh-insight__tile-hint">近 7 天还没有打卡</span>
                )}
              </div>

              <div className="gh-insight__tile">
                <span className="gh-insight__tile-label">较上周</span>
                {counters && counters.moodDelta !== null ? (
                  <>
                    <span className={`gh-insight__tile-value gh-insight__delta ${deltaMeta(counters.moodDelta).cls}`}>
                      <span aria-hidden="true">{deltaMeta(counters.moodDelta).arrow}</span>
                      {counters.moodDelta > 0 ? '+' : ''}
                      {fmt1(counters.moodDelta)}
                      <em className="gh-insight__tile-unit">分</em>
                    </span>
                    <span className="gh-insight__tile-hint">{deltaMeta(counters.moodDelta).text}</span>
                  </>
                ) : (
                  <span className="gh-insight__tile-hint">还不够比较</span>
                )}
              </div>

              <div className="gh-insight__tile">
                <span className="gh-insight__tile-label">已记录</span>
                <span className="gh-insight__tile-value">
                  {counters ? counters.totalCheckIns : 0}
                  <em className="gh-insight__tile-unit">天</em>
                </span>
                <span className="gh-insight__tile-hint">累计打卡天数</span>
              </div>
            </div>

            {keywords.length > 0 && (
              <div className="gh-insight__keywords">
                <span className="gh-insight__tile-label">最近常出现的感受</span>
                <div className="gh-insight__chips">
                  {keywords.map((k) => (
                    <span className="gh-chip" key={k}>
                      {k}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        <p className="gh-insight__foot">数据仅来自你自己的打卡记录，不构成任何医学诊断。</p>
      </div>
    </div>
  );
}
