/**
 * 首页「我的成长」卡片 —— 纯展示组件（自己不发请求）。
 *
 * 数据全部来自 `GrowthOverview`（后端 GET /api/extra/growth 的真实结构），
 * 这里只负责"把已有数字讲清楚"，任何拿不到的数字都走空态，绝不补 0：
 *   · `index.score`     → 圆环中心的大数字（成长指数）
 *   · `level.progress`  → 圆环的填充比例（等级进度，与中心数字是两件事，不要混用）
 *   · `index.dimensions[].score === null` → 样本不足，显示「还没有数据」而不是 0
 *     （后端用 null 明确表示"样本不足"，渲染成 0 会变成"我这个维度是 0 分"的错误结论）
 *
 * 为什么不用 antd 的 Card / Progress：
 *   styles/healing-theme.css 用 `.ant-card { background: 渐变 !important }`、
 *   `.ant-progress-circle .ant-progress-text` 之类的全局规则把卡片压成粉色云朵，
 *   行内样式压不住；同时首页要的是"暖阳色"而不是粉色。所以整卡用原生元素 +
 *   garden.css 的 .gh-card 契约，圆环用内联 SVG 自己画。
 */
import type { GrowthOverview } from '../../types/garden';
import './growth-card.css';

/** 圆环几何：viewBox 160、半径 68、线宽 14，`stroke-dasharray` 用周长精确闭合 */
const RING_R = 68;
const RING_CIRCUMFERENCE = 2 * Math.PI * RING_R;

/** 四个维度卡片的展示顺序与图标（后端按 group 给维度，这里固定成"坚持/心情/投入/探索"一行） */
const TILES: Array<{ key: string; icon: string }> = [
  { key: 'persistence', icon: '🌱' },
  { key: 'mood', icon: '🌈' },
  { key: 'investment', icon: '⏳' },
  { key: 'exploration', icon: '🧭' },
];

/** 把 0-1 或 0-100 的进度值统一夹到 0-100，避免脏数据把圆环/进度条画飞 */
function clampPercent(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(100, Math.max(0, value));
}

export default function GrowthCard({ data, loading, onOpenGrowth }: {
  data: GrowthOverview | null;
  loading: boolean;
  onOpenGrowth: () => void;
}) {
  // 只在数据真的到达后才计算派生值；loading / data 为 null 时下面直接走骨架屏
  const showSkeleton = loading || data === null;

  const ringPercent = data ? clampPercent(data.level.progress) : 0;
  const ringDash = (ringPercent / 100) * RING_CIRCUMFERENCE;
  const score = data ? data.index.score : 0;

  // 维度按固定顺序取；后端若少给了某一维度，用兜底结构显示空态，而不是让整行塌掉
  const tiles = TILES.map((t) => ({
    ...t,
    dim: data?.index.dimensions.find((d) => d.key === t.key) ?? null,
  }));

  return (
    <section className="gh-card gh-card--raise gh-growth" aria-labelledby="gh-growth-title">
      {/* 装饰层：全部 aria-hidden，且 pointer-events:none（见 CSS），不参与无障碍树 */}
      <div className="gh-growth__decor" aria-hidden="true">
        <span className="gh-growth__blob gh-growth__blob--sun" />
        <span className="gh-growth__blob gh-growth__blob--sakura" />
        <span className="gh-growth__blob gh-growth__blob--mint" />
        {/* 角落的叶子水印：原创内联 SVG，低透明度 */}
        <svg className="gh-growth__leaf" viewBox="0 0 180 180" fill="none" focusable="false">
          <path
            d="M172 6C110 8 62 26 36 60 14 89 16 124 34 145c21 24 58 25 88 8 30-17 46-52 50-98 1-16 1-30 0-49Z"
            fill="url(#ghGrowthLeaf)"
          />
          <path
            d="M170 10C132 60 100 96 40 150"
            stroke="rgba(94,191,156,0.55)"
            strokeWidth="3"
            strokeLinecap="round"
          />
          <path d="M120 58c-16-6-30-4-42 6M96 92c-14-9-30-11-46-4M70 126c-11-10-23-15-37-14" stroke="rgba(94,191,156,0.4)" strokeWidth="2.5" strokeLinecap="round" />
          <defs>
            <linearGradient id="ghGrowthLeaf" x1="20" y1="10" x2="160" y2="160" gradientUnits="userSpaceOnUse">
              <stop stopColor="#a6e0cb" />
              <stop offset="1" stopColor="#ffc78a" />
            </linearGradient>
          </defs>
        </svg>
      </div>

      <header className="gh-growth__head">
        <div className="gh-growth__head-main">
          <span className="gh-icon gh-growth__badge" aria-hidden="true">🌿</span>
          <div className="gh-growth__titles">
            <h3 className="gh-growth__title" id="gh-growth-title">我的成长</h3>
            <p className="gh-growth__sub">每一次的坚持，都是更好的自己</p>
          </div>
        </div>
        <button type="button" className="gh-sec__more gh-growth__more" onClick={onOpenGrowth}>
          去看看 <span aria-hidden="true">→</span>
        </button>
      </header>

      {showSkeleton || !data ? (
        /* 骨架屏：形状与真实内容一一对应，避免"数字先蹦 0 再跳真实值"的闪烁 */
        <div className="gh-growth__body" aria-busy="true" aria-live="polite">
          <span className="gh-sr-only">成长数据加载中</span>
          <div className="gh-growth__ring-wrap">
            <div className="gh-skeleton gh-growth__sk-ring" />
          </div>
          <div className="gh-growth__level">
            <div className="gh-skeleton gh-growth__sk-line gh-growth__sk-line--lg" />
            <div className="gh-skeleton gh-growth__sk-line" />
            <div className="gh-skeleton gh-growth__sk-bar" />
            <div className="gh-growth__tiles">
              {TILES.map((t) => (
                <div key={t.key} className="gh-growth__tile">
                  <div className="gh-skeleton gh-growth__sk-line gh-growth__sk-line--sm" />
                  <div className="gh-skeleton gh-growth__sk-line" />
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <div className="gh-growth__body">
          {/* ---- 左：等级进度圆环（填充 = level.progress，中心 = index.score） ---- */}
          <div className="gh-growth__ring-wrap gh-reveal">
            <svg
              className="gh-growth__ring"
              viewBox="0 0 160 160"
              role="img"
              aria-label={`成长指数 ${score} 分，等级进度 ${Math.round(ringPercent)}%`}
            >
              <defs>
                <linearGradient id="ghGrowthRing" x1="24" y1="16" x2="140" y2="148" gradientUnits="userSpaceOnUse">
                  <stop offset="0" stopColor="#ffc78a" />
                  <stop offset="1" stopColor="#ffb3cd" />
                </linearGradient>
              </defs>
              <circle className="gh-growth__ring-track" cx="80" cy="80" r={RING_R} />
              <circle
                className="gh-growth__ring-fill"
                cx="80"
                cy="80"
                r={RING_R}
                strokeDasharray={RING_CIRCUMFERENCE}
                strokeDashoffset={RING_CIRCUMFERENCE - ringDash}
                style={{ '--gh-growth-dash': `${RING_CIRCUMFERENCE - ringDash}` } as React.CSSProperties}
              />
            </svg>
            <div className="gh-growth__ring-center">
              <span className="gh-growth__score">{score}</span>
              <span className="gh-growth__score-label">成长指数</span>
            </div>
          </div>

          {/* ---- 右：等级 + 一句话总结 + 升级进度条 ---- */}
          <div className="gh-growth__level gh-reveal" style={{ '--gh-delay': '90ms' } as React.CSSProperties}>
            <p className="gh-growth__lv">
              <span className="gh-growth__lv-icon" aria-hidden="true">{data.level.icon}</span>
              Lv.{data.level.level} {data.level.name}
            </p>
            <p className="gh-growth__summary">{data.summary}</p>

            <div className="gh-growth__xp">
              <div className="gh-growth__xp-top">
                <span className="gh-growth__xp-label">
                  {data.level.isMax ? '已达最高等级' : `${data.level.xp} / ${data.level.nextLevelXp} 成长值`}
                </span>
                <span className="gh-growth__xp-pct">{Math.round(ringPercent)}%</span>
              </div>
              <div
                className="gh-growth__bar"
                role="progressbar"
                aria-label="距离下一等级的进度"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(ringPercent)}
              >
                <span className="gh-growth__bar-fill" style={{ transform: `scaleX(${ringPercent / 100})` }} />
              </div>
            </div>

            {/* ---- 四维度：null 一律显示"还没有数据"，不显示 0 ---- */}
            <div className="gh-growth__tiles">
              {tiles.map(({ key, icon, dim }) => (
                <div className="gh-growth__tile" key={key}>
                  <div className="gh-growth__tile-head">
                    <span className="gh-growth__tile-icon" aria-hidden="true">{icon}</span>
                    <span className="gh-growth__tile-label">{dim ? dim.label : '—'}</span>
                  </div>
                  {dim && dim.score !== null ? (
                    <>
                      <p className="gh-growth__tile-value">
                        {dim.score}
                        <span className="gh-growth__tile-max">/100</span>
                      </p>
                      <div
                        className="gh-growth__tile-track"
                        role="progressbar"
                        aria-label={`${dim.label} ${dim.score} 分`}
                        aria-valuemin={0}
                        aria-valuemax={100}
                        aria-valuenow={dim.score}
                      >
                        <span
                          className="gh-growth__tile-fill"
                          style={{ transform: `scaleX(${clampPercent(dim.score) / 100})` }}
                        />
                      </div>
                    </>
                  ) : (
                    <p className="gh-growth__tile-empty">还没有数据</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ---- 打卡小条：数字全部直接取自 counters ---- */}
      <footer className="gh-growth__stripe">
        {showSkeleton || !data ? (
          <>
            <span className="gh-skeleton gh-growth__sk-chip" />
            <span className="gh-skeleton gh-growth__sk-chip" />
            <span className="gh-skeleton gh-growth__sk-chip" />
          </>
        ) : (
          <>
            <span className="gh-chip gh-growth__chip">累计打卡 {data.counters.totalCheckIns} 天</span>
            <span className="gh-chip gh-growth__chip">连续 {data.counters.streakDays} 天</span>
            <span className="gh-chip gh-growth__chip">本月活跃 {data.counters.activeDays30} 天</span>
            <span className="gh-chip gh-growth__chip">疗愈 {data.counters.healingSessions} 次</span>
            <span className="gh-chip gh-growth__chip">测评 {data.counters.assessments} 次</span>
          </>
        )}
      </footer>
    </section>
  );
}
