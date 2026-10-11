import type { CSSProperties, ReactNode } from 'react';
import { useNavigate } from 'react-router-dom';
import AiBuddy from './AiBuddy';
import './home-hero.css';

/**
 * `.gh-reveal` 靠内联自定义属性 `--gh-delay` 错峰入场。
 * React 的 `CSSProperties` 不含自定义属性，故用一个显式的小工具函数收回类型，
 * 避免在 JSX 里到处写 `as React.CSSProperties`。
 */
const delay = (ms: number): CSSProperties => ({ ['--gh-delay']: `${ms}ms` } as CSSProperties);

/** 手绘 SVG 的心形：比 emoji 更贴主视觉，也不受系统字体差异影响 */
function HeartMark() {
  return (
    <svg className="gh-hero__heart" viewBox="0 0 32 32" aria-hidden="true">
      <defs>
        <linearGradient id="ghHeroHeart" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#ff9dc4" />
          <stop offset="100%" stopColor="#ef6f9f" />
        </linearGradient>
      </defs>
      <path
        d="M16 27C7.7 21.3 3.6 16.9 3.6 12.3 3.6 8.6 6.4 5.8 9.9 5.8c2.5 0 4.7 1.4 5.9 3.6 1.2-2.2 3.4-3.6 5.9-3.6 3.6 0 6.4 2.8 6.4 6.5 0 4.6-4.1 9-12.1 14.7Z"
        fill="url(#ghHeroHeart)"
      />
      <path
        d="M16 27C7.7 21.3 3.6 16.9 3.6 12.3 3.6 8.6 6.4 5.8 9.9 5.8c2.5 0 4.7 1.4 5.9 3.6 1.2-2.2 3.4-3.6 5.9-3.6 3.6 0 6.4 2.8 6.4 6.5 0 4.6-4.1 9-12.1 14.7Z"
        fill="none"
        stroke="rgba(226,87,138,0.35)"
        strokeWidth="1.2"
      />
    </svg>
  );
}

/** 主视觉文案按钮上的右箭头 */
function ArrowMark() {
  return (
    <svg viewBox="0 0 24 24" width="17" height="17" aria-hidden="true">
      <path
        d="M4 12h14M12.5 5.5 19 12l-6.5 6.5"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/**
 * 主视觉（Hero）—— 首页与疗愈空间共用。
 *
 * 两个变体的差别只在**背景图**与几个文案位（见下方 props）；标题、按钮、
 * AI 陪伴卡片、与下方内容区的衔接渐变全部复用同一套 `.gh-hero*` 样式，
 * 这样两个页面的顶部结构天然一致，不会各写一份而慢慢跑偏。
 *
 *   home    → /assets/hero-lake-sunset.jpg（女孩与猫坐在湖边看日落）
 *   healing → /assets/healing-cat-lake.jpg（窗边小桌上的猫、马克杯与书）
 *
 * 分层要求（来自改版需求）：
 *   背景图是**独立图片资源**（走 background-image），标题 / 按钮 / 卡片全部由
 *   真实 HTML+CSS 实现，绝不把文字画进图里。
 *
 * 人物/主体不被遮挡的做法：
 *   首页插画里女孩位于画面横向约 62% 处，`cover` 在 2.57:1 左右的容器里会按高度
 *   缩放、裁掉少量左右边缘，取 `background-position: center 48%`（移动端 62%）
 *   即把她稳定落在容器 60%~67%，右侧卡片从 74% 起，互不重叠。
 *   疗愈空间那张的主体（睡觉的猫 + 书架 + 窗帘）在画面**右侧 73%~100%**，
 *   因此桌面必须用 `right 48%` 让右侧内容尽量留在画面里，
 *   否则卡片会压住猫 —— 这与首页的取景方向正好相反。
 */
export interface PageHeroProps {
  /** 背景图与取景：home / healing 各自一套（见上） */
  variant?: 'home' | 'healing';
  /** 页面主标题；给 ReactNode 以便单独给某一行上色（如首页的 `.gh-hero__title-accent`） */
  title: ReactNode;
  /** 副标题（一行说明） */
  subtitle: string;
  /**
   * 主按钮文案。
   * 不传就**不渲染按钮那一行**（连 `ctaNote` 一起）—— 疗愈空间只用文案不用按钮，
   * 但仍复用同一套 Hero 布局。
   */
  ctaLabel?: string;
  /** 主按钮行为；传 href 时渲染为锚点（用于页内跳转），否则渲染为按钮 */
  ctaHref?: string;
  /** 锚点按钮点击时接管默认跳转（页内有 fixed 顶栏时要自己算滚动位置） */
  onCtaHrefClick?: (e: React.MouseEvent<HTMLAnchorElement>) => void;
  onCta?: () => void;
  /** 主按钮右侧的次要文案（首页是"关爱心灵，从现在开始"） */
  ctaNote?: string;
  /** 左上角小标签 */
  eyebrow: string;
  /**
   * AI 陪伴卡片：内容与行为由调用方给，结构固定（保证两页一致）。
   * 不传就不渲染这张卡 —— 疗愈空间不需要它，右侧留白交给背景插画。
   */
  buddy?: {
    greeting: string;
    nickname: string;
    /** 伙伴表情：打卡过就开心一点 */
    mood: 'happy' | 'idle';
    title: string;
    desc: string;
    /** 卡片主按钮文案（默认「和我聊聊」） */
    ctaLabel?: string;
    onChat: () => void;
    /** 卡片底部小字（默认「不想打字？试试心理电话」），点击进 /call */
    altLabel?: string;
  };
}

export default function PageHero({
  variant = 'home',
  title,
  subtitle,
  ctaLabel,
  ctaHref,
  onCtaHrefClick,
  onCta,
  ctaNote,
  eyebrow,
  buddy,
}: PageHeroProps) {
  const navigate = useNavigate();

  const ctaInner = (
    <>
      {ctaLabel}
      <ArrowMark />
    </>
  );

  return (
    <section
      className={`gh-hero gh-hero--${variant}${buddy ? '' : ' gh-hero--no-buddy'}`}
      aria-labelledby="gh-hero-title"
    >
      {/* 背景插画（独立资源，可被浏览器单独缓存） */}
      <div className="gh-hero__bg" aria-hidden="true" />
      {/* 左侧提亮蒙版：保证深色标题在天空上仍有充足对比度 */}
      <div className="gh-hero__veil" aria-hidden="true" />

      <div className="gh-hero__inner">
        <div className="gh-hero__copy">
          <span className="gh-eyebrow gh-hero__eyebrow gh-reveal">
            <span className="gh-hero__eyebrow-dot" aria-hidden="true" />
            {eyebrow}
          </span>

          <h1 id="gh-hero-title" className="gh-hero__title gh-reveal" style={delay(90)}>
            {title}
          </h1>

          <p className="gh-hero__sub gh-reveal" style={delay(170)}>{subtitle}</p>

          {/* 没有 ctaLabel 就整行不渲染（含右侧的次要文案） */}
          {ctaLabel && (
            <div className="gh-hero__actions gh-reveal" style={delay(250)}>
              {ctaHref ? (
                <a className="gh-btn gh-btn--primary gh-hero__cta" href={ctaHref} onClick={onCtaHrefClick}>{ctaInner}</a>
              ) : (
                <button type="button" className="gh-btn gh-btn--primary gh-hero__cta" onClick={onCta}>
                  {ctaInner}
                </button>
              )}
              {ctaNote && <span className="gh-hero__secondary">{ctaNote}</span>}
            </div>
          )}
        </div>

        {/* 半透明 AI 陪伴卡片：不传 buddy 就不渲染 */}
        {buddy && (
          <aside className="gh-hero__buddy gh-reveal" style={delay(320)}>
            <div className="gh-hero__buddy-glow" aria-hidden="true" />
            <div className="gh-hero__buddy-top">
              <span className="gh-chip gh-hero__online">
                <span className="gh-hero__online-dot" aria-hidden="true" />
                24 小时在线
              </span>
            </div>

            <div className="gh-hero__buddy-art">
              <AiBuddy size={104} mood={buddy.mood} className="gh-float" />
            </div>

            <div className="gh-hero__buddy-body">
              <p className="gh-hero__buddy-hi">{buddy.greeting}，{buddy.nickname}</p>
              <p className="gh-hero__buddy-title">{buddy.title}</p>
              <p className="gh-hero__buddy-desc">{buddy.desc}</p>
            </div>

            <button type="button" className="gh-btn gh-btn--primary gh-btn--block" onClick={buddy.onChat}>
              {buddy.ctaLabel ?? '和我聊聊'}
            </button>

            <button type="button" className="gh-hero__buddy-alt" onClick={() => navigate('/call')}>
              {buddy.altLabel ?? '不想打字？试试心理电话'}
            </button>
          </aside>
        )}
      </div>

      {/* 与下方奶油色内容区自然衔接 */}
      <div className="gh-hero__fade" aria-hidden="true" />
    </section>
  );
}
