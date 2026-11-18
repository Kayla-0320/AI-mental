/**
 * 今日推荐内容（首页 · 纯展示组件）
 *
 * ⚠️ 当前状态：**首页已不下挂载**。
 *   产品要求把首页的「今日推荐内容」整块下线，Home.tsx 里的引用已移除；
 *   组件与样式文件按仓库既有惯例保留（同 `CrisisInterventionWidget.tsx` /
 *   `AnxietyFloatingWidget.tsx`），需要时把下面这一行挂回 Home.tsx 即可：
 *     <RecommendCards items={items} loading={loading} onOpen={...} onViewMore={...} />
 *   心理知识的检索入口仍在顶栏搜索（components/NavSearch.tsx → ContentPreviewModal）。
 *
 * 数据来源：`LearningContent[]`（GET /api/learning/recommendations），见 types/garden.ts。
 *
 * ⚠️ 为什么不读 `coverImage`：
 *   本项目 learningContent 表的 `coverImage` 恒为 null（没有素材上传链路），
 *   所以每个分类都用本文件手绘的 inline SVG 插画兜底——不用外链图片、
 *   不依赖网络，也不必担心图片挂了以后卡片塌成空白。
 *
 * ⚠️ 布局：桌面是一条横向 scroll-snap 轨道（鼠标/触控板可滚，也可用两侧箭头按钮），
 *   手机改为竖向排列（横滑在窄屏上很容易和页面纵向滚动打架）。
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import type { LearningContent } from '../../types/garden';
import './recommend.css';

/** 卡片数量与骨架屏数量：3 张骨架刚好铺满桌面首屏（2 张可见 + 1 张露出） */
const SKELETON_COUNT = 3;

interface CategoryMeta {
  /** 中文分类标签 */
  label: string;
  /** 插画外层渐变底（与 gh-sky-3 / mint-3 等同族，保持一套色系） */
  bg: string;
  /** 分类药丸的胶囊配色 */
  pillBg: string;
  pillColor: string;
  /** 插画主色（描边 + 大色块） */
  primary: string;
  /** 插画辅色 */
  secondary: string;
}

/**
 * 分类映射表。
 *
 * 取值来自 DB 里实际存在的 5 个 category；未知分类统一落到 `DEFAULT_CATEGORY`
 * （中性天蓝 + 小叶子），保证后端新增分类时卡片不会渲染成空白。
 */
const CATEGORY_MAP: Record<string, CategoryMeta> = {
  emotion: {
    label: '情绪',
    bg: 'linear-gradient(135deg, #e8f4ff 0%, #fff2e6 100%)',
    pillBg: 'rgba(124, 192, 234, 0.24)',
    pillColor: '#2f3d54',
    primary: '#7cc0ea',
    secondary: '#ffc78a',
  },
  stress: {
    label: '减压',
    bg: 'linear-gradient(135deg, #e9f9f2 0%, #eef7ff 100%)',
    pillBg: 'rgba(94, 191, 156, 0.24)',
    pillColor: '#2f3d54',
    primary: '#5ebf9c',
    secondary: '#a9dcf7',
  },
  sleep: {
    label: '睡眠',
    bg: 'linear-gradient(135deg, #f2ecfc 0%, #e7f1fd 100%)',
    pillBg: 'rgba(154, 134, 209, 0.24)',
    pillColor: '#2f3d54',
    primary: '#9a86d1',
    secondary: '#e3f4ff',
  },
  self_growth: {
    label: '自我成长',
    bg: 'linear-gradient(135deg, #fff4e8 0%, #eafaf3 100%)',
    pillBg: 'rgba(247, 159, 94, 0.24)',
    pillColor: '#2f3d54',
    primary: '#f79f5e',
    secondary: '#5ebf9c',
  },
  relationship: {
    label: '人际关系',
    bg: 'linear-gradient(135deg, #ffeef4 0%, #f5efff 100%)',
    pillBg: 'rgba(240, 127, 169, 0.24)',
    pillColor: '#2f3d54',
    primary: '#f07fa9',
    secondary: '#cbbcec',
  },
};

/** 未知分类的兜底：中性天蓝 */
const DEFAULT_CATEGORY: CategoryMeta = {
  label: '心理知识',
  bg: 'linear-gradient(135deg, #eef7ff 0%, #f3f6fb 100%)',
  pillBg: 'rgba(124, 192, 234, 0.22)',
  pillColor: '#2f3d54',
  primary: '#7cc0ea',
  secondary: '#a6e0cb',
};

function metaOf(category: string): CategoryMeta {
  return CATEGORY_MAP[category] ?? DEFAULT_CATEGORY;
}

/** type → 时长的量词（article→阅读 / audio→聆听 / video→观看 / course→学习） */
function durationNoun(type: string): string {
  switch (type) {
    case 'audio':
      return '聆听';
    case 'video':
      return '观看';
    case 'course':
      return '学习';
    case 'article':
    default:
      return '阅读';
  }
}

/* ============================================================================
 * 分类插画：全部手绘 SVG，统一 24 描边、圆角线帽，构成同一个家族的视觉
 * ========================================================================== */

function ArtEmotion({ primary, secondary }: { primary: string; secondary: string }) {
  return (
    <svg className="gh-rec-art__svg" viewBox="0 0 240 135" preserveAspectRatio="xMidYMid slice" aria-hidden="true" focusable="false">
      <circle cx="178" cy="42" r="19" fill={secondary} opacity="0.85" />
      <path d="M0 92 L52 48 L96 92 Z" fill={primary} opacity="0.4" />
      <path d="M64 92 L124 40 L188 92 Z" fill={primary} opacity="0.28" />
      <path d="M-6 135 L54 92 L126 135 Z" fill={primary} opacity="0.55" />
      <path d="M70 135 L152 88 L240 135 Z" fill={primary} opacity="0.38" />
      <path d="M0 92 L52 48 L96 92" fill="none" stroke={primary} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" opacity="0.75" />
      <path d="M64 92 L124 40 L188 92" fill="none" stroke={primary} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" opacity="0.6" />
    </svg>
  );
}

function ArtStress({ primary, secondary }: { primary: string; secondary: string }) {
  return (
    <svg className="gh-rec-art__svg" viewBox="0 0 240 135" preserveAspectRatio="xMidYMid slice" aria-hidden="true" focusable="false">
      <path d="M-10 135 Q42 74 96 106 Q140 130 186 96 Q216 76 250 92 L250 135 Z" fill={primary} opacity="0.35" />
      <path d="M-10 120 Q46 88 104 112 Q154 132 200 106 Q226 92 250 100 L250 135 L-10 135 Z" fill={primary} opacity="0.5" />
      <path d="M18 58 q16 -12 32 0 t32 0" fill="none" stroke={secondary} strokeWidth="2" strokeLinecap="round" opacity="0.9" />
      <path d="M62 32 q14 -10 28 0 t28 0" fill="none" stroke={secondary} strokeWidth="2" strokeLinecap="round" opacity="0.7" />
      <circle cx="196" cy="40" r="13" fill={secondary} opacity="0.55" />
    </svg>
  );
}

function ArtSleep({ primary, secondary }: { primary: string; secondary: string }) {
  return (
    <svg className="gh-rec-art__svg" viewBox="0 0 240 135" preserveAspectRatio="xMidYMid slice" aria-hidden="true" focusable="false">
      <path d="M150 20 a26 26 0 1 0 22 40 a20 20 0 1 1 -22 -40 Z" fill={primary} opacity="0.55" />
      <path d="M150 20 a26 26 0 1 0 22 40 a20 20 0 1 1 -22 -40 Z" fill="none" stroke={primary} strokeWidth="2" strokeLinecap="round" opacity="0.7" />
      {/* 星：一条竖线 + 一条横线组成的四角星，比多边形更贴近"手绘"感 */}
      <path d="M62 30 v14 M55 37 h14" stroke={secondary} strokeWidth="2" strokeLinecap="round" />
      <path d="M100 62 v10 M95 67 h10" stroke={secondary} strokeWidth="2" strokeLinecap="round" opacity="0.8" />
      <path d="M200 84 v10 M195 89 h10" stroke={secondary} strokeWidth="2" strokeLinecap="round" opacity="0.7" />
      <path d="M-10 135 Q40 104 96 124 Q150 142 200 116 Q228 102 250 112 L250 135 Z" fill={primary} opacity="0.28" />
      <path d="M-10 126 Q52 108 110 128 Q164 146 214 124 L250 124 L250 135 L-10 135 Z" fill={primary} opacity="0.45" />
    </svg>
  );
}

function ArtSelfGrowth({ primary, secondary }: { primary: string; secondary: string }) {
  return (
    <svg className="gh-rec-art__svg" viewBox="0 0 240 135" preserveAspectRatio="xMidYMid slice" aria-hidden="true" focusable="false">
      <path d="M120 122 V62" fill="none" stroke={primary} strokeWidth="2" strokeLinecap="round" opacity="0.8" />
      <path d="M120 86 C104 74 96 60 98 44 C114 46 126 58 120 86 Z" fill={primary} opacity="0.45" />
      <path d="M120 100 C136 90 146 76 145 60 C129 62 117 74 120 100 Z" fill={secondary} opacity="0.6" />
      <path d="M120 62 C112 46 114 30 122 18 C132 30 132 48 120 62 Z" fill={secondary} opacity="0.75" />
      <path d="M-10 135 Q40 114 96 126 Q152 138 206 120 Q230 112 250 118 L250 135 Z" fill={primary} opacity="0.3" />
      <path d="M120 122 V62" fill="none" stroke={primary} strokeWidth="2" strokeLinecap="round" opacity="0.55" />
    </svg>
  );
}

function ArtRelationship({ primary, secondary }: { primary: string; secondary: string }) {
  return (
    <svg className="gh-rec-art__svg" viewBox="0 0 240 135" preserveAspectRatio="xMidYMid slice" aria-hidden="true" focusable="false">
      <circle cx="100" cy="70" r="42" fill={primary} opacity="0.35" />
      <circle cx="142" cy="70" r="42" fill={secondary} opacity="0.4" />
      {/* 交叠处用曲线勾一下，暗示"两个人在同一段关系里" */}
      <path d="M121 34 a42 42 0 0 0 0 72" fill="none" stroke={primary} strokeWidth="2" strokeLinecap="round" opacity="0.75" />
      <path d="M121 106 a42 42 0 0 0 0 -72" fill="none" stroke={secondary} strokeWidth="2" strokeLinecap="round" opacity="0.75" />
      <path d="M-10 135 Q50 116 110 128 Q170 140 250 122 L250 135 Z" fill={primary} opacity="0.25" />
    </svg>
  );
}

/** 默认插画：一片叶子（未知分类用） */
function ArtDefault({ primary }: { primary: string }) {
  return (
    <svg className="gh-rec-art__svg" viewBox="0 0 240 135" preserveAspectRatio="xMidYMid slice" aria-hidden="true" focusable="false">
      <path d="M120 112 V66" fill="none" stroke={primary} strokeWidth="2" strokeLinecap="round" opacity="0.8" />
      <path d="M120 70 C120 42 142 22 172 20 C172 52 150 72 120 70 Z" fill={primary} opacity="0.4" />
      <path d="M120 70 C120 42 142 22 172 20" fill="none" stroke={primary} strokeWidth="2" strokeLinecap="round" opacity="0.7" />
      <path d="M-10 135 Q50 116 110 128 Q170 140 250 122 L250 135 Z" fill={primary} opacity="0.28" />
    </svg>
  );
}

/** 按分类挑插画 */
function CategoryArt({ category, meta }: { category: string; meta: CategoryMeta }) {
  const { primary, secondary } = meta;
  switch (category) {
    case 'emotion':
      return <ArtEmotion primary={primary} secondary={secondary} />;
    case 'stress':
      return <ArtStress primary={primary} secondary={secondary} />;
    case 'sleep':
      return <ArtSleep primary={primary} secondary={secondary} />;
    case 'self_growth':
      return <ArtSelfGrowth primary={primary} secondary={secondary} />;
    case 'relationship':
      return <ArtRelationship primary={primary} secondary={secondary} />;
    default:
      return <ArtDefault primary={primary} />;
  }
}

export default function RecommendCards({
  items,
  loading,
  onOpen,
  onViewMore,
}: {
  items: LearningContent[];
  loading: boolean;
  onOpen: (item: LearningContent) => void;
  /** 传了才渲染「查看更多 →」；不传表示调用方没有列表页可跳 */
  onViewMore?: () => void;
}) {
  const railRef = useRef<HTMLDivElement | null>(null);
  const [canPrev, setCanPrev] = useState(false);
  const [canNext, setCanNext] = useState(false);

  /**
   * 量一次轨道的溢出状态。
   *
   * 用 ref 回调 + effect 双通道触发：ref 回调负责"首次挂载就有内容"的场景，
   * effect 负责"items 到达后重新渲染"的场景（ref 回调在 key 不变时不会重跑）。
   */
  const measure = useCallback(() => {
    const el = railRef.current;
    if (!el) return;
    const max = el.scrollWidth - el.clientWidth;
    setCanPrev(el.scrollLeft > 4);
    // max <= 4 表示没有溢出（卡片少或还处于移动端竖排），两个箭头都禁用
    setCanNext(max > 4 && el.scrollLeft < max - 4);
  }, []);

  useEffect(() => {
    measure();
    const el = railRef.current;
    if (!el || typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [measure, items, loading]);

  /** 箭头滚动一步 = 一张卡宽（实测，兼容不同断点下的卡片宽度） */
  const scrollByCard = useCallback((dir: -1 | 1) => {
    const el = railRef.current;
    if (!el) return;
    const card = el.querySelector<HTMLElement>('.gh-rec-card');
    const gap = 16;
    const step = card ? card.offsetWidth + gap : el.clientWidth * 0.8;
    el.scrollBy({ left: dir * step, behavior: 'smooth' });
  }, []);

  const showEmpty = !loading && items.length === 0;

  return (
    <section className="gh-sec gh-rec" aria-labelledby="gh-rec-title">
      <div className="gh-sec__head gh-rec__head">
        <div>
          <h2 className="gh-sec__title" id="gh-rec-title">
            今日推荐内容
          </h2>
          <p className="gh-sec__sub">挑一篇，给自己几分钟</p>
        </div>

        {onViewMore && (
          <button type="button" className="gh-sec__more" onClick={onViewMore}>
            查看更多 <span aria-hidden="true">→</span>
          </button>
        )}
      </div>

      {loading ? (
        <div className="gh-rec__rail gh-rec__rail--static" aria-busy="true">
          {Array.from({ length: SKELETON_COUNT }, (_, i) => (
            <div className="gh-rec-card gh-rec-card--skeleton" key={i}>
              <div className="gh-rec-card__art">
                <span className="gh-skeleton gh-rec__sk-art" />
              </div>
              <div className="gh-rec-card__body">
                <span className="gh-skeleton gh-rec__sk-line gh-rec__sk-line--sm" />
                <span className="gh-skeleton gh-rec__sk-line gh-rec__sk-line--lg" />
                <span className="gh-skeleton gh-rec__sk-line gh-rec__sk-line--md" />
              </div>
            </div>
          ))}
        </div>
      ) : showEmpty ? (
        <div className="gh-empty gh-rec__empty">暂时还没有推荐内容，晚点再来看看</div>
      ) : (
        <div className="gh-rec__rail-wrap">
          <div
            className="gh-rec__rail"
            ref={(el) => {
              railRef.current = el;
              // 首帧内容已在 DOM 里时立刻量一次
              if (el) measure();
            }}
            onScroll={measure}
          >
            {items.map((item) => {
              const meta = metaOf(item.category);
              return (
                <button
                  type="button"
                  className="gh-rec-card"
                  key={item.id}
                  onClick={() => onOpen(item)}
                  aria-label={`${meta.label} · ${item.title}`}
                >
                  <span className="gh-rec-card__art" style={{ background: meta.bg }}>
                    <span className="gh-rec-card__art-inner">
                      <CategoryArt category={item.category} meta={meta} />
                    </span>
                    <span
                      className="gh-rec-card__pill"
                      style={{ background: meta.pillBg, color: meta.pillColor }}
                    >
                      {meta.label}
                    </span>
                  </span>

                  <span className="gh-rec-card__body">
                    <span className="gh-rec-card__title">{item.title}</span>
                    <span className="gh-rec-card__summary">
                      {item.summary ?? '这篇内容暂时还没有简介'}
                    </span>
                    <span className="gh-rec-card__meta">
                      <span className="gh-rec-card__meta-item">
                        {item.type === 'audio' ? '🎧' : item.type === 'video' ? '▶' : item.type === 'course' ? '✎' : '📖'}
                      </span>
                      {typeof item.duration === 'number' && (
                        <span className="gh-rec-card__meta-item">
                          {item.duration} 分钟{durationNoun(item.type)}
                        </span>
                      )}
                      <span className="gh-rec-card__meta-item gh-rec-card__meta-item--tag">{meta.label}</span>
                    </span>
                  </span>
                </button>
              );
            })}
          </div>

          {/* 箭头按钮只服务于桌面横向轨道；移动端是竖排，隐藏（见 CSS） */}
          <button
            type="button"
            className="gh-arrow-btn gh-rec__nav gh-rec__nav--prev"
            onClick={() => scrollByCard(-1)}
            disabled={!canPrev}
            aria-label="查看上一张推荐"
          >
            <span aria-hidden="true">←</span>
          </button>
          <button
            type="button"
            className="gh-arrow-btn gh-rec__nav gh-rec__nav--next"
            onClick={() => scrollByCard(1)}
            disabled={!canNext}
            aria-label="查看下一张推荐"
          >
            <span aria-hidden="true">→</span>
          </button>
        </div>
      )}
    </section>
  );
}
