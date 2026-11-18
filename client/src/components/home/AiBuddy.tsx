/**
 * 「小灵」—— 首页原创 AI 心理伙伴形象。
 *
 * 为什么是内联 SVG 而不是图片：
 *   1. 参考图里的吉祥物是位图素材，直接复用会引入来源不明的第三方资产；
 *      这里用同一套奶油白 + 薄荷绿配色手绘一个，风格与卡片体系一致、可任意缩放；
 *   2. 面部表情可以跟着 `mood` 变，AI 伙伴卡不再是"贴上去的一张图"。
 *
 * 无外部依赖、无网络请求，纯 `<svg>`，因此在任何断网环境下都能渲染。
 */

export type BuddyMood = 'idle' | 'happy' | 'listen';

export default function AiBuddy({
  size = 120,
  mood = 'idle',
  className,
}: {
  size?: number;
  mood?: BuddyMood;
  className?: string;
}) {
  // 三种情绪只改眼睛与嘴，身体保持一致，避免出现"换了个角色"的观感
  const eyes =
    mood === 'happy' ? (
      <>
        <path d="M40 60 q6 -8 12 0" stroke="#5b6b85" strokeWidth="3.4" strokeLinecap="round" fill="none" />
        <path d="M68 60 q6 -8 12 0" stroke="#5b6b85" strokeWidth="3.4" strokeLinecap="round" fill="none" />
      </>
    ) : mood === 'listen' ? (
      <>
        <circle cx="46" cy="60" r="3.6" fill="#5b6b85" />
        <circle cx="74" cy="60" r="3.6" fill="#5b6b85" />
        <path d="M56 72 q4 3 8 0" stroke="#5b6b85" strokeWidth="3" strokeLinecap="round" fill="none" />
      </>
    ) : (
      <>
        <circle cx="46" cy="60" r="4.2" fill="#5b6b85" />
        <circle cx="74" cy="60" r="4.2" fill="#5b6b85" />
        <circle cx="47.4" cy="58.6" r="1.4" fill="#fff" />
        <circle cx="75.4" cy="58.6" r="1.4" fill="#fff" />
      </>
    );

  const mouth =
    mood === 'happy' ? (
      <path d="M55 72 q5 6 10 0" stroke="#5b6b85" strokeWidth="3" strokeLinecap="round" fill="none" />
    ) : (
      <path d="M56 74 q4 3.5 8 0" stroke="#5b6b85" strokeWidth="2.6" strokeLinecap="round" fill="none" />
    );

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 120 120"
      className={className}
      role="img"
      aria-label="AI 心理伙伴小灵"
    >
      <defs>
        <radialGradient id="ghBuddyBody" cx="38%" cy="30%" r="78%">
          <stop offset="0%" stopColor="#ffffff" />
          <stop offset="62%" stopColor="#fdf9f4" />
          <stop offset="100%" stopColor="#f2e9dd" />
        </radialGradient>
        <linearGradient id="ghBuddyLeaf" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#8fd9bb" />
          <stop offset="100%" stopColor="#4fb894" />
        </linearGradient>
        <linearGradient id="ghBuddyAura" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#ffe6b8" stopOpacity="0.85" />
          <stop offset="100%" stopColor="#ffc6dd" stopOpacity="0.6" />
        </linearGradient>
        <radialGradient id="ghBuddyBlush" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#ffb3cd" stopOpacity="0.75" />
          <stop offset="100%" stopColor="#ffb3cd" stopOpacity="0" />
        </radialGradient>
        <filter id="ghBuddyShadow" x="-30%" y="-20%" width="160%" height="160%">
          <feDropShadow dx="0" dy="5" stdDeviation="5" floodColor="#7c98b8" floodOpacity="0.22" />
        </filter>
      </defs>

      {/* 身后光晕：与首页"日落"主视觉呼应，也让人物在浅色卡片上不显得发灰 */}
      <ellipse cx="60" cy="66" rx="52" ry="46" fill="url(#ghBuddyAura)" opacity="0.55" />

      {/* 影子 */}
      <ellipse cx="60" cy="106" rx="27" ry="6.5" fill="#7c98b8" opacity="0.16" />

      {/* 身体 */}
      <g filter="url(#ghBuddyShadow)">
        <path
          d="M60 22c20.5 0 35 14.6 35 36.4 0 20.3-13.6 35.6-35 35.6S25 78.7 25 58.4C25 36.6 39.5 22 60 22Z"
          fill="url(#ghBuddyBody)"
          stroke="rgba(124,152,184,0.28)"
          strokeWidth="1.6"
        />
      </g>

      {/* 头顶嫩芽：与"成长/植物"隐喻一致 */}
      <g>
        <path d="M60 24c0-6 0-9 0-12" stroke="#4fb894" strokeWidth="3" strokeLinecap="round" fill="none" />
        <path d="M60 15c-7-5-13-3-13-3 1 6 6 9 13 8Z" fill="url(#ghBuddyLeaf)" />
        <path d="M60 16c6-5 12-3.5 12-3.5-1 5.6-5.6 8.6-12 7.8Z" fill="url(#ghBuddyLeaf)" opacity="0.9" />
      </g>

      {/* 耳朵 */}
      <ellipse cx="28" cy="44" rx="6.5" ry="10" fill="#fdf9f4" stroke="rgba(124,152,184,0.24)" strokeWidth="1.4" transform="rotate(-18 28 44)" />
      <ellipse cx="92" cy="44" rx="6.5" ry="10" fill="#fdf9f4" stroke="rgba(124,152,184,0.24)" strokeWidth="1.4" transform="rotate(18 92 44)" />

      {/* 腮红 */}
      <ellipse cx="38" cy="70" rx="9" ry="6" fill="url(#ghBuddyBlush)" />
      <ellipse cx="82" cy="70" rx="9" ry="6" fill="url(#ghBuddyBlush)" />

      {eyes}
      {mouth}

      {/* 小肚兜上的心：AI 伙伴的身份符号 */}
      <path
        d="M60 86c-4.4-3-6.6-5.2-6.6-7.6 0-1.9 1.5-3.3 3.3-3.3 1.3 0 2.5.8 3.3 2 .8-1.2 2-2 3.3-2 1.8 0 3.3 1.4 3.3 3.3 0 2.4-2.2 4.6-6.6 7.6Z"
        fill="#f486ae"
        opacity="0.85"
      />
    </svg>
  );
}
