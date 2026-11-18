import type { CSSProperties } from 'react';
import { useLocation } from 'react-router-dom';

/**
 * 页顶大图（PageTopImage）—— 各内容页顶部那张整幅出血的插画 + 一版与首页同款的标题区。
 *
 * 与首页 Hero 的关系：
 *   首页有自己的 `PageHero`（文案 + AI 陪伴卡片）。这里是**同一个标题体系**的轻量版：
 *   同样的小标签 + 主标题 + 一行副标题，字号 / 字重 / 行高 / 颜色**直接复用** `.gh-hero__*`，
 *   区别只是没有主按钮、没有 AI 陪伴卡片，右侧把插画让出来。
 *
 *   为什么不抄一份数值而要复用 class：两边的标题只要有一天不同步，"像首页一样"就不成立。
 *   复用 `.gh-eyebrow` / `.gh-hero__title` / `.gh-hero__sub` 之后，改首页字号会同时改到这里。
 *
 * 为什么按**路由**取图与文案，而不是每个页面自己传：
 *   多个页面共用同一种图条时，格式（高度、整屏出血、文字位置、与顶栏的配合）必须完全一致；
 *   把「路由 → 图 + 文案」的映射收在这一处，页面里只写一行 `<PageTopImage />`，
 *   既不会出现某个页面忘了传、也不会各写一套而慢慢跑偏。
 *
 * ⚠️ 配套改动（改这里要一起看）：
 *   · `PatientLayout` 用同一个 `hasPageTopImage(path)` 判据决定
 *     （a）是否渲染图条包装层 `.gh-app__topstrip`（无条件渲染会在首页顶出一整屏空白）、
 *     （b）内容区形态 `.gh-app__main--top-strip`、顶栏是否透明压图；
 *   · 图条的定位、尺寸、裁切、文字层与渐隐规则都在
 *     `components/home/home-hero.css` 的 `.gh-top-image*` 小节里，
 *     高度变量 `--gh-strip-h` 定义在 `.gh-app__topstrip` 上。
 */

interface StripMeta {
  /** 文件名，位于 client/public/assets/ */
  image: string;
  /** 左上角小标签（与首页那个是同一个 `.gh-eyebrow` 组件） */
  eyebrow: string;
  /** 主标题，按行给 —— 换行位置是版式的一部分，不交给浏览器自动折行 */
  title: string[];
  /** 副标题：一行说明这一页能做什么 */
  sub: string;
}

/**
 * `.gh-reveal` 靠内联自定义属性 `--gh-delay` 错峰入场，与首页 PageHero 用的是同一套。
 * React 的 `CSSProperties` 不含自定义属性，所以用这个小工具函数收回类型。
 */
const delay = (ms: number): CSSProperties => ({ ['--gh-delay']: `${ms}ms` } as CSSProperties);

/** 路由 → 图 + 文案。图片顺序按"按顺序循环分配"落下来的结果（见文件末尾注释） */
const PAGE_META: Record<string, StripMeta> = {
  '/healing': {
    image: 'healing-cat-lake.jpg',
    eyebrow: '疗愈空间 · 放松练习',
    title: ['累了就歇一歇', '让心松口气'],
    sub: '小游戏与放松练习，用几分钟换一点轻盈',
  },
  // AI 倾诉页原用 hero-desk-lake.jpg（4.82:1 的超宽横条），裁成窄条后只剩一层书桌与
  // 花瓶的细带，和其余页的插画不是一套观感 —— 换成「窗边阳台、两只猫挨着睡」这张，
  // 既与其余页同一套画风，也更贴"倾诉 / 陪伴"的语境。
  '/chat': {
    image: 'hero-balcony-cats.jpg',
    eyebrow: 'AI 倾诉 · 24 小时在线',
    title: ['想说什么都可以', '我一直都在'],
    sub: '不打分、不评判 · 24 小时在线，陪你慢慢说',
  },
  '/profile': {
    image: 'hero-dusk-lake.jpg',
    eyebrow: '心理画像 · 多模态评估',
    title: ['看见情绪', '才能更好地照顾自己'],
    sub: '多模态情绪感知，把心理状态画成一张可读的图',
  },
  '/experts': {
    image: 'hero-balcony-cats.jpg',
    eyebrow: '咨询服务 · 持证咨询师',
    title: ['需要帮助的时候', '有人在这里'],
    sub: '持证心理咨询师在线，随时开启一次专业对话',
  },
  '/growth': {
    image: 'hero-sketch-lake.jpg',
    eyebrow: '我的成长 · 情绪记录',
    title: ['每一次坚持', '都在悄悄长大'],
    sub: '记录情绪、累积成长值，看见你走过的每一步',
  },
  '/companions': {
    image: 'hero-marketing-cafe.jpg',
    eyebrow: '同伴社区 · 匿名分享',
    title: ['在这里', '你不是一个人'],
    sub: '和懂你的人聊聊，分享也倾听，彼此照亮',
  },
  '/twin': {
    image: 'hero-desk-lake.jpg',
    eyebrow: '数字孪生 · 情绪预演',
    title: ['遇见另一个你', '先一步看见情绪'],
    sub: 'AI 心理孪生，陪你预演那些不太容易的时刻',
  },
};

/**
 * 需要**窄条**的路由（默认是 500px 大图，与其余页面一致）。
 *
 * 目前是空集合：AI 倾诉页原来放这里（它左侧历史 + 右侧对话，
 * 内容区高度写死为 `calc(100vh - 200px)`），按用户要求也改成与其他页同样的 500px 大图了。
 * 保留这个开关是为了以后某页需要矮图时不用再改结构 —— 加个路径即可（样式见
 * home-hero.css 的 `.gh-top-image--compact` 与 garden-shell.css 的 `--gh-strip-h` 覆盖）。
 *
 * ⚠️ 代价（实测 1440×900）：对话区高度 = 100vh − 200px 是**减去** nav 与图条后的固定值，
 * 500px 大图会把对话区压到约 456px；窗口更矮（如 768px 高）时只剩约 250px，
 * 需要往下滚才能看到输入框。
 */
const COMPACT_ROUTES = new Set<string>([]);

/**
 * 前缀匹配要**排除**的子路由。
 *
 * `/experts/room/:bookingId` 是挂在「咨询服务」下面的**咨询室**（PatientRoom），
 * 不是咨询服务那个板块本身：它是一间要专注会话的房间，顶上再压 500px 营销大图
 * 既挤掉会话区，也不合场景。前缀匹配本身是为了 `/chat/:conversationId` 那种
 * "同一个页面的详情"，这里把"另一个页面"摘出去。
 */
const NO_STRIP_PREFIXES = ['/experts/room'];

/**
 * 取当前路径对应的配置。
 *
 * 先精确匹配；再按**前缀**匹配带参数的路由（`/chat/:conversationId` 也属于 `/chat` 这一页）。
 * ⚠️ 只做精确匹配是个真实的坑：AI 倾诉页点进某个对话后路径变成 `/chat/c1`，
 * 图条会整个消失（实测过）。
 */
function metaFor(pathname: string): StripMeta | undefined {
  if (NO_STRIP_PREFIXES.some((p) => pathname === p || pathname.startsWith(`${p}/`))) return undefined;
  if (PAGE_META[pathname]) return PAGE_META[pathname];
  const hit = Object.keys(PAGE_META).find((p) => pathname.startsWith(`${p}/`));
  return hit ? PAGE_META[hit] : undefined;
}

/** 该路由是否有页顶图条；供 PatientLayout 决定内容区形态与顶栏是否透明压图 */
export function hasPageTopImage(pathname: string): boolean {
  return metaFor(pathname) !== undefined;
}

/**
 * 页顶图条本体。
 *
 * ⚠️ 由 `PatientLayout` 渲染在 `.gh-app__topstrip`（`<main>` 外面的全宽块）里 ——
 * 图条是 `position: absolute; inset: 0` 相对那一层定位，所以能整屏铺满；
 * 放回 `<main>` 里最多只能和内容同宽（两侧留白），就不像首页大图了。
 * 文字层用的是与页面内容相同的 `max-width: var(--gh-sec-w)` + 左右 `--gh-sec-pad`，
 * 所以标题的左边缘与下方卡片的左边缘**严丝合缝**（和首页同一个对齐线）。
 */
export default function PageTopImage() {
  const { pathname } = useLocation();
  const meta = metaFor(pathname);
  // 窄条判断同样按前缀匹配（将来若把某个带参数的页设为窄条，行为才一致）
  const compact = COMPACT_ROUTES.has(pathname)
    || [...COMPACT_ROUTES].some((p) => pathname.startsWith(`${p}/`));
  // 不在映射里的路由（例如设置页）什么都不渲染，免得出现空白条
  if (!meta) return null;

  return (
    <div className={`gh-top-image${compact ? ' gh-top-image--compact' : ''}`}>
      <div className="gh-top-image__bg" style={{ backgroundImage: `url('/assets/${meta.image}')` }} />
      {/* 左侧提亮：深色标题压在插画上时的对比度保障（与首页 .gh-hero__veil 同一套数值） */}
      <div className="gh-top-image__veil" aria-hidden="true" />
      {/* 顶部轻白纱：只服务压在图上的顶栏 */}
      <div className="gh-top-image__blend" aria-hidden="true" />

      {/* 文案层：结构与首页的 .gh-hero__copy 一致（小标签 + 主标题 + 副标题） */}
      <div className="gh-top-image__inner">
        <div className="gh-top-image__copy">
          <span className="gh-eyebrow gh-hero__eyebrow gh-reveal">
            <span className="gh-hero__eyebrow-dot" aria-hidden="true" />
            {meta.eyebrow}
          </span>

          <h1 className="gh-hero__title gh-reveal" style={delay(90)}>
            {meta.title.map((line, i) => (
              <span key={line}>
                {i > 0 && <br />}
                {line}
              </span>
            ))}
          </h1>

          <p className="gh-hero__sub gh-reveal" style={delay(170)}>
            {meta.sub}
          </p>
        </div>
      </div>
    </div>
  );
}

/*
 * 图片分配（五张图全用到）：
 *   /chat       hero-balcony-cats.jpg    （图 3）
 *   /profile    hero-dusk-lake.jpg       （图 2）
 *   /experts    hero-balcony-cats.jpg    （图 3）
 *   /growth     hero-sketch-lake.jpg     （图 4）
 *   /companions hero-marketing-cafe.jpg  （图 5）
 *   /twin       hero-desk-lake.jpg       （图 1）
 *   疗愈空间     healing-cat-lake.jpg     （图 5 的 2172x724 版本，原来的图条即此）
 *   ── 以下不加图条 ──
 *   首页（保持原有的文案 + 陪伴卡片 Hero）、设置页
 *
 * ⚠️ 图 1 是 4.82:1 的超宽条，图 2~5 约 2.14~3:1；`background-size: cover` 下
 * 高度 500px 时前者几乎不裁、后者会裁掉上下约 1/3（主体在画面中部，构图上仍成立）。
 * 想换图或改文案只改这个映射即可。
 */
