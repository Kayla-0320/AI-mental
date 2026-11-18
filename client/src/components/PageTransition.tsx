import { useLayoutEffect, useRef, type ReactNode, type RefObject } from 'react';
import { useLocation } from 'react-router-dom';

/**
 * 路由切换过渡动画。
 *
 * 全站**每一次**页面切换（患者端 / 咨询师端 / 管理端，以及登录、注册、AI 通话
 * 这三个不进布局的整屏路由）都由这里统一供动画，各页面自己不要再写一套。
 *
 * 实现选型（改动前请先读完，这里踩过的坑都在下面）：
 *
 *  1. **只做「进入」动画，不做「退出」动画。**
 *     要淡出旧页面就必须把旧页面继续挂在树上 300ms，而本项目的页面大多带重副作用
 *     （摄像头 / 麦克风 / WebSocket / 会话轮询，如 VoiceCall、Chat、ConsultantRoom）。
 *     新旧页面同挂 = 同一份副作用起两遍，实测这类页面会被搞崩。
 *     现在旧页面在 commit 时立即卸载，新页面从 opacity 0 淡入 —— 视觉上就是一次过渡，
 *     但没有双份副作用。
 *
 *  2. **动画挂在「本来就存在的容器」上，不新增包裹层。**
 *     患者端把动画挂在 `<main>` 与页顶图条包装层上（都已有节点，零 DOM 变更）；
 *     咨询师端 / 管理端 / 整屏路由用 `<PageTransition>` 包一层普通 div。
 *     用 WAAPI（`element.animate`）而不是 CSS `animation` + class 切换：
 *       · class 方案要靠「移除 class → 强制 reflow → 重加 class」才能重播，
 *         而且想靠 `key` 重播就得重挂子树（页面状态全丢）；
 *       · WAAPI 在同一批 DOM 上重放动画，组件不重挂，页面状态、滚动位置、表单都保留。
 *
 *  3. **`screen` 变体不能有 transform。**
 *     通话页（VoiceCall）根节点是 `position: fixed; inset: 0`。祖先上有 transform
 *     （哪怕只有动画那 300ms）会给 fixed 后代换包含块，通话页会瞬间塌成一条 0 高度，
 *     所以整屏路由只用 opacity。opacity 只创建层叠上下文，不影响 fixed 定位。
 *
 *  4. **尊重 `prefers-reduced-motion`。** 命中时直接不播，只留静态切换。
 *
 *  5. **同页换参数不算换页**（`/chat` ←→ `/chat/:id`、`/call` → `/call/:id`）：
 *     不重播动画，否则点一次会话列表就整页闪一下。判定见 `routeTransitionKey`。
 */

export type RouteTransitionVariant = 'page' | 'screen';

/** 与 garden.css 的 `--gh-ease` 同一根曲线，观感统一 */
const EASE = 'cubic-bezier(0.22, 0.8, 0.36, 1)';

const PRESETS: Record<RouteTransitionVariant, { duration: number; keyframes: Keyframe[] }> = {
  /** 布局内页面：淡入 + 轻微上浮（比 ghRise 快一档，切换才跟手） */
  page: {
    duration: 340,
    keyframes: [
      { opacity: 0, transform: 'translate3d(0, 16px, 0)' },
      { opacity: 1, transform: 'translate3d(0, 0, 0)' },
    ],
  },
  /** 整屏页面：只淡入，见文件头第 3 条 */
  screen: {
    duration: 380,
    keyframes: [{ opacity: 0 }, { opacity: 1 }],
  },
};

function prefersReducedMotion(): boolean {
  return typeof window !== 'undefined'
    && typeof window.matchMedia === 'function'
    && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/**
 * 「参数只换内容、不换页面」的路由：同一个页面上换 `:id` 不算页面切换，不重播动画。
 *
 *   · `/chat` ←→ `/chat/:conversationId`
 *     在 AI 倾诉页左侧会话列表里点另一条会话、或新建会话后跳到 `/chat/<id>`，
 *     整个页面（含会话列表本身）都在原地 —— 这时重播一次全屏淡入，看起来像
 *     "整个页面被重新加载了一遍"，而且点一次会话闪一次。所以不播。
 *   · `/call` → `/call/:conversationId`
 *     通话页没有 id 时会自己建一个会话再 `replace` 过去（见 VoiceCall.tsx 的会话生命周期）。
 *     那是同一次进入通话页，重播淡入 = 通话页连着闪两下。
 *
 * 反过来，**不在**这张表里的参数路由要正常播动画：
 * `/consultant/consultations/:bookingId`、`/experts/room/:bookingId` 都是
 * "列表 → 房间"的真实换页。
 */
const PARAM_ONLY_PAGES = ['/chat', '/call'];

/**
 * 得出这次渲染该用的过渡标识。
 * 普通路由用 `location.key`：同页原地重复点导航项、浏览器前进 / 后退都会有反馈；
 * 上面那几个参数页改用「页面名」，于是换 id 不重播、真正换页才播。
 */
export function routeTransitionKey(pathname: string, locationKey: string): string {
  const page = PARAM_ONLY_PAGES.find((p) => pathname === p || pathname.startsWith(`${p}/`));
  return page ? `page:${page}` : locationKey;
}

/**
 * 把「路由进入动画」挂到任意容器上。
 *
 * 动画标识见 `routeTransitionKey`；`useLayoutEffect` 保证在
 * **新页面已经 commit、浏览器还没绘制**的窗口里启动动画：
 * 新内容不会先以最终形态闪一帧再跳回 opacity 0。
 */
export function useRouteTransition<T extends HTMLElement>(
  ref: RefObject<T>,
  variant: RouteTransitionVariant = 'page',
) {
  const { key, pathname } = useLocation();
  const transitionKey = routeTransitionKey(pathname, key);

  useLayoutEffect(() => {
    const el = ref.current;
    // 老浏览器没有 WAAPI 就退化成"直接切换"，不影响功能
    if (!el || typeof el.animate !== 'function' || prefersReducedMotion()) return;

    const { duration, keyframes } = PRESETS[variant];
    const animation = el.animate(keyframes, { duration, easing: EASE });
    // 下一次路由切换（或卸载）时先取消，避免上一段动画的剩余时间影响新页面
    return () => animation.cancel();
  }, [transitionKey, variant, ref]);
}

interface PageTransitionProps {
  children: ReactNode;
  variant?: RouteTransitionVariant;
  className?: string;
}

/**
 * 过渡动画承载层。用在**没有现成容器可挂**的路由元素上
 * （咨询师端 / 管理端的内容区、登录 / 注册 / 通话这些整屏页）。
 * 它只是个普通 block div：不设 overflow / height / transform，
 * 内部页面的高度计算与 absolute / fixed 定位都不受影响。
 */
export default function PageTransition({ children, variant = 'page', className }: PageTransitionProps) {
  const ref = useRef<HTMLDivElement>(null);
  useRouteTransition(ref, variant);

  return (
    <div
      ref={ref}
      className={className ? `gh-route ${className}` : 'gh-route'}
      data-route-transition={variant}
    >
      {children}
    </div>
  );
}
