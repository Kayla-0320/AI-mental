/**
 * 疗愈空间 —— 心理小游戏合集
 *
 * 四大分类（共 12 个小游戏）：
 *   🎯 注意力 & 专注力：舒尔特方格、色词冲突、序列密码、持续注意
 *   🌿 缓解压力：气球呼吸训练、冥想音频、静默计时
 *   🎨 释放压力：涂鸦画板、橡皮人拉扯
 *   ⚡ 反应 & 协调：反应灯（Go / No-Go）、多目标追踪、跟手追踪、选择反应（Simon）、甩点挑战、节奏打击
 *
 * 每个游戏都带统一的「结束并回退」按钮（快捷键 Esc）：
 * 立即中断正在运行的计时器 / 动画帧 / 音频，并把游戏状态回退到初始状态；
 * 切换分类时也会自动中断，避免隐藏标签页里的进程（计时器、白噪音）继续跑。
 *
 * 每个游戏右下角还有一个放大按钮：点一下整局铺满整个页面专注玩，
 * 顶栏的「返回」或 Esc 回到卡片排布。放大只切样式、不动 DOM 结构，
 * 所以来回切换不会把正在玩的一局丢掉（详见 GameFrame 的注释）。
 *
 * 反应 & 协调这一类额外说明：
 * 范式参考电竞选拔与认知心理学里的常用任务，但定位是「自我觉察」而不是「天赋判定」——
 * 浏览器测反应时有约 ±20~30ms 的固定偏差，不足以支撑跨人比较；
 * 因此只呈现自己的数值与纵向变化，不给分数排名，也不做任何能力定性结论。
 */
import { useState, useEffect, useRef, useCallback, useMemo, useContext, createContext } from 'react';
import { Card, Tabs, Tag, Space, Typography, Button, Row, Col, Tooltip, Slider } from 'antd';
import {
  BulbOutlined, CloudOutlined, ExperimentOutlined, StopOutlined, ThunderboltOutlined,
  ExpandOutlined, CompressOutlined, ArrowLeftOutlined,
} from '@ant-design/icons';
import { healingApi } from '../../services';

const { Title, Text } = Typography;

// ============================================================
// 0. 通用能力：可打断运行器 / 结束并回退操作条
// ============================================================

/**
 * 可打断运行器
 *
 * 统一登记定时器、动画帧与自定义清理函数，「结束」时一次性全部中断；
 * 同时用自增 token 作废所有尚未执行的回调，防止打断后旧回调（尤其是
 * 递归定时器、setTimeout 延迟回调）继续推进游戏状态。
 */
function useInterruptible() {
  const timersRef = useRef<Set<any>>(new Set());
  const framesRef = useRef<Set<any>>(new Set());
  const cleanupsRef = useRef<Array<() => void>>([]);
  const tokenRef = useRef(0);

  const clearAll = useCallback(() => {
    timersRef.current.forEach(id => { clearInterval(id); clearTimeout(id); });
    framesRef.current.forEach(id => cancelAnimationFrame(id));
    cleanupsRef.current.forEach(fn => { try { fn(); } catch { /* 清理异常不影响回退流程 */ } });
    timersRef.current.clear();
    framesRef.current.clear();
    cleanupsRef.current = [];
  }, []);

  /** 开启新一轮：先中断上一轮，返回本轮 token */
  const begin = useCallback(() => {
    clearAll();
    tokenRef.current += 1;
    return tokenRef.current;
  }, [clearAll]);

  /** 中断当前进程：清理全部资源并作废待执行回调 */
  const interrupt = useCallback(() => {
    clearAll();
    tokenRef.current += 1;
  }, [clearAll]);

  /** 回调内自检：token 失效说明已被「结束」，应立即返回 */
  const isActive = useCallback((token: number) => token === tokenRef.current, []);
  const token = useCallback(() => tokenRef.current, []);

  const trackTimer = useCallback((id: any) => { timersRef.current.add(id); return id; }, []);
  const trackFrame = useCallback((id: any) => { framesRef.current.add(id); return id; }, []);
  const addCleanup = useCallback((fn: () => void) => { cleanupsRef.current.push(fn); }, []);

  // 组件卸载（离开疗愈空间）时同样中断，避免计时器 / 音频残留
  useEffect(() => clearAll, [clearAll]);

  return useMemo(
    () => ({ begin, interrupt, isActive, token, trackTimer, trackFrame, addCleanup }),
    [begin, interrupt, isActive, token, trackTimer, trackFrame, addCleanup],
  );
}

/**
 * 全屏放大的状态容器
 * 由 GameFrame 提供；游戏组件通过 useIsExpanded() 读它来决定要不要用更大的尺寸。
 */
interface GameFullscreenApi {
  expanded: boolean;
  setExpanded: (v: boolean) => void;
  toggle: () => void;
}

const GameFullscreenContext = createContext<GameFullscreenApi | null>(null);

/** 是否处于放大态；没有套在 GameFrame 里时恒为 false */
function useIsExpanded() {
  const api = useContext(GameFullscreenContext);
  return !!api?.expanded;
}

/** Esc 键 = 结束并回退，仅在确实有进程在跑时挂载 */
function useEscapeToEnd(onEnd: () => void, enabled: boolean) {
  // 放大状态下 Esc 先用来退出全屏（由 GameFrame 在捕获阶段处理），
  // 否则玩家习惯性按 Esc 想退出全屏，结果把正在玩的一局给结束了。
  const blocked = useIsExpanded();
  useEffect(() => {
    if (!enabled || blocked) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onEnd();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onEnd, enabled, blocked]);
}

/** 切换分类时自增，供正在运行的游戏自动结束 */
const HealingAbortContext = createContext(0);

/**
 * 分类切换后自动结束当前游戏
 * @param onAbort 中断回调（通常是 endGame）
 * @param enabled 只有确实在运行时才中断，避免清掉已完成的结果
 */
function useAbortOnTabChange(onAbort: () => void, enabled: boolean) {
  const version = useContext(HealingAbortContext);
  const seenRef = useRef(version);
  useEffect(() => {
    if (seenRef.current === version) return;
    seenRef.current = version;
    if (enabled) onAbort();
  }, [version, enabled, onAbort]);
}

/** 「已回退到初始状态」提示：显示 2.4 秒后自动消失 */
function useEndedFlag(): [boolean, () => void, () => void] {
  const [ended, setEnded] = useState(false);
  const timeoutRef = useRef<any>(null);

  const flashEnded = useCallback(() => {
    setEnded(true);
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    timeoutRef.current = setTimeout(() => setEnded(false), 2400);
  }, []);

  const clearEnded = useCallback(() => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    setEnded(false);
  }, []);

  useEffect(() => () => { if (timeoutRef.current) clearTimeout(timeoutRef.current); }, []);

  return [ended, flashEnded, clearEnded];
}

/** 统一的「结束并回退」操作条：中间放本局按钮，右下角固定为放大 / 返回全屏按钮 */
function GameActions({ children, onEnd, endDisabled, ended, tip }: {
  children?: React.ReactNode;
  onEnd: () => void;
  endDisabled?: boolean;
  ended?: boolean;
  tip?: string;
}) {
  const fs = useContext(GameFullscreenContext);
  return (
    // 注意：marginTop 交给 GAME_LAYOUT_CSS 的 margin-top:auto（压到卡片底部），
    // 这里写行内 marginTop 会盖掉它
    <div className="game-actions" style={{
      paddingTop: 12, borderTop: '1px dashed #f0f0f0',
      display: 'flex', alignItems: 'center', gap: 10,
    }}>
      {/* 左侧等宽占位：有没有放大按钮，中间那组按钮都保持视觉居中 */}
      <div style={{ width: 66, flex: '0 0 auto' }} />
      <div style={{
        flex: '1 1 auto', display: 'flex', alignItems: 'center',
        justifyContent: 'center', gap: 10, flexWrap: 'wrap',
      }}>
        {children}
        <Tooltip title={tip ?? '立即中断当前进程并回退到初始状态（快捷键 Esc）'}>
          <span style={{ display: 'inline-block' }}>
            <Button danger icon={<StopOutlined />} onClick={onEnd} disabled={endDisabled}
              style={{ borderRadius: 20 }}>
              结束并回退
            </Button>
          </span>
        </Tooltip>
        {ended && <Tag color="red">已结束并回退</Tag>}
      </div>
      {/*
        右下角：放大铺满整页 / 退出全屏。
        这里特意留出 32px：页面右下角常驻一组 position:fixed 的悬浮按钮（z-index 1000），
        按钮贴着卡片右边会被它盖住、点了没反应（实测 elementFromPoint 命中浮层）。
        左侧占位同步加宽到 66，保证中间那组按钮仍然视觉居中。
      */}
      <div style={{
        width: 66, flex: '0 0 auto', paddingRight: 32, boxSizing: 'border-box',
        display: 'flex', justifyContent: 'flex-end',
      }}>
        {fs && (
          <Tooltip title={fs.expanded ? '退出全屏，回到卡片（Esc）' : '放大，铺满整个页面'}>
            <Button
              shape="circle"
              className="game-expand-btn"
              aria-label={fs.expanded ? '退出全屏' : '放大'}
              icon={fs.expanded ? <CompressOutlined /> : <ExpandOutlined />}
              onClick={fs.toggle}
            />
          </Tooltip>
        )}
      </div>
    </div>
  );
}

// ============================================================
// 0.6 游戏卡片：放大后铺满整页
// ============================================================

/**
 * 卡片等高布局
 *
 * 问题：每个游戏内容高度差得很多（气球呼吸 400 出头，冥想自然音 500 多），
 * 同一行里卡片底部参差不齐，短卡片下面留一大块空隙，两边的「结束并回退」也不在同一水平线。
 *
 * 做法：把「行 → 列 → 卡片 → 卡体 → 游戏根节点」整条链路都变成纵向 flex，
 * 让卡片撑满行高，再把操作条用 margin-top:auto 压到底部。
 * 这样一行里卡片的上下边界和操作条都严格对齐。
 *
 * 两个坑：
 *  - 不能给这条链路加 min-height:0，否则固有高度计算时内容会被压扁（实测舒尔特方格被压到 197px）；
 *  - 操作条的 margin-top 不能在行内样式里写死，行内优先级高于这里的选择器，
 *    写了 auto 就永远不生效（实测操作条错位 5~74px）。
 *
 * 只作用于卡片排布（.game-frame:not(.game-frame-expanded)），
 * 放大全屏那条链路有自己的内联 flex 样式，完全不受影响。
 */
const GAME_LAYOUT_CSS = `
.game-frame:not(.game-frame-expanded) { height: 100%; display: flex; flex-direction: column; }
.game-frame:not(.game-frame-expanded) .game-frame-body { flex: 1 1 auto; display: flex; flex-direction: column; }
.game-frame:not(.game-frame-expanded) .game-slot { flex: 1 1 auto; display: flex; flex-direction: column; }
.game-frame:not(.game-frame-expanded) .game-slot > .ant-card { height: 100%; display: flex; flex-direction: column; }
.game-frame:not(.game-frame-expanded) .game-slot > .ant-card > .ant-card-head { flex: 0 0 auto; }
.game-frame:not(.game-frame-expanded) .game-slot .ant-card-body { flex: 1 1 auto; display: flex; flex-direction: column; }
.game-frame:not(.game-frame-expanded) .game-slot .ant-card-body > div { flex: 1 1 auto; display: flex; flex-direction: column; }
.game-frame:not(.game-frame-expanded) .game-actions { margin-top: auto; }
.game-frame:not(.game-frame-expanded) .game-slot .ant-card-body > div > *:nth-last-child(2) { margin-bottom: 16px !important; }
`;

/**
 * 游戏卡片（卡片外观 + 放大容器）
 *
 * 两个必须守住的点：
 *
 * 1) 正常态和放大态的 DOM 结构必须完全一致，只切样式。
 *    结构一变（或者改用 createPortal 把节点挂到 body 下），React 会把游戏组件卸载重建，
 *    正在跑的一局就没了 —— 计时器、画布上的球、打了一半的试次全部清零。
 *    所以顶栏始终渲染、只用 display 控制显隐，卡片也始终在同一个位置。
 *
 * 2) 卡片必须放在全屏层的「里面」，不能反过来把全屏层套在卡片里。
 *    .cloud-card 上有 filter: drop-shadow(...)，而 filter 会给它内部所有
 *    position:fixed 的后代建立包含块 —— 全屏层会因此被缩成卡片大小（实测过 570×103）。
 *    把卡片变成全屏层的子节点，全屏层的祖先链上就没有任何 filter / transform 了。
 */
function GameCard({ title, children }: { title: string; children: React.ReactNode }) {
  const [expanded, setExpanded] = useState(false);
  const abortVersion = useContext(HealingAbortContext);
  const seenRef = useRef(abortVersion);

  // 切换分类时收起。正常流程下全屏层盖住了标签栏点不到，这里是兜底。
  useEffect(() => {
    if (seenRef.current === abortVersion) return;
    seenRef.current = abortVersion;
    setExpanded(false);
  }, [abortVersion]);

  // Esc 退出全屏：注册在捕获阶段，抢在游戏自己的「Esc = 结束并回退」之前把事件拦下来
  useEffect(() => {
    if (!expanded) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      e.preventDefault();
      e.stopPropagation();
      e.stopImmediatePropagation();
      setExpanded(false);
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [expanded]);

  const api = useMemo<GameFullscreenApi>(() => ({
    expanded,
    setExpanded,
    toggle: () => setExpanded(v => !v),
  }), [expanded]);

  return (
    <GameFullscreenContext.Provider value={api}>
      <div
        className={expanded ? 'game-frame game-frame-expanded' : 'game-frame'}
        style={expanded ? {
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, zIndex: 1300,
          display: 'flex', flexDirection: 'column',
          padding: '0 24px 24px',
          background: 'linear-gradient(160deg, #ffffff 0%, #fff7f9 55%, #f6f3ff 100%)',
          overscrollBehavior: 'contain',
        } : { position: 'relative' }}
      >
        <div className="game-frame-head" style={{
          display: expanded ? 'flex' : 'none',
          alignItems: 'center', gap: 12, flex: '0 0 auto', height: 64,
          borderBottom: '1px solid rgba(255, 182, 193, 0.25)', marginBottom: 16,
        }}>
          <Button icon={<ArrowLeftOutlined />} onClick={() => setExpanded(false)}
            style={{ borderRadius: 20 }} className="game-back-btn">
            返回
          </Button>
          <Text strong style={{ fontSize: 16, color: '#5a4a6a' }}>{title}</Text>
          <Text type="secondary" style={{ fontSize: 12, marginLeft: 'auto' }}>
            按 Esc 也可以返回
          </Text>
        </div>
        {/*
          正常态的布局（display / flex / min-height）全部交给 GAME_LAYOUT_CSS，
          这里只在放大态写行内样式 —— 行内样式优先级最高，一旦在正常态也写上
          display:block、flex:0 0 auto，就会把等高链路打断（实测卡片被压扁到 299px）。
        */}
        <div className="game-frame-body" style={expanded ? {
          flex: '1 1 auto', display: 'flex', overflow: 'auto',
        } : undefined}>
          {/* margin:auto 让内容在放大态居中，内容比屏幕高时又不会被切掉顶部 */}
          <div className="game-slot" style={{
            width: '100%',
            ...(expanded ? { maxWidth: 920, margin: 'auto' } : null),
          }}>
            <Card
              title={title}
              className="cloud-card"
              styles={{
                // 放大态隐藏卡片自带的标题栏（只是 display，节点仍在，不会引起重建），
                // 标题由上面那条常驻顶栏承担
                header: { display: expanded ? 'none' : undefined },
                body: { padding: 24 },
              }}
            >
              {children}
            </Card>
          </div>
        </div>
      </div>
    </GameFullscreenContext.Provider>
  );
}

// ============================================================
// 0.5 反应测试公共件
//   useReactionClock  计时
//   useTrialEngine    试次序列与计分
//   useStaircase      阶梯难度
//   ReactionReport    成绩卡
//   recordGameResult  成绩落库
// ============================================================

/**
 * 关于精度（写在前面，因为所有反应类结果都要靠它成立）
 *
 * 浏览器测反应时有一段消不掉的固定偏差：
 *   - 屏幕刷新率量化：60Hz 下一档就是 16.7ms
 *   - 显示与输入设备延迟：USB 鼠标 1~8ms，触摸屏 20~50ms
 * 合计约 ±20~30ms，而且逐台设备不同。
 * 所以：同一个人前后的相对变化有意义，跨设备、跨人的绝对值比较没有意义。
 * 页面上要把这句话讲清楚，避免把娱乐数值当成能力判定。
 */
const RT_DISCLAIMER =
  '受屏幕刷新率与输入设备影响，反应时约有 ±20~30ms 的固定偏差，换台设备数值就会变；' +
  '关注自己的相对变化就好，不必和别人比绝对值。';

const RT_MIN_VALID = 100;    // 早于 100ms 的按键不可能是真反应，判为预判（抢跑）
const RT_GO_WINDOW = 1200;   // 绿色刺激的反应窗（ms），超时算漏接
const RT_NOGO_WINDOW = 1200; // 红色刺激的判定窗（ms），整个窗口内不按才算抑制成功

/**
 * 反应时打点器
 * 统一用 performance.now()（单调高精度时钟），而不是 Date.now()：
 * 后者会被系统对时和休眠影响，且只有毫秒整数精度，不适合测反应。
 */
function useReactionClock() {
  const markedRef = useRef<number | null>(null);
  const rafRef = useRef<number | null>(null);

  /** 立即打点（刺激已经在屏幕上时使用） */
  const arm = useCallback(() => {
    markedRef.current = performance.now();
    return markedRef.current;
  }, []);

  /**
   * 下一帧打点
   * setState 之后刺激要到下一帧才真正画到屏幕上，用 rAF 对齐更接近真实呈现时刻。
   */
  const armOnNextFrame = useCallback(() => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      markedRef.current = performance.now();
    });
  }, []);

  /**
   * 取事件时间戳
   * event.timeStamp 记录于事件产生那一刻、且与 performance.now() 同一时基，
   * 比在回调里再调一次 performance.now() 少一段事件派发排队误差。
   * 老浏览器给的是 epoch 毫秒，量级对不上就回退。
   */
  const eventTime = useCallback((e?: { timeStamp?: number }) => {
    const now = performance.now();
    const ts = e?.timeStamp;
    if (typeof ts === 'number' && ts > 0 && ts <= now + 4 && ts >= now - 60000) return ts;
    return now;
  }, []);

  /** 距打点已过去多少毫秒；尚未打点返回 null */
  const elapsed = useCallback((e?: { timeStamp?: number }) => {
    const t0 = markedRef.current;
    if (t0 === null) return null;
    return Math.round(eventTime(e) - t0);
  }, [eventTime]);

  const clear = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    markedRef.current = null;
  }, []);

  useEffect(() => clear, [clear]);

  return { arm, armOnNextFrame, elapsed, eventTime, clear };
}

// ----- 统计 -----

interface RtStats {
  n: number;     // 有效样本数
  mean: number;  // 平均反应时 (ms)
  best: number;  // 最快 (ms)
  sd: number;    // 标准差 (ms)：越小说明节奏越稳
  mad: number;   // 平均绝对偏差 (ms)
}

/** 反应时汇总 */
function summarizeRt(values: number[]): RtStats | null {
  if (values.length === 0) return null;
  const n = values.length;
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
  const mad = values.reduce((a, b) => a + Math.abs(b - mean), 0) / n;
  return {
    n,
    mean: Math.round(mean),
    best: Math.round(Math.min(...values)),
    sd: Math.round(Math.sqrt(variance)),
    mad: Math.round(mad),
  };
}

// ----- 试次引擎 -----

type TrialKind = 'go' | 'nogo';
/** hit 答对；miss 该按没按；wrong 不该按却按了；anticipate 抢跑 */
type TrialOutcome = 'hit' | 'miss' | 'wrong' | 'anticipate';

interface TrialPlanItem { index: number; kind: TrialKind; }
interface TrialLogItem extends TrialPlanItem { outcome: TrialOutcome; rt: number | null; }

/**
 * 生成 Go / No-Go 试次序列
 *  - 第一个试次固定为 Go：先建立「看到就按」的优势习惯，后面的 No-Go 才有抑制意义
 *  - No-Go 之间不相邻：避免形成「连续两次不用按」的节奏预期，那会让抑制变得免费
 */
function buildTrialPlan(total: number, nogoCount: number): TrialPlanItem[] {
  const kinds: TrialKind[] = Array.from({ length: total }, () => 'go');
  const slots = Array.from({ length: Math.max(0, total - 1) }, (_, i) => i + 1);
  for (let i = slots.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [slots[i], slots[j]] = [slots[j], slots[i]];
  }
  const used = new Set<number>();
  let placed = 0;
  for (const s of slots) {
    if (placed >= nogoCount) break;
    if (used.has(s - 1) || used.has(s + 1)) continue;
    kinds[s] = 'nogo';
    used.add(s);
    placed += 1;
  }
  return kinds.map((kind, index) => ({ index, kind }));
}

/** 逐试次计分汇总 */
function summarizeTrialLog(log: TrialLogItem[]) {
  const rts = log.filter(l => l.outcome === 'hit' && l.rt !== null).map(l => l.rt as number);
  const count = (o: TrialOutcome) => log.filter(l => l.outcome === o).length;
  const hit = count('hit');
  const anticipate = count('anticipate');
  // 抢跑会把该试次整轮作废，所以正确率只在有效试次上算 ——
  // 抢跑次数已经单独列出来了，再让它拉低正确率就是同一件事扣两次分。
  const valid = log.length - anticipate;
  return {
    done: log.length,
    valid,
    goTotal: log.filter(l => l.kind === 'go').length,
    nogoTotal: log.filter(l => l.kind === 'nogo').length,
    hit,
    miss: count('miss'),
    wrong: count('wrong'),
    anticipate,
    accuracy: valid > 0 ? hit / valid : 0,
    rt: summarizeRt(rts),
  };
}

type TrialStats = ReturnType<typeof summarizeTrialLog>;

/**
 * 试次引擎
 * 负责试次序列、当前进度、逐次计分与汇总。
 * 真值放在 ref 里、state 只用于渲染 —— 事件回调必须能同步读到当前是第几个试次。
 */
function useTrialEngine(total: number, nogoCount: number) {
  const [plan, setPlan] = useState<TrialPlanItem[]>([]);
  const [cursor, setCursor] = useState(0);
  const [log, setLog] = useState<TrialLogItem[]>([]);
  const planRef = useRef<TrialPlanItem[]>([]);
  const cursorRef = useRef(0);
  const logRef = useRef<TrialLogItem[]>([]);

  const start = useCallback((): TrialPlanItem[] => {
    const next = buildTrialPlan(total, nogoCount);
    planRef.current = next;
    cursorRef.current = 0;
    logRef.current = [];
    setPlan(next);
    setCursor(0);
    setLog([]);
    return next;
  }, [total, nogoCount]);

  const current = useCallback((): TrialPlanItem | null => planRef.current[cursorRef.current] ?? null, []);

  /** 记录一次结果并前进，返回是否已跑完以及最新的完整记录 */
  const submit = useCallback((outcome: TrialOutcome, rt: number | null) => {
    const item = planRef.current[cursorRef.current];
    if (!item) return { finished: true, log: logRef.current };
    logRef.current = [...logRef.current, { ...item, outcome, rt }];
    cursorRef.current += 1;
    setLog(logRef.current);
    setCursor(cursorRef.current);
    return { finished: cursorRef.current >= planRef.current.length, log: logRef.current };
  }, []);

  const reset = useCallback(() => {
    planRef.current = [];
    cursorRef.current = 0;
    logRef.current = [];
    setPlan([]);
    setCursor(0);
    setLog([]);
  }, []);

  const stats: TrialStats = useMemo(() => summarizeTrialLog(log), [log]);

  return { plan, cursor, log, stats, current, start, submit, reset };
}

// ----- 阶梯难度 -----

/**
 * 阶梯难度
 * 表现好升一级、明显吃力降一级，把任务稳在「有点挑战但做得到」的区间。
 * 这也是心理测评里避免挫败感的常规做法：难度跟着人走，而不是让人去撞难度。
 */
function useStaircase(start: number, min: number, max: number) {
  const [level, setLevel] = useState(start);
  const levelRef = useRef(start);

  const reset = useCallback(() => {
    levelRef.current = start;
    setLevel(start);
  }, [start]);

  const adjust = useCallback((accuracy: number) => {
    let next = levelRef.current;
    if (accuracy >= 0.99) next += 1;
    else if (accuracy < 0.6) next -= 1;
    next = Math.max(min, Math.min(max, next));
    levelRef.current = next;
    setLevel(next);
    return next;
  }, [min, max]);

  return { level, adjust, reset };
}

// ----- 成绩卡 -----

function ReactionReport({ title, rows, note, tone = '#722ed1' }: {
  title: string;
  rows: Array<{ label: string; value: string }>;
  note?: string;
  tone?: string;
}) {
  return (
    <div style={{
      background: '#faf7ff', border: '1px solid #efe4ff', borderRadius: 14,
      padding: '14px 16px', marginBottom: 12,
    }}>
      <div style={{ fontWeight: 600, color: tone, marginBottom: 10, textAlign: 'center' }}>{title}</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(84px, 1fr))', gap: 10 }}>
        {rows.map(r => (
          <div key={r.label} style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 12, color: '#999' }}>{r.label}</div>
            <div style={{ fontSize: 20, fontWeight: 700, color: '#333' }}>{r.value}</div>
          </div>
        ))}
      </div>
      {note && (
        <div style={{ marginTop: 10, fontSize: 12, color: '#999', lineHeight: 1.7 }}>{note}</div>
      )}
    </div>
  );
}

// ----- 成绩落库 -----

/**
 * 把一局成绩写进疗愈记录
 * 复用现成的 HealingSession 表（metadata 是 JSON 字符串字段）与 healingApi，
 * 不需要改 Prisma schema。写失败就静默降级：页面照常出成绩，只是不进历史。
 * type 统一加 'game:' 前缀，方便后端统计时把小游戏和冥想 / 呼吸区分开。
 */
function recordGameResult(taskId: string, title: string, durationSec: number, metrics: Record<string, any>) {
  const seconds = Math.max(1, Math.round(durationSec));
  healingApi.createSession({ type: `game:${taskId}`, title, duration: seconds })
    .then((res: any) => {
      const id = res?.data?.id;
      if (!id) return;
      return healingApi.completeSession(id, { duration: seconds, metadata: metrics });
    })
    .catch(() => { /* 记录失败不影响游戏体验 */ });
}

/**
 * 个人最好成绩（localStorage）
 * 只存一个数，用来给玩家一个自己跟自己的参照，不做跨人比较。
 */
function usePersonalBest(taskId: string, higherIsBetter: boolean) {
  const key = `healing_best_${taskId}`;
  const [best, setBest] = useState<number | null>(() => {
    const raw = localStorage.getItem(key);
    const v = raw === null ? NaN : Number(raw);
    return Number.isFinite(v) ? v : null;
  });

  const submit = useCallback((value: number) => {
    setBest(prev => {
      const better = prev === null || (higherIsBetter ? value > prev : value < prev);
      if (!better) return prev;
      try { localStorage.setItem(key, String(value)); } catch { /* 隐私模式写不进去，忽略 */ }
      return value;
    });
  }, [key, higherIsBetter]);

  return { best, submit };
}

/** 游戏内的即时反馈文案：显示 800ms 后自动消失 */
function useFlash() {
  const [feedback, setFeedback] = useState<{ text: string; tone: string } | null>(null);
  const timerRef = useRef<any>(null);

  const flash = useCallback((text: string, tone = '#52c41a') => {
    setFeedback({ text, tone });
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => setFeedback(null), 800);
  }, []);

  const clear = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    setFeedback(null);
  }, []);

  useEffect(() => () => { if (timerRef.current) clearTimeout(timerRef.current); }, []);

  return { feedback, flash, clear };
}

// ============================================================
// 1. 舒尔特方格
// ============================================================
const SCHULTE_SIZE = 5;
const SCHULTE_TOTAL = SCHULTE_SIZE * SCHULTE_SIZE;

function SchulteGrid() {
  const [grid, setGrid] = useState<number[]>([]);
  const [next, setNext] = useState(1);
  const [timer, setTimer] = useState(0);
  const [running, setRunning] = useState(false);
  const [done, setDone] = useState(false);
  const [best, setBest] = useState<number | null>(null);
  const [ended, flashEnded, clearEnded] = useEndedFlag();
  const isExpanded = useIsExpanded();
  const run = useInterruptible();
  const elapsedRef = useRef(0); // 与显示同步的已用时，避免闭包读到旧的 state

  const shuffle = useCallback(() => {
    const arr = Array.from({ length: SCHULTE_TOTAL }, (_, i) => i + 1);
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    const token = run.begin();
    elapsedRef.current = 0;
    setGrid(arr);
    setNext(1);
    setTimer(0);
    setDone(false);
    setRunning(true);
    clearEnded();
    run.trackTimer(setInterval(() => {
      if (!run.isActive(token)) return;
      elapsedRef.current = Math.round((elapsedRef.current + 0.1) * 10) / 10;
      setTimer(elapsedRef.current);
    }, 100));
  }, [run, clearEnded]);

  /** 结束并回退：中断计时，方格停止高亮并回到第 1 格（历史最佳成绩保留） */
  const endGame = useCallback(() => {
    if (!running && !done) return;
    run.interrupt();
    setRunning(false);
    setDone(false);
    setNext(1);
    elapsedRef.current = 0;
    setTimer(0);
    flashEnded();
  }, [run, running, done, flashEnded]);

  const handleClick = (num: number) => {
    if (!running || done) return; // 结束后点击不再推进
    if (num !== next) return;
    if (next === SCHULTE_TOTAL) {
      run.interrupt();
      setDone(true);
      setRunning(false);
      const finalTime = elapsedRef.current;
      if (best === null || finalTime < best) setBest(finalTime);
    } else {
      setNext(n => n + 1);
    }
  };

  useEscapeToEnd(endGame, running);
  useAbortOnTabChange(endGame, running);

  const cellColor = (num: number) => {
    if (!running) return '#f0f0f0';
    if (num < next) return '#b7eb8f';
    if (num === next) return '#ffe58f';
    return '#f0f0f0';
  };

  return (
    <div style={{ textAlign: 'center' }}>
      <Text type="secondary" style={{ display: 'block', marginBottom: 12, fontSize: 13 }}>
        按顺序点击 1~25，越快越好！训练视觉搜索和注意力广度。
      </Text>
      <div style={{ display: 'flex', justifyContent: 'center', gap: 40, marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 12, color: '#999', marginBottom: 4 }}>用时</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: '#722ed1' }}>{timer.toFixed(1)} 秒</div>
        </div>
        {best !== null && (
          <div>
            <div style={{ fontSize: 12, color: '#999', marginBottom: 4 }}>最佳</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: '#52c41a' }}>{best.toFixed(1)} 秒</div>
          </div>
        )}
      </div>
      <div style={{
        display: 'grid', gridTemplateColumns: `repeat(${SCHULTE_SIZE}, 1fr)`,
        gap: isExpanded ? 8 : 6, maxWidth: isExpanded ? 440 : 320, margin: '0 auto 16px',
      }}>
        {grid.map((num, i) => (
          <div key={i} onClick={() => handleClick(num)} style={{
            width: isExpanded ? 74 : 56, height: isExpanded ? 74 : 56,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: cellColor(num), borderRadius: 10, cursor: 'pointer',
            fontSize: isExpanded ? 24 : 18, fontWeight: 600, color: num < next ? '#52c41a' : '#333',
            border: num === next && running ? '2px solid #faad14' : '1px solid #e8e8e8',
            transition: 'all 0.15s', userSelect: 'none',
          }}>{num}</div>
        ))}
      </div>
      {done && <div style={{ marginBottom: 4 }}><Tag color="green" style={{ fontSize: 14, padding: '4px 12px' }}>完成！用时 {timer.toFixed(1)} 秒</Tag></div>}
      <GameActions onEnd={endGame} endDisabled={!running && !done} ended={ended}>
        <Button type="primary" onClick={shuffle} style={{ borderRadius: 20 }}>
          {done ? '再来一局' : running ? '重新开始' : '开始'}
        </Button>
      </GameActions>
    </div>
  );
}

// ============================================================
// 2. 色词冲突 (Stroop)
// ============================================================
const STROOP_COLORS = [
  { name: '红', hex: '#f5222d' },
  { name: '蓝', hex: '#1890ff' },
  { name: '绿', hex: '#52c41a' },
  { name: '黄', hex: '#faad14' },
  { name: '紫', hex: '#722ed1' },
];
const STROOP_ROUND_SEC = 30;

function StroopTest() {
  const [word, setWord] = useState('');
  const [color, setColor] = useState('#722ed1');
  const [score, setScore] = useState(0);
  const [total, setTotal] = useState(0);
  const [timeLeft, setTimeLeft] = useState(STROOP_ROUND_SEC);
  const [playing, setPlaying] = useState(false);
  const [feedback, setFeedback] = useState<'correct' | 'wrong' | null>(null);
  const [ended, flashEnded, clearEnded] = useEndedFlag();
  const run = useInterruptible();
  const timeRef = useRef(STROOP_ROUND_SEC);

  const newRound = useCallback(() => {
    const w = STROOP_COLORS[Math.floor(Math.random() * STROOP_COLORS.length)];
    let c = STROOP_COLORS[Math.floor(Math.random() * STROOP_COLORS.length)];
    // 70% 概率字色不一致
    if (Math.random() < 0.7) {
      while (c.name === w.name) c = STROOP_COLORS[Math.floor(Math.random() * STROOP_COLORS.length)];
    }
    setWord(w.name);
    setColor(c.hex);
    setFeedback(null);
  }, []);

  const start = useCallback(() => {
    const token = run.begin();
    timeRef.current = STROOP_ROUND_SEC;
    setScore(0);
    setTotal(0);
    setTimeLeft(STROOP_ROUND_SEC);
    setPlaying(true);
    setFeedback(null);
    clearEnded();
    newRound();
    run.trackTimer(setInterval(() => {
      if (!run.isActive(token)) return;
      timeRef.current = Math.max(0, Math.round((timeRef.current - 0.1) * 10) / 10);
      setTimeLeft(timeRef.current);
      if (timeRef.current <= 0) {
        run.interrupt(); // 倒计时自然结束：收尾（保留本局成绩）
        setPlaying(false);
      }
    }, 100));
  }, [run, newRound, clearEnded]);

  const answer = (colorName: string) => {
    if (!playing) return;
    const correctColor = STROOP_COLORS.find(c => c.hex === color);
    setTotal(t => t + 1);
    if (correctColor?.name === colorName) {
      setScore(s => s + 1);
      setFeedback('correct');
    } else {
      setFeedback('wrong');
    }
    const token = run.token();
    run.trackTimer(setTimeout(() => {
      if (!run.isActive(token)) return; // 已结束：不再出下一题
      newRound();
    }, 300));
  };

  /** 结束并回退：中断倒计时与待出的题目，回到「未开始」状态（不保留本局成绩） */
  const endGame = useCallback(() => {
    if (!playing && total === 0) return;
    run.interrupt();
    timeRef.current = STROOP_ROUND_SEC;
    setPlaying(false);
    setScore(0);
    setTotal(0);
    setTimeLeft(STROOP_ROUND_SEC);
    setWord('');
    setColor('#722ed1');
    setFeedback(null);
    flashEnded();
  }, [run, playing, total, flashEnded]);

  useEscapeToEnd(endGame, playing);
  useAbortOnTabChange(endGame, playing);

  const idle = !playing && total === 0;

  return (
    <div style={{ textAlign: 'center' }}>
      <Text type="secondary" style={{ display: 'block', marginBottom: 12, fontSize: 13 }}>
        选择字的<strong>颜色</strong>（不是字义）！训练抑制控制和认知灵活性。
      </Text>
      <div style={{ display: 'flex', justifyContent: 'center', gap: 40, marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 12, color: '#999', marginBottom: 4 }}>得分</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: '#722ed1' }}>{score}</div>
        </div>
        <div>
          <div style={{ fontSize: 12, color: '#999', marginBottom: 4 }}>剩余</div>
          <div style={{ fontSize: 28, fontWeight: 700, color: timeLeft < 10 ? '#f5222d' : '#1890ff' }}>{timeLeft.toFixed(1)} 秒</div>
        </div>
      </div>
      {playing && (
        <>
          <div style={{ fontSize: 52, fontWeight: 800, color, marginBottom: 24, height: 70, lineHeight: '70px' }}>{word}</div>
          <Space wrap size={12} style={{ justifyContent: 'center' }}>
            {STROOP_COLORS.map(c => (
              <Button key={c.name} onClick={() => answer(c.name)}
                style={{ width: 72, height: 44, borderRadius: 22, fontSize: 16, fontWeight: 600, color: '#fff', background: c.hex, border: 'none' }}>
                {c.name}
              </Button>
            ))}
          </Space>
          {feedback && (
            <div style={{ marginTop: 12, fontSize: 18, color: feedback === 'correct' ? '#52c41a' : '#f5222d', fontWeight: 600 }}>
              {feedback === 'correct' ? '✓ 正确！' : '✗ 看颜色不是字义！'}
            </div>
          )}
        </>
      )}
      {!playing && total > 0 && (
        <div>
          <Tag color={score / total > 0.7 ? 'green' : score / total > 0.4 ? 'orange' : 'red'}
            style={{ fontSize: 16, padding: '6px 16px' }}>
            正确率 {Math.round(score / total * 100)}%（{score}/{total}）
          </Tag>
        </div>
      )}
      {idle && (
        <div style={{ fontSize: 13, color: '#bbb' }}>30 秒内尽可能多答对</div>
      )}
      <GameActions onEnd={endGame} endDisabled={idle} ended={ended}
        tip="立即中断倒计时并回退到未开始状态（快捷键 Esc）">
        {!playing && (
          <Button type="primary" onClick={start} style={{ borderRadius: 20 }}>
            {total > 0 ? '再来一局' : '开始挑战（30秒）'}
          </Button>
        )}
      </GameActions>
    </div>
  );
}

// ============================================================
// 3. 气球呼吸训练
// ============================================================
type BreathPhase = 'idle' | 'inhale' | 'hold' | 'exhale';

const INHALE_SEC = 4, HOLD_SEC = 4, EXHALE_SEC = 6;
const BALLOON_MAX = 2.2;

function BalloonBreathing() {
  const [phase, setPhase] = useState<BreathPhase>('idle');
  const [phaseLeft, setPhaseLeft] = useState(0);
  const [scale, setScale] = useState(1);
  const [cycles, setCycles] = useState(0);
  const [ended, flashEnded, clearEnded] = useEndedFlag();
  const run = useInterruptible();
  const phaseRef = useRef<BreathPhase>('idle');

  /** 推进单个呼吸阶段；阶段结束后回调 onDone，token 失效则立即停住 */
  const runPhase = useCallback((token: number, name: BreathPhase, seconds: number, onDone: () => void) => {
    phaseRef.current = name;
    setPhase(name);
    let t = 0;
    const apply = (elapsed: number) => {
      if (name === 'inhale') setScale(1 + (BALLOON_MAX - 1) * Math.min(elapsed / INHALE_SEC, 1));
      else if (name === 'hold') setScale(BALLOON_MAX);
      else setScale(BALLOON_MAX - (BALLOON_MAX - 1) * Math.min(elapsed / EXHALE_SEC, 1));
      setPhaseLeft(Math.max(0, seconds - elapsed));
    };
    apply(0);
    const id = setInterval(() => {
      if (!run.isActive(token)) return;
      t += 0.05;
      if (t >= seconds) {
        clearInterval(id);
        setPhaseLeft(0);
        onDone();
        return;
      }
      apply(t);
    }, 50);
    run.trackTimer(id);
  }, [run]);

  /** 吸气 → 屏息 → 呼气，循环往复 */
  const runCycle = useCallback((token: number) => {
    runPhase(token, 'inhale', INHALE_SEC, () => {
      runPhase(token, 'hold', HOLD_SEC, () => {
        runPhase(token, 'exhale', EXHALE_SEC, () => {
          if (!run.isActive(token)) return;
          setCycles(c => c + 1);
          runCycle(token);
        });
      });
    });
  }, [run, runPhase]);

  const startBreathing = useCallback(() => {
    const token = run.begin();
    setCycles(0);
    setScale(1);
    setPhaseLeft(0);
    clearEnded();
    runCycle(token);
  }, [run, runCycle, clearEnded]);

  /** 结束并回退：中断呼吸循环，气球回到初始大小、轮次清零 */
  const endGame = useCallback(() => {
    if (phaseRef.current === 'idle') return;
    run.interrupt();
    phaseRef.current = 'idle';
    setPhase('idle');
    setScale(1);
    setPhaseLeft(0);
    setCycles(0);
    flashEnded();
  }, [run, flashEnded]);

  useEscapeToEnd(endGame, phase !== 'idle');
  useAbortOnTabChange(endGame, phase !== 'idle');

  const phaseText = { idle: '点击开始', inhale: '吸气...', hold: '屏住...', exhale: '呼气...' }[phase];
  const phaseColor = { idle: '#bfbfbf', inhale: '#69c0ff', hold: '#ffc069', exhale: '#95de64' }[phase];

  return (
    <div style={{ textAlign: 'center' }}>
      <Text type="secondary" style={{ display: 'block', marginBottom: 12, fontSize: 13 }}>
        跟随气球节奏呼吸：吸气 4 秒 → 屏息 4 秒 → 呼气 6 秒。缓解焦虑，激活副交感神经。
      </Text>
      <div style={{ position: 'relative', height: 280, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{
          width: 100, height: 120, borderRadius: '50% 50% 50% 50% / 60% 60% 40% 40%',
          background: `radial-gradient(ellipse at 35% 35%, #ffd6e7, #ff85a2)`,
          transform: `scale(${scale})`,
          transition: 'transform 0.05s linear',
          boxShadow: `0 0 ${20 * scale}px rgba(255,133,162,0.4)`,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <div style={{ fontSize: 28 }}>🎈</div>
        </div>
        {/* 气球线 */}
        <div style={{
          position: 'absolute', bottom: 40, width: 2, height: 40 * scale,
          background: '#ccc', left: '50%', transform: 'translateX(-50%)',
        }} />
      </div>
      <div style={{ fontSize: 22, fontWeight: 700, color: phaseColor, marginBottom: 8 }}>
        {phaseText}
        {phase !== 'idle' && <span style={{ fontSize: 14, color: '#999', marginLeft: 8 }}>{phaseLeft.toFixed(1)}s</span>}
      </div>
      <div>
        <Tag color="purple" style={{ fontSize: 14 }}>已完成 {cycles} 轮</Tag>
      </div>
      <GameActions onEnd={endGame} endDisabled={phase === 'idle'} ended={ended}
        tip="立即中断呼吸循环并回退到初始状态（快捷键 Esc）">
        {phase === 'idle' && (
          <Button type="primary" onClick={startBreathing} style={{ borderRadius: 20 }}>开始呼吸</Button>
        )}
      </GameActions>
    </div>
  );
}

// ============================================================
// 4. 冥想音频 (Web Audio API 实时合成)
// ============================================================
/**
 * 冥想自然音 —— 真实录音音源
 *
 * url 为 CDN 直链（Mixkit Free Sound Effects License：可免费商用、无需署名），
 * 用 <audio> 流式播放，音质是真实环境录音，不再是合成的白噪音；
 * 若网络不可达，会自动回退到下方 Web Audio 实时合成音（type 字段）。
 */
const MEDITATION_TRACKS = [
  {
    title: '森林清晨', emoji: '\u{1F332}', desc: '鸟鸣与微风', duration: '3:29', color: '#52c41a', type: 'forest' as const,
    url: 'https://assets.mixkit.co/active_storage/sfx/2472/2472-preview.mp3',
  },
  {
    title: '海浪轻拍', emoji: '\u{1F30A}', desc: '真实的潮汐拍岸', duration: '2:31', color: '#1890ff', type: 'ocean' as const,
    url: 'https://assets.mixkit.co/active_storage/sfx/1189/1189-preview.mp3',
  },
  {
    title: '雨打窗棂', emoji: '\u{1F327}\u{FE0F}', desc: '连绵雨声', duration: '0:57', color: '#722ed1', type: 'rain' as const,
    url: 'https://assets.mixkit.co/active_storage/sfx/2394/2394-preview.mp3',
  },
  {
    title: '篝火噼啪', emoji: '\u{1F525}', desc: '木柴燃烧与爆裂', duration: '0:24', color: '#fa8c16', type: 'fire' as const,
    url: 'https://assets.mixkit.co/active_storage/sfx/1330/1330-preview.mp3',
  },
  {
    title: '溪流潺潺', emoji: '\u{1F4A7}', desc: '山涧流水', duration: '3:00', color: '#13c2c2', type: 'stream' as const,
    url: 'https://assets.mixkit.co/active_storage/sfx/2454/2454-preview.mp3',
  },
  {
    title: '星空静谧', emoji: '\u{1F319}', desc: '夜森林虫鸣', duration: '1:24', color: '#2f54eb', type: 'night' as const,
    url: 'https://assets.mixkit.co/active_storage/sfx/2414/2414-preview.mp3',
  },
];

type MeditationTrack = typeof MEDITATION_TRACKS[number];
type SoundType = MeditationTrack['type'];
type AudioSource = 'online' | 'synth';

/** 合成音（回退方案）没有真实时长，按一次疗愈 5 分钟计算 */
const MEDITATION_DURATION = 300;

/** 秒 → m:ss */
function fmtTime(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds || 0));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

/** 创建噪声缓冲区 */
function createNoiseBuffer(ctx: AudioContext, type: 'white' | 'pink' | 'brown'): AudioBuffer {
  const length = ctx.sampleRate * 2;
  const buffer = ctx.createBuffer(1, length, ctx.sampleRate);
  const data = buffer.getChannelData(0);
  let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
  for (let i = 0; i < length; i++) {
    const white = Math.random() * 2 - 1;
    if (type === 'white') {
      data[i] = white;
    } else if (type === 'pink') {
      b0 = 0.99886 * b0 + white * 0.0555179;
      b1 = 0.99332 * b1 + white * 0.0750759;
      b2 = 0.96900 * b2 + white * 0.1538520;
      b3 = 0.86650 * b3 + white * 0.3104856;
      b4 = 0.55000 * b4 + white * 0.5329522;
      b5 = -0.7616 * b5 - white * 0.0168980;
      data[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362) * 0.11;
      b6 = white * 0.115926;
    } else {
      data[i] = (b0 = (b0 + (0.02 * white)) / 1.02);
    }
  }
  return buffer;
}

function MeditationAudio() {
  const [playing, setPlaying] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(0.6);
  const [source, setSource] = useState<AudioSource>('online');
  const [ended, flashEnded, clearEnded] = useEndedFlag();
  const run = useInterruptible();
  const mediaRef = useRef<HTMLAudioElement | null>(null);
  const synthRef = useRef<{ ctx: AudioContext; nodes: AudioNode[]; intervals: number[]; master: GainNode } | null>(null);
  const trackRef = useRef<MeditationTrack | null>(null);
  const sourceRef = useRef<AudioSource>('online');
  const progressRef = useRef(0);
  const volumeRef = useRef(volume);

  /** 关闭合成音（仅在网络音源不可用时才会用到） */
  const stopSynth = useCallback(() => {
    if (synthRef.current) {
      synthRef.current.intervals.forEach(id => clearInterval(id));
      synthRef.current.nodes.forEach(n => { try { n.disconnect(); } catch {} });
      try { synthRef.current.ctx.close(); } catch { /* 已关闭则忽略 */ }
      synthRef.current = null;
    }
  }, []);

  /** 彻底停止发声：卸载在线音频 + 关闭合成音 */
  const stopAll = useCallback(() => {
    const el = mediaRef.current;
    if (el) {
      el.pause();
      el.removeAttribute('src');
      try { el.load(); } catch { /* 忽略 */ } // 断开下载，避免结束后继续占用带宽
    }
    stopSynth();
  }, [stopSynth]);

  const startSound = useCallback((type: SoundType) => {
    const ctx = new AudioContext();
    const master = ctx.createGain();
    master.gain.value = volumeRef.current;
    master.connect(ctx.destination);
    const nodes: AudioNode[] = [master];
    const intervals: number[] = [];

    if (type === 'ocean') {
      const noise = ctx.createBufferSource();
      noise.buffer = createNoiseBuffer(ctx, 'brown');
      noise.loop = true;
      const filter = ctx.createBiquadFilter();
      filter.type = 'lowpass';
      filter.frequency.value = 400;
      const lfo = ctx.createOscillator();
      lfo.frequency.value = 0.08;
      const lfoGain = ctx.createGain();
      lfoGain.gain.value = 0.15;
      lfo.connect(lfoGain);
      lfoGain.connect(master.gain);
      lfo.start();
      noise.connect(filter);
      filter.connect(master);
      noise.start();
      nodes.push(noise, filter, lfo, lfoGain);
    } else if (type === 'rain') {
      const noise = ctx.createBufferSource();
      noise.buffer = createNoiseBuffer(ctx, 'pink');
      noise.loop = true;
      const bp = ctx.createBiquadFilter();
      bp.type = 'bandpass';
      bp.frequency.value = 3000;
      bp.Q.value = 0.5;
      const hp = ctx.createBiquadFilter();
      hp.type = 'highpass';
      hp.frequency.value = 1000;
      noise.connect(bp);
      bp.connect(hp);
      hp.connect(master);
      noise.start();
      nodes.push(noise, bp, hp);
    } else if (type === 'fire') {
      const noise = ctx.createBufferSource();
      noise.buffer = createNoiseBuffer(ctx, 'brown');
      noise.loop = true;
      const bp = ctx.createBiquadFilter();
      bp.type = 'bandpass';
      bp.frequency.value = 800;
      bp.Q.value = 1;
      const crackleGain = ctx.createGain();
      crackleGain.gain.value = 0.5;
      const cid = window.setInterval(() => {
        if (Math.random() < 0.3 && ctx.state === 'running') {
          crackleGain.gain.setValueAtTime(0.8, ctx.currentTime);
          crackleGain.gain.exponentialRampToValueAtTime(0.3, ctx.currentTime + 0.1);
        }
      }, 200);
      intervals.push(cid);
      noise.connect(bp);
      bp.connect(crackleGain);
      crackleGain.connect(master);
      noise.start();
      nodes.push(noise, bp, crackleGain);
    } else if (type === 'stream') {
      const noise = ctx.createBufferSource();
      noise.buffer = createNoiseBuffer(ctx, 'white');
      noise.loop = true;
      const hp = ctx.createBiquadFilter();
      hp.type = 'highpass';
      hp.frequency.value = 2000;
      const lfo = ctx.createOscillator();
      lfo.frequency.value = 6;
      const lfoGain = ctx.createGain();
      lfoGain.gain.value = 0.1;
      lfo.connect(lfoGain);
      lfoGain.connect(master.gain);
      lfo.start();
      noise.connect(hp);
      hp.connect(master);
      noise.start();
      nodes.push(noise, hp, lfo, lfoGain);
    } else if (type === 'forest') {
      const noise = ctx.createBufferSource();
      noise.buffer = createNoiseBuffer(ctx, 'pink');
      noise.loop = true;
      const lp = ctx.createBiquadFilter();
      lp.type = 'lowpass';
      lp.frequency.value = 2000;
      noise.connect(lp);
      lp.connect(master);
      noise.start();
      nodes.push(noise, lp);
      const bid = window.setInterval(() => {
        if (Math.random() < 0.4 && ctx.state === 'running') {
          const osc = ctx.createOscillator();
          const bg = ctx.createGain();
          osc.frequency.value = 2000 + Math.random() * 3000;
          osc.type = 'sine';
          bg.gain.setValueAtTime(0, ctx.currentTime);
          bg.gain.linearRampToValueAtTime(0.06, ctx.currentTime + 0.05);
          bg.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.3);
          osc.connect(bg);
          bg.connect(ctx.destination);
          osc.start();
          osc.stop(ctx.currentTime + 0.3);
        }
      }, 2000);
      intervals.push(bid);
    } else if (type === 'night') {
      const noise = ctx.createBufferSource();
      noise.buffer = createNoiseBuffer(ctx, 'brown');
      noise.loop = true;
      const lp = ctx.createBiquadFilter();
      lp.type = 'lowpass';
      lp.frequency.value = 200;
      const g = ctx.createGain();
      g.gain.value = 0.15;
      noise.connect(lp);
      lp.connect(g);
      g.connect(master);
      noise.start();
      nodes.push(noise, lp, g);
      const drone = ctx.createOscillator();
      drone.frequency.value = 80;
      drone.type = 'sine';
      const dg = ctx.createGain();
      dg.gain.value = 0.04;
      drone.connect(dg);
      dg.connect(master);
      drone.start();
      nodes.push(drone, dg);
    }

    synthRef.current = { ctx, nodes, intervals, master };
  }, []);

  /** 网络音源不可用时，回退到本地合成音（并如实提示用户） */
  const fallbackToSynth = useCallback((token: number) => {
    if (!run.isActive(token) || sourceRef.current !== 'online') return;
    const track = trackRef.current;
    if (!track) return;
    const el = mediaRef.current;
    if (el) {
      el.pause();
      el.removeAttribute('src');
      try { el.load(); } catch { /* 忽略 */ }
    }
    sourceRef.current = 'synth';
    setSource('synth');
    startSound(track.type);
    progressRef.current = 0;
    setProgress(0);
    setDuration(MEDITATION_DURATION);
    run.trackTimer(setInterval(() => {
      if (!run.isActive(token)) return;
      progressRef.current += 1;
      setProgress(progressRef.current);
      if (progressRef.current >= MEDITATION_DURATION) {
        run.interrupt();
        progressRef.current = 0;
        setProgress(0);
        setPlaying(null);
      }
    }, 1000));
  }, [run, startSound]);

  /** 结束并回退：停止发声（在线音频 / 合成音）、清空进度，回到未播放状态 */
  const endPlay = useCallback(() => {
    if (!playing) return;
    run.interrupt(); // 触发 stopAll：暂停并卸载音频、关闭合成器、清进度定时器
    progressRef.current = 0;
    setProgress(0);
    setDuration(0);
    setPlaying(null);
    flashEnded();
  }, [run, playing, flashEnded]);

  const play = useCallback((track: MeditationTrack) => {
    // 再次点击正在播放的卡片 = 结束并回退
    if (playing === track.title) {
      endPlay();
      return;
    }
    const token = run.begin(); // 中断上一段播放（卸载音频 + 清定时器）
    progressRef.current = 0;
    sourceRef.current = 'online';
    trackRef.current = track;
    setProgress(0);
    setDuration(0);
    setSource('online');
    setPlaying(track.title);
    clearEnded();
    run.addCleanup(stopAll);

    const el = mediaRef.current;
    if (!el) return;
    el.src = track.url;
    el.loop = true;
    el.volume = volumeRef.current;
    el.currentTime = 0;
    const started = el.play();
    if (started && typeof started.catch === 'function') {
      started.catch(() => fallbackToSynth(token)); // 加载/解码失败 → 合成音兜底
    }
  }, [playing, run, stopAll, endPlay, clearEnded, fallbackToSynth]);

  /** 音量同时作用于在线音频与合成音 */
  useEffect(() => {
    volumeRef.current = volume;
    if (mediaRef.current) mediaRef.current.volume = volume;
    if (synthRef.current) synthRef.current.master.gain.value = volume;
  }, [volume]);

  useEscapeToEnd(endPlay, playing !== null);
  useAbortOnTabChange(endPlay, playing !== null);

  const mediaDuration = duration > 0 ? duration : 0;
  const pct = playing && mediaDuration > 0 ? Math.min(100, (progress / mediaDuration) * 100) : 0;

  return (
    <div>
      <Text type="secondary" style={{ display: 'block', marginBottom: 16, fontSize: 13, textAlign: 'center' }}>
        选一个自然声音，闭上眼睛，让声音带你进入平静。
      </Text>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12 }}>
        {MEDITATION_TRACKS.map(track => {
          const isActive = playing === track.title;
          return (
            <Card key={track.title} hoverable onClick={() => play(track)}
              styles={{ body: { padding: '16px 12px', textAlign: 'center' } }}
              style={{
                borderRadius: 16, border: isActive ? `2px solid ${track.color}` : '1px solid #f0f0f0',
                background: isActive ? `${track.color}08` : '#fff',
                transition: 'all 0.3s',
              }}>
              <div style={{ fontSize: 36, marginBottom: 8 }}>{track.emoji}</div>
              <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 4 }}>{track.title}</div>
              <div style={{ fontSize: 12, color: '#999', marginBottom: 8 }}>{track.desc}</div>
              {isActive && (
                <div>
                  <div style={{ height: 4, background: '#f0f0f0', borderRadius: 2, overflow: 'hidden', marginBottom: 6 }}>
                    <div style={{ width: `${pct}%`, height: '100%', background: track.color, borderRadius: 2, transition: 'width 0.3s linear' }} />
                  </div>
                  <Text style={{ fontSize: 11, color: track.color }}>
                    {fmtTime(progress)} / {mediaDuration > 0 ? fmtTime(mediaDuration) : '--:--'}
                  </Text>
                </div>
              )}
              {!isActive && <Tag style={{ fontSize: 11 }}>{track.duration}</Tag>}
            </Card>
          );
        })}
      </div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10, marginTop: 14 }}>
        <Text style={{ fontSize: 12, color: '#999' }}>音量</Text>
        <Slider min={0} max={100} value={Math.round(volume * 100)}
          onChange={(v: number) => setVolume(v / 100)}
          style={{ width: 140, margin: 0 }} tooltip={{ formatter: (v) => `${v}%` }} />
      </div>
      <div style={{
        textAlign: 'center', marginTop: 12, padding: 12, borderRadius: 12,
        background: source === 'synth' ? '#fff7e6' : '#f6ffed',
      }}>
        <Text style={{ fontSize: 12, color: source === 'synth' ? '#fa8c16' : '#52c41a' }}>
          {source === 'synth'
            ? '\u26A0\uFE0F 在线音源加载失败，已切换为本地合成音（检查网络后重试）'
            : `\u{1F3A7} 真实环境录音（Mixkit 免费音源），声音会循环播放；建议佩戴耳机体验更佳。`}
        </Text>
      </div>
      {/* 真实录音播放器：结束后会被卸载 src，避免继续占用带宽 */}
      <audio
        ref={mediaRef}
        preload="none"
        onTimeUpdate={(e) => {
          if (sourceRef.current !== 'online') return;
          const t = e.currentTarget.currentTime;
          progressRef.current = t;
          setProgress(t);
        }}
        onLoadedMetadata={(e) => {
          if (sourceRef.current !== 'online') return;
          const d = e.currentTarget.duration;
          if (Number.isFinite(d) && d > 0) setDuration(d);
        }}
        // VBR MP3 的时长有时要扫描完整文件才确定，这里补一次捕获
        onDurationChange={(e) => {
          if (sourceRef.current !== 'online') return;
          const d = e.currentTarget.duration;
          if (Number.isFinite(d) && d > 0) setDuration(d);
        }}
        onError={() => fallbackToSynth(run.token())}
        style={{ display: 'none' }}
      />
      <GameActions onEnd={endPlay} endDisabled={playing === null} ended={ended}
        tip="立即停止声音并回退到未播放状态（点击正在播放的卡片也可结束）">
        {playing && <Tag color="purple">{playing} 播放中</Tag>}
      </GameActions>
    </div>
  );
}

// ============================================================
// 5. 涂鸦画板
// ============================================================
const DOODLE_COLORS = ['#722ed1', '#f5222d', '#fa8c16', '#52c41a', '#1890ff', '#eb2f96', '#333333', '#ffffff'];
const DOODLE_DEFAULT_COLOR = '#722ed1';
const DOODLE_DEFAULT_BRUSH = 4;

function DoodleCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawingRef = useRef(false);
  const lastPosRef = useRef<{ x: number; y: number } | null>(null);
  const [color, setColor] = useState(DOODLE_DEFAULT_COLOR);
  const [brushSize, setBrushSize] = useState(DOODLE_DEFAULT_BRUSH);
  const [ended, flashEnded, clearEnded] = useEndedFlag();

  const clearCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.fillStyle = '#fafafa';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }, []);

  useEffect(() => { clearCanvas(); }, [clearCanvas]);

  const getPos = (e: React.MouseEvent | React.TouchEvent) => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
    const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
    return {
      x: (clientX - rect.left) * (canvas.width / rect.width),
      y: (clientY - rect.top) * (canvas.height / rect.height),
    };
  };

  const startDraw = (e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault();
    drawingRef.current = true;
    lastPosRef.current = getPos(e);
  };

  const draw = (e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault();
    if (!drawingRef.current || !lastPosRef.current) return;
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext('2d')!;
    const pos = getPos(e);
    ctx.beginPath();
    ctx.moveTo(lastPosRef.current.x, lastPosRef.current.y);
    ctx.lineTo(pos.x, pos.y);
    ctx.strokeStyle = color;
    ctx.lineWidth = brushSize;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.stroke();
    lastPosRef.current = pos;
  };

  const stopDraw = () => {
    drawingRef.current = false;
    lastPosRef.current = null;
  };

  const clear = () => {
    clearCanvas();
    clearEnded();
  };

  /** 结束并回退：清空画布并把画笔恢复到初始状态 */
  const endGame = () => {
    drawingRef.current = false;
    lastPosRef.current = null;
    clearCanvas();
    setColor(DOODLE_DEFAULT_COLOR);
    setBrushSize(DOODLE_DEFAULT_BRUSH);
    flashEnded();
  };

  return (
    <div style={{ textAlign: 'center' }}>
      <Text type="secondary" style={{ display: 'block', marginBottom: 12, fontSize: 13 }}>
        随意涂鸦，把情绪画出来。没有对错，画完可以清空重来。
      </Text>
      <canvas ref={canvasRef} width={400} height={300}
        onMouseDown={startDraw} onMouseMove={draw} onMouseUp={stopDraw} onMouseLeave={stopDraw}
        onTouchStart={startDraw} onTouchMove={draw} onTouchEnd={stopDraw}
        style={{
          width: '100%', maxWidth: 400, height: 'auto', borderRadius: 16,
          border: '2px solid #f0f0f0', cursor: 'crosshair', touchAction: 'none',
          background: '#fafafa',
        }}
      />
      <div style={{ marginTop: 12, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 12, flexWrap: 'wrap' }}>
        {DOODLE_COLORS.map(c => (
          <div key={c} onClick={() => setColor(c)} style={{
            width: 28, height: 28, borderRadius: '50%', background: c, cursor: 'pointer',
            border: color === c ? '3px solid #333' : '2px solid #ddd',
            transition: 'all 0.2s',
          }} />
        ))}
        <span style={{ fontSize: 12, color: '#999', marginLeft: 8 }}>粗细：</span>
        {[2, 4, 8, 14].map(s => (
          <div key={s} onClick={() => setBrushSize(s)} style={{
            width: 28, height: 28, borderRadius: '50%', background: '#f0f0f0',
            display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer',
            border: brushSize === s ? '2px solid #722ed1' : '1px solid #ddd',
          }}>
            <div style={{ width: s, height: s, borderRadius: '50%', background: '#333' }} />
          </div>
        ))}
        <Button size="small" onClick={clear} style={{ borderRadius: 14 }}>清空</Button>
      </div>
      <GameActions onEnd={endGame} ended={ended}
        tip="清空画布并把画笔设置恢复到初始状态">
      </GameActions>
    </div>
  );
}

// ============================================================
// 6. 橡皮人拉扯
// ============================================================

/** 可选的解压卡通形象（内联 SVG，配色沿用全站淡紫 / 粉 / 青的柔和渐变） */
const RUBBER_CHARACTERS = [
  { key: 'bean', name: '豆豆', from: '#d3c4ff', to: '#8a6bff' },
  { key: 'bear', name: '小熊', from: '#ffe0b8', to: '#ff9f68' },
  { key: 'cat', name: '小猫', from: '#ffd9e6', to: '#ff85a2' },
  { key: 'bunny', name: '小兔', from: '#e4ffcc', to: '#73d13d' },
  { key: 'star', name: '星星', from: '#fff3b0', to: '#ffc53d' },
  { key: 'cloud', name: '云朵', from: '#cfeaff', to: '#69c0ff' },
] as const;

type CharacterKey = typeof RUBBER_CHARACTERS[number]['key'];
type CharacterSpec = typeof RUBBER_CHARACTERS[number];

const FACE_COLOR = '#4a3b66';

/** 一张通用表情，叠在身体之上 */
function CharacterFace() {
  return (
    <g>
      {/* 腮红用不透明的暖粉，避免叠在绿色 / 蓝色身体上发灰 */}
      <ellipse cx="62" cy="117" rx="11.5" ry="7.5" fill="#ff8fa8" opacity="0.78" />
      <ellipse cx="138" cy="117" rx="11.5" ry="7.5" fill="#ff8fa8" opacity="0.78" />
      <ellipse cx="80" cy="98" rx="8.5" ry="11" fill={FACE_COLOR} />
      <ellipse cx="120" cy="98" rx="8.5" ry="11" fill={FACE_COLOR} />
      <circle cx="83" cy="93" r="3.2" fill="#fff" />
      <circle cx="123" cy="93" r="3.2" fill="#fff" />
      <path d="M92 116 Q100 126 108 116" stroke={FACE_COLOR} strokeWidth="3.4"
        fill="none" strokeLinecap="round" />
    </g>
  );
}

/** 各形象的身体轮廓 */
function CharacterBody({ spec, fill }: { spec: CharacterSpec; fill: string }) {
  const inner = `${spec.from}`;
  switch (spec.key) {
    case 'bean':
      return <ellipse cx="100" cy="104" rx="62" ry="66" fill={fill} />;
    case 'bear':
      return (
        <g>
          <circle cx="60" cy="58" r="21" fill={fill} />
          <circle cx="140" cy="58" r="21" fill={fill} />
          <circle cx="60" cy="58" r="11" fill={inner} />
          <circle cx="140" cy="58" r="11" fill={inner} />
          <circle cx="100" cy="106" r="60" fill={fill} />
        </g>
      );
    case 'cat':
      return (
        <g>
          <path d="M58 76 L46 24 L94 52 Z" fill={fill} strokeLinejoin="round" />
          <path d="M142 76 L154 24 L106 52 Z" fill={fill} strokeLinejoin="round" />
          <circle cx="100" cy="108" r="57" fill={fill} />
          {/* 内耳画在头之后，否则会被头部圆形完全盖住 */}
          <path d="M61 63 L54 34 L81 50 Z" fill="#ffb3c6" opacity="0.9" />
          <path d="M139 63 L146 34 L119 50 Z" fill="#ffb3c6" opacity="0.9" />
        </g>
      );
    case 'bunny':
      return (
        <g>
          <ellipse cx="78" cy="46" rx="13" ry="42" transform="rotate(-12 78 46)" fill={fill} />
          <ellipse cx="122" cy="46" rx="13" ry="42" transform="rotate(12 122 46)" fill={fill} />
          <ellipse cx="78" cy="46" rx="6" ry="30" transform="rotate(-12 78 46)" fill="#ffb3c6" opacity="0.7" />
          <ellipse cx="122" cy="46" rx="6" ry="30" transform="rotate(12 122 46)" fill="#ffb3c6" opacity="0.7" />
          <circle cx="100" cy="112" r="55" fill={fill} />
        </g>
      );
    case 'star':
      return (
        <path
          d="M100 34 L117.6 79.7 L166.6 82.4 L128.5 113.3 L141.1 160.6 L100 134 L58.9 160.6 L71.5 113.3 L33.4 82.4 L82.4 79.7 Z"
          fill={fill} strokeLinejoin="round" strokeWidth="10" stroke={fill} />
      );
    case 'cloud':
    default:
      return (
        <g>
          <circle cx="72" cy="108" r="37" fill={fill} />
          <circle cx="128" cy="108" r="37" fill={fill} />
          <circle cx="100" cy="84" r="43" fill={fill} />
          <rect x="38" y="108" width="124" height="40" rx="20" fill={fill} />
        </g>
      );
  }
}

/**
 * 单个卡通形象
 * @param idSuffix 同一形象会在选择器与舞台各画一次，用后缀避免 SVG 渐变 id 冲突
 */
function CharacterArt({ variant, size, idSuffix }: { variant: CharacterKey; size: number; idSuffix: string }) {
  const spec = RUBBER_CHARACTERS.find(c => c.key === variant) ?? RUBBER_CHARACTERS[0];
  const gid = `rubber-${variant}-${idSuffix}`;
  return (
    <svg width={size} height={size} viewBox="0 0 200 200" style={{ display: 'block', overflow: 'visible' }}>
      <defs>
        <radialGradient id={gid} cx="34%" cy="28%" r="82%">
          <stop offset="0%" stopColor={spec.from} />
          <stop offset="100%" stopColor={spec.to} />
        </radialGradient>
      </defs>
      <CharacterBody spec={spec} fill={`url(#${gid})`} />
      <CharacterFace />
    </svg>
  );
}

/** 拖拽最远像素（超出即被夹住，避免小人被拖出画框） */
const MAX_PULL_RADIUS = 160;
/** 拉满时的最大拉伸倍数（与画框高度、形象尺寸一起取值，保证拉满也不出框） */
const MAX_STRETCH = 1.8;
/** 形象在舞台中的显示尺寸 */
const RUBBER_SIZE = 140;
/** 水平 / 垂直方向跟随指针的比例（画框扁而宽，故纵向跟随更少） */
const FOLLOW_X = 0.34;
const FOLLOW_Y = 0.22;

/**
 * 把「拖拽位移」换算成 CSS transform
 *
 * 用 R(θ)·S(s, 1/√s)·R(−θ) 组成的矩阵沿拖拽方向拉伸、垂直方向收细，
 * 因此向右、向左、向上、向下、斜向都能均匀拉长（此前只实现了 x 轴正向）。
 */
function pullToTransform(px: number, py: number): string {
  const dist = Math.hypot(px, py);
  const s = 1 + Math.min(dist / 140, MAX_STRETCH - 1);
  const t = 1 / Math.sqrt(s); // 垂直方向收细，接近体积守恒
  const angle = Math.atan2(py, px);
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const a = cos * cos * s + sin * sin * t;
  const b = cos * sin * (s - t);
  const d = sin * sin * s + cos * cos * t;
  const tx = px * FOLLOW_X; // 位移略慢于指针，形成橡皮筋手感
  const ty = py * FOLLOW_Y;
  return `translate(${tx}px, ${ty}px) matrix(${a}, ${b}, ${b}, ${d}, 0, 0)`;
}

function RubberPerson() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [variant, setVariant] = useState<CharacterKey>('bean');
  const [dragging, setDragging] = useState(false);
  const [pull, setPull] = useState({ x: 0, y: 0 });
  const [pullCount, setPullCount] = useState(0);
  const [soundOn, setSoundOn] = useState(true);
  const [ended, flashEnded] = useEndedFlag();
  const run = useInterruptible();
  const dragStartRef = useRef<{ x: number; y: number } | null>(null);
  const pullRef = useRef({ x: 0, y: 0 });
  const audioCtxRef = useRef<AudioContext | null>(null);
  const soundOnRef = useRef(true);

  soundOnRef.current = soundOn;

  /** 回弹音效用的 AudioContext，按需创建 */
  const ensureAudio = useCallback(() => {
    if (!audioCtxRef.current || audioCtxRef.current.state === 'closed') {
      audioCtxRef.current = new AudioContext();
    }
    if (audioCtxRef.current.state === 'suspended') void audioCtxRef.current.resume();
    return audioCtxRef.current;
  }, []);

  /** 弹簧回弹音效：下滑音 + 由快变慢的颤音，音高随拉扯幅度升高 */
  const playBoing = useCallback((strength: number) => {
    try {
      const ctx = ensureAudio();
      const t0 = ctx.currentTime;
      const base = 150 + 150 * strength;
      const osc = ctx.createOscillator();
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(base * 2.2, t0);
      osc.frequency.exponentialRampToValueAtTime(base * 0.55, t0 + 0.3);

      const lfo = ctx.createOscillator();
      lfo.frequency.setValueAtTime(28, t0);
      lfo.frequency.exponentialRampToValueAtTime(9, t0 + 0.3);
      const lfoGain = ctx.createGain();
      lfoGain.gain.setValueAtTime(base * 0.4, t0);
      lfoGain.gain.exponentialRampToValueAtTime(base * 0.02, t0 + 0.3);
      lfo.connect(lfoGain);
      lfoGain.connect(osc.frequency);

      const filter = ctx.createBiquadFilter();
      filter.type = 'lowpass';
      filter.frequency.value = 3200;

      const gain = ctx.createGain();
      gain.gain.setValueAtTime(0.0001, t0);
      gain.gain.exponentialRampToValueAtTime(0.26, t0 + 0.012);
      gain.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.42);

      osc.connect(filter);
      filter.connect(gain);
      gain.connect(ctx.destination);
      osc.start(t0);
      lfo.start(t0);
      osc.stop(t0 + 0.45);
      lfo.stop(t0 + 0.45);
    } catch { /* 浏览器不支持或被拦截时静默失败，不影响玩法 */ }
  }, [ensureAudio]);

  const closeAudio = useCallback(() => {
    const ctx = audioCtxRef.current;
    if (ctx && ctx.state !== 'closed') { try { void ctx.close(); } catch { /* 忽略 */ } }
    audioCtxRef.current = null;
  }, []);

  useEffect(() => () => closeAudio(), [closeAudio]);

  const resetPull = useCallback(() => {
    pullRef.current = { x: 0, y: 0 };
    setPull({ x: 0, y: 0 });
  }, []);

  const handleDown = (e: React.PointerEvent) => {
    e.preventDefault();
    run.interrupt(); // 中断上一次回弹动画
    setDragging(true);
    dragStartRef.current = { x: e.clientX, y: e.clientY };
  };

  useEffect(() => {
    if (!dragging) return;
    const handleMove = (e: PointerEvent) => {
      if (!dragStartRef.current) return;
      let dx = e.clientX - dragStartRef.current.x;
      let dy = e.clientY - dragStartRef.current.y;
      const dist = Math.hypot(dx, dy);
      if (dist > MAX_PULL_RADIUS) { // 夹住最大位移，保证小人不出画框
        dx = (dx / dist) * MAX_PULL_RADIUS;
        dy = (dy / dist) * MAX_PULL_RADIUS;
      }
      const next = { x: dx, y: dy };
      pullRef.current = next;
      setPull(next);
    };
    const handleUp = () => {
      setDragging(false);
      dragStartRef.current = null;
      setPullCount(c => c + 1);
      const start = { ...pullRef.current };
      const strength = Math.min(Math.hypot(start.x, start.y) / MAX_PULL_RADIUS, 1);
      if (strength > 0.06 && soundOnRef.current) playBoing(strength);

      // 欠阻尼弹性回弹：会稍稍反向甩过头再稳定，可被「结束并回退」或下一次拖拽中断
      const token = run.begin();
      const t0 = Date.now();
      const DURATION = 720;
      const animate = () => {
        if (!run.isActive(token)) return;
        const p = Math.min((Date.now() - t0) / DURATION, 1);
        const ease = 1 - Math.pow(1 - p, 3) * Math.cos(p * Math.PI * 2);
        const next = { x: start.x * (1 - ease), y: start.y * (1 - ease) };
        pullRef.current = next;
        setPull(next);
        if (p < 1) {
          run.trackFrame(requestAnimationFrame(animate));
        } else {
          resetPull();
        }
      };
      run.trackFrame(requestAnimationFrame(animate));
    };
    window.addEventListener('pointermove', handleMove);
    window.addEventListener('pointerup', handleUp);
    window.addEventListener('pointercancel', handleUp);
    return () => {
      window.removeEventListener('pointermove', handleMove);
      window.removeEventListener('pointerup', handleUp);
      window.removeEventListener('pointercancel', handleUp);
    };
  }, [dragging, run, playBoing, resetPull]);

  const selectCharacter = useCallback((key: CharacterKey) => {
    run.interrupt();
    setVariant(key);
    resetPull();
  }, [run, resetPull]);

  /** 结束并回退：中断拖拽、回弹动画与音效，小人恢复原状、次数清零 */
  const endGame = useCallback(() => {
    if (!dragging && pullCount === 0 && pull.x === 0 && pull.y === 0) return;
    run.interrupt();
    closeAudio();
    setDragging(false);
    dragStartRef.current = null;
    resetPull();
    setPullCount(0);
    flashEnded();
  }, [run, dragging, pullCount, pull, closeAudio, resetPull, flashEnded]);

  useEscapeToEnd(endGame, dragging || pullCount > 0);
  useAbortOnTabChange(endGame, dragging);

  const stretchLevel = Math.min(Math.hypot(pull.x, pull.y) / MAX_PULL_RADIUS, 1);

  return (
    <div style={{ textAlign: 'center' }}>
      <Text type="secondary" style={{ display: 'block', marginBottom: 12, fontSize: 13 }}>
        抓住小人朝<strong>任意方向</strong>拖拽，松手会「啵」地弹回。先挑一个喜欢的形象吧。
      </Text>
      <div ref={containerRef} style={{
        position: 'relative',
        height: 330, display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: '#fafafa', borderRadius: 16, border: '2px dashed #e8e8e8',
        cursor: dragging ? 'grabbing' : 'grab', userSelect: 'none', overflow: 'hidden',
        touchAction: 'none',
      }}
        onPointerDown={handleDown}>
        {/* 地面投影：保持在原位，衬托小人被拽出去的距离 */}
        <div style={{
          position: 'absolute', bottom: 52, width: 96, height: 14, borderRadius: '50%',
          background: 'rgba(70,50,110,0.07)',
          transform: `translateX(${pull.x * FOLLOW_X}px) scaleX(${1 - stretchLevel * 0.15})`,
          transition: dragging ? 'none' : 'transform 0.06s linear',
        }} />
        <div style={{
          transform: pullToTransform(pull.x, pull.y),
          transition: dragging ? 'none' : 'transform 0.06s linear',
          willChange: 'transform',
        }}>
          <CharacterArt variant={variant} size={RUBBER_SIZE} idSuffix="stage" />
        </div>
      </div>

      <div style={{ marginTop: 12, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, flexWrap: 'wrap' }}>
        {RUBBER_CHARACTERS.map(ch => (
          <div key={ch.key} onClick={() => selectCharacter(ch.key)} title={ch.name} style={{
            width: 42, height: 42, borderRadius: 13, cursor: 'pointer',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: variant === ch.key ? `${ch.to}1f` : '#fafafa',
            border: variant === ch.key ? `2px solid ${ch.to}` : '1px solid #eee',
            transition: 'all 0.2s',
          }}>
            <CharacterArt variant={ch.key} size={32} idSuffix="picker" />
          </div>
        ))}
        <Button size="small" onClick={() => setSoundOn(v => !v)} style={{ borderRadius: 14, marginLeft: 4 }}>
          {soundOn ? '\u{1F50A} 音效开' : '\u{1F507} 音效关'}
        </Button>
      </div>

      <div style={{ marginTop: 12 }}>
        <Tag color="orange" style={{ fontSize: 14 }}>已拉扯 {pullCount} 次</Tag>
        {pullCount > 5 && <Tag color="green" style={{ fontSize: 12, marginLeft: 8 }}>压力释放中... 💨</Tag>}
        {pullCount > 15 && <Tag color="purple" style={{ fontSize: 12, marginLeft: 8 }}>解压大师！🏆</Tag>}
      </div>
      <GameActions onEnd={endGame} endDisabled={!dragging && pullCount === 0 && pull.x === 0 && pull.y === 0}
        ended={ended} tip="立即中断拖拽并让小人回到原状（快捷键 Esc）">
      </GameActions>
    </div>
  );
}

// ============================================================
// 7. 反应灯（红绿 + Go / No-Go）
// ============================================================
const RL_TOTAL = 20;
const RL_NOGO = 5;
const RL_WAIT_MIN = 1200;  // 变灯前的等待下限：太快会被预判
const RL_WAIT_MAX = 3000;  // 上限：再长就开始走神了

type RlPhase = 'idle' | 'wait' | 'go' | 'nogo' | 'done';

const RL_LOOK: Record<RlPhase, { bg: string; text: string }> = {
  idle: { bg: '#f0f0f5', text: '点下面的「开始」' },
  wait: { bg: '#33334d', text: '等待…' },
  go: { bg: '#52c41a', text: '' },
  nogo: { bg: '#ff4d4f', text: '' },
  done: { bg: '#f0f0f5', text: '' },
};

/**
 * 反应灯
 * 范式取自电竞选拔里的视觉反应测试和认知心理学里的 Go / No-Go：
 *   Go 试次（绿）：尽快按 —— 测简单反应时
 *   No-Go 试次（红）：不能按 —— 测抑制控制（冲动性）
 * 红灯不能出现任何文字提示，只靠颜色区分：优势习惯是「看到就按」，
 * 抑制的难度正来自这里，写了「停」就等于把测试变成读字。
 */
function ReactionLight() {
  const {
    current: currentTrial, start: engineStart, submit: engineSubmit,
    reset: engineReset, stats,
  } = useTrialEngine(RL_TOTAL, RL_NOGO);
  const clock = useReactionClock();
  const run = useInterruptible();

  const [phase, setPhase] = useState<RlPhase>('idle');
  const phaseRef = useRef<RlPhase>('idle');
  const [feedback, setFeedback] = useState<{ text: string; tone: string } | null>(null);
  const [ended, flashEnded, clearEnded] = useEndedFlag();
  const isExpanded = useIsExpanded();
  const feedbackTimerRef = useRef<any>(null);
  const recordedRef = useRef(false);
  const startedAtRef = useRef(0);
  const { best, submit: submitBest } = usePersonalBest('reaction-light', false); // 反应时越低越好

  const setPhaseSafe = useCallback((p: RlPhase) => { phaseRef.current = p; setPhase(p); }, []);

  const flash = useCallback((text: string, tone: string) => {
    setFeedback({ text, tone });
    if (feedbackTimerRef.current) clearTimeout(feedbackTimerRef.current);
    feedbackTimerRef.current = setTimeout(() => setFeedback(null), 800);
  }, []);

  useEffect(() => () => { if (feedbackTimerRef.current) clearTimeout(feedbackTimerRef.current); }, []);

  /**
   * 跑一个试次
   * 每轮开头先 run.begin()：清掉上一试次残留的定时器、并让旧回调失效。
   * 少了这一步，「抢跑后立刻进入下一试次」会让上一轮的等待定时器继续跑，界面连跳两轮。
   */
  const advanceRef = useRef<() => void>(() => {});
  const advance = useCallback(() => {
    const token = run.begin();
    clock.clear();
    const item = currentTrial();
    if (!item) { setPhaseSafe('done'); return; }
    setPhaseSafe('wait');
    const delay = RL_WAIT_MIN + Math.random() * (RL_WAIT_MAX - RL_WAIT_MIN);
    run.trackTimer(setTimeout(() => {
      if (!run.isActive(token)) return;
      setPhaseSafe(item.kind === 'go' ? 'go' : 'nogo');
      clock.armOnNextFrame();
      run.trackTimer(setTimeout(() => {
        if (!run.isActive(token)) return;
        if (item.kind === 'go') flash('没接住，慢了', '#fa8c16');
        // 绿灯没按 = 漏接；红灯没按 = 抑制成功
        engineSubmit(item.kind === 'go' ? 'miss' : 'hit', null);
        advanceRef.current();
      }, item.kind === 'go' ? RT_GO_WINDOW : RT_NOGO_WINDOW));
    }, delay));
  }, [run, clock, currentTrial, setPhaseSafe, flash, engineSubmit]);

  useEffect(() => { advanceRef.current = advance; }, [advance]);

  const handleResponse = useCallback((e?: { timeStamp?: number }) => {
    const p = phaseRef.current;
    if (p === 'wait') {
      flash('抢跑了！灯还没变色', '#fa541c');
      engineSubmit('anticipate', null);
    } else if (p === 'nogo') {
      flash('红色不用按哦', '#fa541c');
      engineSubmit('wrong', null);
    } else if (p === 'go') {
      const ms = clock.elapsed(e);
      if (ms === null) return;
      if (ms < RT_MIN_VALID) {
        flash('太快了，算抢跑', '#fa541c');
        engineSubmit('anticipate', null);
      } else {
        flash(`${ms} ms`, ms < 320 ? '#52c41a' : '#fa8c16');
        engineSubmit('hit', ms);
      }
    } else {
      return; // idle / done：这时候按没有意义
    }
    advanceRef.current();
  }, [flash, engineSubmit, clock]);

  const start = useCallback(() => {
    engineStart();
    recordedRef.current = false;
    startedAtRef.current = Date.now();
    setFeedback(null);
    clearEnded();
    advanceRef.current();
  }, [engineStart, clearEnded]);

  const endGame = useCallback(() => {
    if (phaseRef.current === 'idle') return;
    run.interrupt();
    clock.clear();
    engineReset();
    recordedRef.current = true; // 阻断正在进行的结算落库
    setPhaseSafe('idle');
    setFeedback(null);
    flashEnded();
  }, [run, clock, engineReset, setPhaseSafe, flashEnded]);

  const isRunning = phase === 'wait' || phase === 'go' || phase === 'nogo';
  useEscapeToEnd(endGame, isRunning);
  useAbortOnTabChange(endGame, isRunning);

  // 空格 = 反应键。用 keydown 而不是等 click，键盘的输入延迟通常比鼠标更低
  useEffect(() => {
    if (!isRunning) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== 'Space' || e.repeat) return;
      e.preventDefault();
      handleResponse({ timeStamp: e.timeStamp });
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isRunning, handleResponse]);

  // 一局结束：写历史 + 更新个人最快
  useEffect(() => {
    if (phase !== 'done' || recordedRef.current) return;
    recordedRef.current = true;
    const s = stats;
    if (!s.rt) return;
    submitBest(s.rt.best);
    recordGameResult('reaction-light', '反应灯', (Date.now() - startedAtRef.current) / 1000, {
      trials: s.done, valid: s.valid, go: s.goTotal, nogo: s.nogoTotal,
      hit: s.hit, miss: s.miss, wrong: s.wrong, anticipate: s.anticipate,
      accuracy: Math.round(s.accuracy * 1000) / 1000,
      meanRt: s.rt.mean, bestRt: s.rt.best, sdRt: s.rt.sd,
    });
  }, [phase, stats, submitBest]);

  const look = RL_LOOK[phase];
  const progress = phase === 'idle' ? 0 : Math.min(stats.done + (phase === 'done' ? 0 : 1), RL_TOTAL);
  const rt = phase === 'done' ? stats.rt : null;

  return (
    <div style={{ textAlign: 'center' }}>
      <Text type="secondary" style={{ display: 'block', marginBottom: 10, fontSize: 13, lineHeight: 1.7 }}>
        屏幕变<strong style={{ color: '#52c41a' }}>绿</strong>就尽快按（空格或点色块）；
        变<strong style={{ color: '#ff4d4f' }}>红</strong>千万别按。
      </Text>

      <div style={{ display: 'flex', justifyContent: 'center', gap: 32, marginBottom: 10 }}>
        <div>
          <div style={{ fontSize: 12, color: '#999' }}>进度</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: '#722ed1' }}>{progress} / {RL_TOTAL}</div>
        </div>
        {best !== null && (
          <div>
            <div style={{ fontSize: 12, color: '#999' }}>个人最快</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: '#52c41a' }}>{best} ms</div>
          </div>
        )}
        {stats.rt && !rt && (
          <div>
            <div style={{ fontSize: 12, color: '#999' }}>本次均速</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: '#1890ff' }}>{stats.rt.mean} ms</div>
          </div>
        )}
      </div>

      {/* 不加颜色过渡动画：渐变动画会让刺激的「出现时刻」变得模糊，直接影响测量 */}
      <div
        className="reaction-light-stage"
        onPointerDown={e => handleResponse(e)}
        style={{
          width: '100%', maxWidth: isExpanded ? 620 : 420, height: isExpanded ? 360 : 240,
          margin: '0 auto',
          borderRadius: 18, background: look.bg,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: isRunning ? '#fff' : '#8c8c8c', fontSize: isExpanded ? 20 : 16, fontWeight: 600,
          userSelect: 'none', touchAction: 'none',
          cursor: isRunning ? 'pointer' : 'default',
          boxShadow: 'inset 0 0 0 1px rgba(0,0,0,0.04)',
        }}
      >
        {look.text}
      </div>

      <div style={{ height: 24, marginTop: 6 }}>
        {feedback && <Text style={{ color: feedback.tone, fontWeight: 600 }}>{feedback.text}</Text>}
      </div>

      {rt && (
        <ReactionReport
          title="本局成绩"
          rows={[
            { label: '平均反应', value: `${rt.mean} ms` },
            { label: '最快', value: `${rt.best} ms` },
            { label: '稳定性', value: `±${rt.sd} ms` },
            { label: '正确率', value: `${Math.round(stats.accuracy * 100)}%` },
            { label: '抢跑', value: `${stats.anticipate} 次` },
            { label: '误按红灯', value: `${stats.wrong} 次` },
          ]}
          note={
            `共 ${stats.goTotal} 次绿灯、${stats.nogoTotal} 次红灯，有效试次 ${stats.valid} 个` +
            `（抢跑会让该试次作废，不计入正确率）。` +
            `稳定性是反应时的标准差，越小说明节奏越匀；误按红灯越多说明「先按再说」的冲动越强。` +
            RT_DISCLAIMER
          }
        />
      )}

      <GameActions onEnd={endGame} endDisabled={phase === 'idle'} ended={ended}>
        <Button type="primary" onClick={start} style={{ borderRadius: 20 }}>
          {phase === 'idle' ? '开始' : '重新开始'}
        </Button>
      </GameActions>
    </div>
  );
}

// ============================================================
// 8. 多目标追踪（MOT）
// ============================================================
const MOT_SIZE = 360;
const MOT_MARK_MS = 2000;      // 标记目标的时间
const MOT_MOVE_MS = 6000;      // 打散运动的时间
const MOT_FEEDBACK_MS = 1400;  // 展示本轮对错的时间
const MOT_ROUNDS = 5;

/**
 * 难度表
 * 多目标追踪的瓶颈是「物体之间的空间干涉」，不是速度：
 * 球越多、越小，彼此越容易互相遮挡和混淆，难度就上来了。
 * 所以三档主要加数量、缩半径，速度只做小幅配合，避免变成拼手速。
 */
const MOT_LEVELS = [
  { targets: 4, dots: 10, radius: 16, speed: 55 },
  { targets: 5, dots: 13, radius: 14, speed: 62 },
  { targets: 6, dots: 16, radius: 13, speed: 70 },
];

interface MotDot { x: number; y: number; vx: number; vy: number; target: boolean; picked: boolean; }
interface MotRound { level: number; targets: number; hit: number; falseAlarm: number; }
type MotPhase = 'idle' | 'marking' | 'moving' | 'select' | 'feedback' | 'done';

/** 撒点：要求最小间距，保证起手时球是分得开的（间距本身就是难度变量） */
function seedDots(count: number, radius: number, targetCount: number): MotDot[] {
  const dots: MotDot[] = [];
  const minGap = radius * 2 + 16;
  const rand = () => ({
    x: radius + Math.random() * (MOT_SIZE - radius * 2),
    y: radius + Math.random() * (MOT_SIZE - radius * 2),
  });
  let guard = 0;
  while (dots.length < count && guard < 3000) {
    guard += 1;
    const p = rand();
    if (dots.some(d => Math.hypot(d.x - p.x, d.y - p.y) < minGap)) continue;
    const a = Math.random() * Math.PI * 2;
    dots.push({ x: p.x, y: p.y, vx: Math.cos(a), vy: Math.sin(a), target: false, picked: false });
  }
  // 兜底：间距要求太苛刻时放宽，保证球数够
  while (dots.length < count) {
    const p = rand();
    const a = Math.random() * Math.PI * 2;
    dots.push({ x: p.x, y: p.y, vx: Math.cos(a), vy: Math.sin(a), target: false, picked: false });
  }
  // 随机挑目标
  const idx = dots.map((_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  idx.slice(0, targetCount).forEach(i => { dots[i].target = true; });
  return dots;
}

function MultiObjectTracking() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dotsRef = useRef<MotDot[]>([]);
  const picksRef = useRef(0);
  const levelRef = useRef(0);
  const roundsRef = useRef<MotRound[]>([]);
  const recordedRef = useRef(false);
  const startedAtRef = useRef(0);

  const [phase, setPhase] = useState<MotPhase>('idle');
  const phaseRef = useRef<MotPhase>('idle');
  const [picks, setPicks] = useState(0);
  const [rounds, setRounds] = useState<MotRound[]>([]);
  const [showLevel, setShowLevel] = useState(0);
  const [ended, flashEnded, clearEnded] = useEndedFlag();
  const isExpanded = useIsExpanded();
  const motScale = isExpanded ? 2 : 1; // 放大时后备存储 ×2，球才是清晰的

  const run = useInterruptible();
  const { adjust: adjustLevel, reset: resetLevel } = useStaircase(0, 0, MOT_LEVELS.length - 1);
  const { best, submit: submitBest } = usePersonalBest('mot', true); // 正确率越高越好

  const setPhaseSafe = useCallback((p: MotPhase) => { phaseRef.current = p; setPhase(p); }, []);

  const startRound = useCallback((levelIdx: number) => {
    const level = MOT_LEVELS[levelIdx];
    levelRef.current = levelIdx;
    setShowLevel(levelIdx);
    dotsRef.current = seedDots(level.dots, level.radius, level.targets);
    picksRef.current = 0;
    setPicks(0);
    const token = run.begin();
    setPhaseSafe('marking');
    run.trackTimer(setTimeout(() => {
      if (!run.isActive(token)) return;
      setPhaseSafe('moving');
      run.trackTimer(setTimeout(() => {
        if (!run.isActive(token)) return;
        setPhaseSafe('select');
      }, MOT_MOVE_MS));
    }, MOT_MARK_MS));
  }, [run, setPhaseSafe]);

  const startRoundRef = useRef(startRound);
  useEffect(() => { startRoundRef.current = startRound; }, [startRound]);

  const submitRound = useCallback(() => {
    const dots = dotsRef.current;
    const targets = dots.filter(d => d.target).length;
    const hit = dots.filter(d => d.target && d.picked).length;
    const falseAlarm = dots.filter(d => !d.target && d.picked).length;
    const nextRounds = [...roundsRef.current, { level: levelRef.current, targets, hit, falseAlarm }];
    roundsRef.current = nextRounds;
    setRounds(nextRounds);

    // 阶梯：全对升一档，低于 60% 降一档
    const nextLevel = adjustLevel(targets > 0 ? hit / targets : 0);

    const token = run.begin();
    setPhaseSafe('feedback');
    run.trackTimer(setTimeout(() => {
      if (!run.isActive(token)) return;
      if (nextRounds.length >= MOT_ROUNDS) { setPhaseSafe('done'); return; }
      startRoundRef.current(nextLevel);
    }, MOT_FEEDBACK_MS));
  }, [run, setPhaseSafe, adjustLevel]);

  const submitRoundRef = useRef(submitRound);
  useEffect(() => { submitRoundRef.current = submitRound; }, [submitRound]);

  /** 矢量积分 + 方形容器反弹 */
  const integrate = useCallback((dt: number) => {
    const level = MOT_LEVELS[levelRef.current];
    const speed = level.speed;
    for (const d of dotsRef.current) {
      d.x += d.vx * speed * dt;
      d.y += d.vy * speed * dt;
      if (d.x < level.radius) { d.x = level.radius; d.vx = Math.abs(d.vx); }
      if (d.x > MOT_SIZE - level.radius) { d.x = MOT_SIZE - level.radius; d.vx = -Math.abs(d.vx); }
      if (d.y < level.radius) { d.y = level.radius; d.vy = Math.abs(d.vy); }
      if (d.y > MOT_SIZE - level.radius) { d.y = MOT_SIZE - level.radius; d.vy = -Math.abs(d.vy); }
    }
  }, []);

  const drawMot = useCallback((ctx: CanvasRenderingContext2D) => {
    const level = MOT_LEVELS[levelRef.current];
    const p = phaseRef.current;
    const r = level.radius;
    // 物理坐标始终是 360×360，放大时只把画布后备存储放大再整体缩放，
    // 这样球是清晰的，而种子、积分、点选映射全都不用改。
    const s = ctx.canvas.width / MOT_SIZE;
    ctx.setTransform(s, 0, 0, s, 0, 0);
    ctx.clearRect(0, 0, MOT_SIZE, MOT_SIZE);
    ctx.fillStyle = '#fbfaff';
    ctx.fillRect(0, 0, MOT_SIZE, MOT_SIZE);

    for (const d of dotsRef.current) {
      let fill = '#c9c4d8';
      let stroke = '';
      if (p === 'marking') {
        fill = d.target ? '#722ed1' : '#d5d0e0';
      } else if (p === 'moving') {
        fill = '#8a6bff'; // 运动阶段全部同色：只能靠记住身份，不能靠颜色
      } else if (p === 'select') {
        fill = d.picked ? '#ff8fab' : '#8a6bff';
        stroke = d.picked ? '#d4380d' : '';
      } else if (p === 'feedback') {
        if (d.target && d.picked) fill = '#52c41a';
        else if (d.target && !d.picked) { fill = '#8a6bff'; stroke = '#fa8c16'; }
        else if (!d.target && d.picked) fill = '#ff4d4f';
        else fill = '#d5d0e0';
      }
      ctx.beginPath();
      ctx.arc(d.x, d.y, r, 0, Math.PI * 2);
      ctx.fillStyle = fill;
      ctx.fill();
      if (stroke) {
        ctx.lineWidth = 3;
        ctx.strokeStyle = stroke;
        ctx.stroke();
      }
      if (p === 'marking' && d.target) {
        ctx.beginPath();
        ctx.arc(d.x, d.y, r * 0.34, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255,255,255,0.85)';
        ctx.fill();
      }
    }
  }, []);

  // idle 时画一张静态预览（也负责「结束并回退」后把画布刷回中性状态）
  useEffect(() => {
    if (phase !== 'idle') return;
    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx) return;
    levelRef.current = 0;
    dotsRef.current = seedDots(MOT_LEVELS[0].dots, MOT_LEVELS[0].radius, MOT_LEVELS[0].targets);
    drawMot(ctx);
  }, [phase, isExpanded, drawMot]);

  // 只在有回合在跑的时候开动画帧；idle / done 不占资源
  useEffect(() => {
    if (phase === 'idle' || phase === 'done') return;
    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx) return;
    let last = 0;
    let id = 0;
    const frame = (t: number) => {
      const dt = last ? Math.min((t - last) / 1000, 0.05) : 0;
      last = t;
      if (phaseRef.current === 'moving') integrate(dt);
      drawMot(ctx);
      id = requestAnimationFrame(frame);
    };
    id = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(id);
  }, [phase, integrate, drawMot]);

  const handlePick = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (phaseRef.current !== 'select') return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const px = (e.clientX - rect.left) * (MOT_SIZE / rect.width);
    const py = (e.clientY - rect.top) * (MOT_SIZE / rect.height);
    const level = MOT_LEVELS[levelRef.current];
    const dots = dotsRef.current;
    let bestIdx = -1;
    let bestDist = Infinity;
    dots.forEach((d, i) => {
      const dist = Math.hypot(d.x - px, d.y - py);
      if (dist < bestDist) { bestDist = dist; bestIdx = i; }
    });
    if (bestIdx < 0 || bestDist > level.radius * 1.8) return;
    const hit = dots[bestIdx];
    if (hit.picked) {
      hit.picked = false;
      picksRef.current = Math.max(0, picksRef.current - 1);
    } else if (picksRef.current < level.targets) {
      hit.picked = true;
      picksRef.current += 1;
    }
    setPicks(picksRef.current);
    if (picksRef.current >= level.targets) submitRoundRef.current();
  };

  const start = useCallback(() => {
    roundsRef.current = [];
    setRounds([]);
    recordedRef.current = false;
    startedAtRef.current = Date.now();
    clearEnded();
    resetLevel();
    startRoundRef.current(0);
  }, [clearEnded, resetLevel]);

  const endGame = useCallback(() => {
    if (phaseRef.current === 'idle') return;
    run.interrupt();
    recordedRef.current = true;
    picksRef.current = 0;
    setPicks(0);
    roundsRef.current = [];
    setRounds([]);
    levelRef.current = 0;
    setShowLevel(0);
    resetLevel();
    setPhaseSafe('idle');
    flashEnded();
  }, [run, resetLevel, setPhaseSafe, flashEnded]);

  const isRunning = phase !== 'idle' && phase !== 'done';
  useEscapeToEnd(endGame, isRunning);
  useAbortOnTabChange(endGame, isRunning);

  useEffect(() => {
    if (phase !== 'done' || recordedRef.current) return;
    recordedRef.current = true;
    const list = roundsRef.current;
    const totalTargets = list.reduce((a, r) => a + r.targets, 0);
    const totalHit = list.reduce((a, r) => a + r.hit, 0);
    const falseAlarm = list.reduce((a, r) => a + r.falseAlarm, 0);
    const accuracy = totalTargets > 0 ? totalHit / totalTargets : 0;
    submitBest(Math.round(accuracy * 100));
    recordGameResult('mot', '多目标追踪', (Date.now() - startedAtRef.current) / 1000, {
      rounds: list.length, totalTargets, totalHit, falseAlarm,
      accuracy: Math.round(accuracy * 1000) / 1000,
      maxLevel: list.reduce((a, r) => Math.max(a, r.level), 0) + 1,
    });
  }, [phase, submitBest]);

  const level = MOT_LEVELS[showLevel];
  // 回合进度：idle 是 0；本轮进行中是「已完成的 + 1」；本轮已结算（feedback / done）就等于已完成数
  const roundNo = phase === 'idle'
    ? 0
    : (phase === 'done' || phase === 'feedback')
      ? rounds.length
      : Math.min(rounds.length + 1, MOT_ROUNDS);
  const totalTargets = rounds.reduce((a, r) => a + r.targets, 0);
  const totalHit = rounds.reduce((a, r) => a + r.hit, 0);
  const totalFA = rounds.reduce((a, r) => a + r.falseAlarm, 0);
  const finalAccuracy = totalTargets > 0 ? totalHit / totalTargets : 0;

  const hint: Record<MotPhase, string> = {
    idle: '开始后会先高亮几个球，记住它们；散开后它们会混进其他球里，停下来后把它们点出来。',
    marking: '记住这几个亮着的球 👀',
    moving: '它们混进去了，盯住…',
    select: `点出你记住的球，还剩 ${Math.max(0, level.targets - picks)} 个`,
    feedback: '看看认对没有',
    done: '',
  };

  return (
    <div style={{ textAlign: 'center' }}>
      <Text type="secondary" style={{ display: 'block', marginBottom: 10, fontSize: 13, lineHeight: 1.7, minHeight: 44 }}>
        {hint[phase]}
      </Text>

      <div style={{ display: 'flex', justifyContent: 'center', gap: 32, marginBottom: 10 }}>
        <div>
          <div style={{ fontSize: 12, color: '#999' }}>回合</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: '#722ed1' }}>
            {roundNo} / {MOT_ROUNDS}
          </div>
        </div>
        <div>
          <div style={{ fontSize: 12, color: '#999' }}>难度档</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: '#1890ff' }}>
            第 {showLevel + 1} 档 · {level.targets} 目标
          </div>
        </div>
        {best !== null && (
          <div>
            <div style={{ fontSize: 12, color: '#999' }}>个人最佳</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: '#52c41a' }}>{best}%</div>
          </div>
        )}
      </div>

      <canvas
        ref={canvasRef}
        className="mot-canvas"
        width={MOT_SIZE * motScale}
        height={MOT_SIZE * motScale}
        onPointerDown={handlePick}
        style={{
          width: '100%', maxWidth: isExpanded ? 560 : MOT_SIZE, height: 'auto', display: 'block',
          margin: '0 auto', borderRadius: 16, border: '2px solid #f0f0f0',
          background: '#fbfaff', touchAction: 'none',
          cursor: phase === 'select' ? 'crosshair' : 'default',
        }}
      />

      {phase === 'done' && (
        <div style={{ marginTop: 12 }}>
          <ReactionReport
            title="本局成绩"
            rows={[
              { label: '正确率', value: `${Math.round(finalAccuracy * 100)}%` },
              { label: '认对', value: `${totalHit} / ${totalTargets}` },
              { label: '认错', value: `${totalFA} 个` },
              { label: '最高档', value: `第 ${rounds.reduce((a, r) => Math.max(a, r.level), 0) + 1} 档` },
            ]}
            note={
              `每一档的球数：${MOT_LEVELS.map(l => `${l.targets} 目标 / ${l.dots} 球`).join('、')}。` +
              '难度会跟着你的表现走：全对就升档，低于六成就降档，稳定在有点挑战又做得到的区间。' +
              '认错比认漏更能说明问题——那通常是把「跟丢了的那颗」当成了目标。'
            }
          />
        </div>
      )}

      <GameActions onEnd={endGame} endDisabled={phase === 'idle'} ended={ended}>
        <Button type="primary" onClick={start} style={{ borderRadius: 20 }}>
          {phase === 'idle' ? '开始' : '重新开始'}
        </Button>
      </GameActions>
    </div>
  );
}

// ============================================================
// 9. 跟手追踪（眼手协调）
// ============================================================
const HT_W = 360;
const HT_H = 260;
const HT_RADIUS = 26;   // 目标半径，也就是「算命中」的容差
const HT_SPEED = 170;   // 目标移动速度 px/s
const HT_DURATION = 20; // 每局秒数

interface HtResult { hitRate: number; meanDist: number; frames: number; }
type HtPhase = 'idle' | 'running' | 'done';

/**
 * 跟手追踪
 * 对应 Vienna Test System 里的双手协调 / 运动表现系列中的「线追踪」，
 * 只是把手写笔换成了鼠标：目标在框内平滑游走，鼠标尽量贴住它。
 * 核心指标是 time-on-target（命中率），它是眼手协调最直接的一个量。
 */
function HandTracking() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const targetRef = useRef({ x: HT_W / 2, y: HT_H / 2, heading: 0 });
  const pointerRef = useRef({ x: HT_W / 2, y: HT_H / 2, inside: false });
  const metricsRef = useRef({ hitFrames: 0, totalFrames: 0, sumDist: 0, insideFrames: 0 });
  const recordedRef = useRef(false);
  const startedAtRef = useRef(0);

  const [phase, setPhase] = useState<HtPhase>('idle');
  const phaseRef = useRef<HtPhase>('idle');
  const [timeLeft, setTimeLeft] = useState(HT_DURATION);
  const [liveHit, setLiveHit] = useState(0);
  const [result, setResult] = useState<HtResult | null>(null);
  const [ended, flashEnded, clearEnded] = useEndedFlag();
  const isExpanded = useIsExpanded();
  const htScale = isExpanded ? 2 : 1; // 放大时后备存储 ×2

  const run = useInterruptible();
  const { best, submit: submitBest } = usePersonalBest('tracking', true);

  const setPhaseSafe = useCallback((p: HtPhase) => { phaseRef.current = p; setPhase(p); }, []);

  const resetField = useCallback(() => {
    targetRef.current = { x: HT_W / 2, y: HT_H / 2, heading: Math.random() * Math.PI * 2 };
    pointerRef.current = { x: HT_W / 2, y: HT_H / 2, inside: false };
  }, []);

  const emptyMetrics = () => ({ hitFrames: 0, totalFrames: 0, sumDist: 0, insideFrames: 0 });

  /**
   * 平滑游走
   * 航向角缓慢随机变化，比每帧随机抖动更像真实的追踪目标：
   * 纯抖动会变成高频噪声，考验的是鼠标刷新率而不是协调能力。
   */
  const step = useCallback((dt: number) => {
    const t = targetRef.current;
    t.heading += (Math.random() - 0.5) * 2.6 * dt;
    t.x += Math.cos(t.heading) * HT_SPEED * dt;
    t.y += Math.sin(t.heading) * HT_SPEED * dt;
    if (t.x < HT_RADIUS) { t.x = HT_RADIUS; t.heading = Math.PI - t.heading; }
    if (t.x > HT_W - HT_RADIUS) { t.x = HT_W - HT_RADIUS; t.heading = Math.PI - t.heading; }
    if (t.y < HT_RADIUS) { t.y = HT_RADIUS; t.heading = -t.heading; }
    if (t.y > HT_H - HT_RADIUS) { t.y = HT_H - HT_RADIUS; t.heading = -t.heading; }

    const p = pointerRef.current;
    const m = metricsRef.current;
    m.totalFrames += 1;
    if (p.inside) {
      const dist = Math.hypot(p.x - t.x, p.y - t.y);
      m.insideFrames += 1;
      m.sumDist += dist;
      if (dist <= HT_RADIUS) m.hitFrames += 1;
    }
    // 指针不在框内一律算脱靶，否则「把鼠标甩出去」反而成了占便宜
  }, []);

  const drawHt = useCallback((ctx: CanvasRenderingContext2D) => {
    const t = targetRef.current;
    const p = pointerRef.current;
    // 与 MOT 同款做法：逻辑坐标固定 360×260，放大量只放大后备存储
    const s = ctx.canvas.width / HT_W;
    ctx.setTransform(s, 0, 0, s, 0, 0);
    ctx.clearRect(0, 0, HT_W, HT_H);
    ctx.fillStyle = '#fbfaff';
    ctx.fillRect(0, 0, HT_W, HT_H);

    // 目标容差圈
    ctx.beginPath();
    ctx.arc(t.x, t.y, HT_RADIUS, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(114,46,209,0.12)';
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#722ed1';
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(t.x, t.y, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#722ed1';
    ctx.fill();

    if (p.inside) {
      const on = Math.hypot(p.x - t.x, p.y - t.y) <= HT_RADIUS;
      ctx.beginPath();
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = on ? 'rgba(82,196,26,0.75)' : 'rgba(255,77,79,0.6)';
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(t.x, t.y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
      ctx.fillStyle = on ? '#52c41a' : '#ff8fab';
      ctx.fill();
    }
  }, []);

  // idle 时画静态画面（同时负责结束后把画布复位）
  useEffect(() => {
    if (phase !== 'idle') return;
    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx) return;
    resetField();
    drawHt(ctx);
  }, [phase, isExpanded, resetField, drawHt]);

  useEffect(() => {
    if (phase !== 'running') return;
    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx) return;
    let last = 0;
    let id = 0;
    const frame = (t: number) => {
      const dt = last ? Math.min((t - last) / 1000, 0.05) : 0;
      last = t;
      step(dt);
      drawHt(ctx);
      id = requestAnimationFrame(frame);
    };
    id = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(id);
  }, [phase, step, drawHt]);

  const handleMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const p = pointerRef.current;
    p.x = (e.clientX - rect.left) * (HT_W / rect.width);
    p.y = (e.clientY - rect.top) * (HT_H / rect.height);
    p.inside = true;
  };

  const start = useCallback(() => {
    resetField();
    metricsRef.current = emptyMetrics();
    setResult(null);
    setLiveHit(0);
    setTimeLeft(HT_DURATION);
    recordedRef.current = false;
    startedAtRef.current = Date.now();
    clearEnded();
    const token = run.begin();
    setPhaseSafe('running');
    let left = HT_DURATION;
    run.trackTimer(setInterval(() => {
      if (!run.isActive(token)) return;
      left = Math.max(0, left - 0.2);
      setTimeLeft(left);
      const m = metricsRef.current;
      setLiveHit(m.totalFrames > 0 ? m.hitFrames / m.totalFrames : 0);
      if (left <= 0) {
        setResult({
          hitRate: m.totalFrames > 0 ? m.hitFrames / m.totalFrames : 0,
          meanDist: m.insideFrames > 0 ? m.sumDist / m.insideFrames : 0,
          frames: m.totalFrames,
        });
        run.interrupt();
        setPhaseSafe('done');
      }
    }, 200));
  }, [resetField, clearEnded, run, setPhaseSafe]);

  const endGame = useCallback(() => {
    if (phaseRef.current === 'idle') return;
    run.interrupt();
    recordedRef.current = true;
    metricsRef.current = emptyMetrics();
    setResult(null);
    setLiveHit(0);
    setTimeLeft(HT_DURATION);
    setPhaseSafe('idle');
    flashEnded();
  }, [run, setPhaseSafe, flashEnded]);

  const isRunning = phase === 'running';
  useEscapeToEnd(endGame, isRunning);
  useAbortOnTabChange(endGame, isRunning);

  useEffect(() => {
    if (phase !== 'done' || recordedRef.current || !result) return;
    recordedRef.current = true;
    const percent = Math.round(result.hitRate * 100);
    submitBest(percent);
    recordGameResult('tracking', '跟手追踪', HT_DURATION, {
      hitRate: Math.round(result.hitRate * 1000) / 1000,
      meanDist: Math.round(result.meanDist * 10) / 10,
      sampleFrames: result.frames,
      durationSec: HT_DURATION,
    });
  }, [phase, result, submitBest]);

  return (
    <div style={{ textAlign: 'center' }}>
      <Text type="secondary" style={{ display: 'block', marginBottom: 10, fontSize: 13, lineHeight: 1.7 }}>
        把鼠标贴住游走的紫球，尽量别离开圈。目标会不断变向，稳住而不是猛甩。
      </Text>

      <div style={{ display: 'flex', justifyContent: 'center', gap: 32, marginBottom: 10 }}>
        <div>
          <div style={{ fontSize: 12, color: '#999' }}>剩余</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: '#722ed1' }}>{timeLeft.toFixed(1)} s</div>
        </div>
        <div>
          <div style={{ fontSize: 12, color: '#999' }}>命中率</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: liveHit >= 0.6 ? '#52c41a' : '#fa8c16' }}>
            {Math.round(liveHit * 100)}%
          </div>
        </div>
        {best !== null && (
          <div>
            <div style={{ fontSize: 12, color: '#999' }}>个人最佳</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: '#52c41a' }}>{best}%</div>
          </div>
        )}
      </div>

      <canvas
        ref={canvasRef}
        className="tracking-canvas"
        width={HT_W * htScale}
        height={HT_H * htScale}
        onPointerMove={handleMove}
        onPointerDown={handleMove}
        onPointerLeave={() => { pointerRef.current.inside = false; }}
        style={{
          width: '100%', maxWidth: isExpanded ? 560 : HT_W, height: 'auto', display: 'block',
          margin: '0 auto', borderRadius: 16, border: '2px solid #f0f0f0',
          background: '#fbfaff', touchAction: 'none',
          cursor: phase === 'running' ? 'none' : 'default',
        }}
      />

      {phase === 'done' && result && (
        <div style={{ marginTop: 12 }}>
          <ReactionReport
            title="本局成绩"
            tone="#1890ff"
            rows={[
              { label: '命中率', value: `${Math.round(result.hitRate * 100)}%` },
              { label: '平均偏差', value: `${Math.round(result.meanDist)} px` },
              { label: '个人最佳', value: best !== null ? `${best}%` : '—' },
            ]}
            note={
              `命中率 = 鼠标停在圈内的时间占比；平均偏差 = 鼠标到球心的平均距离（容差 ${HT_RADIUS}px）。` +
              '命中率上不去、偏差又不大，通常是「追得太急」；偏差大但命中率还行，说明反应够快但控不住。' +
              '这个任务考验的是持续微调，和瞬间反应是两种不同的能力。'
            }
          />
        </div>
      )}

      <GameActions onEnd={endGame} endDisabled={phase === 'idle'} ended={ended}>
        <Button type="primary" onClick={start} style={{ borderRadius: 20 }}>
          {phase === 'idle' ? '开始' : '重新开始'}
        </Button>
      </GameActions>
    </div>
  );
}

// ============================================================
// 10. 选择反应（Simon 冲突）
// ============================================================
const SIMON_TOTAL = 24;
const SIMON_WINDOW = 1500;  // 反应窗
const SIMON_WAIT_MIN = 900;
const SIMON_WAIT_MAX = 1800;

type SimonSide = 'left' | 'right';
interface SimonTrial { cue: SimonSide; side: SimonSide; }
interface SimonLog extends SimonTrial { outcome: 'hit' | 'wrong' | 'miss' | 'anticipate'; rt: number | null; }

/** 颜色决定按哪边 —— 与出现位置无关，这正是冲突的来源 */
const SIMON_CUE: Record<SimonSide, { name: string; color: string; arrow: string }> = {
  left: { name: '红', color: '#f5222d', arrow: '←' },
  right: { name: '蓝', color: '#1890ff', arrow: '→' },
};

/**
 * 生成试次：一致（红在左 / 蓝在右）与冲突（红在右 / 蓝在左）各一半。
 * 第一个固定为一致试次 —— 一上来就撞冲突会让规则还没建立就先出错。
 */
function buildSimonPlan(total: number): SimonTrial[] {
  const congruentQuota = Math.round(total / 2);
  const plan: SimonTrial[] = [];
  for (let i = 0; i < total; i++) {
    const cue: SimonSide = Math.random() < 0.5 ? 'left' : 'right';
    const congruent = i < congruentQuota;
    plan.push({ cue, side: congruent ? cue : (cue === 'left' ? 'right' : 'left') });
  }
  for (let i = plan.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [plan[i], plan[j]] = [plan[j], plan[i]];
  }
  if (plan[0].cue !== plan[0].side) {
    const k = plan.findIndex(t => t.cue === t.side);
    if (k > 0) [plan[0], plan[k]] = [plan[k], plan[0]];
  }
  return plan;
}

/**
 * 选择反应（Simon 冲突）
 * 圆点随机出现在中心左侧或右侧，但**按哪边由颜色决定**，不由位置决定。
 * 两者一致时是普通选择反应；冲突时位置会强烈地"拉"你去按同侧，需要把这个
 * 优势反应压下去 —— 冲突试次比一致试次慢多少毫秒，就是冲突代价。
 */
function ChoiceReaction() {
  const clock = useReactionClock();
  const run = useInterruptible();
  const { feedback, flash, clear: clearFlash } = useFlash();

  const [phase, setPhase] = useState<'idle' | 'wait' | 'show' | 'done'>('idle');
  const phaseRef = useRef<'idle' | 'wait' | 'show' | 'done'>('idle');
  const [stim, setStim] = useState<SimonTrial | null>(null);
  const [logs, setLogs] = useState<SimonLog[]>([]);
  const [cursor, setCursor] = useState(0);
  const [ended, flashEnded, clearEnded] = useEndedFlag();
  const isExpanded = useIsExpanded();

  const planRef = useRef<SimonTrial[]>([]);
  const cursorRef = useRef(0);
  const logsRef = useRef<SimonLog[]>([]);
  const recordedRef = useRef(false);
  const startedAtRef = useRef(0);
  const { best, submit: submitBest } = usePersonalBest('simon', false); // 反应时越低越好

  const setPhaseSafe = useCallback((p: 'idle' | 'wait' | 'show' | 'done') => {
    phaseRef.current = p;
    setPhase(p);
  }, []);

  /** 记一笔并前进，返回是否已跑完 */
  const record = useCallback((outcome: SimonLog['outcome'], rt: number | null) => {
    const trial = planRef.current[cursorRef.current];
    if (!trial) return true;
    logsRef.current = [...logsRef.current, { ...trial, outcome, rt }];
    setLogs(logsRef.current);
    cursorRef.current += 1;
    setCursor(cursorRef.current);
    return cursorRef.current >= planRef.current.length;
  }, []);

  const advanceRef = useRef<() => void>(() => {});
  const advance = useCallback(() => {
    const token = run.begin();   // 清掉上一试次残留的定时器并作废旧回调
    clock.clear();
    const trial = planRef.current[cursorRef.current];
    if (!trial) { setStim(null); setPhaseSafe('done'); return; }
    setStim(null);
    setPhaseSafe('wait');
    const delay = SIMON_WAIT_MIN + Math.random() * (SIMON_WAIT_MAX - SIMON_WAIT_MIN);
    run.trackTimer(setTimeout(() => {
      if (!run.isActive(token)) return;
      setStim(trial);
      setPhaseSafe('show');
      clock.armOnNextFrame();
      run.trackTimer(setTimeout(() => {
        if (!run.isActive(token)) return;
        flash('没反应', '#fa8c16');
        record('miss', null);
        advanceRef.current();
      }, SIMON_WINDOW));
    }, delay));
  }, [run, clock, setPhaseSafe, flash, record]);

  useEffect(() => { advanceRef.current = advance; }, [advance]);

  const respond = useCallback((side: SimonSide, e?: { timeStamp?: number }) => {
    const p = phaseRef.current;
    const trial = planRef.current[cursorRef.current];
    if (!trial || p === 'idle' || p === 'done') return;
    if (p === 'wait') {
      flash('还没出现，抢跑了', '#fa541c');
      record('anticipate', null);
      advanceRef.current();
      return;
    }
    const ms = clock.elapsed(e);
    if (ms === null) return;
    if (ms < RT_MIN_VALID) {
      flash('太快了，算抢跑', '#fa541c');
      record('anticipate', null);
    } else if (side === trial.cue) {
      flash(`${ms} ms`, '#52c41a');
      record('hit', ms);
    } else {
      flash(`按错了 · 该按 ${SIMON_CUE[trial.cue].arrow}`, '#fa541c');
      record('wrong', ms);
    }
    advanceRef.current();
  }, [flash, record, clock]);

  const start = useCallback(() => {
    planRef.current = buildSimonPlan(SIMON_TOTAL);
    cursorRef.current = 0;
    logsRef.current = [];
    setLogs([]);
    setCursor(0);
    recordedRef.current = false;
    startedAtRef.current = Date.now();
    clearFlash();
    clearEnded();
    advanceRef.current();
  }, [clearFlash, clearEnded]);

  const endGame = useCallback(() => {
    if (phaseRef.current === 'idle') return;
    run.interrupt();
    clock.clear();
    planRef.current = [];
    cursorRef.current = 0;
    logsRef.current = [];
    setLogs([]);
    setCursor(0);
    setStim(null);
    recordedRef.current = true;
    clearFlash();
    setPhaseSafe('idle');
    flashEnded();
  }, [run, clock, clearFlash, setPhaseSafe, flashEnded]);

  const isRunning = phase === 'wait' || phase === 'show';
  useEscapeToEnd(endGame, isRunning);
  useAbortOnTabChange(endGame, isRunning);

  // 键盘：方向键或 A / D，左右各一条
  useEffect(() => {
    if (!isRunning) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.repeat) return;
      let side: SimonSide | null = null;
      if (e.code === 'ArrowLeft' || e.code === 'KeyA') side = 'left';
      if (e.code === 'ArrowRight' || e.code === 'KeyD') side = 'right';
      if (!side) return;
      e.preventDefault();
      respond(side, { timeStamp: e.timeStamp });
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isRunning, respond]);

  const stats = useMemo(() => {
    const hits = logs.filter(l => l.outcome === 'hit');
    const conRt = summarizeRt(hits.filter(l => l.cue === l.side).map(l => l.rt as number));
    const incRt = summarizeRt(hits.filter(l => l.cue !== l.side).map(l => l.rt as number));
    const anticipate = logs.filter(l => l.outcome === 'anticipate').length;
    const valid = logs.length - anticipate;
    return {
      done: logs.length,
      valid,
      hit: hits.length,
      wrong: logs.filter(l => l.outcome === 'wrong').length,
      miss: logs.filter(l => l.outcome === 'miss').length,
      anticipate,
      accuracy: valid > 0 ? hits.length / valid : 0,
      rt: summarizeRt(hits.map(l => l.rt as number)),
      conRt,
      incRt,
      cost: conRt && incRt ? incRt.mean - conRt.mean : null,
    };
  }, [logs]);

  useEffect(() => {
    if (phase !== 'done' || recordedRef.current) return;
    recordedRef.current = true;
    const s = stats;
    if (!s.rt) return;
    submitBest(s.rt.mean);
    recordGameResult('simon', '选择反应', (Date.now() - startedAtRef.current) / 1000, {
      trials: s.done, valid: s.valid, hit: s.hit, wrong: s.wrong, miss: s.miss, anticipate: s.anticipate,
      accuracy: Math.round(s.accuracy * 1000) / 1000,
      meanRt: s.rt.mean, bestRt: s.rt.best, sdRt: s.rt.sd,
      congruentRt: s.conRt ? s.conRt.mean : null,
      incongruentRt: s.incRt ? s.incRt.mean : null,
      conflictCost: s.cost,
    });
  }, [phase, stats, submitBest]);

  const stageW = isExpanded ? 620 : 420;
  const stageH = isExpanded ? 320 : 220;
  const discSize = isExpanded ? 96 : 72;
  const offset = stageW * 0.28;

  const disc = stim && (
    <div className="simon-disc" style={{
      position: 'absolute', top: '50%', left: '50%',
      width: discSize, height: discSize, borderRadius: '50%',
      background: SIMON_CUE[stim.cue].color,
      transform: `translate(-50%, -50%) translateX(${stim.side === 'left' ? -offset : offset}px)`,
      boxShadow: `0 8px 22px ${SIMON_CUE[stim.cue].color}55`,
    }} />
  );

  const rt = phase === 'done' ? stats.rt : null;

  return (
    <div style={{ textAlign: 'center' }}>
      <Text type="secondary" style={{ display: 'block', marginBottom: 10, fontSize: 13, lineHeight: 1.7 }}>
        圆点在左边还是右边<strong>不重要</strong>，按哪边只看颜色：
        <strong style={{ color: '#f5222d' }}>红 → ←</strong>、
        <strong style={{ color: '#1890ff' }}>蓝 → →</strong>。
      </Text>

      <div style={{ display: 'flex', justifyContent: 'center', gap: 30, marginBottom: 10 }}>
        <div>
          <div style={{ fontSize: 12, color: '#999' }}>进度</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: '#722ed1' }}>
            {phase === 'idle' ? 0 : Math.min(cursor + (phase === 'done' ? 0 : 1), SIMON_TOTAL)} / {SIMON_TOTAL}
          </div>
        </div>
        {best !== null && (
          <div>
            <div style={{ fontSize: 12, color: '#999' }}>个人最快均值</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: '#52c41a' }}>{best} ms</div>
          </div>
        )}
        {stats.rt && phase !== 'done' && (
          <div>
            <div style={{ fontSize: 12, color: '#999' }}>本次均速</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: '#1890ff' }}>{stats.rt.mean} ms</div>
          </div>
        )}
      </div>

      <div style={{
        position: 'relative', width: '100%', maxWidth: stageW, height: stageH,
        margin: '0 auto', borderRadius: 18,
        background: phase === 'wait' || phase === 'show' ? '#33334d' : '#f0f0f5',
        overflow: 'hidden',
      }}>
        {phase === 'wait' && (
          <div style={{
            position: 'absolute', top: '50%', left: '50%', width: 10, height: 10,
            borderRadius: '50%', background: 'rgba(255,255,255,0.55)',
            transform: 'translate(-50%, -50%)',
          }} />
        )}
        {disc}
        {phase === 'idle' && (
          <div style={{
            position: 'absolute', inset: 0, display: 'flex', alignItems: 'center',
            justifyContent: 'center', color: '#8c8c8c', fontWeight: 600,
          }}>
            点下面的「开始」
          </div>
        )}
      </div>

      <div style={{ height: 24, marginTop: 6 }}>
        {feedback && <Text style={{ color: feedback.tone, fontWeight: 600 }}>{feedback.text}</Text>}
      </div>

      <div style={{ display: 'flex', justifyContent: 'center', gap: 16, marginBottom: 4 }}>
        {(['left', 'right'] as SimonSide[]).map(side => (
          <Button
            key={side}
            size="large"
            disabled={!isRunning}
            onPointerDown={e => { e.preventDefault(); respond(side, e); }}
            style={{
              width: isExpanded ? 150 : 116, height: isExpanded ? 64 : 52,
              borderRadius: 16, fontSize: 20, fontWeight: 700,
              borderColor: SIMON_CUE[side].color, color: SIMON_CUE[side].color,
            }}
          >
            {SIMON_CUE[side].arrow} {SIMON_CUE[side].name}
          </Button>
        ))}
      </div>

      {rt && (
        <ReactionReport
          title="本局成绩"
          rows={[
            { label: '平均反应', value: `${rt.mean} ms` },
            { label: '最快', value: `${rt.best} ms` },
            { label: '稳定性', value: `±${rt.sd} ms` },
            { label: '正确率', value: `${Math.round(stats.accuracy * 100)}%` },
            { label: '一致试次', value: stats.conRt ? `${stats.conRt.mean} ms` : '—' },
            { label: '冲突试次', value: stats.incRt ? `${stats.incRt.mean} ms` : '—' },
          ]}
          note={
            (stats.cost !== null
              ? `冲突代价 ${stats.cost >= 0 ? '+' : ''}${stats.cost} ms —— 位置和颜色打架时，`
                + `你的反应比它们一致时${stats.cost >= 0 ? '慢' : '快'}了 ${Math.abs(stats.cost)} ms。`
              : '样本还不够，冲突代价这次没算出来。')
            + `共 ${stats.hit} 对 / ${stats.wrong} 错 / ${stats.miss} 漏 / ${stats.anticipate} 次抢跑，`
            + `正确率只统计有效试次（抢跑作废）。`
            + '冲突代价是这套任务里最有意思的数：它衡量的是"忍住不按最顺手那一下"要花多少力气。'
            + '一般几十毫秒都算正常，状态差、疲劳的时候这个数往往会变大。'
            + RT_DISCLAIMER
          }
        />
      )}

      <GameActions onEnd={endGame} endDisabled={phase === 'idle'} ended={ended}>
        <Button type="primary" onClick={start} style={{ borderRadius: 20 }}>
          {phase === 'idle' ? '开始' : '重新开始'}
        </Button>
      </GameActions>
    </div>
  );
}

// ============================================================
// 11. 甩点挑战（Flicking）
// ============================================================
const FLICK_DURATION = 30;
const FLICK_MIN_GAP = 0.32;    // 与上一个目标的最小间距（占舞台的宽/高比例）
const FLICK_DECOY_RATE = 0.2;  // 红点比例：点了要扣分
const FLICK_DECOY_MS = 850;    // 红点只闪这么久

interface FlickTarget { id: number; x: number; y: number; decoy: boolean; }

/** 甩点挑战：目标随机跳，越快越准越好；偶尔出现的红点不能点 */
function FlickingTask() {
  const run = useInterruptible();
  const { flash } = useFlash();
  const [phase, setPhase] = useState<'idle' | 'running' | 'done'>('idle');
  const phaseRef = useRef<'idle' | 'running' | 'done'>('idle');
  const [target, setTarget] = useState<FlickTarget | null>(null);
  const [timeLeft, setTimeLeft] = useState(FLICK_DURATION);
  const [hits, setHits] = useState(0);
  const [decoyHits, setDecoyHits] = useState(0);
  const [ended, flashEnded, clearEnded] = useEndedFlag();
  const isExpanded = useIsExpanded();

  const hitsRef = useRef(0);
  const decoyRef = useRef(0);
  const idRef = useRef(0);
  const lastPosRef = useRef({ x: 0.5, y: 0.5 });
  const lastHitAtRef = useRef(0);
  const intervalSumRef = useRef(0);
  const intervalCountRef = useRef(0);
  const recordedRef = useRef(false);
  const startedAtRef = useRef(0);
  const { best, submit: submitBest } = usePersonalBest('flick', true); // 命中数越多越好

  const setPhaseSafe = useCallback((p: 'idle' | 'running' | 'done') => {
    phaseRef.current = p;
    setPhase(p);
  }, []);

  const decoyTimerRef = useRef<any>(null);
  const clearDecoyTimer = useCallback(() => {
    if (decoyTimerRef.current) { clearTimeout(decoyTimerRef.current); decoyTimerRef.current = null; }
  }, []);

  const nextTargetRef = useRef<() => void>(() => {});
  const nextTarget = useCallback(() => {
    clearDecoyTimer();
    let x = 0.5, y = 0.5;
    // 强制拉开距离，逼出"甩"而不是原地微调
    for (let i = 0; i < 24; i++) {
      x = 0.09 + Math.random() * 0.82;
      y = 0.1 + Math.random() * 0.8;
      if (Math.hypot(x - lastPosRef.current.x, y - lastPosRef.current.y) >= FLICK_MIN_GAP) break;
    }
    lastPosRef.current = { x, y };
    idRef.current += 1;
    const id = idRef.current;
    const decoy = Math.random() < FLICK_DECOY_RATE;
    setTarget({ id, x, y, decoy });
    if (decoy) {
      // 红点必须自己消失。第一版让它一直挂着，结果只要出现红点、玩家又守规矩不点，
      // 局面就永远卡在那里 —— 唯一能继续的办法是"故意点错"，等于反向激励。
      decoyTimerRef.current = setTimeout(() => {
        decoyTimerRef.current = null;
        if (idRef.current === id && phaseRef.current === 'running') nextTargetRef.current();
      }, FLICK_DECOY_MS);
    }
  }, [clearDecoyTimer]);

  useEffect(() => { nextTargetRef.current = nextTarget; }, [nextTarget]);
  useEffect(() => clearDecoyTimer, [clearDecoyTimer]);

  const start = useCallback(() => {
    hitsRef.current = 0;
    decoyRef.current = 0;
    intervalSumRef.current = 0;
    intervalCountRef.current = 0;
    lastPosRef.current = { x: 0.5, y: 0.5 };
    lastHitAtRef.current = 0;
    recordedRef.current = false;
    startedAtRef.current = Date.now();
    setHits(0);
    setDecoyHits(0);
    setTimeLeft(FLICK_DURATION);
    clearEnded();
    const token = run.begin();
    setPhaseSafe('running');
    nextTarget();
    let left = FLICK_DURATION;
    run.trackTimer(setInterval(() => {
      if (!run.isActive(token)) return;
      left = Math.max(0, left - 0.1);
      setTimeLeft(left);
      if (left <= 0) {
        run.interrupt();
        setTarget(null);
        setPhaseSafe('done');
      }
    }, 100));
  }, [run, setPhaseSafe, clearEnded, nextTarget]);

  const endGame = useCallback(() => {
    if (phaseRef.current === 'idle') return;
    run.interrupt();
    clearDecoyTimer();
    recordedRef.current = true;
    hitsRef.current = 0;
    decoyRef.current = 0;
    intervalSumRef.current = 0;
    intervalCountRef.current = 0;
    setTarget(null);
    setHits(0);
    setDecoyHits(0);
    setTimeLeft(FLICK_DURATION);
    setPhaseSafe('idle');
    flashEnded();
  }, [run, clearDecoyTimer, setPhaseSafe, flashEnded]);

  const hitTarget = useCallback((t: FlickTarget) => {
    if (phaseRef.current !== 'running') return;
    const now = performance.now();
    if (t.decoy) {
      decoyRef.current += 1;
      setDecoyHits(decoyRef.current);
      flash('红色不要点！', '#fa541c');
    } else {
      hitsRef.current += 1;
      setHits(hitsRef.current);
      if (lastHitAtRef.current > 0) {
        intervalSumRef.current += now - lastHitAtRef.current;
        intervalCountRef.current += 1;
      }
      lastHitAtRef.current = now;
      flash(`第 ${hitsRef.current} 个`, '#52c41a');
    }
    nextTarget();
  }, [flash, nextTarget]);

  const isRunning = phase === 'running';
  useEscapeToEnd(endGame, isRunning);
  useAbortOnTabChange(endGame, isRunning);

  useEffect(() => {
    if (phase !== 'done' || recordedRef.current) return;
    recordedRef.current = true;
    const meanInterval = intervalCountRef.current > 0
      ? Math.round(intervalSumRef.current / intervalCountRef.current) : null;
    submitBest(hitsRef.current);
    recordGameResult('flick', '甩点挑战', FLICK_DURATION, {
      durationSec: FLICK_DURATION,
      hits: hitsRef.current,
      decoyHits: decoyRef.current,
      meanIntervalMs: meanInterval,
    });
  }, [phase, submitBest]);

  const meanInterval = intervalCountRef.current > 0
    ? Math.round(intervalSumRef.current / intervalCountRef.current) : null;
  const size = isExpanded ? 64 : 46;
  const stageH = isExpanded ? 460 : 320;

  return (
    <div style={{ textAlign: 'center' }}>
      <Text type="secondary" style={{ display: 'block', marginBottom: 10, fontSize: 13, lineHeight: 1.7 }}>
        30 秒内点掉尽可能多的<strong style={{ color: '#722ed1' }}>紫色</strong>小球，
        <strong style={{ color: '#f5222d' }}>红色</strong>的千万别点。目标会跳得很远，靠甩。
      </Text>

      <div style={{ display: 'flex', justifyContent: 'center', gap: 30, marginBottom: 10 }}>
        <div>
          <div style={{ fontSize: 12, color: '#999' }}>剩余</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: '#722ed1' }}>{timeLeft.toFixed(1)} s</div>
        </div>
        <div>
          <div style={{ fontSize: 12, color: '#999' }}>命中</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: '#52c41a' }}>{hits}</div>
        </div>
        <div>
          <div style={{ fontSize: 12, color: '#999' }}>误点红点</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: decoyHits > 0 ? '#fa541c' : '#bbb' }}>{decoyHits}</div>
        </div>
        {best !== null && (
          <div>
            <div style={{ fontSize: 12, color: '#999' }}>个人最佳</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: '#faad14' }}>{best}</div>
          </div>
        )}
      </div>

      <div style={{
        position: 'relative', width: '100%', maxWidth: isExpanded ? 720 : 460,
        height: stageH, margin: '0 auto', borderRadius: 16,
        border: '2px solid #f0f0f0', background: '#fbfaff', overflow: 'hidden',
        touchAction: 'none', cursor: isRunning ? 'crosshair' : 'default',
      }}>
        {phase === 'idle' && (
          <div style={{
            position: 'absolute', inset: 0, display: 'flex', alignItems: 'center',
            justifyContent: 'center', color: '#8c8c8c', fontWeight: 600,
          }}>
            点下面的「开始」
          </div>
        )}
        {isRunning && target && (
          <button
            key={target.id}
            aria-label={target.decoy ? '红色目标' : '紫色目标'}
            onPointerDown={e => { e.preventDefault(); hitTarget(target); }}
            style={{
              position: 'absolute',
              left: `${target.x * 100}%`, top: `${target.y * 100}%`,
              width: size, height: size, marginLeft: -size / 2, marginTop: -size / 2,
              borderRadius: '50%', border: 'none', padding: 0, cursor: 'pointer',
              background: target.decoy
                ? 'radial-gradient(circle at 34% 30%, #ff9a9e, #f5222d)'
                : 'radial-gradient(circle at 34% 30%, #c4b1ff, #722ed1)',
              boxShadow: target.decoy
                ? '0 6px 16px rgba(245,34,45,0.35)' : '0 6px 16px rgba(114,46,209,0.35)',
            }}
          />
        )}
      </div>

      {phase === 'done' && (
        <div style={{ marginTop: 12 }}>
          <ReactionReport
            title="本局成绩"
            tone="#52c41a"
            rows={[
              { label: '命中', value: `${hits} 个` },
              { label: '平均间隔', value: meanInterval ? `${meanInterval} ms` : '—' },
              { label: '误点红点', value: `${decoyHits} 次` },
              { label: '个人最佳', value: best !== null ? `${best} 个` : '—' },
            ]}
            note={
              `平均间隔是连续两次命中之间的时间：${meanInterval ?? '—'} ms，越小说明手眼衔接越快。`
              + '它和反应时的区别在于——反应时测"看到到按下"，这里还额外包含了眼睛找下一个目标和把鼠标移过去的全过程。'
              + '误点红点次数反映的是"先点了再说"的倾向，和反应灯里误按红灯是一回事。'
            }
          />
        </div>
      )}

      <GameActions onEnd={endGame} endDisabled={phase === 'idle'} ended={ended}>
        <Button type="primary" onClick={start} style={{ borderRadius: 20 }}>
          {phase === 'idle' ? '开始' : '重新开始'}
        </Button>
      </GameActions>
    </div>
  );
}

// ============================================================
// 12. 节奏打击（预计时机 CAT）
// ============================================================
const CAT_SIZE = 320;
const CAT_R = 118;              // 轨道半径
const CAT_MS_PER_TURN = 2400;   // 指针转一圈的毫秒数
const CAT_OMEGA = (Math.PI * 2) / CAT_MS_PER_TURN; // 角速度 rad/ms
const CAT_TRIALS = 12;
const CAT_WINDOW_MS = 100;      // 命中判定：误差 ±100ms 以内
const CAT_MISS_MS = 320;        // 越过目标这么多还没按，算漏
const CAT_GAP_MS = 900;         // 两次之间展示结果的时间
const CAT_LEAD_MS = 1800;       // 目标提前多久就出现（3/4 圈：既留够准备时间，又不会和指针起点重合）

/** 角度归一到 [0, 2π)：只在画图时用，算误差时不能取模 */
const wrapAngle = (a: number) => ((a % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2);

interface CatLog { error: number; hit: boolean; }  // error > 0 = 滞后，< 0 = 提前

/**
 * 节奏打击（Coincidence Anticipation Timing）
 * 指针匀速绕圈，绿色扇区就是时机窗，要求指针扫过时按下。
 * 误差单位是毫秒，而且有方向：负数是提前、正数是滞后 ——
 * "总是早一点"和"总是晚一点"是两种很不一样的倾向。
 */
function RhythmTiming() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const run = useInterruptible();
  const { feedback, flash, clear: clearFlash } = useFlash();

  const [phase, setPhase] = useState<'idle' | 'run' | 'gap' | 'done'>('idle');
  const phaseRef = useRef<'idle' | 'run' | 'gap' | 'done'>('idle');
  const [trialNo, setTrialNo] = useState(0);
  const [logs, setLogs] = useState<CatLog[]>([]);
  const [lastError, setLastError] = useState<number | null>(null);
  const [ended, flashEnded, clearEnded] = useEndedFlag();
  const isExpanded = useIsExpanded();

  const runStartRef = useRef(0);
  const targetRef = useRef(0);
  const logsRef = useRef<CatLog[]>([]);
  const markedRef = useRef(false);   // 本轮是否已经判定过
  const recordedRef = useRef(false);
  const startedAtRef = useRef(0);
  const { best, submit: submitBest } = usePersonalBest('cat', false); // 平均绝对误差越小越好

  const setPhaseSafe = useCallback((p: 'idle' | 'run' | 'gap' | 'done') => {
    phaseRef.current = p;
    setPhase(p);
  }, []);

  /** 指针的「未取模」角度 */
  const rawAngle = (now: number) => (now - runStartRef.current) * CAT_OMEGA;

  /**
   * 指针相对目标的带符号误差（毫秒）：负 = 还没到（提前），正 = 已越过（滞后）
   *
   * 这里刻意用未取模的角度差。指针每圈都会经过目标角，一旦取模，
   * 「目标在前方 1.5 圈」会被折算成「刚过去 0.5 圈」，符号翻转，
   * 误差算出来是 +1200ms，第一帧就直接判成错过 —— 实测踩过这个坑。
   */
  const errorMs = useCallback((now: number) => Math.round((rawAngle(now) - targetRef.current) / CAT_OMEGA), []);
  const errRef = useRef(errorMs);
  useEffect(() => { errRef.current = errorMs; }, [errorMs]);

  const settle = useCallback((error: number | null) => {
    if (markedRef.current) return;
    markedRef.current = true;
    const hit = error !== null && Math.abs(error) <= CAT_WINDOW_MS;
    logsRef.current = [...logsRef.current, { error: error ?? CAT_MISS_MS, hit }];
    setLogs(logsRef.current);
    setLastError(error);
    if (error === null) flash('错过了这一圈', '#fa8c16');
    else if (hit) flash(`${error > 0 ? '+' : ''}${error} ms`, '#52c41a');
    else flash(`${error > 0 ? '晚了' : '早了'} ${Math.abs(error)} ms`, '#fa8c16');

    const token = run.begin();
    setPhaseSafe('gap');
    run.trackTimer(setTimeout(() => {
      if (!run.isActive(token)) return;
      if (logsRef.current.length >= CAT_TRIALS) { setPhaseSafe('done'); return; }
      // 下一轮：目标放到指针前方一圈的位置（用未取模角度，避免最短路径歧义）
      targetRef.current = rawAngle(performance.now()) + CAT_LEAD_MS * CAT_OMEGA;
      markedRef.current = false;
      setLastError(null);
      setTrialNo(logsRef.current.length + 1);
      setPhaseSafe('run');
    }, CAT_GAP_MS));
  }, [run, setPhaseSafe, flash]);
  const settleRef = useRef(settle);
  useEffect(() => { settleRef.current = settle; }, [settle]);

  const draw = useCallback((ctx: CanvasRenderingContext2D, now: number, phaseNow: string) => {
    const s = ctx.canvas.width / CAT_SIZE;
    ctx.setTransform(s, 0, 0, s, 0, 0);
    ctx.clearRect(0, 0, CAT_SIZE, CAT_SIZE);
    ctx.fillStyle = '#fbfaff';
    ctx.fillRect(0, 0, CAT_SIZE, CAT_SIZE);
    const cx = CAT_SIZE / 2, cy = CAT_SIZE / 2;

    // 轨道
    ctx.beginPath();
    ctx.arc(cx, cy, CAT_R, 0, Math.PI * 2);
    ctx.lineWidth = 8;
    ctx.strokeStyle = '#eee6fb';
    ctx.stroke();

    // 目标扇区：宽度对应 ±CAT_WINDOW_MS，位置是目标角的取模值
    if (phaseNow !== 'idle') {
      const half = CAT_WINDOW_MS * CAT_OMEGA;
      const tAng = wrapAngle(targetRef.current) - Math.PI / 2;
      ctx.beginPath();
      ctx.arc(cx, cy, CAT_R, tAng - half, tAng + half);
      ctx.lineWidth = 12;
      ctx.strokeStyle = phaseNow === 'run' ? '#52c41a' : 'rgba(82,196,26,0.4)';
      ctx.stroke();
    }

    if (phaseNow === 'idle') return;

    // 指针
    const a = wrapAngle(rawAngle(now)) - Math.PI / 2;
    const px = cx + Math.cos(a) * CAT_R, py = cy + Math.sin(a) * CAT_R;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(px, py);
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#722ed1';
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(px, py, 10, 0, Math.PI * 2);
    ctx.fillStyle = '#722ed1';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#722ed1';
    ctx.fill();
  }, []);
  const drawRef = useRef(draw);
  useEffect(() => { drawRef.current = draw; }, [draw]);

  // 只在有回合时开动画帧
  useEffect(() => {
    if (phase === 'idle' || phase === 'done') return;
    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx) return;
    let id = 0;
    const frame = (t: number) => {
      drawRef.current(ctx, t, phaseRef.current);
      // 指针越过目标太多 => 本轮漏掉
      if (phaseRef.current === 'run' && errRef.current(t) > CAT_MISS_MS) settleRef.current(null);
      id = requestAnimationFrame(frame);
    };
    id = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(id);
  }, [phase]);

  // idle / done 时画一张静态图
  useEffect(() => {
    if (phase !== 'idle' && phase !== 'done') return;
    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx) return;
    drawRef.current(ctx, performance.now(), 'idle');
  }, [phase]);

  const press = useCallback((e?: { timeStamp?: number }) => {
    if (phaseRef.current !== 'run') return;
    const now = typeof e?.timeStamp === 'number' && e.timeStamp > 0 ? e.timeStamp : performance.now();
    settleRef.current(errRef.current(now));
  }, []);

  const start = useCallback(() => {
    logsRef.current = [];
    setLogs([]);
    setLastError(null);
    setTrialNo(1);
    recordedRef.current = false;
    startedAtRef.current = Date.now();
    clearFlash();
    clearEnded();
    run.begin();
    runStartRef.current = performance.now();
    targetRef.current = CAT_LEAD_MS * CAT_OMEGA;   // 指针从 0 起步，目标放在前方一圈
    markedRef.current = false;
    setPhaseSafe('run');
  }, [run, setPhaseSafe, clearFlash, clearEnded]);

  const endGame = useCallback(() => {
    if (phaseRef.current === 'idle') return;
    run.interrupt();
    logsRef.current = [];
    setLogs([]);
    setLastError(null);
    setTrialNo(0);
    recordedRef.current = true;
    clearFlash();
    setPhaseSafe('idle');
    flashEnded();
  }, [run, setPhaseSafe, clearFlash, flashEnded]);

  const isRunning = phase === 'run' || phase === 'gap';
  useEscapeToEnd(endGame, isRunning);
  useAbortOnTabChange(endGame, isRunning);

  useEffect(() => {
    if (!isRunning) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== 'Space' || e.repeat) return;
      e.preventDefault();
      press({ timeStamp: e.timeStamp });
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isRunning, press]);

  const stats = useMemo(() => {
    if (logs.length === 0) return null;
    const errs = logs.map(l => l.error);
    const abs = errs.map(Math.abs);
    const mean = errs.reduce((a, b) => a + b, 0) / errs.length;
    const meanAbs = abs.reduce((a, b) => a + b, 0) / abs.length;
    return {
      n: logs.length,
      hits: logs.filter(l => l.hit).length,
      meanAbs: Math.round(meanAbs),
      bias: Math.round(mean),
      best: Math.round(Math.min(...abs)),
      worst: Math.round(Math.max(...abs)),
    };
  }, [logs]);

  useEffect(() => {
    if (phase !== 'done' || recordedRef.current || !stats) return;
    recordedRef.current = true;
    submitBest(stats.meanAbs);
    recordGameResult('cat', '节奏打击', (Date.now() - startedAtRef.current) / 1000, {
      trials: stats.n, withinWindow: stats.hits,
      meanAbsErrorMs: stats.meanAbs,
      signedBiasMs: stats.bias,
      accuracy: Math.round((stats.hits / stats.n) * 1000) / 1000,
    });
  }, [phase, stats, submitBest]);

  return (
    <div style={{ textAlign: 'center' }}>
      <Text type="secondary" style={{ display: 'block', marginBottom: 10, fontSize: 13, lineHeight: 1.7 }}>
        指针匀速转圈，扫过<strong style={{ color: '#52c41a' }}>绿色扇区</strong>时按空格（或点画面）。
        目标不是越快越好，而是<strong>刚刚好</strong>。
      </Text>

      <div style={{ display: 'flex', justifyContent: 'center', gap: 30, marginBottom: 10 }}>
        <div>
          <div style={{ fontSize: 12, color: '#999' }}>回合</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: '#722ed1' }}>
            {phase === 'idle' ? 0 : Math.min(trialNo, CAT_TRIALS)} / {CAT_TRIALS}
          </div>
        </div>
        {lastError !== null && (
          <div>
            <div style={{ fontSize: 12, color: '#999' }}>上一击</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: Math.abs(lastError) <= CAT_WINDOW_MS ? '#52c41a' : '#fa8c16' }}>
              {lastError > 0 ? '+' : ''}{lastError} ms
            </div>
          </div>
        )}
        {best !== null && (
          <div>
            <div style={{ fontSize: 12, color: '#999' }}>个人最佳</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: '#faad14' }}>±{best} ms</div>
          </div>
        )}
      </div>

      <canvas
        ref={canvasRef}
        className="cat-canvas"
        width={CAT_SIZE * (isExpanded ? 2 : 1)}
        height={CAT_SIZE * (isExpanded ? 2 : 1)}
        onPointerDown={e => { e.preventDefault(); press(e); }}
        style={{
          width: '100%', maxWidth: isExpanded ? 420 : CAT_SIZE, height: 'auto', display: 'block',
          margin: '0 auto', borderRadius: 16, border: '2px solid #f0f0f0',
          background: '#fbfaff', touchAction: 'none',
          cursor: phase === 'run' ? 'pointer' : 'default',
        }}
      />

      <div style={{ height: 24, marginTop: 6 }}>
        {feedback && <Text style={{ color: feedback.tone, fontWeight: 600 }}>{feedback.text}</Text>}
      </div>

      {phase === 'done' && stats && (
        <ReactionReport
          title="本局成绩"
          tone="#52c41a"
          rows={[
            { label: '平均误差', value: `±${stats.meanAbs} ms` },
            { label: '最好一击', value: `±${stats.best} ms` },
            { label: '时机窗内', value: `${stats.hits} / ${stats.n}` },
            { label: '倾向', value: stats.bias > 12 ? '偏慢' : stats.bias < -12 ? '偏快' : '很准' },
          ]}
          note={
            `平均有符号误差 ${stats.bias > 0 ? '+' : ''}${stats.bias} ms —— `
            + (stats.bias > 12 ? '负数才是提前，你是习惯性慢半拍，可以试着更早一点出手。'
              : stats.bias < -12 ? '你是习惯性抢在前面，可以试着再等一点点。'
                : '你的提前和滞后基本互相抵消了，说明不是固定偏向，而是每一击的稳定度问题。')
            + `误差最小的一击是 ±${stats.best} ms，最大的是 ±${stats.worst} ms。`
            + `判定窗是 ±${CAT_WINDOW_MS} ms，指针每 ${CAT_MS_PER_TURN / 1000} 秒转一圈。`
          }
        />
      )}

      <GameActions onEnd={endGame} endDisabled={phase === 'idle'} ended={ended}>
        <Button type="primary" onClick={start} style={{ borderRadius: 20 }}>
          {phase === 'idle' ? '开始' : '重新开始'}
        </Button>
      </GameActions>
    </div>
  );
}

// ============================================================
// 13. 序列密码（视空间工作记忆）
// ============================================================
const CHIMP_GRID = 5;
const CHIMP_START = 4;
const CHIMP_MAX = 9;

type ChimpPhase = 'idle' | 'playing' | 'reveal' | 'done';

/** 从 25 个格子里随机挑 n 个放数字 1..n */
function seedChimp(n: number): number[] {
  const cells = Array.from({ length: CHIMP_GRID * CHIMP_GRID }, (_, i) => i);
  for (let i = cells.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [cells[i], cells[j]] = [cells[j], cells[i]];
  }
  const picked = cells.slice(0, n);
  const map: number[] = new Array(CHIMP_GRID * CHIMP_GRID).fill(0);
  picked.forEach((cell, i) => { map[cell] = i + 1; });
  return map;
}

/**
 * 序列密码
 * 沿用黑猩猩测试的做法：数字摆好、按 1 开始点，**点掉第一个之后所有数字立刻隐藏**，
 * 剩下全靠记住位置。这一点是关键 —— 否则就退化成"照着数字点"，不测记忆。
 * 每过一关加一位，同一难度连续两次失败就结束。
 */
function SequenceMemory() {
  const run = useInterruptible();
  const { feedback, flash, clear: clearFlash } = useFlash();

  const [phase, setPhase] = useState<ChimpPhase>('idle');
  const phaseRef = useRef<ChimpPhase>('idle');
  const [level, setLevel] = useState(CHIMP_START);
  const [map, setMap] = useState<number[]>([]);
  const [next, setNext] = useState(1);
  const [hidden, setHidden] = useState(false);
  const [maxPassed, setMaxPassed] = useState(0);
  const [fails, setFails] = useState(0);
  const [attempts, setAttempts] = useState(0);
  const [wrongClicks, setWrongClicks] = useState(0);
  const [ended, flashEnded, clearEnded] = useEndedFlag();
  const isExpanded = useIsExpanded();

  const mapRef = useRef<number[]>([]);
  const nextRef = useRef(1);
  const levelRef = useRef(CHIMP_START);
  const failsRef = useRef(0);
  const maxPassedRef = useRef(0);
  const attemptsRef = useRef(0);
  const wrongRef = useRef(0);
  const recordedRef = useRef(false);
  const startedAtRef = useRef(0);
  const { best, submit: submitBest } = usePersonalBest('chimp', true);

  const setPhaseSafe = useCallback((p: ChimpPhase) => { phaseRef.current = p; setPhase(p); }, []);

  const setup = useCallback((n: number) => {
    mapRef.current = seedChimp(n);
    nextRef.current = 1;
    levelRef.current = n;
    attemptsRef.current += 1;
    setMap(mapRef.current);
    setNext(1);
    setLevel(n);
    setHidden(false);
    setAttempts(attemptsRef.current);
    setPhaseSafe('playing');
  }, [setPhaseSafe]);

  const start = useCallback(() => {
    failsRef.current = 0;
    maxPassedRef.current = 0;
    attemptsRef.current = 0;
    wrongRef.current = 0;
    recordedRef.current = false;
    startedAtRef.current = Date.now();
    setFails(0);
    setMaxPassed(0);
    setWrongClicks(0);
    clearFlash();
    clearEnded();
    run.begin();
    setup(CHIMP_START);
  }, [run, setup, clearFlash, clearEnded]);

  const endGame = useCallback(() => {
    if (phaseRef.current === 'idle') return;
    run.interrupt();
    mapRef.current = [];
    nextRef.current = 1;
    levelRef.current = CHIMP_START;
    failsRef.current = 0;
    maxPassedRef.current = 0;
    attemptsRef.current = 0;
    wrongRef.current = 0;
    recordedRef.current = true;
    setMap([]);
    setNext(1);
    setLevel(CHIMP_START);
    setHidden(false);
    setMaxPassed(0);
    setFails(0);
    setAttempts(0);
    setWrongClicks(0);
    clearFlash();
    setPhaseSafe('idle');
    flashEnded();
  }, [run, clearFlash, setPhaseSafe, flashEnded]);

  const clickCell = useCallback((cell: number) => {
    if (phaseRef.current !== 'playing') return;
    const value = mapRef.current[cell];
    if (value !== nextRef.current) {
      // 点错：亮出答案，然后重试本关或结束
      wrongRef.current += 1;
      setWrongClicks(wrongRef.current);
      failsRef.current += 1;
      setFails(failsRef.current);
      flash(`该点 ${nextRef.current}`, '#fa541c');
      setHidden(false);
      setPhaseSafe('reveal');
      const token = run.begin();
      run.trackTimer(setTimeout(() => {
        if (!run.isActive(token)) return;
        if (failsRef.current >= 2) { setPhaseSafe('done'); return; }
        setup(levelRef.current);
      }, 1400));
      return;
    }

    // 第一次点对之后就把数字全部藏起来
    if (nextRef.current === 1 && !hidden) setHidden(true);

    if (nextRef.current >= levelRef.current) {
      maxPassedRef.current = Math.max(maxPassedRef.current, levelRef.current);
      setMaxPassed(maxPassedRef.current);
      // 已经打满最高难度就收尾。少了这一步，min(9+1, 9) 会一直等于 9，
      // 游戏会卡在"过关 -> 又发同样的第 9 关"里永远出不来。
      if (levelRef.current >= CHIMP_MAX) { setPhaseSafe('done'); return; }
      setPhaseSafe('reveal');
      const token = run.begin();
      run.trackTimer(setTimeout(() => {
        if (!run.isActive(token)) return;
        failsRef.current = 0;      // 过关就把同难度的失败计数清零
        setFails(0);
        setup(levelRef.current + 1);
      }, 900));
      return;
    }
    nextRef.current += 1;
    setNext(nextRef.current);
  }, [run, setup, setPhaseSafe, flash, hidden]);

  const isRunning = phase === 'playing' || phase === 'reveal';
  useEscapeToEnd(endGame, isRunning);
  useAbortOnTabChange(endGame, isRunning);

  useEffect(() => {
    if (phase !== 'done' || recordedRef.current) return;
    recordedRef.current = true;
    submitBest(maxPassedRef.current);
    recordGameResult('chimp', '序列密码', (Date.now() - startedAtRef.current) / 1000, {
      maxSpan: maxPassedRef.current,
      attempts: attemptsRef.current,
      wrongClicks: wrongRef.current,
    });
  }, [phase, submitBest]);

  const cellSize = isExpanded ? 72 : 52;
  const showAnswer = phase === 'reveal';
  const rt = phase === 'done' ? maxPassed : null;

  return (
    <div style={{ textAlign: 'center' }}>
      <Text type="secondary" style={{ display: 'block', marginBottom: 10, fontSize: 13, lineHeight: 1.7 }}>
        按 1、2、3… 的顺序点。注意：<strong>点掉第一个之后数字就会全部消失</strong>，后面靠记位置。
        过一关加一位，同一难度连续错两次就结束。
      </Text>

      <div style={{ display: 'flex', justifyContent: 'center', gap: 30, marginBottom: 10 }}>
        <div>
          <div style={{ fontSize: 12, color: '#999' }}>本关位数</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: '#722ed1' }}>{phase === 'idle' ? 0 : level}</div>
        </div>
        <div>
          <div style={{ fontSize: 12, color: '#999' }}>已通关</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: '#52c41a' }}>{maxPassed} 位</div>
        </div>
        <div>
          <div style={{ fontSize: 12, color: '#999' }}>本关失败</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: fails > 0 ? '#fa541c' : '#bbb' }}>{fails} / 2</div>
        </div>
        {best !== null && (
          <div>
            <div style={{ fontSize: 12, color: '#999' }}>个人最佳</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: '#faad14' }}>{best} 位</div>
          </div>
        )}
      </div>

      <div style={{
        display: 'grid', gridTemplateColumns: `repeat(${CHIMP_GRID}, ${cellSize}px)`,
        gap: 8, justifyContent: 'center', margin: '0 auto 8px',
      }}>
        {Array.from({ length: CHIMP_GRID * CHIMP_GRID }, (_, i) => {
          const v = map[i] ?? 0;
          const done = v > 0 && v < next;
          // 只有数字还看得见的时候才提示"下一个点哪"。
          // 数字一隐藏还给黄框，等于直接把答案标出来，记忆任务就白做了。
          const isNext = v === next && phase === 'playing' && !hidden;
          const showNum = v > 0 && !hidden;
          return (
            <div
              key={i}
              onClick={() => clickCell(i)}
              style={{
                width: cellSize, height: cellSize, borderRadius: 12,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: isExpanded ? 24 : 19, fontWeight: 700,
                background: !v ? '#fafafa' : done ? '#d9f7be' : '#fff',
                border: !v ? '1px solid #f5f5f5'
                  : isNext ? '2px solid #faad14'
                    : showAnswer ? '2px solid #ffccc7' : '1px solid #e8e8e8',
                color: '#5a4a6a',
                cursor: v && phase === 'playing' ? 'pointer' : 'default',
                userSelect: 'none', transition: 'all 0.15s',
              }}
            >
              {v === 0 ? '' : showNum ? v : ''}
            </div>
          );
        })}
      </div>

      <div style={{ height: 24 }}>
        {feedback && <Text style={{ color: feedback.tone, fontWeight: 600 }}>{feedback.text}</Text>}
        {!feedback && phase === 'playing' && hidden && (
          <Text type="secondary" style={{ fontSize: 12 }}>数字已隐藏，凭记忆继续</Text>
        )}
      </div>

      {rt !== null && (
        <ReactionReport
          title="本局成绩"
          rows={[
            { label: '最长序列', value: `${maxPassed} 位` },
            { label: '尝试次数', value: `${attempts} 次` },
            { label: '点错次数', value: `${wrongClicks} 次` },
            { label: '个人最佳', value: best !== null ? `${best} 位` : '—' },
          ]}
          note={
            `最长一次记住了 ${maxPassed} 位。这个任务测的是视空间工作记忆的容量 —— `
            + '也就是"同时能盯住几件事"。它和舒尔特方格不同：舒尔特考的是找得快不快，这里考的是记不记得住。'
            + '数字一隐藏就点错很多，通常是没等看清就先动手了；可以先在心里默数一遍顺序再点。'
          }
        />
      )}

      <GameActions onEnd={endGame} endDisabled={phase === 'idle'} ended={ended}>
        <Button type="primary" onClick={start} style={{ borderRadius: 20 }}>
          {phase === 'idle' ? '开始' : '重新开始'}
        </Button>
      </GameActions>
    </div>
  );
}

// ============================================================
// 14. 静默计时（时间知觉）
// ============================================================
const TIMING_TARGET = 10;   // 秒
const TIMING_ROUNDS = 3;

/**
 * 静默计时
 * 不看表，凭感觉数到 10 秒就按停。测的是"时间产生"（time production），
 * 是认知心理学里时间知觉维度的经典做法。
 * 全程不显示任何计时 —— 一给数字，任务就变成看表了。
 */
function SilentTiming() {
  const run = useInterruptible();
  const [phase, setPhase] = useState<'idle' | 'running' | 'between' | 'done'>('idle');
  const phaseRef = useRef<'idle' | 'running' | 'between' | 'done'>('idle');
  const [round, setRound] = useState(0);
  const [results, setResults] = useState<number[]>([]);
  const [ended, flashEnded, clearEnded] = useEndedFlag();
  const isExpanded = useIsExpanded();

  const startAtRef = useRef(0);
  const resultsRef = useRef<number[]>([]);
  const recordedRef = useRef(false);
  const startedAtRef = useRef(0);
  const { best, submit: submitBest } = usePersonalBest('timing', false); // 平均绝对偏差越小越好

  const setPhaseSafe = useCallback((p: 'idle' | 'running' | 'between' | 'done') => {
    phaseRef.current = p;
    setPhase(p);
  }, []);

  const beginRound = useCallback((n: number) => {
    startAtRef.current = performance.now();
    setRound(n);
    setPhaseSafe('running');
  }, [setPhaseSafe]);

  const start = useCallback(() => {
    resultsRef.current = [];
    setResults([]);
    recordedRef.current = false;
    startedAtRef.current = Date.now();
    clearEnded();
    run.begin();
    beginRound(1);
  }, [run, beginRound, clearEnded]);

  const stopRound = useCallback(() => {
    if (phaseRef.current !== 'running') return;
    const estimate = (performance.now() - startAtRef.current) / 1000;
    resultsRef.current = [...resultsRef.current, Math.round(estimate * 100) / 100];
    setResults(resultsRef.current);
    const token = run.begin();
    if (resultsRef.current.length >= TIMING_ROUNDS) { setPhaseSafe('done'); return; }
    setPhaseSafe('between');
    run.trackTimer(setTimeout(() => {
      if (!run.isActive(token)) return;
      beginRound(resultsRef.current.length + 1);
    }, 1200));
  }, [run, beginRound, setPhaseSafe]);

  const endGame = useCallback(() => {
    if (phaseRef.current === 'idle') return;
    run.interrupt();
    resultsRef.current = [];
    setResults([]);
    setRound(0);
    recordedRef.current = true;
    setPhaseSafe('idle');
    flashEnded();
  }, [run, setPhaseSafe, flashEnded]);

  const isRunning = phase === 'running' || phase === 'between';
  useEscapeToEnd(endGame, isRunning);
  useAbortOnTabChange(endGame, isRunning);

  const stats = useMemo(() => {
    if (results.length === 0) return null;
    const errs = results.map(v => v - TIMING_TARGET);
    const abs = errs.map(Math.abs);
    const mean = results.reduce((a, b) => a + b, 0) / results.length;
    const meanErr = errs.reduce((a, b) => a + b, 0) / errs.length;
    const meanAbs = abs.reduce((a, b) => a + b, 0) / abs.length;
    return {
      n: results.length,
      mean: Math.round(mean * 100) / 100,
      bias: Math.round(meanErr * 100) / 100,
      meanAbs: Math.round(meanAbs * 100) / 100,
      spread: Math.round((Math.max(...results) - Math.min(...results)) * 100) / 100,
    };
  }, [results]);

  useEffect(() => {
    if (phase !== 'done' || recordedRef.current || !stats) return;
    recordedRef.current = true;
    submitBest(stats.meanAbs);
    recordGameResult('timing', '静默计时', stats.n * TIMING_TARGET, {
      targetSec: TIMING_TARGET, rounds: stats.n,
      estimates: resultsRef.current,
      meanSec: stats.mean, signedBiasSec: stats.bias,
      meanAbsErrorSec: stats.meanAbs, spreadSec: stats.spread,
    });
  }, [phase, stats, submitBest]);

  const pulse = isExpanded ? 220 : 160;

  return (
    <div style={{ textAlign: 'center' }}>
      <Text type="secondary" style={{ display: 'block', marginBottom: 10, fontSize: 13, lineHeight: 1.7 }}>
        凭感觉数到 <strong>{TIMING_TARGET} 秒</strong>就按停，共 {TIMING_ROUNDS} 轮。
        全程不会显示任何计时 —— 一给数字，这题就变成看表了。
      </Text>

      <div style={{ display: 'flex', justifyContent: 'center', gap: 30, marginBottom: 10 }}>
        <div>
          <div style={{ fontSize: 12, color: '#999' }}>第几轮</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: '#722ed1' }}>
            {phase === 'idle' ? 0 : Math.min(round, TIMING_ROUNDS)} / {TIMING_ROUNDS}
          </div>
        </div>
        {best !== null && (
          <div>
            <div style={{ fontSize: 12, color: '#999' }}>个人最准</div>
            <div style={{ fontSize: 18, fontWeight: 700, color: '#52c41a' }}>偏差 {best} s</div>
          </div>
        )}
      </div>

      <div
        className="timing-stage"
        onPointerDown={e => { e.preventDefault(); if (phase === 'running') stopRound(); }}
        style={{
          width: '100%', maxWidth: isExpanded ? 560 : 420, height: isExpanded ? 360 : 260,
          margin: '0 auto', borderRadius: 18, background: '#f7f5ff',
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          gap: 18, userSelect: 'none', touchAction: 'none',
          cursor: phase === 'running' ? 'pointer' : 'default',
        }}
      >
        {phase === 'running' ? (
          <>
            <div style={{
              width: pulse, height: pulse, borderRadius: '50%',
              background: 'radial-gradient(circle at 36% 30%, #e6dbff, #b39ddb)',
              boxShadow: '0 10px 30px rgba(114,46,209,0.22)',
              animation: 'float 4s ease-in-out infinite',
            }} />
            <Text style={{ color: '#8c8c8c', fontSize: 14 }}>觉得到了 {TIMING_TARGET} 秒，就点一下</Text>
          </>
        ) : (
          <Text style={{ color: '#8c8c8c', fontWeight: 600 }}>
            {phase === 'between' ? `第 ${round} 轮完成，准备下一轮…`
              : phase === 'done' ? '三轮都完成了' : '点下面的「开始」'}
          </Text>
        )}
      </div>

      {phase === 'done' && stats && (
        <div style={{ marginTop: 12 }}>
          <ReactionReport
            title="本局成绩"
            tone="#722ed1"
            rows={[
              { label: '平均估计', value: `${stats.mean} s` },
              { label: '平均偏差', value: `±${stats.meanAbs} s` },
              { label: '偏向', value: stats.bias > 0.3 ? '偏高' : stats.bias < -0.3 ? '偏低' : '很准' },
              { label: '三轮极差', value: `${stats.spread} s` },
            ]}
            note={
              `目标 ${TIMING_TARGET} 秒，你三轮平均数是 ${stats.mean} 秒`
              + `（${stats.bias > 0 ? '偏长' : '偏短'} ${Math.abs(stats.bias)} 秒）。`
              + '偏差方向非常个人化，也会随状态浮动：同一处境下有人越紧张越觉得时间拖得长，也有人一忙起来就觉得时间不够用。'
              + '所以这个数更像是"当天状态的一枚小指纹"，而不是任何判断依据。'
              + `三轮极差 ${stats.spread} 秒反映的是稳定性 —— 三轮都很接近，说明你的内部节拍很稳。`
            }
          />
        </div>
      )}

      <GameActions onEnd={endGame} endDisabled={phase === 'idle'} ended={ended}>
        <Button type="primary" onClick={phase === 'running' ? stopRound : start} style={{ borderRadius: 20 }}>
          {phase === 'running' ? '停' : phase === 'idle' ? '开始' : '重新开始'}
        </Button>
      </GameActions>
    </div>
  );
}

// ============================================================
// 15. 持续注意（SART）
// ============================================================
const SART_TRIALS = 36;
const SART_INTERVAL = 1400;  // 每个试次的总时长
const SART_STIM_MS = 400;    // 数字呈现时长，随后遮罩
const SART_NOGO = 3;         // 出现 3 时不能按

interface SartLog { digit: number; outcome: 'hit' | 'wrong' | 'miss' | 'anticipate'; rt: number | null; }

/**
 * 生成数字序列
 * 除 3 以外的数字都要按，所以 3 就是"不能按"的信号。
 * 先填满所有非 3 的数字，再往空位里插 3 —— 这样序列里天然不会有多余的 3，
 * 也不用事后清理。3 之间不相邻，第一个试次也不会是 3：
 * 先建立"看到就按"的习惯，抑制才有意义。
 */
function buildSartPlan(total: number, nogoCount: number): number[] {
  const digits: number[] = Array.from({ length: total }, () => {
    let v = 1 + Math.floor(Math.random() * 8);   // 1..8
    return v >= SART_NOGO ? v + 1 : v;           // 跳过 3，得到 1、2、4..9
  });
  const slots = Array.from({ length: Math.max(0, total - 1) }, (_, i) => i + 1);
  for (let i = slots.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [slots[i], slots[j]] = [slots[j], slots[i]];
  }
  const used = new Set<number>();
  for (const s of slots) {
    if (used.size >= nogoCount) break;
    if (used.has(s - 1) || used.has(s + 1)) continue;
    used.add(s);
  }
  used.forEach(s => { digits[s] = SART_NOGO; });
  return digits;
}

/**
 * 持续注意（SART）
 * 看到任何数字都要按，唯独 3 不能按。任务本身极其单调，考验的是能不能一直守住规则。
 * 关键指标不是平均反应时，而是「误报」和反应时的变异 ——
 * 注意一旦开始溜号，反应时就会忽快忽慢，同时手会不受控地按到 3 上。
 */
function SustainedAttention() {
  const clock = useReactionClock();
  const run = useInterruptible();
  const { feedback, flash, clear: clearFlash } = useFlash();

  const [phase, setPhase] = useState<'idle' | 'wait' | 'show' | 'done'>('idle');
  const phaseRef = useRef<'idle' | 'wait' | 'show' | 'done'>('idle');
  const [digit, setDigit] = useState<number | null>(null);
  const [masked, setMasked] = useState(false);
  const [logs, setLogs] = useState<SartLog[]>([]);
  const [cursor, setCursor] = useState(0);
  const [ended, flashEnded, clearEnded] = useEndedFlag();
  const isExpanded = useIsExpanded();

  const planRef = useRef<number[]>([]);
  const cursorRef = useRef(0);
  const logsRef = useRef<SartLog[]>([]);
  const recordedRef = useRef(false);
  const startedAtRef = useRef(0);
  const { best, submit: submitBest } = usePersonalBest('sart', false); // 误报次数越少越好

  const setPhaseSafe = useCallback((p: 'idle' | 'wait' | 'show' | 'done') => {
    phaseRef.current = p;
    setPhase(p);
  }, []);

  const record = useCallback((outcome: SartLog['outcome'], rt: number | null) => {
    const d = planRef.current[cursorRef.current];
    if (d === undefined) return true;
    logsRef.current = [...logsRef.current, { digit: d, outcome, rt }];
    setLogs(logsRef.current);
    cursorRef.current += 1;
    setCursor(cursorRef.current);
    return cursorRef.current >= planRef.current.length;
  }, []);

  const advanceRef = useRef<() => void>(() => {});
  const advance = useCallback(() => {
    const token = run.begin();
    clock.clear();
    const d = planRef.current[cursorRef.current];
    if (d === undefined) { setDigit(null); setPhaseSafe('done'); return; }
    setDigit(d);
    setMasked(false);
    setPhaseSafe('show');
    clock.armOnNextFrame();
    // 呈现 400ms 后遮罩，剩下的时间留作反应窗
    run.trackTimer(setTimeout(() => {
      if (!run.isActive(token)) return;
      setMasked(true);
    }, SART_STIM_MS));
    run.trackTimer(setTimeout(() => {
      if (!run.isActive(token)) return;
      if (d === SART_NOGO) {
        record('hit', null);   // 忍住没按 = 正确抑制
      } else {
        record('miss', null);  // 该按没按 = 漏报
      }
      advanceRef.current();
    }, SART_INTERVAL));
  }, [run, clock, setPhaseSafe, record]);

  useEffect(() => { advanceRef.current = advance; }, [advance]);

  const press = useCallback((e?: { timeStamp?: number }) => {
    const p = phaseRef.current;
    if (p === 'idle' || p === 'done') return;
    const d = planRef.current[cursorRef.current];
    if (d === undefined) return;
    const ms = clock.elapsed(e);
    if (p === 'show' && ms !== null && ms < RT_MIN_VALID) {
      record('anticipate', null);
      advanceRef.current();
      return;
    }
    if (d === SART_NOGO) {
      flash('这是 3，不该按', '#fa541c');
      record('wrong', null);
    } else {
      record('hit', ms);
    }
    advanceRef.current();
  }, [flash, record, clock]);

  const start = useCallback(() => {
    planRef.current = buildSartPlan(SART_TRIALS, Math.round(SART_TRIALS / 6));
    cursorRef.current = 0;
    logsRef.current = [];
    setLogs([]);
    setCursor(0);
    recordedRef.current = false;
    startedAtRef.current = Date.now();
    clearFlash();
    clearEnded();
    advanceRef.current();
  }, [clearFlash, clearEnded]);

  const endGame = useCallback(() => {
    if (phaseRef.current === 'idle') return;
    run.interrupt();
    clock.clear();
    planRef.current = [];
    cursorRef.current = 0;
    logsRef.current = [];
    setLogs([]);
    setCursor(0);
    setDigit(null);
    setMasked(false);
    recordedRef.current = true;
    clearFlash();
    setPhaseSafe('idle');
    flashEnded();
  }, [run, clock, clearFlash, setPhaseSafe, flashEnded]);

  const isRunning = phase === 'show';
  useEscapeToEnd(endGame, isRunning);
  useAbortOnTabChange(endGame, isRunning);

  useEffect(() => {
    if (!isRunning) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== 'Space' || e.repeat) return;
      e.preventDefault();
      press({ timeStamp: e.timeStamp });
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isRunning, press]);

  const stats = useMemo(() => {
    const hits = logs.filter(l => l.outcome === 'hit' && l.digit !== SART_NOGO);
    const rt = summarizeRt(hits.map(l => l.rt as number));
    const anticipate = logs.filter(l => l.outcome === 'anticipate').length;
    const valid = logs.length - anticipate;
    const half = Math.floor(logs.length / 2);
    const firstHalf = logs.slice(0, half).filter(l => l.outcome === 'wrong').length;
    const secondHalf = logs.slice(half).filter(l => l.outcome === 'wrong').length;
    return {
      done: logs.length,
      valid,
      hit: hits.length,
      wrong: logs.filter(l => l.outcome === 'wrong').length,
      miss: logs.filter(l => l.outcome === 'miss').length,
      anticipate,
      accuracy: valid > 0 ? logs.filter(l => l.outcome === 'hit').length / valid : 0,
      rt,
      cv: rt && rt.mean > 0 ? Math.round((rt.sd / rt.mean) * 1000) / 10 : null,
      firstHalf, secondHalf,
    };
  }, [logs]);

  useEffect(() => {
    if (phase !== 'done' || recordedRef.current) return;
    recordedRef.current = true;
    const s = stats;
    submitBest(s.wrong);
    recordGameResult('sart', '持续注意', (Date.now() - startedAtRef.current) / 1000, {
      trials: s.done, valid: s.valid,
      hit: s.hit, commissionErrors: s.wrong, omissionErrors: s.miss, anticipate: s.anticipate,
      accuracy: Math.round(s.accuracy * 1000) / 1000,
      meanRt: s.rt ? s.rt.mean : null, sdRt: s.rt ? s.rt.sd : null, rtCvPercent: s.cv,
      commissionFirstHalf: s.firstHalf, commissionSecondHalf: s.secondHalf,
    });
  }, [phase, stats, submitBest]);

  const size = isExpanded ? 620 : 420;
  const height = isExpanded ? 320 : 220;
  const rt = phase === 'done' ? stats.rt : null;

  return (
    <div style={{ textAlign: 'center' }}>
      <Text type="secondary" style={{ display: 'block', marginBottom: 10, fontSize: 13, lineHeight: 1.7 }}>
        看到数字就按（空格或点画面），<strong>唯独出现 3 的时候别按</strong>。
        一共 {SART_TRIALS} 个，约 {Math.round((SART_TRIALS * SART_INTERVAL) / 1000)} 秒，中途不能走神。
      </Text>

      <div style={{ display: 'flex', justifyContent: 'center', gap: 30, marginBottom: 10 }}>
        <div>
          <div style={{ fontSize: 12, color: '#999' }}>进度</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: '#722ed1' }}>
            {phase === 'idle' ? 0 : Math.min(cursor + (phase === 'done' ? 0 : 1), SART_TRIALS)} / {SART_TRIALS}
          </div>
        </div>
        <div>
          <div style={{ fontSize: 12, color: '#999' }}>误按 3</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: stats.wrong > 0 ? '#fa541c' : '#52c41a' }}>{stats.wrong}</div>
        </div>
        {rt && <div>
          <div style={{ fontSize: 12, color: '#999' }}>平均反应</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: '#1890ff' }}>{rt.mean} ms</div>
        </div>}
      </div>

      <div
        className="sart-stage"
        onPointerDown={e => { e.preventDefault(); press(e); }}
        style={{
          width: '100%', maxWidth: size, height, margin: '0 auto', borderRadius: 18,
          background: '#33334d', display: 'flex', alignItems: 'center', justifyContent: 'center',
          userSelect: 'none', touchAction: 'none', cursor: isRunning ? 'pointer' : 'default',
        }}
      >
        {phase === 'idle' && (
          <Text style={{ color: 'rgba(255,255,255,0.7)', fontWeight: 600 }}>点下面的「开始」</Text>
        )}
        {phase === 'show' && !masked && digit !== null && (
          <div style={{ fontSize: isExpanded ? 140 : 96, fontWeight: 700, color: '#fff', lineHeight: 1 }}>
            {digit}
          </div>
        )}
      </div>

      <div style={{ height: 24, marginTop: 6 }}>
        {feedback && <Text style={{ color: feedback.tone, fontWeight: 600 }}>{feedback.text}</Text>}
      </div>

      {rt && (
        <ReactionReport
          title="本局成绩"
          rows={[
            { label: '误按 3', value: `${stats.wrong} 次` },
            { label: '漏按', value: `${stats.miss} 次` },
            { label: '正确率', value: `${Math.round(stats.accuracy * 100)}%` },
            { label: '平均反应', value: `${rt.mean} ms` },
            { label: '反应波动', value: stats.cv !== null ? `${stats.cv}%` : '—' },
            { label: '抢跑', value: `${stats.anticipate} 次` },
          ]}
          note={
            `误按 3 共 ${stats.wrong} 次、漏按 ${stats.miss} 次。`
            + (stats.cv !== null
              ? `反应波动（标准差÷平均值）是 ${stats.cv}%，这个数比平均反应时更能说明问题：`
                + '注意力一旦开始溜号，反应时就会忽快忽慢，平均值却可能看起来很正常。'
              : '')
            + `前半段误按 ${stats.firstHalf} 次、后半段 ${stats.secondHalf} 次`
            + (stats.secondHalf > stats.firstHalf
              ? '—— 越到后面越守不住，这是很典型的现象，说明单调任务对注意力的消耗是真实存在的。'
              : '—— 后半段没有变差，说明你在这段时间里守得比较稳。')
            + '这类任务非常反直觉：规则简单到不需要思考，但正因为不需要思考，注意力才有地方溜走。'
          }
        />
      )}

      <GameActions onEnd={endGame} endDisabled={phase === 'idle'} ended={ended}>
        <Button type="primary" onClick={start} style={{ borderRadius: 20 }}>
          {phase === 'idle' ? '开始' : '重新开始'}
        </Button>
      </GameActions>
    </div>
  );
}

// ============================================================
// 主页面
// ============================================================
const gameCategories = [
  {
    key: 'focus',
    label: (
      <span><BulbOutlined style={{ marginRight: 6, color: '#722ed1' }} />注意力 & 专注力</span>
    ),
    children: (
      <Row gutter={[24, 24]} align="stretch">
        <Col xs={24} lg={12}>
          <GameCard title="🔢 舒尔特方格">
            <SchulteGrid />
          </GameCard>
        </Col>
        <Col xs={24} lg={12}>
          <GameCard title="🎨 色词冲突">
            <StroopTest />
          </GameCard>
        </Col>
        <Col xs={24} lg={12}>
          <GameCard title="🔑 序列密码">
            <SequenceMemory />
          </GameCard>
        </Col>
        <Col xs={24} lg={12}>
          <GameCard title="🧭 持续注意">
            <SustainedAttention />
          </GameCard>
        </Col>
      </Row>
    ),
  },
  {
    key: 'relax',
    label: (
      <span><CloudOutlined style={{ marginRight: 6, color: '#52c41a' }} />缓解压力</span>
    ),
    children: (
      <Row gutter={[24, 24]} align="stretch">
        <Col xs={24} lg={12}>
          <GameCard title=" 气球呼吸训练">
            <BalloonBreathing />
          </GameCard>
        </Col>
        <Col xs={24} lg={12}>
          <GameCard title="⏳ 静默计时">
            <SilentTiming />
          </GameCard>
        </Col>
        {/* 冥想自然音内容最高，单独占满一行：并排会把另一半拉出近 300px 空白 */}
        <Col xs={24}>
          <GameCard title=" 冥想自然音">
            <MeditationAudio />
          </GameCard>
        </Col>
      </Row>
    ),
  },
  {
    key: 'release',
    label: (
      <span>🎨 释放压力</span>
    ),
    children: (
      <Row gutter={[24, 24]} align="stretch">
        <Col xs={24} lg={12}>
          <GameCard title="🖌️ 涂鸦画板">
            <DoodleCanvas />
          </GameCard>
        </Col>
        <Col xs={24} lg={12}>
          <GameCard title="🧸 橡皮人拉扯">
            <RubberPerson />
          </GameCard>
        </Col>
      </Row>
    ),
  },
  {
    key: 'react',
    label: (
      <span><ThunderboltOutlined style={{ marginRight: 6, color: '#faad14' }} />反应 & 协调</span>
    ),
    children: (
      <Row gutter={[24, 24]} align="stretch">
        {/* 同一行两张卡的内容高度尽量接近，等高对齐后卡片里的空白最少 */}
        <Col xs={24} lg={12}>
          <GameCard title="🚦 反应灯">
            <ReactionLight />
          </GameCard>
        </Col>
        <Col xs={24} lg={12}>
          <GameCard title="🖱️ 跟手追踪">
            <HandTracking />
          </GameCard>
        </Col>
        <Col xs={24} lg={12}>
          <GameCard title="🎯 多目标追踪">
            <MultiObjectTracking />
          </GameCard>
        </Col>
        <Col xs={24} lg={12}>
          <GameCard title="🥁 节奏打击">
            <RhythmTiming />
          </GameCard>
        </Col>
        <Col xs={24} lg={12}>
          <GameCard title="⚡ 甩点挑战">
            <FlickingTask />
          </GameCard>
        </Col>
        <Col xs={24} lg={12}>
          <GameCard title="🔀 选择反应">
            <ChoiceReaction />
          </GameCard>
        </Col>
      </Row>
    ),
  },
];

export default function Healing() {
  const [activeTab, setActiveTab] = useState('focus');
  const [abortVersion, setAbortVersion] = useState(0);

  const handleTabChange = useCallback((key: string) => {
    setActiveTab(key);
    // 离开当前分类 = 打断正在运行的游戏，避免隐藏标签页里的计时器 / 白噪音继续跑
    setAbortVersion(v => v + 1);
  }, []);

  return (
    <HealingAbortContext.Provider value={abortVersion}>
      <style>{GAME_LAYOUT_CSS}</style>
      <div>
        <div style={{ marginBottom: 24, textAlign: 'center' }}>
          <Title level={3} style={{ marginBottom: 4 }}>
            <ExperimentOutlined style={{ color: '#ff8fab', marginRight: 8 }} />
            心理小游戏
          </Title>
          <Text type="secondary">选一个喜欢的游戏，在玩耍中照顾自己的心情 🌸</Text>
        </div>

        <Tabs items={gameCategories} activeKey={activeTab} onChange={handleTabChange}
          tabBarStyle={{ marginBottom: 20 }}
          size="large"
        />
      </div>
    </HealingAbortContext.Provider>
  );
}
