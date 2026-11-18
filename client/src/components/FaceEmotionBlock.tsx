/**
 * 右栏（陪伴与感知）顶部的「情绪识别」块 —— 三种面部表情读数的常显位置。
 *
 * 背景：这三个数原本只在左下角悬浮工具条（`EmotionStatusDock`）里。倾诉工作区改成
 * 「聊天 + 右栏」以后，悬浮条会压住输入框，已从 `/chat` 撤掉 —— 用户在最需要它的
 * 页面反而看不到了。所以把它**常显在右栏顶部**（不塞进某个 Tab 里：切到「持续陪伴」
 * 时也该看得见），不再依赖悬浮层。
 *
 * 口径全部来自 `faceEmotionView.tsx`，本文件只管摆放与展开收起：
 *   · 三条概率用平滑值上屏，和模型原始判定并列（展开后可见）；
 *   · 读数有有效期，过期就说明"没有新读数"，不把几秒前的值当当前表情；
 *   · 没有读数时百分比显示 `—`，不是 `0%`；
 *   · 措辞非诊断（AGENTS.md 安全红线），只描述"识别到什么"。
 */
import { useState } from 'react';
import {
  useFaceEmotionSnapshot,
  ProbabilityRow,
  StackedBar,
  FaceEmotionDetail,
  BlinkDetail,
  FusionSummaryRows,
  HonestyNote,
} from './faceEmotionView';
import type { FaceEmotionSnapshot } from './faceEmotionView';

/** 一句话说明"现在为什么有/没有数字" —— 不给结论、不给建议 */
function readingHint(snap: FaceEmotionSnapshot): string {
  if (snap.showFaceReading) return '本机识别，只做展示，不参与风险判断';
  if (snap.faceStatus === 'unavailable') return snap.faceReason || '表情模型不可用';
  if (snap.faceStatus === 'loading') return '模型加载中…';
  // 只指路，不在这里再开一个摄像头开关：开关仍只有聊天框左下角那一个入口。
  if (!snap.cameraEnabled) return '打开摄像头后这里会出现读数（视频不出设备）；开关在聊天框左下角的摄像头按钮';
  if (!snap.facePresent) return '画面里暂时没有看到人脸';
  return '暂无新读数，等下一拍';
}

/** 右栏里常显的三种情绪标签 */
export default function FaceEmotionBlock() {
  return <FaceEmotionBlockView snap={useFaceEmotionSnapshot()} />;
}

/**
 * 纯展示版（不含取数）：把 `FaceEmotionBlock` 拆成"取数"与"摆放"两层，
 * 渲染核验时可以喂一份构造好的快照，不必真的开摄像头。
 */
export function FaceEmotionBlockView({ snap }: { snap: FaceEmotionSnapshot }) {
  const [expanded, setExpanded] = useState(false);
  const { probs, dominantStyle, faceSummary, offMap, offMapDominant, showFaceReading } = snap;

  return (
    <section className="chat-emotion" data-testid="emotion-tri" aria-label="此刻的面部表情识别读数">
      <div className="chat-emotion-head">
        {/* 有主导情绪时用对应的表情图标；没有读数时不摆一个"看起来像有结论"的图标 */}
        {dominantStyle && (
          <span className="chat-emotion-icon" aria-hidden="true">{dominantStyle.icon}</span>
        )}
        <span className="chat-emotion-title">情绪识别</span>
        <span
          className="chat-emotion-state"
          style={{ color: dominantStyle?.color ?? '#9a8fae' }}
        >
          {faceSummary}
        </span>
        <span style={{ flex: 1 }} />
        <button
          type="button"
          className="chat-emotion-toggle"
          aria-expanded={expanded}
          onClick={() => setExpanded((v) => !v)}
        >
          {expanded ? '收起' : '详情'}
        </button>
      </div>

      {/* 三条百分比常显；没有读数时是 `—`，不是 `0%` */}
      <ProbabilityRow probs={probs} compact={false} />
      {probs && (
        <div style={{ margin: '6px 0 5px' }}>
          <StackedBar probs={probs} height={6} />
        </div>
      )}

      {/* 三类之外的余量：多数判定落在三类之外时，别把第一名的百分比当结论 */}
      {showFaceReading && offMap > 0.02 && (
        <div className="chat-emotion-extra">
          其他表情 {Math.round(offMap * 100)}%
          {offMapDominant ? '（多数判定落在三类之外，上面三条仅供参考）' : ''}
        </div>
      )}

      <div className="chat-emotion-hint">{readingHint(snap)}</div>

      {expanded && (
        <div className="chat-emotion-detail">
          <FaceEmotionDetail snap={snap} />
          <BlinkDetail snap={snap} />
          <FusionSummaryRows snap={snap} />
          <HonestyNote snap={snap} />
        </div>
      )}
    </section>
  );
}

/**
 * 一行紧凑版三标签（右栏收起为抽屉时，摆在聊天卡片工具栏里，
 * 否则窄屏上这三个数会彻底看不到）。
 */
export function FaceEmotionChips() {
  const snap = useFaceEmotionSnapshot();
  return (
    <span
      className="chat-emotion-chips"
      data-testid="emotion-tri-chips"
      title={readingHint(snap)}
    >
      <ProbabilityRow probs={snap.probs} compact />
      {snap.probs && (
        <span className="chat-emotion-chips-bar">
          <StackedBar probs={snap.probs} height={4} />
        </span>
      )}
    </span>
  );
}
