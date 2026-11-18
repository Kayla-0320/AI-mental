/**
 * 左下角「情绪识别」悬浮工具栏
 *
 * 需求：把**已经识别出来的情绪结果**直接显示在页面上；位置固定在左下角，
 * 做成工具栏的感觉，随视图（路由）切换始终可见。
 *
 * ## 2026-10-10 重做：让三条概率"看得见地在动"
 *
 * 用户反馈两点：
 *   1. 「一次识别间隔太长，很长时间都在等结果」→ 需要三种情绪的百分比**持续实时变化**；
 *   2. 「开心几乎识别不出来」→ 需要能看出开心到底有没有涨。
 *
 * 对应的界面改动：
 *   - **收起态也常显三条百分比**（开心/中性/悲伤 + 一条三色占比条），
 *     不必展开就能看到数字在动；
 *   - 百分比用 `smoothedProbabilities`（时间常数公开的 EMA，见 `faceEmotionPacing.ts`）
 *     上屏，配合 0.35s 的宽度过渡，视觉上是连续变化而不是每几秒跳一次；
 *     卡片里**同时**列出模型未平滑的原始判定，避免把平滑值误当成模型输出；
 *   - **近 10 秒峰值**：短暂的一闪而过的微笑（可能只占一两拍）不会因为
 *     "当前判定是中性"就完全看不见；
 *   - 如实展示**实测刷新率 / 执行后端 / 单次推理耗时** —— 慢后端下刷新就是慢，
 *     界面要说出来，而不是让用户对着不动的数字猜"是不是坏了"。
 *
 * ## 2026-11-18：读数口径搬进 `faceEmotionView.tsx`
 *
 * 倾诉工作区（`/chat`）改成「聊天 + 右栏」以后，这条悬浮条会压住输入区，已从该页撤掉，
 * 同三个数改在右栏顶部常显。取数与上屏口径（平滑 / 有效期 / 三类之外的余量 / 峰值 /
 * 非诊断措辞）统一收在 `faceEmotionView.tsx`，**本文件只负责浮层的位置与展开收起**，
 * 不要再在这里另写一套读数逻辑，否则两处会对同一份数据给出不同说法。
 */
import { useEffect, useState } from 'react';
import {
  useFaceEmotionSnapshot,
  ProbabilityRow,
  StackedBar,
  FaceEmotionDetail,
  BlinkDetail,
  FusionSummaryRows,
  HonestyNote,
} from './faceEmotionView';

export default function EmotionStatusDock() {
  const snap = useFaceEmotionSnapshot();
  const [expanded, setExpanded] = useState(false);
  // 移动端底部有 56px 的 Tab 栏，悬浮件要抬起来，避免被压住
  const [isMobile, setIsMobile] = useState(
    typeof window !== 'undefined' && window.innerWidth < 768,
  );

  useEffect(() => {
    const onResize = () => setIsMobile(window.innerWidth < 768);
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  const { face, faceStatus, cameraEnabled, showFaceReading, probs, dominantStyle, faceSummary, textEmotion } = snap;
  const { blink } = snap;

  return (
    <div
      data-dock="emotion-status"
      style={{
        position: 'fixed',
        left: 16,
        bottom: isMobile ? 68 : 16,
        zIndex: 900,
        fontFamily: 'inherit',
      }}
    >
      {!expanded ? (
        // ── 收起态：一枚小工具栏，三条概率**常显**（不必展开就能看到数字在动）──
        <div
          role="button"
          tabIndex={0}
          onClick={() => setExpanded(true)}
          onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setExpanded(true); }}
          title="点击查看情绪识别详情"
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: 4,
            padding: '6px 10px',
            borderRadius: 10,
            border: '1px solid rgba(0,0,0,0.06)',
            background: 'rgba(255,255,255,0.92)',
            backdropFilter: 'blur(8px)',
            boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
            cursor: 'pointer',
            color: '#333',
            minWidth: 232,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 13 }}>{dominantStyle?.icon ?? '🧠'}</span>
            <span style={{ color: dominantStyle?.color ?? '#8c8c8c', fontWeight: 600, fontSize: 12 }}>
              {faceSummary}
            </span>
            {showFaceReading && face && (
              <span style={{ color: '#bbb', fontSize: 10 }}>{Math.round(face.confidence * 100)}%</span>
            )}
            <span style={{ flex: 1 }} />
            {/* 眨眼频率：`isMeasuring` 为 false 时整段不显示，避免把"没测"显示成 0 */}
            {blink.measuring && (
              <span style={{ color: '#666', fontSize: 11, whiteSpace: 'nowrap' }}>
                👁 {blink.rate}/分
              </span>
            )}
            {textEmotion && (
              <span style={{ color: '#8c8c8c', fontSize: 11, whiteSpace: 'nowrap' }}>
                {textEmotion}
              </span>
            )}
          </div>

          {/* 三条百分比：**常显**，这就是"实时变动的三个数字" */}
          <ProbabilityRow probs={probs} compact />

          {/* 三色占比条：随概率连续伸缩（0.35s 线性过渡） */}
          {probs && <StackedBar probs={probs} height={4} />}

          {!probs && (
            <span style={{ color: '#bbb', fontSize: 10 }}>
              {faceStatus === 'unavailable'
                ? '模型不可用'
                : cameraEnabled
                  ? '暂无新读数'
                  : '打开摄像头后可识别'}
            </span>
          )}
        </div>
      ) : (
        // ── 展开态：一张小卡片，给出完整读数与"为什么没读数" ─────────────
        <div
          style={{
            width: 250,
            padding: '10px 12px',
            borderRadius: 12,
            border: '1px solid rgba(0,0,0,0.06)',
            background: 'rgba(255,255,255,0.96)',
            backdropFilter: 'blur(10px)',
            boxShadow: '0 4px 16px rgba(0,0,0,0.10)',
            fontSize: 12,
            color: '#333',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: 8 }}>
            <span style={{ fontSize: 13, fontWeight: 600, flex: 1 }}>情绪识别</span>
            <button
              type="button"
              onClick={() => setExpanded(false)}
              title="收起"
              style={{
                border: 'none', background: 'transparent', cursor: 'pointer',
                color: '#999', fontSize: 14, padding: 0, lineHeight: 1,
              }}
            >
              ×
            </button>
          </div>

          <FaceEmotionDetail snap={snap} />
          <BlinkDetail snap={snap} />
          <FusionSummaryRows snap={snap} />
          <HonestyNote snap={snap} />
        </div>
      )}
    </div>
  );
}
