import { useState } from 'react';
import { Popover, message } from 'antd';
import { useAnxiety } from '../context/AnxietyContext';

/**
 * 顶栏「多模态感知」开关。
 *
 * 历史：这三个功能（心理电话 / 多模态开关 / 反馈评价）原先挤在首页的「其他功能」卡片区，
 * 而那一整块已按产品要求并入顶栏。多模态是一个**跨页面的全局状态**，放在顶栏比放在
 * 首页更合理：在任何页面都能看到它现在是开是关。
 *
 * 为什么用 Popover 而不是裸开关：
 *   它要申请摄像头与麦克风权限，裸按钮容易被误触。Popover 里同时说明"数据仅本机分析"
 *   与当前实时状态，用户点之前就知道会发生什么。
 *
 * ⚠️ 依赖 `AnxietyProvider`，必须渲染在 PatientLayout 的 Provider 之内。
 */
function SenseIcon({ on }: { on: boolean }) {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
      <circle cx="12" cy="12" r="3.1" fill="none" stroke="currentColor" strokeWidth="1.9" />
      <path
        d="M6.4 17.6a8 8 0 0 1 0-11.2M17.6 6.4a8 8 0 0 1 0 11.2M3.6 20.4a11.7 11.7 0 0 1 0-16.8M20.4 3.6a11.7 11.7 0 0 1 0 16.8"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
        opacity={on ? 1 : 0.32}
      />
      {!on && <path d="M4 20 20 4" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" />}
    </svg>
  );
}

export default function MultimodalToggle() {
  const { cameraEnabled, toggleCamera, mediaNotice } = useAnxiety();
  const [toggling, setToggling] = useState(false);

  /**
   * `toggleCamera()` 返回"是否真的打开了"：麦克风与摄像头**都**起不来才是 false；
   * 只起来一路（常见于只授了麦克风权限）仍算成功，此时 `mediaNotice` 里会有
   * "已仅启用麦克风"之类的说明 —— 由 Popover 里的 ⚠️ 呈现，而不是假装一切正常。
   */
  const handleToggle = async (next: boolean) => {
    if (toggling) return;
    setToggling(true);
    try {
      const ok = await toggleCamera();
      if (!ok) {
        message.warning('麦克风和摄像头都无法启动，请检查浏览器权限设置');
        return;
      }
      message.success(next ? '多模态感知已开启（数据仅在本机分析）' : '多模态感知已关闭');
    } finally {
      setToggling(false);
    }
  };

  const content = (
    <div className="gh-mm">
      <div className="gh-mm__head">
        <span className={`gh-mm__state${cameraEnabled ? ' is-on' : ''}`}>
          <span className="gh-mm__dot" aria-hidden="true" />
          {cameraEnabled ? '感知中' : '已关闭'}
        </span>
        <button
          type="button"
          role="switch"
          aria-checked={cameraEnabled}
          aria-label="多模态感知开关"
          data-quick="multimodal"
          disabled={toggling}
          className={`gh-switch${cameraEnabled ? ' is-on' : ''}`}
          onClick={() => handleToggle(!cameraEnabled)}
        >
          <span className="gh-switch__knob" />
        </button>
      </div>

      <p className="gh-mm__desc">
        摄像头看表情、麦克风听语调。原始音视频<strong>不出本机</strong>，只上传分析后的少量数值特征。
      </p>

      {cameraEnabled && mediaNotice ? (
        <p className="gh-mm__notice">⚠️ {mediaNotice}</p>
      ) : null}
      {!cameraEnabled ? (
        <p className="gh-mm__notice gh-mm__notice--muted">打开后，首页的「情绪识别」会开始工作。</p>
      ) : null}
    </div>
  );

  return (
    <Popover content={content} trigger="click" placement="bottomRight" arrow={false} overlayClassName="gh-mm-pop">
      <button
        type="button"
        className={`gh-nav__icon-btn${cameraEnabled ? ' is-active' : ''}`}
        aria-label={`多模态感知：${cameraEnabled ? '已开启' : '已关闭'}`}
        data-quick="multimodal-open"
      >
        <SenseIcon on={cameraEnabled} />
      </button>
    </Popover>
  );
}
