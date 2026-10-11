import { Modal } from 'antd';
import { useNavigate } from 'react-router-dom';
import { HOTLINES } from './home/hotlines';

/**
 * 「心理电话」弹窗。
 *
 * 为什么是弹窗而不是直接跳 `/call`：
 *   这个弹窗同时承担两件事 —— 一是进入 AI 陪伴通话，二是**在同一个位置**给出
 *   24 小时人工热线。有自伤风险的用户点了"心理电话"，第一眼就该看到真人的号码，
 *   而不是先进一个语音页再找热线。
 *
 * ⚠️ 号码统一来自 `components/home/hotlines.ts`，不要在这里另抄一份。
 */
export default function PhoneHelpModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const navigate = useNavigate();

  return (
    <Modal
      open={open}
      onCancel={onClose}
      footer={null}
      width={470}
      title="心理电话"
      styles={{ body: { paddingTop: 4 } }}
    >
      <p className="gh-modal__text">
        AI 陪伴者会像朋友一样接起电话，听你把话说完。不想打字、眼睛累了，
        或者只是想听见一个声音的时候，都可以打。
      </p>

      <button
        type="button"
        className="gh-btn gh-btn--primary gh-btn--block"
        data-quick="start-call"
        onClick={() => { onClose(); navigate('/call'); }}
      >
        开始 AI 陪伴通话
      </button>
      <p className="gh-modal__hint">建议戴上耳机，找一个安静的地方</p>

      <div className="gh-modal__divider"><span>如果此刻特别难受</span></div>
      <p className="gh-modal__hint gh-modal__hint--left">
        也可以直接拨打人工热线，24 小时有人接听：
      </p>

      {HOTLINES.map((h) => (
        <div key={h.phone} className="gh-modal__hotline">
          <span className="gh-modal__hotline-name">
            {h.name}
            <em>{h.note}</em>
          </span>
          <a href={`tel:${h.phone}`} className="gh-modal__hotline-phone">{h.phone}</a>
        </div>
      ))}

      <p className="gh-modal__foot">
        通话与热线均保密。如果情况紧急，请优先拨打上面的电话。
      </p>
    </Modal>
  );
}
