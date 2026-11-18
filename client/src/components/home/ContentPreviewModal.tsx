import { Modal } from 'antd';
import { useNavigate } from 'react-router-dom';
import type { LearningContent } from '../../types/garden';

/**
 * 心理知识内容预览弹窗。
 *
 * 项目里**没有内容详情页路由**（`/learning` 被 App.tsx 重定向到 `/healing`），
 * 所以搜索命中、推荐卡点击都走这里：直接展示接口返回的真实字段，
 * 而不是把用户送到一个打不开的地址，也不用假数据撑场面。
 *
 * 复用方：`components/NavSearch.tsx`、`pages/patient/Home.tsx`。
 */

/** 分类中文名：搜索结果的副标题与弹窗里的标签共用同一份映射 */
export const CATEGORY_LABEL: Record<string, string> = {
  emotion: '情绪',
  stress: '减压',
  sleep: '睡眠',
  self_growth: '自我成长',
  relationship: '人际关系',
};

/** 内容形态 → 量词（"5 分钟阅读" / "20 分钟聆听"） */
export const TYPE_NOUN: Record<string, string> = {
  article: '阅读',
  audio: '聆听',
  video: '观看',
  course: '学习',
};

export default function ContentPreviewModal({
  item,
  onClose,
}: {
  item: LearningContent | null;
  onClose: () => void;
}) {
  const navigate = useNavigate();

  return (
    <Modal open={item !== null} onCancel={onClose} footer={null} width={580} title={item?.title ?? ''}>
      {item && (
        <>
          <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
            <span className="gh-chip">{CATEGORY_LABEL[item.category] ?? '心理知识'}</span>
            {item.duration ? (
              <span className="gh-chip">
                {item.duration} 分钟{TYPE_NOUN[item.type] ?? '阅读'}
              </span>
            ) : null}
          </div>

          <p className="gh-modal__text">{item.summary ?? '这篇内容暂时还没有简介。'}</p>

          {item.content ? <div className="gh-nav__preview-body">{item.content}</div> : null}

          <button
            type="button"
            className="gh-btn gh-btn--primary gh-btn--block"
            style={{ marginTop: 16 }}
            onClick={() => { onClose(); navigate('/healing'); }}
          >
            去疗愈空间练习
          </button>
          <p className="gh-modal__foot">内容仅供自我参考，不构成医学诊断。</p>
        </>
      )}
    </Modal>
  );
}
