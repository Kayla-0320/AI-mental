import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { learningApi } from '../services';
import type { LearningContent } from '../types/garden';
import ContentPreviewModal, { CATEGORY_LABEL } from './home/ContentPreviewModal';

/**
 * 顶栏搜索。
 *
 * 为什么不是"假搜索框"：
 *   搜索结果分两类，都能真的点进去 ——
 *     1. 平台内的功能页（静态清单，与左侧导航同一份口径）；
 *     2. 心理知识内容，来自真实的 GET /api/learning/contents。
 *   项目里**没有内容详情页路由**（`/learning` 被重定向到 `/healing`），
 *   所以内容点击后在弹窗里直接展示接口返回的标题 / 摘要 / 正文，
 *   而不是把用户踢到一个打不开的地址。
 */

interface NavTarget {
  key: string;
  icon: string;
  title: string;
  desc: string;
  path: string;
  keywords: string;
}

const NAV_TARGETS: NavTarget[] = [
  { key: 'home', icon: '🏠', title: '首页', desc: '今日心情、成长与推荐', path: '/', keywords: '首页 主页 心情 打卡' },
  { key: 'chat', icon: '💬', title: 'AI 倾诉', desc: '和 AI 心理伙伴说说话', path: '/chat', keywords: '倾诉 聊天 ai 对话 情绪' },
  { key: 'assessment', icon: '📋', title: '心理画像', desc: '心理画像、量表测评与结果记录', path: '/profile', keywords: '画像 测评 量表 测试 心理画像 自查' },
  { key: 'experts', icon: '☕', title: '咨询服务', desc: '预约公益咨询师', path: '/experts', keywords: '咨询 咨询师 预约 专家 我的咨询' },
  { key: 'healing', icon: '🌿', title: '疗愈空间', desc: '冥想、呼吸、音乐放松', path: '/healing', keywords: '疗愈 冥想 呼吸 放松 音乐 助眠' },
  { key: 'companions', icon: '🫂', title: '同伴社区', desc: '和有相似经历的人交流', path: '/companions', keywords: '社区 同伴 树洞 分享' },
  { key: 'growth', icon: '🌱', title: '我的成长', desc: '等级、指标与徽章', path: '/growth', keywords: '成长 等级 徽章 打卡 指标' },
  { key: 'twin', icon: '🧠', title: '数字孪生', desc: 'AI 陪伴学习与成长', path: '/twin', keywords: '数字 孪生 twin 数字孪生 学习' },
  { key: 'call', icon: '📞', title: '心理电话', desc: '不想打字时，和 AI 通个电话', path: '/call', keywords: '电话 通话 语音 电话倾诉' },
  { key: 'settings', icon: '⚙️', title: '设置', desc: '账户与偏好设置', path: '/settings', keywords: '设置 账户 偏好 隐私' },
];

const CATEGORY_FALLBACK = '心理知识';

function SearchIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" style={{ flex: '0 0 auto', color: '#7d8ca3' }}>
      <circle cx="11" cy="11" r="6.4" fill="none" stroke="currentColor" strokeWidth="2" />
      <path d="M15.8 15.8 20 20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

export default function NavSearch() {
  const navigate = useNavigate();
  const [query, setQuery] = useState('');
  const [focused, setFocused] = useState(false);
  const [contents, setContents] = useState<LearningContent[]>([]);
  const [contentsLoaded, setContentsLoaded] = useState(false);
  const [preview, setPreview] = useState<LearningContent | null>(null);
  const boxRef = useRef<HTMLDivElement>(null);

  // 内容清单只在第一次聚焦时拉一次：不进首页就永远不会请求
  const loadContents = async () => {
    if (contentsLoaded) return;
    setContentsLoaded(true);
    try {
      const res: any = await learningApi.getContents({ page: 1, pageSize: 30 });
      const list: LearningContent[] = res?.data?.contents ?? [];
      setContents(list);
    } catch {
      // 静默：搜索是一个附加入口，接口失败不应该弹错误打断用户
      setContents([]);
    }
  };

  // 点击外部收起
  useEffect(() => {
    if (!focused) return;
    const onDown = (e: MouseEvent) => {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) setFocused(false);
    };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [focused]);

  const q = query.trim().toLowerCase();

  const pageHits = useMemo(() => {
    if (!q) return NAV_TARGETS.slice(0, 5);
    return NAV_TARGETS.filter(
      (t) => t.title.toLowerCase().includes(q) || t.desc.toLowerCase().includes(q) || t.keywords.includes(q),
    ).slice(0, 5);
  }, [q]);

  const contentHits = useMemo(() => {
    if (!q) return contents.slice(0, 3);
    return contents
      .filter((c) => {
        const cat = CATEGORY_LABEL[c.category] ?? CATEGORY_FALLBACK;
        return (
          c.title.toLowerCase().includes(q) ||
          (c.summary ?? '').toLowerCase().includes(q) ||
          cat.includes(q)
        );
      })
      .slice(0, 5);
  }, [q, contents]);

  const open = focused && (pageHits.length > 0 || contentHits.length > 0 || q.length > 0);

  const go = (path: string) => {
    setFocused(false);
    setQuery('');
    navigate(path);
  };

  return (
    <div className="gh-nav__search" ref={boxRef}>
      <SearchIcon />
      <input
        type="search"
        value={query}
        aria-label="搜索想了解的内容"
        placeholder="搜索你想了解的内容"
        onFocus={() => { setFocused(true); loadContents(); }}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={(e) => {
          if (e.key !== 'Enter') return;
          // 回车直接跳到最匹配的一项，键盘用户不必再用鼠标点结果
          if (pageHits[0]) go(pageHits[0].path);
          else if (contentHits[0]) { setPreview(contentHits[0]); setFocused(false); }
        }}
      />

      {open && (
        <div className="gh-nav__results" role="listbox" aria-label="搜索结果">
          {pageHits.length === 0 && contentHits.length === 0 && (
            <div className="gh-nav__results-empty">
              没有找到「{query}」相关的内容
              <br />
              试试换个说法，或直接去 AI 倾诉聊聊
            </div>
          )}

          {pageHits.map((t) => (
            <button key={t.key} type="button" className="gh-nav__result" role="option" aria-selected={false} onClick={() => go(t.path)}>
              <span className="gh-nav__result-icon" aria-hidden="true">{t.icon}</span>
              <span style={{ minWidth: 0 }}>
                <span className="gh-nav__result-title">{t.title}</span>
                <span className="gh-nav__result-desc">{t.desc}</span>
              </span>
            </button>
          ))}

          {contentHits.length > 0 && (
            <>
              <div className="gh-nav__results-group">心理知识</div>
              {contentHits.map((c) => (
                <button
                  key={c.id}
                  type="button"
                  className="gh-nav__result"
                  role="option"
                  aria-selected={false}
                  onClick={() => { setPreview(c); setFocused(false); }}
                >
                  <span className="gh-nav__result-icon" aria-hidden="true">📖</span>
                  <span style={{ minWidth: 0 }}>
                    <span className="gh-nav__result-title">{c.title}</span>
                    <span className="gh-nav__result-desc">
                      {CATEGORY_LABEL[c.category] ?? CATEGORY_FALLBACK}
                      {c.duration ? ` · ${c.duration} 分钟` : ''}
                    </span>
                  </span>
                </button>
              ))}
            </>
          )}
        </div>
      )}

      <ContentPreviewModal item={preview} onClose={() => setPreview(null)} />
    </div>
  );
}
