import { useEffect, useRef, useState } from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { Avatar, Dropdown, Badge, Popover, List, Button, Empty, Drawer, Space, Typography } from 'antd';
import {
  BellOutlined, LogoutOutlined, SettingOutlined, UserOutlined,
  MenuOutlined, CloseOutlined, DownOutlined,
} from '@ant-design/icons';
import { useAuthStore } from '../../store/authStore';
import api from '../../services/api';
import { AnxietyProvider } from '../../context/AnxietyContext';
import RealityTaskModal from '../../components/RealityTaskModal';
import EmotionStatusDock from '../../components/EmotionStatusDock';
import NavSearch from '../../components/NavSearch';
import PageTopImage, { hasPageTopImage } from '../../components/home/PageTopImage';
import MultimodalToggle from '../../components/MultimodalToggle';
import PhoneHelpModal from '../../components/PhoneHelpModal';
import FeedbackModal from '../../components/FeedbackModal';
import { useRouteTransition } from '../../components/PageTransition';

const { Text } = Typography;

/**
 * 患者端外壳：**顶部导航 + 沉浸式内容区**。
 *
 * 内容区两种形态：
 *   首页与疗愈空间有整幅出血的 Hero（`isFlush`）—— 内容区不加内边距，
 *   顶栏在页顶走透明压图态；其余页面用 `.gh-app__page` 统一留白。
 *   新增带 Hero 的页面时，把路径加进 `isFlush` 即可，别再各写一套。
 *
 * 导航信息架构（本次做过减法，不要再往首页堆入口）：
 *   首页此前同时存在「侧边栏 + 顶栏 + 快捷入口卡片 + 常用功能卡片」四套入口，
 *   同一个页面（如 /profile、/experts）能有三四个按钮指向它。现在收敛为**一处**：
 *
 *     顶栏主项   首页 · AI 倾诉 · 心理画像 · 咨询服务 · 疗愈空间
 *     顶栏「更多」 我的成长 · 同伴社区 · 数字孪生 · 心理电话 · 反馈评价 · 设置
 *     顶栏右侧    搜索 · 多模态感知开关 · 通知 · 头像
 *
 *   划分是数据驱动的：把条目在 `NAV_ITEMS` / `MORE_ITEMS` 之间挪一行即可换位置，
 *   不需要改布局代码。
 *
 * ⚠️ 仍必须保留的东西（删了会掉功能）：
 *   · `AnxietyProvider` —— EmotionStatusDock / MultimodalToggle 依赖它，且
 *     `useVoiceCall` 注释里明确说明通话页依赖这个 context 只在 PatientLayout 挂载；
 *   · `RealityTaskModal` —— 认知行为练习的现实任务弹窗；
 *   · `EmotionStatusDock` —— 左下角情绪识别工具条；
 *   · 移动端底部 Tab —— 手机端的主要导航方式。
 *
 * ⚠️ 危机资源分布（不要以为藏在弹窗里就等于没有）：
 *   · 顶栏「更多 → 心理电话」弹窗内含 AI 通话与全部 24 小时热线；
 *   · 首页 Hero 的「不想打字？试试心理电话」同一弹窗；
 *   · 页脚常驻热线；AI 倾诉 / 陪伴通话页检测到危机时仍强制弹出 `CrisisResourceCard`。
 */

interface NavEntry {
  key: string;
  label: string;
  desc: string;
  icon: string;
  /** 路由型 */
  path?: string;
  /** 动作型：打开全局弹窗，不跳路由 */
  action?: 'phone' | 'feedback';
}

/** 顶栏主项：最常用、最需要在任何页面一步到达 */
const NAV_ITEMS: NavEntry[] = [
  { key: 'home', label: '首页', desc: '今日心情、成长与推荐', icon: '🏠', path: '/' },
  { key: 'chat', label: 'AI 倾诉', desc: '和 AI 心理伙伴说说话', icon: '💬', path: '/chat' },
  { key: 'assessment', label: '心理画像', desc: '多模态心理画像与量表测评', icon: '📋', path: '/profile' },
  { key: 'consult', label: '咨询服务', desc: '预约公益咨询师', icon: '☕', path: '/experts' },
  { key: 'healing', label: '疗愈空间', desc: '冥想、呼吸、音乐放松', icon: '🌿', path: '/healing' },
];

/** 「更多」下拉：都会用到，但不必占用顶栏的横向空间 */
const MORE_ITEMS: NavEntry[] = [
  { key: 'growth', label: '我的成长', desc: '等级、指标与徽章', icon: '🌱', path: '/growth' },
  { key: 'community', label: '同伴社区', desc: '和同龄人分享交流', icon: '🫂', path: '/companions' },
  { key: 'twin', label: '数字孪生', desc: 'AI 陪伴学习与成长', icon: '🧠', path: '/twin' },
  { key: 'phone', label: '心理电话', desc: '不想打字时打个电话', icon: '📞', action: 'phone' },
  { key: 'feedback', label: '反馈评价', desc: '告诉我们哪里不好用', icon: '💌', action: 'feedback' },
  { key: 'settings', label: '设置', desc: '账户与偏好设置', icon: '⚙️', path: '/settings' },
];

function LogoMark({ size = 36 }: { size?: number }) {
  return (
    <svg viewBox="0 0 48 48" width={size} height={size} aria-hidden="true">
      <defs>
        <linearGradient id="ghNavPetal" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#ffb3cd" />
          <stop offset="100%" stopColor="#ef6f9f" />
        </linearGradient>
      </defs>
      {[0, 72, 144, 216, 288].map((deg) => (
        <ellipse
          key={deg}
          cx="24"
          cy="15"
          rx="7.6"
          ry="11"
          fill="url(#ghNavPetal)"
          opacity="0.94"
          transform={`rotate(${deg} 24 24)`}
        />
      ))}
      <circle cx="24" cy="24" r="5.6" fill="#ffd9a0" />
      <circle cx="24" cy="24" r="2.4" fill="#f6b25c" />
    </svg>
  );
}

export default function PatientLayout() {
  const navigate = useNavigate();
  const location = useLocation();
  const { user, logout } = useAuthStore();
  const [notificationCount, setNotificationCount] = useState(0);
  const [notifications, setNotifications] = useState<any[]>([]);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
  /**
   * 中部导航在 <1080px 会被 CSS 收起（见 garden-shell.css 的 .gh-nav__links），
   * 所以汉堡按钮的显示条件必须用**同一个断点**，而不是 768。
   * 曾经用 isMobile 控制：768~959px 之间链接已经被藏起来、汉堡却也不显示，
   * 结果这一段宽度下**完全没有导航入口**。
   */
  const [navCompact, setNavCompact] = useState(window.innerWidth < 1080);
  const [scrolled, setScrolled] = useState(false);

  // 全局弹窗（心理电话 / 反馈评价）：从顶栏「更多」进入，因此在 Layout 层持有状态
  const [phoneOpen, setPhoneOpen] = useState(false);
  const [feedbackOpen, setFeedbackOpen] = useState(false);

  /**
   * 页面切换过渡：动画挂在**本来就存在的两个容器**上，不新增包裹层 ——
   *   · `mainRef`  —— 内容区，所有页面都在这；
   *   · `stripRef` —— 页顶图条包装层，它是 main 的兄弟（图条要整屏出血，必须放外面），
   *                  所以得单独挂一份，否则切页时"内容淡入、图条硬切"。
   * 两处用同一个 `location.key`，同一帧启动，看起来是一次整体过渡。
   * 顶栏 / 底部 Tab / 抽屉都在 main 之外，**不参与动画** —— 导航不该跟着晃。
   */
  const mainRef = useRef<HTMLElement>(null);
  const stripRef = useRef<HTMLDivElement>(null);
  useRouteTransition(mainRef);
  useRouteTransition(stripRef);

  const isHome = location.pathname === '/';
  /**
   * 顶栏是否走 `--over` 透明压图态：首页与所有图条页的页顶都压着一张插画，
   * 实心玻璃顶栏会把图切断，看起来像"图片被顶栏压掉了"。
   */
  const navOverImage = isHome || hasPageTopImage(location.pathname);

  /**
   * 内容区形态：
   *   · 首页 —— 整幅出血（`.gh-app__main--flush`，Hero 自己铺满）；
   *   · 图条页 —— 取消顶部留白、保留左右（`.gh-app__main--top-strip`），
   *     图条用负外边距出血，页面内容仍有 `.gh-app__page` 的左右留白；
   *   · 其余页面 —— `.gh-app__page` 统一留白。
   * ⚠️ 顶部形态必须与 `<PageTopImage />` 的渲染判据一致（同一个 hasPageTopImage），
   * 否则会出现"图条下多出一大块空白"或"内容贴边"。
   */
  const mainClass = isHome
    ? 'gh-app__main gh-app__main--flush'
    : hasPageTopImage(location.pathname)
      ? 'gh-page-shell gh-app__main gh-app__page gh-app__main--top-strip'
      : 'gh-app__main gh-app__page';

  // 通知轮询（沿用旧实现的 30s 周期，行为不变）
  useEffect(() => {
    const loadNotifications = async () => {
      try {
        const res = await api.get('/notifications') as any;
        if (res.data) {
          setNotificationCount(res.data.unreadCount || 0);
          setNotifications(res.data.notifications || []);
        }
      } catch {}
    };
    loadNotifications();
    const interval = setInterval(loadNotifications, 30000);
    return () => clearInterval(interval);
  }, []);

  // 响应式监听
  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth < 768);
      setNavCompact(window.innerWidth < 1080);
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // 滚动状态：只有它决定顶栏是"透明压图"还是"玻璃吸顶"
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  // 切换页面时回到顶部；顺手关掉移动端抽屉
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'auto' });
    setMobileMenuOpen(false);
  }, [location.pathname]);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const userMenu = {
    items: [
      { key: 'settings', icon: <SettingOutlined />, label: '个人设置', onClick: () => navigate('/settings') },
      { type: 'divider' as const },
      { key: 'logout', icon: <LogoutOutlined />, label: '退出登录', onClick: handleLogout },
    ],
  };

  const isActive = (item: NavEntry) => {
    if (!item.path) return false;
    if (item.path === '/') return location.pathname === '/';
    return location.pathname === item.path || location.pathname.startsWith(`${item.path}/`);
  };

  /** 统一的入口分发：路由型跳转，动作型开弹窗 */
  const openEntry = (item: NavEntry) => {
    setMobileMenuOpen(false);
    if (item.action === 'phone') { setPhoneOpen(true); return; }
    if (item.action === 'feedback') { setFeedbackOpen(true); return; }
    if (item.path) navigate(item.path);
  };

  const moreMenu = {
    items: MORE_ITEMS.map((item) => ({
      key: item.key,
      icon: <span style={{ fontSize: 16, lineHeight: 1 }}>{item.icon}</span>,
      label: (
        <span className="gh-more-menu__label">
          <span className="gh-more-menu__title">{item.label}</span>
          <span className="gh-more-menu__desc">{item.desc}</span>
        </span>
      ),
      onClick: () => openEntry(item),
    })),
  };

  const bottomTabs = [
    { key: '/', label: '首页', icon: '🏠' },
    { key: '/chat', label: 'AI倾诉', icon: '💬' },
    { key: '/companions', label: '社区', icon: '🫂' },
    { key: '/healing', label: '疗愈', icon: '🌿' },
    { key: '/settings', label: '我的', icon: '⚙️' },
  ];

  const navClass = `gh-nav ${navOverImage && !scrolled ? 'gh-nav--over' : 'gh-nav--solid'}`;

  return (
    <AnxietyProvider>
      <div className="gh-app">
        <header className={navClass}>
          <div className="gh-nav__inner">
            {/* 品牌 */}
            <button type="button" className="gh-nav__brand" onClick={() => navigate('/')} aria-label="回到心灵花园首页">
              <span className="gh-nav__logo"><LogoMark size={38} /></span>
              <span className="gh-nav__brand-text">
                <span className="gh-nav__brand-name">心灵花园</span>
                <span className="gh-nav__brand-tag">陪你，遇见更好的自己</span>
              </span>
            </button>

            {/* 中部导航（<1080px 收起为抽屉） */}
            <nav className="gh-nav__links" aria-label="主导航">
              {NAV_ITEMS.map((item) => (
                <button
                  key={item.key}
                  type="button"
                  className={`gh-nav__link${isActive(item) ? ' is-active' : ''}`}
                  aria-current={isActive(item) ? 'page' : undefined}
                  onClick={() => openEntry(item)}
                >
                  {item.label}
                </button>
              ))}

              <Dropdown menu={moreMenu} placement="bottom" trigger={['hover', 'click']} overlayClassName="gh-more-menu">
                <button
                  type="button"
                  className={`gh-nav__link${MORE_ITEMS.some(isActive) ? ' is-active' : ''}`}
                  aria-haspopup="menu"
                >
                  更多 <DownOutlined style={{ fontSize: 10 }} />
                </button>
              </Dropdown>
            </nav>

            {/* 右侧动作区 */}
            <div className="gh-nav__actions">
              <NavSearch />
              <MultimodalToggle />

              <Popover
                placement="bottomRight"
                trigger="click"
                arrow={false}
                title={
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Text strong>通知</Text>
                    {notifications.length > 0 && (
                      <Button
                        type="link"
                        size="small"
                        onClick={async () => {
                          try { await api.put('/notifications/read-all'); } catch {}
                          setNotificationCount(0);
                        }}
                      >
                        全部已读
                      </Button>
                    )}
                  </div>
                }
                content={
                  notifications.length === 0 ? (
                    <div style={{ padding: '20px 0', textAlign: 'center', width: 260 }}>
                      <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂无通知" />
                    </div>
                  ) : (
                    <List
                      size="small"
                      style={{ maxHeight: 320, overflow: 'auto', width: 300 }}
                      dataSource={notifications}
                      renderItem={(item: any) => (
                        <List.Item
                          style={{
                            padding: '10px 8px',
                            background: item.isRead ? 'transparent' : '#fffbe6',
                            borderRadius: 8,
                            marginBottom: 4,
                            cursor: 'pointer',
                          }}
                          onClick={async () => {
                            try { await api.put(`/notifications/${item.id}/read`); } catch {}
                            setNotificationCount((prev) => Math.max(0, prev - 1));
                            setNotifications((prev) => prev.map((n) => (n.id === item.id ? { ...n, isRead: true } : n)));
                          }}
                        >
                          <div style={{ width: '100%' }}>
                            <div style={{ fontSize: 13, fontWeight: item.isRead ? 400 : 600 }}>{item.title}</div>
                            <div style={{ fontSize: 12, color: '#8a97a8', marginTop: 2 }}>{item.content}</div>
                            <div style={{ fontSize: 11, color: '#aab4c2', marginTop: 4 }}>
                              {new Date(item.createdAt).toLocaleString('zh-CN')}
                            </div>
                          </div>
                        </List.Item>
                      )}
                    />
                  )
                }
                onOpenChange={(open) => { if (open) setNotificationCount(0); }}
              >
                <button type="button" className="gh-nav__icon-btn" aria-label={`通知，${notificationCount} 条未读`}>
                  <Badge count={notificationCount} size="small" offset={[2, -2]}>
                    <BellOutlined />
                  </Badge>
                </button>
              </Popover>

              <Dropdown menu={userMenu} placement="bottomRight">
                <button type="button" className="gh-nav__user" aria-label="账户菜单">
                  <Avatar size={30} style={{ backgroundColor: '#ffb6c1' }} icon={<UserOutlined />} />
                  <span className="gh-nav__user-name">{user?.nickname || '我'}</span>
                </button>
              </Dropdown>

              {navCompact && (
                <button
                  type="button"
                  className="gh-nav__icon-btn gh-nav__burger"
                  aria-label="打开菜单"
                  onClick={() => setMobileMenuOpen(true)}
                >
                  <MenuOutlined />
                </button>
              )}
            </div>
          </div>
        </header>

        {/*
          页顶图条（路由 → 图片见 components/home/PageTopImage.tsx）。
          刻意放在 `<main>` **外面**：main 是 `max-width: 1240px; margin: 0 auto` 的定宽盒子，
          图条放里面最多只能和内容同宽（两侧留白），做成"首页那样的大图"必须让它整屏铺满。
          它不参与文档流（absolute），高度由包装层变量喂给 main 占位。

          ⚠️ 包装层必须跟着「这一页有没有图条」一起判断：它是固定 500px 高的块，
          无条件渲染会在没有图条的首页顶出一整屏空白（实测踩过）。
        */}
        {hasPageTopImage(location.pathname) && (
          <div className="gh-app__topstrip" ref={stripRef}>
            <PageTopImage />
          </div>
        )}

        {/*
          内容区。形态由 mainClass 决定，见上面的注释；图条页用 `--top-strip`
          （padding-top 由包装层的高度变量算出，内容正好落在图下面）。
          `ref` 只用于路由切换过渡（原地重放动画，不重挂子树，页面状态全保留）。
        */}
        <main className={mainClass} ref={mainRef}>
          <Outlet />
        </main>

        {/*
          右下角悬浮图标列已下线（心理电话 / 多模态开关 / 反馈评价）——
          这三个功能现在都在顶栏：心理电话与反馈评价在「更多」里，多模态开关是右侧的图标按钮。
        */}
        <RealityTaskModal />
        <EmotionStatusDock />

        {/* 移动端底部 Tab */}
        {isMobile && (
          <div
            style={{
              position: 'fixed', bottom: 0, left: 0, right: 0, height: 56,
              background: 'rgba(255,255,255,0.95)', backdropFilter: 'blur(10px)',
              borderTop: '1px solid rgba(124,152,184,0.16)',
              display: 'flex', alignItems: 'center', justifyContent: 'space-around',
              zIndex: 1000, paddingBottom: 'env(safe-area-inset-bottom)',
            }}
          >
            {bottomTabs.map((tab) => {
              const active = location.pathname === tab.key;
              return (
                <button
                  key={tab.key}
                  type="button"
                  onClick={() => navigate(tab.key)}
                  style={{
                    display: 'flex', flexDirection: 'column', alignItems: 'center',
                    border: 'none', background: 'none', cursor: 'pointer',
                    padding: '4px 0', minWidth: 48, font: 'inherit',
                    color: active ? '#e2578a' : '#8a97a8',
                    transition: 'color 0.2s',
                  }}
                >
                  <span style={{ fontSize: 18, lineHeight: 1.2 }}>{tab.icon}</span>
                  <span style={{ fontSize: 10 }}>{tab.label}</span>
                </button>
              );
            })}
          </div>
        )}

        {/* 移动端导航抽屉：列出全部入口（含「更多」里的那 6 项） */}
        <Drawer
          placement="left"
          width={280}
          open={mobileMenuOpen}
          onClose={() => setMobileMenuOpen(false)}
          closeIcon={<CloseOutlined />}
          styles={{ body: { padding: 0 }, header: { display: 'none' } }}
        >
          <div className="gh-nav-drawer-brand">
            <LogoMark size={32} />
            <div>
              <div style={{ fontSize: 16, fontWeight: 800, color: '#26364f', letterSpacing: 1 }}>心灵花园</div>
              <div style={{ fontSize: 11, color: '#7d8ca3' }}>陪你，遇见更好的自己</div>
            </div>
          </div>

          <div className="gh-nav-drawer-list">
            {NAV_ITEMS.map((item) => (
              <button
                key={item.key}
                type="button"
                className={`gh-nav-drawer-item${isActive(item) ? ' is-active' : ''}`}
                onClick={() => openEntry(item)}
              >
                <span aria-hidden="true">{item.icon}</span>
                {item.label}
              </button>
            ))}

            <div className="gh-nav-drawer-group">更多</div>
            {MORE_ITEMS.map((item) => (
              <button
                key={item.key}
                type="button"
                className={`gh-nav-drawer-item${isActive(item) ? ' is-active' : ''}`}
                onClick={() => openEntry(item)}
              >
                <span aria-hidden="true">{item.icon}</span>
                {item.label}
              </button>
            ))}
          </div>

          <div style={{ padding: '4px 14px 24px' }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Button block onClick={() => { setMobileMenuOpen(false); navigate('/chat'); }}>开始 AI 倾诉</Button>
              <Button block onClick={() => { setMobileMenuOpen(false); setPhoneOpen(true); }}>心理电话</Button>
            </Space>
          </div>
        </Drawer>

        {/* 顶栏「更多」里的两个全局弹窗 */}
        <PhoneHelpModal open={phoneOpen} onClose={() => setPhoneOpen(false)} />
        <FeedbackModal open={feedbackOpen} onClose={() => setFeedbackOpen(false)} />
      </div>
    </AnxietyProvider>
  );
}
