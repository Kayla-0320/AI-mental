import PageHero from './PageHero';

/**
 * 首页主视觉 —— 结构与取景都收敛在 `PageHero` 里（首页/疗愈空间共用），
 * 这里只负责把首页自己的文案与行为塞进去，避免两页各维护一份 Hero 标记。
 */
export default function HomeHero({
  greeting,
  nickname,
  checkedInToday,
  onStartChat,
}: {
  greeting: string;
  nickname: string;
  checkedInToday: boolean;
  onStartChat: () => void;
}) {
  return (
    <PageHero
      variant="home"
      eyebrow="心灵花园 · 陪你遇见更好的自己"
      title={
        <>
          你并不孤单，
          <br />
          <span className="gh-hero__title-accent">这里有温暖的陪伴</span>
        </>
      }
      subtitle="AI 心理伙伴 · 24 小时倾听 · 陪你走过每一段情绪"
      ctaLabel="开始心灵对话"
      onCta={onStartChat}
      ctaNote="关爱心灵，从现在开始"
      buddy={{
        greeting,
        nickname,
        mood: checkedInToday ? 'happy' : 'idle',
        title: '你好，我是你的 AI 心理伙伴',
        desc: '今天要不要和我聊聊你的心情？我一直在这里。',
        onChat: onStartChat,
      }}
    />
  );
}
