import { useEffect, useRef, useState } from 'react';

const quotes = [
  '每一天的你，都值得被温柔以待。',
  '你不必完美，也很可爱。',
  '慢慢来，比较快，你已经在路上了。',
  '别急着开花，每棵树都有自己的季节。',
  '乌云遮不住太阳，困难挡不住成长。',
  '休息不是放弃，是给心灵充电。',
  '今天的每一点努力，都是惊喜的伏笔。',
  '每一颗星星，都曾是穿过黑夜的光。',
];

export const SPLASH_EVENT = 'app-splash';

export default function SplashScreen() {
  const [mounted, setMounted] = useState(false);
  const [quoteIn, setQuoteIn] = useState(false);
  const [leaving, setLeaving] = useState(false);
  const [quote, setQuote] = useState('');
  const timersRef = useRef<number[]>([]);

  useEffect(() => {
    const clearTimers = () => {
      timersRef.current.forEach((t) => window.clearTimeout(t));
      timersRef.current = [];
    };

    const play = () => {
      clearTimers();
      setQuote(quotes[Math.floor(Math.random() * quotes.length)]);
      setLeaving(false);
      setQuoteIn(false);
      setMounted(true);
      timersRef.current = [
        window.setTimeout(() => setQuoteIn(true), 700),
        window.setTimeout(() => setLeaving(true), 3800),
        window.setTimeout(() => setMounted(false), 4700),
      ];
    };

    window.addEventListener(SPLASH_EVENT, play);
    return () => {
      window.removeEventListener(SPLASH_EVENT, play);
      clearTimers();
    };
  }, []);

  if (!mounted) return null;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 9999,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '0 24px',
        background: 'var(--bg-gradient, linear-gradient(135deg, #fff0f3 0%, #f0e6ff 30%, #e8f4f8 60%, #fff8f0 100%))',
        opacity: leaving ? 0 : 1,
        transition: 'opacity 0.9s ease',
      }}
    >
      <div
        style={{
          fontSize: 36,
          fontFamily: '"KaiTi", "STKaiti", "楷体", "DFKai-SB", "BiauKai", serif',
          fontWeight: 400,
          letterSpacing: 3,
          lineHeight: 1.8,
          textAlign: 'center',
          color: 'var(--text-primary, #5a4a6a)',
          opacity: quoteIn ? 1 : 0,
          transform: quoteIn ? 'translateY(0)' : 'translateY(14px)',
          transition: 'opacity 1.2s ease, transform 1.2s ease',
        }}
      >
        {quote}
      </div>
    </div>
  );
}
