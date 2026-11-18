import { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { message } from 'antd';
import dayjs from 'dayjs';
import { growthApi, moodApi, extraApi } from '../../services';
import type { GrowthOverview } from '../../types/garden';
import { useAuthStore } from '../../store/authStore';

import HomeHero from '../../components/home/HomeHero';
import GrowthCard from '../../components/home/GrowthCard';
import MoodTodayCard, { MOOD_OPTIONS } from '../../components/home/MoodTodayCard';
import type { MoodTodayController } from '../../components/home/MoodTodayCard';
import MoodInsightCard from '../../components/home/MoodInsightCard';
import HomeQuestionnaire from '../../components/HomeQuestionnaire';
import HomeFooter from '../../components/home/HomeFooter';
import './home-page.css';

/**
 * 「心灵花园」首页。
 *
 * 页面本身**只负责取数与编排**：布局与视觉在 `components/home/*` 里，
 * 因此每个板块都能单独调整而不会牵动打卡 / 成长这些业务逻辑。
 *
 * 数据来源（全部是项目里已有的真实接口，没有新增后端）：
 *   GET /api/extra/growth?days=7        → 我的成长 + 心理状态洞察（一次请求喂两块）
 *   GET /api/mood/checkin/history       → 今天是否已打卡 / 连续天数 / 今天填过的感受
 *   POST|PUT /api/mood/checkin          → 打卡与"点错了改一个"（PUT 原地更正）
 *   POST /api/extra/sleep               → 打卡后的可选睡眠记录
 *
 * ⚠️ 与旧首页相比**刻意保留**的行为（都踩过坑，不要在重构里丢掉）：
 *   · 打卡支持更正：今天已打卡时走 PUT，点错表情不会被锁一整天；
 *   · 连续天数从昨天起算：今天还没打卡时 streak 不归零，与后端成长口径一致；
 *   · `newlyUnlocked` 用累加而不是覆盖：更正打卡返回空数组时不会把刚弹出的徽章吃掉；
 *   · 睡眠记录只报 hours，床/醒时间由前端按今天 07:00 反推，避免 Invalid Date 造成 500。
 */
export default function Home() {
  const navigate = useNavigate();
  const { user } = useAuthStore();

  const [greeting, setGreeting] = useState('你好');
  const [checkedIn, setCheckedIn] = useState(false);
  const [selectedMood, setSelectedMood] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [streak, setStreak] = useState(0);

  // 今天填过的那句感受
  const [note, setNote] = useState('');
  const [noteSaving, setNoteSaving] = useState(false);
  const [noteSaved, setNoteSaved] = useState(false);

  // 睡眠步骤
  const [sleepStep, setSleepStep] = useState(false);
  const [sleepHours, setSleepHours] = useState<number | null>(null);
  const [sleepLoading, setSleepLoading] = useState(false);

  // 打卡顺带解锁的徽章
  const [unlockedBadges, setUnlockedBadges] = useState<Array<{ id: string; name: string; icon: string }>>([]);

  // 成长 / 洞察
  const [growth, setGrowth] = useState<GrowthOverview | null>(null);
  const [growthLoading, setGrowthLoading] = useState(true);

  useEffect(() => {
    const hour = new Date().getHours();
    if (hour >= 5 && hour < 11) setGreeting('早上好');
    else if (hour >= 11 && hour < 14) setGreeting('中午好');
    else if (hour >= 14 && hour < 18) setGreeting('下午好');
    else setGreeting('晚上好');
  }, []);

  /** 最近的打卡记录：数组直接就是 data（旧代码读 res.data.checkIns 恒为 undefined） */
  const fetchCheckIns = async (days: number): Promise<any[]> => {
    try {
      const res: any = await moodApi.getCheckInHistory(days);
      return Array.isArray(res?.data) ? res.data : (res?.data?.checkIns || []);
    } catch {
      return [];
    }
  };

  const checkTodayCheckIn = useCallback(async () => {
    const checkIns = await fetchCheckIns(7);
    const today = dayjs().format('YYYY-MM-DD');
    const todayCheckIn = checkIns.find((c: any) => dayjs(c.checkInDate).format('YYYY-MM-DD') === today);
    if (!todayCheckIn) return;

    setCheckedIn(true);
    // mood 存的是中文标签（如「不错」），selectedMood 却是数组下标，必须反查
    const idx = MOOD_OPTIONS.findIndex((m) => m.label === todayCheckIn.mood);
    setSelectedMood(idx >= 0 ? idx : null);
    // 只有"用户自己写的那句"才回填：默认备注是「😊 不错」这种由系统拼出来的
    if (idx >= 0) {
      const fallback = `${MOOD_OPTIONS[idx].emoji} ${MOOD_OPTIONS[idx].label}`;
      const saved = typeof todayCheckIn.note === 'string' ? todayCheckIn.note.trim() : '';
      if (saved && saved !== fallback) setNote(saved);
    }
  }, []);

  const loadStreak = useCallback(async () => {
    const checkIns = await fetchCheckIns(365);
    const days = new Set(checkIns.map((c: any) => dayjs(c.checkInDate).format('YYYY-MM-DD')));
    // 与后端成长指标同一口径：今天还没打卡就从昨天起算，不因此归零
    let count = 0;
    let cursor = dayjs();
    if (!days.has(cursor.format('YYYY-MM-DD'))) cursor = cursor.subtract(1, 'day');
    while (days.has(cursor.format('YYYY-MM-DD')) && count < 400) {
      count += 1;
      cursor = cursor.subtract(1, 'day');
    }
    setStreak(count);
  }, []);

  const loadGrowth = useCallback(async () => {
    try {
      const res: any = await growthApi.getOverview(7);
      setGrowth((res?.data ?? null) as GrowthOverview | null);
    } catch {
      // 不编造数值：拿不到就停在骨架屏/空态，页面其余部分照常可用
      setGrowth(null);
    } finally {
      setGrowthLoading(false);
    }
  }, []);

  useEffect(() => {
    checkTodayCheckIn();
    loadStreak();
    loadGrowth();
  }, [checkTodayCheckIn, loadStreak, loadGrowth]);

  /**
   * 打卡 / 更正今天的打卡。
   * 今天已经打过卡时走 PUT（原地更正），所以点错表情不会被锁一整天。
   */
  const handleMoodCheckIn = async (moodIdx: number) => {
    if (loading) return;
    const m = MOOD_OPTIONS[moodIdx];
    const revising = checkedIn;
    setLoading(true);
    try {
      const payload = { mood: m.label, score: m.score, note: note.trim() || `${m.emoji} ${m.label}` };
      const res: any = revising ? await moodApi.updateCheckIn(payload) : await moodApi.checkIn(payload);
      // 累加而不是覆盖：更正打卡时接口返回空的 newlyUnlocked，
      // 直接用空数组覆盖会让刚弹出的解锁提示瞬间消失。
      const fresh: Array<{ id: string; name: string; icon: string }> = res?.data?.newlyUnlocked || [];
      if (fresh.length) {
        setUnlockedBadges((prev) => {
          const merged = [...prev];
          fresh.forEach((b) => { if (!merged.some((x) => x.id === b.id)) merged.push(b); });
          return merged;
        });
      }
      setCheckedIn(true);
      setSelectedMood(moodIdx);
      message.success({
        content: revising ? `已改成：${m.emoji} ${m.label}` : `已记录：${m.emoji} ${m.label}`,
        duration: 2,
      });
      loadStreak();
      loadGrowth();
      // 首次打卡后才问睡眠；更正时不再重复问
      if (!revising) setSleepStep(true);
    } catch (e: any) {
      // 旧实现只报笼统的「记录失败」，看不出是「今天已经打过卡」还是网络问题
      message.error(e?.message || '记录失败，再试试？');
      checkTodayCheckIn();
    } finally {
      setLoading(false);
    }
  };

  /** 保存"今天的一句感受"（同一天的打卡原地更正，不会多出一条记录） */
  const handleSaveNote = async () => {
    if (selectedMood === null || noteSaving) return;
    const text = note.trim();
    if (!text) {
      message.warning('先写一句现在的心情吧~');
      return;
    }
    const m = MOOD_OPTIONS[selectedMood];
    setNoteSaving(true);
    try {
      await moodApi.updateCheckIn({ mood: m.label, score: m.score, note: text });
      setNoteSaved(true);
      window.setTimeout(() => setNoteSaved(false), 2600);
      loadGrowth();
    } catch (e: any) {
      message.error(e?.message || '没记上，再试一次？');
    } finally {
      setNoteSaving(false);
    }
  };

  const handleSleepRecord = async (hours: number) => {
    setSleepHours(hours);
    setSleepLoading(true);
    try {
      // 只报"睡了几个小时"时按今天 07:00 起床往前推；直接给 hours，
      // 避免 bedTime:'' / wakeTime:'' 生成 Invalid Date 导致接口 500、前端又静默吞掉。
      const wake = dayjs().hour(7).minute(0).second(0).millisecond(0);
      const bed = wake.subtract(hours, 'hour');
      await extraApi.recordSleep({
        bedTime: bed.toISOString(),
        wakeTime: wake.toISOString(),
        hours,
        // 质量按 1-10 存（旧代码传 80/60/40，会写出「质量 800 分」的脏数据）
        quality: hours >= 7 ? 8 : hours >= 5 ? 6 : 4,
      });
      message.success(`已记录睡眠 ${hours} 小时`);
    } catch {
      // 不再静默：这一步失败过一次（sleep_records 全表 0 行），必须让用户看见
      message.warning('睡眠没记上，稍后再试一次');
    } finally {
      setSleepLoading(false);
      window.setTimeout(() => setSleepStep(false), 1500);
    }
  };

  const moodController: MoodTodayController = useMemo(() => ({
    checkedIn,
    selectedMood,
    checking: loading,
    streak,
    select: handleMoodCheckIn,
    note,
    onNoteChange: setNote,
    saveNote: handleSaveNote,
    noteSaving,
    noteSaved,
    noteMaxLength: 60,
    sleepStep,
    sleepHours,
    sleepLoading,
    recordSleep: handleSleepRecord,
    dismissSleep: () => setSleepStep(false),
  // handleMoodCheckIn / handleSaveNote / handleSleepRecord 是每次渲染新建的函数，
  // 这里刻意只依赖它们读取到的状态，避免每次渲染都把 controller 换新导致子组件重渲染
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }), [checkedIn, selectedMood, loading, streak, note, noteSaving, noteSaved, sleepStep, sleepHours, sleepLoading]);

  return (
    <div className="gh-home">
      {/* 背后的柔光装饰层：把"通透、有层次"做实，纯装饰不参与交互 */}
      <div className="gh-home__glow" aria-hidden="true">
        <span />
        <span />
        <span />
      </div>

      <HomeHero
        greeting={greeting}
        nickname={user?.nickname || '朋友'}
        checkedInToday={checkedIn}
        onStartChat={() => navigate('/chat')}
      />

      <div className="gh-home__body">
        {/* ---------- 我的成长 + 今日心情 ---------- */}
        <section className="gh-sec" aria-label="我的成长与今日心情">
          <div className="gh-split">
            <GrowthCard data={growth} loading={growthLoading} onOpenGrowth={() => navigate('/growth')} />
            <MoodTodayCard controller={moodController} />
          </div>

          {/* 打卡顺带解锁的徽章：接口本来就返回 newlyUnlocked，别把它丢掉 */}
          {unlockedBadges.length > 0 && (
            <div className="gh-home__badges" role="status">
              <span className="gh-home__badges-icon" aria-hidden="true">🎉</span>
              <span className="gh-home__badges-text">
                解锁新徽章：{unlockedBadges.map((b) => `${b.icon} ${b.name}`).join('、')}
              </span>
              <button type="button" className="gh-btn gh-btn--sm gh-btn--glass" onClick={() => navigate('/growth')}>
                去「我的成长」看看
              </button>
            </div>
          )}
        </section>

        {/* ---------- 心理状态 + 心理状态自查：并排一行两栏（窄屏自动回退单栏） ----------
            这一行**外层**才挂 `.gh-sec`：由它负责整行（≥1080 时 1240px）的最大宽度与左右留白，
            两个子块则完全由 `.gh-cols2` 的 grid 决定宽度 —— 子块自己不能再挂 `.gh-sec`，
            否则会各自再缩到半栏宽、并在半栏里叠一层内边距。
            注：MoodInsightCard 内部自带 `.gh-sec` 的那层已下掉，改动见该组件文件头。 */}
        <section className="gh-sec gh-cols2" aria-label="心理状态与心理状态自查">
          <MoodInsightCard data={growth} loading={growthLoading} onOpenProfile={() => navigate('/profile')} />

          <HomeQuestionnaire />
        </section>
      </div>

      <HomeFooter />
    </div>
  );
}
