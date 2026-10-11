import { useState } from 'react';
import { Modal } from 'antd';
import { useNavigate } from 'react-router-dom';
import { HOTLINES } from './hotlines';

/**
 * 首页页脚。
 *
 * 「关于我们 / 隐私政策 / 用户协议」在项目里**没有对应路由**，所以不用死链占位：
 * 点开是弹窗，内容如实说明平台的定位与边界（尤其是"不做医学诊断"这一条）。
 */

function LogoMark({ size = 32 }: { size?: number }) {
  return (
    <svg viewBox="0 0 48 48" width={size} height={size} aria-hidden="true">
      <defs>
        <linearGradient id="ghFootPetal" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#ffb3cd" />
          <stop offset="100%" stopColor="#ef6f9f" />
        </linearGradient>
      </defs>
      <g>
        {[0, 72, 144, 216, 288].map((deg) => (
          <ellipse
            key={deg}
            cx="24"
            cy="15"
            rx="7.6"
            ry="11"
            fill="url(#ghFootPetal)"
            opacity="0.92"
            transform={`rotate(${deg} 24 24)`}
          />
        ))}
        <circle cx="24" cy="24" r="5.6" fill="#ffd9a0" />
        <circle cx="24" cy="24" r="2.4" fill="#f6b25c" />
      </g>
    </svg>
  );
}

const LEGAL: Record<string, { title: string; body: string[] }> = {
  about: {
    title: '关于我们',
    body: [
      '心灵花园是一个面向青少年的 AI 心理健康陪伴平台：用多模态情绪感知、心理评估与智能干预，陪你识别情绪、练习放松、并在需要时找到专业的人。',
      '我们坚持三件事：不做诊断、不推荐药物、不拿你的情绪数据做无关的事。情绪数据只用来生成你自己的状态参考，危机识别到高风险时会引导你联系真人与热线。',
      '平台仍在持续建设中，欢迎通过「反馈评价」告诉我们哪里不好用。',
    ],
  },
  privacy: {
    title: '隐私政策',
    body: [
      '面部表情、语音语调等多模态数据默认在**你的浏览器本机**完成推理，原始音视频不上传；上传的只有分析后的少量数值特征，用于生成你自己的心理画像与趋势。',
      '打卡记录、测评答案、对话内容按账号隔离存储，你可以随时在「设置」中要求导出或删除。',
      '我们不会将你的心理健康数据用于广告投放，也不会未经同意向第三方披露。涉及危机风险时，会优先引导你联系可信任的成年人与专业机构。',
    ],
  },
  terms: {
    title: '用户协议',
    body: [
      '本平台提供的自查、测评与 AI 对话结果仅供自我参考，**不构成医学诊断，也不能替代面诊**。',
      '如果你出现持续两周以上的情绪低落、睡眠或进食明显受影响，或出现伤害自己的念头，请立刻告诉家长、老师，并拨打页脚列出的 24 小时热线。',
      '请勿在对话中填写真实的身份信息、住址等敏感内容；也请尊重社区里其他同伴的隐私。',
    ],
  },
};

export default function HomeFooter() {
  const navigate = useNavigate();
  const [legal, setLegal] = useState<keyof typeof LEGAL | null>(null);

  return (
    <footer className="gh-footer">
      <div className="gh-sec">
        <div className="gh-footer__grid">
          <div>
            <div className="gh-footer__brand">
              <LogoMark size={30} />
              <span className="gh-footer__brand-name">心灵花园</span>
            </div>
            <p className="gh-footer__intro">
              陪你，遇见更好的自己。AI 心理伙伴 24 小时倾听，
              也帮你在需要的时候找到真实的人。
            </p>
          </div>

          <div>
            <div className="gh-footer__col-title">开始使用</div>
            <button type="button" className="gh-footer__link" onClick={() => navigate('/chat')}>AI 倾诉</button>
            <button type="button" className="gh-footer__link" onClick={() => navigate('/profile')}>心理画像</button>
            <button type="button" className="gh-footer__link" onClick={() => navigate('/healing')}>疗愈空间</button>
            <button type="button" className="gh-footer__link" onClick={() => navigate('/companions')}>同伴社区</button>
          </div>

          <div>
            <div className="gh-footer__col-title">我的花园</div>
            <button type="button" className="gh-footer__link" onClick={() => navigate('/growth')}>我的成长</button>
            {/* 页面标题是「我的数字孪生」（MyDigitalTwin.tsx），入口名保持一致，不要写成「数字学生」 */}
            <button type="button" className="gh-footer__link" onClick={() => navigate('/twin')}>数字孪生</button>
            <button type="button" className="gh-footer__link" onClick={() => navigate('/experts')}>我的咨询</button>
            <button type="button" className="gh-footer__link" onClick={() => navigate('/settings')}>设置</button>
          </div>

          <div>
            <div className="gh-footer__col-title">了解更多</div>
            {/* 「心理知识」原先滚到首页的推荐内容区，该区块已下线；内容仍可从顶栏搜索检索 */}
            <button type="button" className="gh-footer__link" onClick={() => setLegal('about')}>关于我们</button>
            <button type="button" className="gh-footer__link" onClick={() => setLegal('privacy')}>隐私政策</button>
            <button type="button" className="gh-footer__link" onClick={() => setLegal('terms')}>用户协议</button>
          </div>
        </div>

        {/* 危机资源常驻页脚：任何时候都够得到，不只藏在弹窗里 */}
        <div className="gh-footer__crisis">
          <span className="gh-footer__crisis-text">
            如果此刻特别难受，别一个人扛着 —— <strong>{HOTLINES[0].name}</strong>
            <a href={`tel:${HOTLINES[0].phone}`}> {HOTLINES[0].phone}</a>
            ，24 小时有人接听。
          </span>
          <a href={`tel:${HOTLINES[0].phone}`}>立即拨打</a>
        </div>

        <div className="gh-footer__bottom">
          <span>© {new Date().getFullYear()} 心灵花园 · 本平台不提供医学诊断，结果仅供自我参考</span>
          <span className="gh-footer__legal">
            <button type="button" onClick={() => setLegal('privacy')}>隐私政策</button>
            <button type="button" onClick={() => setLegal('terms')}>用户协议</button>
            <button type="button" onClick={() => setLegal('about')}>联系我们</button>
          </span>
        </div>
      </div>

      <Modal
        open={legal !== null}
        onCancel={() => setLegal(null)}
        footer={null}
        width={560}
        title={legal ? LEGAL[legal].title : ''}
      >
        {legal
          ? LEGAL[legal].body.map((p) => (
            <p key={p.slice(0, 12)} className="gh-modal__text">{p}</p>
          ))
          : null}
      </Modal>
    </footer>
  );
}
