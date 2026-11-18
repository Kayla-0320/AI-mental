import { useEffect, useState } from 'react';
import { CameraOutlined, CheckOutlined, FilterOutlined, MessageOutlined, PlayCircleOutlined } from '@ant-design/icons';
import { Button, Progress } from 'antd';
import { expertLabel, qualityReason, type InnovationReference } from '../services/innovationPresentation';
import './quality-process.css';

const steps = ['接收画面', '检查问题', '调整参考', '形成理解'];
const percent = (value: number): string => `${Number((value * 100).toFixed(1))}%`;

/** 按本轮真实记录解释纠偏链路；回放顺序不冒充模型运行的实时时序。 */
export default function QualityCorrectionProcess({ reference, waiting, active }: {
  reference: InnovationReference | null; waiting: boolean; active: boolean;
}) {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const quality = reference?.quality.visual;
  const fallback = !!reference?.warnings.some(reason => reason === 'quality_model_fallback_to_text'
    || reason.startsWith('visual_') && reason.includes('fallback'));
  const adopted = reference?.available === true && reference.observed_modalities.includes('visual') && !fallback;
  const ready = !waiting && !!reference && (!!quality || fallback || reference.available);
  const metrics = quality?.metrics ?? {};
  const reasons = quality?.reason_codes.filter(reason => !reason.endsWith('_unverified')) ?? [];
  const signature = JSON.stringify([reference, waiting]);
  useEffect(() => { setPlaying(false); setStep(0); }, [signature]);
  useEffect(() => {
    if (!playing || !active || !ready) return;
    const timer = window.setTimeout(() => {
      if (step >= 3) setPlaying(false);
      else setStep(value => value + 1);
    }, 1800);
    return () => window.clearTimeout(timer);
  }, [playing, step, active, ready]);
  useEffect(() => { if (!active) setPlaying(false); }, [active]);
  const experts = reference?.expert_names.map((name, i) => ({ name, weight: reference.expert_weights[i] }))
    .filter(item => Number.isFinite(item.weight) && item.weight > 0) ?? [];
  const evidence: [string, string][] = [
    ['dark_frame_fraction', '过暗'], ['overexposed_frame_fraction', '过亮'],
    ['low_contrast_frame_fraction', '对比度偏低'], ['blur_proxy_fraction', '模糊'],
  ];
  const hasFrameCount = Number.isFinite(metrics.frame_count);
  return <section className="quality-process" aria-label="质量纠偏过程回放" data-testid="quality-process">
    <div className="quality-process-head"><strong>质量纠偏是怎样发生的</strong><span>{ready ? '本轮记录回放' : '等待本轮记录'}</span></div>
    <ol className="quality-process-steps" aria-label="选择纠偏步骤">{steps.map((title, index) => <li key={title}>
      <button type="button" disabled={!ready} aria-current={step === index ? 'step' : undefined}
        className={ready && index <= step ? 'is-reached' : ''} onClick={() => { setPlaying(false); setStep(index); }}>
        <span>{index + 1}</span><small>{title}</small>
      </button>
    </li>)}</ol>
    <div className="quality-process-stage" aria-live="polite" aria-atomic="true">
      {!ready ? <div className="quality-process-empty"><FilterOutlined /><strong>{waiting ? '正在等待这一轮的检查记录' : '聊一轮后，可以在这里回放纠偏过程'}</strong><p>会逐步展示检查依据、参考调整和最终处理结果。</p></div>
        : <>
          <div className="quality-process-stage-title"><b>0{step + 1}</b><strong>{steps[step]}</strong></div>
          {step === 0 && <>
            <div className="quality-process-input" aria-label={hasFrameCount ? `本轮接收 ${metrics.frame_count} 帧` : '本轮采样帧数未记录'}>
              {[0, 1, 2].map(index => <span key={index} className={hasFrameCount && index < metrics.frame_count ? 'is-captured' : ''}><CameraOutlined /><small>采样 {index + 1}</small></span>)}
            </div>
            <p>{hasFrameCount ? `这一轮实际收到 ${metrics.frame_count} 张缩略帧，先检查画面质量，再决定是否参考。` : fallback ? '这一轮的画面未能通过输入校验，没有可供检查的有效帧记录。' : '这一轮没有画面质量记录，不能还原画面检查过程。'}</p>
            <small className="quality-process-caption">这里只显示采样记录，不保存或补造原始画面。</small>
          </>}
          {step === 1 && <>
            <div className="quality-process-findings">{evidence.map(([key, title]) => {
              const value = metrics[key];
              const measured = Number.isFinite(value) && value >= 0 && value <= 1;
              return <div key={key}><span>{title}的帧</span><strong className={measured && value > .5 ? 'is-issue' : ''}>{measured ? percent(value) : '未记录'}</strong></div>;
            })}</div>
            <p>{reasons.length ? `记录到：${reasons.map(qualityReason).join('、')}。`
              : quality ? '本轮未记录额外的画质问题。' : fallback ? '画面格式或新鲜度检查未通过，后续跳过画面参考。' : '未留下画质检查依据。'}</p>
            {reference?.warnings.includes('quality_model_fallback_to_text') && <p>辅助画质模型触发了退回文字{Number.isFinite(metrics.quality_mos_worst) ? `，本轮最低 MOS 为 ${metrics.quality_mos_worst.toFixed(2)} / 5` : ''}。</p>}
          </>}
          {step === 2 && <>
            <div className="quality-process-gate" aria-label="质量门控后的模态状态">
              <div><MessageOutlined /><strong>文字</strong><span>{reference?.observed_modalities.includes('text') ? '继续参考' : '采用情况未记录'}</span></div>
              <div className={fallback ? 'is-rejected' : adopted ? 'is-kept' : ''}><CameraOutlined /><strong>画面</strong><span>{fallback ? '停止参考' : adopted ? '保留参考' : '未采用'}</span></div>
            </div>
            {quality && <div className="quality-process-factor"><span>画面质量系数</span><strong>{quality.quality_score.toFixed(2)}</strong><span>→</span><strong>{fallback ? '画面不再输入' : adopted ? '送入后续专家' : '本轮未采用画面'}</strong></div>}
            <p>{fallback ? '这次纠偏把不可靠的画面从后续参考中移除，让它不再影响本轮理解。'
              : adopted ? '画面保留，并带着本轮质量系数进入后续专家；质量系数与最终专家权重是两项不同记录。'
                : '这一轮未采用画面；没有记录的中间调整不作推测。'}</p>
          </>}
          {step === 3 && <>
            <div className={`quality-process-outcome${fallback ? ' is-fallback' : ''}`}><CheckOutlined /><strong>{!reference?.available ? '本轮未形成可用模型参考' : fallback ? '纠偏完成 · 已退回文字参考' : adopted ? '检查完成 · 保留图文参考' : '本轮采用文字参考'}</strong></div>
            {reference?.available && experts.length > 0 && <div className="quality-process-weights" aria-label="本轮最终专家权重">{experts.map(expert => <div key={expert.name}><span>{expertLabel(expert.name)}</span><Progress percent={expert.weight * 100} size="small" showInfo={false} strokeColor="#a294bc" /><strong>{percent(expert.weight)}</strong></div>)}</div>}
            <p>{reference?.available ? '以上是本轮实际采用的专家权重。没有保存纠偏前的权重，因此不展示虚构的前后差值。' : '保留已记录的画质检查结果，不把缺失模型结果当作纠偏成功。'}</p>
          </>}
        </>}
    </div>
    <div className="quality-process-controls">
      <Button size="small" disabled={!ready} icon={playing ? <span aria-hidden="true">Ⅱ</span> : <PlayCircleOutlined />} onClick={() => {
        if (playing) setPlaying(false); else { if (step === 3) setStep(0); setPlaying(true); }
      }}>{playing ? '暂停' : step === 3 ? '重新播放' : '播放过程'}</Button>
      <Button size="small" type="text" disabled={!ready || step === 0} onClick={() => { setPlaying(false); setStep(value => value - 1); }}>上一步</Button>
      <Button size="small" type="text" disabled={!ready || step === 3} onClick={() => { setPlaying(false); setStep(value => value + 1); }}>下一步</Button>
    </div>
  </section>;
}
