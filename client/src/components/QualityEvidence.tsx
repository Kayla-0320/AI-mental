import { CameraOutlined, CheckOutlined, FilterOutlined, MessageOutlined } from '@ant-design/icons';
import { Progress, Tag } from 'antd';
import { qualityReason, type InnovationReference } from '../services/innovationPresentation';
import './quality-evidence.css';
import LiveQualityPreview, { type LiveQualityState } from './LiveQualityPreview';
import QualityCorrectionProcess from './QualityCorrectionProcess';

const finite = (value: number | undefined): value is number => typeof value === 'number' && Number.isFinite(value);
const fraction = (value: number | undefined): value is number => finite(value) && value >= 0 && value <= 1;
const format = (value: number): string => Number(value.toFixed(2)).toLocaleString('zh-CN');

/** 只画已记录的画质证据；是否采用画面以本轮结果为准。 */
export default function QualityEvidence({ reference, waiting = false, live, active = true }: {
  reference: InnovationReference | null; waiting?: boolean; live?: LiveQualityState; active?: boolean;
}) {
  const quality = waiting ? undefined : reference?.quality.visual;
  const metrics = quality?.metrics ?? {};
  const fallback = !waiting && !!reference?.warnings.some(reason =>
    reason === 'quality_model_fallback_to_text' || reason.startsWith('visual_') && reason.includes('fallback'));
  const adopted = !waiting && !fallback && reference?.available === true && reference.observed_modalities.includes('visual');
  const textOnly = !waiting && !adopted && reference?.available === true && reference.observed_modalities.includes('text');
  const score = quality && finite(quality.quality_score) ? Math.max(0, Math.min(100, quality.quality_score * 100)) : null;
  const frames = finite(metrics.frame_count) && metrics.frame_count >= 0 ? Math.floor(metrics.frame_count) : null;
  const tone = fallback ? 'warning' : adopted ? 'success' : 'pending';
  const color = fallback ? '#c18a45' : adopted ? '#4d9684' : '#9b91af';
  const label = waiting ? '正在等这一轮' : fallback ? '已退回文字' : adopted ? '已采用画面' : quality ? '画面暂未采用' : textOnly ? '本轮仅用文字' : '等待画面记录';
  const explanation = waiting ? '这一轮回复后，更新画面检查结果。'
    : fallback ? '这次画面未通过检查，我先靠你的文字来理解。'
      : adopted ? '这一轮画面通过了检查，可以作为辅助参考。'
        : quality ? '已记录画质检查；这一轮没有采用画面作为参考。'
          : textOnly ? '这一轮没有画面记录，我先靠你的文字来理解。' : '打开摄像头后再聊一轮，这里就会显示画面检查结果。';
  const details = [
    ['luminance_mean', '平均亮度', ' / 255'], ['contrast_mean', '画面对比度', ''],
    ['laplacian_variance_128px', '清晰度代理', ''], ['quality_mos_worst', '辅助画质 MOS', ' / 5'],
  ];
  const exposure = fraction(metrics.dark_frame_fraction) && fraction(metrics.overexposed_frame_fraction)
    ? 1 - Math.min(1, metrics.dark_frame_fraction + metrics.overexposed_frame_fraction) : null;
  const signals = [
    { key: 'exposure', label: '曝光', value: exposure, note: '未过暗或过亮的采样帧占比' },
    { key: 'contrast', label: '对比度', value: fraction(metrics.low_contrast_frame_fraction) ? 1 - metrics.low_contrast_frame_fraction : null, note: '未触发低对比度检查的采样帧占比' },
    { key: 'sharpness', label: '清晰度', value: fraction(metrics.blur_proxy_fraction) ? 1 - metrics.blur_proxy_fraction : null, note: '未触发模糊检查的采样帧占比' },
  ];
  const issues = quality?.reason_codes.filter(reason => !reason.endsWith('_unverified')) ?? [];
  return <section className={`quality-visual quality-visual--${tone}`} data-testid="quality-evidence" aria-label="本轮画面质量可视化">
    <QualityCorrectionProcess reference={reference} waiting={waiting} active={active} />
    {live && <LiveQualityPreview state={live} active={active} />}
    <div className="quality-visual-heading">
      <span>本轮画面检查</span><span className="quality-visual-badge">{label}</span>
    </div>
    <div className="quality-visual-overview">
      <div className="quality-visual-gauge" role="img" aria-label={score === null ? '画面质量参考：尚无记录' : `画面质量参考：${format(score)} / 100`}>
        <Progress type="circle" percent={score ?? 0} size={104} strokeWidth={7} strokeColor={color} trailColor="#eeeaf2"
          format={() => <div className="quality-visual-number"><strong>{score === null ? '—' : format(score)}</strong><span>{score === null ? '尚无记录' : '/ 100'}</span></div>} />
      </div>
      <div className="quality-visual-summary">
        <strong>画面质量参考</strong>
        <p>{explanation}</p>
        <div className="quality-visual-frames" aria-label={frames === null ? '采样帧数未记录' : `已检查 ${frames} 帧`}>
          <span className="quality-visual-frame-icons" aria-hidden="true">{[0, 1, 2].map(index =>
            <i key={index} className={frames !== null && index < frames ? 'is-recorded' : ''}><CameraOutlined /></i>)}</span>
          <span>{frames === null ? '等待采样记录' : `${frames} 帧已检查`}</span>
        </div>
      </div>
    </div>

    <div className="quality-visual-signals" aria-label="采样帧质量分布">
      <div className="quality-visual-section-heading"><strong>采样帧的情况</strong><span>未触发问题的帧占比</span></div>
      {signals.map(signal => <div className="quality-visual-signal" key={signal.key} title={signal.note}>
        <span>{signal.label}</span>
        <div className={`quality-visual-track${signal.value === null ? ' is-empty' : ''}`} role="img"
          aria-label={`${signal.label}：${signal.value === null ? '未记录' : `${format(signal.value * 100)}% 的采样帧未触发该项问题`}`}>
          {signal.value !== null && <>
            <span className="quality-visual-good" style={{ width: `${signal.value * 100}%` }} />
            <span className="quality-visual-flagged" style={{ width: `${(1 - signal.value) * 100}%` }} />
          </>}
        </div>
        <strong>{signal.value === null ? '—' : `${Math.round(signal.value * 100)}%`}</strong>
      </div>)}
      <div className="quality-visual-legend"><span><i />未触发问题</span><span><i />有异常</span><span><i />未记录</span></div>
    </div>

    <div className="quality-visual-route">
      <div className="quality-visual-section-heading"><strong>这一轮，怎么处理画面</strong></div>
      <ol aria-label="画面处理流程">
        <li className={quality ? 'is-done' : ''}><span className="quality-visual-route-icon"><CameraOutlined /></span><strong>采样</strong><small>{frames === null ? '等待记录' : `${frames} 帧记录`}</small></li>
        <li className={fallback ? 'is-warning' : quality ? 'is-done' : ''}><span className="quality-visual-route-icon"><FilterOutlined /></span><strong>质量检查</strong><small>{fallback ? '已触发纠偏' : quality ? '已检查' : '尚未检查'}</small></li>
        <li className={fallback ? 'is-warning' : adopted || textOnly ? 'is-done' : ''}><span className="quality-visual-route-icon">{adopted ? <CheckOutlined /> : <MessageOutlined />}</span><strong>{adopted ? '图文参考' : fallback || textOnly ? '文字参考' : '等待结果'}</strong><small>{adopted ? '画面已采用' : fallback ? '画面已跳过' : '画面未采用'}</small></li>
      </ol>
    </div>

    {issues.length > 0 && <div className="quality-visual-issues">{issues.map(reason => <Tag key={reason}>{qualityReason(reason)}</Tag>)}</div>}
    {quality && <details className="quality-visual-details"><summary>查看测量数值与检查说明</summary>
      <dl>{details.filter(([key]) => finite(metrics[key])).map(([key, title, unit]) => <div key={key}><dt>{title}</dt><dd>{format(metrics[key])}{unit}</dd></div>)}</dl>
      {quality.reason_codes.filter(reason => reason.endsWith('_unverified')).map(reason => <p key={reason}>{qualityReason(reason)}</p>)}
      {finite(metrics.quality_mos_worst) && <p>MOS 是模型的辅助画质分；是否采用画面以本轮处理结果为准。</p>}
    </details>}
    <p className="quality-visual-note">这里只检查画面质量，不评价你的情绪。记录随每轮对话更新，原始缩略图不保存在档案里。</p>
  </section>;
}
