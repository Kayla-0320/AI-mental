/**
 * 感知能力开关 —— 声明本平台当前「真正具备」的传感器/算法能力
 *
 * 为什么需要这个文件：
 * 有些模态在界面上有卡片，但前端既没有对应硬件，也没有可用算法。
 * 如果仍然让它们输出数值，那些数值只能来自噪声或凭空推算，
 * 会被用户/咨询师误当成真实测量结果（例如「心率 180bpm」其实是摄像头曝光噪声）。
 * 因此统一在这里声明能力，所有相关 Hook / 融合 / UI 都以它为准：
 * 能力为 false 的模态一律不产出数值、不参与融合、不进入报告与上报。
 *
 * 注意：基础感知（键盘/文字/语音声学/面部微表情）、行为激活、眼动、
 * 语音语义、昼夜活动记录这些是真采的，不在本开关的管理范围内。
 */

/** 读取可选环境变量开关，缺省即视为未实现 */
function envFlag(value: unknown): boolean {
  return value === 'true' || value === '1';
}

const env = (import.meta as any).env || {};

/**
 * ❤️ rPPG 心率 / HRV —— 未实现
 *
 * 现状：`useRPPG` 只把摄像头画面的绿色通道均值做了一次极简峰值计数，
 * 摄像头自动曝光、环境光变化和头部晃动都会产生"峰"，
 * 于是输出 150~180bpm 这类恒定偏高的假值。
 * 真正的 rPPG 需要固定曝光 + 带通滤波 + 频域峰值判定，
 * 且手机/笔记本前置摄像头在普通室内光下信噪比通常不足以测心率。
 */
export const RPPG_AVAILABLE = envFlag(env.VITE_ENABLE_RPPG);

/**
 * 🌬️ 呼吸频率 —— 未实现
 *
 * 现状：`useBreathing` 用麦克风 RMS 能量的过零率估算呼吸，
 * 说话、风扇、键盘声都会带偏结果，读数在 5~40 次/分之间乱跳。
 * 按产品口径，呼吸应当从语音识别的停顿/气口节奏来推断，
 * 但语音识别本身在当前环境不稳定，因此一并关闭，不输出数值。
 */
export const BREATHING_AVAILABLE = envFlag(env.VITE_ENABLE_BREATHING);

/**
 * 😴 睡眠 —— 未实现（不可自动检测）
 *
 * 平台没有可穿戴设备与夜间传感器，
 * 既不能测睡眠时长，也不能测睡眠质量。
 * 睡眠数据只允许来自用户「主动手动记录」，任何模型推算都不算测量。
 */
export const SLEEP_AUTO_DETECTION_AVAILABLE = envFlag(env.VITE_ENABLE_SLEEP_AUTO);

/**
 * 👁️ 每眼「睁闭眼」第二判定通道（浏览器内 ONNX，46 KB，Apache-2.0）—— 默认关闭
 *
 * 现状：`useEyeStateModel` 用 `open-closed-eye-0001` 对双眼 patch 各判一次，
 * 与 MediaPipe blendshape 做**一致率统计**，结果只上报、不参与风险。
 *
 * 为什么默认关闭：
 *   1. 首次启用需要下载 onnxruntime-web 的 wasm 运行时（约 13.6 MB，之后浏览器缓存），
 *      对"只多一个 46 KB 模型"来说是一笔真实成本；
 *   2. 模型训练域是**近红外灰度**眼图（MRL Eye），在本平台 RGB 画面上存在域偏移：
 *      实测在 RT-BENE 标注 patch 上 AUC 0.92–0.97，但没有任何阈值能让闭眼召回达到 0.9，
 *      因此**必须先采集真实设备上的分数分布来标定阈值**再启用。
 * 开启方式：`VITE_ENABLE_EYE_STATE_MODEL=1`，并按
 * `docs/blink_signal_source_selection.md` 的步 B 完成标定。
 */
export const EYE_STATE_MODEL_AVAILABLE = envFlag(env.VITE_ENABLE_EYE_STATE_MODEL);

/** 心率不可用时，界面统一展示的说明 */
export const RPPG_UNAVAILABLE_REASON = '本设备无心率传感器，rPPG 未启用';

/** 呼吸不可用时，界面统一展示的说明 */
export const BREATHING_UNAVAILABLE_REASON = '待语音识别稳定后由语音节奏推断';

/** 睡眠不可自动检测时，界面统一展示的说明 */
export const SLEEP_UNAVAILABLE_REASON = '无夜间传感器，睡眠仅支持手动记录';

/** 第二判定通道未启用时，界面统一展示的说明 */
export const EYE_STATE_MODEL_UNAVAILABLE_REASON = '第二眼状态模型未启用（需先标定阈值）';
