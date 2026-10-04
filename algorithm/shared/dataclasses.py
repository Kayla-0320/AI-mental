"""
跨模块统一数据接口定义

所有模块之间的通信必须通过这些数据类，不允许各自定义。
使用 Python 3.10+ dataclass 语法，不引入外部依赖。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RiskLevel(str, Enum):
    """风险等级枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRISIS = "crisis"


@dataclass
class EvidenceItem:
    """证据条目 —— 记录某项判断的依据来源

    Attributes:
        source: 证据来源模块名称（如 perception.text / assessment.risk）
        description: 证据的文本描述
        weight: 证据权重，范围 [0.0, 1.0]
    """
    source: str           # 证据来源模块
    description: str      # 证据描述
    weight: float         # 证据权重 [0.0, 1.0]


@dataclass
class EmotionResult:
    """情绪感知结果 —— 感知层输出统一的情绪向量

    Attributes:
        text_emotion_probs: 文本情绪概率分布，5维向量
            （对应：快乐/悲伤/焦虑/愤怒/中性）
        audio_risk_prob: 语音风险概率，None 表示无语音通道数据
        confidence: 综合置信度，范围 [0.0, 1.0]
        timestamp: Unix 时间戳，标记感知完成时刻
        evidence: 支撑该结果的证据描述列表
        crisis_keywords: 命中的危机关键词列表（安全相关，空列表表示未命中）

    注意：crisis_keywords 非空只表示「文本字面命中了危机词表」，
    不等于临床判断，下游只能据此触发人工复核流程，不得据此输出诊断结论。
    """
    text_emotion_probs: list[float]       # 5维文本情绪概率分布
    audio_risk_prob: Optional[float]      # 语音风险概率（可为空）
    confidence: float                     # 综合置信度 [0.0, 1.0]
    timestamp: float                      # Unix 时间戳
    evidence: list[str]                   # 证据描述列表
    crisis_keywords: list[str] = field(default_factory=list)  # 命中的危机关键词


@dataclass
class RiskAssessment:
    """心理风险评估结果 —— 评估层输出

    Attributes:
        phq9_estimated: PHQ-9 抑郁量表预估分数区间 (下界, 上界)
        gad7_estimated: GAD-7 焦虑量表预估分数区间 (下界, 上界)
        risk_level: 风险等级（low / medium / high / crisis）
        confidence: 评估置信度，范围 [0.0, 1.0]
        evidence: 支撑该评估的证据条目列表
    """
    phq9_estimated: tuple[float, float]   # PHQ-9 预估分数区间
    gad7_estimated: tuple[float, float]   # GAD-7 预估分数区间
    risk_level: RiskLevel                 # 风险等级枚举
    confidence: float                     # 评估置信度 [0.0, 1.0]
    evidence: list[EvidenceItem]          # 证据条目列表


@dataclass
class CrisisAlert:
    """危机警报 —— 升级层触发危机响应时的数据结构

    Attributes:
        user_id: 触发警报的用户唯一标识
        risk_score: 综合风险评分，范围 [0.0, 1.0]
        trigger_evidence: 触发本次危机的证据描述列表
        timestamp: Unix 时间戳，标记警报生成时刻
        recommended_action: 建议采取的干预动作
    """
    user_id: str                          # 用户唯一标识
    risk_score: float                     # 综合风险评分 [0.0, 1.0]
    trigger_evidence: list[str]           # 触发证据描述列表
    timestamp: float                      # Unix 时间戳
    recommended_action: str               # 建议干预动作


@dataclass
class FeatureSummary:
    """特征摘要 —— 跨模块传递的聚合特征数据

    注意：此数据类确保不包含原始文本或可识别信息，
    仅传递统计特征与概率分布，用于保护用户隐私。

    Attributes:
        text_emotion_probs: 文本情绪概率分布
        audio_risk_prob: 语音风险概率，None 表示无语音数据
        confidence: 特征置信度，范围 [0.0, 1.0]
        session_id: 会话唯一标识
        timestamp: Unix 时间戳，标记特征提取时刻
    """
    text_emotion_probs: list[float]       # 文本情绪概率分布
    audio_risk_prob: Optional[float]      # 语音风险概率（可为空）
    confidence: float                     # 特征置信度 [0.0, 1.0]
    session_id: str                       # 会话唯一标识
    timestamp: float                      # Unix 时间戳


# ============================================================
# 安全审计数据类
# ============================================================

class AuditAxis(str, Enum):
    """审计轴枚举 —— 六个安全审计维度"""
    CRISIS_DELAY = "crisis_delay"           # 危机升级延迟
    DELUSION_REINFORCEMENT = "delusion_reinforcement"  # 妄想强化
    STIGMA_REJECTION = "stigma_rejection"   # 污名化与拒绝
    SYCOPHANCY = "sycophancy"               # 谄媚倾向
    TRAJECTORY_DRIFT = "trajectory_drift"   # 轨迹漂移
    SOCRATIC_TIMING = "socratic_timing"     # 苏格拉底反问时机


class AuditAction(str, Enum):
    """审计不通过时的处理动作"""
    PASS = "pass"                           # 通过，原样输出
    REWRITE = "rewrite"                     # 重写回复
    INJECT_RESOURCE = "inject_resource"     # 注入危机资源卡
    ESCALATE_HUMAN = "escalate_human"       # 升级至人工


@dataclass
class AuditResult:
    """单轴审计结果

    Attributes:
        axis: 审计轴名称
        passed: 是否通过
        reason: 不通过的理由（通过时为空字符串）
        suggested_action: 建议的处理动作
        recommended_depth: 建议的反问深度（仅 socratic_timing 轴使用）
            0=共情模式, 1=SHALLOW, 2=MEDIUM, 3=DEEP
    """
    axis: AuditAxis
    passed: bool
    reason: str = ""
    suggested_action: AuditAction = AuditAction.PASS
    recommended_depth: int = -1  # -1 表示不适用


@dataclass
class AuditVerdict:
    """综合审计裁决 —— 五轴审计的最终结论

    Attributes:
        passed: 综合是否通过（所有轴都通过才为 True）
        results: 各轴审计结果列表
        final_action: 最终处理动作（取最严重的）
        rewritten_content: 重写后的内容（如果需要重写）
    """
    passed: bool
    results: list[AuditResult] = field(default_factory=list)
    final_action: AuditAction = AuditAction.PASS
    rewritten_content: str = ""


# ============================================================
# 数字表型特征数据类
# ============================================================

@dataclass
class PhenotypeFeature:
    """单个表型特征

    Attributes:
        name: 特征名称（如 "sleep_duration_mean"）
        value: 特征值（None 表示数据缺失）
        confidence: 特征置信度 [0, 1]
        source: 数据来源（如 "wearable" / "chat_log" / "assessment"）
        unit: 特征单位（如 "hours" / "score" / "count"）
        significant_change: 是否相对上一窗口有显著变化
    """
    name: str
    value: Optional[float]
    confidence: float
    source: str
    unit: str = ""
    significant_change: bool = False


@dataclass
class PhenotypeVector:
    """多模态数字表型向量

    整合五类特征，形成用户的综合数字表型画像。

    Attributes:
        user_id: 用户唯一标识
        time_window_days: 时间窗口（天）
        sleep_features: 睡眠特征列表
        emotion_features: 对话情感特征列表
        behavior_features: 行为特征列表
        assessment_features: 测评历史特征列表
        physiological_features: 生理特征列表
        timestamp: 特征提取时间戳
    """
    user_id: str
    time_window_days: int
    sleep_features: list[PhenotypeFeature] = field(default_factory=list)
    emotion_features: list[PhenotypeFeature] = field(default_factory=list)
    behavior_features: list[PhenotypeFeature] = field(default_factory=list)
    assessment_features: list[PhenotypeFeature] = field(default_factory=list)
    physiological_features: list[PhenotypeFeature] = field(default_factory=list)
    timestamp: float = 0.0

    def all_features(self) -> list[PhenotypeFeature]:
        """返回所有特征的扁平列表"""
        return (
            self.sleep_features
            + self.emotion_features
            + self.behavior_features
            + self.assessment_features
            + self.physiological_features
        )

    def feature_dict(self) -> dict[str, Optional[float]]:
        """返回 {特征名: 值} 字典"""
        return {f.name: f.value for f in self.all_features()}

    def missing_features(self) -> list[str]:
        """返回缺失数据的特征名列表"""
        return [f.name for f in self.all_features() if f.value is None]


# ============================================================
# 危机升级数据类
# ============================================================

class EscalationChannel(str, Enum):
    """升级通道枚举"""
    WEBHOOK = "webhook"               # Webhook 通知（模拟）
    IN_APP = "in_app"                 # 站内通知
    CONSOLE = "console"               # 控制台告警


class EscalationStatus(str, Enum):
    """升级状态枚举"""
    PENDING = "pending"               # 待处理
    DISPATCHED = "dispatched"         # 已派发
    ACKNOWLEDGED = "acknowledged"     # 已确认
    RESOLVED = "resolved"             # 已解决
    FAILED = "failed"                 # 失败


@dataclass
class EscalationResult:
    """危机升级结果 —— 升级层核心输出

    Attributes:
        alert_id: 升级事件唯一标识
        alert: 原始危机警报
        status: 升级处理状态
        channels_notified: 已通知的通道列表
        assigned_counselor: 分配的咨询师 ID（空字符串表示未分配）
        response_time_seconds: 升级响应时间（秒），目标 < 30 秒
        timestamp: 升级完成时间戳
        audit_log_id: 对应的审计日志 ID
    """
    alert_id: str
    alert: CrisisAlert
    status: EscalationStatus
    channels_notified: list[EscalationChannel] = field(default_factory=list)
    assigned_counselor: str = ""
    response_time_seconds: float = 0.0
    timestamp: float = 0.0
    audit_log_id: str = ""


# ============================================================
# 语音识别数据类
# ============================================================

class AsrEngine(str, Enum):
    """语音识别引擎枚举

    当前只落地了本地 sherpa-onnx 两条路径。刻意不提供「浏览器 Web Speech API」
    选项 —— 那条路径的声音会被送到厂商云服务（Edge→微软 Azure、Chrome→谷歌），
    与「敏感原始数据不出设备」冲突。
    """
    PARAFORMER = "sherpa-onnx-paraformer"              # 本地离线整段（/asr/transcribe）
    ZIPFORMER_STREAMING = "sherpa-onnx-streaming-zipformer"  # 本地流式边说边出字（/asr/ws）
    UNAVAILABLE = "unavailable"                        # 模型缺失或加载失败


class AsrEventType(str, Enum):
    """流式识别事件的类型"""
    PARTIAL = "partial"   # 中间结果：会被后续结果覆盖，只用于上屏
    FINAL = "final"       # 定稿：这一段话结束，可以送入下游文本/语义分析


@dataclass
class AsrResult:
    """语音转文字结果 —— 非流式（整段）识别的输出

    由 `asr/engine.py` 产生，经 `/api/v1/asr/transcribe` 暴露给前端。

    Attributes:
        text: 识别文本；可能为空字符串（该段音频无可用语音）
        engine: 实际使用的识别引擎
        duration_ms: 输入音频时长（毫秒）
        latency_ms: 本次识别耗时（毫秒）
        error: 失败原因；为空字符串表示成功
    """
    text: str
    engine: AsrEngine = AsrEngine.PARAFORMER
    duration_ms: float = 0.0
    latency_ms: float = 0.0
    error: str = ""

    @property
    def is_empty(self) -> bool:
        """是否未识别出任何文本"""
        return not self.text.strip()

    @property
    def ok(self) -> bool:
        """本次识别是否成功（成功不等同于识别出了文本）"""
        return self.engine is not AsrEngine.UNAVAILABLE and not self.error


@dataclass
class StreamingAsrUpdate:
    """流式识别的一次增量更新

    由 `asr/streaming.py` 产生，经 `/api/v1/asr/ws` 以 JSON 推给前端。

    为什么把 PARTIAL 和 FINAL 分开：
        中间结果会不断被后续结果覆盖，只适合"上屏给用户看"；下游的文本情感、
        认知扭曲、语音语义分析必须只吃定稿，否则同一句话会被反复分析，
        既浪费也算出不稳定结果。

    Attributes:
        event: partial（中间结果）或 final（定稿）
        text: 当前累计文本（flushing 之前是这一段的完整累计，不是增量片段）
        elapsed_ms: 该段音频已累计的时长（毫秒）
        latency_ms: 本次解码耗时（毫秒）
        reason: 仅 final 时有意义 —— "endpoint"（端点检测）或 "finalize"（用户主动结束）
        engine: 产出这段文本的引擎（定稿若经离线模型纠错，这里是离线引擎）
        refined: 定稿文本是否经过离线大模型复识别纠错
    """
    event: AsrEventType
    text: str
    elapsed_ms: float = 0.0
    latency_ms: float = 0.0
    reason: str = ""
    engine: AsrEngine = AsrEngine.ZIPFORMER_STREAMING
    refined: bool = False

    @property
    def is_final(self) -> bool:
        """是否为定稿事件"""
        return self.event is AsrEventType.FINAL

    @property
    def is_empty(self) -> bool:
        """是否未识别出任何文本"""
        return not self.text.strip()


# ============================================================
# 语音合成数据类
# ============================================================

class TtsEngine(str, Enum):
    """语音合成引擎枚举

    三条路径，通过环境变量 ``TTS_BACKEND`` 切换：
        * ``vits``      —— sherpa-onnx VITS 离线合成（16 kHz，零 GPU 占用）
        * ``qwen3``     —— Qwen3-TTS 1.7B 对话级合成（24 kHz，需 GPU 显存）
        * ``cosyvoice`` —— CosyVoice3 零样本克隆（24 kHz，独立微服务 + GPU）

    前两条跑在算法服务进程内；``cosyvoice`` 走**独立微服务**
    （``tts/cosyvoice_service/``），因为它的 ``torch==2.7`` + ``numpy<2``
    与本进程的 ``torch==2.11`` + ``numpy==2.5`` 直接冲突，放不进同一解释器。
    这与算法侧既有的"ASR 与 TTS 依赖/失败模式独立"是同一条原则。

    刻意不提供浏览器 ``speechSynthesis``：它的音频**不经过 Web Audio**，
    拿不到 ``MediaStream`` / ``AudioNode``，因此既不能作为回声消除的参考信号，
    也做不了文本级自回声过滤（见 docs/voice_call_plan.md §6.3 的否决理由，
    那是架构性的，不是延迟性的）。

    与 ``AsrEngine`` 同一条原则：音频只在本机流转，不出设备。
    """
    VITS = "sherpa-onnx-vits"     # 本地离线合成（/tts/ws）
    QWEN3 = "qwen3-tts"           # 本地对话级合成（/tts/ws，voice design）
    COSYVOICE = "cosyvoice3"      # 独立微服务零样本克隆（HTTP 流式，1.2× 加速）
    UNAVAILABLE = "unavailable"   # 模型缺失或加载失败


@dataclass
class TtsSynthesisResult:
    """一次语音合成的结果（计时与状态；**不承载音频数据**）

    音频本身走 WebSocket 的二进制帧，这里只保留可观测性所需的元数据 ——
    与 ``AsrResult`` 把音频丢给调用方、只回一个结果对象是同一个思路。

    ⚠️ 关于 ``first_chunk_ms``：实测（docs/tts_benchmark.md §1 发现一）
    ``sherpa_onnx.OfflineTts.generate(callback=...)`` 对每一句**只回调一次**，
    携带的是整句合成完的音频。因此 **``first_chunk_ms`` 与 ``synth_ms`` 相等**。
    保留这个字段是为了：(1) 与 benchmark 的产物字段对齐；
    (2) 将来若换成真流式 TTS，协议与消费端都不用改。
    **不要**据此认为"首块比整句快"。

    Attributes:
        seq: 本次合成的序号，与请求里的 seq 对应（客户端据此对号入座）
        engine: 实际使用的合成引擎
        spoken_text: **实际被念出来的文本**（已过术语替换，可能与请求文本不同）。
            客户端做文本级自回声过滤时要拿这个值，而不是原始请求文本 ——
            否则 `CBT → 认知行为疗法` 这类替换会让过滤失效。
        sample_rate: 输出 PCM 的采样率（Hz）
        audio_ms: 产出音频时长（毫秒）
        first_chunk_ms: 从调用到第一次回调的耗时（毫秒）；VITS 下等于 synth_ms
        synth_ms: 合成总耗时（毫秒）
        cancelled: 是否被中途取消（客户端已打断，音频不再下发）
        error: 失败原因；为空字符串表示成功
    """
    seq: int
    engine: TtsEngine = TtsEngine.VITS
    spoken_text: str = ""
    sample_rate: int = 16000
    audio_ms: float = 0.0
    first_chunk_ms: float = 0.0
    synth_ms: float = 0.0
    cancelled: bool = False
    error: str = ""

    @property
    def rtf(self) -> float:
        """实时率 = 合成耗时 / 音频时长

        判据是 **< 1.0**（否则合成追不上播放，队列越积越长，听起来"越说越慢"）。
        `voice_call_plan.md` 阶段 0 的 go 判据更严：< 0.4。
        实测本机 4 线程 0.180、8 线程 0.112（docs/tts_benchmark.md）。
        音频时长为 0 时返回 0.0（表示"无从判断"，不是"很快"）。
        """
        return self.synth_ms / self.audio_ms if self.audio_ms > 0 else 0.0

    @property
    def ok(self) -> bool:
        """本次合成是否成功（成功不等同于产出了音频）

        被取消算成功：取消是正常路径（用户打断），不是故障。
        """
        return self.engine is not TtsEngine.UNAVAILABLE and not self.error

    @property
    def is_silent(self) -> bool:
        """是否一个字都没合成出来

        单独暴露这个判断，是为了让"模型静默失效"（例如 ``max_num_sentences=1``
        导致的截断、或文本全被 OOV 吞掉）能被上游显式发现，而不是当成正常空结果。
        """
        return self.audio_ms <= 0.0


# ============================================================
# 文本流式对话数据类
# ============================================================

class ChatStreamEventType(str, Enum):
    """文本流式对话的事件类型（跨语言契约）

    同一套取值同时出现在三个地方，改动必须三处同步：
        1. Python 产出端 ``algorithm/api/intervention.py``（/smart-chat/stream）
        2. Node 中转端 ``server/src/services/algorithm-bridge.ts``（原样透传）
        3. 浏览器消费端 ``client/src/services/index.ts``

    事件顺序约定：
        meta → (delta*) → speak → done         # 低/中风险：文本流完后给合成计划
        meta → (delta*) → done
        meta → (delta*) → revise → done        # 安全审计改写了回复
        meta → error                            # 前处理失败，未产生任何 delta
        meta → (delta*) → revise → error        # 流中断，revise 携带已审计的兜底文本

    ⚠️ 关于 `SPEAK` 的位置：它**必须在文本流完之后**才能算出来（定稿要看全文），
    所以事件顺序上排在 `delta*` 之后。这不是"定稿晚于产出"—— 定稿不改动一个字，
    它是**用同一份文本**给出"按句怎么合成、按句怎么上屏"的计划；`delta` 拼接与
    `speak.text` 与 `done.text` 三者**逐字相等**（由 `_plan_streamed_reply` 保证）。
    旧行为的问题不是顺序，而是 `done.text` 会比已流出的字**短**（问句上限在音频
    已经合成之后才生效 → 声音念到一半被掐断）。

    Attributes:
        META: 前处理完成，携带风险等级、对话模式与情绪概率；此时还没有任何回复文本
        SPEAK: **定稿文本的合成计划**（按句切好的 ``segments`` + 全文 ``text``）。
            语音端据此合成与上屏，保证"听到的"与"看到的"同源；**纯文字端可忽略**。
            只有低/中风险会下发；`segments.join('')` 恒等于 `text`。
        DELTA: 回复的增量片段，消费端应**追加**到当前气泡
        REVISE: 权威改稿，消费端应**整体替换**当前气泡（安全审计否掉了已生成的内容）
        DONE: 本轮结束，携带与非流式 /smart-chat 完全一致的审计结论
        ERROR: 出错，本轮作废；消费端应回退到非流式路径
    """
    META = "meta"
    SPEAK = "speak"
    DELTA = "delta"
    REVISE = "revise"
    DONE = "done"
    ERROR = "error"


@dataclass
class ChatStreamEvent:
    """文本流式对话的一次事件

    为什么流式路径必须保留 REVISE 这个事件类型：
        本项目的安全审计是**整段回复级**的判据（五轴审计要看完整回复与对话
        历史），因此不可能在逐字吐字之前就拿到审计结论。折中方案是
        「先说、后校」：增量上屏保证体感，审计一旦否掉就整体改稿。
        只有显式区分"追加"与"替换"，消费端才不会把权威改稿拼在错误文本后面。

    Attributes:
        event: 事件类型
        text: DELTA 为增量片段、REVISE 为完整替换文本、DONE 为最终文本；其余为空
        risk_level: 本轮风险等级（META / DONE 携带）
        dialogue_mode: 对话模式 EMPATHY / SOCRATIC（META / DONE 携带）
        emotion_probs: 五维文本情绪概率（META / DONE 携带）
        streaming: META 携带 —— 本轮是否允许增量上屏。
            high / crisis 风险恒为 False：升级流程必须原子下发，
            逐字吐出的安全确认与热线会看起来像普通闲聊。
        audit_passed: DONE 携带 —— 五轴审计是否通过
        requires_escalation: DONE 携带 —— 是否需要触发升级
        fallback: DONE 携带 —— 是否走了模板兜底（未用上真实 LLM）
        error: ERROR 携带 —— 失败原因
    """
    event: ChatStreamEventType
    text: str = ""
    risk_level: str = ""
    dialogue_mode: str = ""
    emotion_probs: list[float] = field(default_factory=list)
    streaming: bool = False
    audit_passed: bool = True
    requires_escalation: bool = False
    fallback: bool = False
    error: str = ""

    @property
    def is_terminal(self) -> bool:
        """是否为终结事件（消费端收到后本轮不再期待新事件）"""
        return self.event in (ChatStreamEventType.DONE, ChatStreamEventType.ERROR)

    @property
    def replaces_bubble(self) -> bool:
        """是否为整体替换语义（消费端必须清空当前气泡再写入 ``text``）"""
        return self.event is ChatStreamEventType.REVISE
