"""
Agent 编排器 —— 协调感知、评估、对话状态机与安全闭环的中枢调度模块
灵感来源：datawhalechina/resonant-soul（ResonantSoul 项目）
子组件：SessionManager / SelfAssessment / MoodDiary / AgentOrchestrator
"""
from __future__ import annotations
import logging
import random
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from shared.dataclasses import (
    EmotionResult, RiskAssessment, RiskLevel, CrisisAlert,
    EvidenceItem, AuditVerdict,
)
from intervention.state_machine import (
    DialogEngine, DialogState, DialogueMode, Action, ActionType,
)
from intervention.safety_loop import SafetyLoop, SafetyLoopResult

logger = logging.getLogger(__name__)
random.seed(42)  # 固定随机种子（项目约定）
_EMOTION_LABELS = ["快乐", "悲伤", "焦虑", "愤怒", "中性"]
_DEFAULT_WINDOW = 5


# ============================================================
# TurnResult —— 单轮对话编排输出
# ============================================================
@dataclass
class TurnResult:
    """单轮对话编排结果 —— 整合感知/评估/状态机/安全闭环的全部输出"""
    turn_id: str = ""                       # 本轮唯一标识
    user_id: str = ""                       # 用户 ID
    reply: str = ""                         # 最终回复文本
    dialog_state: str = ""                  # 当前对话状态
    dialog_mode: str = ""                   # 当前对话模式
    risk_level: str = ""                    # 本轮风险评估等级
    emotion_probs: list[float] = field(default_factory=list)  # 五维情绪概率
    dominant_emotion: str = ""              # 主导情绪标签
    action: Optional[Action] = None         # 状态机选择的动作
    audit_passed: bool = True               # 安全审计是否通过
    safety_action: str = "pass_through"     # 安全闭环采取的动作
    requires_escalation: bool = False       # 是否需要危机升级
    session_summary: str = ""               # 会话摘要（仅转介时填充）
    latency_ms: float = 0.0                 # 处理延迟（毫秒）

# ============================================================
# MoodDiary —— 情绪轨迹记录
# ============================================================
@dataclass
class MoodEntry:
    """单条心情记录（时间戳、情绪概率、主导情绪、心情评分、轮次序号）"""
    timestamp: float = 0.0
    emotion_probs: list[float] = field(default_factory=list)
    dominant_emotion: str = ""
    mood_score: float = 0.0
    turn_index: int = 0

class MoodDiary:
    """情绪轨迹记录器 —— 追踪单次会话的情绪变化弧线

    提供趋势分析（改善/稳定/恶化）和可视化数据结构。
    """

    def __init__(self) -> None:
        self._entries: list[MoodEntry] = []

    def record(self, emotion_probs: list[float], turn_index: int,
               mood_score: float = 0.0) -> MoodEntry:
        """记录一条情绪条目，若未提供自评则从概率推断"""
        dominant_idx = max(range(5), key=lambda i: emotion_probs[i])
        if mood_score == 0.0:
            happy, neg = emotion_probs[0], sum(emotion_probs[1:4])
            mood_score = round(max(1.0, min(5.0, 1.0 + happy * 4.0 - neg * 1.5)), 1)
        entry = MoodEntry(
            timestamp=time.time(), emotion_probs=list(emotion_probs),
            dominant_emotion=_EMOTION_LABELS[dominant_idx],
            mood_score=mood_score, turn_index=turn_index,
        )
        self._entries.append(entry)
        return entry
    def get_trend(self) -> str:
        """分析情绪趋势：比较前后半段平均心情分"""
        if len(self._entries) < 3:
            return "stable"
        mid = len(self._entries) // 2
        avg = lambda s: sum(e.mood_score for e in s) / len(s)
        diff = avg(self._entries[mid:]) - avg(self._entries[:mid])
        if diff > 0.3:
            return "improving"
        if diff < -0.3:
            return "worsening"
        return "stable"
    def get_arc_data(self) -> list[dict]:
        """获取情绪弧线可视化数据（供前端渲染曲线图）"""
        return [
            {"turn": e.turn_index, "mood_score": e.mood_score,
             "dominant_emotion": e.dominant_emotion,
             "emotion_probs": [round(p, 3) for p in e.emotion_probs]}
            for e in self._entries
        ]
    def get_summary(self) -> dict:
        """获取会话情绪摘要"""
        if not self._entries:
            return {"total_entries": 0, "trend": "stable", "avg_mood": 0.0}
        scores = [e.mood_score for e in self._entries]
        return {
            "total_entries": len(self._entries), "trend": self.get_trend(),
            "avg_mood": round(sum(scores) / len(scores), 2),
            "max_mood": round(max(scores), 1), "min_mood": round(min(scores), 1),
            "arc": self.get_arc_data(),
        }

# ============================================================
# SessionManager —— 会话状态管理
# ============================================================
class SessionManager:
    """会话状态管理器 —— 追踪单个用户的对话上下文

    维护情绪历史、风险历史、轮次计数和情绪趋势滚动窗口。
    """

    def __init__(self, user_id: str, window_size: int = _DEFAULT_WINDOW) -> None:
        self.user_id = user_id
        self.turn_count: int = 0
        self.emotion_history: list[EmotionResult] = []
        self.risk_history: list[RiskLevel] = []
        self._window: deque[list[float]] = deque(maxlen=window_size)
        self.mood_diary: MoodDiary = MoodDiary()
        self.dialog_engine: DialogEngine = DialogEngine(user_id)
        self.safety_loop: SafetyLoop = SafetyLoop(
            user_id=user_id, dialog_engine=self.dialog_engine,
        )
        self._start_time: float = time.time()

    def record_emotion(self, emotion: EmotionResult) -> None:
        """记录情绪感知结果并更新滚动窗口"""
        self.emotion_history.append(emotion)
        self._window.append(list(emotion.text_emotion_probs))
        self.mood_diary.record(emotion.text_emotion_probs, self.turn_count)
    def record_risk(self, risk_level: RiskLevel) -> None:
        """记录风险等级"""
        self.risk_history.append(risk_level)
    def get_emotion_trend(self) -> str:
        """获取当前情绪趋势"""
        return self.mood_diary.get_trend()
    def increment_turn(self) -> None:
        """轮次递增"""
        self.turn_count += 1
    def get_session_summary(self) -> str:
        """生成会话摘要（供咨询师转介，不含诊断性结论）"""
        diary = self.mood_diary.get_summary()
        duration_min = (time.time() - self._start_time) / 60
        risk_counts: dict[str, int] = {}
        for r in self.risk_history:
            risk_counts[r.value] = risk_counts.get(r.value, 0) + 1
        return "\n".join([
            f"会话摘要 (用户: {self.user_id})",
            f"持续时长: {duration_min:.1f} 分钟",
            f"对话轮次: {self.turn_count}",
            f"情绪趋势: {self.get_emotion_trend()}",
            f"平均心情评分: {diary.get('avg_mood', 'N/A')}",
            f"风险分布: {risk_counts}",
            f"当前对话状态: {self.dialog_engine.current_state.value}",
            f"当前对话模式: {self.dialog_engine.current_mode.value}",
        ])

# ============================================================
# SelfAssessment —— 结构化自评
# ============================================================
@dataclass
class AssessmentResult:
    """自评结果（量表名、得分、严重程度、建议、各条目得分）"""
    scale_name: str = ""
    score: float = 0.0
    max_score: int = 0
    severity: str = ""
    recommendation: str = ""
    items: list[dict] = field(default_factory=list)

class SelfAssessment:
    """结构化自评量表 —— PHQ-2 / GAD-2 / 心情签到

    临床依据：PHQ-2 (Kroenke 2003), GAD-2 (Kroenke 2007)
    注意：仅提供筛查工具，不生成诊断性结论。
    """

    PHQ2_ITEMS = ["做事时提不起劲或没有兴趣", "感到心情低落、沮丧或绝望"]
    GAD2_ITEMS = ["感觉紧张、焦虑或急切", "不能够停止或控制担忧"]

    @classmethod
    def quick_phq2(cls, answers: list[int]) -> AssessmentResult:
        """PHQ-2 快速抑郁筛查（满分 6，>= 3 建议进一步评估）"""
        if len(answers) != 2:
            raise ValueError("PHQ-2 需要恰好 2 个条目得分")
        score = sum(max(0, min(3, a)) for a in answers)
        return AssessmentResult(
            scale_name="PHQ-2", score=float(score), max_score=6,
            severity="筛查阳性" if score >= 3 else "筛查阴性",
            recommendation=("建议进一步完成 PHQ-9 完整评估或寻求专业咨询"
                            if score >= 3 else "当前状态良好，如有持续困扰建议咨询专业人士"),
            items=[{"text": cls.PHQ2_ITEMS[i], "score": max(0, min(3, a))}
                   for i, a in enumerate(answers)],
        )

    @classmethod
    def quick_gad2(cls, answers: list[int]) -> AssessmentResult:
        """GAD-2 快速焦虑筛查（满分 6，>= 3 建议进一步评估）"""
        if len(answers) != 2:
            raise ValueError("GAD-2 需要恰好 2 个条目得分")
        score = sum(max(0, min(3, a)) for a in answers)
        return AssessmentResult(
            scale_name="GAD-2", score=float(score), max_score=6,
            severity="筛查阳性" if score >= 3 else "筛查阴性",
            recommendation=("建议进一步完成 GAD-7 完整评估或寻求专业咨询"
                            if score >= 3 else "当前状态良好，如有持续困扰建议咨询专业人士"),
            items=[{"text": cls.GAD2_ITEMS[i], "score": max(0, min(3, a))}
                   for i, a in enumerate(answers)],
        )

    @classmethod
    def mood_checkin(cls, mood_score: int) -> AssessmentResult:
        """心情签到（1-5 分）"""
        score = max(1, min(5, mood_score))
        if score <= 2:
            severity, rec = "偏低", "你的心情似乎不太好，愿意多聊聊吗？"
        elif score == 3:
            severity, rec = "一般", "心情处于中等水平，有什么想说的随时告诉我"
        else:
            severity, rec = "良好", "看起来心情不错！继续保持"
        return AssessmentResult(
            scale_name="MoodCheckIn", score=float(score), max_score=5,
            severity=severity, recommendation=rec,
            items=[{"text": "你现在的心情如何？（1=很差，5=很好）", "score": score}],
        )

# ============================================================
# AgentOrchestrator —— 中枢编排器
# ============================================================
class AgentOrchestrator:
    """Agent 编排器 —— 感知 → 评估 → 状态机 → 安全闭环

    使用方式：
        orch = AgentOrchestrator()
        result = orch.process_turn("user_123", "我最近心情很差", [])
    """

    def __init__(self, perception_service: Optional[object] = None) -> None:
        self._perception = perception_service
        self._sessions: dict[str, SessionManager] = {}

    def _get_session(self, user_id: str) -> SessionManager:
        """获取或创建用户会话"""
        if user_id not in self._sessions:
            self._sessions[user_id] = SessionManager(user_id)
        return self._sessions[user_id]

    def _get_perception(self) -> object:
        """懒加载感知服务"""
        if self._perception is None:
            from perception.perception_service import get_perception_service
            self._perception = get_perception_service()
        return self._perception

    def _estimate_risk_level(self, emotion: EmotionResult, session: SessionManager) -> RiskLevel:
        """从情绪结果推断风险等级（临床分层逻辑）

        临床依据：DSM-5 严重程度分级 + C-SSRS 风险评估协议
        - 单次高情绪 ≠ 临床风险（需区分"正常困扰" vs "功能损害"）
        - 危机关键词 = 立即升级（安全第一原则）
        - 持续高风险 = 可能功能损害（需关注）

        Args:
            emotion: 情绪感知结果
            session: 会话管理器（用于检查历史风险）

        Returns:
            RiskLevel: 风险等级
        """
        # ⚠️ 规则 1：危机关键词命中 → 至少 CRISIS（安全第一原则）
        # 注意：关键词命中不等于临床诊断，仅触发人工复核流程
        if emotion.crisis_keywords:
            return RiskLevel.CRISIS

        # 计算负面情绪强度
        neg = emotion.text_emotion_probs[1] + emotion.text_emotion_probs[2] + emotion.text_emotion_probs[3]

        # 统计近期高风险轮次（检查功能损害模式）
        recent_high_count = sum(
            1 for r in session.risk_history[-3:]
            if r in (RiskLevel.HIGH, RiskLevel.CRISIS)
        )

        # 规则 2：连续 3 轮高风险 → HIGH（可能存在功能损害）
        if recent_high_count >= 3:
            return RiskLevel.HIGH

        # 规则 3：单次情绪评估（区分正常困扰 vs 临床风险）
        # - neg > 0.75 且无历史高风险 → MEDIUM（强烈但可能是正常困扰）
        # - neg > 0.50 → MEDIUM（中等困扰）
        # - neg <= 0.50 → LOW（轻度或无困扰）
        if neg > 0.75:
            # 单次高情绪：可能是"老师骂我"这类正常困扰，给 MEDIUM 而非 HIGH
            return RiskLevel.MEDIUM
        elif neg > 0.50:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    @staticmethod
    def _action_to_reply(action: Action, emotion: EmotionResult, crisis_turns: int = 0, turn_count: int = 0) -> str:
        """将状态机动作转换为回复提示（实际话术由 LLM 生成）

        Args:
            action: 状态机选择的动作
            emotion: 情绪感知结果
            crisis_turns: 危机状态持续轮次（用于 C-SSRS 阶梯式响应，0=刚进入危机）
            turn_count: 对话总轮次（用于 INIT 阶段治疗同盟建立）
        """
        dom_idx = max(range(5), key=lambda i: emotion.text_emotion_probs[i])
        dom = _EMOTION_LABELS[dom_idx]

        # INIT 阶段治疗同盟建立（前 1-2 轮）
        if action.state == DialogState.INIT:
            if turn_count == 0:  # 第一轮
                return "（建立治疗同盟：表达欢迎、说明保密原则、创造安全空间）"
            else:  # 第二轮
                return "（继续建立信任：肯定用户愿意分享、邀请用户决定聊什么）"

        # C-SSRS 阶梯式危机响应
        if action.action_type == ActionType.SAFETY_CHECK:
            if crisis_turns == 0:  # 刚进入危机，第一轮
                return "（接纳用户的痛苦，探索这种感觉的时间线和触发因素）"
            elif crisis_turns == 1:  # 第二轮
                return "（直接询问自伤/自杀想法，评估严重程度）"
            else:  # 第三轮及以后
                return "（提供24小时心理援助热线：400-161-9995，建议联系信任的人或就医）"

        reply_map = {
            ActionType.OPEN_QUESTION: f"（以共情方式邀请用户继续分享，关注「{dom}」情绪）",
            ActionType.EMOTION_REFLECTION: f"（反映用户当前的「{dom}」情绪，传递理解与接纳）",
            ActionType.CBT_GUIDE: "（引导用户进行认知重构，探索替代性思维）",
            ActionType.MINDFULNESS_GUIDE: "（引导用户进行正念放松练习）",
            ActionType.RESOURCE_PROVIDE: "（推荐专业心理援助资源和热线信息）",
            ActionType.SUMMARY: "（回顾本次对话要点，总结应对策略）",
        }
        return reply_map.get(action.action_type, "（以温和方式回应用户）")

    def _extract_user_keywords(self, text: str) -> list[str]:
        """从用户输入中提取关键实体（人物、地点、事件）用于回复针对性

        提取策略：
        1. 人物关系词（老师、爸妈、同学、朋友等）
        2. 地点词（学校、家、教室等）
        3. 事件词（吵架、骂、批评、考试等）

        Args:
            text: 用户输入文本

        Returns:
            list[str]: 提取的关键词列表（去重，最多3个）
        """
        # 人物关系词
        person_keywords = [
            "老师", "班主任", "同学", "朋友", "爸妈", "爸爸", "妈妈",
            "父亲", "母亲", "哥哥", "姐姐", "弟弟", "妹妹", "男朋友",
            "女朋友", "喜欢的人", "暗恋", "喜欢的人",
        ]

        # 地点词
        place_keywords = [
            "学校", "教室", "家里", "宿舍", "操场", "食堂",
        ]

        # 事件词
        event_keywords = [
            "吵架", "骂", "批评", "打", "欺负", "孤立", "排挤",
            "考试", "成绩", "分数", "作业", "迟到", "旷课",
        ]

        extracted = []

        # 提取人物
        for kw in person_keywords:
            if kw in text:
                extracted.append(kw)

        # 提取地点
        for kw in place_keywords:
            if kw in text:
                extracted.append(kw)

        # 提取事件
        for kw in event_keywords:
            if kw in text:
                extracted.append(kw)

        # 去重并限制数量
        return list(dict.fromkeys(extracted))[:3]

    def _enhance_reply_with_keywords(self, reply: str, keywords: list[str]) -> str:
        """在回复中回显用户关键词，增加针对性

        策略：
        1. 如果回复是泛化模板（"这些"、"这件事"），用具体关键词替换
        2. 在回复开头或中间插入关键词引用

        Args:
            reply: 原始回复文本
            keywords: 提取的关键词列表

        Returns:
            str: 增强后的回复文本
        """
        if not keywords:
            return reply

        # 策略1：替换泛化表达
        enhanced = reply
        for kw in keywords[:2]:  # 最多用2个关键词
            # 替换"这些"为具体关键词
            if "这些" in enhanced and kw not in enhanced:
                enhanced = enhanced.replace("这些", f"关于{kw}的事", 1)
                break
            # 替换"这件事"为具体关键词
            if "这件事" in enhanced and kw not in enhanced:
                enhanced = enhanced.replace("这件事", f"{kw}这件事", 1)
                break
            # 替换"你说的"为具体关键词
            if "你说的" in enhanced and kw not in enhanced:
                enhanced = enhanced.replace("你说的", f"你提到的{kw}", 1)
                break

        # 策略2：如果回复是占位符格式，直接插入关键词
        if enhanced.startswith("（") and keywords:
            # 在占位符描述的开头插入关键词引用
            kw_str = "、".join(keywords[:2])
            enhanced = enhanced.replace("（", f"（关于{kw_str}，", 1)

        return enhanced

    def process_turn(
        self, user_id: str, message: str, conversation_history: list[dict],
    ) -> TurnResult:
        """处理一轮完整对话：感知 → 风险评估 → 状态机 → 安全闭环

        Args:
            user_id: 用户 ID
            message: 用户当前输入文本
            conversation_history: 历史对话记录

        Returns:
            TurnResult: 本轮编排结果
        """
        start_time = time.time()
        session = self._get_session(user_id)
        turn_id = f"turn_{uuid.uuid4().hex[:8]}"

        # 步骤 1：感知层 —— 情绪分析
        perception = self._get_perception()
        emotion: EmotionResult = perception.analyze_text(message)  # type: ignore[attr-defined]
        session.record_emotion(emotion)

        # 步骤 2：评估层 —— 风险等级推断（需传入 session 以检查历史风险模式）
        risk_level = self._estimate_risk_level(emotion, session)
        session.record_risk(risk_level)

        # 步骤 3：状态机 —— 对话动作选择
        engine = session.dialog_engine
        emotion_trend = session.get_emotion_trend()
        has_strong_neg = sum(emotion.text_emotion_probs[1:4]) > 0.4 * 3
        engine.update_emotion_state(has_strong_neg)
        engine.evaluate_and_update_mode(message, risk_level)
        current_turn = engine.turn_count  # 保存当前轮次（next_turn 会递增）
        action = engine.next_turn(risk_level, emotion_trend)

        # 步骤 4：构建初始回复提示（传入 crisis_turns 和 turn_count 以支持阶梯式响应和治疗同盟）
        initial_reply = self._action_to_reply(action, emotion, engine.crisis_turns, current_turn)

        # 步骤 4.5：关键词回显增强 —— 提取用户提到的具体人物/事件，在回复中点名
        user_keywords = self._extract_user_keywords(message)
        initial_reply = self._enhance_reply_with_keywords(initial_reply, user_keywords)

        # 步骤 5：安全闭环审计
        safety_result: SafetyLoopResult = session.safety_loop.process(
            user_input=message,
            llm_response=initial_reply,
            context={
                "dialog_state": engine.current_state.value,
                "turn_count": engine.turn_count,
                "risk_level": risk_level.value,
                "conversation_history": conversation_history,
                "dialogue_mode": engine.current_mode.value,
                "consecutive_calm_turns": engine.consecutive_calm_turns,
            },
        )
        session.increment_turn()

        # ⚠️ 危机升级警告：requires_escalation 为 True 时必须触发升级层
        requires_escalation = (
            action.requires_escalation
            or safety_result.escalation_status == "dispatched"
        )
        session_summary = session.get_session_summary() if requires_escalation else ""
        dominant_idx = max(range(5), key=lambda i: emotion.text_emotion_probs[i])

        return TurnResult(
            turn_id=turn_id, user_id=user_id,
            reply=safety_result.final_response,
            dialog_state=engine.current_state.value,
            dialog_mode=engine.current_mode.value,
            risk_level=risk_level.value,
            emotion_probs=[round(p, 4) for p in emotion.text_emotion_probs],
            dominant_emotion=_EMOTION_LABELS[dominant_idx],
            action=action,
            audit_passed=(safety_result.audit_verdict.passed
                          if safety_result.audit_verdict else True),
            safety_action=safety_result.action_taken,
            requires_escalation=requires_escalation,
            session_summary=session_summary,
            latency_ms=round((time.time() - start_time) * 1000, 2),
        )

    def get_session_summary(self, user_id: str) -> str:
        """获取用户会话摘要（供咨询师转介）"""
        return self._get_session(user_id).get_session_summary()

    def get_mood_diary(self, user_id: str) -> dict:
        """获取用户情绪轨迹摘要"""
        return self._get_session(user_id).mood_diary.get_summary()

    def run_self_assessment(
        self, scale: str,
        answers: Optional[list[int]] = None,
        mood_score: Optional[int] = None,
    ) -> AssessmentResult:
        """运行自评量表（phq2 / gad2 / mood）"""
        if scale == "phq2":
            return SelfAssessment.quick_phq2(answers or [0, 0])
        if scale == "gad2":
            return SelfAssessment.quick_gad2(answers or [0, 0])
        if scale == "mood":
            return SelfAssessment.mood_checkin(mood_score or 3)
        raise ValueError(f"不支持的量表: {scale}")
