"""
危机干预子模块

核心功能：
    作为对话状态机 CRISIS 状态的专用处理器，整合：
    1. 自伤/自杀风险评估（对接 escalation/self_harm）
    2. 安全计划生成（Safety Plan）
    3. 紧急联系人通知
    4. 危机资源推送
    5. 危机后跟进流程

设计原则：
    1. 安全第一：任何不确定都按高风险处理
    2. 非评判性：温暖、接纳、不说教
    3. 具体行动：给出具体的求助步骤，而非泛泛安慰
    4. 持续监控：危机解除后仍保持关注

⚠️ 比赛 Demo 为模拟实现，生产环境需接入真实危机干预流程
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ============================================================
# 数据结构
# ============================================================

class CrisisLevel(str, Enum):
    """危机等级"""
    NONE = "none"              # 无危机信号
    MONITOR = "monitor"        # 需持续监控（轻度消极）
    ACTIVE = "active"          # 主动干预（明显消极/自伤暗示）
    IMMEDIATE = "immediate"    # 紧急干预（自伤/自杀明确意图）


@dataclass
class CrisisAssessment:
    """危机评估结果"""
    crisis_level: CrisisLevel
    self_harm_risk: float          # 自伤风险 0-1
    suicide_risk: float            # 自杀风险 0-1
    detected_signals: list[str]    # 检测到的危机信号
    recommended_action: str        # 建议动作
    timestamp: float = field(default_factory=time.time)


@dataclass
class SafetyPlan:
    """安全计划"""
    plan_id: str
    user_id: str
    created_at: float = field(default_factory=time.time)
    # 安全计划步骤
    warning_signs: list[str] = field(default_factory=list)     # 预警信号
    coping_strategies: list[str] = field(default_factory=list) # 应对策略
    social_contacts: list[str] = field(default_factory=list)   # 社交支持
    professionals: list[str] = field(default_factory=list)     # 专业人员
    safe_environment: bool = False                              # 环境是否安全
    reasons_for_living: list[str] = field(default_factory=list) # 活下去的理由


@dataclass
class CrisisInterventionResult:
    """危机干预结果"""
    intervention_id: str
    user_id: str
    crisis_level: CrisisLevel
    assessment: CrisisAssessment
    safety_plan: Optional[SafetyPlan] = None
    response_text: str = ""          # 给用户的回复
    resources_provided: list[str] = field(default_factory=list)  # 推送的资源
    escalation_triggered: bool = False  # 是否触发升级
    follow_up_needed: bool = False      # 是否需要后续跟进
    timestamp: float = field(default_factory=time.time)


# ============================================================
# 危机话术库
# ============================================================

CRISIS_RESPONSES: dict[str, list[str]] = {
    CrisisLevel.MONITOR: [
        "我注意到你最近好像不太开心。你愿意跟我多聊聊吗？我一直在这里听你说。",
        "你最近的状态让我有些在意。不需要一次说完，慢慢来，什么时候想聊都可以。",
        "听起来你正在经历一段不容易的时光。你不需要独自面对，我们可以一起想办法。",
    ],
    CrisisLevel.ACTIVE: [
        "你说的话让我很在意。我想直接问你：你有没有想过伤害自己？这个问题很重要，你的回答会帮助我知道怎么更好地支持你。",
        "我能感受到你现在非常痛苦。谢谢你愿意告诉我这些。你现在安全吗？身边有人陪着你吗？",
        "你描述的这些感受，说明你现在承受了很多。我想帮你找到一些支持。你愿意试试拨打24小时心理援助热线吗？400-161-9995",
    ],
    CrisisLevel.IMMEDIATE: [
        "你现在的感受让我非常担心。你的安全是最重要的。请你现在立刻做以下任何一件事：\n1. 拨打 120 或 110\n2. 拨打24小时心理援助热线：400-161-9995\n3. 告诉身边的大人\n我会一直在这里陪着你。",
        "谢谢你告诉我这些，这需要很大的勇气。你现在有伤害自己的计划吗？如果有，请立刻拨打 120 或告诉身边的人。你的生命很重要，请给自己一个机会。",
    ],
}

# 危机资源列表
CRISIS_RESOURCES = [
    {"name": "24小时心理援助热线", "phone": "400-161-9995", "type": "hotline"},
    {"name": "12355青少年服务热线", "phone": "12355", "type": "hotline"},
    {"name": "生命热线", "phone": "400-821-1215", "type": "hotline"},
    {"name": "希望24热线", "phone": "400-161-9995", "type": "hotline"},
    {"name": "北京心理危机研究与干预中心", "phone": "010-82951332", "type": "center"},
]

# 应对策略库
COPING_STRATEGIES = [
    "深呼吸练习：吸气4秒，屏气4秒，呼气6秒，重复5次",
    "蝴蝶拍：双手交叉放在胸前，左右交替轻拍",
    "5-4-3-2-1 着陆练习：说出5样看到的东西、4样能触摸的、3样能听到的、2样能闻到的、1样能尝到的",
    "冰块握持：握住一块冰块，感受温度，帮助回到当下",
    "安全空间想象：闭上眼睛，想象一个让你感到安全的地方",
]


# ============================================================
# 危机干预处理器
# ============================================================

class CrisisInterventionHandler:
    """危机干预处理器

    整合自伤评估、安全计划、资源推送、升级通知的完整流程：
    1. 接收文本输入 → 评估危机等级
    2. 根据等级选择响应策略
    3. 生成安全计划（中高风险）
    4. 推送危机资源
    5. 触发升级通知（高风险）
    6. 安排后续跟进
    """

    def __init__(self):
        self._interventions: list[CrisisInterventionResult] = []
        self._active_sessions: dict[str, dict] = {}

    def assess_crisis(self, text: str, emotion_probs: Optional[list[float]] = None) -> CrisisAssessment:
        """评估危机等级

        Args:
            text: 用户文本
            emotion_probs: 情绪概率分布 [快乐, 悲伤, 愤怒, 恐惧, 焦虑]

        Returns:
            CrisisAssessment: 评估结果
        """
        signals = []
        self_harm_risk = 0.0
        suicide_risk = 0.0

        # 1. 关键词检测
        direct_keywords = [
            "不想活", "想死", "自杀", "自残", "自伤", "割腕",
            "跳楼", "去死", "活着没意思", "没有意义",
            "结束一切", "消失", "不存在",
        ]
        indirect_keywords = [
            "累了", "不想醒来", "如果没有我", "别人会更好",
            "负担", "拖累", "没人会在意", "无所谓了",
        ]

        text_lower = text.lower()
        for kw in direct_keywords:
            if kw in text_lower:
                signals.append(f"直接自伤关键词: {kw}")
                self_harm_risk += 0.3
                suicide_risk += 0.25

        for kw in indirect_keywords:
            if kw in text_lower:
                signals.append(f"间接消极信号: {kw}")
                self_harm_risk += 0.1
                suicide_risk += 0.08

        # 2. 情绪概率评估
        if emotion_probs and len(emotion_probs) >= 5:
            sadness = emotion_probs[1]   # 悲伤
            fear = emotion_probs[3]      # 恐惧
            anxiety = emotion_probs[4]   # 焦虑

            if sadness > 0.6:
                signals.append(f"悲伤概率过高: {sadness:.2f}")
                self_harm_risk += 0.15
            if fear > 0.5:
                signals.append(f"恐惧概率过高: {fear:.2f}")
                self_harm_risk += 0.1
            if anxiety > 0.6:
                signals.append(f"焦虑概率过高: {anxiety:.2f}")
                self_harm_risk += 0.1

        # 3. 限制范围
        self_harm_risk = min(self_harm_risk, 1.0)
        suicide_risk = min(suicide_risk, 1.0)

        # 4. 确定危机等级
        combined_risk = max(self_harm_risk, suicide_risk)
        if combined_risk >= 0.6:
            level = CrisisLevel.IMMEDIATE
            action = "立即触发紧急干预，通知咨询师和紧急联系人"
        elif combined_risk >= 0.3:
            level = CrisisLevel.ACTIVE
            action = "主动干预，推送安全计划和危机资源"
        elif combined_risk >= 0.1 or len(signals) > 0:
            level = CrisisLevel.MONITOR
            action = "持续监控，表达关心"
        else:
            level = CrisisLevel.NONE
            action = "无需干预"

        return CrisisAssessment(
            crisis_level=level,
            self_harm_risk=round(self_harm_risk, 4),
            suicide_risk=round(suicide_risk, 4),
            detected_signals=signals,
            recommended_action=action,
        )

    def intervene(
        self,
        user_id: str,
        text: str,
        emotion_probs: Optional[list[float]] = None,
    ) -> CrisisInterventionResult:
        """执行危机干预

        完整流程：评估 → 响应 → 安全计划 → 资源推送 → 升级

        Args:
            user_id: 用户 ID
            text: 用户文本
            emotion_probs: 情绪概率分布

        Returns:
            CrisisInterventionResult: 干预结果
        """
        intervention_id = f"ci_{uuid.uuid4().hex[:12]}"

        # 1. 评估危机等级
        assessment = self.assess_crisis(text, emotion_probs)

        # 2. 生成响应文本
        response = self._generate_response(assessment)

        # 3. 推送危机资源
        resources = self._select_resources(assessment)

        # 4. 生成安全计划（中高风险）
        safety_plan = None
        if assessment.crisis_level in (CrisisLevel.ACTIVE, CrisisLevel.IMMEDIATE):
            safety_plan = self._create_safety_plan(user_id, assessment)

        # 5. 判断是否需要升级
        escalation = assessment.crisis_level == CrisisLevel.IMMEDIATE

        # 6. 判断是否需要跟进
        follow_up = assessment.crisis_level in (CrisisLevel.MONITOR, CrisisLevel.ACTIVE)

        result = CrisisInterventionResult(
            intervention_id=intervention_id,
            user_id=user_id,
            crisis_level=assessment.crisis_level,
            assessment=assessment,
            safety_plan=safety_plan,
            response_text=response,
            resources_provided=[r["name"] for r in resources],
            escalation_triggered=escalation,
            follow_up_needed=follow_up,
        )

        self._interventions.append(result)

        # 记录活跃会话
        if assessment.crisis_level != CrisisLevel.NONE:
            self._active_sessions[user_id] = {
                "intervention_id": intervention_id,
                "crisis_level": assessment.crisis_level.value,
                "started_at": time.time(),
                "follow_up_count": 0,
            }

        return result

    def follow_up(self, user_id: str, check_in_text: str = "") -> str:
        """危机后跟进

        Args:
            user_id: 用户 ID
            check_in_text: 用户签到文本

        Returns:
            str: 跟进回复
        """
        session = self._active_sessions.get(user_id)
        if not session:
            return "我注意到你最近状态不太好。想聊聊吗？我一直在这里。"

        session["follow_up_count"] += 1

        # 根据跟进次数选择不同的回复
        if session["follow_up_count"] == 1:
            return "嗨，我想看看你今天怎么样。上次聊的事情，你现在感觉好一些了吗？"
        elif session["follow_up_count"] <= 3:
            return "又见面了。你最近的状态有变化吗？好的坏的都可以跟我说。"
        else:
            return "很高兴又看到你。你一直在坚持，这很了不起。今天有什么想聊的吗？"

    def resolve_crisis(self, user_id: str) -> dict:
        """解除危机状态

        Args:
            user_id: 用户 ID

        Returns:
            dict: 解除信息
        """
        session = self._active_sessions.pop(user_id, None)
        if not session:
            return {"status": "no_active_session", "message": "无活跃危机会话"}

        return {
            "status": "resolved",
            "user_id": user_id,
            "intervention_id": session["intervention_id"],
            "duration_minutes": round((time.time() - session["started_at"]) / 60, 1),
            "follow_up_count": session["follow_up_count"],
            "message": "危机状态已解除，建议持续关注。",
        }

    def get_stats(self) -> dict:
        """获取危机干预统计"""
        total = len(self._interventions)
        if total == 0:
            return {"total_interventions": 0, "active_sessions": len(self._active_sessions)}

        level_counts = {}
        for level in CrisisLevel:
            level_counts[level.value] = sum(
                1 for i in self._interventions if i.crisis_level == level
            )

        return {
            "total_interventions": total,
            "active_sessions": len(self._active_sessions),
            "level_distribution": level_counts,
            "escalation_count": sum(1 for i in self._interventions if i.escalation_triggered),
            "avg_risk_score": round(
                sum(max(i.assessment.self_harm_risk, i.assessment.suicide_risk)
                    for i in self._interventions) / total, 4
            ),
        }

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------

    def _generate_response(self, assessment: CrisisAssessment) -> str:
        """根据危机等级生成响应文本"""
        level = assessment.crisis_level
        templates = CRISIS_RESPONSES.get(level, [])
        if not templates:
            return "我注意到你好像有些不太开心。想聊聊吗？"
        return random.choice(templates)

    def _select_resources(self, assessment: CrisisAssessment) -> list[dict]:
        """选择推送的危机资源"""
        if assessment.crisis_level == CrisisLevel.NONE:
            return []
        elif assessment.crisis_level == CrisisLevel.MONITOR:
            return CRISIS_RESOURCES[:2]  # 推送前 2 个热线
        else:
            return CRISIS_RESOURCES  # 推送全部资源

    def _create_safety_plan(self, user_id: str, assessment: CrisisAssessment) -> SafetyPlan:
        """生成安全计划"""
        plan_id = f"sp_{uuid.uuid4().hex[:12]}"

        return SafetyPlan(
            plan_id=plan_id,
            user_id=user_id,
            warning_signs=assessment.detected_signals[:3],
            coping_strategies=random.sample(COPING_STRATEGIES, min(3, len(COPING_STRATEGIES))),
            social_contacts=["请告诉你信任的人：家人/朋友/老师"],
            professionals=["24小时心理援助热线：400-161-9995"],
            safe_environment=assessment.crisis_level != CrisisLevel.IMMEDIATE,
            reasons_for_living=[
                "你在这个世界上是独一无二的",
                "你的感受很重要，有人在乎你",
                "困难是暂时的，你比想象中更强大",
            ],
        )


# ============================================================
# 便捷函数
# ============================================================

import random

_handler: Optional[CrisisInterventionHandler] = None


def get_handler() -> CrisisInterventionHandler:
    """获取全局处理器实例"""
    global _handler
    if _handler is None:
        _handler = CrisisInterventionHandler()
    return _handler


def assess_and_intervene(user_id: str, text: str, emotion_probs: Optional[list[float]] = None) -> CrisisInterventionResult:
    """便捷函数：评估并干预"""
    return get_handler().intervene(user_id, text, emotion_probs)


# ============================================================
# 演示
# ============================================================

def main():
    """演示危机干预流程"""
    print("=" * 60)
    print("危机干预演示")
    print("=" * 60)

    handler = CrisisInterventionHandler()

    test_cases = [
        ("user_001", "最近总是觉得很累，不想醒来", [0.1, 0.5, 0.1, 0.15, 0.4]),
        ("user_002", "活着没意思，想消失", [0.05, 0.7, 0.1, 0.1, 0.3]),
        ("user_003", "我不想活了，想从楼上跳下去", [0.02, 0.8, 0.05, 0.08, 0.2]),
        ("user_004", "今天天气还不错，和朋友去散步了", [0.7, 0.05, 0.02, 0.03, 0.1]),
    ]

    for user_id, text, emotions in test_cases:
        print(f"\n{'─' * 50}")
        print(f"👤 用户：{user_id}")
        print(f"💬 文本：{text}")
        print(f"📊 情绪：快乐={emotions[0]:.2f} 悲伤={emotions[1]:.2f} 愤怒={emotions[2]:.2f} 恐惧={emotions[3]:.2f} 焦虑={emotions[4]:.2f}")

        result = handler.intervene(user_id, text, emotions)

        print(f"\n🔍 危机等级：{result.crisis_level.value}")
        print(f"⚠️ 自伤风险：{result.assessment.self_harm_risk:.2f}")
        print(f"⚠️ 自杀风险：{result.assessment.suicide_risk:.2f}")
        print(f"📡 检测信号：{result.assessment.detected_signals}")
        print(f"\n💬 响应：{result.response_text}")
        print(f"📋 资源推送：{result.resources_provided}")
        print(f"🚨 升级触发：{'是' if result.escalation_triggered else '否'}")
        if result.safety_plan:
            print(f"🛡️ 安全计划：已生成 ({result.safety_plan.plan_id})")
            print(f"   应对策略：{result.safety_plan.coping_strategies[:2]}")

    # 统计
    print(f"\n{'=' * 60}")
    print(f"📊 统计：{handler.get_stats()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
