"""
主动关怀机制单元测试

测试覆盖：
    1. 触发条件评估（基线偏离、App 未使用、历史危机）
    2. 关怀消息生成（记忆引用、长度限制）
    3. 用户授权管理（开启/关闭）
    4. 伦理边界（每天限制、危机转人工）
    5. 跨场景验证
"""
import sys
import time
from pathlib import Path

import pytest

# 添加算法路径
algorithm_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(algorithm_path))

from escalation.proactive_care import (
    ProactiveCareManager,
    TriggerCondition,
    TriggerSignal,
    CareMessage,
    UserConsent,
    reset_proactive_care_manager,
)
from assessment.user_memory import get_memory_store, reset_memory_store


# ============================================================
# 1. 触发条件评估测试
# ============================================================

class TestTriggerEvaluation:
    """测试触发条件评估"""

    def test_baseline_deviation_trigger(self):
        """基线连续偏离触发"""
        manager = reset_proactive_care_manager()
        user_id = "test_user_1"
        manager.grant_consent(user_id)

        # 模拟连续 3 天基线偏离 |Z| > 2.0
        triggers = manager.evaluate_triggers(
            user_id=user_id,
            baseline_deviations=[2.5, 2.8, 3.0],
        )

        assert len(triggers) >= 1
        assert any(t.condition == TriggerCondition.BASELINE_DEVIATION for t in triggers)

    def test_baseline_deviation_below_threshold(self):
        """基线偏离未达阈值不触发"""
        manager = reset_proactive_care_manager()
        user_id = "test_user_2"
        manager.grant_consent(user_id)

        # 模拟基线偏离 |Z| < 2.0
        triggers = manager.evaluate_triggers(
            user_id=user_id,
            baseline_deviations=[1.5, 1.8, 1.9],
        )

        assert not any(t.condition == TriggerCondition.BASELINE_DEVIATION for t in triggers)

    def test_app_inactivity_trigger(self):
        """App 长时间未使用触发"""
        manager = reset_proactive_care_manager()
        user_id = "test_user_3"
        manager.grant_consent(user_id)

        # 模拟 6 天未打开 App
        last_open = time.time() - 6 * 86400
        triggers = manager.evaluate_triggers(
            user_id=user_id,
            last_app_open=last_open,
        )

        assert len(triggers) >= 1
        assert any(t.condition == TriggerCondition.APP_INACTIVITY for t in triggers)

    def test_app_inactivity_below_threshold(self):
        """App 未使用未达阈值不触发"""
        manager = reset_proactive_care_manager()
        user_id = "test_user_4"
        manager.grant_consent(user_id)

        # 模拟 3 天未打开 App（< 5 天阈值）
        last_open = time.time() - 3 * 86400
        triggers = manager.evaluate_triggers(
            user_id=user_id,
            last_app_open=last_open,
        )

        assert not any(t.condition == TriggerCondition.APP_INACTIVITY for t in triggers)

    def test_historical_crisis_trigger(self):
        """历史危机信号触发"""
        manager = reset_proactive_care_manager()
        user_id = "test_user_5"
        manager.grant_consent(user_id)

        triggers = manager.evaluate_triggers(
            user_id=user_id,
            historical_crisis=True,
        )

        assert len(triggers) >= 1
        assert any(t.condition == TriggerCondition.HISTORICAL_CRISIS for t in triggers)


# ============================================================
# 2. 关怀消息生成测试
# ============================================================

class TestCareMessageGeneration:
    """测试关怀消息生成"""

    def test_care_message_with_memory_reference(self):
        """关怀消息引用用户记忆"""
        manager = reset_proactive_care_manager()
        reset_memory_store()
        memory_store = get_memory_store()
        user_id = "test_user_6"
        manager.grant_consent(user_id)

        # 添加用户记忆
        memory_store.add_memory(
            user_id=user_id,
            fact="在准备期末考试",
            category="academic",
            confidence=0.9,
        )

        trigger = TriggerSignal(
            condition=TriggerCondition.BASELINE_DEVIATION,
            user_id=user_id,
            severity=0.5,
        )

        care_msg = manager.generate_care_message(user_id, trigger)

        assert care_msg is not None
        assert "准备考试" in care_msg.content or "期末" in care_msg.content
        assert len(care_msg.content) < 50

    def test_care_message_without_memory(self):
        """无记忆时的通用关怀消息"""
        manager = reset_proactive_care_manager()
        reset_memory_store()
        user_id = "test_user_7"
        manager.grant_consent(user_id)

        trigger = TriggerSignal(
            condition=TriggerCondition.APP_INACTIVITY,
            user_id=user_id,
            severity=0.3,
        )

        care_msg = manager.generate_care_message(user_id, trigger)

        assert care_msg is not None
        assert len(care_msg.content) < 50
        assert "好久" in care_msg.content or "想聊" in care_msg.content

    def test_care_message_length_limit(self):
        """关怀消息长度限制 < 50 字"""
        manager = reset_proactive_care_manager()
        reset_memory_store()
        user_id = "test_user_8"
        manager.grant_consent(user_id)

        trigger = TriggerSignal(
            condition=TriggerCondition.BASELINE_DEVIATION,
            user_id=user_id,
            severity=0.5,
        )

        care_msg = manager.generate_care_message(user_id, trigger)

        assert care_msg is not None
        assert len(care_msg.content) < 50


# ============================================================
# 3. 用户授权管理测试
# ============================================================

class TestConsentManagement:
    """测试用户授权管理"""

    def test_grant_consent(self):
        """用户开启主动关怀"""
        manager = reset_proactive_care_manager()
        user_id = "test_user_9"

        manager.grant_consent(user_id)

        assert manager.has_consent(user_id) is True

    def test_revoke_consent(self):
        """用户关闭主动关怀"""
        manager = reset_proactive_care_manager()
        user_id = "test_user_10"

        manager.grant_consent(user_id)
        assert manager.has_consent(user_id) is True

        manager.revoke_consent(user_id)
        assert manager.has_consent(user_id) is False

    def test_no_consent_no_trigger(self):
        """未授权不触发"""
        manager = reset_proactive_care_manager()
        user_id = "test_user_11"

        # 不授权，直接评估
        triggers = manager.evaluate_triggers(
            user_id=user_id,
            baseline_deviations=[2.5, 2.8, 3.0],
        )

        assert len(triggers) == 0

    def test_consent_revoked_no_trigger(self):
        """撤销授权后不触发"""
        manager = reset_proactive_care_manager()
        user_id = "test_user_12"

        manager.grant_consent(user_id)
        manager.revoke_consent(user_id)

        triggers = manager.evaluate_triggers(
            user_id=user_id,
            baseline_deviations=[2.5, 2.8, 3.0],
        )

        assert len(triggers) == 0


# ============================================================
# 4. 伦理边界测试
# ============================================================

class TestEthicalBoundaries:
    """测试伦理边界"""

    def test_daily_care_limit(self):
        """每天最多 1 条关怀消息"""
        manager = reset_proactive_care_manager()
        reset_memory_store()
        user_id = "test_user_13"
        manager.grant_consent(user_id)

        trigger1 = TriggerSignal(
            condition=TriggerCondition.BASELINE_DEVIATION,
            user_id=user_id,
            severity=0.5,
        )
        trigger2 = TriggerSignal(
            condition=TriggerCondition.APP_INACTIVITY,
            user_id=user_id,
            severity=0.3,
        )

        # 第一条应该成功
        msg1 = manager.generate_care_message(user_id, trigger1)
        assert msg1 is not None

        # 第二条应该被限制
        msg2 = manager.generate_care_message(user_id, trigger2)
        assert msg2 is None

    def test_daily_evaluation_limit(self):
        """每天最多评估一次"""
        manager = reset_proactive_care_manager()
        user_id = "test_user_14"
        manager.grant_consent(user_id)

        # 第一次评估
        triggers1 = manager.evaluate_triggers(
            user_id=user_id,
            baseline_deviations=[2.5, 2.8, 3.0],
        )
        assert len(triggers1) >= 1

        # 第二次评估（同一天）应该返回空
        triggers2 = manager.evaluate_triggers(
            user_id=user_id,
            baseline_deviations=[2.5, 2.8, 3.0],
        )
        assert len(triggers2) == 0

    def test_crisis_should_escalate_to_human(self):
        """危机信号必须转人工"""
        manager = reset_proactive_care_manager()
        user_id = "test_user_15"

        trigger = TriggerSignal(
            condition=TriggerCondition.HISTORICAL_CRISIS,
            user_id=user_id,
            severity=0.8,
        )

        assert manager.should_escalate_to_human(trigger) is True

    def test_severe_baseline_should_escalate(self):
        """严重基线偏离建议转人工"""
        manager = reset_proactive_care_manager()
        user_id = "test_user_16"

        trigger = TriggerSignal(
            condition=TriggerCondition.BASELINE_DEVIATION,
            user_id=user_id,
            severity=0.8,  # > 0.7
        )

        assert manager.should_escalate_to_human(trigger) is True

    def test_mild_baseline_no_escalation(self):
        """轻度基线偏离不转人工"""
        manager = reset_proactive_care_manager()
        user_id = "test_user_17"

        trigger = TriggerSignal(
            condition=TriggerCondition.BASELINE_DEVIATION,
            user_id=user_id,
            severity=0.5,  # < 0.7
        )

        assert manager.should_escalate_to_human(trigger) is False


# ============================================================
# 5. 跨场景验证
# ============================================================

class TestCrossScenario:
    """跨场景验证"""

    def test_full_scenario_with_consent_and_memory(self):
        """完整场景：授权 + 记忆 + 触发"""
        manager = reset_proactive_care_manager()
        reset_memory_store()
        memory_store = get_memory_store()
        user_id = "test_user_18"

        # 1. 用户授权
        manager.grant_consent(user_id)
        assert manager.has_consent(user_id) is True

        # 2. 添加记忆
        memory_store.add_memory(
            user_id=user_id,
            fact="最近和朋友闹矛盾",
            category="social",
            confidence=0.85,
        )

        # 3. 评估触发
        triggers = manager.evaluate_triggers(
            user_id=user_id,
            baseline_deviations=[2.5, 2.8, 3.0],
        )
        assert len(triggers) >= 1

        # 4. 生成关怀消息
        care_msg = manager.generate_care_message(user_id, triggers[0])
        assert care_msg is not None
        assert "朋友" in care_msg.content or "矛盾" in care_msg.content

    def test_consent_revoked_scenario(self):
        """用户关闭关怀后不触发"""
        manager = reset_proactive_care_manager()
        user_id = "test_user_19"

        # 1. 先授权
        manager.grant_consent(user_id)

        # 2. 再关闭
        manager.revoke_consent(user_id)

        # 3. 评估触发应该返回空
        triggers = manager.evaluate_triggers(
            user_id=user_id,
            baseline_deviations=[2.5, 2.8, 3.0],
        )
        assert len(triggers) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
