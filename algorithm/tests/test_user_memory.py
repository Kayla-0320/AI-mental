"""
用户长期记忆系统单元测试

测试覆盖：
    1. 记忆存储（add/delete/get）
    2. 内容安全检查（拒绝危机/诊断/情绪信息）
    3. 记忆提取（规则提取 + 分类）
    4. 记忆召回（相关性排序 + 引用更新）
    5. 记忆降权（超期未引用）
    6. 跨会话记忆场景验证
"""
import json
import os
import sys
import time
import pytest
import tempfile
from pathlib import Path

# 添加算法路径
algorithm_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(algorithm_path))

from assessment.user_memory import (
    UserMemory,
    MemoryCategory,
    MemoryStore,
    is_memory_safe,
    get_memory_store,
    reset_memory_store,
)
from assessment.memory_extractor import (
    MemoryExtractor,
    ExtractedFact,
    build_memory_context,
)


# ============================================================
# 1. 记忆存储测试
# ============================================================

class TestMemoryStore:
    """测试 MemoryStore 基本操作"""

    def setup_method(self):
        """每个测试前创建临时存储"""
        self.temp_file = tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        )
        self.temp_file.close()
        self.store = MemoryStore(storage_path=self.temp_file.name)

    def teardown_method(self):
        """每个测试后清理临时文件"""
        try:
            os.unlink(self.temp_file.name)
        except OSError:
            pass

    def test_add_memory(self):
        """添加记忆"""
        mem = self.store.add_memory("user_1", "正在准备中考", "academic", 0.9)
        assert mem is not None
        assert mem.user_id == "user_1"
        assert mem.fact == "正在准备中考"
        assert mem.category == "academic"
        assert mem.confidence == 0.9

    def test_add_memory_rejects_unsafe(self):
        """添加不安全记忆被拒绝"""
        mem = self.store.add_memory("user_1", "我想自杀", "other", 0.9)
        assert mem is None

    def test_add_memory_rejects_diagnosis(self):
        """添加诊断信息被拒绝"""
        mem = self.store.add_memory("user_1", "我确诊了抑郁症", "health", 0.9)
        assert mem is None

    def test_get_all_memories(self):
        """获取用户所有记忆"""
        self.store.add_memory("user_1", "正在准备中考", "academic")
        self.store.add_memory("user_1", "喜欢打篮球", "interest")
        self.store.add_memory("user_2", "在读高一", "academic")

        memories = self.store.get_all_memories("user_1")
        assert len(memories) == 2

    def test_delete_memory(self):
        """删除单条记忆"""
        mem = self.store.add_memory("user_1", "正在准备中考", "academic")
        assert self.store.delete_memory("user_1", mem.id) is True
        assert len(self.store.get_all_memories("user_1")) == 0

    def test_delete_memory_wrong_user(self):
        """不能删除别人的记忆"""
        mem = self.store.add_memory("user_1", "正在准备中考", "academic")
        assert self.store.delete_memory("user_2", mem.id) is False
        assert len(self.store.get_all_memories("user_1")) == 1

    def test_clear_user_memories(self):
        """清空用户所有记忆"""
        self.store.add_memory("user_1", "正在准备中考", "academic")
        self.store.add_memory("user_1", "喜欢打篮球", "interest")
        count = self.store.clear_user_memories("user_1")
        assert count == 2
        assert len(self.store.get_all_memories("user_1")) == 0

    def test_persistence(self):
        """记忆持久化到 JSON 文件"""
        self.store.add_memory("user_1", "正在准备中考", "academic")

        # 重新加载
        store2 = MemoryStore(storage_path=self.temp_file.name)
        memories = store2.get_all_memories("user_1")
        assert len(memories) == 1
        assert memories[0].fact == "正在准备中考"


# ============================================================
# 2. 内容安全检查测试
# ============================================================

class TestMemorySafety:
    """测试内容安全检查"""

    def test_safe_fact(self):
        """安全事实通过检查"""
        assert is_memory_safe("我正在准备中考") is True
        assert is_memory_safe("我喜欢打篮球") is True
        assert is_memory_safe("我爸妈在外地工作") is True

    def test_crisis_signal_rejected(self):
        """危机信号被拒绝"""
        assert is_memory_safe("我想自杀") is False
        assert is_memory_safe("不想活了") is False
        assert is_memory_safe("死了算了") is False
        assert is_memory_safe("割腕") is False

    def test_diagnosis_rejected(self):
        """诊断信息被拒绝"""
        assert is_memory_safe("我确诊了抑郁症") is False
        assert is_memory_safe("我被诊断为焦虑症") is False
        assert is_memory_safe("患有双相情感障碍") is False

    def test_emotion_state_rejected(self):
        """情绪状态被拒绝（属于基线系统职责）"""
        assert is_memory_safe("我感到悲伤") is False
        assert is_memory_safe("感到焦虑") is False


# ============================================================
# 3. 记忆提取测试
# ============================================================

class TestMemoryExtractor:
    """测试记忆提取器"""

    def test_extract_academic_fact(self):
        """提取学业事实"""
        extractor = MemoryExtractor()
        facts = extractor.extract_from_text("我在准备中考，最近压力很大")
        academic_facts = [f for f in facts if f.category == "academic"]
        assert len(academic_facts) >= 1

    def test_extract_family_fact(self):
        """提取家庭事实"""
        extractor = MemoryExtractor()
        facts = extractor.extract_from_text("我妈妈是老师，爸爸在外地工作")
        family_facts = [f for f in facts if f.category == "family"]
        assert len(family_facts) >= 1

    def test_extract_interest_fact(self):
        """提取兴趣事实"""
        extractor = MemoryExtractor()
        facts = extractor.extract_from_text("我平时喜欢打篮球和听音乐")
        interest_facts = [f for f in facts if f.category == "interest"]
        assert len(interest_facts) >= 1

    def test_extract_social_fact(self):
        """提取社交事实"""
        extractor = MemoryExtractor()
        facts = extractor.extract_from_text("我有一个好朋友叫小明，我们经常一起打球")
        social_facts = [f for f in facts if f.category == "social"]
        assert len(social_facts) >= 1

    def test_extract_from_conversation(self):
        """从对话历史中提取"""
        extractor = MemoryExtractor()
        history = [
            {"role": "user", "content": "我在读初三，准备中考"},
            {"role": "assistant", "content": "中考确实是个重要的阶段..."},
            {"role": "user", "content": "我妈妈是老师，她对我期望很高"},
        ]
        facts = extractor.extract_from_conversation(history)
        assert len(facts) >= 1

    def test_no_extraction_from_empty(self):
        """空文本不提取"""
        extractor = MemoryExtractor()
        facts = extractor.extract_from_text("")
        assert len(facts) == 0

    def test_no_extraction_from_short(self):
        """过短文本不提取"""
        extractor = MemoryExtractor()
        facts = extractor.extract_from_text("嗯")
        assert len(facts) == 0

    def test_extract_and_store(self):
        """提取并存储"""
        temp_file = tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        )
        temp_file.close()
        try:
            # 重置全局存储并使用临时文件
            reset_memory_store()
            store = MemoryStore(storage_path=temp_file.name)

            extractor = MemoryExtractor()
            stored = extractor.extract_and_store(
                user_id="user_1",
                conversation_history=[
                    {"role": "user", "content": "我在准备中考"},
                ],
                max_facts=3,
            )
            # 应该提取并存储了至少一条
            assert len(stored) >= 1
        finally:
            os.unlink(temp_file.name)
            reset_memory_store()


# ============================================================
# 4. 记忆召回测试
# ============================================================

class TestMemoryRecall:
    """测试记忆召回"""

    def setup_method(self):
        self.temp_file = tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        )
        self.temp_file.close()
        self.store = MemoryStore(storage_path=self.temp_file.name)

    def teardown_method(self):
        try:
            os.unlink(self.temp_file.name)
        except OSError:
            pass

    def test_relevant_memories_returned(self):
        """返回相关记忆"""
        self.store.add_memory("user_1", "正在准备中考", "academic")
        self.store.add_memory("user_1", "喜欢打篮球", "interest")

        memories = self.store.get_relevant_memories("user_1", "中考", top_k=5)
        assert len(memories) >= 1
        # 中考相关记忆应排在前面
        assert any("中考" in m.fact for m in memories)

    def test_top_k_respected(self):
        """top_k 限制生效"""
        for i in range(10):
            self.store.add_memory("user_1", f"记忆{i}", "other")

        memories = self.store.get_relevant_memories("user_1", "", top_k=3)
        assert len(memories) <= 3

    def test_reference_count_updated(self):
        """引用次数更新"""
        self.store.add_memory("user_1", "正在准备中考", "academic")
        memories = self.store.get_relevant_memories("user_1", "中考")
        assert memories[0].reference_count >= 1

    def test_empty_for_unknown_user(self):
        """未知用户返回空"""
        memories = self.store.get_relevant_memories("unknown_user", "中考")
        assert len(memories) == 0


# ============================================================
# 5. 记忆降权测试
# ============================================================

class TestMemoryDecay:
    """测试记忆降权"""

    def setup_method(self):
        self.temp_file = tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        )
        self.temp_file.close()
        self.store = MemoryStore(storage_path=self.temp_file.name)

    def teardown_method(self):
        try:
            os.unlink(self.temp_file.name)
        except OSError:
            pass

    def test_old_memories_decayed(self):
        """超期记忆降权"""
        mem = self.store.add_memory("user_1", "正在准备中考", "academic")
        # 手动设置 last_referenced 为 100 天前
        mem.last_referenced = time.time() - 100 * 86400
        self.store._save()

        count = self.store.decay_old_memories(days=90)
        assert count == 1
        # 置信度应降低
        memories = self.store.get_all_memories("user_1")
        assert memories[0].confidence < 0.9  # 原始 0.8 * 0.5 = 0.4

    def test_recent_memories_not_decayed(self):
        """近期记忆不降权"""
        self.store.add_memory("user_1", "正在准备中考", "academic", confidence=0.8)
        count = self.store.decay_old_memories(days=90)
        assert count == 0


# ============================================================
# 6. 记忆上下文构建测试
# ============================================================

class TestMemoryContext:
    """测试记忆上下文构建"""

    def setup_method(self):
        self.temp_file = tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        )
        self.temp_file.close()
        # 使用临时文件的全局存储
        reset_memory_store()
        self.store = MemoryStore(storage_path=self.temp_file.name)
        import assessment.user_memory as um
        um._global_store = self.store

    def teardown_method(self):
        try:
            os.unlink(self.temp_file.name)
        except OSError:
            pass
        reset_memory_store()

    def test_build_memory_context_with_memories(self):
        """有记忆时构建上下文"""
        self.store.add_memory("user_1", "正在准备中考", "academic")
        context = build_memory_context("user_1", "学习压力")
        assert "准备中考" in context
        assert "学业" in context

    def test_build_memory_context_empty(self):
        """无记忆时返回空"""
        context = build_memory_context("unknown_user", "学习")
        assert context == ""


# ============================================================
# 7. 跨会话场景验证
# ============================================================

class TestCrossSessionScenario:
    """跨会话记忆场景验证"""

    def setup_method(self):
        self.temp_file = tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        )
        self.temp_file.close()
        self.store = MemoryStore(storage_path=self.temp_file.name)
        self.extractor = MemoryExtractor()

    def teardown_method(self):
        try:
            os.unlink(self.temp_file.name)
        except OSError:
            pass

    def test_session1_store_memory(self):
        """对话1：存储记忆"""
        conversation = [
            {"role": "user", "content": "我最近在准备中考，压力好大"},
            {"role": "assistant", "content": "中考确实是个重要的阶段..."},
        ]
        facts = self.extractor.extract_from_conversation(conversation)
        for fact in facts:
            self.store.add_memory("user_1", fact.fact, fact.category)

        memories = self.store.get_all_memories("user_1")
        assert len(memories) >= 1
        assert any("中考" in m.fact for m in memories)

    def test_session2_recall_memory(self):
        """对话2：召回记忆"""
        # 先存储
        self.store.add_memory("user_1", "正在准备中考", "academic")

        # 隔天新对话
        memories = self.store.get_relevant_memories("user_1", "今天好累", top_k=5)
        assert len(memories) >= 1
        # 应该召回中考相关记忆
        assert any("中考" in m.fact for m in memories)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
