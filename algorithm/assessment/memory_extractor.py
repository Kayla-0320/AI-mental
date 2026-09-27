"""
记忆提取器 —— 从对话中提取用户长期记忆

功能：
    1. 从对话历史中提取 3-5 条关键事实
    2. 自动分类（家庭、学业、兴趣、社交、健康、其他）
    3. 内容安全检查（过滤危机信号、诊断信息、情绪状态）
    4. 置信度评估

提取规则：
    - 只提取用户主动透露的、稳定的、非敏感的事实
    - 不提取情绪状态（那是基线的职责）
    - 不提取危机信号（那是审计器的职责）
    - 不提取诊断性信息（那是咨询师的职责）

提取方式：
    - 规则提取：基于关键词模式匹配（当前实现）
    - LLM 提取：调用大语言模型提取（预留接口）

设计依据：
    - 青少年在对话中常以间接方式透露个人信息
    - 规则提取可解释、可审计，符合安全要求
    - 预留 LLM 接口便于未来升级
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from assessment.user_memory import (
    MemoryCategory,
    MemoryStore,
    is_memory_safe,
    get_memory_store,
)


# ============================================================
# 提取结果
# ============================================================

@dataclass
class ExtractedFact:
    """提取的事实

    Attributes:
        fact: 事实内容
        category: 分类
        confidence: 置信度 (0-1)
        source_text: 原始文本片段
    """
    fact: str
    category: str = "other"
    confidence: float = 0.7
    source_text: str = ""


# ============================================================
# 分类关键词模式
# ============================================================

# 每个分类对应一组正则模式，匹配用户输入中的事实性表达
_CATEGORY_PATTERNS: dict[str, list[tuple[str, str]]] = {
    # 家庭
    "family": [
        (r"(?:我|我家)(?:的)?(?:爸爸|妈妈|父亲|母亲|爹|娘).{0,20}(?:是|在|做|工作|当).{2,30}", "family"),
        (r"(?:我有|家里有).{0,5}(?:哥|姐|弟|妹).{0,20}", "family"),
        (r"(?:父母|爸妈).{0,20}(?:离婚|分开|不在|在外).{0,20}", "family"),
        (r"(?:我跟|我和).{0,10}(?:奶奶|爷爷|外婆|外公).{0,20}(?:住|一起|生活).{0,10}", "family"),
        (r"(?:家里|我家).{0,10}(?:有|没).{0,20}(?:人|兄弟姐妹).{0,10}", "family"),
    ],
    # 学业
    "academic": [
        (r"(?:我在|我就).{0,5}(?:读|上).{0,10}(?:初[一二三]|高[一二三]|中[学专]|大[学专]).{0,20}", "academic"),
        (r"(?:准备|在准备|复习).{0,10}(?:中考|高考|期末|考试|考研).{0,20}", "academic"),
        (r"(?:我.{0,5}成绩|成绩).{0,10}(?:还[行好可以]|不[太好理想]|一般).{0,20}", "academic"),
        (r"(?:我的|我).{0,5}(?:专业|方向|科目).{0,5}(?:是|选).{0,20}", "academic"),
        (r"(?:作业|课[堂业外]|补习|辅导).{0,20}(?:很|太|比较多).{0,10}(?:多|难|重).{0,10}", "academic"),
        (r"(?:老师|班主任).{0,20}(?:说|让|要求|布置).{0,20}", "academic"),
    ],
    # 兴趣爱好
    "interest": [
        (r"(?:我(?:喜欢|爱|热爱)|我平时).{0,10}(?:打|玩|听|看|画|弹|唱|跳|跑).{0,20}", "interest"),
        (r"(?:我的|我).{0,5}(?:爱好|兴趣).{0,5}(?:是|有).{0,20}", "interest"),
        (r"(?:我在|我花).{0,10}(?:课余|业余|周末|假期).{0,10}(?:会|都|一般).{0,10}.{2,20}", "interest"),
        (r"(?:最近|现在).{0,5}(?:在追|在看|在玩|在学).{0,20}", "interest"),
    ],
    # 社交
    "social": [
        (r"(?:我的|我有).{0,5}(?:好朋友|闺蜜|兄弟|朋友).{0,20}", "social"),
        (r"(?:我和|我跟).{0,10}(?:同学|同桌|室友|朋友).{0,20}", "social"),
        (r"(?:在学[校校]|班里|班上).{0,20}(?:有|没|和).{0,10}(?:人|同学|朋友).{0,20}", "social"),
        (r"(?:被|遭).{0,5}(?:欺负|排挤|孤立|霸凌).{0,20}", "social"),
        (r"(?:参加了|加入了).{0,10}(?:社团|球队|乐队|组织).{0,20}", "social"),
    ],
    # 健康
    "health": [
        (r"(?:我.{0,5})?(?:睡眠|睡觉|失眠|早睡|熬夜).{0,20}(?:不[好太好]|很[差好]|一般).{0,10}", "health"),
        (r"(?:我.{0,5})?(?:吃饭|胃口|食欲|饮食).{0,20}(?:不[好太好]|很[差好]|一般).{0,10}", "health"),
        (r"(?:我.{0,5})?(?:运动|锻炼|跑步|健身).{0,20}(?:习[惯惯]|频率|时间).{0,10}", "health"),
        (r"(?:我有|在吃).{0,10}(?:药|中药|西药).{0,20}", "health"),
    ],
}


# ============================================================
# 记忆提取器
# ============================================================

class MemoryExtractor:
    """记忆提取器

    从对话中提取用户长期记忆。

    使用方式：
        extractor = MemoryExtractor()
        facts = extractor.extract_from_text("我在准备中考，最近压力很大")
        # -> [ExtractedFact(fact="正在准备中考", category="academic", ...)]
    """

    def extract_from_text(
        self,
        text: str,
        max_facts: int = 5,
    ) -> list[ExtractedFact]:
        """从文本中提取事实

        Args:
            text: 用户输入文本
            max_facts: 最多提取条数

        Returns:
            list[ExtractedFact]: 提取的事实列表
        """
        if not text or len(text.strip()) < 5:
            return []

        facts: list[ExtractedFact] = []

        # 遍历每个分类的模式
        for category, patterns in _CATEGORY_PATTERNS.items():
            for pattern, cat in patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    # 清理提取的文本
                    fact_text = self._clean_fact(match if isinstance(match, str) else match[0])
                    if not fact_text or len(fact_text) < 4:
                        continue

                    # 安全检查
                    if not is_memory_safe(fact_text):
                        continue

                    # 去重
                    if any(f.fact == fact_text for f in facts):
                        continue

                    facts.append(ExtractedFact(
                        fact=fact_text,
                        category=cat,
                        confidence=0.7,
                        source_text=match if isinstance(match, str) else match[0],
                    ))

                    if len(facts) >= max_facts:
                        return facts

        return facts

    def extract_from_conversation(
        self,
        conversation_history: list[dict],
        max_facts: int = 5,
    ) -> list[ExtractedFact]:
        """从对话历史中提取事实

        遍历对话中的所有用户消息，提取事实。

        Args:
            conversation_history: 对话历史 [{role: "user", content: "..."}]
            max_facts: 最多提取条数

        Returns:
            list[ExtractedFact]: 提取的事实列表
        """
        all_facts: list[ExtractedFact] = []
        seen_facts: set[str] = set()

        for msg in conversation_history:
            if msg.get("role") != "user":
                continue

            content = msg.get("content", "")
            facts = self.extract_from_text(content, max_facts=max_facts)

            for fact in facts:
                if fact.fact not in seen_facts:
                    seen_facts.add(fact.fact)
                    all_facts.append(fact)

            if len(all_facts) >= max_facts:
                break

        return all_facts[:max_facts]

    def extract_and_store(
        self,
        user_id: str,
        conversation_history: list[dict],
        conversation_id: str = "",
        max_facts: int = 5,
    ) -> list[ExtractedFact]:
        """提取并存储记忆

        从对话历史中提取事实，并通过安全检查后存储。

        Args:
            user_id: 用户 ID
            conversation_history: 对话历史
            conversation_id: 对话 ID
            max_facts: 最多提取条数

        Returns:
            list[ExtractedFact]: 成功存储的事实列表
        """
        facts = self.extract_from_conversation(conversation_history, max_facts)
        store = get_memory_store()
        stored = []

        for fact in facts:
            memory = store.add_memory(
                user_id=user_id,
                fact=fact.fact,
                category=fact.category,
                confidence=fact.confidence,
                conversation_id=conversation_id,
            )
            if memory is not None:
                stored.append(fact)

        return stored

    @staticmethod
    def _clean_fact(text: str) -> str:
        """清理提取的事实文本

        去除多余空白、标点，使表达更简洁。

        Args:
            text: 原始文本

        Returns:
            str: 清理后的文本
        """
        # 去除首尾空白
        text = text.strip()
        # 去除多余空白
        text = re.sub(r'\s+', '', text)
        # 去除首尾标点
        text = text.strip("，。！？、；：""''（）")
        return text


# ============================================================
# 记忆召回 → System Prompt 注入
# ============================================================

def build_memory_context(
    user_id: str,
    current_topic: str = "",
    top_k: int = 5,
) -> str:
    """构建记忆上下文，用于注入 system prompt

    Args:
        user_id: 用户 ID
        current_topic: 当前话题
        top_k: 返回前 k 条记忆

    Returns:
        str: 格式化的记忆上下文文本
    """
    store = get_memory_store()
    memories = store.get_relevant_memories(user_id, current_topic, top_k)

    if not memories:
        return ""

    lines = ["[关于这个用户的**背景**（你以前听 ta 提过的，仅供你理解这个人）：]"]
    for mem in memories:
        category_label = _category_label(mem.category)
        lines.append(f"- ({category_label}) {mem.fact}")

    lines.append(
        "[⚠️ 这些是**过去**的背景，不代表 ta 现在遇到的事。"
        "只有在 ta **本轮自己主动提到**相关话题时，才可以顺着接一句；"
        "ta 没提，就**绝对不要**把这些背景当成 ta 刚说的话去回应、去点破、去追问，"
        "也不要说'我记得你说过'。宁可当作不知道。]"
    )

    return "\n".join(lines)


def _category_label(category: str) -> str:
    """分类标签映射"""
    labels = {
        "family": "家庭",
        "academic": "学业",
        "interest": "兴趣",
        "social": "社交",
        "health": "健康",
        "other": "其他",
    }
    return labels.get(category, "其他")
