# -*- coding: utf-8 -*-
"""送合成前的术语替换 —— 把念不出来的拉丁文换成可发音的中文

**为什么需要这个模块（实测依据）**

``vits-zh-hf-fanchen-C`` 的 ``lexicon.txt`` **不含任何 ASCII 字母**。合成时
sherpa-onnx 会打：

    character-lexicon.cc:ConvertTextToTokenIds:181 Ignore OOV 'CBT'
    character-lexicon.cc:ConvertTextToTokenIds:181 Ignore OOV '-'

然后**把整个词跳过**。于是"我们试试 CBT 里的一个方法"会被念成
"我们试试 里的一个方法" —— 少了两个字，没有异常、没有日志、用户也不知道。

对一个会大量使用 CBT / DBT 这类术语的心理学产品，这不是小事。

**边界（重要）**

这是一个**发音保真**组件，不是安全组件：

* 它只做"把 A 念成 B"的等价替换，**不改变语义、不新增/删除内容**；
* 它**不做**任何内容审查 —— 安全筛查在 ``_StreamingSafetyGate`` 与
  ``_audit_and_finalize`` 里，早于本模块，且本模块**不参与**那里的判断；
* 因此往 ``PRONOUNCE_FIX`` 里加词**不属于** AGENTS.md 的"安全规则变更"。
  但它是**用户能听见的文案**，建议仍由话术负责人过一眼。
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

#: 术语 → 可发音的中文。键必须是**纯 ASCII 字母**（含数字也可，如 ``PHQ9``），
#: 匹配时大小写不敏感、要求两侧不是字母/数字（避免 ``CBTs`` 被误替换）。
#:
#: 维护约定：只收录"产品里真的会说出来、且念不出来会丢信息"的词。
#: 不要为了让语音好听而改写句子 —— 那是话术层的职责，不是本表的。
PRONOUNCE_FIX: dict[str, str] = {
    # 疗法名（最高频，且念不出来影响最大）
    "CBT": "认知行为疗法",
    "DBT": "辩证行为疗法",
    "ACT": "接纳承诺疗法",
    "MBCT": "正念认知疗法",
    # 常用缩写
    "AI": "人工智能",
    "HRV": "心率变异性",
    "PTSD": "创伤后应激障碍",
    "ADHD": "注意缺陷多动障碍",
    "SOS": "紧急求助",
}

#: 匹配 ASCII 字母/数字串；两侧用 `[A-Za-z0-9]` 而不是 `\b` ——
#: Python 的 `\b` 以 `\w` 为界，而 `\w` **包含中文**，于是 `试试CBT里`
#: 里 `CBT` 两侧都不构成词边界，`\bCBT\b` 会漏匹配。这是本模块最容易写错的一处。
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")

#: 长键优先替换，避免短键先把长键的一部分吃掉（如 `ACT` 出现在 `MBCT` 之后）
_SORTED_KEYS: tuple[str, ...] = tuple(
    sorted(PRONOUNCE_FIX, key=len, reverse=True)
)


def _looks_pronounceable(word: str) -> bool:
    """该词是否在替换表里（大小写不敏感）"""
    return word.upper() in PRONOUNCE_FIX


def unpronounceable_spans(text: str) -> list[str]:
    """找出**替换表没覆盖**的拉丁文串（它们会被合成器静默跳过）。

    单独暴露这个函数，是为了让"有内容被丢掉"这件事**可被观测**：
    ``engine.synthesize`` 会据此打一条 warning，前端也可以把它显示成
    "这句话里有读不出来的词"，而不是让内容无声消失。

    Args:
        text: 原始文本（替换前）。

    Returns:
        list[str]: 未覆盖的 ASCII 词，按出现顺序、可能重复。
    """
    return [w for w in _TOKEN_RE.findall(text or "") if not _looks_pronounceable(w)]


def prepare_for_speech(text: str) -> str:
    """把文本里念不出来的术语替换成中文，返回**实际会被念出来**的文本。

    替换规则：
        * 大小写不敏感（``cbt`` 等价于 ``CBT``）；
        * 只替换独立的词 —— ``CBTs`` / ``xCBT`` 不匹配（避免误伤）；
        * 长键优先；
        * 表中未覆盖的拉丁文**原样保留**（合成器会跳过它），但会打 warning，
          因为"静默丢内容"是本项目最怕的失效类型。

    Args:
        text: 待合成文本。

    Returns:
        str: 替换后的文本；输入为空时返回空串。
    """
    raw = text or ""
    if not raw:
        return ""

    missing = unpronounceable_spans(raw)
    if missing:
        # 不抛错、不拦截 —— 但要留痕。丢内容必须是"可见的"。
        logger.warning(
            "[TTS] 以下词不在发音替换表里，合成时会被跳过（内容会丢）：%s",
            sorted(set(missing)),
        )

    def _sub(match: re.Match[str]) -> str:
        word = match.group(0)
        return PRONOUNCE_FIX.get(word.upper(), word)

    return _TOKEN_RE.sub(_sub, raw)
