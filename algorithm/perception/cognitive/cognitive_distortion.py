"""
认知扭曲深度分析模块 —— 基于 CBT 理论的七类认知扭曲检测

临床依据：
- Beck (1979): 认知扭曲理论 — 抑郁的核心维持因素
- Garber & Hollon (1991): 青少年认知扭曲发展特征
- Burns (1980): Feeling Good — 十类认知扭曲

七类认知扭曲：
1. 灾难化 (Catastrophizing) — 把小事想成天大的灾难
2. 非黑即白 (Black-and-White) — 全或无思维
3. 过度概括 (Overgeneralization) — 一次失败=永远失败
4. 自我归咎 (Self-Blame) — 一切都是我的错
5. 无望感 (Hopelessness) — 未来不会好
6. 读心术 (Mind Reading) — 别人一定觉得我很差
7. 应该陈述 (Should Statements) — 我应该/必须...

年龄差异化：
- 初中：表达直接，关键词匹配率高，阈值低
- 高中：开始使用隐喻和反讽，需要更多模式匹配
- 大学：表达隐晦、哲学化，需要更灵敏的检测
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from perception.age_config import CognitiveDistortionAgeParams, AgeProfile


@dataclass
class CognitiveDistortionFeatures:
    """认知扭曲分析特征"""
    text: str = ""
    # 可选：历史文本列表（用于趋势分析）
    text_history: list[str] = field(default_factory=list)


def analyze_cognitive_distortions(
    features: CognitiveDistortionFeatures,
    age_profile: Optional[AgeProfile] = None,
) -> dict:
    """分析认知扭曲风险

    Args:
        features: 认知扭曲特征
        age_profile: 年龄画像

    Returns:
        {
            "risk_score": float (0-1),
            "categories": {
                "catastrophizing": float,
                "black_and_white": float,
                "overgeneralization": float,
                "self_blame": float,
                "hopelessness": float,
                "mind_reading": float,
                "should_statements": float,
            },
            "total_distortion_count": int,
            "evidence": list[str],
        }
    """
    if age_profile is not None:
        cp = age_profile.cognitive
    else:
        cp = CognitiveDistortionAgeParams()

    text = features.text.lower()
    evidence: list[str] = []

    def _count_matches(keywords: list[str]) -> tuple[float, list[str]]:
        matched = [kw for kw in keywords if kw in text]
        score = min(cp.max_risk_per_category, len(matched) * cp.max_risk_per_category / 3)
        return score, matched

    # 七类认知扭曲检测
    cat_scores: dict[str, float] = {}
    cat_matches: dict[str, list[str]] = {}

    categories = [
        ("catastrophizing", cp.catastrophizing_keywords),
        ("black_and_white", cp.black_and_white_keywords),
        ("overgeneralization", cp.overgeneralization_keywords),
        ("self_blame", cp.self_blame_keywords),
        ("hopelessness", cp.hopelessness_keywords),
        ("mind_reading", cp.mind_reading_keywords),
        ("should_statements", cp.should_statements_keywords),
    ]

    total_count = 0
    for name, keywords in categories:
        score, matched = _count_matches(keywords)
        cat_scores[name] = round(score, 4)
        cat_matches[name] = matched
        total_count += len(matched)

    # 综合风险：各类风险的加权和
    risk_score = sum(cat_scores.values())
    risk_score = min(1.0, max(0.0, risk_score))

    # 多类扭曲同时出现 → 额外风险加成
    active_categories = sum(1 for s in cat_scores.values() if s > 0)
    if active_categories >= 3:
        risk_score = min(1.0, risk_score * 1.2)
        evidence.append(f"多类认知扭曲同时出现({active_categories}类)")

    # 生成证据
    category_labels = {
        "catastrophizing": "灾难化思维",
        "black_and_white": "非黑即白",
        "overgeneralization": "过度概括",
        "self_blame": "自我归咎",
        "hopelessness": "无望感",
        "mind_reading": "读心术",
        "should_statements": "应该陈述",
    }
    for name, matched in cat_matches.items():
        if matched:
            evidence.append(f"{category_labels[name]}: {', '.join(matched[:3])}")

    if not evidence:
        evidence.append("未检测到明显认知扭曲")

    return {
        "risk_score": round(risk_score, 4),
        "categories": cat_scores,
        "total_distortion_count": total_count,
        "evidence": evidence,
    }
