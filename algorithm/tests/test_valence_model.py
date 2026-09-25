"""效价 RoBERTa 适配层测试

覆盖两件容易出错、且一旦出错就没人发现的事：
  1. 3类效价 → 平台5类 的映射是否遵守「不凭空捏造型别」的约定；
  2. 词典类型线索的门槛是否真的挡住了「没匹配到词也返回均匀基线」的情况。

映射函数是纯函数，先测它；涉及 transformers/jieba 的部分缺依赖时跳过。
"""
from __future__ import annotations

import math
import unittest

from perception.text_emotion.valence_model import (
    IDX_ANGER,
    IDX_ANXIETY,
    IDX_DEPRESSION,
    IDX_NEUTRAL,
    IDX_POSITIVE,
    MAP_TYPED,
    MAP_VALENCE_ONLY,
    PLATFORM_LABELS,
    dominant_label,
    map_valence_to_platform,
)


class TestValenceToPlatformMapping(unittest.TestCase):
    """3 类效价 → 平台 5 类的映射"""

    def test_positive_and_neutral_pass_through(self):
        """积极与中性直接搬运，不缩放"""
        probs, _ = map_valence_to_platform(0.1, 0.3, 0.6)
        self.assertAlmostEqual(probs[IDX_POSITIVE], 0.6, places=6)
        self.assertAlmostEqual(probs[IDX_NEUTRAL], 0.3, places=6)

    def test_no_typed_weights_puts_all_negative_on_depression(self):
        """无线索时负向质量全部给悲伤，且标记 valence_only"""
        probs, mapping = map_valence_to_platform(0.8, 0.1, 0.1)
        self.assertEqual(mapping, MAP_VALENCE_ONLY)
        self.assertAlmostEqual(probs[IDX_DEPRESSION], 0.8, places=6)
        self.assertAlmostEqual(probs[IDX_ANXIETY], 0.0, places=6)
        self.assertAlmostEqual(probs[IDX_ANGER], 0.0, places=6)

    def test_none_typed_weights_same_as_absent(self):
        a, m1 = map_valence_to_platform(0.5, 0.3, 0.2)
        b, m2 = map_valence_to_platform(0.5, 0.3, 0.2, typed_weights=None)
        self.assertEqual(m1, m2)
        self.assertEqual(a, b)

    def test_typed_weights_split_proportionally(self):
        """有线索时按相对强度拆分"""
        probs, mapping = map_valence_to_platform(0.6, 0.2, 0.2, typed_weights=(3.0, 1.0, 0.0))
        self.assertEqual(mapping, MAP_TYPED)
        self.assertAlmostEqual(probs[IDX_ANXIETY], 0.45, places=6)
        self.assertAlmostEqual(probs[IDX_DEPRESSION], 0.15, places=6)
        self.assertAlmostEqual(probs[IDX_ANGER], 0.0, places=6)

    def test_zero_and_negative_weights_are_ignored(self):
        """全 0 / 含负值的线索视为无线索，不得凭空造型别"""
        for w in (None, (0.0, 0.0, 0.0), (-1.0, -2.0, 0.0), (-0.5, 0.0, 0.0)):
            probs, mapping = map_valence_to_platform(0.7, 0.2, 0.1, typed_weights=w)
            self.assertEqual(mapping, MAP_VALENCE_ONLY, f"weights={w}")
            self.assertAlmostEqual(probs[IDX_DEPRESSION], 0.7, places=6)

    def test_negative_components_clamped(self):
        """负概率被截断为 0，不产生负的输出"""
        probs, _ = map_valence_to_platform(-0.5, 1.2, 0.3)
        self.assertTrue(all(p >= 0.0 for p in probs))

    def test_sums_to_one(self):
        """任意输入下 5 维之和为 1"""
        cases = [
            (0.5, 0.3, 0.2, None),
            (0.9, 0.05, 0.05, (1.0, 1.0, 1.0)),
            (0.0, 0.0, 1.0, (5.0, 0.1, 0.1)),
            (1.0, 0.0, 0.0, (0.0, 0.0, 9.0)),
        ]
        for pn, pu, pp, w in cases:
            probs, _ = map_valence_to_platform(pn, pu, pp, typed_weights=w)
            self.assertAlmostEqual(sum(probs), 1.0, places=6, msg=f"{pn},{pu},{pp},{w}")

    def test_degenerate_input_does_not_nan(self):
        """全 0 输入不得产生 NaN/inf"""
        probs, _ = map_valence_to_platform(0.0, 0.0, 0.0)
        self.assertTrue(all(math.isfinite(p) for p in probs))
        self.assertAlmostEqual(sum(probs), 1.0, places=6)

    def test_length_and_order(self):
        probs, _ = map_valence_to_platform(0.2, 0.3, 0.5)
        self.assertEqual(len(probs), 5)
        self.assertEqual(len(PLATFORM_LABELS), 5)
        self.assertEqual(PLATFORM_LABELS, ("anxiety", "depression", "anger", "neutral", "positive"))


class TestDominantLabel(unittest.TestCase):
    def test_dominant(self):
        label, conf = dominant_label([0.1, 0.7, 0.1, 0.05, 0.05])
        self.assertEqual(label, "depression")
        self.assertAlmostEqual(conf, 0.7, places=6)

    def test_dominant_positive(self):
        label, _ = dominant_label([0.0, 0.0, 0.0, 0.2, 0.8])
        self.assertEqual(label, "positive")


class TestLexiconTypedWeightsGating(unittest.TestCase):
    """词典类型线索的门槛 —— 防止把中性基线当成类型证据"""

    def setUp(self) -> None:
        try:
            import jieba  # noqa: F401
            import transformers  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"缺依赖: {exc}")

    def test_no_match_returns_none(self):
        """一个情感词都没匹配到 → 无类型线索"""
        from perception.text_emotion.predict import _lexicon_typed_weights
        self.assertIsNone(_lexicon_typed_weights("今天坐地铁去公司然后回家"))

    def test_anxiety_cue_is_used(self):
        """明确的焦虑词应给出类型线索"""
        from perception.text_emotion.predict import _lexicon_typed_weights
        w = _lexicon_typed_weights("我最近特别焦虑，晚上睡不着，心慌")
        self.assertIsNotNone(w)
        self.assertGreater(w[0], 0.0, f"焦虑通道应有权重: {w}")

    def test_positive_text_has_no_negative_type(self):
        """纯积极文本不应在负向类型上给出线索"""
        from perception.text_emotion.predict import _lexicon_typed_weights
        self.assertIsNone(_lexicon_typed_weights("今天超级开心，特别幸福"))


class TestPredictIntegration(unittest.TestCase):
    """predict() 的端到端契约（模型未就绪时也应返回合法结构）"""

    def setUp(self) -> None:
        try:
            import transformers  # noqa: F401
            import jieba  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"缺依赖: {exc}")

    def test_returns_valid_contract(self):
        from perception.text_emotion.predict import predict
        r = predict("我今天心情很低落，什么都不想做")
        self.assertIn(r["label"], PLATFORM_LABELS)
        self.assertEqual(len(r["probs"]), 5)
        self.assertAlmostEqual(sum(r["probs"]), 1.0, places=3)
        self.assertIn("engine", r)

    def test_engine_and_mapping_reported(self):
        """必须如实上报实际引擎；效价引擎必须带 mapping 字段"""
        from perception.text_emotion.predict import predict
        r = predict("最近压力很大，总是睡不着")
        self.assertIn(r["engine"], ("roberta_valence", "roberta5", "lexicon"))
        if r["engine"] == "roberta_valence":
            self.assertIn(r["mapping"], (MAP_TYPED, MAP_VALENCE_ONLY))
            self.assertEqual(len(r["valence"]), 3)

    def test_lexicon_fallback_still_five_dims(self):
        """词典降级路径同样返回 5 维"""
        from perception.text_emotion.predict import _lexicon_predict
        r = _lexicon_predict("今天很开心")
        self.assertEqual(len(r["probs"]), 5)
        self.assertEqual(r["engine"], "lexicon")


if __name__ == "__main__":
    unittest.main()
