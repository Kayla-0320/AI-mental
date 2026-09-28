# -*- coding: utf-8 -*-
"""本地离线语音合成引擎 —— sherpa-onnx VITS

设计上与 `asr/engine.py` 的 `ParaformerRecognizer` **同构**：单例懒加载、
失败原因记在 `load_error` 而不是抛异常。理由也一样 ——
服务能起来、但某个能力静默不能用，是本项目已经栽过的坑
（8001 用错解释器：`/health` 照常 200，只有 `/asr/ws` 404）。

三条**实测得出**、不可省略的配置（依据：`docs/tts_benchmark.md`）：

1. ``max_num_sentences = -1``
   设成 1 时 `generate()` **只合成第一句，其余静默丢弃** ——
   66 字的 4 句文本只产出 4258 ms 音频，与 16 字单句的 4218 ms 几乎一样。
   没有异常、没有日志、没有报错。

2. ``num_threads`` 是最重要的性能杠杆
   实测 1 线程 RTF 0.568（不达标）→ 8 线程 0.112。默认按
   ``clamp(CPU核数 - 2, 2, 8)`` 取，**给流式 ASR 和浏览器留核**。

3. 送合成前必须过 :func:`tts.pronunciation.prepare_for_speech`
   词表**不含任何 ASCII 字母**，`CBT` 会被 `Ignore OOV` 静默吞掉。

音频通道：``/api/v1/tts/ws``（见 `api/tts.py`）。
模型：``vits-zh-hf-fanchen-C``，16 kHz 单声道，多说话人（``num_speakers = 187``）。
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from shared.dataclasses import TtsEngine, TtsSynthesisResult

logger = logging.getLogger(__name__)

#: **默认**模型名（落盘目录名与仓库名一致）。可被 ``TTS_MODEL_DIR`` 或显式路径覆盖 ——
#: 换模型不需要改代码。
#:
#: ⚠️ 不要用它去**校验**别的模型目录：`vits-zh-hf-*` 系列的 onnx 文件名各不相同
#: （`vits-zh-hf-fanchen-C.onnx` / `theresa.onnx` / `eula.onnx` …），
#: 按固定文件名判断可用性会把好模型判成"不可用"，然后静默回退到默认模型。
#: （这不是假设 —— 本项目真的这么错过一次：`TTS_MODEL_DIR` 指向 theresa 时无声生效。）
MODEL_NAME = "vits-zh-hf-fanchen-C"

#: 缺了就不能加载的文件。**刻意不含 onnx** —— onnx 按目录内容自动识别（见 `_find_onnx`），
#: 因为各模型的 onnx 文件名不同，写死会把换模型这条路的门焊死。
#: `dict/` 那几项是 jieba 词典，vits-zh-hf-* 系列**必须**有，否则加载直接失败。
REQUIRED_FILES: tuple[str, ...] = (
    "lexicon.txt",
    "tokens.txt",
    "dict/jieba.dict.utf8",
    "dict/hmm_model.utf8",
    "dict/user.dict.utf8",
    "dict/idf.utf8",
    "dict/stop_words.utf8",
)

#: 文本正则化所用的 fst（日期/数字/电话/多音字）。缺失不会导致失败，但会退化成逐字读。
RULE_FSTS: tuple[str, ...] = ("date.fst", "number.fst", "phone.fst", "new_heteronym.fst")

#: 单次合成的文本上限。`_StreamingSafetyGate` 的 unit 上限是 200 字，
#: 但那对语音太长（≈36 秒音频），所以调用方应先切句；这里是兜底。
MAX_TEXT_CHARS = 200

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL_ROOT = _REPO_ROOT / "algorithm" / "models" / "tts"


# ── 模型定位 ────────────────────────────────────────────────────────────────
def _find_onnx(d: Path) -> Optional[Path]:
    """在目录里找出该用的 onnx 模型文件。

    规则：
        * 优先非 int8 的（int8 是量化版，音质通常更差；只有量化版时才用它）；
        * 同优先级取**体积最大**的（多文件时通常是最完整的那个）。

    Returns:
        Optional[Path]: 找到的 onnx；目录不存在或没有 onnx 时返回 None。
    """
    if not d.is_dir():
        return None
    cands = sorted(d.glob("*.onnx"))
    if not cands:
        return None
    full = [p for p in cands if ".int8." not in p.name]
    pool = full or cands
    return max(pool, key=lambda p: p.stat().st_size)


def _usable(d: Path) -> bool:
    """目录是否是一个可加载的模型（有 onnx + 全部必需文件）"""
    return d.is_dir() and _find_onnx(d) is not None and all(
        (d / f).is_file() for f in REQUIRED_FILES
    )


def describe_model_dir(d: Path) -> str:
    """给一个目录写一份"为什么不能用"的说明（用于错误信息，不用于判断）"""
    if not d.exists():
        return f"目录不存在：{d}"
    if not d.is_dir():
        return f"不是目录：{d}"
    missing = [f for f in REQUIRED_FILES if not (d / f).is_file()]
    if _find_onnx(d) is None:
        missing.insert(0, "*.onnx")
    return f"目录缺少文件：{missing}"


def resolve_tts_model_dir(
    explicit: Optional[str | Path] = None,
) -> tuple[Optional[Path], str]:
    """定位 VITS 模型目录。

    查找顺序：
        1. 显式传入的路径
        2. 环境变量 ``TTS_MODEL_DIR``
        3. ``<repo>/algorithm/models/tts/<MODEL_NAME>``（默认模型）
        4. ``<repo>/algorithm/models/tts/`` 下第一个可用的子目录

    ⚠️ **第 1、2 条是"指定"，不是"建议"**：指定了却不可用时**直接失败**，
    绝不回退到别的模型。理由是本项目已经因此坏过一次 ——
    `TTS_MODEL_DIR` 指向一个 onnx 文件名不同的模型，`_usable()` 判它不可用，
    然后**无声地**用了默认模型；调用方看到的是"换模型没生效"，
    而日志里一个字的抱怨都没有。同类的静默降级是这个项目反复踩的坑
    （8001 用错解释器、`max_num_sentences=1` 静默截断）。

    Returns:
        tuple[Optional[Path], str]: ``(模型目录, 失败原因)``。
            成功时原因为空串；失败时目录为 ``None`` 且原因可直接展示给用户。
    """
    # 指定来源：不可用就报错，不回退
    for label, raw in (("显式传入的路径", explicit), ("TTS_MODEL_DIR", os.environ.get("TTS_MODEL_DIR"))):
        if not raw:
            continue
        d = Path(raw)
        if _usable(d):
            return d, ""
        return None, f"{label} 指定的 TTS 模型不可用（{describe_model_dir(d)}）"

    # 默认来源：可以自动挑一个可用的
    default = _DEFAULT_MODEL_ROOT / MODEL_NAME
    if _usable(default):
        return default, ""

    root = _DEFAULT_MODEL_ROOT
    if root.is_dir():
        for sub in sorted(root.iterdir()):
            if _usable(sub):
                logger.info("[TTS] 未找到默认模型 %s，改用 %s", MODEL_NAME, sub.name)
                return sub, ""

    return None, (
        f"未找到任何可用的 TTS 模型（默认位置 {default}）。"
        f"请先运行：python algorithm/tools/download_asr_model.py --preset tts-zh-fanchen-C"
    )


def default_num_threads() -> int:
    """合成线程数：``clamp(CPU核数 - 2, 2, 8)``。

    为什么是"核数 - 2"而不是"全部核"：通话时同机还要跑流式 ASR（16 kHz
    zipformer）与浏览器侧的音频图/视觉管线。把 CPU 吃光会让**识别**变慢，
    而识别变慢直接表现为"AI 反应迟钝"，比合成慢一点更伤体验。

    为什么上限是 8：实测 1→0.568 / 2→0.293 / 4→0.176 / 8→0.128，
    8 之后收益递减而争用风险上升（§4.3）。

    Returns:
        int: 线程数，至少 2（1 线程时 RTF 0.568，达不到 go 判据）。
    """
    cores = os.cpu_count() or 4
    return max(2, min(8, cores - 2))


class VitsTtsEngine:
    """本地离线中文 TTS 引擎（sherpa-onnx ``OfflineTts`` 单例）

    线程安全说明：本类只保护**加载**（双检锁）。``synthesize`` 本身**不加锁** ——
    并发安全由调用方保证（``api/tts.py`` 用单工作线程池串行调用，先例是
    `asr/punctuation.py:161` 的"会话不保证并发安全 → 加锁串行化"）。
    """

    _instance: Optional["VitsTtsEngine"] = None
    _instance_lock = threading.Lock()

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self._model_dir = model_dir
        self._tts = None  # type: ignore[var-annotated]  # sherpa_onnx.OfflineTts
        self._load_error: str = ""
        self._num_threads: int = default_num_threads()

    @classmethod
    def instance(cls) -> "VitsTtsEngine":
        """取全局单例（双检锁；与 `ParaformerRecognizer.instance()` 同构）"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── 状态 ────────────────────────────────────────────────────────────────
    @property
    def available(self) -> bool:
        """是否已成功加载并可用（会触发懒加载）"""
        return self.load()

    @property
    def load_error(self) -> str:
        """最近一次加载失败的原因；为空字符串表示没有失败"""
        return self._load_error

    @property
    def engine(self) -> TtsEngine:
        """当前实际使用的引擎"""
        return TtsEngine.VITS if self.available else TtsEngine.UNAVAILABLE

    @property
    def sample_rate(self) -> int:
        """输出采样率（Hz）；未加载时返回 0（**不要**猜成 16000）"""
        return int(getattr(self._tts, "sample_rate", 0) or 0)

    @property
    def num_speakers(self) -> int:
        """说话人数量。实测 fanchen-C 是 187 —— 多说话人模型，``sid`` 可挑音色。"""
        return int(getattr(self._tts, "num_speakers", 0) or 0)

    @property
    def model_dir(self) -> Optional[str]:
        """实际加载到的模型目录（排查"模型放错位置"用）"""
        return str(self._model_dir) if self._model_dir else None

    @property
    def num_threads(self) -> int:
        """合成线程数（`/tts/status` 会如实报出，便于现场核对性能配置）"""
        return self._num_threads

    # ── 加载 ────────────────────────────────────────────────────────────────
    def load(self) -> bool:
        """懒加载模型；返回是否可用。失败原因记在 :attr:`load_error`。

        三重错误都要如实区分，因为它们对应完全不同的处置：
            * 未安装 sherpa-onnx  → 依赖问题，换解释器
            * 找不到模型目录      → 跑下载脚本
            * 模型加载抛异常      → 文件损坏 / 版本不匹配
        把它们混成一句"不可用"，现场就无法判断该做什么。
        """
        if self._tts is not None:
            return True

        with self._instance_lock:
            if self._tts is not None:
                return True

            try:
                import sherpa_onnx  # noqa: PLC0415 —— 懒导入，缺依赖时不影响其它端点
            except ImportError as exc:  # pragma: no cover —— 依赖缺失
                self._load_error = f"未安装 sherpa-onnx：{exc}"
                logger.warning("[TTS] %s", self._load_error)
                return False

            model_dir, resolve_error = resolve_tts_model_dir(self._model_dir)
            if model_dir is None:
                self._load_error = resolve_error
                logger.warning("[TTS] %s", self._load_error)
                return False

            onnx = _find_onnx(model_dir)
            if onnx is None:  # 竞态兜底：resolve 之后文件被删
                self._load_error = f"模型目录里没有 onnx：{model_dir}"
                logger.warning("[TTS] %s", self._load_error)
                return False

            self._model_dir = model_dir
            dict_dir = model_dir / "dict"
            fsts = [str(model_dir / n) for n in RULE_FSTS if (model_dir / n).is_file()]

            try:
                self._num_threads = default_num_threads()
                config = sherpa_onnx.OfflineTtsConfig(
                    model=sherpa_onnx.OfflineTtsModelConfig(
                        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                            model=str(onnx),
                            lexicon=str(model_dir / "lexicon.txt"),
                            tokens=str(model_dir / "tokens.txt"),
                            dict_dir=str(dict_dir),
                        ),
                        num_threads=self._num_threads,
                        provider="cpu",
                        debug=False,
                    ),
                    # ⚠️ 见模块 docstring 第 1 条：设成 1 会静默丢弃第一句之后的全部内容
                    max_num_sentences=-1,
                    rule_fsts=",".join(fsts),
                )
                self._tts = sherpa_onnx.OfflineTts(config)
                if self.sample_rate <= 0:
                    raise RuntimeError("初始化后 sample_rate 非法，模型未真正加载")
            except Exception as exc:  # noqa: BLE001 —— 加载失败必须转成状态，不能抛给请求
                self._tts = None
                self._load_error = f"VITS 模型加载失败：{exc}"
                logger.exception("[TTS] %s", self._load_error)
                return False

            self._load_error = ""
            logger.info(
                "[TTS] 已加载 %s（采样率 %d，说话人 %d，线程 %d）",
                model_dir.name, self.sample_rate, self.num_speakers, self._num_threads,
            )
            return True

    # ── 合成 ────────────────────────────────────────────────────────────────
    def synthesize(
        self,
        text: str,
        seq: int = 0,
        on_chunk: Optional[Callable[[np.ndarray], int]] = None,
        sid: int = 0,
        speed: float = 1.0,
    ) -> TtsSynthesisResult:
        """合成一句话，通过 ``on_chunk`` 把音频交给调用方。

        Args:
            text: 待合成文本。**调用方保证它已过安全闸门**（本函数不做安全判断）；
                术语替换在 :func:`tts.pronunciation.prepare_for_speech` 里做，
                调用方应先调用它。
            seq: 本次合成的序号，原样带回结果。
            on_chunk: 每产出一块 float32 波形调用一次。**返回非 0 会让
                sherpa-onnx 中止生成**。⚠️ 实测 VITS 对每一句只回调**一次**
                （合成结束后），所以返回值实际上**省不下计算**，只能阻止
                音频被下发。取消要靠"清空待合成队列"，不是靠这个返回值。
                为 None 时音频会被丢弃（只用于自检/benchmark）。
            sid: 说话人 ID。**各模型的说话人数差别很大**：`fanchen-C` 是 187，
                `theresa` / `eula` 是 804。越界时以模型自身行为为准（可能报错、
                也可能落到某个默认音色），所以调用方应先读
                :attr:`num_speakers` 再决定。
            speed: 语速，1.0 为原速。

        Returns:
            TtsSynthesisResult: 计时与状态；音频已通过 ``on_chunk`` 推走。
                失败时 ``error`` 非空且 ``engine`` 为 UNAVAILABLE ——
                **不抛异常**，与 ``/asr/transcribe`` 用 200 + error 字段
                如实上报"服务未就绪"的处理一致。
        """
        import time  # noqa: PLC0415 —— 只在本函数用到，避免污染模块顶层

        if not self.load():
            return TtsSynthesisResult(
                seq=seq, engine=TtsEngine.UNAVAILABLE,
                spoken_text=text, error=self._load_error or "TTS 引擎不可用",
            )

        stripped = (text or "").strip()
        if not stripped:
            # 空文本是**正常**情况（例如闸门只放行了标点），不是错误。
            return TtsSynthesisResult(seq=seq, spoken_text="", audio_ms=0.0)
        if len(stripped) > MAX_TEXT_CHARS:
            # 不截断、不静默处理：让上游知道它违反了切句约定（静默截断是本项目最怕的失效）
            return TtsSynthesisResult(
                seq=seq, engine=TtsEngine.UNAVAILABLE, spoken_text=stripped,
                error=f"文本过长（{len(stripped)} > {MAX_TEXT_CHARS} 字），调用方应先切句",
            )

        # 只累计**已交付**的样本数，不把音频留在内存里 —— 音频的归属是调用方。
        # 之所以要单独计数而不是用 len(audio.samples)：被打断时 sherpa 返回的
        # audio 是完整音频，但实际交付出去的只有前一部分，两者不能混。
        delivered_samples = 0
        first_chunk_ms: list[float] = []
        cancelled = False
        t0 = time.perf_counter()

        def _on_chunk(samples: np.ndarray, _progress: float) -> int:
            nonlocal cancelled, delivered_samples
            if not first_chunk_ms:
                first_chunk_ms.append((time.perf_counter() - t0) * 1000.0)
            if on_chunk is not None and on_chunk(samples) != 0:
                cancelled = True
                return 1  # 停止生成（实测省不下计算，只是不再往后交付）
            delivered_samples += len(np.asarray(samples))
            return 0

        try:
            audio = self._tts.generate(  # type: ignore[union-attr]
                stripped, sid=sid, speed=speed, callback=_on_chunk,
            )
        except Exception as exc:  # noqa: BLE001 —— 单次合成失败不应让整条 WS 断开
            logger.exception("[TTS] 合成失败")
            return TtsSynthesisResult(
                seq=seq, engine=TtsEngine.UNAVAILABLE, spoken_text=stripped,
                error=f"合成失败：{exc}",
                synth_ms=(time.perf_counter() - t0) * 1000.0,
            )

        synth_ms = (time.perf_counter() - t0) * 1000.0
        sample_rate = int(getattr(audio, "sample_rate", 0) or self.sample_rate)
        total_samples = delivered_samples
        if total_samples == 0 and not cancelled:
            # 回调一次都没触发（异常路径）→ 退回整段音频，避免报成"静音"
            total_samples = len(np.asarray(audio.samples, dtype=np.float32))
        audio_ms = total_samples / sample_rate * 1000.0 if sample_rate else 0.0

        result = TtsSynthesisResult(
            seq=seq,
            engine=TtsEngine.VITS,
            spoken_text=stripped,
            sample_rate=sample_rate,
            audio_ms=audio_ms,
            first_chunk_ms=first_chunk_ms[0] if first_chunk_ms else synth_ms,
            synth_ms=synth_ms,
            cancelled=cancelled,
        )
        # 静默失效必须留痕：模型"成功返回但没产出音频"是本项目最怕的失效类型
        # （max_num_sentences=1 的截断、或文本全被 OOV 吞掉都会长这样）。
        if result.is_silent and not cancelled:
            logger.warning("[TTS] 合成结果为空音频（seq=%d，文本 %d 字）：%s",
                           seq, len(stripped), stripped[:40])
        return result

    # ── 自检 ────────────────────────────────────────────────────────────────
    def status(self) -> dict[str, object]:
        """引擎状态字典，供 ``/tts/status`` 如实上报。

        刻意包含 ``num_threads`` / ``num_speakers`` / ``model_dir`` / ``sample_rate``：
        现场出问题时，"用了几个线程""加载的是哪个目录""采样率多少"是最常见的几个根因。
        **`model_name` 报的是实际加载到的目录名**，不是默认模型名 ——
        否则"换模型没生效"这种问题在状态里根本看不出来。
        """
        ok = self.load()
        return {
            "available": ok,
            "engine": self.engine.value,
            "model_dir": self.model_dir,
            # 用实际目录名，而不是 MODEL_NAME（默认模型名）
            "model_name": self._model_dir.name if (ok and self._model_dir) else None,
            "sample_rate": self.sample_rate,
            "num_speakers": self.num_speakers,
            "num_threads": self.num_threads,
            "max_num_sentences": -1,   # 如实报出：截断坑的护栏值
            "error": self._load_error,
        }
