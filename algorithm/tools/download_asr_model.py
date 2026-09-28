#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""下载本地语音模型 —— sherpa-onnx 中文模型（走 hf-mirror）

为什么需要这个脚本：
    本机 **huggingface.co 不可达**（实测 timeout），但 **hf-mirror.com 可达**。
    GitHub release 资产同样不可达。因此模型必须走镜像下载，且必须可复现。
    另外 hf-mirror 对非浏览器 UA 返回 403（见下方 _BROWSER_UA 注释）。

三个预设对应语音链路的三段：
    streaming-zipformer-zh-14m     流式识别 24 MB   —— /asr/ws 边说边出字
    paraformer-zh-small            离线识别 78 MB   —— /asr/transcribe + 定稿纠错
    punct-zh-en                    标点恢复 294 MB  —— 定稿时补标点

用法：
    python algorithm/tools/download_asr_model.py                          # 默认（离线）
    python algorithm/tools/download_asr_model.py --preset streaming-zipformer-zh-14m
    python algorithm/tools/download_asr_model.py --preset punct-zh-en
    python algorithm/tools/download_asr_model.py --preset all             # 三个都下
    set HF_ENDPOINT=https://hf-mirror.com                                 # 覆盖镜像

落盘位置：
    algorithm/models/<子目录>/<repo 名>/     （asr / punct）
（algorithm/models/ 已在 .gitignore 中排除，模型不入版本库）
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Preset:
    """一个模型预设：仓库名 + 必需文件 + 可选自检样本

    Attributes:
        repo: HuggingFace 仓库名
        required: 缺了就不能用的文件
        optional: 自检用的小样本等
        subdir: 落在 algorithm/models/ 下的哪个子目录
        note: 用途说明
    """
    repo: str
    required: tuple[str, ...]
    optional: tuple[str, ...] = field(default_factory=tuple)
    subdir: str = "asr"
    note: str = ""


#: 自检样本三个 ASR 仓库同名，抽出来避免重复
_TEST_WAVS = ("test_wavs/0.wav", "test_wavs/1.wav", "test_wavs/8k.wav")

PRESETS: dict[str, Preset] = {
    # 离线整段识别。am.mvn 是 ModelScope 原始导出的遗留文件，
    # sherpa-onnx 的 from_paraformer() 并不消费它（CMVN 已烘焙进 ONNX），
    # 但保留它与上游仓库一致，也方便换用其它工具链时对齐。
    "paraformer-zh-small": Preset(
        repo="csukuangfj/sherpa-onnx-paraformer-zh-small-2024-03-09",
        required=("model.int8.onnx", "tokens.txt", "am.mvn"),
        optional=_TEST_WAVS,
        note="离线整段识别 78 MB —— 供 /asr/transcribe，以及流式定稿后的复识别纠错",
    ),
    # 流式识别。14M 参数、中文专用，实测 RTF 远低于实时，适合边说边出字。
    # 对比 streaming-paraformer-bilingual-zh-en（226 MB）与
    # streaming-zipformer-bilingual-zh-en（189 MB），这个只有 24 MB。
    "streaming-zipformer-zh-14m": Preset(
        repo="csukuangfj/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23",
        required=(
            "encoder-epoch-99-avg-1.int8.onnx",
            "decoder-epoch-99-avg-1.int8.onnx",
            "joiner-epoch-99-avg-1.int8.onnx",
            "tokens.txt",
        ),
        optional=_TEST_WAVS,
        note="流式识别 24 MB（中文专用）—— 供 /asr/ws 边说边出字使用",
    ),
    # 标点恢复。流式与离线 ASR 都不输出标点，不定稿补标点就会几段话连成一片。
    # 注意体积：280 MB，比两个 ASR 模型加起来还大 —— 这是目前唯一的中文标点模型。
    "punct-zh-en": Preset(
        repo="csukuangfj/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12",
        required=("model.onnx", "tokens.json"),
        subdir="punct",
        note="中英标点恢复 280 MB —— 定稿时给识别文本补标点",
    ),
    # 离线中文 TTS。**必需文件清单来自 hf-mirror 的仓库文件列表（实测），
    # 不是照抄文档** —— vits-zh-hf-* 这一系列除了 onnx/lexicon/tokens 之外，
    # 还必须带 jieba 词典目录（dict/），否则 sherpa-onnx 加载时直接失败。
    # fst 四个是文本正则化（日期/数字/电话/多音字）用的，缺了会退化成逐字读。
    "tts-zh-fanchen-C": Preset(
        repo="csukuangfj/vits-zh-hf-fanchen-C",
        required=(
            "vits-zh-hf-fanchen-C.onnx",
            "lexicon.txt",
            "tokens.txt",
            "date.fst",
            "number.fst",
            "phone.fst",
            "new_heteronym.fst",
            "dict/jieba.dict.utf8",
            "dict/hmm_model.utf8",
            "dict/user.dict.utf8",
            "dict/idf.utf8",
            "dict/stop_words.utf8",
            "dict/pos_dict/char_state_tab.utf8",
            "dict/pos_dict/prob_emit.utf8",
            "dict/pos_dict/prob_start.utf8",
            "dict/pos_dict/prob_trans.utf8",
        ),
        optional=("G_C.json", "dict/README.md"),
        subdir="tts",
        note="离线中文 TTS 121 MB —— 陪伴通话的出声端（音频不出设备）",
    ),
    # 兜底 TTS：体积只有 fanchen-C 的一半，同架构、同采样率。
    # ⚠️ 它的 MODEL_CARD 写 License: Unknown，用于竞赛 Demo 前必须确认授权。
    # ⚠️ 它走 espeak-ng 音素化，因此要额外下整个 espeak-ng-data/ 目录（数百个小文件），
    #    本预设只列必需的那几个；真要用时按报错补。
    "tts-zh-huayan": Preset(
        repo="csukuangfj/vits-piper-zh_CN-huayan-medium",
        required=(
            "zh_CN-huayan-medium.onnx",
            "tokens.txt",
            "espeak-ng-data/phondata",
            "espeak-ng-data/phontab",
            "espeak-ng-data/phonindex",
            "espeak-ng-data/cmn_dict",
        ),
        optional=("MODEL_CARD", "zh_CN-huayan-medium.onnx.json"),
        subdir="tts",
        note="备选中文 TTS 63 MB —— ⚠️ License: Unknown，用前须确认授权",
    ),
    # 对话级 TTS（Qwen3-TTS 0.6B）。用 snapshot_download 整仓下拉，不逐文件列：
    # 模型由 transformers 自动加载，配置文件（config/tokenizer/preprocessor）
    # 的版本与权重强耦合，少一个文件都会让 AutoProcessor.from_pretrained 失败。
    # ⚠️ FP16 约 1.2 GB；INT4 量化版约 600 MB（需要另找量化仓库或本地量化）。
    # 首次运行时若本地目录不存在，引擎会尝试从 HF_ENDPOINT 自动拉取作为兜底。
    "qwen3-tts-voicedesign": Preset(
        repo="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        required=(),
        subdir="tts",
        note="对话级中文 TTS ≈3.4 GB —— voice design 驱动音色（知心姐姐），需 GPU",
    ),
}

#: 同一系列的其他中文音色（`vits-zh-hf-*`）。
#:
#: 为什么单列出来：`vits-zh-hf-fanchen-C` **自称有 187 个说话人**（实测确为真，
#: 不同 sid 产出完全不同的波形），所以换音色的第一选择是**换 sid**，不用下载。
#: 只有在 187 个都不满意时，才需要下这些**独立模型**（每个约 120 MB）。
#:
#: 结构与 fanchen-C 完全一致（onnx + lexicon + tokens + 4 个 fst + dict/ jieba 词典），
#: 只是 onnx 文件名各不相同，所以这里参数化生成而不是手写四遍。
_HF_VOICE_ONNX: dict[str, str] = {
    "fanchen-wnj": "vits-zh-hf-fanchen-wnj.onnx",
    "theresa": "theresa.onnx",
    "eula": "eula.onnx",
    "keqing": "keqing.onnx",
}

_HF_VOICE_COMMON: tuple[str, ...] = (
    "lexicon.txt",
    "tokens.txt",
    "date.fst",
    "number.fst",
    "phone.fst",
    "new_heteronym.fst",
    "dict/jieba.dict.utf8",
    "dict/hmm_model.utf8",
    "dict/user.dict.utf8",
    "dict/idf.utf8",
    "dict/stop_words.utf8",
    "dict/pos_dict/char_state_tab.utf8",
    "dict/pos_dict/prob_emit.utf8",
    "dict/pos_dict/prob_start.utf8",
    "dict/pos_dict/prob_trans.utf8",
)

for _name, _onnx in _HF_VOICE_ONNX.items():
    PRESETS[f"tts-zh-{_name}"] = Preset(
        repo=f"csukuangfj/vits-zh-hf-{_name}",
        required=(_onnx, *_HF_VOICE_COMMON),
        subdir="tts",
        note=f"中文音色 {_name}（约 120 MB）—— 187 个 sid 都不满意时的备选",
    )

DEFAULT_PRESET = "paraformer-zh-small"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_ROOT = REPO_ROOT / "algorithm" / "models"


def mirror_endpoint() -> str:
    """模型源。HuggingFace 直连在本网络环境不可达，默认走 hf-mirror。"""
    return os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")


# ⚠️ hf-mirror 对**非 LFS 的小文件**（tokens.txt / am.mvn / test_wavs）会按 User-Agent
# 拦截：urllib 默认的 "Python-urllib/3.x" 直接返回 403 Forbidden，而 LFS 大文件
# （model.int8.onnx）却正常。实测带浏览器 UA 后即可 200，因此这里必须显式设置。
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36"
)


def download(repo: str, filename: str, dest: Path, retries: int = 5) -> None:
    """下载单个文件，**支持断点续传**。

    为什么必须支持续传：镜像上大文件（TTS 的 onnx 121 MB、标点模型 280 MB）在
    本网络环境下会在跑到一半时 `The read operation timed out`（实测：121 MB 跑到
    82% 断掉）。不支持续传时每次重试都从 0 开始，重试 3 次也大概率还是失败，
    而"已下 100 MB 全部作废"的代价随时间线性上升。续传把一次断连的代价降到
    "只丢最后一块"。

    实现要点：
        * 落盘写 `.part` 临时文件，**成功才改名**，避免半成品被下次误判为"已存在"；
        * 有 `.part` 就带 `Range: bytes=<已有长度>-` 续；
        * 服务端若不支持 Range 会返回 200（而非 206），此时必须**清空重下**，
          否则会把新内容的开头追加到旧内容后面，得到一个大小对但内容坏的文件。
    """
    url = f"{mirror_endpoint()}/{repo}/resolve/main/{filename}"
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [跳过] {filename}  已存在（{dest.stat().st_size / 1e6:.1f} MB）")
        return

    tmp = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, retries + 1):
        resume_from = tmp.stat().st_size if tmp.exists() else 0
        try:
            headers = {"User-Agent": _BROWSER_UA}
            if resume_from:
                headers["Range"] = f"bytes={resume_from}-"
                print(f"  [续传] {filename}  从 {resume_from / 1e6:.1f} MB 继续")
            else:
                print(f"  [下载] {filename}  <- {url}")

            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                # 服务端忽略了 Range → 200 全量返回，必须从头写，不能追加
                if resume_from and resp.status != 206:
                    print("         （服务端不支持续传，从头下载）")
                    resume_from = 0
                total_header = int(resp.headers.get("Content-Length") or 0)
                total = total_header + resume_from if total_header else 0
                done = resume_from
                mode = "ab" if resume_from else "wb"
                with open(tmp, mode) as fh:
                    while True:
                        chunk = resp.read(1 << 16)
                        if not chunk:
                            break
                        fh.write(chunk)
                        done += len(chunk)
                        if total:
                            pct = done * 100 / total
                            print(f"\r         {pct:5.1f}%  {done / 1e6:7.1f}/{total / 1e6:.1f} MB", end="")
                print()

                # 校验：拿到了 Content-Length 就必须下满，否则当作本次失败（保留 .part 续传）
                if total and done < total:
                    raise OSError(f"连接提前结束：{done / 1e6:.1f}/{total / 1e6:.1f} MB")

                tmp.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            remaining = tmp.stat().st_size if tmp.exists() else 0
            print(f"         第 {attempt}/{retries} 次失败：{exc}（已保留 {remaining / 1e6:.1f} MB）")
            if attempt == retries:
                raise SystemExit(
                    f"下载失败：{filename}\n"
                    f"  已下载的部分保留在 {tmp.name}，重跑本命令会自动续传。\n"
                    f"  请确认镜像可达，或用 HF_ENDPOINT 指定其它镜像。"
                )


def _force_utf8_stdout() -> None:
    """Windows 控制台默认 GBK，打印 ✅/❌ 会 UnicodeEncodeError。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, OSError):
            pass


def _download_preset(name: str, preset: Preset, models_root: Path, skip_test_wavs: bool) -> bool:
    """下载单个预设，返回是否成功

    两条路径：
        * ``preset.required`` 非空 → 按文件清单逐个下载（支持续传，适合大 onnx）
        * ``preset.required`` 为空 → 走 ``huggingface_hub.snapshot_download``
          整仓下拉（适合 transformers 类模型，配置文件多且耦合）
    """
    dest = models_root / preset.subdir / preset.repo.split("/")[-1]

    print(f"── {name} ──")
    print(f"  说明     : {preset.note}")
    print(f"  仓库     : {preset.repo}")
    print(f"  落盘目录 : {dest}")
    print()

    if not preset.required:
        # 整仓下拉路径（Qwen3-TTS 等）
        try:
            from huggingface_hub import snapshot_download  # noqa: PLC0415
        except ImportError as exc:
            print(f"\n❌ {name} 需要 huggingface_hub：uv pip install huggingface_hub（{exc}）\n")
            return False

        endpoint = mirror_endpoint()
        try:
            snapshot_download(
                repo_id=preset.repo,
                local_dir=str(dest),
                endpoint=endpoint if endpoint != "https://huggingface.co" else None,
                resume_download=True,
            )
        except Exception as exc:  # noqa: BLE001 —— 下载脚本里直接报错给用户
            print(f"\n❌ {name} snapshot_download 失败：{exc}\n")
            return False

        print(f"  ✅ {name} 就绪（snapshot_download）\n")
        return True

    files = list(preset.required)
    if not skip_test_wavs:
        files += list(preset.optional)

    for fname in files:
        try:
            download(preset.repo, fname, dest / fname)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001 —— 可选文件失败不阻断
            print(f"  [警告] {fname} 下载失败，跳过：{exc}")

    missing = [f for f in preset.required if not (dest / f).exists()]
    if missing:
        print(f"\n❌ {name} 缺少必需文件：{missing}\n")
        return False

    total = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    print(f"  ✅ {name} 就绪（{total / 1e6:.1f} MB）\n")
    return True


def main(argv: Iterable[str] | None = None) -> int:
    _force_utf8_stdout()
    parser = argparse.ArgumentParser(description="下载本地 ASR 模型（走 hf-mirror）")
    parser.add_argument(
        "--preset",
        default=DEFAULT_PRESET,
        choices=[*PRESETS.keys(), "all"],
        help=f"要下载的模型预设（默认 {DEFAULT_PRESET}；all = 全部）",
    )
    parser.add_argument("--dest-root", default=None, help="落盘根目录（默认 algorithm/models）")
    parser.add_argument("--skip-test-wavs", action="store_true", help="不下载自检样本")
    args = parser.parse_args(list(argv) if argv is not None else None)

    # 注意：dest_root 是 algorithm/models（各预设自带 subdir，如 asr / punct）
    models_root = Path(args.dest_root) if args.dest_root else DEFAULT_MODELS_ROOT

    print(f"镜像     : {mirror_endpoint()}")
    print(f"落盘根目录: {models_root}")
    print()

    names = list(PRESETS.keys()) if args.preset == "all" else [args.preset]
    ok = True
    for n in names:
        ok = _download_preset(n, PRESETS[n], models_root, args.skip_test_wavs) and ok

    if not ok:
        return 1
    print("全部就绪。可用环境变量指定其它位置：ASR_MODEL_DIR=<某个模型目录>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
