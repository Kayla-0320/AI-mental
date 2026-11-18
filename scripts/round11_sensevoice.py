"""严格加载冻结的中文SenseVoiceSmall，保存真实语音状态、原生标签和ASR。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "algorithm"))

import av
import numpy as np
import torch
import yaml
from experiments.round11_common import ROOT, OUTPUT, digest, save

MODEL = Path(r"D:\ai\models\encoders\sensevoice-small")
REVISION = "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7"
WEIGHT_SHA256 = "833ca2dcfdf8ec91bd4f31cfac36d6124e0c459074d5e909aec9cabe6204a3ea"


def build_model() -> tuple[Any, dict[str, Any]]:
    """本地固定配置构建，strict核对全部权重，禁止隐式下载和忽略失配。"""
    from funasr import AutoModel
    import funasr.models.sense_voice.model  # 显式注册官方实现。
    if digest(MODEL / "model.pt") != WEIGHT_SHA256:
        raise ValueError("中文语音权重摘要不符")
    config = yaml.safe_load((MODEL / "config.yaml").read_text("utf-8"))
    keep = ("encoder", "encoder_conf", "model", "model_conf", "tokenizer", "tokenizer_conf", "frontend", "frontend_conf")
    config = {key: config[key] for key in keep}
    config["tokenizer_conf"]["bpemodel"] = str(MODEL / "chn_jpn_yue_eng_ko_spectok.bpe.model")
    config["frontend_conf"]["cmvn_file"] = str(MODEL / "am.mvn")
    # 推理禁用训练用的随机抖动，使单条重放和断点续编不依赖先前样本顺序。
    config["frontend_conf"]["dither"] = 0.
    config.update(device="cuda", ncpu=1, seed=42)
    model, kwargs = AutoModel.build_model(**config)
    weights = torch.load(MODEL / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    model = model.cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, kwargs


def decode(path: Path) -> tuple[np.ndarray, float, bool]:
    """实际音轨重采样；十秒以上居中截取并保留原时长。"""
    try:
        parts = []
        with av.open(str(path)) as container:
            stream = container.streams.audio[0]
            stream.codec_context.thread_count = 1
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
            for frame in container.decode(audio=0):
                parts.extend(out.to_ndarray().reshape(-1) for out in resampler.resample(frame))
            parts.extend(out.to_ndarray().reshape(-1) for out in resampler.resample(None))
        values = np.concatenate(parts).astype(np.float32) if parts else np.empty(0, np.float32)
        duration = len(values)/16000
        valid = len(values) >= 1600 and np.isfinite(values).all()
        if len(values) > 160000:
            start = (len(values)-160000)//2
            values = values[start:start+160000]
        return np.clip(values, -1, 1), duration, bool(valid)
    except (OSError, ValueError, IndexError, av.error.FFmpegError):
        return np.empty(0, np.float32), 0., False


def encode(model: Any, kwargs: dict[str, Any], waves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """直接调用官方推理，hook保存同一次推理的状态，不注入转录或金标。"""
    captured: list[Any] = []
    def capture(_module: Any, _inputs: Any, result: Any) -> None:
        captured.append(result)
    hook = model.encoder.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            results, _ = model.inference(data_in=waves, tokenizer=kwargs["tokenizer"],
                frontend=kwargs["frontend"], language="zh", use_itn=False, device="cuda",
                fs=16000, data_type="sound", output_timestamp=False)
            if len(captured) != 1:
                raise ValueError("编码状态调用次数不符")
            states, lengths = captured[0]
            states = states[0] if isinstance(states, tuple) else states
            probabilities = model.ctc.log_softmax(states[:, :4]).float()
            emo_ids = [model.emo_dict[key] for key in ("happy", "sad", "angry", "neutral", "unk")]
            pooled = []
            for i, length in enumerate(lengths.tolist()):
                values = states[i, 4:int(length)].float()
                if not len(values):
                    raise ValueError("没有语音状态")
                pooled.append(torch.cat((values.mean(0), values.std(0, unbiased=False))))
            return (torch.stack(pooled).cpu().numpy(), states[:, :4].float().cpu().numpy(),
                    probabilities[:, :, emo_ids].cpu().numpy(), [row["text"] for row in results])
    finally:
        hook.remove()


def acquire_split(model: Any, kwargs: dict[str, Any], split: str, batch: int) -> None:
    """先提取全部可解码语音；旧mask与修正mask在训练侧分别控制观测。"""
    output = OUTPUT / "sensevoice_deterministic_features"
    output.mkdir(parents=True, exist_ok=True)
    completed = output / f"{split}.json"
    if completed.exists():
        raise ValueError("拒绝覆盖已完成中文语音编码")
    with np.load(ROOT / f"_国赛优化/round5/text_features/{split}.npz", allow_pickle=False) as data:
        identifiers = data["sample_ids"].astype(str)
    source = ROOT / "data/public/ch_sims_v2/supervised"
    cache = output / "samples"
    cache.mkdir(exist_ok=True)
    missing = [key for key in identifiers if not (cache/f"{key}.npz").exists()]
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0, len(missing), batch):
            keys = missing[start:start+batch]
            decoded = list(pool.map(decode, [source/f"{key}.mp4" for key in keys]))
            indices = [i for i, row in enumerate(decoded) if row[2]]
            if indices:
                pooled, query, emo, transcripts = encode(model, kwargs, [decoded[i][0] for i in indices])
                encoded = {index: j for j, index in enumerate(indices)}
            else:
                encoded = {}
            for i, key in enumerate(keys):
                j = encoded.get(i)
                np.savez_compressed(cache/f"{key}.npz", sample_id=np.array(key),
                    sensevoice=pooled[j] if j is not None else np.zeros(1024, np.float32),
                    query=query[j] if j is not None else np.zeros((4,512), np.float32),
                    emotion_logprobs=emo[j] if j is not None else np.zeros((4,5), np.float32),
                    transcript=np.array(transcripts[j] if j is not None else ""),
                    available=np.array(decoded[i][2]), original_duration=np.array(decoded[i][1]),
                    encoder_sha256=np.array(WEIGHT_SHA256))
            if start % 128 == 0 or start+batch >= len(missing):
                print(json.dumps({"split": split, "encoded": min(len(missing), start+batch),
                                  "remaining_at_start": len(missing), "seconds": round(time.monotonic()-started, 1)}), flush=True)
    records = []
    for key in identifiers:
        with np.load(cache/f"{key}.npz", allow_pickle=False) as record:
            if str(record["sample_id"]) != key or str(record["encoder_sha256"]) != WEIGHT_SHA256:
                raise ValueError("缓存身份不符")
            records.append({field: record[field] for field in record.files})
    fields = ("sensevoice", "query", "emotion_logprobs", "transcript", "available", "original_duration")
    np.savez_compressed(output/f"{split}.npz", sample_ids=identifiers,
                        **{field: np.stack([row[field] for row in records]) for field in fields})
    save(completed, {"n": len(identifiers), "available": sum(bool(row["available"]) for row in records),
                    "feature_sha256": digest(output/f"{split}.npz"), "encoder_sha256": WEIGHT_SHA256,
                    "code_sha256": digest(Path(__file__)), "target_labels_read": False})


def main() -> None:
    """保存固定模型来源与数值环境；开发默认不读取旧calibration和test。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["train", "selection"])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model, kwargs = build_model()
    if args.smoke or args.verify:
        path = MODEL / "example/zh.mp3"
        waveform, duration, valid = decode(path)
        if not valid:
            raise ValueError("官方中文示例解码失败")
        values, query, emo, text = encode(model, kwargs, [waveform])
        if not np.isfinite(values).all() or not np.isfinite(emo).all():
            raise ValueError("语音状态包含非有限值")
        report = {"strict_weights": True, "parameters": sum(p.numel() for p in model.parameters()),
                  "duration": duration, "pooled_shape": list(values.shape), "query_shape": list(query.shape),
                  "emotion_logprobs_shape": list(emo.shape), "transcript": text, "valid": valid}
        if args.verify:
            repeated = encode(model,kwargs,[waveform])
            batched = encode(model,kwargs,[waveform,waveform[:max(1600,len(waveform)//2)]])
            report.update(dither=0., repeat_pooled_max_error=float(np.abs(values-repeated[0]).max()),
                batch_pooled_max_error=float(np.abs(values-batched[0][:1]).max()),
                batch_query_max_error=float(np.abs(query-batched[1][:1]).max()))
            if report["repeat_pooled_max_error"] > 1e-6 or report["batch_pooled_max_error"] > 1e-3:
                raise ValueError("重放或批次长度一致性不满足数值容差")
        save(OUTPUT/"sensevoice_deterministic_verification.json", report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return
    for split in args.splits:
        if split not in ("train", "selection"):
            raise ValueError("开发编码阶段只允许train和selection")
        acquire_split(model, kwargs, split, args.batch)
    manifest = OUTPUT/"sensevoice_deterministic_features/manifest.json"
    if not manifest.exists():
        save(manifest, {"repo": "FunAudioLLM/SenseVoiceSmall", "revision": REVISION,
            "weight_sha256": WEIGHT_SHA256, "funasr": version("funasr"), "torch": torch.__version__,
            "numpy": np.__version__, "frozen": True, "strict_weights": True,
            "max_seconds": 10, "language_query": "zh", "precision": "fp32",
            "frontend_dither": 0., "reason": "deterministic inference independent of prior batches and resume position",
            "pooled": "mean+population_std of 512D states excluding four query tokens",
            "query": "four 512D query states; 4x5 native emotion logprobs stored without sentiment remapping",
            "attribution": "Alibaba SenseVoiceSmall; FunASR Model Open Source License Agreement 1.1",
            "license_url": "https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE",
            "not_claimed": "ASR text and emotion tags are model predictions, not gold annotations"})


if __name__ == "__main__":
    main()
