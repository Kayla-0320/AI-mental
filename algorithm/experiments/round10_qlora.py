"""来源折内QLoRA文字适配与冻结语音/表情融合，保留匹配冻结对照。"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import time
from typing import Any

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, TaskType, PeftModel, get_peft_model, prepare_model_for_kbit_training
from safetensors import safe_open

from experiments import round10_speech_evidence as speech
from experiments import round10_represented_fusion as represented
from experiments.round10_evidence import supervision
from experiments.round10_domain import bias_select
from experiments.round10_fusion import OUTPUT, folds, take, save, load_npz
from experiments.round7_decision_oof import metric

MODEL = Path(r"D:\ai\models\Qwen2.5-3B-Instruct-nf4")
FEATURE_CONFIG = {"pack": "qlora_numeric", "face_weight": 1., "components": 32}
represented.PACKS["qlora_numeric"] = ("aux_mean", "bge_cls", "face_summary", "audio", "visual", "deep_face", "speech", "speech_emotion")
CONFIG = {"rank": 8, "last_layers": 6, "adapter_lr": .0002, "head_lr": .001,
          "adapter_dropout": .05, "head_dropout": .15, "auxiliary_T5_weight": .3,
          "class_weight_power": .5, "weight_decay": .01, "effective_batch": 16,
          "micro_batch": 4, "max_tokens": 192, "epochs": [1, 2, 3],
          "frozen_prefix_layers": 30, "prefix_cache_dtype": "bfloat16", "canonical_position_ids": True,
          "rotary_embedding_dtype": "float32 matching prepared full encoder"}


def seed_all(seed: int = 42) -> None:
    """训练、打乱和初始化共用显式种子。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False


def tokenizer_and_tokens(data: dict[str, np.ndarray]) -> tuple[Any, dict[str, Tensor]]:
    """与冻结文字任务使用相同提示；不提供任何情绪答案。"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    prompts = [tokenizer.apply_chat_template([
        {"role": "system", "content": "你是中文情感分析助手。根据句子的整体表达区分负向、中性和正向。"},
        {"role": "user", "content": f"句子：{text}\n只回答“负向”“中性”或“正向”。"},
    ], tokenize=False, add_generation_prompt=True) for text in data["raw_text"].astype(str)]
    tokens = tokenizer(prompts, max_length=CONFIG["max_tokens"], truncation=True,
                       padding="max_length", return_tensors="pt")
    return tokenizer, dict(tokens)


def load_encoder(adapter: bool, adapter_path: Path | None = None) -> nn.Module:
    """冻结NF4基础权重，仅训练最后六层的低秩Q/V适配器。"""
    encoder = AutoModel.from_pretrained(MODEL, local_files_only=True, dtype=torch.bfloat16,
                                      device_map={"": "cuda:0"}, attn_implementation="sdpa")
    encoder.config.use_cache = False
    encoder = prepare_model_for_kbit_training(encoder, use_gradient_checkpointing=False)
    if adapter_path is not None:
        encoder = PeftModel.from_pretrained(encoder, adapter_path, is_trainable=False)
    elif adapter:
        layers = list(range(encoder.config.num_hidden_layers-CONFIG["last_layers"], encoder.config.num_hidden_layers))
        # 裸Qwen2Model的模块名从layers开头，避免PEFT层号匹配器要求前缀而漏选。
        targets = rf"(?:.*\.)?layers\.(?:{'|'.join(str(i) for i in layers)})\.self_attn\.(?:q_proj|v_proj)"
        encoder = get_peft_model(encoder, LoraConfig(r=CONFIG["rank"], lora_alpha=2*CONFIG["rank"],
            lora_dropout=CONFIG["adapter_dropout"], target_modules=targets,
            bias="none", task_type=TaskType.FEATURE_EXTRACTION))
    return encoder


def slice_tokens(tokens: dict[str, Tensor], indices: np.ndarray | Tensor) -> dict[str, Tensor]:
    """按当前微批裁掉左侧公共padding，不改变句子结尾与注意力掩码。"""
    part = {name: values[indices].cuda() for name, values in tokens.items()}
    width = int(part["attention_mask"].sum(1).max())
    part = {name: values[:, -width:] for name, values in part.items()}
    part["position_ids"] = (part["attention_mask"].cumsum(1)-1).clamp_min(0)
    return part


def cache_frozen(data: dict[str, np.ndarray], tokens: dict[str, Tensor], output: Path) -> tuple[np.ndarray, list[Tensor]]:
    """一次提取无标签冻结前缀和匹配完整输出，双方在同一点转换为BF16。"""
    target = output / "frozen_text.npz"
    prefix_path = output / "frozen_prefix.pt"
    if target.exists() and prefix_path.exists():
        cached = load_npz(target)
        if not np.array_equal(cached["sample_ids"], data["sample_ids"]):
            raise ValueError("冻结对照缓存ID不一致")
        return cached["last"], torch.load(prefix_path, weights_only=True)["prefix"]
    encoder = load_encoder(False).eval()
    values, prefix = [], []
    current_mask = None

    def intercept(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        states = args[0].to(torch.bfloat16)
        for hidden, mask in zip(states, current_mask):
            prefix.append(hidden[mask.bool()].clone().cpu())
        return (states, *args[1:]), kwargs

    handle = encoder.layers[CONFIG["frozen_prefix_layers"]].register_forward_pre_hook(intercept, with_kwargs=True)
    with torch.inference_mode():
        for start in range(0, len(data["sample_ids"]), 16):
            row = slice_tokens(tokens, np.arange(start, min(start+16, len(data["sample_ids"]))))
            current_mask = row["attention_mask"]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                values.append(encoder(**row, use_cache=False).last_hidden_state[:, -1].float().cpu().numpy())
            if (start//16+1) % 40 == 0:
                print(json.dumps({"frozen_prefix_cached": min(start+16, len(data["sample_ids"]))}), flush=True)
    handle.remove()
    hidden = np.concatenate(values)
    np.savez_compressed(target, last=hidden, sample_ids=data["sample_ids"])
    torch.save({"prefix": prefix, "sample_ids": data["sample_ids"].tolist(), "config": CONFIG}, prefix_path)
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    return hidden, prefix


def slice_prefix(prefix: list[Tensor], indices: np.ndarray) -> tuple[Tensor, dict[str, Tensor]]:
    """固定前缀状态仅按观测序列左侧补零，位置编号由有效token决定。"""
    selected = [prefix[int(index)] for index in indices]
    width = max(len(value) for value in selected)
    values = torch.stack([F.pad(value, (0, 0, width-len(value), 0)) for value in selected]).cuda()
    attention = torch.zeros(len(selected), width, dtype=torch.long, device="cuda")
    for i, value in enumerate(selected):
        attention[i, -len(value):] = 1
    return values, {"attention_mask": attention, "position_ids": (attention.cumsum(1)-1).clamp_min(0)}


def full_rotary_precision(module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
    """完整kbit模型的embedding为FP32，缓存路径必须保持相同RoPE精度。"""
    return (args[0].float(), *args[1:])


class AdaptedFusion(nn.Module):
    """文字与数值两路融合，独立文字头使用官方T五档而非整体M标签。"""

    def __init__(self, dimensions: int, adapter: bool, adapter_path: Path | None = None) -> None:
        super().__init__()
        self.encoder = load_encoder(True, adapter_path) if adapter else None
        if adapter:
            base = self.encoder.get_base_model()
            # 普通list保留冻结层用于按原始层号序列化，训练前向只执行后六层。
            self.__dict__["_prefix_layers"] = list(base.layers[:CONFIG["frozen_prefix_layers"]])
            base.layers = nn.ModuleList(list(base.layers[CONFIG["frozen_prefix_layers"]:]))
            base.rotary_emb.register_forward_pre_hook(full_rotary_precision)
        self.text = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, 128), nn.GELU(), nn.Dropout(CONFIG["head_dropout"]))
        self.numeric = nn.Sequential(nn.LayerNorm(dimensions), nn.Linear(dimensions, 128), nn.GELU(), nn.Dropout(CONFIG["head_dropout"]))
        self.overall = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Dropout(CONFIG["head_dropout"]), nn.Linear(64, 3))
        self.text_only = nn.Linear(128, 5)
        self.text.cuda()
        self.numeric.cuda()
        self.overall.cuda()
        self.text_only.cuda()

    def forward(self, numeric: Tensor, tokens: dict[str, Tensor] | None = None,
                hidden: Tensor | None = None, prefix: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """真实监督标签均不作为前向输入。"""
        if self.encoder is not None:
            if tokens is None or prefix is None:
                raise ValueError("适配编码器需要真实文字的固定前缀及注意力掩码")
            hidden = self.encoder(inputs_embeds=prefix, **tokens, use_cache=False).last_hidden_state[:, -1]
        if hidden is None:
            raise ValueError("冻结对照需要匹配的真实文字表征")
        text = self.text(hidden)
        return self.overall(torch.cat([text, self.numeric(numeric)], 1)), self.text_only(text)

    def save_adapter(self, target: Path) -> None:
        """保存前恢复原层号，保证标准PEFT载入器可直接加载适配器。"""
        if self.encoder is None:
            raise ValueError("冻结对照没有适配器")
        base = self.encoder.get_base_model()
        tail = base.layers
        try:
            base.layers = nn.ModuleList(self.__dict__["_prefix_layers"]+list(tail))
            self.encoder.save_pretrained(target)
        finally:
            base.layers = tail


def inference(model: AdaptedFusion, numeric: np.ndarray, tokens: dict[str, Tensor],
              frozen: np.ndarray, indices: np.ndarray, prefix: list[Tensor]) -> np.ndarray:
    """预测只读取观测特征，未读取M或T真实标签。"""
    model.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(indices), 8):
            selected = indices[start:start+8]
            cached, row = slice_prefix(prefix, selected) if model.encoder is not None else (None, None)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(torch.tensor(numeric[selected], dtype=torch.float32, device="cuda"), row,
                                  torch.tensor(frozen[selected], device="cuda"), cached)
            values.append(logits.float().cpu().numpy())
    return np.concatenate(values)


def train_fold(data: dict[str, np.ndarray], tokens: dict[str, Tensor], frozen: np.ndarray,
               tr: np.ndarray, va: np.ndarray, fold: int, adapter: bool, output: Path,
               prefix: list[Tensor], epochs: int = 3, seed: int = 42) -> dict[int, np.ndarray]:
    """只在来源训练折拟合归一化与监督；保存每个预注册轮次，不用留出折早停。"""
    seed_all(seed)
    training = take(data, tr)
    _, scaler = represented.transform(training, FEATURE_CONFIG)
    numeric, _ = represented.transform(data, FEATURE_CONFIG, scaler)
    numeric = numeric.astype(np.float32)
    _, continuous = supervision(training)
    text_targets = np.digitize(continuous[:, 1], [-.5, -.1, .1, .5])
    m_labels = torch.tensor(data["labels"][tr], dtype=torch.long, device="cuda")
    t_labels = torch.tensor(text_targets, dtype=torch.long, device="cuda")
    m_weights = (len(tr)/(3*torch.bincount(m_labels, minlength=3).float().clamp_min(1))).pow(CONFIG["class_weight_power"])
    t_weights = (len(tr)/(5*torch.bincount(t_labels, minlength=5).float().clamp_min(1))).pow(CONFIG["class_weight_power"])
    model = AdaptedFusion(numeric.shape[1], adapter)
    trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    if adapter and any("lora_" not in name for name, _ in trainable if name.startswith("encoder.")):
        raise ValueError("基础编码器出现非LoRA可训练权重")
    optimizer = torch.optim.AdamW([
        {"params": [p for name, p in trainable if name.startswith("encoder.")], "lr": CONFIG["adapter_lr"]},
        {"params": [p for name, p in trainable if not name.startswith("encoder.")], "lr": CONFIG["head_lr"]},
    ], weight_decay=CONFIG["weight_decay"])
    steps = math.ceil(len(tr)/CONFIG["effective_batch"])*epochs

    def schedule(step: int) -> float:
        warmup = max(1, int(.1*steps))
        if step < warmup:
            return (step+1)/warmup
        progress = (step-warmup)/max(1, steps-warmup)
        return .1+.9*.5*(1+math.cos(math.pi*progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    predictions = {}
    rng = np.random.default_rng(seed)
    start_time = time.monotonic()
    kind = "qlora" if adapter else "frozen"
    target_dir = output / f"{kind}_fold{fold}"
    target_dir.mkdir(parents=True, exist_ok=True)
    with (target_dir / "scaler.pkl").open("wb") as handle:
        pickle.dump(scaler, handle)
    for epoch in range(1, epochs+1):
        model.train()
        order = rng.permutation(len(tr))
        losses = []
        for step, start in enumerate(range(0, len(tr), CONFIG["effective_batch"])):
            chunk = order[start:start+CONFIG["effective_batch"]]
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            for micro in range(0, len(chunk), CONFIG["micro_batch"]):
                local = chunk[micro:micro+CONFIG["micro_batch"]]
                actual = tr[local]
                cached, row = slice_prefix(prefix, actual) if adapter else (None, None)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    m, t = model(torch.tensor(numeric[actual], device="cuda"), row, torch.tensor(frozen[actual], device="cuda"), cached)
                loss = F.cross_entropy(m.float(), m_labels[local], weight=m_weights)
                loss += CONFIG["auxiliary_T5_weight"]*F.cross_entropy(t.float(), t_labels[local], weight=t_weights)
                loss *= len(local)/len(chunk)
                if not torch.isfinite(loss):
                    raise ValueError("训练损失非有限")
                loss.backward()
                total += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.)
            if not torch.isfinite(norm):
                raise ValueError("梯度非有限")
            optimizer.step()
            scheduler.step()
            losses.append(total)
            if (step+1) % 25 == 0:
                print(json.dumps({"kind": kind, "fold": fold, "epoch": epoch, "step": step+1,
                    "steps": math.ceil(len(tr)/CONFIG["effective_batch"]), "loss": float(np.mean(losses[-25:])),
                    "seconds": round(time.monotonic()-start_time, 1)}), flush=True)
        predictions[epoch] = inference(model, numeric, tokens, frozen, va, prefix) if len(va) else np.empty((0, 3))
        if adapter:
            model.save_adapter(target_dir / f"epoch{epoch}_adapter")
        head = {name: value.detach().cpu() for name, value in model.state_dict().items() if not name.startswith("encoder.")}
        torch.save({"state_dict": head, "numeric_dimensions": numeric.shape[1], "seed": seed,
                    "training_sample_ids": data["sample_ids"][tr].tolist(), "training_groups": np.unique(data["source_groups"][tr]).tolist(),
                    "epoch": epoch, "config": CONFIG}, target_dir / f"epoch{epoch}_head.pt")
        print(json.dumps({"kind": kind, "fold": fold, "epoch_completed": epoch,
            "validation_argmax": metric(data["labels"][va], predictions[epoch].argmax(1))["macro_f1"] if len(va) else None,
            "seconds": round(time.monotonic()-start_time, 1), "peak_gpu_mb": torch.cuda.max_memory_allocated()/1024**2}), flush=True)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return predictions


def search(output: Path, smoke: bool = False) -> None:
    """冻结对照与适配模型共享同一来源折、监督及头部训练配方。"""
    if (output / "search_complete.json").exists():
        raise ValueError("拒绝覆盖QLoRA选型")
    if smoke and (output / "smoke_passed.json").exists():
        raise ValueError("拒绝覆盖已完成冒烟记录")
    data = speech.data_for(("train", "selection"))
    _, tokens = tokenizer_and_tokens(data)
    if smoke:
        indices = np.concatenate([np.flatnonzero(data["labels"] == k)[:24] for k in range(3)])
        data = take(data, indices)
        tokens = {name: value[indices] for name, value in tokens.items()}
    registration = {"config": CONFIG, "model": str(MODEL), "peft_version": "0.18.0",
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "m_target": "unchanged3classM", "text_aux_target": "officialT discretized5, neverM-as-T",
        "protocol": "4 source folds; all preprocessing and adapters fit training fold only; epochs1/2/3 global OOF selection",
        "frozen_control": "same heads, supervised targets, batch16/micro4, optimizer schedule; independently recomputed frozen states after kbit preparation",
        "smoke_only": smoke, "historical_exposure": "adaptive prior diagnostic; not fresh confirmation"}
    existing_path = output / "preregister.json"
    if existing_path.exists() and not smoke:
        if json.loads(existing_path.read_text("utf-8")) != registration:
            raise ValueError("未完成实验的协议或代码已变动，请使用新输出目录，禁止混用折结果")
    else:
        save(existing_path, registration)
    frozen, prefix = cache_frozen(data, tokens, output)
    if smoke:
        # 冒烟只检查真实反向、非有限值和显存，不报告为开发成绩。
        tr = np.concatenate([np.arange(k*24, k*24+18) for k in range(3)])
        va = np.setdiff1d(np.arange(72), tr)
        model = AdaptedFusion(4, True).eval()
        indices = np.arange(16)
        hidden, row = slice_prefix(prefix, indices)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            actual = model.encoder(inputs_embeds=hidden, **row, use_cache=False).last_hidden_state[:, -1].float().cpu().numpy()
        relative_error = float(np.sqrt(np.mean((actual-frozen[indices])**2))/np.sqrt(np.mean(frozen[indices]**2)))
        if relative_error > .001:
            raise ValueError(f"固定前缀前向与完整前向不一致: relative_RMSE={relative_error}")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        expected = train_fold(data, tokens, frozen, tr, va, 0, True, output, prefix, epochs=1)[1]
        checkpoint = output / "qlora_fold0"
        with safe_open(checkpoint / "epoch1_adapter" / "adapter_model.safetensors", framework="pt") as tensors:
            b_keys = [name for name in tensors.keys() if ".lora_B." in name]
            adapter_changed = max(float(tensors.get_tensor(name).abs().max()) for name in b_keys) > 0
        if not adapter_changed:
            raise ValueError("LoRA权重没有发生实际更新")
        saved = torch.load(checkpoint / "epoch1_head.pt", weights_only=True)
        replay_model = AdaptedFusion(saved["numeric_dimensions"], True, checkpoint / "epoch1_adapter")
        loaded = replay_model.load_state_dict(saved["state_dict"], strict=False)
        if loaded.unexpected_keys or any(not name.startswith("encoder.") for name in loaded.missing_keys):
            raise ValueError("回放头权重缺失或多余")
        with (checkpoint / "scaler.pkl").open("rb") as handle:
            scaler = pickle.load(handle)
        numeric, _ = represented.transform(data, FEATURE_CONFIG, scaler)
        actual = inference(replay_model, numeric.astype(np.float32), tokens, frozen, va, prefix)
        replay_error = float(np.max(abs(expected-actual)))
        if replay_error > 1e-6:
            raise ValueError(f"保存适配器回放不一致: {replay_error}")
        save(output / "smoke_passed.json", {"backward_finite": True, "adapter_weights_changed": adapter_changed,
             "serialized_replay_max_error": replay_error, "peak_gpu_mb": torch.cuda.max_memory_allocated()/1024**2})
        save(output / "prefix_equivalence.json", {"relative_rmse": relative_error, "n": 16, "tolerance": .001})
        return
    grid = [{"kind": kind, "epoch": epoch} for kind in ("frozen", "qlora") for epoch in CONFIG["epochs"]]
    logits = np.full((len(grid), len(data["labels"]), 3), np.nan)
    for fold, (tr, va) in enumerate(folds(data)):
        for adapter in (False, True):
            kind = "qlora" if adapter else "frozen"
            done = output / f"{kind}_fold{fold}" / "predictions.npz"
            if done.exists():
                cached = load_npz(done)
                if not np.array_equal(cached["validation_sample_ids"], data["sample_ids"][va]):
                    raise ValueError("缓存来源折ID不一致")
                values = {epoch: cached[f"epoch{epoch}"] for epoch in CONFIG["epochs"]}
            else:
                values = train_fold(data, tokens, frozen, tr, va, fold, adapter, output, prefix)
                np.savez_compressed(done, **{f"epoch{epoch}": value for epoch, value in values.items()},
                                    validation_sample_ids=data["sample_ids"][va], training_sample_ids=data["sample_ids"][tr])
            for i, config in enumerate(grid):
                if config["kind"] == kind:
                    logits[i, va] = values[config["epoch"]]
    results = [{"config": config, **bias_select(data, z/max(.1, float(z.std(1).mean()))),
                "scale": max(.1, float(z.std(1).mean()))} for config, z in zip(grid, logits)]
    winner = max(results, key=lambda row: (row["selection_score"], row["metrics"]["macro_f1"]))
    np.savez_compressed(output / "candidates_oof.npz", logits=logits, labels=data["labels"],
                        sample_ids=data["sample_ids"], source_groups=data["source_groups"])
    save(output / "search_complete.json", {"results": results})
    save(output / "selection_locked.json", winner)
    print(json.dumps({"winner": winner}), flush=True)


def main() -> None:
    """冒烟与完整开发实验隔离，不接触历史测试入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = OUTPUT / ("qlora_prefix_smoke_replay" if args.smoke else "qlora_prefix")
    output.mkdir(parents=True, exist_ok=True)
    search(output, args.smoke)


if __name__ == "__main__":
    main()
