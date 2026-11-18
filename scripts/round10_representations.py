"""提取目标标签无关的辅助文字表征和表情网络倒数层特征。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "algorithm"))

import numpy as np
from experiments.round10_fusion import ROOT, OUTPUT, load_npz, save

WORKSPACE = ROOT.parent


def auxiliary(splits: list[str], output: Path) -> None:
    """冻结只经ECISA训练的编码器；任何CH-SIMS标签都未进入该权重。"""
    import torch
    from transformers import AutoModel, AutoTokenizer
    from experiments.round10_text import ENCODER
    checkpoint = OUTPUT / "text" / "auxiliary_only.pt"
    identity = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model = AutoModel.from_pretrained(ENCODER, local_files_only=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict({name.removeprefix("encoder."): value for name, value in state.items() if name.startswith("encoder.")})
    model = model.cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(ENCODER, local_files_only=True)
    output.mkdir(parents=True, exist_ok=True)
    for split in splits:
        target = output / f"{split}.npz"
        if target.exists():
            raise ValueError("拒绝覆盖编码工件")
        data = load_npz(ROOT / "round5" / "text_features" / f"{split}.npz")
        cls, mean = [], []
        texts = data["raw_text"].astype(str).tolist()
        with torch.inference_mode():
            for start in range(0, len(texts), 64):
                row = tokenizer(texts[start:start+64], padding=True, truncation=True, max_length=96, return_tensors="pt").to("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = model(**row).last_hidden_state.float()
                mask = row["attention_mask"].unsqueeze(-1)
                cls.append(hidden[:, 0].cpu().numpy())
                mean.append(((hidden * mask).sum(1) / mask.sum(1)).cpu().numpy())
        np.savez_compressed(target, aux_cls=np.concatenate(cls), aux_mean=np.concatenate(mean),
                            sample_ids=data["sample_ids"], encoder_sha256=np.array(identity))
        print(json.dumps({"stage": "auxiliary_features", "split": split, "n": len(texts)}), flush=True)
    save(output / "manifest.json", {"checkpoint": str(checkpoint), "sha256": identity,
         "supervised_training": "ECISA official training only; no target labels", "max_length": 96})


def deep_face(splits: list[str], output: Path) -> None:
    """沿用相同五帧与检测器，只把八类输出扩展为1280维表情表示。"""
    import cv2
    import onnx
    import onnxruntime as ort
    original = WORKSPACE / "_fer_candidates" / "hsemotion_enet_b0_8_best_vgaf.onnx"
    detector = WORKSPACE / "_model_cache" / "round5" / "yunet.onnx"
    output.mkdir(parents=True, exist_ok=True)
    embedding_model = output / "hsemotion_embedding.onnx"
    graph = onnx.load(original)
    flatten = next(node for node in reversed(graph.graph.node) if node.op_type == "Gemm").input[0]
    graph.graph.output.append(onnx.helper.make_tensor_value_info(flatten, onnx.TensorProto.FLOAT, [None, 1280]))
    onnx.checker.check_model(graph)
    onnx.save(graph, embedding_model)
    cache = output / "samples"
    cache.mkdir(exist_ok=True)
    binding = hashlib.sha256(original.read_bytes() + detector.read_bytes() + Path(__file__).read_bytes()).hexdigest()
    local = threading.local()
    cv2.setNumThreads(1)
    media = WORKSPACE / "data" / "public" / "ch_sims_v2" / "supervised"

    def process(identifier: str) -> tuple[np.ndarray, float]:
        """每线程持有独立会话，缓存与模型及源码绑定。"""
        target = cache / f"{identifier}.npz"
        if target.exists():
            saved = load_npz(target)
            if str(saved["binding"]) != binding:
                raise ValueError("人脸特征缓存绑定不一致")
            return saved["embedding"], float(saved["coverage"])
        if not hasattr(local, "detector"):
            local.detector = cv2.FaceDetectorYN.create(str(detector), "", (320, 320), .8, .3, 5000)
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            local.model = ort.InferenceSession(str(embedding_model), options, providers=["CPUExecutionProvider"])
        path = media / f"{identifier}.mp4"
        cap = cv2.VideoCapture(str(path))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        crops = []
        for fraction in (.1, .3, .5, .7, .9):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(count - 1, round(count * fraction))))
            ok, frame = cap.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            scaled = cv2.resize(frame, (320, max(32, round(height * 320 / width))))
            local.detector.setInputSize((scaled.shape[1], scaled.shape[0]))
            _, faces = local.detector.detect(scaled)
            if faces is None or not len(faces):
                continue
            face = max(faces, key=lambda row: row[2] * row[3])
            x, y, fw, fh = face[:4]
            sx, sy = width / scaled.shape[1], height / scaled.shape[0]
            left, top = max(0, int((x-.1*fw)*sx)), max(0, int((y-.1*fh)*sy))
            right, bottom = min(width, int((x+1.1*fw)*sx)), min(height, int((y+1.1*fh)*sy))
            if right <= left or bottom <= top:
                continue
            crop = cv2.cvtColor(cv2.resize(frame[top:bottom, left:right], (224, 224)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            crop = ((crop - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)).transpose(2, 0, 1)
            crops.append(crop)
        cap.release()
        if crops:
            values = local.model.run([flatten], {local.model.get_inputs()[0].name: np.stack(crops)})[0]
            embedding = np.concatenate([values.mean(0), values.std(0)]).astype(np.float32)
        else:
            embedding = np.zeros(2560, np.float32)
        coverage = len(crops) / 5
        if not np.isfinite(embedding).all():
            raise ValueError("非有限表情表示")
        np.savez_compressed(target, embedding=embedding, coverage=np.array(coverage), binding=np.array(binding))
        return embedding, coverage

    for split in splits:
        target = output / f"{split}.npz"
        if target.exists():
            raise ValueError("拒绝覆盖编码工件")
        data = load_npz(ROOT / "round5" / "face_features" / f"{split}.npz")
        results = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for index, item in enumerate(pool.map(process, data["sample_ids"].astype(str).tolist())):
                results.append(item)
                if (index + 1) % 200 == 0:
                    print(json.dumps({"stage": "deep_face", "split": split, "n": index+1, "total": len(data["labels"])}), flush=True)
        np.savez_compressed(target, deep_face=np.stack([row[0] for row in results]), coverage=np.array([row[1] for row in results]), sample_ids=data["sample_ids"])
    save(output / "manifest.json", {"binding": binding, "backbone": "frozen HSEmotion EfficientNet-B0 penultimate 1280",
         "pooling": "five-frame mean and standard deviation", "speaker_verified": False,
         "target_label_training": False, "frames": [.1, .3, .5, .7, .9]})


def main() -> None:
    """只处理显式指定的划分，测试可延后到选型锁定后提取。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("auxiliary", "deep_face"))
    parser.add_argument("--splits", nargs="+", default=["train", "selection"])
    args = parser.parse_args()
    (auxiliary if args.kind == "auxiliary" else deep_face)(args.splits, OUTPUT / f"{args.kind}_features")


if __name__ == "__main__":
    main()
