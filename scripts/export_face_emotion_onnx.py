#!/usr/bin/env python
"""把「Face」项目训练好的 3 类情绪模型导出为浏览器可用的 ONNX。

## 权重来源（已核对）
- 文件：`D:\\Develop\\AIC\\Face\\ResNet101_WholeData.pth`
  - 名字写着 ResNet101，**实际是 ResNet50**：stage 块数 3/4/6/3，`fc.weight=(3,2048)`，
    共 23,567,352 个参数（用 `torch.load` + 本脚本的严格加载双重核对）。
  - 训练脚本：`D:\\Develop\\AIC\\Face\\3.1.train_ResNet101.py`（`models.resnet50(...)`，`fc` 换成 3 类）。
- 预处理（必须与训练一致，否则精度无从谈起）：
  训练时 `transforms.Compose([transforms.ToTensor()])` —— **只做 ToTensor**：
  RGB、HWC→CHW、除以 255，**没有 ImageNet 归一化、没有 resize/crop**。
  因此这里做同样的处理：RGB 通道序 + `x/255`。
- 类别顺序：`happy(0) / neutral(1) / sad(2)`。
  依据：`1.1.split_datasets.py` 先 `classes_name_list.sort()` 再建子目录，
  `3.1.train_ResNet101.py` 用 `os.listdir()` 得到的顺序即目录创建顺序；
  用户已确认这三类。

## 为什么手写 ResNet 而不 import torchvision
当前 venv（`D:\\Develop\\AIC\\.venv`）有 torch 2.11 + onnx + onnxruntime + PIL，但**没有 torchvision**。
本脚本只用已装好的依赖，避免为了导出一次而装包；手写的结构靠 `load_state_dict(strict=True)`
保证与 torchvision 的实现逐张量一致（键名、形状、BN 统计量全部对齐才算通过）。

## 用法
    & "D:\\Develop\\AIC\\.venv\\Scripts\\python.exe" scripts/export_face_emotion_onnx.py
    # 拿到数据集之后可做输入尺寸扫描 + 混淆矩阵（用导出的 ONNX 本身，验证的是最终产物）
    & ... scripts/export_face_emotion_onnx.py --eval-dir "D:\\data\\Face\\testing"

输出：`client/public/models/face_emotion_3cls.onnx`（Vite 的 public/ 直接按 `/models/...` 提供）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = Path(r"D:\Develop\AIC\Face\ResNet101_WholeData.pth")
DEFAULT_OUT = APP_ROOT / "client" / "public" / "models" / "face_emotion_3cls.onnx"

# 类别顺序（模型输出索引 → 标签）。
#
# ⚠️ 训练脚本用的是 `classes = os.listdir(data_dir)`，**不是 sorted()**，
# 所以这个顺序推不出来，只能实测。2026-10-10 用 --eval-dir 扫全部 6 种排列确认：
# 真实顺序是 neutral(0) / happy(1) / sad(2)（validation n=1615 得 95.2%；
# 用 happy, neutral, sad 只有 40.6%）。
CLASS_LABELS = ["neutral", "happy", "sad"]

# 输入边长。数据集图片原生 = 96×96，训练脚本没有 resize（只有 ToTensor），
# 因此模型只在 96×96 上训过。实测 96px 95.2% vs 224px 40.6%。
DEFAULT_INPUT_SIZE = 96


# ────────────────────────────────────────────────────────────────────────────
# 1. ResNet（与 torchvision 的键名/形状逐一对齐）
# ────────────────────────────────────────────────────────────────────────────

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, layers: list[int], num_classes: int = 3):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )
        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * Bottleneck.expansion
        layers += [Bottleneck(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_and_load(weights_path: Path) -> tuple[ResNet, dict]:
    """构建 ResNet50 并**严格**加载权重；不一致就直接失败（不静默跳过）。"""
    sd = torch.load(weights_path, map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        # train_AI.py 存的是 checkpoint dict；3.1 存的是裸 state_dict
        sd = sd["model_state_dict"]

    model = ResNet([3, 4, 6, 3], num_classes=len(CLASS_LABELS))
    keys = set(model.state_dict().keys())
    missing = sorted(keys - set(sd.keys()))
    unexpected = sorted(set(sd.keys()) - keys)
    if missing or unexpected:
        raise SystemExit(
            "权重与结构不匹配，已中止。\n"
            f"  缺失 {len(missing)}: {missing[:5]}\n"
            f"  多余 {len(unexpected)}: {unexpected[:5]}"
        )
    model.load_state_dict(sd, strict=True)
    model.eval()

    report = {
        "tensors": len(sd),
        "params": int(sum(v.numel() for v in sd.values() if hasattr(v, "numel"))),
        "fc_weight_shape": list(sd["fc.weight"].shape),
        "layer_blocks": {
            name: len({k.split(".")[1] for k in sd if k.startswith(name + ".")})
            for name in ("layer1", "layer2", "layer3", "layer4")
        },
    }
    return model, report


# ────────────────────────────────────────────────────────────────────────────
# 2. 导出 ONNX + 数值对齐校验
# ────────────────────────────────────────────────────────────────────────────

def export_onnx(model: ResNet, out_path: Path, input_size: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, input_size, input_size)
    kwargs = dict(
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        # H/W 动态：训练时没有 resize，真实输入尺寸待数据集核对，先不把尺寸焊死
        dynamic_axes={"input": {2: "height", 3: "width"}, "logits": {0: "batch"}},
    )
    try:
        torch.onnx.export(model, dummy, str(out_path), dynamo=False, **kwargs)
    except TypeError:
        # 老/新版本没有 dynamo 形参时的回退
        torch.onnx.export(model, dummy, str(out_path), **kwargs)
    print(f"[导出] {out_path}  ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")


def verify_parity(model: ResNet, onnx_path: Path, sizes: tuple[int, ...] = (224, 64)) -> dict:
    """torch 与 ONNX 在同一输入上的输出必须一致（证明导出没有改变计算）。"""
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(str(onnx_path))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    print(f"[ONNX] 输入名={input_name} 形状={sess.get_inputs()[0].shape} 输出名={sess.get_outputs()[0].name}")

    result = {}
    torch.manual_seed(42)
    for size in sizes:
        x = torch.randn(1, 3, size, size)
        with torch.no_grad():
            ref = model(x).numpy()
        got = sess.run(None, {input_name: x.numpy()})[0]
        diff = float(np.max(np.abs(ref - got)))
        result[int(size)] = diff
        print(f"[对齐] 输入 {size}×{size}: torch 与 onnx 最大绝对差 = {diff:.3e}")
        if diff > 1e-4:
            raise SystemExit(f"ONNX 导出结果与 torch 不一致（{size}px，差 {diff}）")
    return result


# ────────────────────────────────────────────────────────────────────────────
# 3. 可选：用导出的 ONNX 在真实数据集上扫输入尺寸 + 混淆矩阵
# ────────────────────────────────────────────────────────────────────────────

def preprocess(img_path: Path, size: int) -> np.ndarray:
    """与前端 `packFaceImage` 保持同一口径：RGB、正方形 resize、x/255、NCHW。"""
    from PIL import Image

    with Image.open(img_path) as im:
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))[None, ...].copy()


def eval_sizes(onnx_path: Path, eval_dir: Path, sizes: list[int], limit_per_class: int) -> None:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    classes = sorted([d.name for d in eval_dir.iterdir() if d.is_dir()])
    print(f"[数据集] {eval_dir}  类别={classes}（模型标签顺序={CLASS_LABELS}）")
    if len(classes) != len(CLASS_LABELS):
        print(f"  ⚠️ 类别数 {len(classes)} 与模型 {len(CLASS_LABELS)} 不一致，标签映射需要人工确认")

    # ⚠️ 准确率必须**按类别名**对齐，不能用 confusion[i][i]：
    # 数据集目录是 sorted() 排的，而模型索引顺序来自训练的 os.listdir()，两者不一定同序。
    label_index = {name: i for i, name in enumerate(CLASS_LABELS)}

    best: tuple[float, int] | None = None
    for size in sizes:
        confusion = np.zeros((len(classes), len(CLASS_LABELS)), dtype=np.int64)
        for ci, cname in enumerate(classes):
            files = sorted([f for f in (eval_dir / cname).iterdir() if f.is_file()])[:limit_per_class]
            for f in files:
                x = preprocess(f, size)
                out = sess.run(None, {input_name: x})[0][0]
                confusion[ci, int(np.argmax(out))] += 1
        correct = int(
            sum(confusion[ci, label_index[cname]] for ci, cname in enumerate(classes) if cname in label_index)
        )
        total = int(confusion.sum())
        acc = correct / total if total else 0.0
        print(f"\n[尺寸 {size}×{size}] 总体准确率 {acc * 100:.1f}%  (n={total})")
        header = "        " + "".join(f"{lbl:>9}" for lbl in CLASS_LABELS)
        print(header)
        for ci, cname in enumerate(classes):
            row = "".join(f"{int(confusion[ci, j]):>9}" for j in range(len(CLASS_LABELS)))
            print(f"{cname:>8}{row}")
        if best is None or acc > best[0]:
            best = (acc, size)
    if best:
        print(f"\n结论：本数据集上最佳输入尺寸 = {best[1]}×{best[1]}（{best[0] * 100:.1f}%）"
              f" → 把 client 的 FACE_EMOTION_INPUT_SIZE 改成 {best[1]}")


def main() -> int:
    # Windows 控制台默认 GBK，日志里的箭头/符号会直接把脚本崩掉（UnicodeEncodeError）
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - 环境不支持时不影响主流程
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE, help="导出时的示例输入边长（图里是动态维）")
    ap.add_argument("--eval-dir", type=Path, default=None, help="可选：类别子目录结构的数据集，用于扫尺寸/算准确率")
    ap.add_argument("--eval-sizes", type=str, default="48,96,128,160,224,256")
    ap.add_argument("--eval-limit", type=int, default=200, help="每类最多评估多少张")
    args = ap.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"找不到权重：{args.weights}")

    sha = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    print(f"[权重] {args.weights}\n  SHA-256 = {sha}\n  大小 = {args.weights.stat().st_size / 1024 / 1024:.1f} MB")

    model, report = build_and_load(args.weights)
    print("[结构] 严格加载通过：" + json.dumps(report, ensure_ascii=False))

    export_onnx(model, args.out, args.input_size)
    diffs = verify_parity(model, args.out)

    # 参考推理耗时（原生 CPU，仅用于排查"是不是慢到不能用"；浏览器 wasm 会明显更慢）
    x = torch.randn(1, 3, args.input_size, args.input_size)
    with torch.no_grad():
        model(x)
        t0 = time.perf_counter()
        for _ in range(5):
            model(x)
        ms = (time.perf_counter() - t0) / 5 * 1000
    print(f"[参考延迟] 原生 CPU {args.input_size}px: {ms:.1f} ms/次（浏览器 wasm 单线程会慢数倍）")
    print(f"[结论] 导出完成，torch↔onnx 最大差 = {max(diffs.values()):.3e}")

    if args.eval_dir:
        sizes = [int(s) for s in args.eval_sizes.split(",") if s.strip()]
        eval_sizes(args.out, args.eval_dir, sizes, args.eval_limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
