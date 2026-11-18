"""表情模型精度体检：输入尺寸扫描 + 类别排列扫描 + 混淆矩阵。

为什么需要它
------------
2026-10-10 用这个脚本查出两个**部署侧**常量错误，把 validation 准确率从 40.6% 修到 95.2%：

1. **输入尺寸**：数据集里所有图片原生就是 96×96，而训练脚本
   `Face/3.1.train_ResNet101.py` 的 transform 只有 `Compose([ToTensor()])`（不 resize），
   所以模型只看过 96×96。前端却用 224，等于把 96 拉大再喂。
2. **类别顺序**：训练脚本用 `classes = os.listdir(data_dir)` 分配索引 ——
   **`os.listdir` 既不排序也不保证等于创建顺序**。实测当时顺序是 `neutral, happy, sad`；
   部署侧按目录名 `sort()` 假设成 `happy, neutral, sad`，于是 happy/neutral 两个名字整体对调，
   模型其实全预测对了、名字全贴错了。

交叉验证：`Face/ResNet101_best9517.pth` 的名字恰为 **95.17%**，
与 96px + 正确映射下独立测得的 95.2% 吻合。

⚠️ 重新训练 / 重新导出后**必须重跑本脚本**：`os.listdir` 的顺序换机器就可能变，
不要沿用 `client/src/services/faceEmotionClassifier.ts` 里现有的常量。

用法
----
    & "D:\\Develop\\AIC\\.venv\\Scripts\\python.exe" scripts/face_emotion_accuracy_check.py
    # 只扫尺寸（快）：
    ... --sizes 96,224 --limit 100
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = APP_ROOT / "client" / "public" / "models" / "face_emotion_3cls.onnx"
DEFAULT_DATA = Path(r"D:\Develop\AIC\数据集\数据集\Face")
# 与 client/src/services/faceEmotionClassifier.ts 的 FACE_EMOTION_LABELS 保持一致
FRONTEND_LABELS = ["neutral", "happy", "sad"]


def preprocess(path: Path, size: int) -> np.ndarray:
    """与前端 `packFaceImage` 同口径：RGB、双线性 resize、x/255、NCHW。"""
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))[None, ...].copy()


def load_split(root: Path, limit_per_class: int | None = None):
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    samples = []
    for cname in classes:
        files = sorted([f for f in (root / cname).iterdir() if f.is_file()])
        if limit_per_class:
            files = files[:limit_per_class]
        samples += [(f, cname) for f in files]
    return samples, classes


def infer(sess, input_name: str, samples, size: int) -> np.ndarray:
    preds = np.empty(len(samples), dtype=np.int64)
    for i, (path, _) in enumerate(samples):
        out = sess.run(None, {input_name: preprocess(path, size)})[0][0]
        preds[i] = int(np.argmax(out))
    return preds


def acc_for_perm(preds: np.ndarray, truth: list[str], perm: tuple[str, ...]) -> float:
    """把模型输出索引 i 解释成 `perm[i]`，再与真标签比较。"""
    return sum(1 for p, t in zip(preds, truth) if perm[p] == t) / len(truth)


def dump_rgba(split_root: Path, limit: int, out_dir: Path) -> int:
    """把图像导出成裸 RGBA（`*.rgba` + `images.json`），供前端 TS 代码做端到端精度复核。

    为什么要走这条路：Node 侧没有任何图像解码库（无 sharp/jimp/pngjs/jpeg-js），
    而 `packFaceImage` 要的正好是 RGBA 像素。用 Python 解码后落地成裸像素，
    就能让 `scripts/verify_face_emotion_accuracy.ts` 直接喂**真正的前端代码**
    （`FaceEmotionClassifier.classify` → `packFaceImage` → `toReading`），
    从而验证"常量 + 索引→标签映射"这一环，而不只是验证 ONNX 本身。

    导出的是**原生尺寸**（本数据集为 96×96）：前端生产路径里
    `FacePatchExtractor` 已把画面裁成 `FACE_EMOTION_INPUT_SIZE` 的正方形，
    所以裸像素尺寸 = 模型输入尺寸，`packFaceImage` 不再走缩放分支。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    samples, classes = load_split(split_root, limit or None)
    manifest = []
    for i, (path, label) in enumerate(samples):
        with Image.open(path) as im:
            rgba = im.convert("RGBA")
            w, h = rgba.size
            data = rgba.tobytes()
        name = f"{i:06d}.rgba"
        (out_dir / name).write_bytes(data)
        manifest.append({"file": name, "width": w, "height": h, "label": label, "source": path.name})
    (out_dir / "images.json").write_text(
        json.dumps({"classes": classes, "images": manifest}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return len(manifest)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="表情模型精度体检")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--sizes", type=str, default="64,96,128,160,224,256")
    ap.add_argument("--limit", type=int, default=150, help="尺寸扫描阶段每类取样上限；0=全量")
    ap.add_argument("--dump-rgba", type=Path, default=None,
                    help="把 validation 图像导出成裸 RGBA（供 verify_face_emotion_accuracy.ts 复核前端代码）")
    ap.add_argument("--dump-split", type=str, default="validation")
    ap.add_argument("--dump-limit", type=int, default=200, help="每个类别导出多少张；0=全量")
    args = ap.parse_args()

    if args.dump_rgba:
        n = dump_rgba(args.data / args.dump_split, args.dump_limit, args.dump_rgba)
        print(f"[导出] {n} 张 {args.dump_split} 图像的裸 RGBA → {args.dump_rgba}")
        print(f"       接着跑：node \"server/node_modules/tsx/dist/cli.mjs\" "
              f"scripts/verify_face_emotion_accuracy.ts --data \"{args.dump_rgba}\"")
        return 0

    import onnxruntime as ort

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    if not args.model.exists():
        raise SystemExit(f"模型不存在：{args.model}\n先跑 scripts/export_face_emotion_onnx.py 导出。")
    if not args.data.exists():
        raise SystemExit(f"数据集不存在：{args.data}")

    print(f"[模型] {args.model.name}  ({args.model.stat().st_size / 1048576:.1f} MB)")
    sess = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    limit = args.limit or None
    samples, classes = load_split(args.data / "validation", limit)
    truth = [c for _, c in samples]
    perms = list(itertools.permutations(classes))
    print(f"[数据] validation 类别={classes}  n={len(samples)}")
    print(f"[前端] 当前假定的标签顺序 = {FRONTEND_LABELS}")

    print("\n" + "=" * 78)
    print("阶段 1：输入尺寸扫描（按前端假定的标签顺序）")
    print("=" * 78)
    print(f"{'尺寸':>6} {'准确率':>9} {'耗时/张':>10}")
    cache: dict[int, np.ndarray] = {}
    for size in sizes:
        t0 = time.time()
        preds = infer(sess, input_name, samples, size)
        dt = (time.time() - t0) / len(samples) * 1000
        cache[size] = preds
        print(f"{size:>6} {acc_for_perm(preds, truth, tuple(FRONTEND_LABELS)) * 100:>8.1f}% {dt:>9.1f}ms")

    print("\n" + "=" * 78)
    print("阶段 2：类别排列扫描（模型输出索引 → 真实类别名）")
    print("=" * 78)
    for size in sizes:
        preds = cache[size]
        scored = sorted(((acc_for_perm(preds, truth, p), p) for p in perms), reverse=True)
        best_acc, best_perm = scored[0]
        front = acc_for_perm(preds, truth, tuple(FRONTEND_LABELS))
        tag = "✅ 与前端一致" if best_perm == tuple(FRONTEND_LABELS) else "❗与前端不一致"
        print(f"  {size:>4}px  最佳映射={best_perm} {best_acc * 100:>5.1f}%  "
              f"前端映射={front * 100:>5.1f}%  {tag}")

    # ── 阶段 3：最佳配置下的全量 validation + 混淆矩阵 ──
    best_size = max(sizes, key=lambda s: max(acc_for_perm(cache[s], truth, p) for p in perms))
    print("\n" + "=" * 78)
    print(f"阶段 3：全量 validation，尺寸={best_size}")
    print("=" * 78)
    full, full_classes = load_split(args.data / "validation", None)
    full_truth = [c for _, c in full]
    full_preds = infer(sess, input_name, full, best_size)
    scored = sorted(((acc_for_perm(full_preds, full_truth, p), p) for p in perms), reverse=True)
    for acc, perm in scored[:3]:
        print(f"  映射 {perm} → {acc * 100:.1f}%  (n={len(full)})")
    best_perm = scored[0][1]
    front_acc = acc_for_perm(full_preds, full_truth, tuple(FRONTEND_LABELS))
    print(f"\n  前端当前映射 {tuple(FRONTEND_LABELS)} → {front_acc * 100:.1f}%")
    print(f"\n  混淆矩阵（行=真实，列=预测；按最佳映射 {best_perm} 命名）")
    print("        " + "".join(f"{lbl:>9}" for lbl in best_perm))
    for cname in full_classes:
        row = [0] * len(best_perm)
        for i, t in enumerate(full_truth):
            if t == cname:
                row[full_preds[i]] += 1
        print(f"{cname:>8}" + "".join(f"{v:>9}" for v in row))

    # ── 阶段 4：testing（可能含模型认不出的类别）──
    test_root = args.data / "testing"
    if test_root.exists():
        print("\n" + "=" * 78)
        print("阶段 4：testing 集")
        print("=" * 78)
        t_samples, t_classes = load_split(test_root, limit or 150)
        t_truth = [c for _, c in t_samples]
        t_preds = infer(sess, input_name, t_samples, best_size)
        known = [c for c in t_classes if c in best_perm]
        n_known = sum(1 for t in t_truth if t in known)
        hit = sum(1 for p, t in zip(t_preds, t_truth) if t in known and best_perm[p] == t)
        print(f"  类别={t_classes}  n={len(t_samples)}"
              f"（可判 {n_known} 张，模型无此类别 {len(t_samples) - n_known} 张）")
        print(f"  在可判类别上 = {hit / max(1, n_known) * 100:.1f}%")
        print(f"  含不可判类别的总体 = {hit / len(t_samples) * 100:.1f}%"
              f"  ← 上限 {n_known / len(t_samples) * 100:.1f}%")
        unknown_idx = [i for i, t in enumerate(t_truth) if t not in known]
        if unknown_idx:
            print("  模型认不出的那些样本被归到了：")
            c = Counter(best_perm[t_preds[i]] for i in unknown_idx)
            for k, v in c.most_common():
                print(f"    {k:>8}: {v:>4}  ({v / sum(c.values()) * 100:.1f}%)")

    print("\n" + "=" * 78)
    print("结论")
    print("=" * 78)
    print(f"  · 最佳输入尺寸 = {best_size}  → 改 client 的 FACE_EMOTION_INPUT_SIZE")
    print(f"  · 最佳类别映射 = {best_perm}  → 改 client 的 FACE_EMOTION_LABELS")
    if tuple(FRONTEND_LABELS) != best_perm:
        print("  ❗前端当前映射与实测不一致，必须同步（否则准确率会掉一半以上）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
