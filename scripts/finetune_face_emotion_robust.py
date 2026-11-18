"""对三分类表情模型做**抗干扰微调**（nuisance-factor augmentation fine-tune）。

为什么需要它 —— 全部来自本机实测
--------------------------------
`scripts/face_fix_diagnosis.md` 记了完整证据，核心是三条：

1. **模型的"开心"判据极度脆弱，且失败模式正是"被判成悲伤"**：
   | 人脸在 96px patch 里上移 | 开心召回 | 被判成悲伤 |
   |---|---|---|
   | 0px（数据集口径） | 96.2% | 0.0% |
   | 2px（前端现状偏差） | 95.0% | 0.0% |
   | 4px | 52.5% | **35.0%** |
   | 6px | 11.2% | **76.2%** |
   同理：JPEG q30 → 30% 的开心被判成悲伤；分辨率降到 1/3 → 52.5% 掉到中性。

2. **类别与"干扰因素"混杂了**：数据集里 `sad` 的脸部占画面高度中位数是 **0.77**，
   而 `happy` 只有 **0.64**（用 MediaPipe FaceLandmarker 量的，见 `face_framing_audit.py`）。
   也就是说模型完全可以靠"脸裁得紧不紧"来猜类别 —— 这是捷径，不是表情。

3. **模型几乎只靠嘴判断"开心"**：抹掉眼/眉区，开心召回 96.2% → 97.5%（几乎无损）；
   抹掉嘴区，塌到 17.5%。所以任何破坏嘴部细节/位置的干扰都会直接改写结论。

修法：在原权重上做**小学习率**微调，训练时随机施加**平移 / 缩放 / 模糊 / 降采样 / JPEG 压缩 /
亮度对比度抖动 / 水平翻转**，逼模型不再依赖这些捷径。评测用现成的两个 harness：
`face_emotion_domain_shift_check.py`（鲁棒性）与 `verify_face_emotion_accuracy.ts`（前端一致性）。

⚠️ 不改变类别定义与顺序：`neutral(0) / happy(1) / sad(2)`，与
`client/src/services/faceEmotionClassifier.ts` 的 `FACE_EMOTION_LABELS` 一致。
微调后**必须重跑**类别排列扫描，确认这个顺序仍然成立。

用法
----
    & "D:\\Develop\\AIC\\.venv\\Scripts\\python.exe" scripts/finetune_face_emotion_robust.py \
        --epochs 8 --batch 96 --lr 2e-4 --out "D:\\Develop\\AIC\\Face\\ResNet50_finetuned_robust.pth"
"""
from __future__ import annotations

import argparse
import io
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT / "scripts"))

# 复用导出脚本里**手写的 ResNet50**（venv 有 torch 无 torchvision），
# 它已经用 `load_state_dict(strict=True)` 与官方权重逐键对齐过，结构可信。
from export_face_emotion_onnx import ResNet  # noqa: E402

DEFAULT_DATA = Path(r"D:\Develop\AIC\数据集\数据集\Face")
DEFAULT_WEIGHTS = Path(r"D:\Develop\AIC\Face\ResNet101_WholeData.pth")
INPUT_SIZE = 96

# ⚠️ 顺序必须与前端 FACE_EMOTION_LABELS 一致（实测顺序，不是字典序）
CLASSES = ["neutral", "happy", "sad"]

# 选模型时允许干净精度最多掉多少（见训练循环里的说明）
MAX_CLEAN_DROP = 0.010


# ── 数据 ───────────────────────────────────────────────────────────────────

def _pil(a: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _gamma_pil(im: Image.Image, g: float) -> Image.Image:
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return _pil((arr ** g) * 255.0)


def _wb_pil(im: Image.Image, gains) -> Image.Image:
    arr = np.asarray(im, dtype=np.float32) * np.array(gains, dtype=np.float32)
    return _pil(arr)


def _noise_pil(im: Image.Image, sigma: float, rng=None) -> Image.Image:
    r = rng if rng is not None else np.random
    arr = np.asarray(im, dtype=np.float32)
    arr = arr + r.normal(0, sigma, arr.shape)
    return _pil(arr)


def _jpeg_pil(im: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    with Image.open(buf) as j:
        return j.convert("RGB")


class FaceDataset(torch.utils.data.Dataset):
    """读 96×96 图 + 在线施加"上线会遇到"的干扰。

    增广强度是按**实测的失效模式**选的，不是随便调大：
      · 平移 ±8px     ← 实测 4px 就能把 35% 的开心判成悲伤
      · 缩放 0.85~1.18 ← 实测类别与"裁得紧不紧"混杂（sad 0.77 vs happy 0.64）
      · 高斯模糊 0~2.0 ← 实测 blur2 让开心召回掉到 28.7%
      · 降采样 0.5~1.0 ← 实测源分辨率减半 → 77.5%，1/3 → 47.5%
      · JPEG q20~85    ← 实测 q30 让 30% 的开心变成悲伤
      · 亮度/对比度    ← 实测影响很小，但成本几乎为零，一并加上
      · **叠加链**（概率 0.45）← 第二版新增：真实摄像头不会只犯一个错，而是
        "脸偏小 + 略糊 + 偏色 + 噪声 + 压缩"一起上。实测这正是本模型唯一还明显
        吃亏的场景（`webcam` 行：真 sad 有 33.8% 被判成中性，而单独施加半分辨率、
        单独施加 blur2 时 sad 召回都在 95% 以上 —— 典型的"千刀万剐"型退化）。
        第一版微调的选模型集里没有这个场景，所以从没被优化过。
    """

    def __init__(self, root: Path, limit_per_class: int | None = None):
        self.items: list[tuple[Path, int]] = []
        for idx, c in enumerate(CLASSES):
            files = sorted([f for f in (root / c).iterdir() if f.is_file()])
            if limit_per_class:
                files = files[:limit_per_class]
            self.items += [(f, idx) for f in files]

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def augment(im: Image.Image) -> Image.Image:
        # 0) 叠加型退化（与 domain-shift 脚本的 `webcam` / `webcam_hard` 同口径）
        if random.random() < 0.45:
            f = random.uniform(0.5, 0.85)
            small = max(16, int(INPUT_SIZE * f))
            im = im.resize((small, small), Image.BILINEAR).resize(
                (INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
            im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.3)))
            im = _gamma_pil(im, random.uniform(0.85, 1.3))
            im = _wb_pil(im, (random.uniform(0.90, 1.12), 1.0, random.uniform(0.88, 1.10)))
            im = _noise_pil(im, random.uniform(0, 9))
            if random.random() < 0.8:
                im = _jpeg_pil(im, random.randint(20, 65))
        # 1) 随机仿射：平移 + 缩放（越界用均值色填充，不引入训练里不存在的纯黑边）
        if random.random() < 0.9:
            s = random.uniform(0.85, 1.18)
            tx = random.uniform(-8, 8)
            ty = random.uniform(-8, 8)
            fill = tuple(int(v) for v in np.asarray(im).reshape(-1, 3).mean(axis=0))
            im = im.transform(im.size, Image.AFFINE, (s, 0, tx, 0, s, ty),
                              resample=Image.BILINEAR, fillcolor=fill)
        # 2) 降采样（模拟人脸在画面里偏小 → patch 是插值出来的）
        if random.random() < 0.35:
            f = random.uniform(0.5, 0.9)
            small = max(16, int(INPUT_SIZE * f))
            im = im.resize((small, small), Image.BILINEAR).resize(
                (INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        # 3) 模糊（失焦 / 运动 / 低照度下的降噪涂抹）
        if random.random() < 0.45:
            im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 2.0)))
        # 4) JPEG 压缩（摄像头逐帧编码）
        if random.random() < 0.45:
            im = _jpeg_pil(im, random.randint(20, 85))
        # 5) 亮度/对比度/饱和
        if random.random() < 0.6:
            im = ImageEnhance.Brightness(im).enhance(random.uniform(0.75, 1.3))
            im = ImageEnhance.Contrast(im).enhance(random.uniform(0.7, 1.35))
        # 6) 水平翻转（摄像头不镜像，但人脸左右不对称不该成为判据）
        if random.random() < 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        return im

    def __getitem__(self, i: int):
        path, label = self.items[i]
        with Image.open(path) as im:
            im = im.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
            im = self.augment(im)
            arr = np.asarray(im, dtype=np.float32) / 255.0
        x = torch.from_numpy(np.transpose(arr, (2, 0, 1)).copy())
        return x, label


class CleanEval(torch.utils.data.Dataset):
    """验证集：**不做任何增广**，用来守住"干净精度不许掉"。"""

    def __init__(self, root: Path):
        self.items: list[tuple[Path, int]] = []
        for idx, c in enumerate(CLASSES):
            files = sorted([f for f in (root / c).iterdir() if f.is_file()])
            self.items += [(f, idx) for f in files]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label = self.items[i]
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(np.transpose(arr, (2, 0, 1)).copy()), label


class RobustEval(torch.utils.data.Dataset):
    """鲁棒性验证集：对验证图施加**固定的**几种干扰（与 domain-shift 脚本同口径）。

    每个样本对应"某张图 × 某种干扰"，用样本序号确定性决定，所以同一轮内可比、
    跨轮也可比。只用来**如实观察趋势**，不参与"干净精度"约束，但**参与选模型**
    （否则会像第一版那样：`webcam` 叠加场景从没被优化过，留下 sad→neutral 33.8% 的洞）。

    默认每类抽 80 张 × 9 种干扰 = 2160 个样本，与
    `face_emotion_domain_shift_check.py --limit 80` 的口径一致，便于横向对照。
    """

    OPS = ["clean", "shift_up4", "shift_up6", "blur2", "halfres", "thirdres",
           "jpeg30", "webcam", "webcam_hard"]

    def __init__(self, root: Path, limit_per_class: int = 80):
        self.pairs: list[tuple[Path, int]] = []
        for idx, c in enumerate(CLASSES):
            files = sorted([f for f in (root / c).iterdir() if f.is_file()])
            step = max(1, len(files) // limit_per_class) if limit_per_class else 1
            picked = files[::step][:limit_per_class]
            self.pairs += [(f, idx) for f in picked]

    def __len__(self) -> int:
        return len(self.pairs) * len(self.OPS)

    def __getitem__(self, i: int):
        pi = i % len(self.pairs)
        op = self.OPS[i // len(self.pairs)]
        path, label = self.pairs[pi]
        with Image.open(path) as im:
            a = np.asarray(im.convert("RGB"))
        arr = apply_op(a, op).astype(np.float32) / 255.0
        return torch.from_numpy(np.transpose(arr, (2, 0, 1)).copy()), label


def apply_op(a: np.ndarray, op: str) -> np.ndarray:
    """与 `face_emotion_domain_shift_check.py` 同口径的退化（保持同一套评测语言）。

    含噪声的算子用**固定种子**的局部 RNG：这样"鲁棒验证"每一轮都是同一份数据，
    逐轮可比；否则噪声每次不同，趋势里会混进采样抖动。
    """
    h, w = a.shape[:2]
    rng = np.random.default_rng(20261010)

    def shift(dy):
        out = np.pad(a, ((max(dy, 0), max(-dy, 0)), (0, 0), (0, 0)), mode="edge")
        y0 = max(-dy, 0)
        return out[y0:y0 + h]

    if op == "clean":
        return a
    if op == "shift_up4":
        return shift(-4)
    if op == "shift_up6":
        return shift(-6)
    if op == "blur2":
        return np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(2.0)))
    if op in ("halfres", "thirdres"):
        f = 2 if op == "halfres" else 3
        s = max(8, INPUT_SIZE // f)
        im = Image.fromarray(a).resize((s, s), Image.BILINEAR)
        return np.asarray(im.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR))
    if op == "jpeg30":
        return np.asarray(_jpeg_pil(Image.fromarray(a), 30))
    if op == "webcam":
        # 与 face_emotion_domain_shift_check.py 的 op_webcam 逐参数一致：
        # 半分辨率 + 模糊 0.6 + gamma 1.12 + 暖白平衡 + 噪声 5
        im = Image.fromarray(a).resize((INPUT_SIZE // 2, INPUT_SIZE // 2), Image.BILINEAR)
        im = im.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        im = im.filter(ImageFilter.GaussianBlur(0.6))
        im = _gamma_pil(im, 1.12)
        im = _wb_pil(im, (1.06, 1.0, 0.95))
        im = _noise_pil(im, 5, rng)
        return np.asarray(im)
    if op == "webcam_hard":
        s = max(8, INPUT_SIZE // 3)
        im = Image.fromarray(a).resize((s, s), Image.BILINEAR).resize(
            (INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        im = im.filter(ImageFilter.GaussianBlur(1.0))
        im = _gamma_pil(im, 1.28)
        im = _wb_pil(im, (1.10, 1.0, 0.90))
        im = _noise_pil(im, 10, rng)
        im = _jpeg_pil(im, 30)
        return np.asarray(im)
    raise ValueError(op)


# ── 训练 ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, np.ndarray, list[int]]:
    model.eval()
    conf = np.zeros((3, 3), dtype=np.int64)
    labels_all: list[int] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(1).cpu().numpy()
        for p, t in zip(pred, y.numpy()):
            conf[t, p] += 1
        labels_all += list(y.numpy())
    acc = float(np.trace(conf) / max(1, conf.sum()))
    return acc, conf, labels_all


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="表情模型抗干扰微调")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--out", type=Path,
                    default=Path(r"D:\Develop\AIC\Face\ResNet50_finetuned_robust.pth"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader 进程数。**默认 0**：本机实测 workers>0 会在 Windows 上"
                         "卡死（不是慢，是挂住）；单进程下增广只要 1.8ms/张，够用。")
    ap.add_argument("--limit", type=int, default=0, help="每类限流（调试用，0=全量）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}" + (f" {torch.cuda.get_device_name(0)}" if device.type == "cuda" else ""))

    # ── 模型：复用导出脚本里手写的 ResNet50（layers 3/4/6/3，与权重严格对齐）──
    model = ResNet([3, 4, 6, 3], num_classes=3)
    sd = torch.load(str(args.weights), map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"⚠️ strict=False：missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("   missing[:8]:", missing[:8])
        if unexpected:
            print("   unexpected[:8]:", unexpected[:8])
    else:
        print("[权重] 逐键严格对齐 ✅")
    model.to(device)

    train_ds = FaceDataset(args.data / "training", args.limit or None)
    val_ds = CleanEval(args.data / "validation")
    rob_ds = RobustEval(args.data / "validation", limit_per_class=80)
    print(f"[数据] 训练 {len(train_ds)} / 干净验证 {len(val_ds)} / 鲁棒验证 {len(rob_ds)}")

    dl_kw = dict(num_workers=args.workers, pin_memory=(device.type == "cuda"),
                 persistent_workers=args.workers > 0)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                           drop_last=True, **dl_kw)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=256, shuffle=False,
                                         num_workers=min(2, args.workers))
    rob_dl = torch.utils.data.DataLoader(rob_ds, batch_size=256, shuffle=False,
                                         num_workers=min(2, args.workers))

    # 微调：小学习率 + 余弦退火；标签平滑 0.05 缓解"噪声标签"（数据集里 sad 混进了
    # fear/surprise 之类的极端表情，硬标签会把这些噪声当成真理去拟合）
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    base_clean, base_conf, _ = evaluate(model, val_dl, device)
    base_rob, _, _ = evaluate(model, rob_dl, device)
    print(f"\n[基线] 微调前：干净验证 = {base_clean * 100:.2f}%   鲁棒验证 = {base_rob * 100:.2f}%")
    print("        混淆矩阵（行=真实，列=预测 neutral/happy/sad）")
    print(base_conf)
    for i, c in enumerate(CLASSES):
        tot = int(base_conf[i].sum())
        print(f"        {c:>8} 召回 {base_conf[i, i] / max(1, tot) * 100:5.1f}%  (n={tot})")

    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss = 0.0
        seen = 0
        correct = 0
        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * x.size(0)
            correct += int((logits.argmax(1) == y).sum())
            seen += x.size(0)
        sched.step()
        clean, conf, _ = evaluate(model, val_dl, device)
        rob, _, _ = evaluate(model, rob_dl, device)
        print(f"[E{ep}/{args.epochs}] 训练loss={run_loss / max(1, seen):.4f} "
              f"训练acc={correct / max(1, seen) * 100:.1f}% | "
              f"干净验证={clean * 100:.2f}%（基线 {base_clean * 100:.2f}%）| "
              f"鲁棒验证={rob * 100:.2f}% | {time.time() - t0:.0f}s")
        print(f"        混淆矩阵 {conf.tolist()}")
        # 选模型：干净精度允许掉最多 `MAX_CLEAN_DROP`，在此约束下取**鲁棒性最好**的一轮。
        #
        # 为什么允许掉 1pp：实测第 1 轮就是 干净 95.23→94.67%（−0.56pp）
        # 换 鲁棒 72.08→92.56%（**+20.5pp**）。用户的实际症状发生在上线域，
        # 用一个百分点换二十个百分点是明显划算的交换 —— 但必须写清楚，不能偷偷换。
        if clean >= base_clean - MAX_CLEAN_DROP and rob > best:
            best = rob
            best_clean = clean
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(args.out))
            print(f"        ✅ 保存（干净 {clean * 100:.2f}% / 鲁棒 {rob * 100:.2f}%）→ {args.out}")
        elif clean < base_clean - MAX_CLEAN_DROP:
            print(f"        ⏭ 跳过：干净精度掉超 {MAX_CLEAN_DROP * 100:.0f}pp"
                  f"（{clean * 100:.2f}%）")
        # 无论是否入选，都留一份"最后一轮"，便于事后对比
        torch.save(model.state_dict(), str(args.out) + ".last.pth")

    if best < 0:
        print("\n❌ 没有任何一轮满足「干净精度不掉」的约束，未保存。"
              "建议降低 --lr 或增强度。")
        return 1

    print(f"\n完成：最佳鲁棒验证 {best * 100:.2f}%，权重 → {args.out}")
    print("下一步（**必须按顺序**）：")
    print("  1) 导出 ONNX 并扫类别排列：scripts/export_face_emotion_onnx.py + "
          "scripts/face_emotion_accuracy_check.py")
    print("  2) 鲁棒性对照：scripts/face_emotion_domain_shift_check.py（对比新旧模型）")
    print("  3) 前端一致性：scripts/verify_face_emotion_accuracy.ts（必须 ≥95% 且 exit 0）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
