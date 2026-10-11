"""表情模型「上线域」体检：把 validation 图像按**摄像头真实会遇到的退化**变换后重测。

为什么需要它
------------
`face_emotion_accuracy_check.py` 测的是**数据集原始分布**上的精度（95.2%）。
但线上这条链路和数据集之间至少差四件事：

  1. **分辨率**：人脸在 640×480 画面里可能只占 ~150px，裁成 96×96 是**下采样**，
     而数据集是原生 96×96 —— 下采样会丢高频，嘴角/眼角的细纹首当其冲。
  2. **光照/白平衡**：`3.1.train_ResNet101.py` 的 transform 只有 `ToTensor()`，
     **没有 mean/std 归一化**。模型直接吃原始像素值，所以对亮度/对比度/色温敏感。
  3. **运动模糊 / 传感器噪声**：摄像头逐帧压缩与手抖。
  4. **取景比例**：前端 `FACE_PATCH_MARGIN` 裁出来的是 **1.5×脸长** 的方框
     （左/右各 0.15、上 0.35、下 0.15，再取最大边做正方形），
     与训练图里脸的占比未必一致。

本脚本把这些因素**逐个单独**施加在真实带标签图像上，看准确率（尤其开心类召回）
掉多少 —— 这样才能判断"开心识别不出来"到底是哪一个环节造成的，而不是靠猜。

用法
----
    & "D:\\Develop\\AIC\\.venv\\Scripts\\python.exe" scripts/face_emotion_domain_shift_check.py
    ... --limit 80 --splits validation,testing
    ... --only clean,halfres,blur2,dark,flat,wb_warm,webcam
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = APP_ROOT / "client" / "public" / "models" / "face_emotion_3cls.onnx"
DEFAULT_DATA = Path(r"D:\Develop\AIC\数据集\数据集\Face")
# 与 client/src/services/faceEmotionClassifier.ts 的 FACE_EMOTION_LABELS 一致
LABELS = ["neutral", "happy", "sad"]
INPUT_SIZE = 96
RNG = np.random.default_rng(42)


# ── 退化算子：输入/输出都是 uint8 的 HWC RGB ────────────────────────────────

def op_clean(a: np.ndarray) -> np.ndarray:
    return a


def op_halfres(a: np.ndarray) -> np.ndarray:
    """人脸只占画面一小块 → 前端裁出的 patch 是下采样结果。

    等价于把一个 96px 的人脸先用 48px 采样再放大回去：高频全部丢失。
    """
    im = Image.fromarray(a).resize((INPUT_SIZE // 2, INPUT_SIZE // 2), Image.BILINEAR)
    return np.asarray(im.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR))


def op_thirdres(a: np.ndarray) -> np.ndarray:
    """更极端：人脸只占 ~1/3 画幅（640×480 里 96px 左右的脸）。"""
    im = Image.fromarray(a).resize((INPUT_SIZE // 3, INPUT_SIZE // 3), Image.BILINEAR)
    return np.asarray(im.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR))


def _blur(a: np.ndarray, radius: float) -> np.ndarray:
    return np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(radius)))


def op_blur1(a: np.ndarray) -> np.ndarray:
    return _blur(a, 1.0)


def op_blur2(a: np.ndarray) -> np.ndarray:
    return _blur(a, 2.0)


def _gamma(a: np.ndarray, g: float) -> np.ndarray:
    x = a.astype(np.float32) / 255.0
    return np.clip((x ** g) * 255.0, 0, 255).astype(np.uint8)


def op_dark(a: np.ndarray) -> np.ndarray:
    """偏暗的房间（gamma 1.35，等效整体压暗）。"""
    return _gamma(a, 1.35)


def op_bright(a: np.ndarray) -> np.ndarray:
    """过曝 / 背光（gamma 0.75）。"""
    return _gamma(a, 0.75)


def op_flat(a: np.ndarray) -> np.ndarray:
    """低对比：逆光或廉价摄像头的"灰蒙蒙"。"""
    x = a.astype(np.float32)
    return np.clip((x - 127.5) * 0.6 + 127.5, 0, 255).astype(np.uint8)


def _noise(a: np.ndarray, sigma: float) -> np.ndarray:
    n = RNG.normal(0, sigma, a.shape)
    return np.clip(a.astype(np.float32) + n, 0, 255).astype(np.uint8)


def op_noise8(a: np.ndarray) -> np.ndarray:
    return _noise(a, 8)


def op_noise16(a: np.ndarray) -> np.ndarray:
    return _noise(a, 16)


def _wb(a: np.ndarray, gains: tuple[float, float, float]) -> np.ndarray:
    out = a.astype(np.float32) * np.array(gains, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def op_wb_warm(a: np.ndarray) -> np.ndarray:
    """暖光（白炽灯/暖白 LED）—— 自动白平衡没压住时的常见偏移。"""
    return _wb(a, (1.15, 1.0, 0.85))


def op_wb_cool(a: np.ndarray) -> np.ndarray:
    """冷光（日光灯/窗外阴天）。"""
    return _wb(a, (0.88, 1.0, 1.15))


def op_jpeg30(a: np.ndarray) -> np.ndarray:
    """webcam 逐帧 JPEG 压缩（质量 30 是低带宽下的常见档位）。"""
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="JPEG", quality=30)
    buf.seek(0)
    with Image.open(buf) as im:
        return np.asarray(im.convert("RGB"))


def op_webcam(a: np.ndarray) -> np.ndarray:
    """把上面几项**叠加**成一个"典型"的摄像头画面（不是极端情况）。"""
    x = op_halfres(a)
    x = _blur(x, 0.6)
    x = _gamma(x, 1.12)
    x = _wb(x, (1.06, 1.0, 0.95))
    x = _noise(x, 5)
    return x


def op_webcam_hard(a: np.ndarray) -> np.ndarray:
    """更差的条件：人脸更小 + 更暗 + 噪声更大 + JPEG 更狠。"""
    x = op_thirdres(a)
    x = _blur(x, 1.0)
    x = _gamma(x, 1.28)
    x = _wb(x, (1.10, 1.0, 0.90))
    x = _noise(x, 10)
    x = op_jpeg30(x)
    return x


def _mask_rows(a: np.ndarray, y0: float, y1: float) -> np.ndarray:
    """把纵向 [y0, y1) 这一段替换成该图的整体均值色。

    用途：**定位模型依赖哪一块区域**。表情识别里"嘴"和"眼"是两条独立证据，
    把其中一条抹掉后看预测倒向哪一类，就能判断模型的判据到底是哪一条。
    抹成均值色而不是纯黑，是为了不引入一个训练里不存在的强边缘。
    """
    out = a.copy().astype(np.float32)
    h = out.shape[0]
    lo, hi = int(round(h * y0)), int(round(h * y1))
    out[lo:hi, :, :] = a.reshape(-1, 3).mean(axis=0)
    return np.clip(out, 0, 255).astype(np.uint8)


def op_mask_eyes(a: np.ndarray) -> np.ndarray:
    """抹掉眼/眉区（上 40%）——只剩嘴。"""
    return _mask_rows(a, 0.0, 0.40)


def op_mask_mouth(a: np.ndarray) -> np.ndarray:
    """抹掉嘴区（下 35%）——只剩眼/眉。"""
    return _mask_rows(a, 0.65, 1.0)


def op_mask_top_bottom(a: np.ndarray) -> np.ndarray:
    """眼和嘴都抹掉，只留中段——近似"什么都没有"的对照。"""
    x = _mask_rows(a, 0.0, 0.40)
    return _mask_rows(x, 0.65, 1.0)


def _shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """整幅平移 (dy, dx) 像素，越界用**边缘复制**填充（不用 np.roll：回卷会造出假边缘）。"""
    h, w = a.shape[:2]
    out = np.pad(a, ((max(dy, 0), max(-dy, 0)), (max(dx, 0), max(-dx, 0)), (0, 0)), mode="edge")
    y0 = max(-dy, 0)
    x0 = max(-dx, 0)
    return out[y0:y0 + h, x0:x0 + w]


def op_shift_up6(a: np.ndarray) -> np.ndarray:
    """人脸在 patch 里上移 6px（96px 的 6.25%）。"""
    return _shift(a, -6, 0)


def op_shift_down6(a: np.ndarray) -> np.ndarray:
    return _shift(a, 6, 0)


def op_shift_down12(a: np.ndarray) -> np.ndarray:
    return _shift(a, 12, 0)


def op_shift_up12(a: np.ndarray) -> np.ndarray:
    return _shift(a, -12, 0)


def op_shift_left6(a: np.ndarray) -> np.ndarray:
    return _shift(a, 0, -6)


def op_shift_right6(a: np.ndarray) -> np.ndarray:
    return _shift(a, 0, 6)


def op_shift_up1(a: np.ndarray) -> np.ndarray:
    return _shift(a, -1, 0)


def op_shift_up2(a: np.ndarray) -> np.ndarray:
    """2px ≈ 前端现状相对数据集取景的纵向偏差（实测 0.0217 × 96 ≈ 2.1px）。"""
    return _shift(a, -2, 0)


def op_shift_up3(a: np.ndarray) -> np.ndarray:
    return _shift(a, -3, 0)


def op_shift_up4(a: np.ndarray) -> np.ndarray:
    return _shift(a, -4, 0)


def op_zoom90(a: np.ndarray) -> np.ndarray:
    """中心裁到 90% 再放大 —— 人脸在 patch 里**变大**（前端把框裁得更紧时就是这个效果）。"""
    return apply_zoom(a, 0.90)


def op_zoom80(a: np.ndarray) -> np.ndarray:
    return apply_zoom(a, 0.80)


def op_zoom110(a: np.ndarray) -> np.ndarray:
    """人脸在 patch 里变小（前端把框裁得更宽）。"""
    return apply_pad(a, 0.05)


OPS = {
    "clean": op_clean,
    "halfres": op_halfres,
    "thirdres": op_thirdres,
    "blur1": op_blur1,
    "blur2": op_blur2,
    "dark": op_dark,
    "bright": op_bright,
    "flat": op_flat,
    "noise8": op_noise8,
    "noise16": op_noise16,
    "wb_warm": op_wb_warm,
    "wb_cool": op_wb_cool,
    "jpeg30": op_jpeg30,
    "webcam": op_webcam,
    "webcam_hard": op_webcam_hard,
    "mask_eyes": op_mask_eyes,
    "mask_mouth": op_mask_mouth,
    "mask_both": op_mask_top_bottom,
    # 位置敏感性：模型（尤其"开心"这一类）几乎完全靠嘴部判据，
    # 所以人脸在 96px patch 里偏几个像素就可能改写结论。前端裁剪框与训练取景
    # 之间只要有一点几何差，就会以这种形式暴露出来。
    "shift_up6": op_shift_up6,
    "shift_up12": op_shift_up12,
    "shift_down6": op_shift_down6,
    "shift_down12": op_shift_down12,
    "shift_left6": op_shift_left6,
    "shift_right6": op_shift_right6,
    "shift_up1": op_shift_up1,
    "shift_up2": op_shift_up2,
    "shift_up3": op_shift_up3,
    "shift_up4": op_shift_up4,
    "zoom80": op_zoom80,
    "zoom90": op_zoom90,
    "zoom110": op_zoom110,
}
DEFAULT_OPS = list(OPS)


# ── 补偿算子：作用在**已经裁好的 96×96 patch** 上（前端能做的就只有这一步）──
#
# 动机：上一个表说明「光照」不是瓶颈（dark/flat/wb_* 都在 95% 以上），
# 掉分全部集中在**高频细节丢失**（blur*/halfres/thirdres/jpeg30）。而真实摄像头
# 画面天然比数据集的清晰裁图软 —— 所以值得试的是"在下采样后把细节补回来"。
# 这些算子只用到当前 patch 的像素，不需要额外信息，因此**可以真的搬到前端**。

def comp_none(a: np.ndarray) -> np.ndarray:
    return a


def _unsharp(a: np.ndarray, radius: float, percent: int, threshold: int = 0) -> np.ndarray:
    return np.asarray(
        Image.fromarray(a).filter(
            ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold)
        )
    )


def comp_unsharp_1_100(a: np.ndarray) -> np.ndarray:
    return _unsharp(a, 1.0, 100)


def comp_unsharp_1_150(a: np.ndarray) -> np.ndarray:
    return _unsharp(a, 1.0, 150)


def comp_unsharp_2_150(a: np.ndarray) -> np.ndarray:
    return _unsharp(a, 2.0, 150)


def comp_unsharp_1_250(a: np.ndarray) -> np.ndarray:
    return _unsharp(a, 1.0, 250)


def comp_sharpen(a: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(a).filter(ImageFilter.SHARPEN))


def comp_autocontrast(a: np.ndarray) -> np.ndarray:
    """逐通道拉满对比度（把暗部细节拉开），用来对照"光照类补偿是否真的没用"。"""
    from PIL import ImageOps
    return np.asarray(ImageOps.autocontrast(Image.fromarray(a), cutoff=1))


COMPENSATIONS = {
    "none": comp_none,
    "unsharp_1_100": comp_unsharp_1_100,
    "unsharp_1_150": comp_unsharp_1_150,
    "unsharp_2_150": comp_unsharp_2_150,
    "unsharp_1_250": comp_unsharp_1_250,
    "sharpen": comp_sharpen,
    "autocontrast": comp_autocontrast,
}


# ── 取景比例：pad（裁得更宽）/ zoom（裁得更紧）──────────────────────────────

def apply_pad(a: np.ndarray, frac: float) -> np.ndarray:
    """给图像四周补 `frac` 倍宽度的"背景"再缩回 96 —— 模拟前端裁得更宽。

    用**边缘复制**而不是填纯色：真实背景是人物的头发/衣服/房间，纯色会引入
    一个训练里根本不存在的强边缘，反而失真。
    """
    if frac <= 0:
        return a
    b = int(round(INPUT_SIZE * frac))
    padded = np.pad(a, ((b, b), (b, b), (0, 0)), mode="edge")
    return np.asarray(Image.fromarray(padded).resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR))


def apply_zoom(a: np.ndarray, frac: float) -> np.ndarray:
    """中心裁掉 `1-frac` 再缩放回 96 —— 模拟前端裁得更紧（脸更大）。"""
    if frac >= 1.0:
        return a
    side = max(8, int(round(INPUT_SIZE * frac)))
    off = (INPUT_SIZE - side) // 2
    crop = a[off:off + side, off:off + side]
    return np.asarray(Image.fromarray(crop).resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR))


# ── 数据与推理 ─────────────────────────────────────────────────────────────

def load_split(root: Path, split: str, limit: int | None):
    d = root / split
    if not d.is_dir():
        return [], []
    classes = sorted([p.name for p in d.iterdir() if p.is_dir()])
    samples: list[tuple[Path, str]] = []
    for c in classes:
        files = sorted([f for f in (d / c).iterdir() if f.is_file()])
        if limit:
            # 均匀取样，避免只看目录开头那批（常常同源）
            step = max(1, len(files) // limit)
            files = files[::step][:limit]
        samples += [(f, c) for f in files]
    return samples, classes


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def build_views(arr: np.ndarray, tta: str) -> list[np.ndarray]:
    """一次采样的多个"视角"。多视角时输出取各视角概率的**平均**（测试时增强）。

    `sharpen*` 是"原始 + 锐化"的集成：锐化单独用会在**干净**样本上掉点
    （见补偿 A/B 表），但把两者平均可以既保住干净精度、又拿回模糊下的细节 ——
    这正是要验证的点。
    """
    if tta == "none":
        return [arr]
    if tta == "hflip":
        return [arr, arr[:, ::-1].copy()]
    if tta == "multicrop":
        return [arr, apply_zoom(arr, 0.88), apply_pad(arr, 0.10)]
    if tta == "sharpen":
        return [arr, comp_unsharp_1_100(arr)]
    if tta == "sharpen2":
        return [arr, comp_unsharp_2_150(arr)]
    if tta == "sharpen_both":
        return [arr, comp_unsharp_1_100(arr), comp_unsharp_2_150(arr)]
    raise SystemExit(f"未知 TTA：{tta}")


def run_grid(sess, input_name: str, samples, ops: list[str], pad: float, zoom: float,
             tta: str, comp: str = "none") -> dict[str, np.ndarray]:
    """对每个退化算子跑一遍全量，返回 {op: probs[n,3]}。"""
    comp_fn = COMPENSATIONS[comp]
    probs: dict[str, list[np.ndarray]] = {op: [] for op in ops}
    for path, _label in samples:
        with Image.open(path) as im:
            base = np.asarray(im.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR))
        base = apply_pad(base, pad)
        base = apply_zoom(base, zoom)
        for op in ops:
            views = build_views(comp_fn(OPS[op](base)), tta)
            p = np.zeros(3, dtype=np.float64)
            for v in views:
                tensor = np.transpose(v.astype(np.float32) / 255.0, (2, 0, 1))[None, ...].copy()
                logits = sess.run(None, {input_name: tensor})[0][0]
                p += softmax(np.asarray(logits, dtype=np.float64))
            probs[op].append(p / len(views))
    return {op: np.stack(v) for op, v in probs.items()}


def report_destinations(probs: dict[str, np.ndarray], truth: list[str]) -> None:
    """每个真实类别的**完整去向行** —— 用来定位某一对类别之间的混淆。

    用户反馈分三个阶段：先是"开心识别不出来"，然后"咧嘴笑→悲伤"，现在
    "中性和悲伤分不出来"。后者的指纹是 **真 sad 大量落到 neutral**，
    所以必须按行看，而不是只看总体准确率。
    """
    print("\n" + "=" * 100)
    print("各类别去向分解（行 = 真实类别，值 = 被判成 中性/开心/悲伤 的比例）")
    print("=" * 100)
    print(f"{'退化':<14}{'真实类别':<10}{'→中性':>10}{'→开心':>10}{'→悲伤':>10}{'样本数':>9}")
    for op, p in probs.items():
        pred = p.argmax(axis=1)
        for cls in LABELS:
            m = [i for i, t in enumerate(truth) if t == cls]
            if not m:
                continue
            cells = [sum(1 for i in m if LABELS[pred[i]] == l) / len(m) for l in LABELS]
            print(f"{op:<14}{cls:<10}" + "".join(f"{c * 100:>9.1f}%" for c in cells)
                  + f"{len(m):>9}")


def compare_compensations(split: str, collected: dict[str, dict[str, np.ndarray]],
                          truth: list[str], classes: list[str]) -> None:
    """补偿算子 A/B 决策表：**clean 不许掉，退化场景要明显回升**才算有用。"""
    known = [c for c in classes if c in LABELS]
    idx = {l: i for i, l in enumerate(LABELS)}
    print("\n" + "=" * 92)
    print(f"{split} —— 变体 A/B（每格 = 总体 / 开心召回），行 = 补偿/TTA")
    print("=" * 92)
    ops = list(next(iter(collected.values())).keys())
    header = f"{'变体':<24}" + "".join(f"{o:>14}" for o in ops)
    print(header)
    for comp, probs in collected.items():
        cells = []
        for op in ops:
            p = probs[op]
            pred = p.argmax(axis=1)
            n = sum(1 for t in truth if t in known)
            hit = sum(1 for i, t in enumerate(truth) if t in known and LABELS[pred[i]] == t)
            hi = [i for i, t in enumerate(truth) if t == "happy"]
            hrec = sum(1 for i in hi if LABELS[pred[i]] == "happy") / len(hi) if hi else float("nan")
            cells.append(f"{hit / n * 100:>5.1f}/{hrec * 100:>5.1f}%")
        print(f"{comp:<24}" + "".join(f"{c:>14}" for c in cells))
    print("\n  （读法：`clean` 列不能比 none 差；`blur*/halfres/webcam*` 列要明显变好，才值得搬进前端。）")
    print(f"  开心样本基准概率均值：n={sum(1 for t in truth if t == 'happy')}")


def report(title: str, probs: dict[str, np.ndarray], truth: list[str], classes: list[str]) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    known = [c for c in classes if c in LABELS]
    idx = {l: i for i, l in enumerate(LABELS)}
    # 开心样本的**完整去向分解**：用户反馈"咧嘴笑被判成悲伤"，
    # 所以必须同时看 →开心 / →中性 / →悲伤 三条；只看"→中性"会漏掉真正的症状。
    print(f"{'退化':<14}{'总体':>7}{'开心召回':>9}{'中性召回':>9}{'悲伤召回':>9}"
          f"{'开心→开心':>10}{'开心→中性':>10}{'开心→悲伤':>10}{'开心均概率':>11}")
    rows = []
    for op, p in probs.items():
        pred = p.argmax(axis=1)
        n = sum(1 for t in truth if t in known)
        hit = sum(1 for i, t in enumerate(truth) if t in known and LABELS[pred[i]] == t)
        recalls = {}
        for c in known:
            m = [i for i, t in enumerate(truth) if t == c]
            recalls[c] = (sum(1 for i in m if LABELS[pred[i]] == c) / len(m)) if m else float("nan")
        hi = [i for i, t in enumerate(truth) if t == "happy"]
        mean_happy = float(p[hi, idx["happy"]].mean()) if hi else float("nan")
        to = {l: (sum(1 for i in hi if LABELS[pred[i]] == l) / len(hi)) if hi else float("nan")
              for l in LABELS}
        rows.append((op, hit / n, recalls, to, mean_happy))
    for op, acc, recalls, to, mh in sorted(rows, key=lambda r: -r[1]):
        print(f"{op:<14}{acc * 100:>6.1f}%" +
              "".join(f"{recalls[c] * 100:>8.1f}%" for c in known) +
              "".join(f"{to[l] * 100:>9.1f}%" for l in LABELS) +
              f"{mh * 100:>10.1f}%")


def report_happy_row(probs: dict[str, np.ndarray], truth: list[str]) -> None:
    """只聚焦「真实为开心」那批样本的完整混淆行 —— 直接对应用户的症状。"""
    idx = {l: i for i, l in enumerate(LABELS)}
    hi = [i for i, t in enumerate(truth) if t == "happy"]
    if not hi:
        return
    print("\n" + "=" * 100)
    print("聚焦：真实为「开心」的样本被判到哪里去（→悲伤 那一列就是「咧嘴笑被判成悲伤」）")
    print("=" * 100)
    print(f"{'退化':<14}{'→开心':>10}{'→中性':>10}{'→悲伤':>10}"
          f"{'happy概率均值':>14}{'sad概率均值':>13}")
    for op, p in probs.items():
        pred = p.argmax(axis=1)
        cells = [sum(1 for i in hi if LABELS[pred[i]] == l) / len(hi) for l in LABELS]
        print(f"{op:<14}" + "".join(f"{c * 100:>9.1f}%" for c in cells) +
              f"{p[hi, idx['happy']].mean() * 100:>13.1f}%"
              f"{p[hi, idx['sad']].mean() * 100:>12.1f}%")
    print(f"  （n={len(hi)} 张真实开心样本）")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="表情模型上线域体检")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--splits", type=str, default="validation,testing")
    ap.add_argument("--limit", type=int, default=80, help="每个类别取样上限（均匀取样）")
    ap.add_argument("--only", type=str, default="", help="只跑这些退化（逗号分隔）")
    ap.add_argument("--pad", type=float, default=0.0, help="模拟裁得更宽（0.15≈多带一圈背景）")
    ap.add_argument("--zoom", type=float, default=1.0, help="模拟裁得更紧（0.85≈脸放大 18%%）")
    ap.add_argument("--tta", type=str, default="none",
                    help="测试时增强：none,hflip,multicrop,sharpen,sharpen2,sharpen_both（可逗号分隔做 A/B）")
    ap.add_argument("--comp", type=str, default="none",
                    help="在裁好的 patch 上做的补偿算子（逗号分隔可做 A/B）；可选 %s"
                         % ",".join(COMPENSATIONS))
    args = ap.parse_args()

    import onnxruntime as ort

    ops = [o.strip() for o in args.only.split(",") if o.strip()] or DEFAULT_OPS
    bad = [o for o in ops if o not in OPS]
    if bad:
        raise SystemExit(f"未知退化算子：{bad}；可选 {DEFAULT_OPS}")
    comps = [c.strip() for c in args.comp.split(",") if c.strip()]
    badc = [c for c in comps if c not in COMPENSATIONS]
    if badc:
        raise SystemExit(f"未知补偿算子：{badc}；可选 {list(COMPENSATIONS)}")
    ttas = [t.strip() for t in args.tta.split(",") if t.strip()]
    variants = [(c, t) for t in ttas for c in comps]
    # 先构造一次，未知的 TTA 会在 build_views 里抛错 —— 提前在这里暴露
    for _c, _t in variants:
        build_views(np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8), _t)
    if not args.model.exists():
        raise SystemExit(f"模型不存在：{args.model}")

    print(f"[模型] {args.model.name} ({args.model.stat().st_size / 1048576:.1f} MB)")
    sess = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    print(f"[配置] 输入 {INPUT_SIZE}px | 标签顺序 {LABELS} | pad={args.pad} zoom={args.zoom}")
    print(f"[变体] {[f'{c}/{t}' for c, t in variants]}")

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        samples, classes = load_split(args.data, split, args.limit)
        if not samples:
            print(f"\n[{split}] 无数据，跳过")
            continue
        truth = [c for _, c in samples]
        print(f"\n[{split}] n={len(samples)} 类别={classes}")
        collected: dict[str, dict[str, np.ndarray]] = {}
        for comp, tta in variants:
            key = f"{comp}/{tta}"
            t0 = time.time()
            collected[key] = run_grid(sess, input_name, samples, ops, args.pad, args.zoom, tta, comp)
            print(f"  [变体 {key}] 完成，耗时 {time.time() - t0:.0f}s")
        if len(variants) == 1:
            report(f"{split} —— 逐个退化（{len(samples)} 张 × {len(ops)} 种）",
                   collected[variants[0][0] + "/" + variants[0][1]], truth, classes)
            report_happy_row(collected[variants[0][0] + "/" + variants[0][1]], truth)
            report_destinations(collected[variants[0][0] + "/" + variants[0][1]], truth)
        else:
            compare_compensations(split, collected, truth, classes)

    print("\n" + "=" * 92)
    print("怎么读这张表")
    print("=" * 92)
    print("  · 若某一行只有 `clean` 高、其他行崩 → 线上瓶颈是**成像条件**，不是模型。")
    print("  · 若 `halfres/thirdres/blur*` 掉得最狠 → 去提高有效分辨率（别让脸小于 96px）。")
    print("  · 若 `dark/flat/wb_*` 掉得最狠 → 预处理缺少归一化/白平衡，考虑加光照归一化。")
    print("  · 若 `webcam` 行的「开心召回」极低而「开心判定为中性」很高 → 线上确实存在「微笑被读成中性」。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
