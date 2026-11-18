"""针对「悲伤识别」的独立审计（**不使用 testing 集、不使用 fear 类**）。

为什么单独做这个
----------------
用户反馈"悲伤完全出不来"，但前面所有评测都显示模型在数据集上 sad 召回 97%+。
两边对不上，就必须把 sad 这一项**单独、干净地**验证一遍，而且满足三条约束：

  1. **只用 training + validation**（这两份才是 3 类数据），完全不碰 `testing/`
     —— 那份含 `fear`，会把"模型没有第 4 类"的账算到 sad 头上；
  2. **分类别给精确率/召回率/F1**，而不是只看总体准确率；
  3. **给出 P(sad) 的分离度（AUC）** —— 召回率只回答"赢没赢"，
     AUC 回答"到底能不能分得开"，这才是"是不是彻底出不来"的答案；
  4. **把判错的 sad 图拼出来看** —— 数字说不了"它到底错在什么样的脸上"。

另外同时给出**线上渲染**（`face_live_render_eval.py` 的几何）下的结果，
因为那才是模型线上真正看到的输入。

用法
----
    & "D:\\Develop\\AIC\\.venv\\Scripts\\python.exe" scripts/face_sad_audit.py
    ... --limit 400          # 每类抽样上限（默认全量）
    ... --skip-live          # 不做线上渲染那一路（省时间）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = APP_ROOT / "client" / "public" / "models" / "face_emotion_3cls.onnx"
DEFAULT_DATA = Path(r"D:\Develop\AIC\数据集\数据集\Face")
DEFAULT_TASK = Path(r"D:\Develop\AIC\_mediapipe_models\face_landmarker.task")

# ⚠️ 只有这 3 类。`testing` 里的 `fear` 是模型没有的第 4 类，本审计**刻意排除**。
CLASSES = ["neutral", "happy", "sad"]
LABELS = CLASSES  # 前端索引顺序
INPUT_SIZE = 96
LIVE_H, LIVE_CY = 0.6608, 0.5884   # 前端裁剪框的实际产出（见 face_framing_audit.py）


def collect(root: Path, split: str, limit: int):
    out = []
    for c in CLASSES:
        d = root / split / c
        if not d.is_dir():
            continue
        files = sorted([f for f in d.iterdir() if f.is_file()])
        if limit:
            step = max(1, len(files) // limit)
            files = files[::step][:limit]
        out += [(f, c) for f in files]
    return out


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def render_live(arr: np.ndarray, h_n: float, cy_n: float, cx_n: float,
                target_h=LIVE_H, target_cy=LIVE_CY) -> np.ndarray:
    """把这张图重渲染成"前端裁剪框会产出的样子"（详见 face_live_render_eval.py）。"""
    if h_n <= 1e-6:
        return arr
    s = target_h / h_n
    cx_in, cy_in = cx_n * INPUT_SIZE, cy_n * INPUT_SIZE
    cx_out, cy_out = 0.5 * INPUT_SIZE, target_cy * INPUT_SIZE
    im = Image.fromarray(arr)
    fill = tuple(int(v) for v in arr.reshape(-1, 3).mean(axis=0))
    return np.asarray(im.transform((INPUT_SIZE, INPUT_SIZE), Image.AFFINE,
                                   (1 / s, 0, cx_in - cx_out / s, 0, 1 / s, cy_in - cy_out / s),
                                   resample=Image.BILINEAR, fillcolor=fill))


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """P(sad) 区分 sad / 非sad 的 ROC-AUC（用秩和公式，无阈值）。"""
    pos = scores[positive]
    neg = scores[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    # 处理并列：用平均秩
    allv = np.concatenate([pos, neg])
    uniq, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.zeros(len(uniq))
        np.add.at(sums, inv, ranks)
        avg = sums / cnt
        ranks = avg[inv]
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def metrics(conf: np.ndarray) -> None:
    print(f"{'类别':<10}{'精确率':>9}{'召回率':>9}{'F1':>9}{'n':>8}")
    for i, c in enumerate(CLASSES):
        tp = conf[i, i]
        fp = conf[:, i].sum() - tp
        fn = conf[i].sum() - tp
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        print(f"{c:<10}{prec * 100:>8.1f}%{rec * 100:>8.1f}%{f1 * 100:>8.1f}%{int(conf[i].sum()):>8}")
    print(f"{'总体准确率':<10}{'':>27}{np.trace(conf) / max(1, conf.sum()) * 100:>8.1f}%")


def montage(items, path: Path, title: str) -> None:
    """items: [(arr, 真实, 预测, P(sad)), ...]；拼成一张图便于肉眼看"错在什么样的脸"."""
    if not items:
        return
    cell, cols = 96, 8
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cell * cols, cell * rows + 20 * rows), (245, 245, 245))
    dr = ImageDraw.Draw(sheet)
    y = 0
    for r in range(rows):
        chunk = items[r * cols:(r + 1) * cols]
        dr.text((4, y + 5), title if r == 0 else "", fill=(0, 0, 0))
        y += 20
        for i, (arr, t, p, psad) in enumerate(chunk):
            sheet.paste(Image.fromarray(arr), (i * cell, y))
            dr.text((i * cell + 2, y + cell - 11), f"{t[:3]}>{p[:3]} {psad:.2f}", fill=(255, 0, 0))
        y += cell
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    print(f"  -> {path}")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="悲伤识别的独立审计（排除 fear / testing）")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--task", type=Path, default=DEFAULT_TASK)
    ap.add_argument("--splits", type=str, default="validation,training")
    ap.add_argument("--limit", type=int, default=0, help="每类抽样上限（0=全量）")
    ap.add_argument("--skip-live", action="store_true")
    ap.add_argument("--dump", type=Path, default=Path(r"D:\Develop\AIC\_face_sad_audit"))
    args = ap.parse_args()

    import onnxruntime as ort

    print(f"[模型] {args.model}")
    print(f"[口径] 只用 {args.splits} 的 {CLASSES} 三类；**不使用 testing 集、不涉及 fear**")
    sess = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    rows = []
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        got = collect(args.data, split, args.limit)
        print(f"[数据] {split}: {len(got)} 张")
        rows += [(p, c, split) for p, c in got]

    imgs, y_true, y_pred, p_sad = [], [], [], []
    for path, c, _split in rows:
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"))
        x = np.transpose(arr.astype(np.float32) / 255.0, (2, 0, 1))[None, ...].copy()
        probs = softmax(sess.run(None, {input_name: x})[0][0])
        imgs.append(arr)
        y_true.append(CLASSES.index(c))
        y_pred.append(int(np.argmax(probs)))
        p_sad.append(float(probs[CLASSES.index("sad")]))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    p_sad = np.array(p_sad)

    conf = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        conf[t, p] += 1
    print("\n" + "=" * 84)
    print("A) 原图口径（训练分布）")
    print("=" * 84)
    print(f"{'':<10}" + "".join(f"{'→' + c:>11}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"{c:<10}" + "".join(f"{conf[i, j] / max(1, conf[i].sum()) * 100:>10.1f}%" for j in range(3)))
    metrics(conf)

    is_sad = y_true == CLASSES.index("sad")
    a = auc(p_sad, is_sad)
    print(f"\n  P(悲伤) 的分离度 AUC(sad vs 其余) = {a:.4f}")
    print(f"  真实 sad  的 P(悲伤)：中位数 {np.median(p_sad[is_sad]):.3f}  "
          f"p10 {np.percentile(p_sad[is_sad], 10):.3f}")
    print(f"  真实非 sad 的 P(悲伤)：中位数 {np.median(p_sad[~is_sad]):.3f}  "
          f"p90 {np.percentile(p_sad[~is_sad], 90):.3f}")
    thr = 0.5
    print(f"  P(悲伤) ≥ {thr} 的样本里：真 sad 命中 {int((is_sad & (p_sad >= thr)).sum())}"
          f"/{int(is_sad.sum())}，误报 {int((~is_sad & (p_sad >= thr)).sum())}/{int((~is_sad).sum())}")

    # 判错的 sad：肉眼看"错在什么样的脸"
    wrong_sad = [i for i in range(len(rows)) if y_true[i] == CLASSES.index("sad") and y_pred[i] != CLASSES.index("sad")]
    right_sad = [i for i in range(len(rows)) if y_true[i] == CLASSES.index("sad") and y_pred[i] == CLASSES.index("sad")]
    print(f"\n  sad 判错 {len(wrong_sad)} 张 / 共 {int(is_sad.sum())} 张")
    montage([(imgs[i], "sad", CLASSES[y_pred[i]], p_sad[i]) for i in wrong_sad[:40]],
            args.dump / "sad_wrong.png", "判错的 sad（行=真实>预测 P(sad)）")
    montage([(imgs[i], "sad", CLASSES[y_pred[i]], p_sad[i]) for i in right_sad[:40]],
            args.dump / "sad_right.png", "判对的 sad")

    if args.skip_live:
        return 0

    # B) 线上渲染口径
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(args.task)),
        running_mode=vision.RunningMode.IMAGE, num_faces=1)
    landmarker = vision.FaceLandmarker.create_from_options(options)

    conf2 = np.zeros((3, 3), dtype=np.int64)
    p_sad2, y_true2, rendered = [], [], []
    for i, (path, c, _s) in enumerate(rows):
        arr = imgs[i]
        try:
            res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
        except Exception:
            res = None
        if not res or not res.face_landmarks:
            continue
        lm = res.face_landmarks[0]
        xs = [q.x for q in lm]
        ys = [q.y for q in lm]
        r = render_live(arr, max(ys) - min(ys), (min(ys) + max(ys)) / 2, (min(xs) + max(xs)) / 2)
        x = np.transpose(r.astype(np.float32) / 255.0, (2, 0, 1))[None, ...].copy()
        probs = softmax(sess.run(None, {input_name: x})[0][0])
        conf2[CLASSES.index(c), int(np.argmax(probs))] += 1
        p_sad2.append(float(probs[CLASSES.index("sad")]))
        y_true2.append(CLASSES.index(c))
        rendered.append(r)

    print("\n" + "=" * 84)
    print("B) 线上渲染口径（把每张脸归一化到前端裁剪框的实际产出）")
    print("=" * 84)
    print(f"{'':<10}" + "".join(f"{'→' + c:>11}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"{c:<10}" + "".join(f"{conf2[i, j] / max(1, conf2[i].sum()) * 100:>10.1f}%" for j in range(3)))
    metrics(conf2)
    p_sad2 = np.array(p_sad2)
    y_true2 = np.array(y_true2)
    is_sad2 = y_true2 == CLASSES.index("sad")
    print(f"\n  P(悲伤) 的分离度 AUC = {auc(p_sad2, is_sad2):.4f}")
    print(f"  真实 sad 的 P(悲伤)：中位数 {np.median(p_sad2[is_sad2]):.3f}")

    print("\n" + "=" * 84)
    print("结论")
    print("=" * 84)
    rec_raw = conf[2, 2] / max(1, conf[2].sum())
    rec_live = conf2[2, 2] / max(1, conf2[2].sum())
    print(f"  sad 召回：原图 {rec_raw * 100:.1f}%   线上渲染 {rec_live * 100:.1f}%")
    print(f"  P(悲伤) 分离度：原图 AUC {auc(p_sad, is_sad):.4f}   线上渲染 AUC {auc(p_sad2, is_sad2):.4f}")
    if min(rec_raw, rec_live) > 0.9 and min(auc(p_sad, is_sad), auc(p_sad2, is_sad2)) > 0.97:
        print("  → 模型**能**正确识别数据集里的 sad，且在线上几何下不掉。")
        print("     所以线上'悲伤出不来'不是模型/取景的问题，而是**你的脸不在训练分布里**。")
    else:
        print("  → 模型对 sad 本身就不行，问题在模型/数据侧。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
