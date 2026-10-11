"""「模型在线上到底看到什么」的忠实渲染评测。

要解决的问题
------------
用户反馈「悲伤完全出不来」。而 `face_framing_audit.py` 量出一个尖锐的事实：

    数据集里各类别的人脸包围盒高占比（h_n）中位数差别很大：
        happy 0.644 / neutral 0.635 / **sad 0.774**
    而前端裁剪框是**按关键点归一化**的，无论谁来、坐多远，产出永远是
        h_n ≈ 0.661、cy_n ≈ 0.588

也就是说：**线上 sad 人脸被渲染成"脸只有 0.661 满"，而训练集里的 sad 全是 0.774 满**。
如果模型学的是"脸占得满 → sad"这条与表情无关的捷径，那线上 sad 就**永远出不来** ——
这与本次症状完全吻合。

本脚本做一次**忠实的线上渲染**：拿带标签的数据集图，用它的关键点算出前端会产出的
几何（把脸归一化到 h_n=0.661 / cy_n=0.588），再喂模型，看每一类的召回怎么变。

- 如果 sad 召回在"线上渲染"下崩塌 → **病因是取景，不是你的表情**，而且**不需要采集链路**：
  重新按正确口径训练（或把线上裁剪改成匹配 sad 的口径）即可。
- 如果 sad 召回基本不变 → 取景不是病因，剩下的确实是"你的脸/你的表情不在训练分布里"，
  那时采集+标注+微调才是唯一解。

⚠️ 局限：越界区域只能用**边缘复制**填充（真实背景是头发/房间）。几何是精确的，
填充不是。所以它回答"取景会不会杀掉 sad"，不能回答"线上准确率是多少"。

用法
----
    & "D:\\Develop\\AIC\\.venv\\Scripts\\python.exe" scripts/face_live_render_eval.py --limit 150
    ... --target-h 0.774        # 反事实：如果线上也渲染成 sad 的典型占比，会怎样
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = APP_ROOT / "client" / "public" / "models" / "face_emotion_3cls.onnx"
DEFAULT_DATA = Path(r"D:\Develop\AIC\数据集\数据集\Face")
DEFAULT_TASK = Path(r"D:\Develop\AIC\_mediapipe_models\face_landmarker.task")

LABELS = ["neutral", "happy", "sad"]
INPUT_SIZE = 96
# 前端裁剪框的实际产出（由 face_framing_audit.py 在 1195 张图 + 4:3 画面下算出）
LIVE_H = 0.6608
LIVE_CY = 0.5884
# 数据集里 sad 的典型占比（反事实对照用）
SAD_TYPICAL_H = 0.7738


def measure_landmarks(paths, landmarker):
    import mediapipe as mp

    out = []
    for p, lab in paths:
        try:
            with Image.open(p) as im:
                arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
            res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
        except Exception:
            continue
        if not res.face_landmarks:
            continue
        lm = res.face_landmarks[0]
        xs = [q.x for q in lm]
        ys = [q.y for q in lm]
        out.append({
            "path": p, "label": lab, "img": arr,
            "w_n": max(xs) - min(xs), "h_n": max(ys) - min(ys),
            "cy_n": (min(ys) + max(ys)) / 2, "cx_n": (min(xs) + max(xs)) / 2,
        })
    return out


def render_live(row: dict, target_h: float, target_cy: float) -> np.ndarray:
    """把这张图**按前端裁剪框的口径重渲染**：脸归一化到 h_n=target_h、cy_n=target_cy。"""
    im = Image.fromarray(row["img"])
    h_n = row["h_n"]
    if h_n <= 1e-6:
        return None
    s = target_h / h_n                       # 需要的缩放比
    cx_in, cy_in = row["cx_n"] * INPUT_SIZE, row["cy_n"] * INPUT_SIZE
    cx_out, cy_out = 0.5 * INPUT_SIZE, target_cy * INPUT_SIZE
    # PIL.AFFINE 的 data 是 **输出坐标 → 输入坐标**：(x,y) → (a·x+b·y+c, d·x+e·y+f)
    a = e = 1.0 / s
    c = cx_in - cx_out / s
    f = cy_in - cy_out / s
    fill = tuple(int(v) for v in row["img"].reshape(-1, 3).mean(axis=0))
    return np.asarray(im.transform((INPUT_SIZE, INPUT_SIZE), Image.AFFINE,
                                   (a, 0, c, 0, e, f), resample=Image.BILINEAR, fillcolor=fill))


def run(sess, input_name, rows, target_h, target_cy, tag, render: bool = True):
    conf = np.zeros((3, 3), dtype=np.int64)
    for r in rows:
        if render:
            arr = render_live(r, target_h, target_cy)
            if arr is None:
                continue
        else:
            arr = r["img"]
        x = np.transpose(arr.astype(np.float32) / 255.0, (2, 0, 1))[None, ...].copy()
        logits = sess.run(None, {input_name: x})[0][0]
        conf[LABELS.index(r["label"]), int(np.argmax(logits))] += 1
    show(conf, tag)
    return conf


def show(conf: np.ndarray, tag: str) -> None:
    used = int(conf.sum())
    print(f"\n{'=' * 84}\n{tag}   (n={used})\n{'=' * 84}")
    print(f"{'真实':<10}{'→中性':>10}{'→开心':>10}{'→悲伤':>10}{'召回':>9}{'n':>7}")
    for i, c in enumerate(LABELS):
        tot = int(conf[i].sum())
        print(f"{c:<10}" + "".join(f"{conf[i, j] / max(1, tot) * 100:>9.1f}%" for j in range(3))
              + f"{conf[i, i] / max(1, tot) * 100:>8.1f}%{tot:>7}")
    print(f"{'总体':<10}{'':>30}{np.trace(conf) / max(1, used) * 100:>8.1f}%")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="线上渲染忠实评测")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--task", type=Path, default=DEFAULT_TASK)
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--limit", type=int, default=150, help="每类取样上限")
    ap.add_argument("--target-h", type=float, default=LIVE_H,
                    help=f"要渲染成的人脸高占比；默认 {LIVE_H}（前端实际产出）")
    args = ap.parse_args()

    import mediapipe as mp
    import onnxruntime as ort
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    print(f"[模型] {args.model.name}")
    sess = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    options = vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(args.task)),
        running_mode=vision.RunningMode.IMAGE, num_faces=1)
    landmarker = vision.FaceLandmarker.create_from_options(options)

    root = args.data / args.split
    paths: list[tuple[Path, str]] = []
    for c in LABELS:
        files = sorted([f for f in (root / c).iterdir() if f.is_file()])
        step = max(1, len(files) // args.limit)
        paths += [(f, c) for f in files[::step][:args.limit]]
    rows = measure_landmarks(paths, landmarker)
    print(f"[数据] {args.split} 共取 {len(paths)} 张，关键点可用 {len(rows)} 张")
    for c in LABELS:
        hs = [r["h_n"] for r in rows if r["label"] == c]
        if hs:
            print(f"       {c:<8} h_n 中位数 = {np.median(hs):.4f}  "
                  f"cy_n 中位数 = {np.median([r['cy_n'] for r in rows if r['label'] == c]):.4f}")

    # 1) 原图（训练口径，什么都不改）—— 基线
    raw = run(sess, input_name, rows, 0, 0, "对照组 A：原图直接喂（训练口径）", render=False)

    # 2) 忠实线上渲染：脸归一化到前端实际产出的口径
    live = run(sess, input_name, rows, LIVE_H, LIVE_CY,
               f"实验组 B：**线上渲染**（h_n={LIVE_H}, cy_n={LIVE_CY}）")

    # 3) 反事实：如果线上也渲染成 sad 的典型占比
    counter = run(sess, input_name, rows, SAD_TYPICAL_H, LIVE_CY,
                  f"反事实 C：渲染成 sad 的典型占比 h_n={SAD_TYPICAL_H}")

    def sad_recall(conf):
        tot = int(conf[LABELS.index("sad")].sum())
        return conf[LABELS.index("sad"), LABELS.index("sad")] / max(1, tot)

    print("\n" + "=" * 84)
    print("结论")
    print("=" * 84)
    print(f"  sad 召回：原图 {sad_recall(raw) * 100:.1f}%"
          f"  →  线上渲染 {sad_recall(live) * 100:.1f}%"
          f"  →  反事实 h_n={SAD_TYPICAL_H} {sad_recall(counter) * 100:.1f}%")
    print(f"  总体    ：原图 {np.trace(raw) / max(1, raw.sum()) * 100:.1f}%"
          f"  →  线上渲染 {np.trace(live) / max(1, live.sum()) * 100:.1f}%"
          f"  →  反事实 {np.trace(counter) / max(1, counter.sum()) * 100:.1f}%")
    drop = sad_recall(raw) - sad_recall(live)
    if drop > 0.25:
        print(f"\n  **sad 在线上渲染下掉了 {drop * 100:.1f}pp —— 病因是取景口径，不是用户的表情。**")
        print("  修法：把线上裁剪与训练口径统一（**不需要**采集链路），改完用本脚本复核。")
    else:
        print(f"\n  sad 在线上渲染下只变 {drop * 100:+.1f}pp —— 取景不是病因。")
        print("  剩下的解释是「训练域里没有你的脸 / 你的表情」，此时采集+标注+微调才是唯一解。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
