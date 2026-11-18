"""「成人真实人脸」跨域对照：外部 FER 模型 vs 我们线上模型，在 **sad** 这一格上谁更强。

为什么必须单独做这一件事
------------------------
本地域（`数据集/Face`）上外部模型全线落后（见 `_fer_eval_cache/probe2.md`），
但那**不能直接推出**"外来的都不如自己的"：
本数据集的 sad 大量是**婴幼儿嚎哭/满脸泪水**（见 docs/face_fix_diagnosis.md），
而用户线上的痛点是**成年人的低强度难过**被读成中性。
两者根本不是同一个分布问题，必须换一个**成人、真实场景、彩色**的数据集来问。

本脚本就读 HF 上已下载的 **RAF-DB**（真实世界成人人脸，7 类，100×100 已对齐彩色图）
parquet，直接内存解码，不落中间图片，只输出对比表。

⚠️ 两个必须自己验证、不能照抄的点：
1. **数据集标签的整数含义**。RAF-DB 的类名顺序在不同镜像里 0/1 起始不一致，
   这里用 `CONVENTIONS` 同时跑两种，并打印 7×7 混淆矩阵 ——
   哪个约定下模型（尤其 HSEmotion 这种公开准确率 60%+ 的）对角明显，
   哪个才是真的。**这是"用模型反推标签含义"，不是相信 README。**
2. **跨模型的类别顺序**各自用自己的规格（已在 probe 阶段实测确认）。

用法
----
    & "D:\\Develop\\AIC\\.venv\\Scripts\\python.exe" scripts/fer_cross_domain_bench.py ^
        --parquet "D:\\Develop\\AIC\\_fer_bench\\_raw\\elipaluma_rafdb\\data\\test-00000-of-00001.parquet" ^
        --specs "D:\\Develop\\AIC\\_fer_eval_cache\\specs_variants.json" --per-class 150
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

# 复用同一套规格/预处理/推理（**必须是同一份代码**，否则跨域对比就失去意义）
from fer_external_eval import (DISPLAY, LIVE_CY, LIVE_H, Model, auc_sad, our_spec,
                               render_framing, to_probs)

# RAF-DB 的 7 类整数标签：两种常见起始约定都跑，让数据自己说话
CONVENTIONS: dict[str, dict[int, str]] = {
    "rafdb1(1起)": {1: "surprise", 2: "fear", 3: "disgust", 4: "happy", 5: "sad", 6: "angry", 7: "neutral"},
    "rafdb0(0起)": {0: "surprise", 1: "fear", 2: "disgust", 3: "happy", 4: "sad", 5: "angry", 6: "neutral"},
}

DEFAULT_TASK = Path(r"D:\Develop\AIC\_mediapipe_models\face_landmarker.task")


def measure_lm(images: list[np.ndarray], task: Path) -> list[dict | None]:
    """对内存里的图量人脸关键点框（跨域图不能复用数据集那份缓存）。"""
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(task)),
        running_mode=vision.RunningMode.IMAGE, num_faces=1)
    lm = vision.FaceLandmarker.create_from_options(options)
    out: list[dict | None] = []
    for arr in images:
        try:
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
        except Exception:
            out.append(None)
            continue
        if not res.face_landmarks:
            out.append(None)
            continue
        f = res.face_landmarks[0]
        xs = [q.x for q in f]
        ys = [q.y for q in f]
        out.append({"h_n": max(ys) - min(ys), "w_n": max(xs) - min(xs),
                    "cx_n": (min(xs) + max(xs)) / 2, "cy_n": (min(ys) + max(ys)) / 2})
    return out


def load_parquet(path: Path, per_class: int) -> tuple[list[np.ndarray], list[int]]:
    """读 `image`(struct<bytes>) + `label`(int) 两列；每类均匀取 `per_class` 张。"""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["image", "label"])
    imgs_col = table.column("image").to_pylist()
    labels = table.column("label").to_pylist()
    buckets: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        buckets.setdefault(int(lab), []).append(i)

    out_img: list[np.ndarray] = []
    out_lab: list[int] = []
    for lab, idxs in sorted(buckets.items()):
        step = max(1, len(idxs) // per_class)
        for i in idxs[::step][:per_class]:
            raw = imgs_col[i]["bytes"] if isinstance(imgs_col[i], dict) else imgs_col[i]
            try:
                with Image.open(__import__("io").BytesIO(raw)) as im:
                    arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
            except Exception:
                continue
            out_img.append(arr)
            out_lab.append(lab)
    return out_img, out_lab


def matrix(model: Model, probs: np.ndarray, truth_int: list[int],
           conv: dict[int, str]) -> tuple[np.ndarray, list[str]]:
    """→ (行=真实类 的 4 列计数矩阵[中性/开心/悲伤/其他], 真实类名列表)"""
    names = [conv.get(l, f"label{l}") for l in truth_int]
    order = sorted(set(names))
    conf = np.zeros((len(order), 4), dtype=int)
    for i, t in enumerate(names):
        p = probs[i]
        j = int(np.argmax(p))
        cname = model.classes[j].lower() if j < len(model.classes) else ""
        d = {"happy": "happy", "happiness": "happy", "joy": "happy",
             "neutral": "neutral", "neutrality": "neutral", "calm": "neutral",
             "sad": "sad", "sadness": "sad", "sorrow": "sad"}.get(cname)
        conf[order.index(t), DISPLAY.index(d) if d in DISPLAY else 3] += 1
    return conf, order


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="成人跨域 sad 对照")
    ap.add_argument("--parquet", type=Path, action="append", required=True)
    ap.add_argument("--specs", type=Path, default=None)
    ap.add_argument("--per-class", type=int, default=150)
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--heavy", action="store_true", help="也跑 327 MB 的 ViT（很慢，浏览器里不现实）")
    args = ap.parse_args()

    # 第一个 = 线上真实口径（重渲染）；第二个 = 故意喂原图的对照，
    # 用来暴露"紧取景会让我们的模型把所有成人脸都读成悲伤"这条取景捷径。
    specs: list[dict] = [
        dict(our_spec(), render_live=True),
        dict(our_spec(), render_live=False, name="ours(对照: RAF-DB 原图紧取景)"),
    ]
    if args.specs and args.specs.exists():
        import json
        raw = json.loads(args.specs.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = raw.get("models", [])
        # 去掉"故意做对照"的退化变体，别把结论搞浑
        specs += [s for s in raw if "no-norm" not in s["name"] and "(255)" not in s["name"]
                  and "mirror" not in s["name"]
                  and (args.heavy or "vit" not in s["name"].lower())]
    if args.only:
        keys = [k.strip().lower() for k in args.only.split(",") if k.strip()]
        specs = [s for s in specs if any(k in s["name"].lower() for k in keys)]

    lines: list[str] = ["# 成人真实人脸跨域对照（RAF-DB）", ""]
    for pq in args.parquet:
        images, labels = load_parquet(pq, args.per_class)
        lines.append(f"- `{pq.name}`：取 {len(images)} 张（每类上限 {args.per_class}），"
                     f"整数标签分布 {dict(sorted(Counter(labels).items()))}")
    lines.append("")

    # 我们线上模型必须**按线上裁剪口径重渲染**再测：
    # RAF-DB 是"脸几乎占满画面"的紧对齐图（h_n 接近 0.9~1.0），而线上永远是 h_n≈0.66。
    # 直接喂原图会落进模型的"脸占满→悲伤"捷径里（实测 7 类全部 100% 判成 sad），
    # 那是取景产物，不是悲伤识别能力 —— 这一点本身就是重要结论，所以两种都跑、都列出来。
    lms = measure_lm(images, DEFAULT_TASK) if any(s.get("render_h") or s.get("render_live")
                                                  for s in specs) else []
    ok_lm = sum(1 for r in lms if r)
    own_h = [r["h_n"] for r in lms if r]
    if lms:
        lines.append(f"- MediaPipe 在 RAF-DB 上测到关键点 {ok_lm}/{len(images)} 张"
                     f"（h_n 中位数 {np.median(own_h):.4f}，即**人脸本来几乎占满画面**；"
                     f"线上口径是 {LIVE_H}）")
        lines.append("")

    for conv_name, conv in CONVENTIONS.items():
        lines.append(f"\n## 标签约定：{conv_name}  `{conv}`")
        for spec in specs:
            try:
                model = Model(spec)
            except Exception as e:
                lines.append(f"\n### ❌ {spec.get('name')}：加载失败 {e}")
                continue
            # 每个模型吃"它自己的口径"：可以指定把它喂成哪种取景
            # （`render_h` = 人脸在裁框里的高度占比；不指定 = 喂原图）。
            # 为什么必须能调：线上裁剪框永远是 h_n≈0.66，而 AffectNet 系模型想要 ~0.9 的紧框；
            # 若不给它合适的取景，换模型省下的错误会被取景不匹配又还回去。
            rh = spec.get("render_h")
            if rh or spec.get("render_live"):
                rh = rh or LIVE_H
                rcy = float(spec.get("render_cy", LIVE_CY if spec.get("render_live") else 0.5))
                imgs, labs = [], []
                for arr, lm, lab in zip(images, lms, labels):
                    if not lm:
                        continue
                    out = render_framing(arr, lm["h_n"], lm["cx_n"], lm["cy_n"],
                                         float(rh), rcy, int(spec["input_size"]))
                    if out is not None:
                        imgs.append(out)
                        labs.append(lab)
                tag = f"（**重渲染成 h_n={float(rh)} / cy_n={rcy}**，{len(imgs)}/{len(images)} 张可用）"
            else:
                imgs, labs = images, list(labels)
                tag = "（原图）"
            probs = model.infer(imgs)
            conf, order = matrix(model, probs, labs, conv)
            lines.append(f"\n### {spec.get('name')} · {conv_name}　{tag}")
            lines.append("| 真实 | 中性 | 开心 | 悲伤 | 其他 | 召回 | n |")
            lines.append("|---|---|---|---|---|---|---|")
            hit = mapped = tp = fp = fn = 0
            for r, name in enumerate(order):
                tot = int(conf[r].sum())
                if name in DISPLAY:
                    correct = int(conf[r, DISPLAY.index(name)])
                    mapped += tot
                    hit += correct
                    if name == "sad":
                        tp += correct
                        fn += tot - correct
                    else:
                        fp += int(conf[r, DISPLAY.index("sad")])
                    rec = f"{correct / max(1, tot) * 100:.1f}%"
                else:
                    rec = "—"
                lines.append(f"| {name} | " +
                             " | ".join(f"{v / max(1, tot) * 100:.1f}%" for v in conf[r]) +
                             f" | {rec} | {tot} |")
            acc = hit / max(1, mapped)
            prec = tp / max(1, tp + fp)
            rc = tp / max(1, tp + fn)
            f1 = 2 * prec * rc / max(1e-9, prec + rc)
            # sad AUC：P(sad) 的秩和（不需要选阈值）
            scores = np.array([model.p_sad(probs[i]) for i in range(len(imgs))])
            is_sad = np.array([conv.get(l) == "sad" for l in labs])
            auc = auc_sad(["sad" if b else "x" for b in is_sad], scores)
            cover = float(np.mean([1.0 if (model.display_index(probs[i])[0] in DISPLAY) else 0.0
                                   for i in range(len(imgs))]))
            lines.append(f"\n- 三分类准确率（只看 happy/neutral/sad 三类）**{acc * 100:.1f}%**"
                         f"　sad：P {prec * 100:.1f}% / R {rc * 100:.1f}% / F1 {f1 * 100:.1f}%"
                         f"　AUC {auc:.4f}　折叠覆盖率 {cover * 100:.1f}%"
                         f"　（{model.bytes / 1048576:.1f} MB）")

    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"\n[写出] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
