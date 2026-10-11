"""外部 FER 模型「能不能接进来、接进来准不准」的统一评估脚本。

要回答的问题
------------
用户问：**FER 模型能不能集成到我们这个板块里？**

能接的前提有三个，缺一不可，本脚本把三个都量出来：

1. **规格可描述**：输入尺寸/通道序/归一化/类别名都能写清楚（`specs.json`），
   而不是"下载来直接用"的黑盒 —— 归一化错了准确率会掉到随机水平。
2. **在我们自己的域上不掉分**：本项目数据集 `Face/`（96×96 原生）上，
   把外部模型的 7/8 类**折叠成** neutral/happy/sad 之后，准确率要能看。
3. **在真实取景下不掉分**：线上 `FacePatchExtractor` 永远产出
   `h_n≈0.661 / cy_n≈0.588`，而数据集里 sad 的典型占比是 0.774。
   于是**同一张图、同一模型**在不同取景下准确率可以差十几个百分点。
   外部模型的训练口径（AffectNet 是紧贴人脸的方框）**和我们的裁剪框不一样**，
   所以必须逐取景测，否则"接进来变差"会被误判成模型不行。

三段式输出（每个模型 × 每个取景）
--------------------------------
- `本地域`：`Face/validation`，真实标签只有 neutral/happy/sad
- `成人跨域`：`_fer_bench/<name>/test`，真实标签是 7 类
  （`fear/angry/disgust/surprise` 这些"折叠不到三分类"的类**不丢**，
   而是作为**假阳性来源**如实列出：真·恐惧被预测成 sad 就是 sad 的假阳性）
- `判定质量`：sad 的 precision/recall/F1 + AUC（按 P(sad) 秩和，无需阈值）

用法
----
    # 全部候选 + 我们的线上模型，三种取景
    & "D:\\Develop\\AIC\\.venv\\Scripts\\python.exe" scripts/fer_external_eval.py ^
        --framings native,live,tight --limit 150 --split validation

    # 单跑某个模型
    ... --specs D:\\Develop\\AIC\\_fer_candidates\\specs.json --only hsemotion

    # 加成人跨域基准
    ... --bench D:\\Develop\\AIC\\_fer_bench\\fer2013

    # 写报告
    ... --out D:\\Develop\\AIC\\_fer_candidates\\REPORT.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path(r"D:\Develop\AIC\数据集\数据集\Face")
DEFAULT_TASK = Path(r"D:\Develop\AIC\_mediapipe_models\face_landmarker.task")
DEFAULT_SPECS = Path(r"D:\Develop\AIC\_fer_candidates\specs.json")
DEFAULT_CACHE = Path(r"D:\Develop\AIC\_fer_eval_cache")
OUR_MODEL = APP_ROOT / "client" / "public" / "models" / "face_emotion_3cls.onnx"

# 三分类展示口径（= client/src/services/faceEmotionClassifier.ts 的 FACE_EMOTION_LABELS）
DISPLAY = ["neutral", "happy", "sad"]

# 外部模型自己的类名 → 我们的三分类。（None = 折叠不到三分类）
ALIASES: dict[str, str | None] = {
    "neutral": "neutral", "neutrality": "neutral", "calm": "neutral",
    "happy": "happy", "happiness": "happy", "joy": "happy", "amusement": "happy",
    "sad": "sad", "sadness": "sad", "sorrow": "sad",
    # 折叠不到三分类的：不算命中，但要统计它们被误判成什么
    "anger": None, "angry": None, "contempt": None, "disgust": None,
    "fear": None, "surprise": None, "fearful": None, "surprised": None,
    "disgusted": None, "contemptuous": None,
}

# 线上裁剪框的口径（由 face_framing_audit.py 实测：1195 张 + 4:3 画面）
LIVE_H = 0.6608
LIVE_CY = 0.5884
# AffectNet 系的训练口径：人脸方框基本填满画面
TIGHT_H = 0.90
TIGHT_CY = 0.50

FRAMINGS = {
    "native": None,                 # 数据集原图直接喂（= 各自的训练口径，上界参考）
    "live": (LIVE_H, LIVE_CY, 0.50),   # 线上实际产出
    "tight": (TIGHT_H, TIGHT_CY, 0.50),  # AffectNet 风格紧框（外部模型的期望）
}


# ─────────────────────────── 规格与预处理 ───────────────────────────

def our_spec() -> dict:
    return {
        "name": "ours_resnet50_v2_3cls",
        "onnx": str(OUR_MODEL),
        "input_size": 96,
        "color": "RGB",
        "grayscale": False,
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "classes": ["neutral", "happy", "sad"],
        "source_url": "in-repo (docs/face_emotion_model_integration.md)",
        "license": "project-internal",
        "notes": "线上模型；训练口径 = RGB + x/255，无 mean/std 归一化",
    }


def resolve_paths(spec: dict, specs_dir: Path) -> dict:
    """`onnx` 允许是相对 specs.json 的路径。"""
    p = Path(str(spec["onnx"]))
    if not p.is_absolute():
        cand = (specs_dir / p)
        p = cand if cand.exists() else (Path.cwd() / p)
    spec = dict(spec)
    spec["onnx"] = p
    return spec


def preprocess(arr: np.ndarray, spec: dict) -> np.ndarray:
    """RGB uint8 (H,W,3) → NCHW float32，按 spec 的尺寸/灰度/缩放/归一化。"""
    size = int(spec["input_size"])
    channels = int(spec.get("channels", 3))
    im = Image.fromarray(arr).convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.BILINEAR)
    if spec.get("grayscale"):
        im = im.convert("L").convert("RGB")
    x = np.asarray(im, dtype=np.float32)
    # `scale` 默认 1/255（绝大多数模型）。CNTK 导出的 FER+ ONNX 吃 **0~255 原始像素**，
    # 这时必须显式写 `scale: 1.0` —— 猜错就等于"在另一个分布上跑模型"，输出会退化成噪声。
    x = x * float(spec.get("scale", 1.0 / 255.0))
    mean = spec.get("mean")
    std = spec.get("std")
    if mean is not None and std is not None:
        m = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        s = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        if not (np.allclose(m, 0) and np.allclose(s, 1)):
            x = (x - m) / np.maximum(s, 1e-6)
    if str(spec.get("color", "RGB")).upper() == "BGR":
        x = x[:, :, ::-1].copy()
    if channels == 1:
        # 单通道模型（FER+）：BT.601 亮度，形状 [1,1,S,S]
        y = 0.299 * x[:, :, 0] + 0.587 * x[:, :, 1] + 0.114 * x[:, :, 2]
        return y[None, None, ...].copy()
    return np.transpose(np.ascontiguousarray(x), (2, 0, 1))[None, ...]


def to_probs(raw: np.ndarray) -> np.ndarray:
    """logits 或已经 softmax 过的都接：和≈1 且非负就原样用。"""
    raw = np.asarray(raw, dtype=np.float64).ravel()
    if raw.size == 0:
        return raw
    if raw.min() >= 0 and abs(raw.sum() - 1.0) < 1e-3:
        return raw
    e = np.exp(raw - raw.max())
    return e / e.sum()


# ─────────────────────────── 取景重渲染 ───────────────────────────

# OpenCV Zoo `facial_expression_recognition_mobilefacenet_2022july` 的标准 5 点模板（112×112）
OPENCV_5PT = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                       [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """相似变换（旋转+等比缩放+平移，**不含镜像**）的 3×3 矩阵：`dst ≈ M @ src`。

    与 OpenCV Zoo 的 `FaceAlignment.__findNonreflectiveSimilarity` 同一模型：
        dst_x = a·src_x − b·src_y + tx
        dst_y = b·src_x + a·src_y + ty
    4 个未知量用最小二乘解（5 个点 → 10 个方程）。
    """
    n = src.shape[0]
    A = np.zeros((2 * n, 4), dtype=np.float64)
    y = np.zeros(2 * n, dtype=np.float64)
    for i in range(n):
        x, yy = src[i]
        A[2 * i] = [x, -yy, 1, 0]
        A[2 * i + 1] = [yy, x, 0, 1]
        y[2 * i] = dst[i, 0]
        y[2 * i + 1] = dst[i, 1]
    a, b, tx, ty = np.linalg.lstsq(A, y, rcond=None)[0]
    return np.array([[a, -b, tx], [b, a, ty], [0.0, 0.0, 1.0]])


def align_face(arr: np.ndarray, pts5: list[list[float]], out_size: int = 112,
               mirror: bool = False) -> np.ndarray:
    """按 5 点把原图**相似变换对齐**到标准模板（OpenCV Zoo 的做法）。

    这一步正是 OpenCV Zoo 那个 4.6 MB 模型能打的原因之一：它的输入永远是
    "眼睛在同一高度、脸占满画面"的标准正面图，**取景/远近/轻微转头都被归一化掉了**。
    我们线上裁框恰恰相反（按关键点归一化，产出固定的 0.66 占比），
    对这类要求对齐的模型是**分布外**输入 —— 这也是本次要实测的假设。
    """
    im = Image.fromarray(arr)
    w, h = im.size
    src = np.array([[p[0] * w, p[1] * h] for p in pts5], dtype=np.float64)
    dst = OPENCV_5PT.copy()
    if mirror:
        # 对照：故意把左右点对调（验证"点序猜反"会不会明显变差）
        dst = dst[::-1]
    M = similarity_transform(src, dst)
    Minv = np.linalg.inv(M)
    data = (Minv[0, 0], Minv[0, 1], Minv[0, 2], Minv[1, 0], Minv[1, 1], Minv[1, 2])
    fill = tuple(int(v) for v in arr.reshape(-1, 3).mean(axis=0))
    out = im.transform((out_size, out_size), Image.AFFINE, data,
                       resample=Image.BILINEAR, fillcolor=fill)
    return np.asarray(out.convert("RGB"), dtype=np.uint8)


def render_framing(arr: np.ndarray, h_n: float, cx_n: float, cy_n: float,
                   target_h: float, target_cy: float, out_size: int) -> np.ndarray:
    """把整张图重采样，使**人脸关键点框**在输出里占 `target_h`、中心在 `target_cy`。

    公式（输出像素 → 输入像素的仿射，PIL.AFFINE 用的就是这个方向）：
        x_in = a·x_out + c,  a = (W_in/S)·(h_n/target_h),  c = cx_n·W_in − 0.5·S·a
        y_in = e·y_out + f,  e = (H_in/S)·(h_n/target_h),  f = cy_n·H_in − target_cy·S·e
    越界区域用**边缘均值**填充（真实背景是头发/房间，这里只能近似）。
    """
    im = Image.fromarray(arr)
    w_in, h_in = im.size
    if h_n <= 1e-6:
        return None
    k = h_n / max(1e-6, target_h)
    a = (w_in / out_size) * k
    e = (h_in / out_size) * k
    c = cx_n * w_in - 0.5 * out_size * a
    f = cy_n * h_in - target_cy * out_size * e
    fill = tuple(int(v) for v in arr.reshape(-1, 3).mean(axis=0))
    out = im.transform((out_size, out_size), Image.AFFINE, (a, 0, c, 0, e, f),
                       resample=Image.BILINEAR, fillcolor=fill)
    return np.asarray(out.convert("RGB"), dtype=np.uint8)


def measure_landmarks(paths: list[tuple[Path, str]], task: Path, cache_key: str) -> list[dict]:
    """MediaPipe 量关键点框；结果缓存到磁盘（一次几秒，反复跑不用重算）。"""
    cache = DEFAULT_CACHE / f"landmarks_{cache_key}.json"
    if cache.exists():
        rows = json.loads(cache.read_text(encoding="utf-8"))
        ok = {r["path"] for r in rows}
        if all(str(p) in ok for p, _ in paths):
            return rows

    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(task)),
        running_mode=vision.RunningMode.IMAGE, num_faces=1)
    lm = vision.FaceLandmarker.create_from_options(options)

    rows = []
    for p, lab in paths:
        try:
            with Image.open(p) as im:
                arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
        except Exception:
            continue
        if not res.face_landmarks:
            continue
        f = res.face_landmarks[0]
        xs = [q.x for q in f]
        ys = [q.y for q in f]
        # 5 点（OpenCV Zoo 的 facial_expression_recognition 用相似变换对齐到 112×112 模板）：
        #   右眼中心 = 33/133 中点，左眼中心 = 362/263 中点，鼻尖 = 1，
        #   右嘴角 = 61，左嘴角 = 291
        def mid(a: int, b: int) -> list[float]:
            return [(f[a].x + f[b].x) / 2, (f[a].y + f[b].y) / 2]
        pts5 = [mid(33, 133), mid(362, 263), [f[1].x, f[1].y], [f[61].x, f[61].y], [f[291].x, f[291].y]]
        rows.append({"path": str(p), "label": lab,
                     "w_n": max(xs) - min(xs), "h_n": max(ys) - min(ys),
                     "cx_n": (min(xs) + max(xs)) / 2, "cy_n": (min(ys) + max(ys)) / 2,
                     "pts5": pts5})
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return rows


# ─────────────────────────── 数据加载 ───────────────────────────

def load_flat(root: Path, limit: int | None, want: list[str] | None = None) -> list[tuple[Path, str]]:
    """`root/<class>/*` 布局，类名小写。"""
    out: list[tuple[Path, str]] = []
    for d in sorted([q for q in root.iterdir() if q.is_dir()], key=lambda q: q.name):
        cname = d.name.lower()
        if want and cname not in want:
            continue
        files = sorted([f for f in d.iterdir() if f.is_file()])
        if limit:
            # 均匀取样，不要只取前 limit 张（前若干张常常是同一个人）
            step = max(1, len(files) // limit)
            files = files[::step][:limit]
        out += [(f, cname) for f in files]
    return out


def load_dataset(root: Path, split: str, limit: int | None) -> list[tuple[Path, str]]:
    return load_flat(root / split, limit, want=list(DISPLAY))


def probe_label_order(model: "Model", images: list[np.ndarray], truth: list[str],
                      lines: list[str], top_k: int = 6) -> None:
    """**实测**模型的类别顺序，不依赖任何 README。

    做法有两步，互相独立：
      1. 打一张 `P(模型的第 j 类 | 真实第 i 类)` 均值表 —— 正确映射一眼可见：
         真实 sad 那一行，概率最高的模型类就是它的 sad。
      2. 穷举"从 C 个模型类里挑 3 个、对应到 neutral/happy/sad"的全部排列
         （8 类 = 8·7·6 = 336 种），按本地域准确率排序取前几名。

    为什么不能只信 README：类别顺序是"部署常量"，本项目自己就栽过两次
    （输入尺寸、`os.listdir` 顺序）。外部模型的名字顺序在不同导出之间也会变，
    所以最终以**在带标签数据上的实测一致性**为准。
    """
    probs = model.infer(images)
    if len(probs) == 0:
        return
    C = probs.shape[1]
    shown = [c for c in DISPLAY if c in set(truth)]

    lines.append("\n**实测类别顺序（方法 1）：`P(模型类 | 真实类)` 均值表**")
    lines.append("| 模型类＼真实 | " + " | ".join(shown) + " |")
    lines.append("|---|" + "---|" * len(shown))
    means = {}
    for j in range(C):
        row = []
        for c in shown:
            idx = [i for i, t in enumerate(truth) if t == c]
            row.append(float(probs[idx, j].mean()) if idx else float("nan"))
            means[(j, c)] = row[-1]
        lines.append(f"| {j}:{model.classes[j]} | " + " | ".join(f"{v * 100:.1f}%" for v in row) + " |")
    for c in shown:
        best = max(range(C), key=lambda j: means[(j, c)])
        lines.append(f"- 真实 `{c}` 最像的模型类 = **{best}:{model.classes[best]}** "
                     f"（{means[(best, c)] * 100:.1f}%）")

    # 方法 2：穷举排列，看本地域准确率
    import itertools
    best = []
    for combo in itertools.permutations(range(C), 3):
        hit = 0
        for i, t in enumerate(truth):
            if DISPLAY[int(np.argmax(probs[i][list(combo)]))] == t:
                hit += 1
        best.append((hit / len(truth), combo))
    best.sort(reverse=True)
    lines.append(f"\n**实测类别顺序（方法 2）：穷举 {C}·(C-1)·(C-2) 种映射（n={len(truth)}）**")
    for acc, combo in best[:top_k]:
        lines.append(f"- {acc * 100:5.1f}%  " +
                     " + ".join(f"{model.classes[j]}→{DISPLAY[k]}" for k, j in enumerate(combo)))
    lines.append(f"- 规格里声明的类名顺序（仅供参考，最终以方法 1/2 的实测为准）："
                 f"{model.classes}")


# ─────────────────────────── 推理 ───────────────────────────

class Model:
    def __init__(self, spec: dict):
        import onnxruntime as ort

        self.spec = spec
        self.path = Path(spec["onnx"])
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1  # 贴近浏览器单线程 wasm 的处境
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(self.path), so, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        self.classes = [str(c) for c in spec["classes"]]
        self.bytes = self.path.stat().st_size
        self.sha = hashlib.sha256(self.path.read_bytes()).hexdigest()[:16]

    def infer_one(self, arr: np.ndarray) -> np.ndarray:
        x = preprocess(arr, self.spec)
        out = self.sess.run(None, {self.input_name: x})[0][0]
        return to_probs(out)

    def infer(self, images: list[np.ndarray]) -> np.ndarray:
        return np.stack([self.infer_one(a) for a in images]) if images else np.zeros((0, len(self.classes)))

    def display_index(self, probs: np.ndarray) -> tuple[str | None, float, int | None]:
        """→ (折叠后的三分类名 或 None, 该类概率, 原索引)"""
        i = int(np.argmax(probs))
        cname = self.classes[i].lower() if i < len(self.classes) else ""
        # 找 P(sad)：类名里含 sad 的取最大
        return ALIASES.get(cname, None), float(probs[i]), i

    def p_sad(self, probs: np.ndarray) -> float:
        idx = [i for i, c in enumerate(self.classes) if c.lower() in ("sad", "sadness", "sorrow")]
        return float(probs[idx].sum()) if idx else float("nan")

    def p_map(self, probs: np.ndarray, target: str) -> float:
        idx = [i for i, c in enumerate(self.classes) if ALIASES.get(c.lower()) == target]
        return float(probs[idx].sum()) if idx else float("nan")


def auc_sad(labels: list[str], scores: np.ndarray) -> float:
    """AUC(true=sad vs 其余)，用秩和，不需要选阈值。"""
    labels = np.asarray(labels)
    m = ~np.isnan(scores)
    labels, scores = labels[m], scores[m]
    if len(set(labels)) < 2:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    pos = labels == "sad"
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# ─────────────────────────── 报告 ───────────────────────────

def report(model: Model, images: list[np.ndarray], truth: list[str], title: str,
           lines: list[str]) -> dict:
    probs = model.infer(images)
    preds = [model.display_index(probs[i]) for i in range(len(images))]
    shown = sorted(set(truth))
    offmap = "—"

    lines.append(f"\n### {title}")
    lines.append(f"- 模型：`{model.path.name}`  {model.bytes / 1048576:.1f} MB  "
                 f"输入 {model.spec['input_size']}px  sha256:{model.sha}…")
    lines.append(f"- 类别（模型自报顺序）：{model.classes}")
    lines.append(f"- n = {len(images)}，真实类别 = {shown}")
    if not images:
        return {}

    # 1) 模型**自己**的 argmax 落到哪一类（不做任何折叠）
    lines.append("\n**模型原始判定分布**（行=真实，列=模型自己的类名 argmax）")
    used = [c for c in model.classes]
    lines.append("| 真实 | " + " | ".join(used) + " |")
    lines.append("|---|" + "---|" * len(used))
    for t in shown:
        idx = [i for i, x in enumerate(truth) if x == t]
        if not idx:
            continue
        cnt = [0] * len(used)
        for i in idx:
            cnt[int(np.argmax(probs[i]))] += 1
        row = " | ".join(f"{c / len(idx) * 100:.1f}%" for c in cnt)
        lines.append(f"| {t} (n={len(idx)}) | {row} |")

    # 2) 折叠成三分类后的混淆矩阵
    lines.append("\n**折叠成三分类**（模型无对应类的预测记在 `其他`，不计入准确率分母）")
    lines.append("| 真实 | 中性 | 开心 | 悲伤 | 其他 | 召回 |")
    lines.append("|---|---|---|---|---|---|")
    conf = np.zeros((len(shown), 4), dtype=int)
    for i, t in enumerate(truth):
        r = shown.index(t)
        d = preds[i][0]
        c = DISPLAY.index(d) if d in DISPLAY else 3
        conf[r, c] += 1
    for r, t in enumerate(shown):
        tot = conf[r].sum()
        rec = conf[r, DISPLAY.index(t)] / max(1, tot) if t in DISPLAY else float("nan")
        lines.append(f"| {t} (n={tot}) | " + " | ".join(f"{v / max(1, tot) * 100:.1f}%" for v in conf[r])
                     + f" | {rec * 100:.1f}% |" if t in DISPLAY else
                     f"| {t} (n={tot}) | " + " | ".join(f"{v / max(1, tot) * 100:.1f}%" for v in conf[r])
                     + " | — |")
    mapped = sum(conf[r, :3].sum() for r in range(len(shown)) if shown[r] in DISPLAY)
    hit = sum(conf[r, DISPLAY.index(shown[r])] for r in range(len(shown)) if shown[r] in DISPLAY)
    if mapped:
        lines.append(f"\n- **三分类准确率 = {hit / mapped * 100:.1f}%**（可折叠样本 n={mapped}）")
    cover = sum(conf[r, :3].sum() for r in range(len(shown))) / max(1, conf.sum())
    lines.append(f"- 折叠覆盖率（预测落在三分类里的比例）= {cover * 100:.1f}%")

    # 3) sad 的判定质量
    scores = np.array([model.p_sad(probs[i]) for i in range(len(images))])
    tp = sum(1 for i, t in enumerate(truth) if t == "sad" and preds[i][0] == "sad")
    fp = sum(1 for i, t in enumerate(truth) if t != "sad" and preds[i][0] == "sad")
    fn = sum(1 for i, t in enumerate(truth) if t == "sad" and preds[i][0] != "sad")
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    lines.append(f"\n**sad 判定质量**：precision {prec * 100:.1f}%  recall {rec * 100:.1f}%  "
                 f"F1 {2 * prec * rec / max(1e-9, prec + rec) * 100:.1f}%  "
                 f"AUC(sad vs 其余) {auc_sad(truth, scores):.4f}")
    return {"acc": hit / mapped if mapped else float("nan"), "prec": prec, "rec": rec,
            "auc": auc_sad(truth, scores), "cover": cover,
            "onmap_rate": sum(1 for p in preds if p[0] in DISPLAY) / max(1, len(preds))}


def latency(model: Model, images: list[np.ndarray], n: int = 15) -> tuple[float, float]:
    sub = images[:n]
    if not sub:
        return float("nan"), float("nan")
    ts = []
    for a in sub:
        t0 = time.perf_counter()
        model.infer_one(a)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts) // 2], sum(ts) / len(ts)


# ─────────────────────────── 主流程 ───────────────────────────

def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="外部 FER 模型接入评估")
    ap.add_argument("--specs", type=Path, default=DEFAULT_SPECS)
    ap.add_argument("--only", type=str, default=None, help="只跑名字含该子串的模型")
    ap.add_argument("--no-ours", action="store_true", help="不跑线上模型基线")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--split", type=str, default="validation")
    ap.add_argument("--bench", type=Path, action="append", default=[],
                    help="成人跨域基准根目录（内含 <class>/*），可重复")
    ap.add_argument("--limit", type=int, default=150, help="每类取样上限")
    ap.add_argument("--framings", type=str, default="native",
                    help="native / live / tight 的逗号组合")
    ap.add_argument("--task", type=Path, default=DEFAULT_TASK)
    ap.add_argument("--latency", action="store_true", help="测单线程 CPU 单张延迟")
    ap.add_argument("--probe-labels", action="store_true",
                    help="实测模型的类别顺序（P(模型类|真实类) 均值表 + 穷举映射扫描），不依赖 README")
    ap.add_argument("--out", type=Path, default=None, help="把报告写到这里（markdown）")
    args = ap.parse_args()

    specs: list[dict] = []
    if args.specs.exists():
        raw = json.loads(args.specs.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = raw.get("models", [])
        specs = [resolve_paths(s, args.specs.parent) for s in raw]
    else:
        print(f"⚠️ 没有找到 {args.specs}，只评估线上模型")
    if not args.no_ours:
        specs = [our_spec()] + specs
    if args.only:
        specs = [s for s in specs if args.only.lower() in str(s.get("name", "")).lower()]
    if not specs:
        print("没有可评估的模型")
        return 2

    lines: list[str] = ["# 外部 FER 模型接入评估", ""]
    lines.append(f"- 数据集：`{args.data}\\{args.split}`，每类上限 {args.limit}")
    lines.append(f"- 取景：`{args.framings}`（live = h_n {LIVE_H} / cy_n {LIVE_CY}；"
                 f"tight = h_n {TIGHT_H} / cy_n {TIGHT_CY}）")
    lines.append("- 本报告的「三分类准确率」= 外部模型自己的类名折叠成 neutral/happy/sad 后，"
                 "在**可折叠**样本上的准确率；折叠不到的类（fear/angry/…）单独列在 `其他` 列，"
                 "不算命中也不算误判。")

    # ── 数据 ──
    local = load_dataset(args.data, args.split, args.limit)
    lines.append(f"\n本地域样本 n={len(local)}")
    benches: dict[str, list[tuple[Path, str]]] = {}
    for b in args.bench:
        if not b.exists():
            print(f"⚠️ 基准不存在，跳过：{b}")
            continue
        root = b / "test" if (b / "test").is_dir() else b
        benches[b.name] = load_flat(root, args.limit)
        lines.append(f"跨域基准 {b.name} n={len(benches[b.name])}")

    sources: list[tuple[str, list[tuple[Path, str]]]] = [("本地域 Face/" + args.split, local)]
    for name, items in benches.items():
        sources.append((f"跨域基准 {name}", items))

    framing_names = [f.strip() for f in args.framings.split(",") if f.strip() in FRAMINGS]

    # 关键点只量一次，全取景/全模型共用。
    # ⚠️ 触发条件必须包含"有模型要对齐"：曾漏掉这一条，于是 `--framings native` 下
    # 对齐模型拿到空的关键点表、一张图都没跑出来，报告静默缺行（不报错）——
    # 这类"没跑到却不报错"的坑比报错更危险。
    need_landmarks = any(f != "native" for f in framing_names) or any(s.get("align") for s in specs)
    lm_rows: dict[str, list[dict]] = {}
    if need_landmarks:
        for sname, items in sources:
            key = "v2_" + hashlib.sha1((sname + str(len(items)) + args.split).encode()).hexdigest()[:12]
            lm_rows[sname] = measure_landmarks(items, args.task, key)

    summary: list[dict] = []
    for spec in specs:
        try:
            model = Model(spec)
        except Exception as e:
            lines.append(f"\n## ❌ {spec.get('name')}：加载失败 {e}")
            continue
        lines.append(f"\n## {spec.get('name')}")
        lines.append(f"- 来源：{spec.get('source_url', '—')}    许可：{spec.get('license', '—')}")
        if spec.get("notes"):
            lines.append(f"- 说明：{spec['notes']}")
        if args.probe_labels:
            imgs = []
            tru = []
            for p, lab in local:
                with Image.open(p) as im:
                    imgs.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
                tru.append(lab)
            probe_label_order(model, imgs, tru, lines)

        align = str(spec.get("align") or "")
        # 对齐模型对取景不敏感（相似变换把取景归一化掉了），跑一遍即可
        run_framings = framing_names[:1] if align else framing_names
        for fname in run_framings:
            fr = FRAMINGS[fname]
            for sname, items in sources:
                images: list[np.ndarray] = []
                truth: list[str] = []
                idx = {r["path"]: r for r in lm_rows.get(sname, [])} if align else {}
                for p, lab in items:
                    with Image.open(p) as im:
                        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
                    if align:
                        # 对齐模型：取景被相似变换归一化掉了，framing 不再是变量
                        r = idx.get(str(p))
                        if not r or not r.get("pts5"):
                            continue
                        out = align_face(arr, r["pts5"], int(spec["input_size"]),
                                         mirror=bool(spec.get("align_mirror")))
                    elif fr is None:
                        out = arr
                    else:
                        r = idx.get(str(p))
                        if not r:
                            continue
                        target_h, target_cy, _ = fr
                        out = render_framing(arr, r["h_n"], r["cx_n"], r["cy_n"],
                                             target_h, target_cy, int(spec["input_size"]))
                    if out is None:
                        continue
                    images.append(out)
                    truth.append(lab)
                if not images:
                    lines.append(f"\n### [{fname}] {sname}\n"
                                 "- ⚠️ **0 张可用**（关键点缺失 / 对齐失败），本节为空 —— "
                                 "不要把它当成“跑过了”")
                    continue
                stats = report(model, images, truth, f"[{fname}] {sname}", lines)
                if stats:
                    stats.update({"model": spec.get("name"), "framing": fname, "source": sname})
                    summary.append(stats)
            if args.latency and fname == framing_names[0]:
                images = []
                for p, _ in local[:20]:
                    with Image.open(p) as im:
                        images.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
                med, avg = latency(model, images)
                lines.append(f"\n- 单线程 CPU 延迟：中位 {med:.0f} ms / 平均 {avg:.0f} ms（n={len(images)}）")

    # ── 汇总 ──
    lines.append("\n## 汇总")
    lines.append("| 模型 | 取景 | 数据 | 三分类准确率 | sad P | sad R | sad AUC | 折叠覆盖率 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in summary:
        lines.append(f"| {s['model']} | {s['framing']} | {s['source']} | {s['acc'] * 100:.1f}% | "
                     f"{s['prec'] * 100:.1f}% | {s['rec'] * 100:.1f}% | {s['auc']:.4f} | "
                     f"{s['cover'] * 100:.1f}% |")

    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"\n[写出] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
