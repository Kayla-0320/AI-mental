"""表情模型「取景口径」审计：量出数据集 96×96 裁图里人脸到底占多大、在什么位置。

为什么需要它
------------
`face_emotion_domain_shift_check.py` 测出两件要命的事：

1. **模型几乎只靠"嘴"判"开心"**：把眼/眉区抹掉，开心召回 96.2% → 97.5%（几乎无损）；
   把嘴区抹掉，开心召回塌到 17.5%。所以嘴在 96×96 patch 里的**位置和大小**是决定性的。
2. **模型对纵向位置极度敏感**：整幅上移 6px（96px 的 6.25%）→ 开心召回 96.2% → 11.2%，
   而且 **76.2% 的开心样本被判成"悲伤"** —— 这正是用户反馈的「咧嘴笑识别成悲伤」。
   下移 6px 则倒向"中性"（73.8%）。

结论：前端 `FACE_PATCH_MARGIN` 裁出来的框，必须让**人脸在 96×96 patch 里的占比与位置
和训练图一致**，否则结论会被直接改写。而"一致"是**可测量**的：数据集里的图就是模型
训练时的理想 patch，所以只要量出它们的人脸占比/位置，就能反解出前端该用的几何。

本脚本用**与前端同一个** MediaPipe FaceLandmarker（同一套关键点定义）量：

    h_n  = 关键点包围盒高度 / 图高      （人脸在 patch 里占的高度比例）
    w_n  = 关键点包围盒宽度 / 图宽
    cy_n = 包围盒中心 y / 图高          （人脸在 patch 里的纵向位置）
    cx_n = 包围盒中心 x / 图宽

再按这些统计量反解前端几何，并打印「现在 vs 应该」的对照。

用法
----
    & "D:\\Develop\\AIC\\.venv\\Scripts\\python.exe" scripts/face_framing_audit.py --limit 200
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path(r"D:\Develop\AIC\数据集\数据集\Face")
DEFAULT_TASK = Path(r"D:\Develop\AIC\_mediapipe_models\face_landmarker.task")

# ⚠️ 这是 `client/src/hooks/facePatchCanvas.ts` 里 `FACE_PATCH_MARGIN` 的**副本**。
#    那边改了这里就必须同步改，否则"现状 vs 目标"的对照会算错、把人带沟里。
#    改前（2026-10-10 之前）的值是 left/right 0.15 / top 0.35 / bottom 0.15，
#    它让人脸在 patch 里偏高 2.1px（详见该文件里的推导）。
FRONT_MARGIN = {"left": 0.147, "right": 0.149, "top": 0.390, "bottom": 0.123}
# 前端采集到的画面宽高比（useFacialAnalysis 请求 640×480）
FRAME_W, FRAME_H = 640, 480


def list_images(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in d.iterdir() if p.suffix.lower() in exts)


def measure(paths: list[Path], landmarker, limit: int | None):
    import numpy as np
    import mediapipe as mp
    from PIL import Image

    rows = []
    failed = 0
    for p in paths[:limit] if limit else paths:
        try:
            # ⚠️ 不要用 `mp.Image.create_from_file`：本机实测它在这些 PNG 上一律返回
            # "无脸"（200/200 失败），改用 PIL 解码 + `mp.Image(SRGB, ndarray)` 就正常。
            # 关键点是**归一化坐标**，与喂进去的边长无关，所以 96×96 直接喂即可
            # （已验证 96/192/256/384 四种边长结果一致到小数点后三位）。
            with Image.open(p) as im:
                arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
            res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
        except Exception:
            failed += 1
            continue
        if not res.face_landmarks:
            failed += 1
            continue
        lm = res.face_landmarks[0]
        xs = [q.x for q in lm]
        ys = [q.y for q in lm]
        w_n = max(xs) - min(xs)
        h_n = max(ys) - min(ys)
        if h_n <= 0 or w_n <= 0:
            failed += 1
            continue
        rows.append({
            "file": p.name,
            "w_n": w_n,
            "h_n": h_n,
            "cy_n": (min(ys) + max(ys)) / 2,
            "cx_n": (min(xs) + max(xs)) / 2,
            "aspect": w_n / h_n,
        })
    return rows, failed


def pct(vals, q):
    vals = sorted(vals)
    if not vals:
        return float("nan")
    i = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
    return vals[i]


def summarize(name: str, rows: list[dict], failed: int) -> dict:
    n = len(rows)
    if n == 0:
        print(f"[{name}] 未检测到任何人脸（失败 {failed}）")
        return {}
    out = {}
    print(f"\n[{name}] 检测成功 {n} 张，失败 {failed} 张"
          f"（成功率 {n / max(1, n + failed) * 100:.1f}%）")
    print(f"  {'量':<8}{'中位数':>9}{'p10':>9}{'p90':>9}{'最小':>9}{'最大':>9}")
    for key, label in (("h_n", "h_n"), ("w_n", "w_n"), ("cy_n", "cy_n"),
                       ("cx_n", "cx_n"), ("aspect", "w/h")):
        vals = [r[key] for r in rows]
        out[key] = st.median(vals)
        print(f"  {label:<8}{st.median(vals):>9.4f}{pct(vals, 0.10):>9.4f}"
              f"{pct(vals, 0.90):>9.4f}{min(vals):>9.4f}{max(vals):>9.4f}")
    return out


def front_current(h_n: float) -> tuple[float, float, float]:
    """按前端现有 `computeFacePatchBox` 算：人脸在 96px patch 里的 (高度占比, 中心y占比, 侧边/包围盒高)。

    推导（设关键点包围盒在 4:3 画面里是 wb×hb 像素，纵向中心 cyb）：
      top = y0 - a·hb, bottom = y1 + b·hb   → 框高 = (1+a+b)·hb
      left = x0 - c·wb, right = x1 + d·wb   → 框宽 = (1+c+d)·wb
      side = max(框宽, 框高)；patch 里人脸高占比 = hb/side
      patch 里人脸纵向中心 = 0.5 - (b-a)·(hb/side)/2
    """
    a, b = FRONT_MARGIN["top"], FRONT_MARGIN["bottom"]
    c, d = FRONT_MARGIN["left"], FRONT_MARGIN["right"]
    # 用数据集自己的 w/h 比例把"像素"换算统一到 hb 的倍数上
    wb_over_hb = None  # 由调用方注入
    return a, b, c  # 占位，真实计算在 recommend() 里


def recommend(stats: dict) -> None:
    """反解出前端应该用的几何，并与现状对照。"""
    h_n, w_n = stats["h_n"], stats["w_n"]
    cy_n, cx_n = stats["cy_n"], stats["cx_n"]
    a, b = FRONT_MARGIN["top"], FRONT_MARGIN["bottom"]
    c, d = FRONT_MARGIN["left"], FRONT_MARGIN["right"]

    # ── 现状：在真实 4:3 画面里算 ──
    # 用数据集量到的 w/h 比例当作"真人脸"的像素比例（同一批脸，比例可迁移）
    aspect = stats["aspect"]                      # wb/hb
    # 画面里 hb 取 1 个单位做归一化
    hb = 1.0
    wb = aspect
    box_h = (1 + a + b) * hb
    box_w = (1 + c + d) * wb * (FRAME_H / FRAME_W)  # 归一化坐标→像素时的 4:3 修正
    side = max(box_h, box_w)
    cur_face_h = hb / side
    cur_face_cy = 0.5 - (b - a) * (hb / side) / 2
    cur_face_cx = 0.5 - (d - c) * (wb * (FRAME_H / FRAME_W) / side) / 2

    print("\n" + "=" * 88)
    print("现状 vs 目标：人脸在 96×96 patch 里应该占多大、在什么位置")
    print("=" * 88)
    print(f"  {'':<18}{'人脸高占比':>12}{'人脸中心y':>12}{'人脸中心x':>12}")
    print(f"  {'数据集（目标）':<18}{h_n:>12.4f}{cy_n:>12.4f}{cx_n:>12.4f}")
    print(f"  {'前端现状':<18}{cur_face_h:>12.4f}{cur_face_cy:>12.4f}{cur_face_cx:>12.4f}")
    print(f"  {'偏差':<18}{cur_face_h - h_n:>+12.4f}"
          f"{cur_face_cy - cy_n:>+12.4f}{cur_face_cx - cx_n:>+12.4f}")

    # ── 反解：要求 side = hb/h_n，且 patch 内中心落在 (cx_n, cy_n) ──
    side = hb / h_n
    # cropTop = cyb - cy_n·side；同时 cropTop = (cyb + (b-a)/2·hb) - side/2
    # → (b-a)/2·hb = side·(0.5 - cy_n)
    b_minus_a = 2 * (0.5 - cy_n) * hb / h_n
    a_sum_b = 1 / h_n - 1
    # 横向：框宽必须 ≤ side（否则 max() 会选中宽度而改变缩放）
    # patch 内人脸宽占比 = wb_px/side，希望等于 w_n（数据集口径，图宽归一）
    # wb_px = w_n·FRAME_W, side = hb_px/h_n；两者相除即可，这里给的是 c/d 的解
    wb_px_over_hb_px = w_n * FRAME_W / (h_n * FRAME_H)
    d_minus_c = 2 * (0.5 - cx_n) * (wb_px_over_hb_px / h_n)

    rec_a = (a_sum_b - b_minus_a) / 2
    rec_b = (a_sum_b + b_minus_a) / 2

    # 横向：希望框宽恰好等于 side → (1+c+d)·wb_px = side = hb_px/h_n
    c_sum_d = (1 / h_n) / wb_px_over_hb_px - 1
    rec_c = (c_sum_d - d_minus_c) / 2
    rec_d = (c_sum_d + d_minus_c) / 2

    print("\n" + "=" * 88)
    print("反解出的前端裁剪参数（替代 facePatchCanvas.ts 的 FACE_PATCH_MARGIN）")
    print("=" * 88)
    print(f"  {'':<10}{'top':>10}{'bottom':>10}{'left':>10}{'right':>10}")
    print(f"  {'现状':<10}{a:>10.3f}{b:>10.3f}{c:>10.3f}{d:>10.3f}")
    print(f"  {'建议':<10}{rec_a:>10.3f}{rec_b:>10.3f}{rec_c:>10.3f}{rec_d:>10.3f}")
    print(f"\n  等效方框边长 / 关键点包围盒高 = {1 / h_n:.3f}（现状是 {1 + a + b:.3f}）")
    print(f"  → 现状把脸**缩小**了 {cur_face_h / h_n:.2f} 倍；模型只在这种占比上训练过。")
    print(f"  → 纵向偏移 {cur_face_cy - cy_n:+.4f}（patch 高度的比例）；"
          f"实测上移 0.0625 就会把 76% 的开心改判成悲伤。")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="数据集取景口径审计（MediaPipe FaceLandmarker）")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--task", type=Path, default=DEFAULT_TASK)
    ap.add_argument("--splits", type=str, default="validation,training")
    ap.add_argument("--limit", type=int, default=200, help="每个类别取样上限（0=全量）")
    args = ap.parse_args()

    if not args.task.exists():
        raise SystemExit(f"缺少 FaceLandmarker 模型：{args.task}\n"
                         f"下载：Invoke-WebRequest "
                         f"'https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                         f"face_landmarker/float16/1/face_landmarker.task' -OutFile <path>")

    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(args.task)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    all_rows: list[dict] = []
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        root = args.data / split
        if not root.is_dir():
            print(f"[{split}] 目录不存在，跳过")
            continue
        for cname in sorted([d.name for d in root.iterdir() if d.is_dir()]):
            files = list_images(root / cname)
            if not files:
                continue
            lim = args.limit or None
            step = max(1, len(files) // lim) if lim else 1
            picked = files[::step][:lim] if lim else files
            rows, failed = measure(picked, landmarker, None)
            s = summarize(f"{split}/{cname}", rows, failed)
            all_rows += rows

    if not all_rows:
        raise SystemExit("没量到任何人脸，无法给出建议")

    print("\n" + "=" * 88)
    print("合计")
    print("=" * 88)
    overall = summarize("全部", all_rows, 0)
    recommend(overall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
