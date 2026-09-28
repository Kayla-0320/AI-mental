"""用真实千问复现一段对话，检查"套路感"是否消失。

背景（2026-09-26 用户截图）：连续 4 轮回复都是同一个骨架 ——
    「我能感受到你现在X。……发生了什么事 / 是什么让你……？」
用户反馈："这种回答很套路，只能令患者更加无语。"

改完 system prompt 后，这个脚本把截图里那段对话**原样重放**，
并把命中套路特征的地方标出来，用真实输出（而不是模拟）判断效果。

用法（从 ``algorithm/`` 目录运行）：

    python tools/replay_chat_style.py

只读：只调用 DashScope，不写任何文件。需要 ``DASHSCOPE_API_KEY``。
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _find_env() -> Path | None:
    """向上找到同时含 ``algorithm/`` 与 ``server/`` 的那层目录下的 .env。

    刻意不写死 ``parents[N]``：这个仓库的嵌套是
    ``AIC/AI-mental-main (2)/AI-mental-main/AI-mental-main/``，
    写死深度后只要目录再嵌一层就会静默读不到 key，
    而这正是 2026-09-26 那次"聊天失败卡片"的同一个坑（Node 读了错误的 .env）。
    """
    for base in Path(__file__).resolve().parents:
        if (base / "algorithm").is_dir() and (base / "server").is_dir():
            env_path = base / ".env"
            return env_path if env_path.is_file() else None
    return None


def _load_env() -> str:
    """按 main.py 的方式加载应用根 .env，返回 key 的来源说明。

    这里用直接赋值而不是 ``setdefault``：本脚本的目的是实测**当前仓库里这把 key**，
    若父进程恰好设过一个坏值，``setdefault`` 会静默沿用坏值，测出来的就是别人的结果。
    """
    env_path = _find_env()
    if env_path is None:
        return "（未找到应用根 .env）"
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ[k.strip()] = v.strip().strip('"').strip("'")
    return str(env_path)


from api import intervention as iv  # noqa: E402

from intervention.style_detector import UserStyle  # noqa: E402

# ⚠️ 必须在 import api.intervention 之前把 .env 灌进 os.environ：
# 该模块在模块级就把 ``_LLM_BASE_URL`` / ``_LLM_MODEL`` 固化成常量了，
# 先 import 再设环境变量，基址会一直是默认值（静默走错服务）。
_LOADED_ENV = _load_env()

from api import intervention as iv  # noqa: E402

# 场景一：截图里那段对话（用户原话，逐字）
SCREENSHOT: list[str] = [
    "我现在很难受",
    "我考试没考好",
    "老师骂我了",
    "他说我很废物",
]

# 场景二：换一组完全不同的处境，用来确认第一版的改善不是"只对截图过拟合"。
# 这条线是"回家/父母"场景，且第 1 句就含具体事件（与场景一不同）。
PARENTS: list[str] = [
    "我一想到明天要去学校就心慌",
    "爸妈昨天又吵架了",
    "我妈说要不是我他们早就离了",
    "我不知道该怪谁",
]

# 场景三：用户第二张截图里的真实路径（寒暄 → 一句泛泛的难受）。
GREETING: list[str] = [
    "你好",
    "我有点难受",
]

REPLAYS: dict[str, list[str]] = {
    "场景一 · 截图原对话（老师/考试）": SCREENSHOT,
    "场景二 · 家庭冲突（父母争吵）": PARENTS,
    # 2026-09-26 第二张截图：用户只说「你好」，AI 回"你今天看起来有点没精神啊？"
    # —— 凭空捏造用户状态。这组用来验证寒暄轮次不再编状态。
    "场景三 · 寒暄开场（你好 / 我有点难受）": GREETING,
}

# 套路特征（出现即扣分，用于人工判断，不做自动化断言）
_BANNED_OPENING = re.compile(r"^(我能感受到你|听起来你|我理解你)")
_SKELETON = "我能感受到你现在"


def _question_count(text: str) -> int:
    return text.count("？") + text.count("?")


def _call_and_capture(user_msg: str, history: list, probs: list) -> tuple[str, str]:
    """调用一次并同时返回（定稿后回复, 模型原话）。

    只看最终文本无法判断"这句话是不是被代码改坏的"（本次就遇到过
    「老师骂你了，当时大概不好受？」被压成「当时大概不好受。」）。
    所以这里复制一份 _call_llm 的流程，只为把**原始输出**也拿到手。
    这是只读的诊断工具，不参与生产路径。

    Args:
        user_msg: 用户本轮发言。
        history: 对话历史。
        probs: 情绪概率。

    Returns:
        tuple[str, str]: ``(定稿后, 模型原话)``。
    """
    from intervention.style_detector import UserStyle

    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    ask_question = iv._ask_question_this_turn(
        UserStyle.CALM,
        "low",
        user_message=user_msg,
        negative_intensity=iv._negative_emotion_intensity(probs),
    )
    messages = iv._build_llm_messages(
        user_msg, history, probs, "low", style=UserStyle.CALM, ask_question=ask_question
    )
    resp = iv.requests.post(
        f"{iv._LLM_BASE_URL}/chat/completions",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        json={
            "model": iv._LLM_MODEL,
            "messages": messages,
            "temperature": 0.85,
            "max_tokens": 200,
        },
        timeout=iv._LLM_TIMEOUT,
    )
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    final = iv._finalize_reply(
        raw,
        ask_question and not iv._is_social_only(user_msg),
        user_message=user_msg,
    )
    return final, raw


def main() -> int:
    src = _LOADED_ENV
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    print(f"[env] {src} → key {'已加载' if api_key else '缺失'}")
    print(f"[env] base_url={iv._LLM_BASE_URL} model={iv._LLM_MODEL}")
    if not api_key:
        print("缺少 DASHSCOPE_API_KEY，无法实测")
        return 1

    # 情绪概率用中性偏高的占位值：这里只考察**话术形态**，
    # 不考察情绪分类精度（分类器由 perception 层负责）。
    probs = [0.13, 0.07, 0.0, 0.0, 0.80]

    total_banned = 0
    total_turns = 0
    for scene_name, turns in REPLAYS.items():
        print("\n" + "=" * 72)
        print(f"### {scene_name}")
        print("=" * 72)

        history: list[dict] = []
        replies: list[str] = []
        for turn, user_msg in enumerate(turns, 1):
            # 先取一份 prompt 看本轮实际注入了哪条 ask_rule ——
            # 模型不遵守时，第一步是确认规则到底有没有送到它手里。
            sys_prompt = iv._build_llm_messages(
                user_msg, history, probs, "low", style=UserStyle.CALM
            )[0]["content"]
            rule_line = [
                ln
                for ln in sys_prompt.splitlines()
                if "本轮**" in ln or "前面已经连着" in ln
            ]
            repeated = iv._assistant_asked_repeatedly(history)

            reply, raw = _call_and_capture(user_msg, history, probs)
            replies.append(reply)
            print(f"[{turn}] 用户: {user_msg}")
            print(f"     AI : {reply}")
            print(f"          （问号 {_question_count(reply)} 个，{len(reply)} 字）")
            print(f"          [注入] 连续追问检测={repeated} → {' / '.join(rule_line)[:70]}")
            if raw != reply:
                print(f"          [定稿改过] 模型原话={raw}")
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": reply})
            print("-" * 72)

        # ── 套路特征统计 ──
        n = len(turns)
        total_turns += n
        print("\n【套路特征检查】")
        banned_hits = [r for r in replies if _BANNED_OPENING.match(r.strip())]
        total_banned += len(banned_hits)
        skeleton_hits = [r for r in replies if _SKELETON in r]
        print(f"  以「我能感受到你/听起来你/我理解你」开头: {len(banned_hits)}/{n}")
        for r in banned_hits:
            print(f"      ! {r}")
        print(f"  含「{_SKELETON}」: {len(skeleton_hits)}/{n}")
        for r in skeleton_hits:
            print(f"      ! {r}")

        asking = [r for r in replies if _question_count(r) > 0]
        print(f"  含问句的轮次: {len(asking)}/{n}")
        # 注意用 1 基编号，和人读对话的编号对齐（此前用 0 基，报出来的位置是错的）
        consecutive = [
            (i + 1, i + 2)
            for i in range(n - 1)
            if _question_count(replies[i]) > 0 and _question_count(replies[i + 1]) > 0
        ]
        print(f"  连续两轮都追问的位置: {consecutive or '无'}")

        # 是否接住了用户说过的具体内容（截图里 AI 四轮都没接住"老师""废物"）
        said = [t for t in turns]
        caught = [
            i
            for i, r in enumerate(replies, 1)
            # 取用户该轮发言里的 2 字以上片段做粗匹配
            if any(
                frag in r
                for frag in re.findall(r"[\u4e00-\u9fa5]{2,}", said[i - 1])
                if frag not in ("一个", "什么", "怎么")
            )
        ]
        print(f"  回应里出现用户说过的具体词: 第 {caught or '（无）'} 轮")

    print("\n" + "=" * 72)
    print(f"【合计】被禁开场: {total_banned}/{total_turns}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
