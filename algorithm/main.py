"""
青少年 AI 心理健康平台 —— 后端算法服务入口
"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

# uvicorn 的日志配置只挂到 "uvicorn" 这个 logger 上，root 仍无 handler ——
# 本模块 logger.info 的消息会被静默丢弃（lastResort 只输出 WARNING 以上），
# 导致 [startup] 预加载的日志一行都看不到。
logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时顺序预加载所有 HuggingFace 模型。

    目的：
    1. 用户打开网页时无需等待 40+ 秒的冷启动
    2. 顺序加载避免并发 from_pretrained 段错误（exit 139）

    注意：TTS 不在此预加载，因为它占用 4.3GB VRAM，会导致合成时显存不足。
    TTS 保持懒加载，第一次合成时才加载到 GPU。

    即使 startup 预加载失败，后续的懒加载仍会尝试（兜底逻辑不变）。
    """
    logger.info("[startup] 开始顺序预加载模型（约 15-20 秒）...")

    # 1. ASR 流式识别（sherpa-onnx，约 2 秒，CPU）
    try:
        from asr.streaming import StreamingRecognizer
        StreamingRecognizer.instance()
        logger.info("[startup] ASR 流式识别就绪")
    except Exception as exc:
        logger.warning("[startup] ASR 预加载失败（懒加载仍可工作）：%s", exc)

    # 2. 场景检索编码器（BGE，约 5 秒，CPU）
    try:
        from intervention.scene_retrieval.embedder import get_default_embedder
        embedder = get_default_embedder()
        if hasattr(embedder, "_ensure_loaded"):
            embedder._ensure_loaded()
        logger.info("[startup] 场景检索编码器就绪")
    except Exception as exc:
        logger.warning("[startup] 场景检索编码器预加载失败（懒加载仍可工作）：%s", exc)

    # 3. 认知扭曲分类器（约 10 秒，CPU）
    try:
        from perception.cognitive.distortion_classifier import DistortionClassifier
        classifier = DistortionClassifier.load()
        if hasattr(classifier, "_ensure_loaded"):
            classifier._ensure_loaded()
        logger.info("[startup] 认知扭曲分类器就绪")
    except Exception as exc:
        logger.warning("[startup] 认知扭曲分类器预加载失败（懒加载仍可工作）：%s", exc)

    logger.info("[startup] 小模型预加载完成（TTS 保持懒加载以节省 VRAM）")
    yield  # 应用运行
    logger.info("[shutdown] 服务关闭")


def _load_env() -> None:
    """加载仓库根目录的 .env。

    背景：此前 algorithm/ 侧完全没有加载 .env 的逻辑（全目录既无 load_dotenv
    也无 pydantic-settings），而 api/intervention.py 却用
    ``os.environ.get('DASHSCOPE_API_KEY', '')`` 取 key —— 结果是永远取到空串，
    /smart-chat 静默退回模板回复：看似接入了千问，实际没接。

    必须在导入任何「模块级读取环境变量」的子模块之前调用
    （例如 perception/text_emotion/config.py 在模块级读 USE_CPU）。

    优先用 python-dotenv；未安装时退回极简解析器，避免新增硬依赖。
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    try:
        from dotenv import load_dotenv

        # override=False：真实环境变量优先于 .env，符合十二要素应用惯例
        load_dotenv(env_path, override=False)
        return
    except Exception:
        pass

    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            name = name.strip()
            if name and name not in os.environ:
                os.environ[name] = value.strip().strip('"').strip("'")
    except Exception:
        pass


_load_env()

from api.router import api_router  # noqa: E402  (必须在 _load_env() 之后)

app = FastAPI(
    title="青少年AI心理健康平台 - 算法服务",
    description="多模态情绪感知、心理评估、智能干预、危机升级一体化算法后端",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 中间件（允许前端跨域调用）
# 注意：allow_origins=["*"] 与 allow_credentials=True 不兼容（CORS 规范）
# 前端 fetch 调用不携带 credentials，因此 allow_credentials 设为 False
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册所有路由
app.include_router(api_router)


@app.get("/health", tags=["系统"])
async def health_check():
    """健康检查接口"""
    return {"status": "ok", "service": "algorithm-backend"}
