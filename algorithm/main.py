"""
青少年 AI 心理健康平台 —— 后端算法服务入口
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.router import api_router

app = FastAPI(
    title="青少年AI心理健康平台 - 算法服务",
    description="多模态情绪感知、心理评估、智能干预、危机升级一体化算法后端",
    version="0.1.0",
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
