"""
API 路由注册表 —— 统一注册所有模块路由
"""
from fastapi import APIRouter

from api.perception import router as perception_router
from api.assessment import router as assessment_router
from api.intervention import router as intervention_router
from api.escalation import router as escalation_router
from api.audit import router as audit_router
from api.privacy import router as privacy_router
from api.federated import router as federated_router
from api.sim_vail import router as sim_vail_router
from api.phenotype import router as phenotype_router
from api.digital_twin import router as digital_twin_router
from api.asr import router as asr_router
from api.tts import router as tts_router
from api.research import router as research_router
from api.innovation import router as innovation_router
from api.knowledge import router as knowledge_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(perception_router)
api_router.include_router(assessment_router)
api_router.include_router(intervention_router)
api_router.include_router(escalation_router)
api_router.include_router(audit_router)
api_router.include_router(privacy_router)
api_router.include_router(federated_router)
api_router.include_router(sim_vail_router)
api_router.include_router(phenotype_router)
api_router.include_router(digital_twin_router)
# 本地语音识别（替代浏览器 Web Speech API）
api_router.include_router(asr_router)
# 本地语音合成（陪伴通话的出声端；音频不出设备）
api_router.include_router(tts_router)
# 第二轮预测单独鉴权，不接入现有风险与升级链路。
api_router.include_router(research_router)
api_router.include_router(innovation_router)
# 知识图谱只读研究接口，默认由 SUPPORT_KNOWLEDGE_ENABLED 关闭。
api_router.include_router(knowledge_router)
