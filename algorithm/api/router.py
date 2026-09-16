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

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(perception_router)
api_router.include_router(assessment_router)
api_router.include_router(intervention_router)
api_router.include_router(escalation_router)
api_router.include_router(audit_router)
api_router.include_router(privacy_router)
api_router.include_router(federated_router)
