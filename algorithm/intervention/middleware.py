"""
安全审计中间件 —— 包装 LLM 调用，输入输出经过审计层

架构：
    用户输入 → [预处理] → LLM → [审计层] → 用户输出
                                      ↓
                                  不通过 → 重写/注入资源/升级人工

不依赖具体 LLM 实现，只定义接口。
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

from intervention.auditor import (
    AuditContext,
    SafetyAuditor,
    generate_crisis_resource_card,
    load_audit_config,
)
from shared.dataclasses import AuditAction, AuditVerdict


# ============================================================
# 日志配置
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# LLM 接口定义（不依赖具体实现）
# ============================================================

class LLMProvider(Protocol):
    """LLM 提供者接口

    任何 LLM 实现只需实现此接口即可接入审计中间件。
    """

    def generate(
        self,
        prompt: str,
        context: Optional[dict] = None,
    ) -> "LLMResponse":
        """生成回复

        Args:
            prompt: 输入提示
            context: 上下文信息

        Returns:
            LLMResponse: LLM 响应
        """
        ...


@dataclass
class LLMResponse:
    """LLM 响应数据结构

    Attributes:
        content: 生成的文本内容
        confidence: 置信度 [0, 1]
        metadata: 额外元数据
    """
    content: str
    confidence: float = 0.5
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ============================================================
# 重写器接口
# ============================================================

class ResponseRewriter(Protocol):
    """回复重写器接口

    当审计不通过时，用于重写 LLM 回复。
    """

    def rewrite(
        self,
        original_response: str,
        audit_verdict: AuditVerdict,
        context: AuditContext,
    ) -> str:
        """重写回复

        Args:
            original_response: 原始 LLM 回复
            audit_verdict: 审计裁决
            context: 审计上下文

        Returns:
            重写后的回复
        """
        ...


class DefaultRewriter:
    """默认重写器 —— 基于规则的简单重写

    实际生产中可替换为 LLM 驱动的重写器。
    """

    def rewrite(
        self,
        original_response: str,
        audit_verdict: AuditVerdict,
        context: AuditContext,
    ) -> str:
        """基于审计结果进行简单重写"""
        rewritten = original_response

        # 移除诊断性标签（简单替换）
        diagnostic_patterns = [
            ("你患有", "你可能正在经历"),
            ("你得了", "你可能正在经历"),
            ("你确诊", "你被评估为"),
            ("你是个抑郁症", "你正在经历抑郁情绪"),
            ("你疯了", "你的感受是可以理解的"),
        ]
        for old, new in diagnostic_patterns:
            rewritten = rewritten.replace(old, new)

        # 移除污名化用语
        stigma_patterns = [
            ("疯子", "正在经历困难的人"),
            ("神经病", "正在经历困扰的人"),
            ("矫情", "敏感"),
            ("太脆弱", "正在经历挑战"),
            ("不够坚强", "正在寻找力量"),
        ]
        for old, new in stigma_patterns:
            rewritten = rewritten.replace(old, new)

        return rewritten


# ============================================================
# 审计中间件
# ============================================================

@dataclass
class MiddlewareConfig:
    """中间件配置

    Attributes:
        enable_audit: 是否启用审计
        max_rewrite_attempts: 最大重写次数
        enable_logging: 是否记录日志
        auto_escalate: 是否自动升级人工
    """
    enable_audit: bool = True
    max_rewrite_attempts: int = 2
    enable_logging: bool = True
    auto_escalate: bool = True


class AuditedLLMMiddleware:
    """审计中间件 —— 包装 LLM 调用

    使用方式：
        # 1. 创建 LLM 提供者（实现 LLMProvider 接口）
        class MyLLM:
            def generate(self, prompt, context=None):
                return LLMResponse(content="生成的回复", confidence=0.8)

        # 2. 创建中间件
        middleware = AuditedLLMMiddleware(
            llm_provider=MyLLM(),
            rewriter=DefaultRewriter(),
        )

        # 3. 调用（自动经过审计层）
        response = middleware.generate(
            prompt="用户输入",
            dialog_state="EXPLORE",
            turn_count=5,
        )
    """

    def __init__(
        self,
        llm_provider: Any,  # LLMProvider
        rewriter: Optional[Any] = None,  # ResponseRewriter
        config: Optional[MiddlewareConfig] = None,
    ) -> None:
        self.llm_provider = llm_provider
        self.rewriter = rewriter or DefaultRewriter()
        self.config = config or MiddlewareConfig()
        self.auditor = SafetyAuditor()
        self._audit_config = load_audit_config()

    def generate(
        self,
        prompt: str,
        dialog_state: str = "INIT",
        turn_count: int = 0,
        crisis_turns: int = 0,
        risk_level: str = "low",
        conversation_history: Optional[list[dict]] = None,
        current_topic: str = "",
        **kwargs,
    ) -> LLMResponse:
        """生成经过审计的回复

        Args:
            prompt: 用户输入
            dialog_state: 当前对话状态
            turn_count: 当前轮次
            crisis_turns: CRISIS 状态持续轮次
            risk_level: 风险等级
            conversation_history: 对话历史
            current_topic: 当前治疗话题
            **kwargs: 传递给 LLM 的额外参数

        Returns:
            LLMResponse: 经过审计的回复
        """
        start_time = time.time()

        # 1. 调用 LLM 生成原始回复
        raw_response = self.llm_provider.generate(prompt, kwargs.get("context"))

        if not self.config.enable_audit:
            return raw_response

        # 2. 构建审计上下文
        audit_context = AuditContext(
            llm_output=raw_response.content,
            dialog_state=dialog_state,
            turn_count=turn_count,
            crisis_turns=crisis_turns,
            risk_level=risk_level,
            conversation_history=conversation_history or [],
            llm_confidence=raw_response.confidence,
            current_topic=current_topic,
        )

        # 3. 执行审计（支持重试）
        verdict = None
        current_content = raw_response.content
        attempts = 0

        while attempts <= self.config.max_rewrite_attempts:
            # 审计当前内容
            audit_context.llm_output = current_content
            verdict = self.auditor.audit(audit_context)

            if verdict.passed:
                break

            # 根据审计结果处理
            if verdict.final_action == AuditAction.ESCALATE_HUMAN:
                # 升级人工 —— 返回预设消息
                current_content = self._get_escalation_message()
                break

            elif verdict.final_action == AuditAction.INJECT_RESOURCE:
                # 注入危机资源卡
                resource_card = generate_crisis_resource_card(self._audit_config)
                current_content = current_content + "\n\n" + resource_card
                break

            elif verdict.final_action == AuditAction.REWRITE:
                # 重写回复
                current_content = self.rewriter.rewrite(
                    current_content, verdict, audit_context
                )
                attempts += 1

            else:
                break

        # 4. 记录审计日志
        if self.config.enable_logging and verdict:
            self._log_audit(verdict, audit_context, time.time() - start_time)

        # 5. 返回最终回复
        return LLMResponse(
            content=current_content,
            confidence=raw_response.confidence,
            metadata={
                **raw_response.metadata,
                "audit_passed": verdict.passed if verdict else True,
                "audit_action": verdict.final_action.value if verdict else "pass",
                "rewrite_attempts": attempts,
            },
        )

    def _get_escalation_message(self) -> str:
        """获取升级人工的预设消息"""
        return (
            "我注意到你可能正在经历一些困难。为了更好地帮助你，"
            "我建议你将情况告知专业人士。你可以拨打以下热线获得即时支持：\n\n"
            + generate_crisis_resource_card(self._audit_config)
        )

    def _log_audit(
        self,
        verdict: AuditVerdict,
        context: AuditContext,
        duration: float,
    ) -> None:
        """记录审计日志"""
        log_entry = {
            "timestamp": time.time(),
            "dialog_state": context.dialog_state,
            "turn_count": context.turn_count,
            "risk_level": context.risk_level,
            "audit_passed": verdict.passed,
            "final_action": verdict.final_action.value,
            "failed_axes": [
                r.axis.value for r in verdict.results if not r.passed
            ],
            "duration_ms": round(duration * 1000, 2),
        }
        logger.info(f"Audit: {log_entry}")


# ============================================================
# 便捷工厂函数
# ============================================================

def create_audited_middleware(
    llm_provider: Any,
    rewriter: Optional[Any] = None,
    enable_audit: bool = True,
) -> AuditedLLMMiddleware:
    """创建审计中间件的工厂函数

    Args:
        llm_provider: LLM 提供者（实现 LLMProvider 接口）
        rewriter: 重写器（可选，默认使用 DefaultRewriter）
        enable_audit: 是否启用审计

    Returns:
        AuditedLLMMiddleware: 配置好的中间件实例
    """
    config = MiddlewareConfig(enable_audit=enable_audit)
    return AuditedLLMMiddleware(llm_provider, rewriter, config)
