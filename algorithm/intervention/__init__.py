"""
对话状态机 —— 状态机驱动的受控 AI 咨询对话引擎

五状态有限自动机：
    INIT → EXPLORE → INTERVENE → CLOSE
                      ↓
                    CRISIS → (升级层)

核心安全约束：
    - CRISIS 状态必须触发升级层接口
    - 任何状态禁止生成诊断性语言
    - 所有转移规则不依赖 LLM，纯逻辑实现
"""
