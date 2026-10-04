"""
生成 TAMER 项目答辩 PPT（改进版）
按竞赛建议：前3页讲清项目做什么、解决谁的问题、AI技术在哪里、完成程度
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

def add_title_slide(prs, title, subtitle=""):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    txBox = slide.shapes.add_textbox(Inches(1), Inches(2.5), Inches(11), Inches(1.5))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(44)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0x1a, 0x1a, 0x2e)
    p.alignment = PP_ALIGN.CENTER
    if subtitle:
        txBox2 = slide.shapes.add_textbox(Inches(2), Inches(4.2), Inches(9), Inches(1))
        tf2 = txBox2.text_frame
        p2 = tf2.paragraphs[0]
        p2.text = subtitle
        p2.font.size = Pt(20)
        p2.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
        p2.alignment = PP_ALIGN.CENTER
    return slide

def add_content_slide(prs, title, bullets):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    shape = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.333), Inches(1.2))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(0x63, 0x66, 0xf1)
    shape.line.fill.background()
    txBox = slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(12), Inches(0.8))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0xff, 0xff, 0xff)
    txBox2 = slide.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.5), Inches(5.5))
    tf2 = txBox2.text_frame
    tf2.word_wrap = True
    for i, bullet in enumerate(bullets):
        if i == 0:
            p = tf2.paragraphs[0]
        else:
            p = tf2.add_paragraph()
        p.text = bullet
        p.font.size = Pt(18)
        p.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        p.space_after = Pt(12)
        if bullet.startswith("  "):
            p.level = 1
            p.font.size = Pt(16)
    return slide

# ===== Slide 1: 封面 =====
add_title_slide(prs, "TAMER 青少年 AI 心理健康平台", "治疗联盟感知的多模态情绪识别框架")

# ===== Slide 2: 一句话讲清项目（新增） =====
add_content_slide(prs, "项目概述", [
    "▎TAMER 是什么？",
    "  首个将临床心理治疗理论系统性融入 AI 的青少年心理健康平台",
    "",
    "解决谁的问题？",
    "  12-22 岁青少年 —— 抑郁检出率 24.6%，但每 10 万人仅 3.5 名心理咨询师",
    "",
    "▎AI 技术在哪里？",
    "  ① RQI 关系质量指数（治疗联盟理论）→ 动态调整 AI 回复策略",
    "  ② MIS 多模态矛盾检测（跨通道理论）→ 识别「说没事但声音紧张」",
    "  ③ CUSUM 时序追踪（统计过程控制）→ 捕捉渐进性情绪恶化",
    "",
    "▎完成到什么程度？",
    "  四端系统已上线 | 风险预警准确率 92.6% | 124 人用户测试满意度 4.08/5.0",
])

# ===== Slide 3: 核心技术与完成度（新增） =====
add_content_slide(prs, "AI 技术详解与完成度", [
    "▎具体技术实现",
    "  文本情绪分析：5 维概率分布 [快乐, 悲伤, 焦虑, 愤怒, 中性]",
    "  声学特征提取：能量均值、基频范围、语速、停顿比",
    "  大模型对话：通义千问（DashScope API），TAMER 分析结果注入 system prompt",
    "  风险分类：基于融合情绪向量的阈值分类 + CUSUM 时序追踪",
    "",
    "▎相比传统方法的提升",
    "  固定阈值 → 个人基线：准确率 64.8% → 92.6%（+42.9%）",
    "  单模态文本 → 跨模态融合：掩饰检测召回 0% → 87%",
    "  无关系建模 → RQI 动态策略：「被理解感」+22.2%",
    "",
    "▎系统完成度",
    "  前端 React + 后端 Node.js + 算法 Python + 数据库 PostgreSQL 四端全通",
    "  文字对话、语音通话、心理画像、危机干预四大功能均已实现",
])

# ===== Slide 4: 问题背景（压缩） =====
add_content_slide(prs, "问题背景", [
    "▎青少年心理健康现状",
    "  抑郁检出率 24.6%，焦虑检出率 31.2%",
    "  每 10 万青少年仅 3.5 名心理咨询师（供需比 1:28,000）",
    "",
    "▎现有 AI 心理产品的三大不足",
    "  ① 静态评估：量表是单次快照，无法捕捉渐进性恶化",
    "  ② 单模态分析：忽视「说没事但声音紧张」的掩饰行为",
    "  ③ 无关系建模：对所有用户使用相同策略，缺乏临床依据",
])

# ===== Slide 5: 创新点① RQI =====
add_content_slide(prs, "创新点① RQI 关系质量指数", [
    "▎临床依据：治疗联盟理论（Horvath & Greenberg, 1989）",
    "",
    "▎算法：RQI(t) = α·RQI(t-1) + (1-α)·Σ(wᵢ·signal)",
    "  5 维信号：互动频率 | 自我暴露深度 | 话题多样性 | 信任信号 | 情感共鸣",
    "  α = 0.7（EMA 时间衰减）",
    "",
    "▎阶段划分 → AI 策略自适应",
    "  RQI < 0.3  → 陌生期：温和引导，不追问私人问题",
    "  0.3~0.6    → 熟悉期：共情倾听，逐步深入",
    "  RQI ≥ 0.6  → 信任期：可使用更深层干预策略",
    "",
    "▎效果：多次对话后「被理解感」+22.2%，「信任度」+25.4%",
])

# ===== Slide 6: 创新点② MIS =====
add_content_slide(prs, "创新点② MIS 多模态矛盾检测", [
    "▎临床依据：跨通道情绪不一致（Ekman, 2003）",
    "",
    "▎算法：MIS = 1 - cosine_similarity(text_vec, voice_vec)",
    "  自适应阈值：threshold = μ + k·σ",
    "",
    "▎矛盾类型判定",
    "  masking（掩饰型）：言语平静 + 声音紧张 → 可能在隐藏",
    "  performative（表演型）：言语痛苦 + 声音平淡 → 可能非真实危机",
    "",
    "▎注意力融合：模态权重由 RQI 和 MIS 动态调整",
    "  高 RQI → 增加文本权重 | 高 MIS → 增加声学权重",
    "",
    "▎效果：掩饰检测准确率 86.7%，表演检测 82.3%",
])

# ===== Slide 7: 创新点③ CUSUM =====
add_content_slide(prs, "创新点③ CUSUM 时序追踪 + 个人基线", [
    "▎临床依据：统计过程控制（Page, 1954）",
    "  心理恶化是渐进过程，单轮评估容易漏检",
    "",
    "▎个人基线（EMA + Z-score）",
    "  μ_t = α·μ_{t-1} + (1-α)·x_t",
    "",
    "▎CUSUM 变点检测",
    "  S_t = max(0, S_{t-1} + (x_t - μ₀ - δ))",
    "  S_t > h → 恶化 | S_t < -h → 好转",
    "",
    "▎效果：准确率 92.6% vs 固定阈值 64.8%（+42.9%）",
    "  误报率 12.1% vs 31.2%（-61.2%）",
])

# ===== Slide 8: 实验结果 =====
add_content_slide(prs, "实验结果", [
    "▎实验 1：个人基线 vs 固定阈值",
    "  准确率 92.6% vs 64.8% | F1 0.912 vs 0.589",
    "",
    "▎实验 2：跨模态 vs 单模态文本",
    "  总体 91.0% vs 33.3% | 掩饰召回 87%（单模态 0%）",
    "",
    "▎实验 3：消融实验",
    "  Full TAMER: 92.0% / F1 0.897",
    "  w/o MIS: 91.3% | w/o CUSUM: 91.5% | Simple MLP: 72.7%",
    "",
    "用户测试（124 人，13 天）",
    "  整体满意度 4.08/5.0 | NPS 23.3 | 共情能力 4.28/5.0",
    "  对话数据 3,847 轮 | 语音通话 412 分钟",
])

# ===== Slide 9: 系统架构 =====
add_content_slide(prs, "系统架构与实施", [
    "▎四端架构",
    "  前端 React (:5173) → 后端 Node.js (:3000) → 算法 Python (:8001) → PostgreSQL (:5432)",
    "",
    "▎TAMER 已接入对话链路",
    "  intervention.py 每轮调用 TAMERFramework.process()",
    "  RQI/MIS/CUSUM 结果注入 system prompt，影响 AI 回复行为",
    "  前端组件实时展示后端 TAMER 输出（非前端自算）",
    "",
    "▎开发周期：4 个月（2026.06 - 2026.10）",
])

# ===== Slide 10: 应用成效 =====
add_content_slide(prs, "应用成效与社会价值", [
    "▎核心指标全部达成",
    "  风险预警准确率 92.6%（目标 ≥85%）✓",
    "  误报率 12.1%（目标 ≤20%）✓",
    "  用户满意度 4.08/5.0（目标 ≥4.0）✓",
    "",
    "▎社会价值",
    "  漏检率从 35.2% 降至 8.7%（-69.5%）",
    "  7×24 小时可及，覆盖夜间高危时段（47% 对话在 22:00-01:00）",
    "  弥补咨询资源不足（供需比 1:28,000）",
])

# ===== Slide 11: 总结与展望 =====
add_content_slide(prs, "总结与展望", [
    "▎核心贡献",
    "  首次将临床心理治疗理论系统性融入 AI 系统",
    "  TAMER 框架可推广到成人咨询、危机热线等场景",
    "",
    "▎当前不足",
    "  模拟数据为主，需真实临床数据验证",
    "  RQI 权重人工设定，未来可用注意力机制学习",
    "",
    "▎未来规划",
    "  短期：与学校合作收集 500+ 真实用户数据",
    "  中期：引入面部表情、生理信号等多模态",
    "  长期：建立青少年心理健康大数据平台",
])

# ===== Slide 12: 致谢 =====
add_title_slide(prs, "谢谢", "TAMER 青少年 AI 心理健康平台")

output_path = r"c:\Users\11127\Desktop\李泓毅新版\AI-mental-main\algorithm\artifacts\metrics\TAMER_答辩PPT_v2.pptx"
prs.save(output_path)
print(f"PPT 已生成：{output_path}")
print(f"共 {len(prs.slides)} 页")
