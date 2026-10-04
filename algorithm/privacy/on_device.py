"""
端侧推理模块 —— 模型量化与端侧部署

核心功能：
    1. quantize_model(): 动态范围量化（TFLite / PyTorch）
    2. benchmark(): 测量推理时间、准确率、模型大小
    3. generate_report(): 输出量化对比 Markdown 报告

基础模型：DistilBERT（66M 参数，比 BERT 小 40%）
    - 当 TFLite 可用时使用 TFLite 动态范围量化
    - 否则使用 PyTorch dynamic quantization（int8）

目标约束：
    - 模型大小 < 25MB
    - 推理延迟 < 50ms/样本（CPU）
    - 准确率下降 > 3% 时建议 QAT
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ============================================================
# 路径配置
# ============================================================

_MODULE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = _MODULE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = str(OUTPUT_DIR / "quantization_report.md")
MODEL_DIR = OUTPUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 基础模型名称（DistilBERT —— 轻量级 BERT 替代）
BASE_MODEL_NAME = "distilbert-base-uncased"
# 序列长度
MAX_SEQ_LENGTH = 128
# 分类标签
NUM_LABELS = 5  # anxiety / depression / anger / neutral / positive


# ============================================================
# 数据结构
# ============================================================

@dataclass
class BenchmarkResult:
    """基准测试结果

    Attributes:
        model_type: 模型类型（original / quantized）
        model_size_mb: 模型大小（MB）
        avg_latency_ms: 平均推理延迟（ms/样本）
        p50_latency_ms: P50 延迟
        p95_latency_ms: P95 延迟
        accuracy: 准确率
        f1_macro: Macro F1
        throughput: 吞吐量（样本/秒）
    """
    model_type: str
    model_size_mb: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    accuracy: float
    f1_macro: float
    throughput: float


@dataclass
class QuantizationReport:
    """量化报告数据

    Attributes:
        base_model: 基础模型名称
        original: 原始模型基准结果
        quantized: 量化模型基准结果
        size_reduction_pct: 大小缩减百分比
        latency_speedup: 延迟加速比
        accuracy_drop_pct: 准确率下降百分比
        meets_size_target: 是否满足大小目标
        meets_latency_target: 是否满足延迟目标
        accuracy_warning: 准确率下降是否超过阈值
        recommendations: 建议列表
    """
    base_model: str
    original: BenchmarkResult
    quantized: BenchmarkResult
    size_reduction_pct: float
    latency_speedup: float
    accuracy_drop_pct: float
    meets_size_target: bool
    meets_latency_target: bool
    accuracy_warning: bool
    recommendations: list[str] = field(default_factory=list)


# ============================================================
# 模型构建（轻量文本分类器）
# ============================================================

def _build_text_classifier(num_labels: int = NUM_LABELS):
    """构建基于 DistilBERT 的文本分类模型

    Returns:
        (model, tokenizer)
    """
    import torch
    from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

    model_path = os.environ.get("MODEL_PATH", BASE_MODEL_NAME)

    try:
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    except Exception:
        # 使用镜像
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)

    try:
        model = DistilBertForSequenceClassification.from_pretrained(
            model_path, num_labels=num_labels
        )
    except Exception:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        model = DistilBertForSequenceClassification.from_pretrained(
            model_path, num_labels=num_labels
        )

    model.eval()
    return model, tokenizer


# ============================================================
# 假数据生成
# ============================================================

def _generate_test_data(
    tokenizer,
    n_samples: int = 100,
    seed: int = 42,
) -> tuple:
    """生成假数据用于基准测试

    Returns:
        (input_ids, attention_mask, labels)
    """
    import torch

    rng = np.random.RandomState(seed)
    texts = [
        "我感到很焦虑，不知道该怎么办",
        "今天心情不错，阳光很好",
        "最近总是失眠，很累",
        "我觉得生活没有意义",
        "和朋友聊了很久，感觉好多了",
        "工作压力很大，想辞职",
        "每天都在哭泣",
        "学会了冥想，感觉平静了很多",
        "总是担心未来",
        "今天做了一件很有意义的事",
    ]

    sampled = [texts[rng.randint(0, len(texts))] for _ in range(n_samples)]
    encoded = tokenizer(
        sampled,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt",
    )

    labels = torch.tensor(rng.randint(0, NUM_LABELS, n_samples))
    return encoded["input_ids"], encoded["attention_mask"], labels


# ============================================================
# 模型大小计算
# ============================================================

def _get_model_size_mb(model, path: Optional[str] = None) -> float:
    """计算模型大小（MB）"""
    import torch

    if path and Path(path).exists():
        return Path(path).stat().st_size / (1024 * 1024)

    # 计算参数内存
    total_params = sum(p.numel() for p in model.parameters())
    # float32 = 4 bytes
    size_bytes = total_params * 4
    # 加上缓冲区
    total_buffers = sum(b.numel() for b in model.buffers())
    size_bytes += total_buffers * 4

    return size_bytes / (1024 * 1024)


def _get_quantized_model_size_mb(model) -> float:
    """计算量化模型大小（int8 = 1 byte/param）"""
    total_params = sum(p.numel() for p in model.parameters())
    # int8 = 1 byte
    size_bytes = total_params * 1
    total_buffers = sum(b.numel() for b in model.buffers())
    size_bytes += total_buffers * 4  # 部分缓冲区仍为 float32

    return size_bytes / (1024 * 1024)


# ============================================================
# 量化
# ============================================================

def quantize_model(
    model_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> tuple:
    """动态范围量化

    优先使用 PyTorch dynamic quantization（int8）。
    如果 TFLite 可用，同时生成 TFLite 量化模型。

    Args:
        model_path: 原始模型路径（None 则下载 DistilBERT）
        output_path: 量化模型输出路径

    Returns:
        (original_model, quantized_model, tokenizer)
    """
    import torch

    print("[INFO] 加载基础模型 DistilBERT...")
    model, tokenizer = _build_text_classifier()

    if output_path is None:
        output_path = str(MODEL_DIR / "distilbert_quantized.pt")

    # 保存原始模型
    original_path = str(MODEL_DIR / "distilbert_original.pt")
    torch.save(model.state_dict(), original_path)
    print(f"[INFO] 原始模型已保存 → {original_path}")

    # PyTorch 动态范围量化（int8）
    print("[INFO] 执行 PyTorch 动态范围量化 (int8)...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # 量化所有 Linear 层
        dtype=torch.qint8,
    )

    # 保存量化模型
    torch.save(quantized_model.state_dict(), output_path)
    print(f"[INFO] 量化模型已保存 → {output_path}")

    # 尝试 TFLite 量化（可选）
    tflite_path = None
    try:
        tflite_path = _convert_to_tflite(model, tokenizer)
    except Exception as e:
        print(f"[WARN] TFLite 转换不可用: {e}")

    return model, quantized_model, tokenizer


def _convert_to_tflite(model, tokenizer) -> Optional[str]:
    """尝试转换为 TFLite 并量化

    需要 TensorFlow 安装。如果不可用则跳过。
    """
    try:
        import tensorflow as tf
        from transformers import TFDistilBertForSequenceClassification

        tf_model = TFDistilBertForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME, num_labels=NUM_LABELS
        )

        # 转换为 ConcreteFunction
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[1, MAX_SEQ_LENGTH], dtype=tf.int32),
            tf.TensorSpec(shape=[1, MAX_SEQ_LENGTH], dtype=tf.int32),
        ])
        def serve(input_ids, attention_mask):
            return tf_model(input_ids=input_ids, attention_mask=attention_mask)

        concrete = serve.get_concrete_function()
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 动态范围量化
        tflite_model = converter.convert()

        tflite_path = str(MODEL_DIR / "distilbert_quantized.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        print(f"[INFO] TFLite 量化模型已保存 → {tflite_path}")
        return tflite_path

    except ImportError:
        return None


# ============================================================
# 基准测试
# ============================================================

def benchmark(
    model,
    tokenizer,
    test_data: Optional[tuple] = None,
    n_warmup: int = 10,
    n_runs: int = 100,
    model_type: str = "original",
) -> BenchmarkResult:
    """基准测试 —— 测量推理时间、准确率、模型大小

    Args:
        model: 模型实例
        tokenizer: 分词器
        test_data: (input_ids, attention_mask, labels) 或 None（自动生成）
        n_warmup: 预热轮数
        n_runs: 测试轮数
        model_type: 模型类型标签

    Returns:
        BenchmarkResult
    """
    import torch

    model.eval()

    # 生成测试数据
    if test_data is None:
        test_data = _generate_test_data(tokenizer, n_samples=n_runs)

    input_ids, attention_mask, labels = test_data

    # 模型大小
    if "quantized" in model_type.lower():
        size_mb = _get_quantized_model_size_mb(model)
    else:
        size_mb = _get_model_size_mb(model)

    # 预热
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_ids[:2], attention_mask=attention_mask[:2])

    # 推理延迟测量
    latencies = []
    predictions = []

    with torch.no_grad():
        for i in range(min(n_runs, len(input_ids))):
            x_ids = input_ids[i:i+1]
            x_mask = attention_mask[i:i+1]

            start = time.perf_counter()
            outputs = model(x_ids, attention_mask=x_mask)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            latencies.append(elapsed)

            preds = torch.argmax(outputs.logits, dim=-1).item()
            predictions.append(preds)

    latencies = np.array(latencies)
    predictions = np.array(predictions)
    labels_np = labels[:len(predictions)].numpy()

    # 准确率
    accuracy = float(np.mean(predictions == labels_np))

    # Macro F1
    from sklearn.metrics import f1_score
    f1 = float(f1_score(labels_np, predictions, average="macro", zero_division=0))

    # 延迟统计
    avg_latency = float(np.mean(latencies))
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    throughput = 1000.0 / avg_latency if avg_latency > 0 else 0

    return BenchmarkResult(
        model_type=model_type,
        model_size_mb=round(size_mb, 2),
        avg_latency_ms=round(avg_latency, 2),
        p50_latency_ms=round(p50, 2),
        p95_latency_ms=round(p95, 2),
        accuracy=round(accuracy, 4),
        f1_macro=round(f1, 4),
        throughput=round(throughput, 2),
    )


# ============================================================
# 报告生成
# ============================================================

def generate_report(
    original: BenchmarkResult,
    quantized: BenchmarkResult,
    save_path: str = REPORT_PATH,
) -> QuantizationReport:
    """生成量化对比 Markdown 报告

    Args:
        original: 原始模型基准结果
        quantized: 量化模型基准结果
        save_path: 报告保存路径

    Returns:
        QuantizationReport
    """
    # 计算对比指标
    size_reduction = (1 - quantized.model_size_mb / original.model_size_mb) * 100 if original.model_size_mb > 0 else 0
    latency_speedup = original.avg_latency_ms / quantized.avg_latency_ms if quantized.avg_latency_ms > 0 else 1.0
    accuracy_drop = (original.accuracy - quantized.accuracy) * 100

    # 目标检查
    meets_size = quantized.model_size_mb < 25.0
    meets_latency = quantized.avg_latency_ms < 50.0
    accuracy_warning = accuracy_drop > 3.0

    # 建议
    recommendations = []
    if not meets_size:
        recommendations.append(f"模型大小 {quantized.model_size_mb:.1f}MB 超过 25MB 目标，建议使用更小的基础模型（如 TinyBERT）")
    if not meets_latency:
        recommendations.append(f"推理延迟 {quantized.avg_latency_ms:.1f}ms 超过 50ms 目标，建议考虑 INT4 量化或模型蒸馏")
    if accuracy_warning:
        recommendations.append(f"准确率下降 {accuracy_drop:.2f}% 超过 3% 阈值，强烈建议使用 QAT（量化感知训练）替代 PTQ")
        recommendations.append("QAT 参考：训练时插入 fake-quantization 节点，模拟量化误差")
    if meets_size and meets_latency and not accuracy_warning:
        recommendations.append("所有目标均已满足，模型可用于端侧部署")

    report_data = QuantizationReport(
        base_model=BASE_MODEL_NAME,
        original=original,
        quantized=quantized,
        size_reduction_pct=round(size_reduction, 2),
        latency_speedup=round(latency_speedup, 2),
        accuracy_drop_pct=round(accuracy_drop, 2),
        meets_size_target=meets_size,
        meets_latency_target=meets_latency,
        accuracy_warning=accuracy_warning,
        recommendations=recommendations,
    )

    # 生成 Markdown
    md = _render_markdown(report_data)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"[INFO] 量化报告已保存 → {save_path}")
    return report_data


def _render_markdown(report: QuantizationReport) -> str:
    """渲染 Markdown 报告"""
    lines = [
        "# 端侧模型量化报告",
        "",
        f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"> 基础模型：{report.base_model}",
        f"> 量化方式：动态范围量化（INT8）",
        "",
        "## 1. 模型大小对比",
        "",
        "| 指标 | 原始模型 | 量化模型 | 变化 |",
        "|------|---------|---------|------|",
        f"| 模型大小 | {report.original.model_size_mb:.2f} MB | {report.quantized.model_size_mb:.2f} MB | **-{report.size_reduction_pct:.1f}%** |",
        f"| 目标 < 25MB | {'✅' if report.original.model_size_mb < 25 else '❌'} | {'✅' if report.quantized.model_size_mb < 25 else '❌'} | |",
        "",
        "## 2. 推理延迟对比",
        "",
        "| 指标 | 原始模型 | 量化模型 | 变化 |",
        "|------|---------|---------|------|",
        f"| 平均延迟 | {report.original.avg_latency_ms:.2f} ms | {report.quantized.avg_latency_ms:.2f} ms | **{report.latency_speedup:.2f}x 加速** |",
        f"| P50 延迟 | {report.original.p50_latency_ms:.2f} ms | {report.quantized.p50_latency_ms:.2f} ms | |",
        f"| P95 延迟 | {report.original.p95_latency_ms:.2f} ms | {report.quantized.p95_latency_ms:.2f} ms | |",
        f"| 吞吐量 | {report.original.throughput:.1f} 样本/s | {report.quantized.throughput:.1f} 样本/s | |",
        f"| 目标 < 50ms | {'✅' if report.original.avg_latency_ms < 50 else '❌'} | {'✅' if report.quantized.avg_latency_ms < 50 else '❌'} | |",
        "",
        "## 3. 准确率对比",
        "",
        "| 指标 | 原始模型 | 量化模型 | 变化 |",
        "|------|---------|---------|------|",
        f"| Accuracy | {report.original.accuracy:.4f} | {report.quantized.accuracy:.4f} | -{report.accuracy_drop_pct:.2f}% |",
        f"| Macro F1 | {report.original.f1_macro:.4f} | {report.quantized.f1_macro:.4f} | |",
        f"| 准确率下降 ≤ 3% | {'✅' if not report.accuracy_warning else '❌ **超过阈值**'} | | |",
        "",
        "## 4. 目标达成情况",
        "",
        f"| 目标 | 状态 | 说明 |",
        f"|------|------|------|",
        f"| 模型大小 < 25MB | {'✅ 达成' if report.meets_size_target else '❌ 未达成'} | 量化后 {report.quantized.model_size_mb:.1f}MB |",
        f"| 推理延迟 < 50ms | {'✅ 达成' if report.meets_latency_target else '❌ 未达成'} | 量化后 {report.quantized.avg_latency_ms:.1f}ms |",
        f"| 准确率下降 ≤ 3% | {'✅ 达成' if not report.accuracy_warning else '❌ 未达成'} | 下降 {report.accuracy_drop_pct:.2f}% |",
        "",
        "## 5. 建议",
        "",
    ]

    for i, rec in enumerate(report.recommendations, 1):
        lines.append(f"{i}. {rec}")

    lines.extend([
        "",
        "## 6. 隐私保护说明",
        "",
        "量化后的模型可部署在用户设备上，实现 **敏感数据不出设备**：",
        "",
        "- 用户文本无需上传云端，直接在本地推理",
        "- 模型大小经过量化压缩，适合移动端/嵌入式部署",
        "- 配合 TFLite / PyTorch Mobile 可实现毫秒级端侧推理",
        "",
        "## 7. 技术细节",
        "",
        "- **基础模型**: DistilBERT（66M 参数，比 BERT-base 小 40%）",
        "- **量化方式**: 动态范围量化（Dynamic Range Quantization）",
        "  - Linear 层权重从 FP32 → INT8",
        "  - 推理时动态计算激活量化参数",
        "  - 无需校准数据集",
        "- **如果准确率下降超过 3%**: 建议改用 QAT（量化感知训练）",
        "  - 在训练过程中插入 fake-quantization 节点",
        "  - 模型学习适应量化误差，准确率损失更小",
        "",
    ])

    return "\n".join(lines)


# ============================================================
# 主流程
# ============================================================

def main():
    """运行完整量化流程"""
    print("=" * 60)
    print("端侧模型量化流程")
    print("=" * 60)

    # 1. 量化模型
    original_model, quantized_model, tokenizer = quantize_model()

    # 2. 基准测试 - 原始模型
    print("\n[INFO] 基准测试：原始模型...")
    original_result = benchmark(
        original_model, tokenizer, model_type="original"
    )
    print(f"  大小: {original_result.model_size_mb:.2f} MB")
    print(f"  延迟: {original_result.avg_latency_ms:.2f} ms")
    print(f"  准确率: {original_result.accuracy:.4f}")

    # 3. 基准测试 - 量化模型
    print("\n[INFO] 基准测试：量化模型...")
    quantized_result = benchmark(
        quantized_model, tokenizer, model_type="quantized"
    )
    print(f"  大小: {quantized_result.model_size_mb:.2f} MB")
    print(f"  延迟: {quantized_result.avg_latency_ms:.2f} ms")
    print(f"  准确率: {quantized_result.accuracy:.4f}")

    # 4. 生成报告
    print("\n[INFO] 生成量化报告...")
    report = generate_report(original_result, quantized_result)

    print("\n" + "=" * 60)
    print("量化完成！")
    print(f"  大小缩减: {report.size_reduction_pct:.1f}%")
    print(f"  延迟加速: {report.latency_speedup:.2f}x")
    print(f"  准确率下降: {report.accuracy_drop_pct:.2f}%")
    print(f"  准确率警告: {'是' if report.accuracy_warning else '否'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
