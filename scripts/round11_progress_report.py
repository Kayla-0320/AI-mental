"""从已落盘的开发证据生成阶段报告与可导出的科研图，不提前宣告外部验证通过。"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
from experiments.round11_common import OUTPUT,digest,save


def batch_f1(matrix: np.ndarray) -> np.ndarray:
    """从可批处理混淆矩阵计算固定三类macro-F1。"""
    denominator=matrix.sum(-1)+matrix.sum(-2)
    return np.divide(2*np.diagonal(matrix,axis1=-2,axis2=-1),denominator,
        out=np.zeros_like(denominator,dtype=float),where=denominator>0).mean(-1)


def interval(y: np.ndarray, old: np.ndarray, new: np.ndarray, groups: np.ndarray) -> dict[str,Any]:
    """固定开发预测条件下的来源成组重采样，不重选超参数。"""
    names=np.unique(groups)
    matrices=[]
    for prediction in (old,new):
        matrices.append(np.stack([np.bincount(3*y[groups==name]+prediction[groups==name],
            minlength=9).reshape(3,3) for name in names]))
    rng=np.random.default_rng(42)
    multiplicity=rng.multinomial(len(names),np.full(len(names),1/len(names)),size=10000)
    first,second=[batch_f1(np.einsum("bg,gij->bij",multiplicity,matrix)) for matrix in matrices]
    return {"source_groups":len(names),"bootstrap_samples":10000,
        "absolute_gain_95":np.quantile(second-first,[.025,.975]).tolist(),
        "relative_gain_95":np.quantile(second/first-1,[.025,.975]).tolist(),
        "scope":"conditional on fixed source-OOF predictions;not independent external confirmation;no hyperparameter reselection"}


def main() -> None:
    """只汇总完成的开发实验；整合报告允许随未完成实验进展更新。"""
    root=OUTPUT
    rows=json.loads((root/"audio_screen_deterministic/complete.json").read_text("utf-8"))["results"]
    def find(q: str,e: str) -> tuple[int,dict[str,Any]]:
        return next((i,row) for i,row in enumerate(rows) if row["config"]=={"quality":q,"encoder":e,"C":.1,"target":"A"})
    with np.load(root/"audio_screen_deterministic/oof.npz",allow_pickle=False) as values:
        old_index,old=find("legacy","msp")
        new_index,new=find("legacy","sense")
        summary=interval(values["A_labels"],values["probabilities"][old_index].argmax(1),
            values["probabilities"][new_index].argmax(1),values["source_groups"])
    summary.update(old_argmax_f1=old["argmax"]["macro_f1"],new_argmax_f1=new["argmax"]["macro_f1"],
        old_bias_selected_f1=old["metrics"]["macro_f1"],new_bias_selected_f1=new["metrics"]["macro_f1"])
    save(root/"audio_development_paired_bootstrap.json",summary,immutable=False)
    font=FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
    figures=root/"figures"
    figures.mkdir(exist_ok=True)
    labels=["旧MSP语音","中文SenseVoice","新旧语音联合"]
    series=[find("legacy",e)[1] for e in ("msp","sense","both")]
    fig,axes=plt.subplots(1,2,figsize=(10,4.3),layout="constrained")
    for ax,field,title in zip(axes,("macro_f1","neutral"),("音频情感宏F1","音频中性类别F1")):
        numbers=[row["metrics"]["macro_f1"] if field=="macro_f1" else row["metrics"]["per_class_f1"][1] for row in series]
        bars=ax.bar(range(3),numbers,color=["#7B8A9A","#2863A5","#21A28F"],width=.58)
        ax.set_xticks(range(3),labels,fontproperties=font)
        ax.set_ylim(0,.7 if field=="macro_f1" else .5)
        ax.set_title(title,fontproperties=font,fontsize=13)
        ax.grid(axis="y",alpha=.2)
        ax.set_axisbelow(True)
        for bar,number in zip(bars,numbers):
            ax.text(bar.get_x()+bar.get_width()/2,number+.012,f"{number:.4f}",ha="center",fontsize=11)
    fig.suptitle("固定旧质量条件、C=0.1、相同四折来源OOF",fontproperties=font,fontsize=14)
    fig.savefig(figures/"audio_development_comparison.png",dpi=220)
    plt.close(fig)
    audit=json.loads((root/"input_audit/complete.json").read_text("utf-8"))
    acquisition=json.loads((root/"emotiontalk_acquisition/complete.json").read_text("utf-8"))
    text=[]
    text.extend(["# 第十一轮答辩优化：阶段进展与证据报告","",
        "本报告为开发阶段记录，目标仍在执行。未解封外部test极性成绩，也未重跑原821条历史成绩；不将以下开发分数称为最终独立验证。","",
        "## 已落实的改进","",
        f"- 对全部{audit['n']}条开发样本解码时间戳，恢复{audit['restored_audio']}条原被时长元数据误拒的音频/画面观测，仍保留{audit['remaining_audio_rejected']}条质量拒绝。质量阈值保持0.25秒/10%，没有通过放宽阈值放行。",
        "- 中文SenseVoiceSmall严格加载233,999,167个参数，固定上游权重与摘要。关闭推理前端随机抖动后，重复编码最大误差0；混合长度批次误差约0.000003。",
        "- 补齐旧编码器先前跳过的56条真实语音缓存；原有效行逐位保持不变，使质量与表征可以分别对照。",
        f"- 完整获取并核验EmotionTalk固定版本：{acquisition['clips']}个标注媒体ID，原归档903个重复成员均经SHA256确认完全相同，只输入一次。外部测试1929条极性成绩保持封存。",
        "- 取得19,250条官方时间边界；742段对话的文件编号与标注开始时间均无顺序冲突。存在108处同说话人区间重叠，后续序列窗口需按结束时间检查未来泄漏。","",
        "## 相同观测与预算下的中文语音对照","",
        "任务为官方单模态音频A3极性，非整体M3。2,876条全部保留、4折整来源隔离、C固定为第十轮A专家使用的0.1；标准化只拟合训练来源。固定12点偏置选择分数为开发成绩。","",
        "| 编码方案 | 未校准argmax宏F1 | 开发偏置后宏F1 | 音频中性F1 |",
        "|---|---:|---:|---:|"])
    for label,row in zip(labels,series):
        text.append(f"| {label} | {row['argmax']['macro_f1']:.4f} | {row['metrics']['macro_f1']:.4f} | {row['metrics']['per_class_f1'][1]:.4f} |")
    text.extend(["",f"中文方案的开发偏置后相对提升为{(new['metrics']['macro_f1']/old['metrics']['macro_f1']-1)*100:.2f}%；未校准相对提升为{(summary['new_argmax_f1']/summary['old_argmax_f1']-1)*100:.2f}%。后者固定预测的来源bootstrap差值95%条件区间为[{summary['absolute_gain_95'][0]:.4f},{summary['absolute_gain_95'][1]:.4f}]。它不能替代独立外部验证。","",
        "![中文语音开发对照](figures/audio_development_comparison.png)","",
        "## 完整融合的嵌套来源开发对照","",
        "4折外层、3折内层生成元训练证据；M/T/A/V用各自官方标注。下表统一列出depth4且不加可靠性附加特征的融合头，保留同预算控制。每家族另有depth5和可靠性输入消融，详见对应complete.json。","",
        "| 质量 | 语音 | 专家粒度 | 宏F1 | 中性F1 | 来源等权F1 |",
        "|---|---|---:|---:|---:|---:|"])
    completed=[]
    for path in sorted((root/"evidence_deterministic").glob("family*/complete.json")):
        family=json.loads(path.read_text("utf-8"))
        completed.append(family["family"])
        row=family["results"][0]
        c=row["config"]
        text.append(f"| {c['quality']} | {c['encoder']} | {c['classes']} | {row['metrics']['macro_f1']:.4f} | {row['metrics']['per_class_f1'][1]:.4f} | {row['source_balanced_f1']:.4f} |")
    text.extend(["",f"当前已完成{len(completed)}/12个固定家族。开发集全局选型仍有选择偏差；主模型须锁定后再解封外部test并做原历史集诊断。","",
        "## 运行质量与资源恢复","",
        "已完成外部19,250条转录的BGE、ECISA辅助RoBERTa与Qwen2.5-7B三套表征，以及19,250条SenseVoice与MSP语音表征，目标标签未进入编码。新增及相关回归36项通过。",
        "初次并行大语义编码与高清视频提取触发提交内存不足，保留了中断记录。第12家族从已完成的外层折继续，最终12家族均完成。媒体提取改为单解码线程、32片段有界队列、原子缓存和提交余量保护；旧缓存只复用完整采样条目，初版3条四帧人脸及1条两帧信号必须重算。不完整新采样须在内存充足时重复核验，不能将资源故障当成真实缺失。",
        "原MFCC流程在3个固定开发片段的最大数值误差小于0.000004，旧质量mask完全相同；人脸统计与深层表示误差0。","",
        "说话人字段核对：train含13种speaker_id、valid4种、test4种，训练与验证/测试均无交集；valid与test共享02和13。后续外部模型只拟合官方train，valid用于选择，不并入最终训练。结论依据作者的公开ID字段，未自行验证真实人物身份。","",
        "## 与图片批注的对应关系","",
        "| 批注方向 | 本轮已完成 | 仍需验收 |",
        "|---|---|---|",
        "| 数据质量/跨模态纠偏 | 时间戳审计、47条恢复、相同观测对照 | 缺失/噪声/错配压力测试 |",
        "| 中文多模态感知 | 中文语音替换及M/T/A/V嵌套融合 | 新外部测试及中性/冲突分层 |",
        "| vision | 保留CLIP与HSEmotion，外部人脸实现与旧缓存3例误差0 | 完整外部视觉泛化与说话人限制 |",
        "| MoE/Top-k/反事实 | 强化表征后9组路由/注意力/匹配软教师对照已完成，均未超过证据融合 | 保留负结果，主模型按开发证据选型 |",
        "| 情绪时序 | 真实时间标注、音轨时长审计、来源交叉拟合的真实观测概率已完成 | horizon1/3/5预测、实际提前量和最终留出验收 |","",
        "## 当前限制与下一步","",
        "第十轮五种子整体M3历史macro-F1=0.7032028498保持原样。本轮尚未给出更新的历史或外部主成绩。初版随机抖动实验保留在audio_screen与evidence目录；正式开发使用带deterministic字样的目录，不挑两套中更好的一套混报。","",
        "外部媒体哈希未发现跨录制组或跨CH-SIMS的字节级重复；近似/重编码重复仍需继续核验。EmotionTalk含表演式短对话，不能称青少年临床或跨天长期随访。","",
        "继续完成：外部特征与来源审计、同预算新模型验证、锁定后的跨数据集测试、真实观测序列、原媒体单请求延迟/质量消融、逐样本概率和答辩图表案例。","",
        "来源：[EmotionTalk固定版本](https://huggingface.co/datasets/BAAI/Emotiontalk/tree/adbc17fc944e8cf2873643906160c6ca0259ab61)、[SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall/tree/3847d57b6bdf2dd8875cb1508d2af43d80a16bf7)。","",
        "复现入口见本目录PROTOCOL.md及仓库scripts/round11_*.py、algorithm/experiments/round11_*.py。"])
    if (root/"neural/summary.json").exists():
        neural=json.loads((root/"neural/summary.json").read_text("utf-8"))
        text.extend(["","## 后续实测补充","",
            "强化表征的9组神经对照已完成。最优为交换一致性代理，开发宏F1=0.6821；软教师交换=0.6611，未超过主证据融合。这里只说明本轮固定配方未胜出，不据此推断该类方法普遍无效。","",
            "固定融合配方的3B语义版本开发宏F1=0.7169，轻量文本版本=0.7083；实际延迟尚待测量，目前不能据模型大小宣称端到端提速。","",
            "新旧模型五档融合与固定676点偏置搜索选择新模型权重1.0，偏置[0.15,0,0]，宏F1=0.7284、来源等权F1=0.7360。混合旧模型未胜出；这仍是开发选择，不是最终独立外部验收。","",
            "相同搜索预算的文字/语音相反分层198条、55来源，宏F1从0.4748到0.5416，来源条件bootstrap差值95%区间[0.0183,0.1183]。整体M3差值条件区间[-0.0046,0.0296]，尚不能声称整体泛化提升已获得独立确认。","",
            "外部七类感知器以官方train拟合、valid选择：文字+真实语音验证宏F1=0.5698。训练历史为5折整来源交叉拟合识别概率，留出历史由仅训练组拟合的模型识别；未来预测成绩还未评测。","",
            "已核对全部19,250条音轨的实际时长，和官方段长最大差约23.2毫秒；1,240条长于10秒，冻结语音编码采用居中10秒，不能把全样本覆盖称为全长音轨覆盖。","",
            "序列进一步采用实际音轨结束与标注结束的较晚者，并为全部过去观测保留截止检查。0.25秒对齐余量下，一步测试候选1,000个；额外1秒推理余量下749个。三步/五步仍为808/545。早期1106/796是仅标注时间的几何统计，最终不能混用。","",
            "[官方论文§3.2](https://aclanthology.org/2026.findings-acl.440.pdf#page=4)定义了连续极性的方向。17,321条开发发布值均与五位标注者均值一致，但happy类别中1,849/3,346条连续值为负，原因尚未核实。保留原发布值、分别评测三类极性和七类情绪，不按类别推测或重写极性金标。"])
    (root/"阶段进展与证据报告.md").write_text("\n".join(text)+"\n","utf-8")
    save(root/"progress_snapshot.json",{"completed_evidence_families":completed,
        "original_history_re_evaluated":False,"external_test_polarity_unsealed":False,
        "goal_complete":False,"stage_report_sha256":digest(root/"阶段进展与证据报告.md")},immutable=False)
    print(json.dumps({"report":str(root/"阶段进展与证据报告.md"),"completed_families":len(completed),"audio_bootstrap":summary}),flush=True)


if __name__=="__main__":
    main()
