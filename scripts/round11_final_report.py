"""仅在所有验收证据存在时生成最终报告、科研图、答辩讲稿和交付清单。"""
from __future__ import annotations
import csv
from datetime import datetime
from importlib.metadata import version,PackageNotFoundError,distributions
import json
from pathlib import Path
import shutil
import sys
from typing import Any
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
from experiments.round11_common import OUTPUT,PREVIOUS,digest,save


def read(name: str) -> Any:
    """缺少必需证据就失败，不把尚未完成写成已完成。"""
    return json.loads((OUTPUT/name).read_text("utf-8"))


def historical_interval() -> dict[str,Any]:
    """同一821样本、配对来源重采样；先每种子F1，再取五种子均值。"""
    with np.load(OUTPUT/"historical_diagnostic/7b_predictions.npz",allow_pickle=False) as z:
        y=z["labels"];new=z["probabilities"];groups=z["source_groups"];ids=z["sample_ids"]
    with np.load(PREVIOUS/"refined_final/historical_predictions.npz",allow_pickle=False) as z:
        if not np.array_equal(ids,z["sample_ids"]) or not np.array_equal(y,z["labels"]):raise ValueError("主指标区间身份不符")
        old=z["probabilities"]
    names=np.unique(groups);rng=np.random.default_rng(42)
    multiplicity=rng.multinomial(len(names),np.full(len(names),1/len(names)),size=5000)
    values=[]
    for probabilities in (old,new):
        seed_scores=[]
        for p in probabilities:
            pred=p.argmax(1);cm=np.stack([np.bincount(3*y[groups==g]+pred[groups==g],minlength=9).reshape(3,3) for g in names])
            draw=np.einsum("bg,gij->bij",multiplicity,cm);den=draw.sum(1)+draw.sum(2)
            seed_scores.append(np.divide(2*np.diagonal(draw,axis1=1,axis2=2),den,out=np.zeros_like(den,dtype=float),where=den>0).mean(1))
        values.append(np.mean(seed_scores,axis=0))
    return {"unit":"original source video","n_sources":len(names),"draws":5000,
        "new_five_meta_seed_mean_95ci":np.quantile(values[1],[.025,.975]).tolist(),
        "paired_delta_95ci":np.quantile(values[1]-values[0],[.025,.975]).tolist(),
        "limitation":"conditional on fixed models;historical set previously exposed;five meta seeds share deterministic experts"}


def main() -> None:
    """将交付完成率与性能目标是否达标分开评价。"""
    historical=read("historical_diagnostic/results.json");external=read("external_m3/results.json")
    previous_result=json.loads((PREVIOUS/"refined_final/results.json").read_text("utf-8"))
    forecast=read("forecast_final/results.json");robust=read("robustness/results.json")
    runtime=read("runtime_benchmark/results.json");staged=read("runtime_staged/results.json")
    language_equivalence={mode:read(f"validation/language_runtime_{mode}_equivalence.json") for mode in ("7b","3b")}
    replay=read("validation/final_replay.json");raw=read("forecast_replay/results.json")
    audio=read("audio_development_paired_bootstrap.json");quality=read("input_audit/complete.json")
    restored=read("validation/restored_inputs_bitwise.json")
    neural=read("neural/summary.json");policy=read("missing_route/selection_locked.json")
    near=read("near_duplicate_audit.json");conflict=read("development_conflict_audit_v2.json")
    quality_slices=read("quality_strata.json")
    suite=ET.parse(OUTPUT/"validation/final_tests.xml").getroot()
    suites=[suite] if suite.tag=="testsuite" else list(suite.iter("testsuite"))
    test_summary={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ("tests","failures","errors","skipped")}
    if test_summary["failures"] or test_summary["errors"]:raise ValueError("必要测试未通过")
    figures=OUTPUT/"figures";figures.mkdir(exist_ok=True)
    font=FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
    def labels(ax: Any,title: str,x: list[str]) -> None:
        ax.set_title(title,fontproperties=font,fontsize=13);ax.set_xticks(range(len(x)),x,fontproperties=font)
        ax.set_ylim(0,1);ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    selected=[next(r for r in forecast["rows"] if r["setting"]==f"h{h}_lead0.25") for h in (1,3,5)]
    fig,ax=plt.subplots(figsize=(9,4.6),layout="constrained");x=np.arange(3);width=.34
    for offset,key,color,label in ((-.17,"baseline","#8292A5","保持最近观测"),(.17,"selected","#1978A5","验证集锁定预测器")):
        vals=[r["metrics"][key]["macro_f1"] for r in selected];bars=ax.bar(x+offset,vals,width,color=color,label=label)
        for bar,value in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,value+.012,f"{value:.4f}",ha="center")
    labels(ax,"真实观测历史 → 未来七类情绪（官方测试录制组）",[f"跨度{h}\nn={r['eligible_test_windows']}" for h,r in zip((1,3,5),selected)])
    ax.set_ylim(0,.65);ax.legend(prop=font);fig.savefig(figures/"forecast_test_comparison.png",dpi=220);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout="constrained")
    vals=[historical["round10_mean_macro_f1"],historical["7b"]["mean_macro_f1"],historical["3b"]["mean_macro_f1"],historical["light"]["mean_macro_f1"]]
    bars=axes[0].bar(range(4),vals,color=["#8292A5","#1978A5","#21A28F","#E4AA38"])
    labels(axes[0],"原821：五个元分类头种子的平均宏F1",["第十轮","第十一轮7B","3B","轻量"])
    for bar,value in zip(bars,vals):axes[0].text(bar.get_x()+bar.get_width()/2,value+.015,f"{value:.4f}",ha="center")
    erows=external["splits"]["test_inputs"];keys=["old","new","light","adapted"]
    vals=[erows[k]["ensemble"]["macro_f1"] for k in keys]
    bars=axes[1].bar(range(4),vals,color=["#8292A5","#1978A5","#E4AA38","#21A28F"])
    labels(axes[1],"外部1929：固定发布极性金标、概率集成",["冻结第十轮","冻结第十一轮","冻结轻量","额外训练监督"])
    for bar,value in zip(bars,vals):axes[1].text(bar.get_x()+bar.get_width()/2,value+.015,f"{value:.4f}",ha="center")
    fig.savefig(figures/"main_and_external.png",dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4.8),layout="constrained");rows=robust["rows"];x=np.arange(len(rows))
    for offset,key,color,label in ((-.17,"old_same_budget","#8292A5","旧表征同预算控制"),(.17,"new","#1978A5","新表征+已锁定缺字退化路径")):
        ax.bar(x+offset,[r[key]["macro_f1"] for r in rows],.34,color=color,label=label)
    labels(ax,"全2876开发来源压力诊断：不删困难样本",[r["condition"].replace("waveform_SNR_","SNR ") for r in rows])
    ax.tick_params(axis="x",labelrotation=30);ax.legend(prop=font);fig.savefig(figures/"robustness_development.png",dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,4.8),layout="constrained")
    latency=[runtime["hot_p50_seconds"],staged["3b"]["warm_policy_p50_seconds"],staged["7b"]["warm_policy_p50_seconds"]]
    quality_scores=[historical[m]["mean_macro_f1"] for m in ("light","3b","7b")]
    ax.scatter(latency,quality_scores,s=130,c=["#E4AA38","#21A28F","#1978A5"])
    three_policy="3B：常驻" if "co-resident" in staged["3b"]["policy"] else "3B：隔离进程加载"
    for a,b,label in zip(latency,quality_scores,("轻量：常驻",three_policy,"7B：隔离进程加载")):
        ax.annotate(label,(a,b),xytext=(-6,7) if a==max(latency) else (6,7),
            ha="right" if a==max(latency) else "left",textcoords="offset points",fontproperties=font)
    ax.set_xlabel("原媒体请求 p50 / 秒（常驻与按需策略分列）",fontproperties=font)
    ax.set_ylabel("原821五元种子平均宏F1",fontproperties=font);ax.grid(alpha=.2)
    ax.set_ylim(.65,.75)
    fig.savefig(figures/"measured_quality_cost.png",dpi=220);plt.close(fig)
    interval=historical_interval();save(OUTPUT/"historical_diagnostic/paired_interval.json",interval,immutable=False)
    # 原始类别分数和全窗口概率另外导出CSV，方便答辩现场抽查。
    exports=OUTPUT/"exports";exports.mkdir(exist_ok=True)
    with (exports/"metric_summary.csv").open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.writer(handle);writer.writerow(["task","split","method","macro_f1","n","status"])
        for m in ("7b","3b","light"):writer.writerow(["M3","historical821",m,historical[m]["mean_macro_f1"],821,"five meta seed mean;preexposed"])
        for m in keys:writer.writerow(["M3","EmotionTalk test",m,erows[m]["ensemble"]["macro_f1"],1929,"same released labels;adapted extra supervision"])
        for r in forecast["rows"]:
            for m in ("selected","baseline","linear","gru","observed_mean"):
                writer.writerow(["futureM7",r["setting"],m,r["metrics"][m]["macro_f1"],r["eligible_test_windows"],"validation locked;earlier project M7 exposure disclosed"])
    for name in ("historical_diagnostic/7b_predictions.npz","external_m3/test_inputs_scored_predictions.npz","forecast_final/h1_lead0.25/test_predictions.npz"):
        with np.load(OUTPUT/name,allow_pickle=False) as z:
            p=z["probabilities"] if "probabilities" in z.files else z["new"]
            if p.ndim==3:p=p.mean(0)
            with (exports/(Path(name).parent.name+"_predictions.csv")).open("w",encoding="utf-8-sig",newline="") as handle:
                writer=csv.writer(handle);writer.writerow(["sample_id","gold","prediction",*[f"p{i}" for i in range(p.shape[1])]])
                writer.writerows([str(k),int(y),int(q.argmax()),*q.tolist()] for k,y,q in zip(z["sample_ids"],z["labels"],p))
    components=["signal_features","face_features","clip_features","speech_features","sensevoice_deterministic_features","bge_features","auxiliary_features","llm_large_features"]
    coverage=[]
    for split,n in (("train",15413),("valid",1908),("test_inputs",1929)):
        ids=None
        for component in components:
            path=OUTPUT/"emotiontalk_features"/component/f"{split}.npz"
            with np.load(path,allow_pickle=False) as z:
                current=z["sample_ids"]
                if len(current)!=n or ids is not None and not np.array_equal(ids,current):raise ValueError("完整外部表征覆盖不符")
                ids=current.copy()
                for field in z.files:
                    values=z[field]
                    if values.dtype.kind in "fc" and not np.isfinite(values).all():raise ValueError("最终外部表征存在非有限值")
            coverage.append({"component":component,"split":split,"n":n,"sha256":digest(path)})
    save(OUTPUT/"validation/external_feature_coverage.json",coverage,immutable=False)
    if read("wave_stress/complete.json")["n"]!=2876:raise ValueError("波形压力没有覆盖全开发集")
    if "independent_refit" not in replay:raise ValueError("独立重训未验证")
    checks=[("12个固定家族来源嵌套对照",len(list((OUTPUT/"evidence_deterministic").glob("family*/complete.json")))==12),
        ("中文语音与输入质量审计",quality["n"]==2876 and restored["n"]==47),("MoE/反事实等9个同预算神经对照",len(neural["results"])==9),
        ("8种完整外部表征19250条、来源与重复审计",len(coverage)==24),
        ("锁定外部M3全1929条验证",erows["new"]["n"]==1929),("真实观测预测6设置、GRU和投票",len(forecast["rows"])==6),
        ("全2876缺失/错配/真实波形压力及可观测质量分层",len(robust["rows"])==9 and bool(quality_slices["external_test"])),
        ("3档模型和5个元头种子",all((OUTPUT/f"final_models/{m}/model_manifest.json").exists() for m in ("7b","3b","light"))),
        ("模型及选型哈希、独立重训与语言等价重放",replay["independent_refit"]["probability_max_error"]<=1e-6 and all(max(r["max_abs_errors"].values())<=1e-5 for r in language_equivalence.values())),
        ("真实历史媒体至未来预测的端到端重放",raw["forecast_max_probability_error"]<=1e-3),
        ("三档原媒体推理、冷/热策略和显存实测",runtime["n_requests"]==24 and all(staged[m]["n_requests"]==12 for m in ("7b","3b"))),
        ("第十轮126个冻结工件保持",replay["round10_preservation"]["n"]==126 and not replay["round10_preservation"]["failures"]),
        ("关键协议与因果输入测试",test_summary["tests"]>=36 and not test_summary["failures"] and not test_summary["errors"])]
    if not all(ok for _,ok in checks):raise ValueError("最终交付清单仍有未完成项")
    audit={"engineering_checks":[{"item":name,"complete":bool(ok)} for name,ok in checks],
        "delivered_checks":sum(ok for _,ok in checks),"total_checks":len(checks),"engineering_completion_percent":100.,
        "original_performance_goal":{"definition":"original request:unchanged821 five meta seed mean macroF1 >=1.10 * Round9 reported .6388",
            "threshold":.70268,"preserved_round10_achieved":historical["preserved_round10_meets_original_10percent_threshold"],
            "round11_achieved":historical["round11_meets_original_10percent_threshold"],
            "round11_relative_gain_vs_round9":historical["round11_relative_gain_vs_original_round9"]},
        "incremental_round11_vs_round10":historical["main_relative_gain_vs_round10"],
        "tests":test_summary,"do_not_conflate":"engineering completeness is not evidence of reaching performance target"}
    save(OUTPUT/"COMPLETION-AUDIT.json",audit,immutable=False)
    now=datetime.now().astimezone().isoformat(timespec="seconds")
    main_old=historical["round10_mean_macro_f1"];main_new=historical["7b"]["mean_macro_f1"];gain=historical["main_relative_gain_vs_round10"]
    goal_sentence=f"原始10%目标的参照是第九轮报告0.6388，阈值0.70268；第十轮已达到该历史点估计目标。本轮相对原始参照变化{historical['round11_relative_gain_vs_original_round9']:+.2%}，{'达到' if historical['round11_meets_original_10percent_threshold'] else '未达到'}该阈值；相对第十轮的新增变化单独报告。"
    h1=selected[0];h_gain=h1["metrics"]["selected"]["macro_f1"]/h1["metrics"]["baseline"]["macro_f1"]-1
    report=f'''# 第十一轮完整优化与独立验证报告

生成时间：{now}。本报告依据已锁定模型和逐条预测生成；原第十轮工件保持只读。

## 1. 最终结论与完成度

本轮完成全部{len(checks)}项工程交付验收，交付完成度100%。这表示约定实验、验证和交付材料已经完成，并不表示所有方法都有收益。

主任务仍是原821条、原三类标签、五个元分类头种子的平均macro-F1：第十轮 **{main_old:.6f}** → 本轮7B **{main_new:.6f}**，相对变化 **{gain:+.2%}**。{goal_sentence}

{'**整体主指标基本持平，当前证据不支持主分类取得实质提升。** 本轮主要贡献在子任务与证据完整性，不能用工程完成率或其他任务的较大提升掩盖这一点。' if abs(gain)<.01 else '主分类点估计与统计区间均按原口径保留，是否具有稳健增益须同时看来源区间。'}

工程交付完整不等于科研性能全面成功。本轮不以7B历史微小变化或子任务收益作为全面替换第十轮主分类模型的依据；第十轮保留，新增预测链路和部署对照按各自已验证范围使用。

真实观测历史的下一话轮七类预测：**{h1['metrics']['baseline']['macro_f1']:.6f} → {h1['metrics']['selected']['macro_f1']:.6f}（{h_gain:+.2%}）**，覆盖全部1,000个符合预注册时序条件的测试窗口。它是不同任务的有效亮点，不能写成整体M3分类提升{h_gain:.1%}。

外部1,929条发布M3标签的冻结迁移，集成macro-F1为 **{erows['old']['ensemble']['macro_f1']:.6f}→{erows['new']['ensemble']['macro_f1']:.6f}**；{'点估计回退，不能声称跨数据集整体增强。' if erows['new']['ensemble']['macro_f1']<erows['old']['ensemble']['macro_f1'] else '该发布测试下点估计改善，仍只支持有限录制来源。'}额外EmotionTalk训练监督的适应对照为{erows['adapted']['ensemble']['macro_f1']:.6f}，单列训练条件。

主任务历史集在项目早期已多次接触；本轮锁定后仅做一次诊断，不将其称为从未见过的独立测试。外部M3本轮先锁定模型/决策/逐条预测，再提取官方M3目标评分；M7使用独立受锁约束的目标入口，同一发布标注文件包含不同任务字段，不声称整个文件此前从未被读取。M7曾在项目旧轮用于人工状态序列，当前轮仍坚持仅用官方valid选择，并披露项目既往接触。

## 2. 三种任务与口径

| 任务 | 金标与输入 | 用途 | 不能混同 |
|---|---|---|---|
| M3 | 官方整体负/中/正；当下T/A/V输入 | 原821与外部1929的主分类 | 七类情绪分类、预测 |
| A3 | 官方语音单模态极性；音轨输入 | 中文语音表征公平控制 | 整体M3收益 |
| 未来M7 | 官方七类整体情绪；过去已完成话语的识别概率 | 同说话人跨度1/3/5预测 | 人工历史状态、长期心理状态预测 |

五种子42–46仅改变元分类头，基础专家/编码器固定；种子标准差不等于泛化置信区间。宏F1、准确率和中性召回分别报告。决策截距后的softmax不自动具有概率校准保证。

## 3. 输入与数据质量

- 对全部2,876条开发媒体解码音视频时间戳；同一0.25秒/10%规则下恢复 **{quality['restored_audio']}条** 原误拒观测，仍拒绝 **{quality['remaining_audio_rejected']}条**。保持原音频、画面质量标准，未放宽阈值。
- 恢复的全部{restored['n']}条音频/视觉向量与最初真实媒体缓存逐位相同，音频向量均非零；没有用补零或修改人工标签制造恢复，核验见validation/restored_inputs_bitwise.json。
- MSP原已编码有效行逐位保持，只补先前跳过的56条，避免把输入完整性收益都归到中文模型。最终703/821的同规则输入审计在input_audit中单独记录。
- SenseVoiceSmall为引入的冻结开源编码器，233,999,167参数；推理dither=0，严格加载并核验权重。均值/总体标准差1024维、原生查询状态和原生情绪logprob均来自真实音轨，不是人工情绪金标。
- 视觉使用实际五帧HSEmotion表情/深层表示与三帧Chinese-CLIP；“最大脸”只是可核验代理，未做主动说话人身份确认。它们沿用既有编码器，不宣称本轮从零发明视觉基础模型。
- EmotionTalk固定提交adbc17fc944e8cf2873643906160c6ca0259ab61，共19,250个唯一媒体；903个重复归档成员逐字节确认相同，只输入一次。8种表征在train15,413/valid1,908/test1,929完整对齐。
- 4,400条CH媒体与外部媒体无字节级重复；长文本近似跨来源检查产生{near['candidate_count']}对候选，音轨相关复核疑似重编码重复{near['possible_reencoded_pairs']}对。该候选规则不能排除短改写、大剪辑或未知预训练重合。
- 官方speaker_id显示train与valid/test不交叉；valid/test共享部分ID。因此外部监督模型只拟合官方train，不把valid并入最后训练。
- 19,250条音轨时长与标注段长最大差约23.2毫秒；742段对话、1,476个同说话人序列存在108处区间重叠，窗口按真实结束时间核验。语音编码为居中最多10秒，1,240条长片段被截取；100%样本覆盖不等于全长覆盖。

## 4. 真实极性语义审计

外部M3严格使用发布的Continuous label_result符号，未由七类标签自动推断，未删surprised、未改符号。开发17,321条连续值都等于五位标注者均值，聚合错误为0；但happy中1,849/3,346的连续值为负，原因尚未确认。连续极性与七类情绪的关系具有明显差异，不能把外部M3与M7互相替代。详见corpus/label_semantics_audit.json。

来源：[EmotionTalk官方数据](https://huggingface.co/datasets/BAAI/Emotiontalk/tree/adbc17fc944e8cf2873643906160c6ca0259ab61)、[官方论文§3.2](https://aclanthology.org/2026.findings-acl.440.pdf)。本报告的标签统计来自本地发布标注逐条复核。

## 5. 开发选型、正结果与负结果

主线采用4折整来源外层、3折内层。M/T/A/V专家各用官方对应标注，折内拟合StandardScaler与FP64 PCA，内层OOF证据训练融合头。预先固定2种质量×3种语音×2种粒度、4种融合头；另有9个同预算神经对照、7B/3B/light部署对照。703仅用于锁定后的最终拟合，821不参与本轮候选筛选。

冲突在**各模态监督对齐的三类概率空间**中建模：JS(p_T,p_A)、极性期望差、各自回归强度、熵和可用性共40维证据。没有直接对不同维度、不同编码空间的原始文字/语音向量取余弦，并将其当可解释冲突真值。JS、GRU等是既有方法；本项目贡献在任务监督与证据链的实现和验证。

![当前识别与未来预测的可核验证据链](figures/evidence_chain_architecture.png)

相同旧质量和C=.1的A3控制：MSP开发偏置后 **{audio['old_bias_selected_f1']:.6f}** → 中文 **{audio['new_bias_selected_f1']:.6f}（{audio['new_bias_selected_f1']/audio['old_bias_selected_f1']-1:+.2%}）**；未校准argmax {audio['old_argmax_f1']:.6f}→{audio['new_argmax_f1']:.6f}。这是开发子任务收益。

同搜索预算完整M3开发对照约0.716913→0.728883，中性F1约0.4630→0.4959；T/A相反198条/55来源，macro-F1约0.474760→0.541593（+14.08%）。固定开发预测的来源条件区间：整体差[-0.00457,0.02957]跨0；T/A相反差[0.01835,0.11833]为正。它们不是独立外部确认。

最终来源选型使用new_weight=1、bias=[.15,0,0]，开发macro-F1=0.728403、来源等权=0.736028。旧模型混合、五档专家粒度和附加可靠性没有胜出。不能用原始家族0.728883冒充最终截距后的0.728403。

文字缺失专用退化策略：仅合并真实可用的A/V证据，开发macro-F1 **{policy['plain_meta_missing_T']['macro_f1']:.6f}→{policy['winner']['metrics']['macro_f1']:.6f}**；T完整时逐位保持原决策。极端全缺失使用训练先验，不把缺失伪装成中性。

MoE/Top-k/反事实对照最好交换一致性约0.682051，未超过证据融合。Top-k确实只执行选中的M专家头，但基础模态投影仍需执行；没有实际端到端加速证据时不声称加速。交换教师为eval/no_grad，生成样本不赋虚构整体硬标签；它也不能识别真实因果效应。个人Welford/EMA/CUSUM的线性时序增强没有胜出，不宣称是已验证的性能创新。

![中文语音开发对照](figures/audio_development_comparison.png)

固定成功/回退案例的M/T/A/V来源留出预测另见cases/expert_evidence.json。图中是预测概率，人工T/A/V参考标签单列，二者不混同；这不是因果归因或独立测试证明。

![开发案例的真实专家预测证据](figures/case_expert_evidence.png)

## 6. 原821锁定历史诊断

| 方案 | 五元头种子平均macro-F1 | 种子标准差 | 概率集成macro-F1 | 集成中性F1 | 集成中性召回 |
|---|---:|---:|---:|---:|---:|
| 第十轮冻结对照 | {main_old:.6f} | {previous_result['std_macro_f1']:.6f} | {previous_result['ensemble']['macro_f1']:.6f} | {previous_result['ensemble']['per_class_f1'][1]:.6f} | {previous_result['ensemble']['per_class_recall'][1]:.2%} |
'''
    for mode in ("7b","3b","light"):
        r=historical[mode];m=r["ensemble"]
        report+=f"| 第十一轮{mode} | {r['mean_macro_f1']:.6f} | {r['std_macro_f1']:.6f} | {m['macro_f1']:.6f} | {m['per_class_f1'][1]:.6f} | {m['per_class_recall'][1]:.2%} |\n"
    report+=f'''
821个ID、原标签和覆盖率均保持不变。配对30来源、5,000次重采样，五元种子平均分数差的95%条件区间为{interval['paired_delta_95ci']}。此前历史暴露使其仍属于诊断；不追加根据该分数循环选型。三档对照即便历史排序与开发排序不同，也不据此重选主模型或改权重。第十轮五元头中性召回均值约22.35%，与本表集成中性召回约21.18%是不同口径。

原始0.70268门槛仅按五元头均值验收。本轮7B有{sum(r['macro_f1']>=.70268 for r in historical['7b']['seeds'])}/5个单独元头达到门槛，概率集成{'达到' if historical['7b']['ensemble']['macro_f1']>=.70268 else '未达到'}门槛；不表述为每个种子或每个部署模型都稳定提高10%。

## 7. 外部1929条M3锁定测试

| 方案 | 训练条件 | 五元头平均macro-F1 | 集成macro-F1 | 集成中性F1 | 集成中性召回 | 覆盖率 |
|---|---|---:|---:|---:|---:|---:|
'''
    for key,name in (("old","冻结第十轮"),("new","冻结第十一轮"),("light","冻结轻量"),("adapted","域适应对照")):
        m=erows[key]["ensemble"]
        seed_mean=f"{erows[key]['five_seed_mean_macro_f1']:.6f}" if key!='adapted' else "单模型"
        report+=f"| {name} | {'CH训练后直接迁移' if key!='adapted' else '额外EmotionTalk train监督'} | {seed_mean} | {m['macro_f1']:.6f} | {m['per_class_f1'][1]:.6f} | {m['per_class_recall'][1]:.2%} | 100% |\n"
    report+=f'''
冻结第十一轮相对冻结第十轮集成的绝对F1变化：{erows['new']['ensemble']['macro_f1']-erows['old']['ensemble']['macro_f1']:+.6f}；按会话配对重采样的95%区间{erows['new_vs_old_paired']['delta_macro_f1_95ci']}。域适应是额外真人监督条件，不拿它与冻结迁移当作同预算提升。全部候选只用valid选择，测试前保存模型、决策、无标签逐条概率与摘要；只计算一次测试结果。

官方test只有两个录制组，结果仅支持这些发布来源下的跨数据集验证，不能宣称跨所有人群普适。各来源分数、各类召回和混淆矩阵保存在external_m3/results.json。即使某个迁移结果回退，也完整保留。

![主任务与外部测试](figures/main_and_external.png)

## 8. 从真实媒体观测到未来预测

感知器以当下官方转录+原始音轨识别七类概率；训练历史使用5折整录制组OOF，valid/test只用官方train拟合感知器。未来标签只作为预测监督目标，人工历史情绪不进入部署输入。不是自动在线转录条件。

对比观测保持、历史均值、EMA、线性历史、个人漂移、GRU，以及验证集固定四分之一网格概率投票。GRU隐藏32、上下文5/10/20、检查点5/15轮；训练和时长标准化只用train。

| 预测设置 | 测试窗口 | 保持最近观测 | 验证锁定预测器 | 相对变化 | 配对差95%区间 |
|---|---:|---:|---:|---:|---|
'''
    for r in forecast["rows"]:
        a=r["metrics"]["baseline"]["macro_f1"];b=r["metrics"]["selected"]["macro_f1"]
        report+=f"| {r['setting']} | {r['eligible_test_windows']} | {a:.6f} | {b:.6f} | {b/a-1:+.2%} | {r['paired']['delta_macro_f1_95ci']} |\n"
    report+=f'''
lead=.25含时轴对齐余量；1.25额外预留1秒推理预算。截止时刻取全部已完成过去话语的最大结束时间，目标开始不得早于截止。所有合格窗口都预测；“覆盖100%合格窗口”不能写成全部1,929话语都能预测。h3/h5两种lead资格相同，不是四次独立重复试验；区间未做多重比较校正，条件于现有录制来源。

真实音轨时长与发布段长一致，并不独立证明每条片段在完整录制中的绝对起点；时序结论以官方起止标注与实际解码时长为条件。若接入平台实际输入，还需测量转录与说话人关联的运行误差。

某些测试控制（如h1线性）比验证锁定胜者更好，仍不据测试结果换方案。未来预测的F1并非七类当下感知F1（valid约0.569767），更不是长期情绪趋势或心理问题分类的证据。

固定第一个合格测试窗口的原媒体重放：真实历史{raw['history_size']}条；历史识别最大概率误差{raw['observation_max_probability_error']:.8g}，未来预测最大误差{raw['forecast_max_probability_error']:.8g}，重复单条误差{raw['repeat_observation_max_error']:.8g}。没有按预测成功挑重放例子，也没读取该窗口目标金标。详见forecast_replay/results.json。

![未来情绪预测](figures/forecast_test_comparison.png)

## 9. 缺失、错配与噪声压力

全部2,876条开发样本保留原ID/金标，压力输入只经过来源留出模型。错配供体限制在同一外层留出折、不同来源，避免把训练来源供体带入留出验证。

旧控制为family00/depth4及其开发偏置；新方案为family10/depth4、最终开发偏置和缺字退化路径。它们共享外层来源与头训练预算，包含表征、质量与决策组件的组合差异；完整冻结第十轮模型的迁移比较在上一节单列。因此这里的全部差距不能归因于单个中文编码器。

真实波形加入20dB/10dB高斯噪声，按输入RMS定标、截幅[-1,1]；对同一居中10秒波形重新提取MFCC、SenseVoice、MSP，另算相同区间干净控制。不是向embedding加随机数；也不是实际环境录音。为隔离表征退化，观测质量mask固定，不能用它评估自动噪声检测器。

| 条件 | n | 旧表征同预算macro-F1 | 新锁定方案macro-F1 |
|---|---:|---:|---:|
'''
    for r in robust["rows"]:report+=f"| {r['condition']} | {r['n']} | {r['old_same_budget']['macro_f1']:.6f} | {r['new']['macro_f1']:.6f} |\n"
    report+="\n固定可观测质量分层（质量阈值0.5、人脸覆盖阈值0.6；总体样本不删除）：\n\n| 范围 | 分层 | n | 旧方案宏F1 | 新方案宏F1 |\n|---|---|---:|---:|---:|\n"
    for split in ("development","external_test"):
        for row in quality_slices[split]:
            report+=f"| {split} | {row['stratum']} | {row['n']} | {row['old']['macro_f1']:.6f} | {row['new']['macro_f1']:.6f} |\n"
    report+='''
压力成绩仍是开发诊断；文字缺失策略也在开发端选型，不将其当独立确认。各压力条件是否改善逐项判断，不能用“整体鲁棒性全面提高”概括任何回退。

![压力诊断](figures/robustness_development.png)

## 10. 实际推理成本与展示建议

'''
    report+=f"轻量版12条固定开发片段、每条2次、batch1，全程读取MP4并重新解码/编码/融合：模型加载{runtime['model_startup_seconds']:.3f}s，加载+首次推理{runtime['model_start_plus_first_inference_seconds']:.3f}s，热请求p50={runtime['hot_p50_seconds']:.3f}s、p95={runtime['hot_p95_seconds']:.3f}s；CUDA峰值分配{runtime['cuda_peak_allocated_MiB']:.0f}MiB。\n\n"
    report+="| 策略 | 请求数 | 首次请求秒 | 热策略p50秒 | 热策略p95秒 | 峰值分配MiB |\n|---|---:|---:|---:|---:|---:|\n"
    for mode in ("3b","7b"):
        r=staged[mode];policy_name=f"{mode}常驻" if "co-resident" in r["policy"] else f"{mode}隔离进程按需加载"
        report+=f"| {policy_name} | {r['n_requests']} | {r['first_request_seconds']:.3f} | {r['warm_policy_p50_seconds']:.3f} | {r['warm_policy_p95_seconds']:.3f} | {r['peak_cuda_allocated_MiB']:.0f} |\n"
    report+=f'''
3B能够与感知编码器常驻，7B使用独立进程分阶段：新进程加载感知编码器→处理本请求真实媒体→退出→新进程加载语言分支→处理本请求文本→退出→CPU融合。7B的同请求中间NPZ只作进程通信，是请求开始后新提取的值；进程启动、模型加载与IPC均计入。其“热策略”仅表示OS缓存未清空，GPU和CPU模型上下文仍逐请求重建。与轻量/3B常驻请求的策略不同，不能称同一常驻条件下的纯架构速度比。

语言分支保留原NF4解码器和三个实际使用词的原输出权重行，省略未执行的全词表输出头。按原冻结开发批次核验：7B的8条、3B的24条，llm_last/llm_mean/llm_scores最大差均为0；详见validation/language_runtime_*_equivalence.json。原全头加载路径曾出现原生进程退出，失败记录保留在runtime_staged/attempt1和attempt2。该等价分支不改变已锁定主分类权重或测试决策。

这是单台共享笔记本、有限片段样本的观测，未清OS文件缓存，不是服务端冷启动分布、网络延迟或生产SLA。输入假设已有官方转录，未计在线ASR生成转录。T+A预测重放单条热识别{raw['warm_last_observation_seconds']:.3f}s、预测头{raw['forecast_head_seconds']:.6f}s，单例不能建立p95或严格实时保证。

答辩现场建议以轻量原媒体推理作为操作演示，完整7B结果用已锁定表格/概率文件呈现；时序演示播放固定因果窗口。明确显示“当前识别”“未来预测”“模态可用性”与验证范围。

![实测质量成本](figures/measured_quality_cost.png)

## 11. 可复现性与完成度验收

关键测试{test_summary['tests']}项，失败{test_summary['failures']}、错误{test_summary['errors']}。独立重训基于同一冻结表征，从头拟合四折归一化/PCA、专家和元分类头，不是重新预训练基础编码器；副本与冻结7B模型概率最大误差{replay['independent_refit']['probability_max_error']:.8g}。冻结历史821与外部1929逐条概率重放一致，改变传入金标不影响推理。锁定清单内模型/选型摘要逐项复核，第十轮126项工件0变化。

COMPLETION-AUDIT.json逐项列出交付验收，原始相对第九轮的10%目标与本轮相对第十轮的增量分开记录。源代码快照、运行环境、模型、逐样本概率、CSV、来源区间、正负案例和复现/重放命令均在REVIEW-PACKAGE.md导航。各负结果保留，不通过换指标、换样本、反复看旧测试制造增益。

## 12. 答辩创新的准确定位

1. **可追溯的数据质量与跨模态证据建模**：先纠正真实观测时轴，再用各自真人监督的M/T/A/V证据表达冲突、可用性与强度；收益分别做因子与同预算控制。
2. **适合中文的语音适配与缺失退化**：冻结中文语音模型和旧编码器联合，按来源隔离验证；文字缺失有明确A/V退化路径。创新在任务适配和证据组织，不冒称发明SenseVoice或MoE。
3. **由当下感知走向未来预测的可重放链路**：从真实过去媒体识别概率出发，按真实完成时间形成因果窗口，比较简单基线/GRU/投票，在锁定测试上报告全合格窗口、统计区间与实际成本。

长期跟踪、心理问题类别、跨未见人群普适性、因果机制均未被这批实验直接证实。EMA/CUSUM/GRU/Top-k本身是已有方法，不能单独包装成原创算法。详细话术、五页展示安排和问答见《答辩创新点与演示讲稿.md》。

## 13. 客观评价与后续最值得投入的方向

实验交付完成，性能收益仍有明确边界：主线7B整体历史指标基本持平；预设3B对照的历史中性表现更好，但不能据此反复用历史测试选择方案。缺音频、仅文字和语音错配的开发压力出现回退，因此本轮没有证明全场景鲁棒性提升。

若开展下一轮，优先级应为：

1. **训练侧的缺音频/错配退化**：在训练来源内加入模态缺失增广与质量条件路由，直接解决本轮测出的回退；需要新的封存验证来源，不能继续围着已接触的821条筛选。
2. **中性与跨语料标签可比性**：保留原标签，增加多人独立标注的一致性审计，核实连续极性与七类情绪差异；不擅自将happy重标为正向。青少年真实场景与当前公开语料的外推还需独立采样。
3. **面向部署的蒸馏与输入条件**：用训练来源OOF教师蒸馏轻量模型，评估自动转录/说话人关联误差下的真实延迟与质量；量化或改编码器后重新验证，不能沿用未重新计算的质量数字。
4. **更长时间范围的预测**：当前是话轮级短期情绪预测。跨天个人趋势需要新的连续真实观测及时间边界，不能把现有跨度1/3/5直接表述为长期心理状态预测。

这些是下一轮研究方向，不计入本轮已实现的性能收益；本轮13项交付清单中的无效方法也完整保留了对照证据。
'''
    (OUTPUT/"第十一轮完整优化与独立验证报告.md").write_text(report,"utf-8")
    defense=f'''# 答辩创新点与演示讲稿

## 一句话定位

我们把多模态情绪识别做成可追溯、可退化、可预测的证据链：先确认真实观测质量，再联合文字/语音/视觉判断当下情绪，最后只用过去已经完成的观测预测下一次发言。

## 约90秒精简讲稿

“我们完成了三个实质改进。第一，核验真实音视频时间戳，在原质量阈值下恢复47条误拒观测。第二，让各模态使用对应人工监督，对齐到同一概率空间，再融合冲突与可用性证据；开发来源中，198条文字语音相反样本的宏F1从0.475提高到0.542。第三，只使用已经完成的真实过去观测预测未来，不输入人工历史情绪；在官方测试录制组的1,000个合格下一话轮窗口，七类预测宏F1从0.365提高到0.448，相对约22.6%。

主分类仍保留原821条，整体指标约0.703，与上一轮基本持平，外部迁移也有回退。所有负结果完整保留。我们的项目贡献在于把可靠观测、软冲突证据和真实未来预测做成可复核链路；3B常驻版原媒体请求实测约2秒，可现场演示。”

## 约2–3分钟展开讲稿

“我们的改进主要落在三个环节。首先是数据质量。我们发现一些音轨并未缺失，而是容器时长元数据导致观测被错误拒绝。通过实际解码时间戳，在原质量阈值下恢复了47条开发观测，并用相同观测条件分离模型本身的收益。

其次是跨模态证据。文字积极不代表声音也积极，因此我们不把整体情绪标签直接当作每个模态的真值。各模态使用对应人工监督，再把概率、强度、冲突和可用性交给融合器。固定开发来源中，文字与语音相反的198条样本，宏F1从0.4748提高到0.5416；这个结果明确标注为开发验证。文字缺失时则退化到仍有证据的语音和视觉。

第三是预测链路。我们不把人工标好的历史情绪送进预测器，而是先识别真实过去媒体，并严格核验历史已结束、未来尚未开始。在官方测试录制组的1,000个合格下一话轮窗口上，七类预测宏F1从保持最近观测的{h1['metrics']['baseline']['macro_f1']:.4f}提高到{h1['metrics']['selected']['macro_f1']:.4f}，相对{h_gain:.1%}，并提供逐窗口概率、来源区间和原媒体重放。

主分类仍保持原821条样本，五个元分类头种子的均值是{main_new:.4f}，第十轮是{main_old:.4f}。我们把主分类、语音子任务、未来预测分别汇报，完整保留没有胜出的MoE和反事实实验，也测量了原始媒体推理成本。”

## 五页展示安排（约4分钟）

| 页 | 标题 | 主画面 | 讲述重点 | 时间 |
|---|---|---|---|---:|
| 1 | 一个难点：同一句话，文字和声音未必一致 | 先固定的开发成功案例+一个失败案例 | 真人T/A/V标签可以相反；单例用于解释，整体表用于验证 | 35秒 |
| 2 | 从可靠观测到冲突证据 | 实际PTS→中文语音/视觉→M/T/A/V证据→融合器 | 47条恢复、同观测/同预算控制、各模态自己的监督 | 50秒 |
| 3 | 原主任务与外部检验 | figures/main_and_external.png | 原821五元种子均值、外部1929覆盖；域适应额外监督单列 | 45秒 |
| 4 | 只用过去预测未来 | 时间轴+figures/forecast_test_comparison.png | 显示截止线；未来媒体不进入输入；1000窗口和条件区间 | 65秒 |
| 5 | 能演示，也能复核 | 轻量原媒体结果、质量成本图、重放误差 | 实测策略和硬件范围、开源模块与团队贡献、保留负结果 | 45秒 |

## 演示步骤

1. 使用cases/defense_cases.json中按既定开发规则保存的案例，播放原片段并并列显示T/A/V人工标签与来源留出预测。案例页固定显示“开发解释案例，不是独立测试成绩”。同时展示一个回退例。
2. 展示来源质量记录：旧时长、实际解码时间戳、原阈值和新观测可用性；解释这是观测测量纠错，阈值没有放宽。
3. 展示逐窗口预测CSV与forecast_replay/results.json的固定窗口。先显示历史ID和完成时间，再显示未来概率，最后才播放未来片段。不要把未来原文提前放在模型输入栏。
4. 现场推理优先使用轻量版。先完成模型加载，再运行原媒体请求；页面/讲解标明当前演示是否实时重新计算。演示固定回放时直接称“可复核回放”，不把缓存播放伪称现场在线推理。

现场可运行`scripts/round11_demo.py`：默认轻量版、固定第一个外部输入；`--mode 3b`使用已实测的3B常驻路径。读取真实MP4与当前转录，输出五元头集成概率、可用性与实际耗时；也支持`--clip`和`--text`传入新片段。这是功能演示，不利用人工情绪金标选成功例或重新估计泛化成绩。

## 常见追问及回答

**“你们的算法创新到底是什么？是不是堆开源模型？”**

“基础编码器来自开源。团队工作在观测质量纠错、对应模态监督、来源交叉拟合的冲突证据融合、明确缺失退化，以及真实过去观测到未来预测的因果时间链路。我们用同观测、同预算和负结果区分哪些改动有效。我们不把已有的GRU、CUSUM或MoE说成独创基础算法。”

**“为什么没有采用批注建议的MoE和反事实训练？”**

“我们确实实现了动态/Top-k专家、交换一致性和软教师等9组同预算对照。最好约0.6821，未超过证据融合，所以没有为架构名称牺牲实测效果。反事实合成样本也没有虚构整体硬金标，因此不声称识别了因果效应。”

**“22%以上提升是整体分类吗？”**

“不是。它是未来七类情绪预测相对观测保持基线的测试提升。主任务M3单独按原821条、五个元分类头种子均值报告：{main_old:.4f}→{main_new:.4f}，相对{gain:+.2%}。{goal_sentence}”

**“是不是把人工历史情绪直接喂给模型？”**

“没有。训练历史由整录制组交叉拟合的感知器给出识别概率，测试历史由只在官方train训练的感知器给出。未来文本、未来音轨和未来标签不进入预测输入；时间戳检查与变异测试、原媒体重放均已保存。”

**“为什么7B在历史诊断中没有胜过3B？”**

“模型规模不保证任务效果，开发与历史来源的排序也不完全一致。3B是预先设定的部署对照，结果完整保留；我们没有据历史分数重新调权重或换主模型。部署要按明确的质量/资源约束选择，并用新的独立来源确认这种排序是否稳定。”

**“为什么外部测试的happy可能是负向？”**

“我们按官方发布连续极性原值评测，逐条核验了五位标注者均值。七类情绪和连续极性存在明显差异，原因仍待数据作者确认；我们没有用happy重标正向，也没有删除冲突样本。M3和M7保持为不同任务。”

**“这个结果能推广到所有学生或长期心理状态吗？”**

“这批结果支持当前发布语料、录制来源和短期话轮情绪预测。外部测试只有两个录制组，不能据此证明所有人群或长期心理状态效果。我们报告来源分数、会话条件区间和失败案例。”

**“速度是不是只测了缓存分类头？”**

“轻量版24次请求都从MP4重新解码并运行编码器，热p50约{runtime['hot_p50_seconds']:.3f}秒，p95约{runtime['hot_p95_seconds']:.3f}秒。3B常驻路径和7B隔离加载路径另列，后者完整包含每次初始化与进程通信成本；这是两种可运行策略，不能称同常驻条件下纯架构速度比较。”

**“有没有证明任何模型都更强？”**

“没有。更复杂的MoE、反事实和CUSUM分支没有胜出；某些压力或外部来源也可能回退。我们只对实际通过对照的范围作结论，原始逐样本预测和负结果都可抽查。”

## 答辩前核对

- 页面上的M3/A3/未来M7、开发/历史诊断/外部测试、五种子均值/概率集成必须写清。
- 不把相对提升百分比当作绝对百分点，不把预测合格窗口覆盖率当作全部话语覆盖率。
- 原始10%目标参照第九轮0.6388，本轮相对第十轮的新增变化另列；用已证实的贡献组织展示，不转换验收口径。
- 先固定演示片段及失败例、检查本地模型路径和供电状态，预留模型加载时间；保留一份明确标注为回放的结果作为展示工件。
'''
    (OUTPUT/"答辩创新点与演示讲稿.md").write_text(defense,"utf-8")
    casefolder=OUTPUT/"cases";casefolder.mkdir(exist_ok=True)
    save(casefolder/"defense_cases.json",{"comparison":conflict["cases_comparison"],"cases":conflict["cases"],
        "scope":"preexisting development source-OOF controlled comparison;not full3579 in-sample predictions or independent test performance",
        "media_root":str(OUTPUT.parents[1]/"data/public/ch_sims_v2/supervised"),"both_success_and_regression":True},immutable=False)
    repro='''# 复现与重放

在Windows PowerShell中，工作目录为本项目代码库。所有旧工件只读；完整数据/模型路径以PROTOCOL与manifest记录为准。

```powershell
Set-Location -LiteralPath 'D:\\Develop\\AIC\\AI-mental-main (2)\\AI-mental-main\\AI-mental-main'
$taskPython = 'D:\\Develop\\AIC\\.venv\\Scripts\\python.exe'
$env:PYTHONPATH = 'algorithm'
& $taskPython -m pytest algorithm/tests/test_round10_protocol.py algorithm/tests/test_round11_protocol.py -q
& $taskPython scripts/round11_replay_bundle.py --retrained 'D:\\Develop\\AIC\\_国赛优化\\round11\\validation\\refit7b\\model.pkl'
& $taskPython scripts/round11_demo.py
```

重新训练到全新目录（保留冻结模型，不复用最终训练OOF；预先准备已核验的冻结输入特征）：

```powershell
& $taskPython -m experiments.round11_final fit --mode 7b --destination 'D:\\Develop\\AIC\\_国赛优化\\round11\\validation\\another_refit7b'
& $taskPython scripts/round11_replay_bundle.py --retrained 'D:\\Develop\\AIC\\_国赛优化\\round11\\validation\\another_refit7b\\model.pkl'
```

已完成的实验入口拒绝覆盖选型/测试结果。`round11_forecast_replay.py`/`round11_runtime.py`/`round11_runtime_staged.py`内提供实际输入推理实现，完成标记存在时不会重复跑整批基准。逐样本重放入口始终从保存权重重新计算概率并验证锁，不再重选或重评候选。

开发主实验：`-m experiments.round11_evidence`；神经消融：`-m experiments.round11_neural`；未来线性：`-m experiments.round11_forecast`；GRU/投票与锁定评测：`-m experiments.round11_forecast_finish`；外部域适应与评测：`-m experiments.round11_external_finish adapt/evaluate`。准确参数、随机种子、模型版本和代码摘要以各目录preregister与selection_locked为准。

完整媒体提取：`scripts/round11_external_text.py`、`round11_external_gpu.py`、`round11_cpu_resume.py`；输入校验与实际读取由各自manifest绑定。大型GPU工作依次执行，不启动相互争用显存的作业。所有CPU解码仅1线程、队列有界；资源中断不记为媒体缺失。

最终源码快照在source/final，依赖版本在environment-final.json。原始媒体、预训练权重和早期冻结缓存是运行前置条件，不打包成可随意再分发的文件。
'''
    (OUTPUT/"REPRODUCE.md").write_text(repro,"utf-8")
    index=f'''# 第十一轮交付包

- [完整优化与独立验证报告](第十一轮完整优化与独立验证报告.md)
- [答辩创新点与演示讲稿](答辩创新点与演示讲稿.md)
- [逐项完成度验收](COMPLETION-AUDIT.json)
- [复现与重放命令](REPRODUCE.md)
- 最终模型：final_models/{{7b,3b,light}}/model.pkl；均有锁定配方、3579训练ID/来源与五元头种子。
- 原821一次历史诊断：historical_diagnostic/results.json、7b/3b/light_predictions.npz、paired_interval.json。
- 外部1929 M3：external_m3/results.json、evaluation_lock.json、test_inputs_scored_predictions.npz；额外监督适应在adaptation单列。
- 真实观测未来M7：forecast_final/results.json，各设置test_predictions.npz；原媒体重放在forecast_replay/results.json。
- 实际推理：runtime_benchmark/results.json（轻量常驻24请求）、runtime_staged/results.json（3B常驻与7B隔离按需策略各12请求）。语言分支与原冻结表示误差0，失败加载路径单列归档。
- 全2876压力：robustness/results.json与逐条概率；真实波形重新编码wave_stress。
- 来源/时轴/重复：emotiontalk_acquisition、timestamps、input_audit、near_duplicate_audit.json。
- 消融：evidence_deterministic、neural、deployment_3b、deployment_light、missing_route；负结果保留。
- 统计图：figures；汇总及逐条CSV：exports；成功与失败案例：cases。
- 复核：validation/final_tests.xml、final_replay.json、external_feature_coverage.json；环境environment-final.json；源码source/final；摘要artifact_manifest_final.json。

工程交付{len(checks)}/{len(checks)}项完成。主指标较第十轮相对变化{gain:+.2%}；相对第九轮原始报告0.6388变化{historical['round11_relative_gain_vs_original_round9']:+.2%}，原始10%阈值：{'达到' if historical['round11_meets_original_10percent_threshold'] else '本轮未达到，第十轮保留模型已达到'}。未来预测等子任务较大收益不能代替主指标验收。

阶段进展与RUN_STATE是历史记录；最终结论以本页链接的完整报告及锁定结果为准。
'''
    (OUTPUT/"REVIEW-PACKAGE.md").write_text(index,"utf-8")
    packages=("numpy","scipy","scikit-learn","torch","transformers","catboost","funasr","opencv-python","av","librosa","onnxruntime","bitsandbytes")
    env={"time":now,"python":sys.version,"packages":{},"platform":sys.platform}
    for package in packages:
        try:env["packages"][package]=version(package)
        except PackageNotFoundError:env["packages"][package]="not-installed-under-this-distribution-name"
    save(OUTPUT/"environment-final.json",env,immutable=False)
    package_versions={str(dist.metadata["Name"]):dist.version for dist in distributions() if dist.metadata.get("Name")}
    (OUTPUT/"environment-final-requirements.txt").write_text("\n".join(f"{name}=={package_versions[name]}" for name in sorted(package_versions,key=str.lower))+"\n","utf-8")
    repo=Path(__file__).resolve().parents[1];sources=[]
    for parent,pattern in ((repo/"scripts","round11*.py"),(repo/"scripts","round11*.ps1"),(repo/"algorithm/experiments","round11*.py"),(repo/"algorithm/tests","test_round11*.py")):
        for path in sorted(parent.glob(pattern)):
            relative=path.relative_to(repo);target=OUTPUT/"source/final"/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
            sources.append({"path":str(relative),"sha256":digest(path)})
    audit_folders={OUTPUT/name for name in ("validation","cases","figures","historical_diagnostic","external_m3","robustness",
        "runtime_benchmark","runtime_staged","observed_perception","exports")}
    artifact_paths=[p for p in OUTPUT.rglob("*") if p.is_file() and (p.name in ("model.pkl","selection_locked.json","evaluation_lock.json","results.json","test_predictions.npz","model_manifest.json","final_replay.json","final_tests.xml","paired_interval.json","locked_predictions.npz")
        or p.parent==OUTPUT and p.suffix in (".md",".json",".txt")
        or p.parent in audit_folders and p.suffix in (".json",".xml",".txt",".npz",".csv",".png",".svg"))]
    artifact_paths=[p for p in artifact_paths if "source" not in p.relative_to(OUTPUT).parts and p.name not in ("artifact_manifest_final.json","review_bundle.json")]
    save(OUTPUT/"artifact_manifest_final.json",{"generated":now,"sources":sources,"external_features":coverage,
        "artifacts":[{"path":str(p.relative_to(OUTPUT)),"sha256":digest(p),"bytes":p.stat().st_size} for p in sorted(artifact_paths)],
        "scope":"final result/model/decision files;external split features;earlier media per-ID hashes in acquisition records"},immutable=False)
    print(json.dumps({"reports_complete":True,"engineering_checks":len(checks),"main_F1":main_new,"relative_change":gain,
        "original_10percent_threshold_achieved":historical["round11_meets_original_10percent_threshold"]}),flush=True)


if __name__=="__main__":main()
