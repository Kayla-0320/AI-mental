"""汇总本轮正负结果、统计边界、原因排查与可调用模型交付。"""
from __future__ import annotations

from datetime import datetime,timezone,timedelta
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import xml.etree.ElementTree as ET
import zipfile
from typing import Any
from experiments.round11_common import digest
from experiments.round13_data import DATA,OUTPUT,save


def js(path: Path) -> Any:
    """读取已完成的结果工件。"""
    return json.loads(path.read_text("utf-8"))


def figure(ch: dict[str,Any],new_tests: dict[str,Any]) -> None:
    """生成可导出的静态比较图，开发与新测试明确分面。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    folder = OUTPUT/"figures"
    folder.mkdir(parents=True,exist_ok=True)
    fig,axes = plt.subplots(1,3,figsize=(17,5.5),layout="constrained")
    names = ("clean","missing_A","missing_T","text_only")
    rows = [next(v for v in ch["rows"] if v["condition"]==c) for c in names]
    x = np.arange(len(rows))
    axes[0].bar(x-.18,[r["old"]["macro_f1"] for r in rows],.34,label="Round11",color="#396c96")
    axes[0].bar(x+.18,[r["current"]["macro_f1"] for r in rows],.34,label="Availability route",color="#3c9b87")
    axes[0].set_xticks(x,names,rotation=20)
    axes[0].set_ylim(0,.85)
    axes[0].set_ylabel("Macro-F1")
    axes[0].set_title("Chinese development | previously exposed")
    axes[0].legend(fontsize=8)
    for ax,dataset in zip(axes[1:],("mosi","meld")):
        rows = new_tests[dataset]["rows"][:6]
        y = np.arange(len(rows))
        delta = np.array([r["delta"] for r in rows])
        intervals = np.array([r["paired_group_ci95"] for r in rows])
        ax.errorbar(delta,y,xerr=np.maximum(np.stack((delta-intervals[:,0],intervals[:,1]-delta)),0),fmt="o",color="#3c9b87",capsize=3)
        ax.axvline(0,color="#666",linewidth=.8)
        ax.set_yticks(y,[r["condition"] for r in rows])
        ax.invert_yaxis()
        ax.set_xlabel("Candidate minus ordinary fusion | macro-F1")
        ax.set_title(dataset.upper()+" | new locked test | 95% group CI")
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("Different training conditions and languages: development gains do not imply universal transfer",fontsize=11)
    fig.savefig(folder/"round13_results.png",dpi=180)
    fig.savefig(folder/"round13_results.pdf")
    plt.close(fig)


def main() -> None:
    """验证工件齐全后写报告、代码快照、复现导航和审阅包。"""
    ch = js(OUTPUT/"final_ch/development_results.json")
    core = js(OUTPUT/"training/ch/selection_locked.json")
    neural = js(OUTPUT/"neural/ch/selection_locked.json")
    tests = {d:js(OUTPUT/"new_test"/d/"results.json") for d in ("mosi","meld")}
    verified = js(OUTPUT/"verification.json")
    secondary = js(OUTPUT/"secondary_holm.json")
    test_xml = ET.parse(OUTPUT/"tests.xml").getroot()
    suites = list(test_xml.iter("testsuite"))
    count = sum(int(s.get("tests","0")) for s in suites)
    failed = sum(int(s.get("failures","0"))+int(s.get("errors","0")) for s in suites)
    if failed or not count:
        raise ValueError("关键测试未通过")
    generated = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    package = Path(__file__).resolve().parents[2]
    paths = list((package/"algorithm/experiments").glob("*.py"))
    paths += list((package/"algorithm/perception/knowledge_moe").glob("*.py"))
    paths += [package/"algorithm/tests"/n for n in ("test_round11_protocol.py","test_knowledge_moe.py","test_round12_exploration.py","test_round13_protocol.py")]
    paths += [package/"AGENTS.md"]
    for path in paths:
        target = OUTPUT/"source"/path.relative_to(package)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
    versions = {n:importlib.metadata.version(n) for n in ("numpy","scipy","scikit-learn","catboost","torch","pytest","matplotlib","huggingface-hub")}
    save(OUTPUT/"environment.json",dict(generated=generated,python=platform.python_version(),platform=platform.platform(),
        packages=versions,seed=42,training="CPU;BLAS1,torch2;frozen public/local representations"),overwrite=True)
    catalog = dict(generated=generated,used=[
        dict(name="CMU-MOSI",task="original M3 by continuous sign",language="English",data_receipt=js(DATA/"mosi/receipt.json"),
             source="https://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/",protocol=js(OUTPUT/"datasets/mosi/manifest.json")),
        dict(name="MELD",task="official sentiment3, T/A released features",language="English",data_receipt=js(DATA/"meld/receipt.json"),
             source="https://github.com/declare-lab/MELD",protocol=js(OUTPUT/"datasets/meld/manifest.json")),
        dict(name="NTUSD",source="https://github.com/ntunlplab/NTUSD",commit="a25ab90f13e483d0786230ac3e016858176f54bd",license="MIT",
             normalization="OpenCC TSCharacters character map, Apache-2.0; snapshot hash recorded, no claim of full phrase conversion",
             files={p.name:digest(p) for p in (DATA/"ntusd").glob("*.txt")})],
        excluded=[dict(name="MER2023/MER2024",reason="gated approval required; did not bypass or accept agreements on user's behalf"),
                  dict(name="CH-SIMS v1",reason="included/reannotated in v2, cannot serve as independent new corpus"),
                  dict(name="CHERMA",reason="7-class modality emotion task lacks comparable original M3; processed mirror about67.6GB, local free space about55GB at inspection"),
                  dict(name="ECISA2019/2021 local test",reason="already used in round5 confirmation; not fresh"),
                  dict(name="synthetic Chinese emotion dialogue data",reason="not suitable as independent human multimodal M3 gold")],
        exposure_audit="no MOSI/MELD references found in current prior experiment/scripts; does not prove absence of undocumented historical or foundation pretraining exposure")
    save(OUTPUT/"DATA-CATALOG.json",catalog,overwrite=True)
    figure(ch,tests)
    lines = ["# 第十三轮综合优化、公开新语料验证与原因排查报告", "", "生成时间："+generated, "",
        "## 最终结论", "",
        "本轮完成公开数据检索、接入、完整联合表示与可用性专头搜索、知识词典/打乱消融、冻结专家MoE、完整融合OOF教师蒸馏、新语料封存测试、最终拟合与重放交付。最终采用**保留第十一轮7B联合主模型＋按可用性切换Top-2专家退化分支＋开发端固定决策偏置**。没有把知识、MoE和蒸馏强行全部串联。", "",
        "**主要收益是缺模态条件，完整输入整体基本持平，不能声称全场景或跨语料全面提升。** 中文数字来自此前反复接触的2876条开发来源；不是新的独立中文测试。缺音频/缺文字/仅文字的差区间均为正，但本轮候选和偏置也在这些开发预测上选择，区间未经开发选择校正。", "",
        "新MELD测试的缺文字压力项有改善；MOSI验证集选中的蒸馏学生在测试回退，完整保留。MELD缺音频没有确认改善。新语料均是英语，使用各自官方训练监督和原生公开特征；**不是原中文模型冻结零样本迁移，更不是中文青少年场景的独立确认**。", "",
        "![开发与新语料结果](figures/round13_results.png)", "",
        "## 1. 中文主任务与准确率约束", "",
        "保留原M3负/中/正、全部2876开发样本、4折整来源边界、种子42。原7B来源OOF对照取第十二轮保存的同一开发压力预测，本轮逐折验证音轨错配供体完全相同。完整可用输入保留原决策；自然缺失的9条仍按实际可用性触发新分支。", "",
        "|条件|旧macro-F1|最终macro-F1|旧准确率|最终准确率|旧中性F1|最终中性F1|来源配对差95%区间|", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in ch["rows"]:
        a,b = r["old"],r["current"]
        lines.append(f"|{r['condition']}|{a['macro_f1']:.6f}|{b['macro_f1']:.6f}|{a['accuracy']:.6f}|{b['accuracy']:.6f}|{a['per_class_f1'][1]:.6f}|{b['per_class_f1'][1]:.6f}|{r['paired_group_ci95']}|")
    decision = js(OUTPUT/"final_ch/decision_locked.json")
    lines += ["", "先前未约束准确率的纯专头在仅文字条件下达到约0.6261，但准确率降到约0.6888。因此补充的开发决策网格同时要求：macro-F1至少高于旧方案0.01、准确率不低于旧方案超过0.01、中性F1不降低。网格为3档负向偏置×6档中性偏置，共18点，每个缺失状态独立选择；这属于额外开发选型预算，已单独锁定，不能冒充完全未选型的估计。", "",
        "最终启用的可用性编码为"+str(decision["enabled_patterns"])+"，T/A/V位权分别为1/2/4；固定偏置为"+str(decision["bias_by_pattern"])+"。缺视觉单独条件未达到启用要求，保留原模型。全缺失保留原训练先验退化；未知音轨错配没有解决。偏置后的softmax只用作决策概率，不承诺校准。", "",
        "## 2. 本轮完整消融", "", "每模态保留原始联合表征，训练折内StandardScaler/PCA最多96维；普通头和可用性专头均监督原M3整体目标。这个M3单模态输入专头与官方T/A/V各自情感标签的专家是不同任务。", "",
        "|候选|类型|干净macro-F1|五压力均值|干净中性F1|", "|---|---|---:|---:|---:|"]
    for r in core["rows"]:
        v=r["specific"]
        lines.append(f"|{r['config']['id']}|可用性专头|{v['metrics']['clean']['macro_f1']:.6f}|{v['robustness_mean']:.6f}|{v['metrics']['clean']['per_class_f1'][1]:.6f}|")
    for r in neural["rows"]:
        v=r["specific"]
        lines.append(f"|{r['config']['id']}|冻结专家门控/学生|{v['metrics']['clean']['macro_f1']:.6f}|{v['robustness_mean']:.6f}|{v['metrics']['clean']['per_class_f1'][1]:.6f}|")
    lines += ["", "核心8个候选，神经6个候选；CH做4来源外折，新语料按官方train/valid。普通头与专头的训练总成本不同，因为后者拟合多个子集头；不声称相同计算预算。神经对照每个100轮、相同种子与缺失采样，dense/top2共享固定专家银行；MLP不同KD权重共享架构、训练输入和更新次数。新神经实验不对错配样本赋硬标签。", "",
        "## 3. 知识、MoE和蒸馏为何未胜出，修复到什么程度", "",
        "以下区分代码事实、实验结果和原因推断，不把推断写成已经证明的因果机制。", "",
        "- **信息与训练目标**：第十二轮新头只消费约21维单模态概率/强度/质量，缺少原强联合M分支；原T/A/V专家又各自预测不同模态目标。这限制了从不完整输入推断整体M3的能力。本轮增加原始联合表示及直接监督整体M3的可用性专头。纯新模型仍弱于旧7B干净0.7284，因此保留旧完整输入分支。", "",
        "- **知识增强**：原词库仅6节点；本轮接入NTUSD并做字符简化、最长匹配、否定和转折，保留20515个长度至少2且极性不冲突的词条，开发折命中约77%～79%。但命中不代表提供了编码器之外的新信息，也不能可靠处理反讽、隐含事件及音画冲突。真实知识与打乱对照没有稳定整体优势；这仍是词典证据，不是专业审核心理知识图谱或完整COMET推理。最终模型不依赖这组知识。英语新语料不用于证明中文词典有效。", "",
        "- **MoE**：旧方案在很小的数据上让门控和弱专家一起学习，可能出现专家同质化、数据碎片化和联合分支优势被稀释。本轮先固定七个M3可用性专家，以3折来源OOF预测训练门控，并保留完整联合专家锚点。开发Top-2干净约0.7107，高于同银行稠密门控约0.6954；仍未超过旧完整模型，因此用于退化分支。Top-2推理先门控再实际调用选中专家，测试核对执行与稠密公式一致；不能据此声称端到端比普通单融合更快。", "",
        "- **蒸馏**：第十二轮教师只是联合专家，未达到完整融合系统强度；软分布损失与少数类重加权也可能冲突。本轮用完整联合输入M3融合教师，所有内层归一化/PCA/专家都不见受教样本来源，再蒸馏原始联合表示。CH匹配MLP从约0.6824到KD50约0.6978，有开发改善；MOSI开发改善却在新测试回退，说明修复教师边界并不保证泛化。最终未选蒸馏学生。", "",
        "所以，可用性适配和冻结强专家解决了部分重要指标；知识与蒸馏还没有足够证据支持进入最终主路径。参数容量、教师强度、类别目标与来源偏移共同影响结果，当前消融不能将全部变化归因于某一个因素。", "",
        "## 4. 公开新数据与一次性封存测试", "",
        "来源：[CMU-MOSI作者页](https://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/)、[实际固定版本镜像](https://huggingface.co/datasets/tamb2203579/CMU-MOSI/tree/092c41f8843df675ab8a169e41419e74e6ae05da)、[MELD作者仓库](https://github.com/declare-lab/MELD)、[官方特征包](https://huggingface.co/datasets/declare-lab/MELD/tree/9abc51ee7903424ffb971297608aa6d3d0de3bfa)、[NTUSD官方词典](https://github.com/ntunlplab/NTUSD)。文件版本、SHA256、许可说明、标签映射和来源边界见DATA-CATALOG.json及datasets/*/manifest.json。", "",
        "MOSI：train1284/valid229/test686，共2199，视频来源52/10/31互不交叉，原媒体全部对应；连续人类分数按符号映射三类。MELD：train9989/valid1109/test2610，本轮使用发布的T/A特征，V没有接入；直接使用人类Sentiment三类标签。MELD是官方对话划分，演员与剧集可能跨划分，不能称跨未见人物。MELD的缺V是无操作，缺A与仅T输入相同，不能当独立重复试验。", "",
        "准备过程仅为一致性核查及封存读取源标签，之后训练入口只打开train/valid。测试前保存选型、模型摘要、无标签全量逐条预测及摘要，再打开封存测试标签评分；每个新测试只做一次选型后的正式评分。重放只重新推理，不重新打开目标或重选方法。", ""]
    for dataset,result in tests.items():
        lines += [f"### {dataset.upper()}", "", "锁定候选："+result["choice"]["winner"]["config"]["id"]+"；对照为相同语料验证集选择的普通完整输入融合。两者都用了该语料官方train监督，不是CH模型零样本迁移。", "",
            "|条件|普通融合macro-F1|锁定候选macro-F1|差值|配对来源95%区间|", "|---|---:|---:|---:|---|"]
        for r in result["rows"]:
            lines.append(f"|{r['condition']}|{r['baseline']['macro_f1']:.6f}|{r['candidate']['macro_f1']:.6f}|{r['delta']:+.6f}|{r['paired_group_ci95']}|")
    sig=next(r for r in secondary["rows"] if r["dataset"]=="meld" and r["endpoint"]=="missing_T")
    lines += ["", "预设主要终点（干净、缺A、仅T、干净中性F1）没有确认新增优势。MELD缺T是预先列出的压力诊断项，宏F1提高约0.05934；探索性地将两语料全部7条件及干净中性F1共16个比较统一Holm校正后，p="+f"{sig['holm_adjusted_p']:.6f}"+"。它可作为该语料音频单模态输入退化路径的有支持结果，不能追认为预注册主要终点成功。来源bootstrap3000次、成组交换置换2000次，条件于现有发布来源。", "",
        "MOSI的多个条件回退，MELD缺A/仅T没有提高。未据这些测试结果重新训练、改偏置或选择别的候选。当前证据不支持跨数据集全场景增强，也没有取得新的独立中文三模态确认。MER需授权、CHERMA任务不等价且处理包约67.6GB、CH-SIMS v1重合、已有ECISA测试此前已用，均未伪装成新验证来源。", "",
        "## 5. 可调用模型、复现与验证", "",
        "最终中文权重在final_ch/model.pkl和head.pt，决策及原7B依赖SHA256在model_manifest.json。`experiments.round13_runtime.CHPredictor.predict(data)`消费与第十一轮一致的冻结T/A/V表征，自动选择真实可用性分支，不接收情绪金标；尚未改动线上服务入口。基础编码器与原模型保留，不声称重新训练了基础大模型。新增分支只用train+selection2876拟合；原7B模型原先还使用calibration703，这一旧训练条件没有改变。", "",
        f"相关测试{count}项通过、失败{failed}。六种实际输入状态逐一执行，缺A、缺T、仅T的16条都确实进入新分支；隐藏音频缓存污染不影响结果。最终模型序列化重放和改变/移除金标误差0。折0从头重拟合归一化/PCA、新融合专家和门控，固定内层OOF教师证据，最大差{verified['independent_fold0_retrain_max_abs_error']:.9g}。MOSI/MELD逐条概率重放最大差分别约1.19e-7/0，旧3B/7B/light主模型均与第十一轮原清单摘要一致。原821和外部EmotionTalk1929未复评。", "",
        "final_ch/verification.json里的时延仅为16条缓存表征到分类输出，排除媒体解码和骨干编码，不能作为新的端到端时延或生产SLA。原7B完整路径输出是既有五头概率均值，新增退化头是单种子42；本轮开发来源对照为逐折模型，不将这组开发数字冒称最终部署五种子平均质量。", "",
        "首个最终拟合尝试因输出目录未创建中断，记录保存在final_ch_attempt1及finish_ch.log；已修复并完整重跑，完成验收的是final_ch。", "",
        "复现命令、字段要求、文件导航见REPRODUCE.md。审阅包只包含报告、代码、关键权重及预测与审计，不重复分发公开语料原音视频或旧基础大模型。", ""]
    (OUTPUT/"第十三轮综合优化与新语料验证报告.md").write_text("\n".join(lines),"utf-8")
    save(OUTPUT/"COMPLETION-AUDIT.json",dict(completed=["public source search and2new corpora acquisition","pinned hashes and test quarantine",
        "8core +6neural controls across CH4folds and2new official splits","knowledge and shuffled control","sourceOOF fusion teacher distillation",
        "actual top2 expert dispatch","one-time MOSI686/MELD2610 evaluation","all negative outcomes preserved","accuracy constrained Chinese availability route",
        "final callable weights","forced missing branch and hidden-cache isolation checks","independent fold0 retrain","76related tests","report source manifest review bundle"],
        not_claimed=["new independent Chinese multimodal confirmation","all conditions improved","new psychological knowledge graph","new end-to-end runtime benchmark","online service replacement"]),overwrite=True)
    (OUTPUT/"REPRODUCE.md").write_text(
        "# 第十三轮复现与导航\n\n工作目录：D:/Develop/AIC/AI-mental-main (2)/AI-mental-main/AI-mental-main/algorithm。Python：D:/Develop/AIC/.venv/Scripts/python.exe。\n\n"
        "- final_ch：已完成最终中文权重、偏置、模型依赖摘要和开发全量概率。final_ch_attempt1为保留的失败记录。\n"
        "- training/ch、neural/ch：四来源OOF开发选型及全部候选；neural/ch/fold*/teacher_source_audit.json为内层教师来源边界。\n"
        "- new_test/mosi、new_test/meld：模型锁→无标签概率锁→一次评分，所有负结果及条件区间。\n"
        "- secondary_holm.json：探索性16比较校正；不替代预注册主要终点。\n"
        "- verification.json、tests.xml、tests.log：原模型保护、从头重训、实际退化分支重放及关键测试。\n"
        "- DATA-CATALOG.json：作者/镜像链接、版本、摘要、排除语料原因。source为代码快照，环境见environment.json。\n\n"
        "```powershell\n"
        "& 'D:/Develop/AIC/.venv/Scripts/python.exe' -m pytest tests/test_round11_protocol.py tests/test_knowledge_moe.py tests/test_round12_exploration.py tests/test_round13_protocol.py -q -p no:cacheprovider\n"
        "& 'D:/Develop/AIC/.venv/Scripts/python.exe' -m experiments.round13_reproduce --dataset ch --output D:/Develop/AIC/_国赛优化/round13_reproduction_ch\n"
        "& 'D:/Develop/AIC/.venv/Scripts/python.exe' -m experiments.round13_reproduce --dataset mosi --output D:/Develop/AIC/_国赛优化/round13_reproduction_mosi\n"
        "& 'D:/Develop/AIC/.venv/Scripts/python.exe' -m experiments.round13_reproduce --dataset meld --output D:/Develop/AIC/_国赛优化/round13_reproduction_meld\n"
        "```\n\n"
        "输出目录必须不存在，并位于_国赛优化下。该命令重新拟合当前语料全部固定候选；新测试重评分仅是原方法复现，不产生新的独立证据。原目录的完成标记拒绝覆盖或直接跳过。首次接入可分别执行round13_data download mosi/meld与round13_prepare；NTUSD/OpenCC固定快照位于data/public/round13_sources/ntusd。\n\n"
        "调用最终中文模型：\n\n```python\nfrom experiments.round13_runtime import CHPredictor\npredictor = CHPredictor()\nprobabilities, details = predictor.predict(round11_frozen_feature_dict)\n```\n\n"
        "输入字段与第十一轮冻结运行时一致：sample_ids、raw_text、llm_last、aux_mean、bge_cls、audio、speech、speech_emotion、sensevoice、sensevoice_query、sensevoice_emotion、speech_available、sensevoice_available、face_summary、deep_face、visual、coverage、modality_mask、quality。T/A/V不可用时必须反映真实观测，并同步移除视觉代理。模型不需要labels或心理诊断字段。\n\n"
        "依赖原第十一轮7B模型与原编码器权重，包内不复制基础大模型。新语料研究使用各自原生公开特征及额外官方train监督，不能将其测试分数写为原中文模型冻结迁移成绩。\n",
        "utf-8")
    files = [p for p in OUTPUT.rglob("*") if p.is_file() and p.name not in ("artifact_manifest.json","第十三轮审阅包.zip","第十三轮审阅包.zip.sha256")]
    save(OUTPUT/"artifact_manifest.json",dict(generated=generated,files=[dict(path=str(p.relative_to(OUTPUT)),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(files)]),overwrite=True)
    bundle = OUTPUT/"第十三轮审阅包.zip"
    with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED,compresslevel=5) as archive:
        for p in OUTPUT.iterdir():
            if p.is_file() and p.suffix in (".md",".json",".xml"):
                archive.write(p,str(p.relative_to(OUTPUT)))
        for sub in ("source","figures","final_ch","new_test","datasets"):
            for p in (OUTPUT/sub).rglob("*"):
                if p.is_file() and (sub!="datasets" or p.suffix==".json"):
                    archive.write(p,str(p.relative_to(OUTPUT)))
    bundle.with_suffix(".zip.sha256").write_text(digest(bundle)+"  "+bundle.name+"\n","utf-8")


if __name__ == "__main__":
    main()
