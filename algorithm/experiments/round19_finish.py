"""第十九轮知识图谱专项交付，保留真图/平面定义/乱图/无知识全部结果。"""
from __future__ import annotations
import csv
import hashlib
import importlib.metadata
import json
from collections import Counter
from pathlib import Path
import re
import shutil
import zipfile
import requests
from experiments.round14_common import ROOT,REPO,save,sha256
from experiments.round19_prepare import OUT,DATA


def read(relative: str) -> dict:
    """读取已固定工件，不在报告阶段调整图谱、模型或对照条件。"""
    return json.loads((OUT/relative).read_text(encoding="utf-8"))


def interval(pair: list[float]) -> str:
    """区间端点统一四位小数并保留符号，靠近零的下界另在正文给出精确值。"""
    return f"[{pair[0]:+.4f}, {pair[1]:+.4f}]"


def distribution(counter: Counter) -> str:
    """按数量降序输出构成，便于核对图谱不是由单一来源堆出来的。"""
    return "、".join(f"{key} {value}" for key, value in counter.most_common())


def finish() -> None:
    """核对冻结来源、默认待审、真实零人评和逐条回放，打包并逐项校验。"""
    result=read("benchmark/results.json");verification=read("verification.json");status=read("knowledge_graph/status.json");graph=read("knowledge_graph/graph.json");audit=read("benchmark/split_audit.json")
    for file,protocol in (("round19_prepare.py","research_protocol.json"),("round19_graph.py","knowledge_graph/build_protocol.json"),("round19_encode.py","benchmark/encoding_protocol.json"),("round19_evaluate.py","benchmark/evaluation_protocol.json")):
        assert sha256(REPO/"algorithm/experiments"/file)==read(protocol)["source_sha256"]
    assert read("benchmark/encoding_protocol.json")["graph_sha256"]==sha256(OUT/"knowledge_graph/graph.json")
    assert sha256(DATA/"ESConv.json")==read("source_lock.json")["ESCONV"]["sha256"]
    assert all(verification["protected_files_unchanged"].values())
    baseline=read("workspace_baseline.json");assert all(sha256(REPO/p)==h for p,h in baseline["protected_sha256"].items())
    assert all(r["max_error"]<1e-12 for r in verification["replays"])
    assert all(item["review_status"]=="pending" for item in graph["nodes"]+graph["edges"])
    text=(OUT/"tests.log").read_text(encoding="utf-8");counts=re.findall(r"(\d+) passed",text);assert counts and "FAILED" not in text
    actual_human=json.loads((ROOT/"round16/human/semantic_analysis.json").read_text(encoding="utf-8"));assert actual_human["completed_semantic_rows"]==0 and not actual_human["model_used_as_rater"]
    save(OUT/"human_status.json",{"project_reply_human_blind_ratings":0,"graph_human_expert_reviews":0,"model_rater_used":False,"published_ESConv_labels":"existing original public human annotations, not newly completed project ratings","pending_review_items":len(graph["nodes"])+len(graph["edges"]),"expert_identity_cannot_be_verified_by_software":True})
    save(OUT/"promotion_decisions.json",{"graph":"v0.2 engineering draft, pending expert review, default withheld","strategy_models":"offline annotation-consistency experiment only; no production or user advice","no_claim_of_emotion_fusion_or_support_effectiveness":True,"audio":"deferred","platform":"deferred","real_human_reviews":"pending"})
    save(OUT/"knowledge_graph/change_log.json",{"parent_round18_sha256":graph["parent_sha256"],"version":"0.2.0","nodes_before":28,"nodes_after":63,"edges_before":29,"edges_after":56,"added":"UNICEF context-scoped communication principles; ESConv strategy definitions and engineering recognition facets; version/fingerprint bound review records; conflict withholding","clinical_rules_modified":False})
    rows=[]
    for task,label in (("attribution","识别已有回复策略"),("next_strategy","根据过去预测下一策略")):
        for arm,m in result[task]["test"].items():rows.append(f'| {label} | {arm} | {m["n"]} | {m["macro_f1"]:.4f} | {m["accuracy"]:.4f} |')
    comparisons=[]
    for task,record in result.items():
        for reference,c in record["comparisons"].items():comparisons.append(f'| {task} | true_graph − {reference} | {interval(c["macro_f1_delta_ci95"])} | {interval(c["accuracy_delta_ci95"])} |')
    a,b=result["attribution"],result["next_strategy"]
    def delta(record: dict,reference: str) -> float:
        return record["test"]["true_graph"]["macro_f1"]-record["test"][reference]["macro_f1"]
    claim=("本轮不能证明图结构相对平面定义有稳定增益。" if a["comparisons"]["definitions_only"]["macro_f1_delta_ci95"][0]<=0 or b["comparisons"]["definitions_only"]["macro_f1_delta_ci95"][0]<=0 else "本轮在该公开来源的两个任务上观察到图结构候选的正向差值区间，仍需新来源确认。")
    names=["Question","Paraphrase","Reflection","Self-disclosure","Affirmation","Suggestions","Information","Others"]
    per_class=[f'| {name} | {a["test"]["true_graph"]["per_class_f1"][i]-a["test"]["definitions_only"]["per_class_f1"][i]:+.4f} | {b["test"]["true_graph"]["per_class_f1"][i]-b["test"]["definitions_only"]["per_class_f1"][i]:+.4f} |' for i,name in enumerate(names)]
    composition="\n".join((f'| 节点类型 | {distribution(Counter(n.get("kind") for n in graph["nodes"]))} |',f'| 关系谓词 | {distribution(Counter(e["predicate"] for e in graph["edges"]))} |',f'| 关系来源 | {distribution(Counter(e["source_id"] for e in graph["edges"]))} |'))
    removed=audit["duplicate_inputs_removed"];duplicate_note=f'识别任务验证集 {removed["attribution"]["validation"]} 条、测试集 {removed["attribution"]["test"]} 条；预测任务验证集 {removed["next_strategy"]["validation"]} 条、测试集 {removed["next_strategy"]["test"]} 条'
    tiny=f'{b["comparisons"]["no_graph"]["macro_f1_delta_ci95"][0]:+.2e}'
    report=f'''# 第十九轮知识图谱专项报告（2026-11-17）

## 本轮完成内容

图谱由28节点/29边升级到63节点/56边，保留全部出处和待审状态。补入两篇UNICEF亲子沟通资料及ACL2021的八类支持策略定义；增加来源场景隔离、关系内容指纹、版本父链、绑定具体内容的人工审核导入、同范围正反冲突隔离及完整出处证据包。工程连接、定义拆分与文献原有概念明确区分。

真人盲评仍0份、专业图谱审核仍0条。默认不返回待审图谱；研究使用须同时显式开启research和include_unreviewed，也只返回概念与出处。未改动平台、音频、原安全阈值或升级规则。原KnowledgeIndex与旧种子文件保持不动，新增内容均在离线研究模块。

## 结果先行

- 公开ESConv标注对照已跑完1300段对话，约70/15/15按整段对话哈希隔离，实际{len(audit['partitions']['train'])}/{len(audit['partitions']['validation'])}/{len(audit['partitions']['test'])}段。后组中与先前组完全相同的规范化输入按预先协议剔除，数量完整保留。
- 已有回复策略识别：真图Macro-F1={a['test']['true_graph']['macro_f1']:.4f}，相对无知识差{delta(a,'no_graph'):+.4f}、相对平面定义差{delta(a,'definitions_only'):+.4f}。
- 根据过去对话预测下一策略：真图Macro-F1={b['test']['true_graph']['macro_f1']:.4f}，相对无知识差{delta(b,'no_graph'):+.4f}、相对平面定义差{delta(b,'definitions_only'):+.4f}。
- {claim} 不能把识别既有回复的准确率说成“知道下一句该如何支持”，也不能把这两个指标说成项目回复效果、情绪融合或青少年实际收益。
- 相关测试{counts[-1]}项通过；JSON-LD经rdflib解析保留全部56条关系；两任务四臂模型逐条回放误差<1e-12。已有十八轮图谱与本轮前记录的保护文件摘要不变。

## 图谱的来源和范围

| 来源 | 本轮用途 | 适用范围与限制 |
|---|---|---|
| [WHO压力管理指南](https://cdn.who.int/media/docs/default-source/thailand/doing-what-matters-in-times-of-stress-eng.pdf) | 继承五类压力管理概念及原出处 | 概念描述；不作为具体用户的原因、诊断或疗效证据 |
| [UNICEF沟通建议](https://www.unicef.org/parenting/mental-health/how-to-talk-to-kids-mental-health) | 倾听、询问回应偏好、沟通时机 | 来源为亲子场景，不能直接声称AI对青少年有效 |
| [UNICEF 11–13岁沟通资料](https://www.unicef.org/parenting/mental-health/conversation-starters-11-13-years) | 承认感受、共同探索、避免淡化和比较 | 年龄与亲子场景保存在每条关系范围中 |
| [ACL2021原论文](https://aclanthology.org/2021.acl-long.269/) | 八类支持策略及定义，工程拆分16个识别facet | 英语众包支持体系；不是临床协议，facet不是作者发布的正式本体 |

全图有63节点、56边。相比十八轮新增的27条边包括8条沟通原则、16条策略定义的工程拆分和3条阶段概念的工程组织连接；后两类不冒称原文提供了这一形式的知识图谱。无原始用户数据、个体心理属性或自动生成诊断写入图谱。

每条边包含source_id、原文URL、页码/段落定位、原文文件SHA256、来源范围、提取/工程推导类型、内容指纹和审核状态。UNICEF缓存检查了定位关键词的存在，PDF记录指纹并依据已查阅的原文附录定位；这些是出处检查，不等于专家核实中文释义或适用性。来源为相同机构的两篇资料，也不当成两项独立疗效研究。

图谱JSON、JSON-LD、节点/关系CSV、版本变更记录、{len(graph['nodes'])+len(graph['edges'])}项空审核表均交付。JSON-LD以RDF Statement保留关系和出处；独立解析器读到{read('knowledge_graph/rdf_validation.json')['rdf_triples']}个三元组及56条重述关系。

## 审核、冲突与检索

内容指纹覆盖关系、定义、来源和范围，审核人、日期与决定另记。人工导入必须提供当前内容SHA、human审核方法、审核人、ISO时间和资格依据，拒绝模型代审、空身份和过期内容。软件只能核验记录格式和内容绑定，不能证明现实身份或资质；实际专业审核必须另外完成。本轮没有填写任何真实审核。

同一主体/关系/客体在同一来源范围出现正反断言时，相关边从检索中隔离，不用来源数量或多数票判断医学真假。当前真实资料没有检出的正反冲突；冲突功能只通过明确标记的合成测试验证，不编造已解决的真实文献争议。

检索保留显式目标、引用、近邻否定、最新同概念否定与场景限制；用户暂停则不继续检索，亲子条目不在默认一般语境返回。当前检索为中文/英文别名加字符TFIDF，并不宣称完成了复杂跨句否定、讽刺、所有指代或中文语义理解。所有返回条目均标记不属于用户事实、不代表个人建议。证据预算按完整概念块截取，引用不会被截断。

## 外部人工标签评测

[ESConv作者仓库](https://github.com/thu-coai/Emotional-Support-Conversation)的固定提交见source_lock.json；原始语料SHA256={read('source_lock.json')['ESCONV']['sha256']}。[作者数据卡](https://huggingface.co/datasets/thu-coai/esconv)标注CC BY-NC4.0，仓库要求学术研究用途。使用既有人工策略标签，不生成或修改任何标签。数据主要为英语众包对话，不能推广成中国青少年项目的真人盲评。

任务一输入当前已有支持回复，识别其原标注策略。任务二只输入目标支持回复之前最近六句、末1800字符，并使用左侧截断最多256token；不使用目标回复、后续对话、整段总长度、前后问卷或情绪改善结果。原标签代表当时采用的策略，不是唯一正确或效果最好的策略，尤其不能把预测不一致一律视为不恰当支持。

按完整对话文本哈希分组，不使用标签选择分组；这是本项目新划分，不冒称论文官方6/2/2划分。缺少足够身份信息，不能称参与者身份互斥。重复输入剔除：{duplicate_note}。没有根据新测试得分换来源、换标签、改图或只保留容易的策略。

四臂使用同一冻结MiniLM384维表征、只在训练集拟合的词TFIDF、相同LinearSVC与相同C=[.1,1.]验证搜索预算。八个附加通道在无知识时为零；平面定义使用策略定义相似度；真图使用0.75定义相似度+0.25相邻facet最大相似度；乱图只置换facet链接，保持节点文本、边数与每类度数。所有图特征不读取标签。两个任务所有臂都完成验证和锁型后，才计算测试性能。

这里用于分类的图关系是ACL策略节点到16个识别facet，**没有把WHO或UNICEF概念的效果混算进分类收益**。冻结MiniLM属于通用预训练编码器，历史预训练数据与公开语料是否重叠未知。首次GPU编码出现驱动错误，改CPU继续，原失败日志保留，没有重置其他进程或改变实验参数。

| 任务 | 条件 | 测试句数 | Macro-F1 | Accuracy |
|---|---|---:|---:|---:|
{chr(10).join(rows)}

下表为3000次按对话配对bootstrap的探索区间，不是预注册临床显著性或多来源泛化结论。比较家族未作确认性多重比较控制，不能择一条正区间夸大成全面稳定收益。

| 任务 | 比较 | Macro-F1差值95%区间 | Accuracy差值95%区间 |
|---|---|---|---|
{chr(10).join(comparisons)}

区间端点保留四位小数；预测任务相对无知识的 Macro-F1 下界为 {tiny}，只是略高于 0，不能读成稳定收益。

## 分策略观察

| 策略类 | 识别已有回复 ΔF1（真图−平面定义） | 预测下一策略 ΔF1（真图−平面定义） |
|---|---|---:|
{chr(10).join(per_class)}

八类差值都在 ±0.01 以内，没有“某一类被图结构稳定救回”的现象；识别任务里 Others 与 Suggestions 略降，方向与邻接 facet 带来的额外相似度把边界样本推向相邻类一致，但幅度小于噪声，不宜逐类解读。

## 图谱构成

| 维度 | 分布 |
|---|---|
{composition}

63 节点中 27 个来自 ACL2021（八类策略、16 个工程识别 facet、3 个阶段概念），18 条关系来自 WHO，8 条来自两篇 UNICEF，11 条是项目内工程组织连接；亲子来源单独标记为 parent_child，默认一般语境不返回。

## 判决与下一步

- **结论：** 图结构（facet 邻接）在两个公开任务上都没有超过平面定义；识别任务相对平面定义的区间整体为负，预测任务相对无知识的下界仅 {tiny}、相对平面定义跨零。因此不把图谱计入分类性能叙事，也不以节点数代替效果。
- **保留理由：** 图谱当前的价值是“可溯源的概念检索层”——来源场景隔离、审核门必须显式开启、同范围冲突隔离、每条边带 URL／定位／文件指纹；这是可解释与合规价值，不是准确率。
- **若继续：** 先补中文独立材料的概念检索标注与专家审核，再决定图结构是否值得保留；不在同一测试集上调权重、换来源或只保留容易的类别。

## 可复现交付与仍缺的证据

source包含本轮源码及运行依赖快照，benchmark包含划分、特征、模型、锁型与逐句预测；原始ESConv和版权说明在data。UNICEF完整HTML、论文PDF和大型编码器不重复打包，原文件指纹与本地缓存位置保留。raw语料和模型均为本地研究用途，不进行平台安装、发布或接线。

补齐流程时，round19_evaluate 的冻结协议写入出现一次“元组与 JSON 列表”的比较差异：重写前后 evaluation_protocol.json 的 SHA256 完全一致，臂、C、选型规则与被测脚本摘要均未改变，只完成了一次内容相同的重写；本次没有在读取测试标签后修改任何实验定义。

仍需专业人员核查来源释义与适用边界；独立中文概念检索标注、对中文回复事实支撑与具体回应的真人盲评、真实图文情绪融合以及前瞻支持效果都尚未完成。现有策略标签不能替代这些证据。下一步应根据无知识/平面定义/真图结果判断知识结构是否值得保留，优先减少无关检索、扩大中文语义覆盖，再用独立材料验证；不以增加节点数代替效果提升。
'''
    (OUT/"第十九轮知识图谱专项报告-20261117.md").write_text(report,encoding="utf-8")
    (OUT/"使用说明.md").write_text('''# 第十九轮离线知识图谱

设置PYTHONPATH=algorithm，用D:/Develop/AIC/.venv/Scripts/python.exe执行。

1. round19_prepare固定公开来源、研究协议和MiniLM编码器。
2. round19_graph生成独立v0.2图谱和空专家审核表。
3. round19_encode生成严格划分和不依赖标签的冻结表征；GPU不可用可设置CUDA_VISIBLE_DEVICES=-1运行CPU。
4. round19_evaluate在验证集选型，两个任务全部锁型后评测测试集。
5. round19_verify按缓存回放生成verification.json；再用UTF-8写入本轮pytest结果到tests.log，最后运行round19_finish归档。

已有冻结工件不得覆盖以试不同参数，应新开轮次/目录。工作路径默认D:/Develop/AIC/_国赛优化/round19。

检索接口：round19_runtime.evidence_packet(query,goal)。默认不返回待审条目；离线研究须显式research=True, include_unreviewed=True；亲子来源另需setting="parent_child"。返回出处概念，不能直接当用户事实或个人支持建议。

人工审核接口：round19_graph.apply_reviews，先读expert_review_pending.csv及来源，只有实际人工审核完成后才提供绑定当前内容的记录。当前全部待审，不得填造审核人或模型评分。

策略识别接口：round19_runtime.predict_strategy，要求同一冻结编码器和knowledge_channels生成的特征。分数只表示与八类公开策略标注的对应程度。平台、音频和真人采集均未在本轮执行。
''',encoding="utf-8")
    notice='''# Sources and attribution

Siyang Liu, Chujie Zheng, Orianna Demasi, Sahand Sabour, Yu Li, Zhou Yu, Yong Jiang, Minlie Huang. 2021. Towards Emotional Support Dialog Systems. ACL-IJCNLP2021,3469–3483. https://doi.org/10.18653/v1/2021.acl-long.269 . ESConv author repository https://github.com/thu-coai/Emotional-Support-Conversation and author dataset card https://huggingface.co/datasets/thu-coai/esconv . Dataset cardCC BY-NC4.0 https://creativecommons.org/licenses/by-nc/4.0/ ; original README: Copyright2021 CoAI Group, Tsinghua University, all rights reserved, data and codes for academic research use only. Original ESConv JSON and README preserved; project splits/embeddings are derived research artifacts. No clinical or project-human effectiveness is inferred from public strategy labels.

UNICEF Parenting, How to talk to your kids about mental health, https://www.unicef.org/parenting/mental-health/how-to-talk-to-kids-mental-health ; Mental health conversation starters:11–13 years, https://www.unicef.org/parenting/mental-health/conversation-starters-11-13-years . Copyright UNICEF. Only brief engineering paraphrases, concept links, scope and source fingerprints distributed here; full pages not redistributed. Parent-child guidance is not evidence of AI support efficacy.

World Health Organization.2020. Doing what matters in times of stress:an illustrated guide. Geneva:WHO. https://cdn.who.int/media/docs/default-source/thailand/doing-what-matters-in-times-of-stress-eng.pdf . Inherited guideline-derived concept adaptationCC BY-NC-SA3.0IGO https://creativecommons.org/licenses/by-nc-sa/3.0/igo/ . Chinese terms are engineering adaptations, not an official WHO translation; the English original is authoritative. WHO does not endorse this project. All graph expert review is pending. Graph adaptations remain subject to applicable source terms; no broader permission is conferred for underlying pages or paper.

W3C SKOS https://www.w3.org/TR/skos-reference/ and RDF reification used for representation, not clinical validation. Sentence Transformers all-MiniLM-L6-v2 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 , Apache2.0, pinned revision and weight hashes in encoder_lock.json. Frozen encoder used for features, never as a human rater. Historical pretraining overlap unknown; weights not redistributed in this delivery.
'''
    (OUT/"SOURCE_NOTICE.md").write_text(notice,encoding="utf-8")
    (OUT/"data").mkdir(exist_ok=True)
    for name in ("ESConv.json","README.md"):shutil.copy2(DATA/name,OUT/"data"/name)
    with (OUT/"results_summary.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["task","arm","n","macro_f1","accuracy"]);writer.writeheader()
        writer.writerows({"task":task,"arm":arm,**{k:m[k] for k in ("n","macro_f1","accuracy")}} for task,r in result.items() for arm,m in r["test"].items())
    shutil.copytree(ROOT/"round18/source",OUT/"source",dirs_exist_ok=True)
    for path in (REPO/"algorithm/experiments").glob("round19_*.py"):shutil.copy2(path,OUT/"source/algorithm/experiments"/path.name)
    shutil.copy2(REPO/"algorithm/tests/test_round19_knowledge.py",OUT/"source/algorithm/tests/test_round19_knowledge.py")
    save(OUT/"environment_versions.json",{n:importlib.metadata.version(n) for n in ("numpy","scipy","scikit-learn","torch","transformers","rdflib","requests")})
    files=sorted(p for p in OUT.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p!=OUT/"manifest.json")
    manifest={str(p.relative_to(OUT)).replace("\\","/"):{"sha256":sha256(p),"bytes":p.stat().st_size} for p in files};save(OUT/"manifest.json",manifest)
    destination=ROOT/"第十九轮知识图谱交付包-20261117.zip"
    with zipfile.ZipFile(destination,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files+[OUT/"manifest.json"]:archive.write(path,str(path.relative_to(OUT)).replace("\\","/"))
    with zipfile.ZipFile(destination) as archive:
        for name,item in manifest.items():assert hashlib.sha256(archive.read(name)).hexdigest()==item["sha256"]
    print(json.dumps({"report":str(OUT/"第十九轮知识图谱专项报告-20261117.md"),"zip":str(destination),"tests":int(counts[-1]),"files":len(files),"bytes":destination.stat().st_size},ensure_ascii=False),flush=True)


if __name__=="__main__":finish()
