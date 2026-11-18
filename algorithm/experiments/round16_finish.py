"""第十六轮证据核验、来源说明、报告、源码与离线模型交付。"""
from __future__ import annotations
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import re
import shutil
import zipfile
import numpy as np
from experiments.round14_common import ROOT,REPO,save,sha256
from experiments.round16_external import OUT,DATA,SOURCES
from experiments.round16_human import analyze


def read(relative: str) -> dict:
    """读取本轮已生成的真实证据。"""
    return json.loads((OUT/relative).read_text(encoding="utf-8"))


def source_snapshot() -> int:
    """保存本轮代码及静态本地导入依赖，跳过环境密钥和大型历史缓存。"""
    base=REPO/"algorithm";pending=list((base/"experiments").glob("round16_*.py"))+[base/"tests/test_round16_protocol.py"];seen=set()
    prefixes={"experiments","assessment","intervention","perception","shared","data_quality"}
    while pending:
        path=pending.pop()
        if path in seen or not path.is_file():continue
        seen.add(path);text=path.read_text(encoding="utf-8-sig")
        destination=OUT/"source/algorithm"/path.relative_to(base);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
        for node in ast.walk(ast.parse(text)):
            modules=[]
            if isinstance(node,ast.Import):modules=[item.name for item in node.names]
            if isinstance(node,ast.ImportFrom):
                module=node.module or ""
                if node.level:
                    parts=list(path.relative_to(base).parent.parts)
                    if node.level>1:parts=parts[:-(node.level-1)]
                    module=".".join(parts+[module])
                modules=[module]+[module+"."+item.name for item in node.names]
            for module in modules:
                if module.split(".")[0] not in prefixes:continue
                rel=Path(*module.split("."));candidates=[base/rel.with_suffix(".py"),base/rel/"__init__.py"]
                pending.extend(p for p in candidates if p.is_file())
    for path in (base/"config").glob("*.yaml"):
        destination=OUT/"source/algorithm/config"/path.name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
    return len(seen)


def finish() -> None:
    """检查冻结方法、真实空人评和导出模型，再生成可审阅证据包。"""
    lock=read("forecast/source_selection_lock.json");protocol=read("forecast/external_protocol.json")
    assert lock["code_sha256"]==sha256(REPO/"algorithm/experiments/round16_external.py")
    assert protocol["methods"]["source_selected"]==lock["winner"] and protocol["methods"]["robust"]==lock["robust_candidate"]
    assert not protocol["fresh_source_training"]
    external={};count=0
    for name,spec in SOURCES.items():
        path=DATA/spec["filename"];assert hashlib.md5(path.read_bytes()).hexdigest()==spec["md5"]
        result=read(f"forecast/{name}/results.json");external[name]=result
        assert result["audit"]["source_sha256"]==sha256(path)
        rows=json.loads((OUT/f"forecast/{name}/windows.json").read_text(encoding="utf-8"))
        for row in rows:
            assert row["target_day"]==row["origin"]+1 and max(row["history_days"])==row["origin"] and len(row["history"])>=3
            assert max(row["history_days"])<row["target_day"]
        count+=len(rows)
        for m in result["metrics"].values():assert m["coverage"]==1.
        shutil.copy2(path,OUT/"data"/path.name)
    human=analyze();assert human["completed_semantic_rows"]==0 and human["model_used_as_rater"] is False
    old=json.loads((ROOT/"round14/scope_and_promotion.json").read_text(encoding="utf-8"))["protected_current_sha256"]
    assert all(sha256(REPO/name)==digest for name,digest in old.items())
    runtime=read("runtime_verification.json");assert all(row["maximum_error"]<1e-12 for row in runtime.values())
    tv=read("moe/export_verification.json");assert len(tv)==48 and max(row["max_error"] for row in tv.values())<1e-12
    for fold in range(4):
        folder=OUT/f"moe/export/fold{fold}"
        with (folder/"projection.pkl").open("rb") as stream:p=pickle.load(stream)
        with (folder/"experts.pkl").open("rb") as stream:b=pickle.load(stream)
        assert p.members["A"] is None and set(b["models"])=={1,4,5}
    tests=(OUT/"tests.log").read_text(encoding="utf-8");matches=re.findall(r"(\d+) passed",tests);assert matches and "FAILED" not in tests
    verification={"new_sources":2,"external_windows":count,"forecast_replay_comparisons":2*count,"forecast_max_error":max(r["maximum_error"] for r in runtime.values()),
        "tv_export_comparisons":48,"tv_max_error":max(row["max_error"] for row in tv.values()),"audio_removed_from_all_exported_projections_and_heads":True,
        "tests_passed":int(matches[-1]),"human_ratings_completed":0,"semantic_metrics_imputed":False,"protected_existing_files_unchanged":True,
        "new_sources_used_for_tuning":False,"all_verified":True}
    save(OUT/"verification.json",verification)
    rows=[]
    for name,result in external.items():
        for method in ("last","mean7","previous_ema","robust"):
            m=result["metrics"][method];rows.append(f'| {name} | {method} | {m["people"]}/{m["windows"]} | {m["person_mae_0_100"]:.3f} | {m["rmse_0_100"]:.3f} | {m["q90_abs_error"]:.3f} | {m["worst_quarter_person_mae"]:.3f} |')
    inference=read("forecast/inference.json");moe=read("moe/results.json");complement=read("moe/complementarity.json")
    selected_holm=[r for r in inference["person_mae_comparisons"] if (r["dataset"],r["candidate"],r["baseline"]) in
        {("sicorello","robust","mean7"),("sicorello","robust","previous_ema"),("dejonckheere","source_selected","last")}]
    report=f'''# 第十六轮优化报告：首次外部验证、真实语义评审及文本视觉专家（2026-11-17）

本轮按优先级完成可自动执行部分。音频、专业知识图谱和平台接入继续暂缓；真人评分不由模型代替，当前独立评分仍为0。新代码均为离线研究模块，原安全阈值、审计、升级及默认模型文件指纹保持不变。

## 1. 优先补强独立语义评价

基于第十五轮同一40个场景的F/G共80条回复，交付两份独立空评分表，共160行待填写。新增“行动是否对应当前关切、是否复问已知信息、是否编造解释”三项及用户/回复逐字出处。评分导入要求独立真人声明、原六项完整评分及出处与真实文本匹配；NA不计为0，缺失不补分。分歧清单保留空裁决位，不把裁决替换原始一致性结果。语义按臂统计只在真实填写之后计算，当前均为null。

机器排查列表在human/private，与盲评隔离，不能只评价被标记条目。未使用LLM或其他模型产生人类评分；本轮也未把自动关切词命中率包装成理解质量或策略疗效。没有招募或发送消息，实际独立评分这一步仍待真实评审实施。

## 2. 新用户预测首次在两个未参与本项目选型的来源验证

来源A：[Sicorello开放来源说明](https://openesmdata.org/datasets/0073_sicorello/)，[固定版本22809936](https://zenodo.org/records/22809936)，openESM整理的情绪调节日常自评，许可CC BY-NC4.0。本轮只使用时间序列表，不下载静态个人表。目标为overwhelming_emotions，原题1–5；按发布date聚合实际日期。合格每日数据222个ID，满足过去三日与下一日标签条件的195个ID、1194个预测窗口。

来源B：[Dejonckheere开放来源说明](https://openesmdata.org/datasets/0012_dejonckheere/)，[固定版本17347570](https://zenodo.org/records/17347570)，许可CC BY4.0，目标sad为0–100悲伤自评；100个ID、1079个窗口。发布的是研究日索引，没有原始日历时间戳，因此只能称下一研究日，不能捏造真实日期或周末信息。

两个来源此前未参与本项目训练、调参或选型；此次查阅来源/字段定义后，先在旧StudentLife验证组锁定方法，再下载并评测全部新来源。全过程未用外部结局重选方法、调整门槛或筛掉结果较弱的来源。source_selected为EMA(alpha0.3)，与previous_ema完全同一预测器；robust参数由旧来源锁定为eta10、mix1。两者同时预先列入外部协议，不在看到结果后按来源二选一。

所有目标仅以量表理论上下界线性归一化为0–100展示误差，预测任务分别是情绪压倒感和悲伤。两者不是PAM效价、项目心情分或临床结局，不合并为一个准确率，也不宣称已验证中国青少年或零历史用户。新用户在预测时至少有3个过去的日均自评，这是有少量历史的新用户泛化。

每日至少2份有效响应，冲突同人同日同beep记录全组隔离；单次响应日剔除数量保留在字段审计；未响应不补零。历史最多14日，超过3日空档重新暖启动；窗口覆盖率100%指合格窗口的覆盖率，不是全部原始参与者/冷启动日覆盖率。

| 来源 | 方法 | ID/窗口 | ID等权MAE | RMSE | 绝对误差P90 | 最差四分之一用户MAE |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

首次外部结果：Sicorello稳健候选的ID等权MAE相对保持法约改善6.3%，相对mean7改善3.7%，相对旧EMA改善2.1%；误差P90由25.000降到20.644，约改善17.4%。最差四分之一用户MAE由19.189降到17.528，改善约8.7%。Dejonckheere原先锁定的EMA相对保持法RMSE由11.314降到9.677，改善约14.5%，其ID等权MAE由7.568降到6.802，改善约10.1%。

代价同样保留：稳健候选在Sicorello的RMSE从旧EMA的12.996变为13.096；Dejonckheere从9.677变为9.856，且P90也高于旧EMA。因此稳健候选改善绝对误差/尾部负担，但尚不支持取代EMA成为所有来源的统一默认方法。

按用户聚类的MAE区间详见各来源results.json；12个MAE比较同时做Holm校正，部分关键比较为：
'''
    for r in selected_holm:report+=f'- {r["dataset"]}/{r["candidate"]}对{r["baseline"]}：MAE差{r["person_mae_delta"]:.4f}，Holm p={r["holm_p_12_comparisons"]:.6f}。\n'
    report+='''
Sicorello稳健候选对mean7的校正p值接近0.05边界，对旧EMA的未校正区间虽支持改善，多重比较校正后不再达到0.05，不能称所有比较均已稳定胜出。统计比较家族是在结果后明确的探索性分析，符号翻转检验也有交换性假设；不当作临床确认性研究。数据固定版本、发布MD5、实际SHA256、字段来源和全部正负结果均归档。

## 3. 高误差用户的保守个人适应

新增稳健候选只使用个人已到达的历史，在历史内部按先预测、后见结局的顺序计算保持/短均值/中位数/EMA四个专家的误差，用有限样本保守加权。历史不足6个观测日时用冻结EMA；不足3日或长缺测后历史不足则拒绝预测，不用群体均值压向“正常”。没有根据实际测试误差挑出高误差用户再只给他们换模型。

导出JSON协议中的两套固定配置及round16_runtime.predict_observations，接口检查同一匿名ID、递增日索引、理论量表、未来/目标日混入和缺测重启；未经验证的项目心情/压力量表拒绝自动迁移。两来源2273个窗口×两个候选共4546次逐项回放一致，最大数值误差低于1e-12。

## 4. 文本/视觉MoE实际收益检查

本部分是此前已曝光CH开发来源的诊断，不是新的独立测试证据，不打开历史封存测试集。复用旧来源隔离的T/V/TV专家和内部OOF概率，只执行模式1、4、5；输入中的既有音频缓存立即丢弃，未新增音频提取、训练或音频专家推理。门控只见T/V概率与可用性，不见标签或来源ID。门控训练用内部来源OOF专家结果，外层107个来源组互斥。

| TV条件方法 | Macro-F1 | Accuracy |
|---|---:|---:|
'''
    for name,m in moe["metrics"]["TV"].items():report+=f'| {name} | {m["macro_f1"]:.6f} | {m["accuracy"]:.6f} |\n'
    report+=f'''
简单专家均值相对普通TV头的Macro-F1提高约0.71个百分点；学习门控约提高0.49个百分点，未超过简单均值。学习门控纠正{complement['conditions']['TV']['learned']['rescued_errors']}个普通头错误，同时引入{complement['conditions']['TV']['learned']['newly_introduced_errors']}个新错误；简单均值则为44个纠正、29个新错。更多门控收益被新错误抵消，且宏F1与总体准确率目标不完全一致。学习门控F1差来源聚类区间跨零，不推荐仅凭结构复杂度晋升。

T-only/V-only时多个混合方法相同，因为只剩一个合格专家；各自Macro-F1虽小幅增加，Accuracy下降，全部保留。概率门控需要先执行专家，最多3个分类头对普通头1个，不能称稀疏省算力或端到端加速。最值得保留的是简单均值离线参考及门控错误分析，而非强行宣布复杂MoE胜出。

已导出四折仅T/V模型：删除A投影和所有音频专家，提供round16_tv_runtime.predict_tv与16条冻结表征烟测输入。48组“4折×3条件×4方法”回放一致。这里处理匹配冻结编码器表征，不声称已接入平台、原始视频或新编码器；重训还需此前本地冻结缓存。

## 验证、交付与状态

{verification['tests_passed']}项相关测试通过，1项既有Starlette弃用提示。保护文件摘要与第十四轮一致，未改变线上风险、审计、升级或默认路径。真人独立语义评分为0，所有对应质量均值保持null，未招募和发送消息。

范围结论：首次新来源验证完成；稳健历史适应及可调用运行时完成；T/V专家对照、裁剪导出与得失分析完成；真实人评所需材料、出处校验和分歧分析工具完成，实际评分待真人执行。新来源证明的是所述预测方法在所述自评任务上的外部表现，不把这一结果计作对话支持疗效。
'''
    (OUT/"第十六轮优化报告-20261117.md").write_text(report,encoding="utf-8")
    usage=r'''# 使用与复算

工作目录：D:\Develop\AIC\AI-mental-main (2)\AI-mental-main\AI-mental-main
解释器：D:\Develop\AIC\.venv\Scripts\python.exe
PowerShell先设置 $env:PYTHONPATH='algorithm' 和 $env:PYTHONIOENCODING='utf-8'。

只读预测回放：python -m experiments.round16_runtime
单人输入预测：python -m experiments.round16_runtime --history 路径.json
输入含dataset、source_schema、target_day、observations及可选method=source_selected或robust。
source_schema见源码固定来源量表声明；observations每行含person、day、原量表value。不接受未来/重复日期和未经验证量表。

真人评分分析：python -m experiments.round16_human
只向两名独立评审提供human/blind目录；不得提供human/private。不能用模型代填。
真实完成后添加本人rater_declarations.json，再导入两份完整CSV；分析输出含原始一致性、语义分臂描述和待真人裁决的分歧表。
空表会报告待采集，不会得到0分或满分。

文本视觉运行时：导入experiments.round16_tv_runtime.predict_tv(T,V,mask,folder,method='uniform')。
folder示例：D:\Develop\AIC\_国赛优化\round16\moe\export\fold0。
T/V为匹配既有冻结编码器的特征，mask为N×2布尔值，无音频/标签/来源ID入口。
moe/export/smoke_input.npz附16条已冻结输入及expected_uniform，可独立检查推理；完整重训需此前round13/round11缓存。

新来源处理和锁型：experiments.round16_external；所有新来源结果已保存，默认再次执行跳过已有结果。
统计：experiments.round16_statistics；模型导出回放：experiments.round16_tv_runtime；打包：experiments.round16_finish。
重新计算应在独立输出目录副本进行，保留原协议、原型、方法和全部负结果，不根据外部测试再挑模型。
基础模型表征和全部旧训练缓存未塞入ZIP，代码依赖快照、匹配表征烟测和新来源TSV已提供。
'''
    (OUT/"使用说明.md").write_text(usage,encoding="utf-8")
    notice='''# Source attribution and transformations

Sicorello, Heyse, Pruessner, Elsaesser, Kolar: Emotion regulation difficulties in daily life, https://doi.org/10.31234/osf.io/k7mgv_v1. Original repository https://github.com/MaurizioSicorello/SDERSvalid_Analysis. Harmonized by openESM, record https://zenodo.org/records/22809936, CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/).

Dejonckheere, Kalokerinos, Bastian, Kuppens (2019): Poor emotion regulation ability mediates the link between depressive symptoms and affective bipolarity, https://doi.org/10.1080/02699931.2018.1524747. Original https://figshare.com/articles/dataset/7150664. Harmonized by openESM, https://zenodo.org/records/17347570, CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

The downloaded harmonized TSV files are copied unchanged with publisher MD5 and SHA256. Derived daily tables select declared items, quarantine conflicting beeps, exclude days with fewer than two observed responses, average observed responses and linearly normalize using theoretical bounds. No missing response is replaced with zero. The two constructs and time semantics remain distinct. This package is research evidence, not a grant of additional commercial rights.
'''
    (OUT/"data/SOURCE_NOTICE.md").write_text(notice,encoding="utf-8")
    copied=source_snapshot()
    save(OUT/"environment.json",{"packages":{name:importlib.metadata.version(name) for name in ("numpy","pandas","scikit-learn","scipy","pytest","requests","catboost")},"source_python_files":copied})
    files=sorted(p for p in OUT.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name!="manifest.json")
    manifest={str(p.relative_to(OUT)).replace("\\","/"):{"sha256":sha256(p),"bytes":p.stat().st_size} for p in files};save(OUT/"manifest.json",manifest)
    destination=ROOT/"第十六轮优化交付包-20261117.zip"
    with zipfile.ZipFile(destination,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files+[OUT/"manifest.json"]:archive.write(path,str(path.relative_to(OUT)))
    with zipfile.ZipFile(destination) as archive:
        for name,item in manifest.items():assert hashlib.sha256(archive.read(name)).hexdigest()==item["sha256"]
    print(json.dumps({"report":str(OUT/"第十六轮优化报告-20261117.md"),"zip":str(destination),"files":len(files),"bytes":destination.stat().st_size,"verification":verification},ensure_ascii=False),flush=True)


if __name__=="__main__":finish()
