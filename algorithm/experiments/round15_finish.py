"""第十五轮证据复核、统计、报告和自包含源码交付。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile
import numpy as np
from experiments.round14_common import ROOT,REPO,save,sha256
from experiments.round14_human import analyze_ratings
from experiments.round15_forecast import OUT,forecast,choose,configs


def read(path: str) -> dict:
    """读取本轮已完成的证据。"""
    return json.loads((OUT/path).read_text(encoding="utf-8"))


def finish() -> None:
    """核对逐用户排除和运行时预测，统计聚类区间，保留所有正负结果。"""
    results=read("forecast/results.json"); repairs=read("support_repair/results.json"); support=read("support/results.json")
    locks=read("forecast/lopo_selection_locks.json"); protocol=read("forecast/protocol.json")
    rows=json.loads((OUT/"forecast/lopo_windows.json").read_text(encoding="utf-8"))
    arrays=np.load(OUT/"forecast/lopo_predictions.npz",allow_pickle=False)
    replay=np.array([forecast(row["history"],locks[row["person"]]["winner"]) for row in rows])
    assert np.max(np.abs(replay-arrays["selected"]))<1e-12
    for person,lock in locks.items():
        assert person not in lock["validation_people"] and lock["validation_max_target"]<protocol["strict_split"]["test_start_day"]
    groups=arrays["people"];truth=arrays["truth"];people=np.unique(groups)
    count=np.array([(groups==p).sum() for p in people]);ix=np.random.default_rng(42).integers(0,len(people),(10000,len(people)))
    names=["selected","last","mean7","old_ema_0.1"]
    sums={name:np.array([np.square(arrays[name][groups==p]-truth[groups==p]).sum() for p in people]) for name in names}
    samples={name:3*np.sqrt(sums[name][ix].sum(1)/count[ix].sum(1)) for name in names}
    intervals={name:np.quantile(samples["selected"]-samples[name],[.025,.975]).tolist() for name in names[1:]}
    generated=[json.loads(x) for x in (OUT/"support/generations.jsonl").read_text(encoding="utf-8").splitlines()]
    paired=[r for r in generated if r["split"]=="new_authored_challenge"]
    from experiments.round15_repair import extra_checks
    from experiments.round15_support import new_cases
    case_map={c["case_id"]:c for c in new_cases()}
    repaired=json.loads((OUT/"support_repair/generations.json").read_text(encoding="utf-8"))
    extended_joint={}
    for name,items in (("F",[r for r in paired if r["arm"]=="F"]),("G",repaired)):
        active=[r for r in items if not r["case_id"].endswith("pause")]
        extended_joint[name]={"n":len(active),"passed":sum(r["checks"]["corrected_topic_mentioned_proxy"] and not r["checks"]["goal_violation_proxy"] and not any(extra_checks(case_map[r["case_id"]],r["final_reply"]).values()) for r in active)}
    topic_deltas=[]
    for family in sorted({r["case_id"].split("_")[0] for r in paired}):
        pair={arm:[r for r in paired if r["arm"]==arm and r["case_id"].startswith(family+"_") and not r["case_id"].endswith("pause")] for arm in ("E","F")}
        topic_deltas.append(np.mean([r["checks"]["corrected_topic_mentioned_proxy"] for r in pair["F"]])-np.mean([r["checks"]["corrected_topic_mentioned_proxy"] for r in pair["E"]]))
    values=np.array(topic_deltas);ix2=np.random.default_rng(42).integers(0,len(values),(10000,len(values)))
    protect=json.loads((ROOT/"round14/scope_and_promotion.json").read_text(encoding="utf-8"))["protected_current_sha256"]
    assert all(sha256(REPO/name)==digest for name,digest in protect.items())
    tests=(OUT/"tests.log").read_text(encoding="utf-8"); match=re.search(r"(\d+) passed",tests);assert match and "FAILED" not in tests
    verification={"lopo_users_excluded":49,"evaluated_users":len(people),"runtime_windows":len(rows),"max_replay_error":float(np.max(np.abs(replay-arrays["selected"]))),
        "paired_challenge_replies":len(paired),"rmse_cluster_ci95":intervals,"topic_family_cluster_delta_ci95":np.quantile(values[ix2].mean(1),[.025,.975]).tolist(),
        "statistics":"exploratory percentile intervals, seed42,10000 cluster bootstrap replicates, no multiplicity correction",
        "tests_passed":int(match[1]),"protected_files_unchanged":True,"support_extended_joint":extended_joint,"source_exposure":"StudentLife reused; authored support challenge newly frozen but not independent humans","human_ratings":0}
    save(OUT/"verification.json",verification)
    for directory in ("support","support_repair"):
        analyze_ratings(OUT/directory/"blind",OUT/directory/"private/arm_key.json",OUT/directory/"independent_ratings_analysis.json")
        for name in ("评分指南.md","rater_declarations.template.json"):
            shutil.copy2(ROOT/"round14/support/blind"/name,OUT/directory/"blind"/name)
    reliability=read("forecast/reliability_results.json")
    lopo=results["lopo"];chosen=lopo["selected"]
    table=[]
    for name in names:
        m=lopo[name];table.append(f'| {name} | {m["participant_equal_mae"]:.4f} | {m["mae"]:.4f} | {m["rmse"]:.4f} |')
    report=f'''# 第十五轮优化报告（2026-11-17）

本轮专注新用户泛化和支持策略具体回应，音频、知识图谱及平台接入继续暂缓。全部修改属于独立离线研究模块；线上安全阈值、审计与升级规则、默认模型均未修改。

## 结果与采用范围

支持回复的具体关切提及代理在40个新编写场景（非暂停30条）中由10/30提高到22/30，即33.3%→73.3%，增加40个百分点；同时满足具体关切提及及原有目标表面检查的比例也从33.3%→73.3%。按10个事件家族聚类的探索性差值95%区间为{verification["topic_family_cluster_delta_ci95"]}。这支持在该编写场景上的具体回应改善，不是独立人评或真实帮助程度。

新用户留一评测在23个有未来标签的用户、150个窗口中，RMSE由保持法的{lopo["last"]["rmse"]:.4f}降至{chosen["rmse"]:.4f}，改善{100*(1-chosen["rmse"]/lopo["last"]["rmse"]):.1f}%；相对滚动均值改善{100*(1-chosen["rmse"]/lopo["mean7"]["rmse"]):.1f}%。两个RMSE差的聚类区间均在零以下，见下表后统计。主指标ID等权MAE仍未形成区间不跨零的优势，不能宣称全部指标或新来源泛化已经解决。

建议将F提示和G契约修复作为下一次独立评审候选；新用户预测保留带证据量回退的研究运行时，不替换平台默认模型。所有结果和原始失败均公开。

## 新用户：具体改动与公平比较

此前慢EMA在只有3个验证ID、43个验证窗口上选中，严格测试只有4个ID、13个窗口，难以代表新用户。首先冻结18个个人历史候选（保持、短均值/中位数、保守混合、不同EMA、因果局部选择），完全不把目标用户纳入超参数选择。窗口继续使用既定全局日期前推、14日历史、至少3个观测日、长缺测重启；不裁剪尾部日期、不换标签、不引入全程统计。

初版尾部误差/RMSE验证约束仍选中旧慢EMA，严格测试未改善，完整保存在forecast/results.json的strict字段。随后增加研究证据量回退：验证少于10个ID或100个窗口时回退保持法。这两个门槛是看到初版结果后加入的启发式，尚未独立验证。原严格划分ID等权MAE因此从1.4670回到0.8720，约减少40.6%，与保持法完全相同，**不是超过保持法的学习增益**。相对旧EMA的差值区间仍跨零。

另对49个发布ID逐一留出：每次选型只使用其余用户、测试日期之前的验证数据。26个ID在既定晚期没有合格标签，照实报告；23个ID进入评测。目标用户可以提供已过去的个人自评，这是“训练/选型未见用户、有少量个人历史”，不是零历史预测。验证的日期严格早于预测日期；没有用目标用户的历史预测误差挑选群体超参数。局部候选若被选中，其个人适应也只在过去窗口中先预测后看结果。

| 留一用户汇总方法 | ID等权MAE | 窗口MAE | RMSE |
|---|---:|---:|---:|
{chr(10).join(table)}

RMSE相对保持法差值95%区间{intervals["last"]}；相对mean7为{intervals["mean7"]}；相对旧慢EMA为{intervals["old_ema_0.1"]}。ID等权MAE区间见forecast/results.json，均仍跨零。所有区间为按用户聚类的探索性分析、未作多重比较校正。23用户与此前4用户协议不可横向比较其绝对误差；每个改进比例都在同一行协议内对相同窗口比较。

StudentLife来源、日期和原表与上一轮相同，数据已曝光，本轮留一协议属于复用来源的探索性评测，不是新独立数据确认。PAM类别间距单位仍是1–4效价的建模约定，不对应项目心情/压力分或临床风险。推理函数要求量表声明，错误量表拒绝迁移。

## 支持策略：具体回应和目标不再互相挤压

F在同一原始对话、同一3B本地NF4模型、确定性解码、相同审计和最多两次表面目标复查下，只增加要求：先准确概括当前修正后的具体任务/物品/情境，再执行倾听、梳理或一个小步骤；暂停时收束。不把评分关键词、参考答案或评分结果送入提示，也不根据关键词命中决定重试。

冻结的40个新编写场景是10类事件×4目标，E/F共80条；旧32条作为已曝光开发诊断，F具体提及22/24=91.7%，旧轮E为12/24=50%，不能当独立确认。新场景F为22/30，E为10/30，暂停不要求重述关切，故排除在具体提及分母之外。全部原始和重写回复、现有审计结果均保留，含未通过者，无通过后筛选。

逐条抽查发现旧自动指标会漏掉问题：F的“你先联系管理员吧”仍在暂停后追加行动；“你更关心的是车票问题吗”又询问已明确的关切。因此**原指标0违约不能等于真实0违约**。增加独立保留的G探索修复：明确暂停使用“好的，今天先到这里。”确认模板（不是模型生成收益）；重复核对已知关切时，以同一原始内容重新生成陈述式回应。不拼接关键词、不删除句子伪造LLM回复。

G修复后，具体关切提及27/30=90%，额外重复核对/暂停追加行动代理问题从F的11/40降到2/40；原目标违约代理由0/40变成1/40，现有审计不通过均为0。要求“具体提及、原目标检查和扩展检查同时满足”时，F为{extended_joint['F']['passed']}/30，G为{extended_joint['G']['passed']}/30。额外检查是少量可观察规则，仍不能覆盖所有隐含建议、无依据推断或语义重复；评分指南和F/G盲评包同时交付。G在看到F后开发，属于探索修复，没有独立新测试集。人工评分仍为0，不声称真实策略有效性、临床收益或广泛安全性。

## 验证与运行

相关回归测试{verification["tests_passed"]}项通过，现有Starlette弃用提示1项。49个外层用户选择记录、150个预测逐项回放通过；运行时与保存预测最大误差{verification["max_replay_error"]}。原线上受保护文件指纹与第十四轮交付一致。

项目根目录D:/Develop/AIC/AI-mental-main (2)/AI-mental-main/AI-mental-main；Python为D:/Develop/AIC/.venv/Scripts/python.exe；PowerShell先设 $env:PYTHONPATH='algorithm' 和 $env:PYTHONIOENCODING='utf-8'。

入口：experiments.round15_forecast为冻结候选评测；experiments.round15_runtime生成保守运行时artifact，predict_new_user为可导入推理函数（须先按时间和缺测契约整理历史）；experiments.round15_support生成E/F对照，messages(case,'F')为可复用提示构建器；experiments.round15_repair生成G探索修复；experiments.round15_finish完成证据复核和打包。已有冻结输出不应覆盖：完整重跑前在隔离副本中调整输出根目录，保持协议不变。

forecast/runtime_artifact.json包含可加载的确定配置与证据量回退原因，lopo_selection_locks.json保留每个用户的配置；所有方法推理共用forecast函数。support/blind为80条E/F评审，support_repair/blind为80条F/G评审；仅发送blind目录，不向评审提供private条件映射。两人完成声明和评分后可复用round14_human.analyze_ratings，适切性及忠实度不能自动填入。真人研究仍须实际发生，本轮未招募或发送消息。
'''
    (OUT/"第十五轮优化报告-20261117.md").write_text(report,encoding="utf-8")
    # 包含依赖快照、真实每日输入和来源记录；不含多GB基础模型。
    shutil.copytree(ROOT/"round14/source",OUT/"source",dirs_exist_ok=True)
    for path in (REPO/"algorithm/experiments").glob("round15_*.py"):
        shutil.copy2(path,OUT/"source/algorithm/experiments"/path.name)
    shutil.copy2(REPO/"algorithm/tests/test_round15_protocol.py",OUT/"source/algorithm/tests/test_round15_protocol.py")
    for filename in ("daily_observations.csv","preregister.json"):
        target=OUT/"data"/filename;target.parent.mkdir(exist_ok=True);shutil.copy2(ROOT/"round14/longitudinal/studentlife_calendar_days"/filename,target)
    shutil.copy2(ROOT/"round14/model_fingerprints.json",OUT/"model_fingerprints.json")
    files=sorted(p for p in OUT.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name!="manifest.json")
    manifest={str(p.relative_to(OUT)).replace("\\","/"):{"sha256":sha256(p),"bytes":p.stat().st_size} for p in files};save(OUT/"manifest.json",manifest)
    destination=ROOT/"第十五轮优化交付包-20261117.zip"
    with zipfile.ZipFile(destination,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files+[OUT/"manifest.json"]:archive.write(path,str(path.relative_to(OUT)))
    with zipfile.ZipFile(destination) as archive:
        for name,item in manifest.items():assert hashlib.sha256(archive.read(name)).hexdigest()==item["sha256"]
    print(json.dumps({"report":str(OUT/"第十五轮优化报告-20261117.md"),"files":len(manifest),"zip":str(destination),"verification":verification},ensure_ascii=False),flush=True)


if __name__=="__main__":finish()
