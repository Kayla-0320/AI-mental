"""第十七轮双指标与质量专家实验的证据复核和交付。"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile
import numpy as np
from experiments.round14_common import ROOT,REPO,save,sha256
from experiments.round17_forecast import OUT,DATA,SOURCES


def read(relative: str) -> dict:
    """读取已完成的本轮证据，不重选候选。"""
    return json.loads((OUT/relative).read_text(encoding="utf-8"))


def finish() -> None:
    """核对锁型、运行时、真实空人评和保护文件，保留全部正负结果后归档。"""
    lock=read("forecast/selection_lock.json");protocol=read("forecast/protocol.json")
    assert protocol["selected"]==lock["winner"] and not protocol["new_source_training"]
    assert lock["source_sha256"]==sha256(REPO/"algorithm/experiments/round17_forecast.py")
    runtime=read("runtime_verification.json");quality=read("quality/guarded_results.json");correction=read("quality/correction_results.json")
    qverify=read("quality/verification.json");cverify=read("quality/correction_verification.json")
    for row in runtime["forecast"].values():assert row["max_error"]<1e-12
    for row in runtime["quality"]+runtime["correction"]:assert row["max_error"]<1e-12
    scopes=json.loads((ROOT/"round14/scope_and_promotion.json").read_text(encoding="utf-8"))
    assert all(sha256(REPO/name)==digest for name,digest in scopes["protected_current_sha256"].items())
    human=json.loads((ROOT/"round16/human/semantic_analysis.json").read_text(encoding="utf-8"))
    assert human["completed_semantic_rows"]==0 and not human["model_used_as_rater"]
    save(OUT/"human_status.json",{"status":"awaiting_real_independent_ratings","actual_human_ratings":0,"model_rater_used":False,"new_dialogue_effectiveness_claim":False,
        "materials":"D:/Develop/AIC/_国赛优化/round16/human/blind (existing 80-response semantic pack remains pending)"})
    # 训练后仅新增推理入口，精确还原训练脚本并核对原预注册摘要。
    current=(REPO/"algorithm/experiments/round17_correction.py").read_bytes();start=current.index(b"def predict_correction(");end=current.index(b"def run()",start)
    training=current[:start]+current[end:];locked=OUT/"source_locked/correction_training.py";locked.parent.mkdir(parents=True,exist_ok=True);locked.write_bytes(training)
    assert sha256(locked)==read("quality/correction_protocol.json")["source_sha256"]
    for code,record in (("round17_quality.py","quality/protocol.json"),("round17_pixels.py","quality/pixels_protocol.json"),("round17_switch.py","quality/switch_protocol.json")):
        assert sha256(REPO/"algorithm/experiments"/code)==read(record)["source_sha256"]
    tests=(OUT/"tests.log").read_text(encoding="utf-8");count=re.findall(r"(\d+) passed",tests);assert count and "FAILED" not in tests
    raw=read("quality/raw_quality_v2.json");partial=sum(r["measured"] and r["metrics"]["frame_count"]<3 for r in raw.values())
    verification={"new_forecast_sources":2,"forecast_windows":sum(r["windows"] for r in runtime["forecast"].values()),"forecast_runtime_max_error":max(r["max_error"] for r in runtime["forecast"].values()),
        "natural_quality_clips":len(raw),"partial_frame_measurements":partial,"quality_runtime_rows":sum(r["samples"] for r in runtime["quality"]),
        "correction_runtime_rows":sum(r["samples"] for r in runtime["correction"]),"all_runtime_errors_below_1e_12":True,"tests_passed":int(count[-1]),
        "human_ratings":0,"production_files_unchanged":True,"test_labels_used_for_selection":False,"all_verified":True}
    save(OUT/"verification.json",verification)
    results={name:read(f"forecast/{name}/results.json") for name in SOURCES};table=[]
    for name,result in results.items():
        meta=read(f"data/{name}_metadata.json");item=next(x for x in meta["files"] if x["key"]==SOURCES[name]["filename"])
        path=DATA/item["key"];assert hashlib.md5(path.read_bytes()).hexdigest()==item["checksum"].split(":")[-1]
        assert sha256(path)==result["audit"]["source_sha256"]
        shutil.copy2(path,OUT/"data"/path.name)
        for method in ("selected","previous_ema","previous_robust","last","mean7"):
            m=result["metrics"][method];table.append(f'| {name} | {method} | {m["people"]}/{m["windows"]} | {m["person_mae_0_100"]:.4f} | {m["rmse_0_100"]:.4f} | {m["worst_quarter_person_mae"]:.4f} |')
    report=f'''# 第十七轮优化报告：预测双指标取舍与图文质量/专家切换（2026-11-17）

本轮只推进离线预测与文本视觉质量/切换。平台接入、音频与专业知识图谱继续暂缓；真人盲评保持待采集，实际评分0，未调用模型替代评审。原安全阈值、审计与升级、平台默认模型文件保持不变。

## 结果与采用建议

1. 双指标锁型后，在两个首次使用的新来源上评测3268个窗口。相对滚动均值，Westhoff的ID等权MAE改善约3.3%、RMSE改善约3.7%；Wang分别改善约7.9%、5.8%。两来源对应的MAE/RMSE聚类差值区间都支持优于滚动均值。
2. 相对上一轮EMA，Wang的MAE改善约2.4%、RMSE改善约0.8%；Westhoff的MAE退步约0.13%、RMSE改善约0.17%，均属很小变化。没有把结果写成所有指标完全解决；相对旧EMA的RMSE区间仍跨零。
3. 实际读取2876个自然原片目标三帧测量，另有48片×5种真实像素探针。新增均匀中心块遮挡代理在48个受控遮挡探针的标记率从0/48变为48/48，48个原片探针标记数仍为0。自然原片质量门控额外切换0，分类指标保持不变，不能把探针检出率称为分类增益或真实遮挡泛化。
4. 新纠错收益切换器相对简单均值的Accuracy由0.7559升到0.7656（约+0.97个百分点），Macro-F1由0.6877降到0.6868（约-0.095个百分点）。因此保留为准确率取向的探索候选，不替换宏F1默认参考。

## 预测：开发角色、锁型和外部来源

StudentLife及第十六轮的Sicorello、Dejonckheere数据明确转为本轮开发数据，不能再称本轮独立测试。固定20个候选包括旧EMA、稳健混合、较快/较慢EMA及个人历史AR。按来源等权优化MAE/RMSE相对旧EMA的平均比值，要求两个指标的跨来源均值都不退步，且任一开发来源/指标退步不超过2%。选中EMA alpha=0.4；其他AR和混合候选的全部分数保留。

新来源在锁型后下载并评测，没有根据它们再选alpha、换目标或改变门槛；也没有按来源各取最好候选。Wang概念记录接口一度500，改取同一已公开固定版本元数据后恢复，日志保留；没有更改预测算法或结果筛选。

- [Westhoff来源说明](https://openesmdata.org/datasets/0008_westhoff/)，[固定版本21997291](https://zenodo.org/records/21997291)，CC BY4.0。使用affect_valence自评，0–100，实际可用113个ID、2014个窗口。
- [Wang来源说明](https://openesmdata.org/datasets/0055_wang/)，[固定版本22809736](https://zenodo.org/records/22809736)，CC BY-NC4.0。使用momentary sad自评，原量表1–100；72个18–25岁来源ID、1254个窗口。没有把sad_daily与瞬时sad混作同一变量。

只下载整理后的时间序列表，不下载静态个人信息。按发布研究日聚合每日至少2份有效响应，检查冲突同人/同日/beep、原量表边界；缺失不补零。历史至少3日、最多14日，超过3日缺测后重新暖启动。两个来源都按发布day索引评测下一研究日，不捏造日历时间。误差归一化0–100只是展示约定，两种情绪构念不能合成一个心理分，也不等于项目用户或中国青少年效果。

| 来源 | 方法 | ID/窗口 | ID等权MAE | RMSE | 最差四分之一用户MAE |
|---|---|---:|---:|---:|---:|
{chr(10).join(table)}

统计见forecast/inference.json：按用户聚类5000次配对bootstrap、种子42；8个MAE比较做Holm校正，RMSE区间为探索性分析。Wang对旧EMA的MAE差-0.1760、校正p约0.011；Westhoff对旧EMA的MAE差+0.0112、区间跨零。相对旧稳健候选，Westhoff两项改善，Wang则以极小MAE代价换取较低RMSE。覆盖率100%仅指合格预测窗口，不含所有缺失/冷启动日。

## 图文质量：原始可观察证据与语义分歧分开

复用原曝光、对比度、拉普拉斯清晰度等质量证据，新增“中心大块均匀遮挡”代理：中心64像素区域方差极低、周边仍有对比，且至少80%抽样帧满足时，软质量分数最高0.25。该规则只针对均匀中心块，不声称涵盖手部、头发、真实口罩等所有自然遮挡；静态场景本身不被判为无效。短文本、否定句和单字“好”保留，只有空白、纯标点、纯链接缺少文字证据。

质量测量函数不接收情绪标签、专家概率或图文一致性，因而不会把情绪分歧当作图像损坏。替代文字专家的信心得分只在决定软降级时使用，是分类器得分而不是已校准的人类可信度。

原片目标三帧位于20%/50%/80%，缩略至128像素，仅解码视频，不打开音轨。实际完成2876片测量，部分FFmpeg解码告警和{partial}片不足三帧的记录保留。未测得质量时标为未核实，不擅自判坏。帧摘要、逐片统计、未核实状态和像素变换探针均归档。

黑屏、暗光、模糊与中心遮挡探针来自48个ID哈希预选原片，真实变换RGB帧。此阶段**只评质量检测器**，没有重新提取完整视觉情绪表征，所以不把缓存特征的缩放伪装成原片损坏后的分类结果。原片探针标记率也不是有真人质量标签的假阳性率。

专家降级的阈值和替代文字信心门槛只在外层训练来源的内部OOF概率上选定，外层来源不参与选择。自然原片的旧质量门控与新增质量门控均切换0，Macro-F1维持0.687701、Accuracy维持0.755911。这支持当前样本上的不干扰结果，**没有证明自然低质量样本分类改善**；真实遮挡、低照度困难样本仍需独立标签和评测。

## 专家切换：预测替换收益，明确保留取舍

除质量降级，另测试一个固定纠错候选：三个小HGB回归器预测“T/V/TV替代均值后的净正确性变化”，目标为-1/0/+1；不是心理状态、语义质量或真人评分。训练专家证据来自内部来源OOF，门槛由元OOF选择；外层来源对专家、门控、阈值均保持隔离。

本轮纠错器的92次类别切换中，56次纠正均值错误，28次把原先正确变错，8次仍错误；不是所有切换均有效。Accuracy改善，但宏F1略降，说明收益在类别间不均衡，不能只报准确率。来源聚类区间和相对同一均值基线的旧学习门控得失均保存在quality/correction_verification.json，不把不同基线的误伤计数直接比较。

默认仍保留简单均值作为宏F1参考，纠错器作为准确率取向的离线候选。硬缺失和专家资格仍约束所有输出，全缺失使用训练先验；不可用专家不能贡献结果。概率型纠错器需要先有专家概率，不声称稀疏或编码器加速。质量优先路由可先算文字专家，只有合格时执行V/TV头，实际调用数单独记录，不外推端到端速度。

## 验证与交付

相关测试{verification['tests_passed']}项通过，1项现有Starlette弃用提示。3268个预测窗口、2876个质量路由样本和2876个纠错样本的可调用运行时逐项回放一致，误差低于1e-12；原保护文件摘要不变。纠错训练后仅添加推理入口，原训练源代码已精确还原并核对预注册摘要，保存在source_locked。

首次像素测量因使用错误字段名失败，修正为共享数据类的reason_codes后继续，失败代码/日志/协议保留。质量单测先前误把低质量代理要求为严格0，已改为验证低分和曝光原因；没有改动生产质量或安全规则来迎合测试。

真人盲评继续使用第十六轮80条回复的双人语义包，真实评分0，没有模型代填，也没有把本轮分类模型称为真人评价。平台接入、音频及知识图谱维持暂缓。本轮不将某个离线候选自动设为平台默认方案。
'''
    (OUT/"第十七轮优化报告-20261117.md").write_text(report,encoding="utf-8")
    usage=r'''# 使用与复算

项目目录：D:\Develop\AIC\AI-mental-main (2)\AI-mental-main\AI-mental-main
解释器：D:\Develop\AIC\.venv\Scripts\python.exe
PowerShell：$env:PYTHONPATH='algorithm'；$env:PYTHONIOENCODING='utf-8'。

预测推理：experiments.round17_runtime.predict_day(payload)。字段dataset为westhoff或wang，source_schema须精确匹配openesm:westhoff:affect_valence:0_100或openesm:wang:sad:1_100；observations含同一person、递增day、原量表value；target_day等于最新日+1。未来/重复日和错误量表被拒绝，缺测后历史不足会拒绝预测。
回放预测：python -m experiments.round17_runtime。

质量检测：experiments.round17_pixels.quality_from_frames(RGB_frames)，返回共享RawSignalQuality；只提供非语义可观察代理。
质量优先图文推理：experiments.round17_runtime.predict_tv_quality(T,V,mask,texts,quality,folder,config)。mask为N×2 bool；quality为N个原始测量分数，NaN为未核实；folder使用已有round16/moe/export/fold0，config使用本轮quality/guarded_selection_locks.json对应折的winner.config。
纠错推理：experiments.round17_correction.predict_correction(bank,mask,quality,folder,prior)，bank为T/V/TV三专家的M3概率，mask为N×3且A必须False，folder为本轮quality/correction_models/fold0，prior来自配套旧专家。模型摘要经过检查，不接收标签或来源ID。

完整核对：python -m experiments.round17_verify；构建报告和ZIP：python -m experiments.round17_finish。
重训/重新测量应在独立输出副本中进行，保留原协议和全部负结果。source_locked/correction_training.py为实际训练版本；新推理入口在当前源码中。MoE重训及原片测量依赖此前本地冻结缓存和公开视频；这些大型原片/编码器不重复打包。可复用第十六轮T/V导出专家，包内提供匹配副本。

真人盲评材料仍在D:\Develop\AIC\_国赛优化\round16\human\blind。未有真实人评前不得填造得分。
'''
    (OUT/"使用说明.md").write_text(usage,encoding="utf-8")
    notice='''# Source attribution

Westhoff, Heshmati, Siepe, Vogelbacher, Ciarrochi, Hayes, Hofmann (2024), Psychological flexibility and cognitive-affective processes in young adults’ daily lives, https://doi.org/10.1038/s41598-024-58598-3. Harmonized by openESM: https://zenodo.org/records/21997291, CC BY4.0, https://creativecommons.org/licenses/by/4.0/.

Wang, McGowan, Fosco, Falk, Bassett, Lydon-Staley (2024), A socioemotional network perspective on momentary experiences of family conflict in young adults, https://doi.org/10.1111/famp.12995. Harmonized by openESM: https://zenodo.org/records/22809736, CC BY-NC4.0, https://creativecommons.org/licenses/by-nc/4.0/.

TSV copies are unchanged, verified against publisher MD5. Derived tables select declared momentary targets, quarantine conflicting beeps, require at least2 daily responses, average observed values and normalize with theoretical scale bounds. Days are published study-day indices; calendar dates are not inferred. Static personal data are not downloaded. Research evidence does not grant additional commercial rights.

Visual measurements use previously available CH-SIMS public media. Original video is not redistributed here. RGB probe descriptions are controlled quality-detection experiments, not labels of natural clinical states or user effects.
'''
    (OUT/"data/SOURCE_NOTICE.md").write_text(notice,encoding="utf-8")
    shutil.copytree(ROOT/"round16/source",OUT/"source",dirs_exist_ok=True)
    for path in (REPO/"algorithm/experiments").glob("round17_*.py"):shutil.copy2(path,OUT/"source/algorithm/experiments"/path.name)
    shutil.copy2(REPO/"algorithm/tests/test_round17_protocol.py",OUT/"source/algorithm/tests/test_round17_protocol.py")
    for name in ("__init__.py","raw_signals.py"):
        target=OUT/"source/algorithm/data_quality"/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/"algorithm/data_quality"/name,target)
    shutil.copytree(ROOT/"round16/moe/export",OUT/"models/tv_experts",dirs_exist_ok=True)
    files=sorted(p for p in OUT.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p!=OUT/"manifest.json")
    manifest={str(p.relative_to(OUT)).replace("\\","/"):{"sha256":sha256(p),"bytes":p.stat().st_size} for p in files};save(OUT/"manifest.json",manifest)
    destination=ROOT/"第十七轮优化交付包-20261117.zip"
    with zipfile.ZipFile(destination,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files+[OUT/"manifest.json"]:archive.write(path,str(path.relative_to(OUT)))
    with zipfile.ZipFile(destination) as archive:
        for fold in range(4):assert f"models/tv_experts/fold{fold}/manifest.json" in {n.replace("\\","/") for n in archive.namelist()}
        for name,item in manifest.items():assert hashlib.sha256(archive.read(name)).hexdigest()==item["sha256"]
    print(json.dumps({"report":str(OUT/"第十七轮优化报告-20261117.md"),"zip":str(destination),"files":len(files),"bytes":destination.stat().st_size,"verification":verification},ensure_ascii=False),flush=True)


if __name__=="__main__":finish()
