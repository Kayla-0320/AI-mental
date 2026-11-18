"""完整保留第十八轮正负结果、冻结源代码、真实待采集状态并核验交付ZIP。"""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import zipfile
from experiments.round14_common import ROOT,REPO,save,sha256

OUT=ROOT/"round18"


def read(name: str) -> dict:
    """读取本轮已固定结果，不在交付阶段调整候选。"""
    return json.loads((OUT/name).read_text(encoding="utf-8"))


def finish() -> None:
    """核验协议哈希、评测角色和真实零人评，提供可回放源码与模型。"""
    verification=read("verification.json");neutral=read("neutral/results.json");images=read("images/results.json");guard=read("images/conservative_results.json");natural=read("natural_quality/results.json");quality=read("quality_model/results.json");prediction=read("forecast/results.json");graph=read("knowledge_graph/status.json")
    for file,protocol in (("round18_neutral.py","neutral/protocol.json"),("round18_images.py","images/protocol.json"),("round18_visual_guard.py","images/conservative_protocol.json"),("round18_uncertainty.py","forecast/protocol.json"),("round18_natural_quality.py","natural_quality/protocol_v3.json"),("round18_quality_model.py","quality_model/protocol_v2.json")):
        assert sha256(REPO/"algorithm/experiments"/file)==read(protocol)["source_sha256"]
    assert sha256(OUT/"natural_quality/source_locked_download_v1.py")==read("natural_quality/protocol.json")["source_sha256"]
    assert sha256(OUT/"natural_quality/source_locked_join_v2.py")==read("natural_quality/protocol_v2.json")["source_sha256"]
    assert sha256(OUT/"quality_model/source_locked_training_v1.py")==read("quality_model/protocol.json")["source_sha256"]==read("quality_model/selection_lock.json")["source_sha256"]
    assert sha256(OUT/"forecast/calibration_lock.json")==read("forecast/protocol.json")["calibration_sha256"]
    assert all(sha256(REPO/p)==v for p,v in verification["protected_current_sha256"].items())
    human=json.loads((ROOT/"round16/human/semantic_analysis.json").read_text(encoding="utf-8"));assert human["completed_semantic_rows"]==0 and not human["model_used_as_rater"]
    save(OUT/"human_status.json",{"actual_project_human_blind_ratings":0,"model_rater_used":False,"status":"awaiting_real_independent_ratings","existing_materials":str(ROOT/"round16/human/blind"),"public_image_MOS_not_project_reply_ratings":True,"graph_human_expert_reviews":0})
    tests=(OUT/"tests.log").read_text(encoding="utf-8");counts=re.findall(r"(\d+) passed",tests);assert counts and "FAILED" not in tests
    verification["tests_passed"]=int(counts[-1]);save(OUT/"verification.json",verification)
    save(OUT/"promotion_decisions.json",{"neutral":"offline neutral-priority candidate; baseline overall intervals cross zero, no promotion","visual_guard":"controlled-development repair only; preserve other visual evidence","quality_model":"standalone natural-quality research head; no automatic fusion switching","forecast_point":"retain previous EMA .4, new tail candidates rejected","forecast_interval":"constant empirical reference only; adaptive undercoverage candidate rejected","knowledge_graph":"traceable v0.1 draft pending human expert review, default excludes pending","platform":"deferred","audio":"deferred","new_production_changes":False,"project_human_ratings":0,"graph_human_expert_reviews":0})
    rows=[]
    for name,label in (("baseline","同一均值融合参考"),("previous","第十七轮纠错候选"),("selected","本轮中性保护候选")):
        m=neutral[name];rows.append(f'| {label} | {m["macro_f1"]:.6f} | {m["neutral_f1"]:.6f} | {m["neutral_recall"]:.6f} | {m["accuracy"]:.6f} |')
    visual=[]
    for c,r in images["conditions"].items():visual.append(f'| {c} | {r["ordinary_TV"]["accuracy"]:.4f} | {r["uniform"]["accuracy"]:.4f} | {r["quality"]["accuracy"]:.4f} | {guard[c]["selected"]["accuracy"]:.4f} | {guard[c]["drop_count"]} |')
    qm=quality["test"];bm=quality["training_mean"];fm=prediction["point"];ci=read("neutral/inference.json")
    report=f'''# 第十八轮优化报告（2026-11-17）

本轮依次推进中性识别与专家切换、真实像素重编码、自然画质公开评分验证、预测尾部误差与不确定性，并启动知识图谱。所有新增算法为离线可调用研究模块，未接入平台。音频继续暂缓。项目真人盲评仍0份，模型没有替代人评；图谱专家审核仍0条。

## 本轮结论

- 中性召回相对第十七轮纠错候选从48.42%恢复到55.01%，中性F1从0.4178恢复到0.4369。代价是准确率从76.56%回落到75.70%；相对均值参考的提升很小，不宣称综合表现稳定提高。
- 真实RGB损坏后重新提取视觉特征，发现“检出质量低就丢视觉”会误伤。保守降级仅在每帧空间完全恒定或无帧时使用文字：黑屏准确率相对均值从65.04%升到66.80%，正常、暗光、模糊、部分遮挡保持均值结果。
- 旧画质代理在独立自然照片探针上全给1分，无法排序。新增冻结ChineseCLIP+画质回归头，在256张严格留出的新图像上Spearman={qm['spearman']:.4f}、低画质AUROC={qm['auc_MOS_below_2_5']:.4f}；这是画质判断进展，尚不是图文情绪分类增益。
- 尾部优化候选没有胜过旧EMA，保留EMA .4。新来源Beck的171名参与者、1753个窗口上点预测不变；常数经验误差带覆盖92.47%，自适应带仅88.42%，后者不采用。
- 图谱初版28节点/29边，其中18条来自公开指南概念，11条是工程契约。均有出处及限制，全部待人工专家审核；默认拒绝待审核内容，不生成个人诊断或疗效结论。

## 中性识别：恢复与取舍

沿用2876条CH开发数据和107个来源分组，外层四折保留来源隔离。外层训练集内部使用元特征OOF训练；选择要求中性F1不低于参考、中性召回下降不超过0.01、准确率下降不超过0.002，失败则精确回退均值。内层条件不等于外层结果保证。元分类器固定小容量，搜索中性样本权重、混合权重、中性偏置与离开中性的最小分类差值。它们是情绪分类参数，不是安全阈值。

| 方法 | Macro-F1 | 中性F1 | 中性召回 | Accuracy |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

相对同一均值参考，本轮纠正29个错误、引入26个新错；上轮对应56/28。本轮少引入2个错误，也少纠正27个错误，不能只报误伤减少。相对上轮中性召回增加6.59个百分点，按来源聚类探索区间为{ci['previous']['delta_ci95']['neutral_recall']}；准确率差值区间为{ci['previous']['delta_ci95']['accuracy']}，代价必须一起报告。

相对均值参考，中性F1增约0.00159、Macro-F1增约0.00032；Macro-F1差值95%区间{ci['baseline']['delta_ci95']['macro_f1']}，中性F1区间{ci['baseline']['delta_ci95']['neutral_f1']}，均跨零。中性召回比均值仍低约0.86个百分点。因此保留“中性优先”候选，不替代默认整体参考。旧来源已经反复曝光；四折与bootstrap都不能重新把它变成首次独立测试。

## 图文质量与专家切换：真正重新编码

按ID哈希固定512段视频，实际抽取1522帧，生成正常、黑屏、暗光、模糊、中心遮挡五种RGB条件，共进行7610帧真实视觉编码，没有用特征缩放代替损坏。T/V/TV头共享训练行、干净训练条件及每折训练内PCA/归一化，按来源四折；音轨未解码。

此实验视觉为ChineseCLIP512维，文字沿用冻结7B缓存；没有拼入旧模型的人脸块，故其干净准确率不能与旧强模型直接横比。所有方法只在同一新特征体系内比较。

| 条件 | 普通TV准确率 | 均值准确率 | 质量≤.51直接退出V | 保守像素降级 | 保守退出数 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(visual)}

暗光、模糊和部分遮挡仍有情绪信息，直接退出V的负结果完整保留。保守规则只看实际RGB，不接收标签、故障条件名称或专家情绪分歧。该修复在看到初次结果后提出，因此明确标记为同批素材上的探索修复，黑屏收益不是新独立泛化结论。图像清晰度也不代表文字语义可靠，文字质量和图文语义冲突仍需分别验证。

## 自然画质：公开真人MOS验证与新画质头

[KonIQ-10k作者介绍](https://database.mmsp-kn.de/koniq-10k-database.html)、[作者固定归档V1](https://doi.org/10.18419/DARUS-2435)。先从图像归档按文件名哈希取128张，与质量分无关。归档有部分未出现在MOS表的图片，2张未匹配，隔离且不补选，实际评分对齐126张。原代理质量分全部1，Spearman因常数而未定义，低MOS AUROC=0.5、低MOS召回=0。不能将受控黑屏/遮挡检出率当作自然画质排序能力。

在这个诊断后，从未参与该探针的其他文件名按固定哈希划分768训练、192验证、256最终留出，封存所有ID。冻结ChineseCLIP视觉512维加10项RGB统计；训练候选Ridge alpha1/10/100与ExtraTrees，按验证Spearman选{quality['selected']}，在训练+验证960图重拟合后只评一次256图。最终测试图像不用于选模型、阈值或特征；1216个图像文件SHA256互不重复。冻结编码器的历史预训练图像重叠未知，不宣称完全排除互联网预训练接触。

| 最终256图 | Spearman | 低MOS(<2.5) AUROC | MOS MAE | MOS RMSE |
|---|---:|---:|---:|---:|
| 新画质头 | {qm['spearman']:.4f} | {qm['auc_MOS_below_2_5']:.4f} | {qm['mae_MOS_1_5']:.4f} | {qm['rmse_MOS_1_5']:.4f} |
| 训练均值 | 未定义 | 0.5000 | {bm['mae_MOS_1_5']:.4f} | {bm['rmse_MOS_1_5']:.4f} |

相对训练均值的MAE差值探索bootstrap区间：{quality['mae_delta_bootstrap_ci95']}。这是同一自然照片来源的图像留出验证，未覆盖自然视频、人脸专用画质、与文字匹配程度或情绪可靠性。画质头仅独立输出预测MOS，未把它套入旧.51阈值或强行作为情绪专家权重；这样做仍需配对情绪数据验证。公开MOS是真实既有人工图片评分，不是项目支持回复的真人盲评，更不能把预测MOS记为新增真人评分。

下载使用作者归档HTTP Range、每个JPEG校验ZIP CRC并保存SHA256，不伪称验证整个767MB归档。原始图片留在本机公开数据缓存，未放入交付ZIP；归档声明元数据CC BY4.0，单图各自遵循YFCC100M原有许可。首次对齐与常数相关性序列化错误保留日志，原冻结源码与协议保留；修复仅隔离缺失标签、将未定义相关性写为null，没有改样本、阈值或重新选型。

## 预测尾部与不确定性

开发数据为StudentLife、Sicorello、Dejonckheere、Westhoff、Wang，均已曝光。各来源按人70/30分隔选型与校准，不共享用户。比较EMA .4、噪声时减慢与单向趋势时加快，联合MAE/RMSE/最差四分之一用户MAE选择，最终仍选原EMA .4，未带来新的点误差改善。

[Beck公开说明](https://openesmdata.org/datasets/0060_beck/)、[固定记录22764751](https://zenodo.org/records/22764751)、[Beck与Jackson 2022论文](https://doi.org/10.1177/09567976221093307)。该来源本轮首次用于结局验证，未用于训练、校准或选型。8672原始行，2475合格人日，171人/1753预测窗口。sad为1–5重复性格项目“常感到悲伤”的回答，不能等同临床状态或青少年项目目标。beep字段全缺失，按人/日期/小时/分钟核验重复，冲突隔离；每日至少2响应、下一实测日预测、至少3天历史、14天回看、长缺测重新暖启动。

归一化0–100展示：用户等权MAE={fm['person_mae_0_100']:.4f}，RMSE={fm['rmse_0_100']:.4f}，P90绝对误差={fm['q90_abs_error']:.4f}，最差四分之一用户MAE={fm['worst_quarter_person_mae']:.4f}。与上轮点预测完全相同，不把新数据上的分数当成本轮提升。

| 经验误差带 | 覆盖率 | 平均宽度(0–100) | 区间分数，低好 |
|---|---:|---:|---:|
| 常数半径参考 | 92.47% | 45.83 | 59.38 |
| 过去波动自适应候选 | 88.42% | 44.62 | 63.47 |

误差带由旧来源隔离用户校准，均未看Beck目标；自适应只看过去相邻变化，但新来源覆盖不足，不能因宽度稍窄就采用。常数带也较宽，只是经验参考，没有分布无关或个体90%覆盖保证。forecast/coverage_inference.json给出用户聚类覆盖区间，不使用新结局重新调整半径。

## 知识图谱开工范围

基于[WHO《Doing what matters in times of stress》](https://cdn.who.int/media/docs/default-source/thailand/doing-what-matters-in-times-of-stress-eng.pdf)和[SKOS标准](https://www.w3.org/TR/skos-reference/)整理压力管理概念。五类核心概念包括锚定当下、与想法解钩、重视的方向、善待、容纳困难体验；每条来源关系提供工具/页码、URL和证据类型。项目倾听/梳理/行动/暂停、本人修正、原话出处等契约另标工程关系，不冒称指南结论。

交付JSON、JSON-LD、CSV节点/边、空专家审核表；检查悬空边、重复标识、缺来源、非法关系与无审核人却标通过。检索默认排除待审核及已拒绝项；研究模式必须双重显式开启才返回带出处概念，且不应用于具体用户。倾听/暂停、否定、引述与超出概念范围的查询有拒答测试。五道自编关系恢复题及乱图/无图负对照仅验证关系管线，不是独立问答精度、真人评价或知识增强效果。图谱28/29是可审查初版规模，不能称为已经做实的专业心理知识图谱。

WHO概念改编按CC BY-NC-SA3.0IGO归属；中文术语是工程整理，以英文原作核对，WHO未背书。待专业人员核查关系、适用边界、出处与中文术语，再增加多来源冲突和版本维护。专家审核与真人采集的实际完成数保持0，未由模型伪填。

## 检查、复现和边界

相关回归{verification['tests_passed']}项通过。2876条中性接口、五条件共2560条图像分类、256张画质头和1753个预测窗口已逐项回放，最大误差小于1e-12。源码、协议、划分、逐样本预测、负结果和摘要随包。source为本轮脚本及运行依赖快照，不表示其中历史模块都在本轮修改。

本轮仅新增离线实验/测试；当前工作区已有与第十四轮不同的support_understanding/shared数据类版本，verification.json如实列出差异，没有覆盖或回滚这些既有修改，也没有声称全部文件仍与第十四轮相同。当前保护文件摘要在交付前复核一致。未修改生产风险阈值、升级条件或执行平台接入。

下步最值得做：保留普通均值参照，在全新来源验证中性召回与准确率取舍；用真实配对视频/文字及情绪标签验证画质头能否改善融合；对预测极端变化日增加真正有效的新观测；专业图谱人工核查；真人盲评采集。已有公开数据不能替代后三者所需的特定证据。
'''
    (OUT/"第十八轮优化报告-20261117.md").write_text(report,encoding="utf-8")
    (OUT/"使用说明.md").write_text('''# 第十八轮离线研究交付

入口位于source/algorithm/experiments/round18_*.py，运行前设置PYTHONPATH=algorithm。默认输出路径D:/Develop/AIC/_国赛优化/round18，外部模型D:/ai/models/encoders/chinese-clip-vit-base-patch16。不是平台安装或生产替换包。

可直接调用round18_runtime.predict_neutral(bank,mask,quality,folder)、predict_quality_features(values,folder)、predict_day(payload)，以及round18_graph.retrieve；各自输入含义见中文docstring和测试。元分类器只接受完整图文无音频；缺模态仍由原合格专家均值回退，不把未验证输入交给新模型。画质MOS不自动控制情绪切换。

已有运行结果：round18_verify执行完整工作区回放；round18_replay可用本包缓存和模型独立回放。原始视频和大编码器不重复打包，重新提取原始特征需要现有本地公开数据与模型。生成协议为冻结文件，不得覆盖历史记录以试新参数，应使用新输出目录/新轮次。

data/0060_beck_ts.tsv为发布原件；data/koniq_scores.tab为公开MOS表，图像未打包。各来源许可与局限见SOURCE_NOTICE.md。human_status.json中的项目真人盲评与图谱专家审核仍是0；原盲评材料仅供将来真实采集。
''',encoding="utf-8")
    (OUT/"SOURCE_NOTICE.md").write_text('''# Attribution and scope

Beck, E. D. & Jackson, J. J. (2022), https://doi.org/10.1177/09567976221093307. Harmonized time series from openESM, fixed Zenodo record22764751, https://zenodo.org/records/22764751, CC BY-NC4.0 https://creativecommons.org/licenses/by-nc/4.0/. Original TSV preserved; no static personal table downloaded. Repeated BFI2 sad1–5 is not a clinical or project target substitute.

Hosu, V., Lin, H., Sziranyi, T., & Saupe, D. (2020), KonIQ-10k: An ecologically valid database for deep learning of blind image quality assessment. IEEE TIP29:4041–4056. Author archiveV1 https://doi.org/10.18419/DARUS-2435 ; https://database.mmsp-kn.de/koniq-10k-database.html . Dataset metadataCC BY4.0 https://creativecommons.org/licenses/by/4.0/ ; individual photos retain their YFCC100M Creative Commons licenses. Original photos remain in local research cache and are not redistributed here. Public MOS human annotations evaluate image quality only; predicted MOS and project reply ratings are separate.

World Health Organization (2020). Doing what matters in times of stress: an illustrated guide. Geneva:WHO. https://cdn.who.int/media/docs/default-source/thailand/doing-what-matters-in-times-of-stress-eng.pdf . Guideline-derived concept adaptationCC BY-NC-SA3.0IGO https://creativecommons.org/licenses/by-nc-sa/3.0/igo/ . Chinese concept terms are engineering adaptations, not an official WHO translation; the English original is authoritative. WHO does not endorse this project. No WHO logo or original guide is redistributed. Project contract relations are separately identified and are not guideline claims. All expert review remains pending.

W3C SKOS Reference, https://www.w3.org/TR/skos-reference/ , used to represent concepts; custom project relations are not represented as validated clinical ontology facts. CH-SIMS development media and existing frozen text embeddings are reused for source-group diagnostics; original videos and large third-party encoders are not redistributed. No new audio decoding or audio expert training.
''',encoding="utf-8")
    (OUT/"data").mkdir(exist_ok=True)
    for src,name in ((Path(r"D:\Develop\AIC\data\public\round18_sources\0060_beck_ts.tsv"),"0060_beck_ts.tsv"),(Path(r"D:\Develop\AIC\data\public\round18_sources\koniq\scores.tab"),"koniq_scores.tab")):shutil.copy2(src,OUT/"data"/name)
    shutil.copytree(ROOT/"round17/source",OUT/"source",dirs_exist_ok=True)
    for path in (REPO/"algorithm/experiments").glob("round18_*.py"):shutil.copy2(path,OUT/"source/algorithm/experiments"/path.name)
    shutil.copy2(REPO/"algorithm/tests/test_round18_protocol.py",OUT/"source/algorithm/tests/test_round18_protocol.py")
    for name in ("shared/dataclasses.py","shared/__init__.py","perception/knowledge_moe/features.py","data_quality/raw_signals.py","data_quality/__init__.py"):
        dest=OUT/"source/algorithm"/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/"algorithm"/name,dest)
    save(OUT/"environment_versions.json",{n:importlib.metadata.version(n) for n in ("numpy","scipy","pandas","scikit-learn","torch","transformers","Pillow","requests")})
    shutil.copytree(ROOT/"round16/moe/export",OUT/"models/tv_experts",dirs_exist_ok=True)
    files=sorted(p for p in OUT.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p!=OUT/"manifest.json")
    manifest={str(p.relative_to(OUT)).replace("\\","/"):{"sha256":sha256(p),"bytes":p.stat().st_size} for p in files};save(OUT/"manifest.json",manifest)
    destination=ROOT/"第十八轮优化交付包-20261117.zip"
    with zipfile.ZipFile(destination,"w",compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files+[OUT/"manifest.json"]:archive.write(path,str(path.relative_to(OUT)).replace("\\","/"))
    with zipfile.ZipFile(destination) as archive:
        assert "quality_model/manifest.json" in archive.namelist()
        for fold in range(4):assert f"models/tv_experts/fold{fold}/manifest.json" in archive.namelist()
        for name,item in manifest.items():assert hashlib.sha256(archive.read(name)).hexdigest()==item["sha256"]
    print(json.dumps({"report":str(OUT/"第十八轮优化报告-20261117.md"),"zip":str(destination),"files":len(files),"bytes":destination.stat().st_size,"tests":verification["tests_passed"]},ensure_ascii=False),flush=True)


if __name__=="__main__":finish()
