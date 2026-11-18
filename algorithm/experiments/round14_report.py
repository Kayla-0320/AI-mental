"""整理第十四轮报告、冻结源码、原始表字段证据与可审阅交付包。"""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from experiments.round14_common import DATA, OUT, REPO, ROOT, save, sha256
from experiments.round14_human import analyze_ratings, analyze_study
from experiments.round14_runtime import verify_artifacts


def snapshot() -> None:
    """保留当前代码和早先仅CLI量纲分支不同的青少年运行版本，并核对原摘要。"""
    for path in (REPO / "algorithm" / "experiments").glob("round14_*.py"):
        target = OUT / "source" / "algorithm" / "experiments" / path.name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    for relative in ("algorithm/tests/test_round14_protocol.py", "algorithm/intervention/auditor.py", "algorithm/intervention/teen_language_detector.py",
        "algorithm/intervention/support_understanding.py", "algorithm/assessment/personal_baseline.py", "algorithm/shared/dataclasses.py",
        "algorithm/shared/__init__.py", "algorithm/config/audit_rules.yaml", "algorithm/config/baseline.yaml"):
        path = REPO / relative; target = OUT / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    current = (REPO / "algorithm/experiments/round14_longitudinal.py").read_bytes()
    older = current.replace(b'data, info, 3., 1 / 3)', b'data, info, 4., .25)')
    for name, content, experiment in (("current", current, "studentlife_calendar_days"), ("adolescent_locked", older, "adolescent_sampling_days_reused")):
        target = OUT / "source_locked" / name / "round14_longitudinal.py"
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(content)
        protocol = json.loads((OUT / "longitudinal" / experiment / "preregister.json").read_text(encoding="utf-8"))
        if sha256(target) != protocol["code_sha256"]: raise ValueError("无法还原冻结源代码，拒绝生成可复现声明")
    for source, filename in ((DATA / "SL_tbl_trans.R", "PAM_mapping_SL_tbl_trans.R"), (DATA / "ema_rds/PAM.Rds", "PAM.Rds"),
        (ROOT / "round2/forecast/minimal_dataset_emolife.xlsx", "adolescent_previously_used.xlsx"),
        (DATA / "studentlife_DESCRIPTION", "studentlife_DESCRIPTION"), (DATA / "GPL-3.0.txt", "GPL-3.0.txt")):
        shutil.copy2(source, OUT / "data" / filename)
    (OUT / "data/SOURCE_NOTICE.md").write_text(
        "# Source attribution\n\nStudentLife original study: Rui Wang et al., UbiComp 2014, "
        "https://studentlife.cs.dartmouth.edu/. Converted dataset: Daniel Fryer, 2019, "
        "https://doi.org/10.5281/zenodo.3529253, published CC BY 4.0. "
        "PAM.Rds is an unmodified extracted source file; the daily CSV removes conflicting timestamp groups, "
        "collapses identical duplicates, maps picture choices and aggregates observed responses by day.\n\n"
        "PAM_mapping_SL_tbl_trans.R is an unmodified source snapshot from Daniel Fryer et al., "
        "studentlife R package version 1.1.0, GPL-3. See studentlife_DESCRIPTION and GPL-3.0.txt. "
        "Original URL: https://raw.githubusercontent.com/frycast/studentlife/master/R/SL_tbl_trans.R. "
        "No downstream project-wide license assertion is made here.\n\n"
        "Adolescent ESM: https://zenodo.org/records/17341614, CC BY 4.0, "
        "previously used by this project. Its original workbook is copied without modification; "
        "derived daily CSV uses observed negative-affect means grouped by four scheduled samples per sampling day. "
        "Full source identifiers, transformations and digests are retained in each protocol.\n", encoding="utf-8")


def documents() -> None:
    """生成评审说明、采集说明、证据边界和复算命令，不伪造待采集数据。"""
    support = json.loads((OUT / "support/results.json").read_text(encoding="utf-8"))
    newer = json.loads((OUT / "support_refined/results.json").read_text(encoding="utf-8"))
    student = json.loads((OUT / "longitudinal/studentlife_calendar_days/results.json").read_text(encoding="utf-8"))
    adolescent = json.loads((OUT / "longitudinal/adolescent_sampling_days_reused/results.json").read_text(encoding="utf-8"))
    audit = json.loads((OUT / "data/studentlife_field_audit.json").read_text(encoding="utf-8"))
    selected = student["results"]["same_person_future"]
    old, current = selected["test"]["last"], selected["test"][selected["winner"]]
    rmse_gain = 100 * (1 - current["rmse"] / old["rmse"])
    rows = []
    for mode, label in (("same_person_future", "同一用户未来日期"), ("unseen_person_future", "未见用户且未来日期")):
        r = student["results"][mode]
        for method, name in (("last", "上一日保持"), ("mean7", "最近7个观测日均值"), (r["winner"], "验证集选中 " + r["winner"])):
            m = r["test"][method]
            rows.append(f'| {label} | {name} | {m["n"]}/{m["people"]} | {m["mae"]:.4f} | {m["participant_equal_mae"]:.4f} | {m["rmse"]:.4f} |')
    primary_report = f"""# 第十四轮优化报告：跨日追踪、支持策略与独立评价

日期：2026-11-17。范围：离线算法、真实公开数据实验、最终回复对照与评价工具。音频、专业知识图谱和平台集成继续暂缓；线上风险阈值、审计规则及升级条件未修改。

## 完成结果与采用决定

1. 完成公开 StudentLife 新来源接入，完整下载230,469,489字节，发布MD5校验通过。PAM原表9,040行，冲突时间记录全组隔离后保留{audit['pam_rows']:,}行，形成{audit['daily_rows']:,}个有观测的用户日，覆盖{audit['pam_people']}个匿名ID。原研究描述48名学生，发布转换表有49个ID，这一差异未消除，因此这里报告ID数量，不认定新增一名经过核实的参与者。
2. 完成仅使用过去信息的跨日候选、简单基线、个人参考和独立运行时。同一用户未来测试的逐窗口RMSE由{old['rmse']:.4f}降到{current['rmse']:.4f}，约改善{rmse_gain:.1f}%。这是次要指标的观测改善，不代表总体效果已被证实。
3. 完成32个编写场景、四个匹配条件的128条真实本地模型最终回复，以及32个场景的探索候选。独立盲评包已准备；没有真人评分。
4. 完成成人先导采集表、24个预分配槽位、28天协议、跨日个人摘要、校验和统计命令。分配槽位不是已招募人员；真实参与者为0。

**采用决定：所有新模型和策略继续保留为离线研究候选，暂不替换默认方案。** 主指标没有形成稳定优越性；修正/目标状态的加入也没有证实回复质量更好。完整保存负结果，避免为追求单个指标而牺牲具体回应。

## 真实跨日数据与任务定义

来源：[StudentLife原项目](https://studentlife.cs.dartmouth.edu/dataset.html)、[维护者发布转换版](https://zenodo.org/records/3529253)、[维护者PAM映射源码](https://raw.githubusercontent.com/frycast/studentlife/master/R/SL_tbl_trans.R)。归档许可CC BY4.0，原始归档摘要353ae79f157097a57bff1c004634ba1b；字段审计和实际源码一起归档。

数据此前只进入项目数据目录清单，未找到此前StudentLife训练/调参脚本；本轮首次用于项目实验。无法排除通用基础模型预训练中见过公开来源。它是大学生来源自评，不是项目青少年用户或干预效果数据。

压力表转换结果只有null/timestamp/uid字段，2017行中仅122个null值为1–5数字，混有坐标字符串且缺少level字段，因此整体不用于压力预测；不能把缺失补成无压力。原压力题编码本身也不是压力从低到高排列。

改用完整得多的PAM图片选择表。按维护者16项映射转为1–4效价，按America/New_York自然日聚合；日内未响应时点不补零。此处等距编码是明确的建模约定，不是临床评分单位。记录日期{audit['calendar_start']}至{audit['calendar_end']}，尾部较稀疏，按协议保留而不事后裁剪来改善结果。

每个预测在当日结束后预测下一自然日的已观测日均效价。历史限14日，至少3个观测日，缺测超过3日重新暖启动。缺明日标签时不制造评测样本。全程均值、问卷结局、定位、未来观测不进入特征。全局日期60%/20%/20%前推；第二协议再叠加70%/15%/15%参与者隔离。参数只在训练组拟合，候选只由验证组选择，测试前保存选型锁，不用测试再挑模型。

| 协议 | 方法 | 测试窗口/ID | MAE | ID等权MAE（主指标） | RMSE |
|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

同一用户候选相对上一日保持的ID聚类MAE差95%区间为{selected['selected_minus_last_participant_mae_ci95']}，仍跨零。其ID等权MAE还略高于简单滚动均值，因此不能说超过全部基线。严格未见用户协议仅4个ID、13个窗口，选中方法退步；验证组也只有43个窗口，泛化和选型不稳定。

方向命中率只在真实变化≥1个类别间距时统计；保持法预测不变，方向分数为0是定义结果，不能据此渲染巨大提升。验证误差带复用了选型验证组，只是描述性误差范围，不提供个人覆盖保证。PAM误差不能与三模态情绪分类Macro-F1相加为一个“综合分”。

青少年ESM旧来源同时增加了按4次采样划分采样日的诊断：同人MAE由9.0410降到8.3660，未见用户由8.1420变为11.7930，仍不稳定。它已在第二轮等实验使用过；没有真实日期，下一采样日可能跨周末，不能叫新来源自然日验证。

## 个人基线和旧创新点

离线轨迹以个人前7个已观测日建立参考，跨度不超过14日；当前日不进入用于评价当前日的参考，长缺测重建，标准差设下限以免恒定历史导致除零。与固定0.3/0.1参考对照，数值漂移标记在真实数据上111/1675日，对照575/1675日；相邻小变动日的标记比例9.93%对38.60%。**这只是标记负担和稳定步代理，不是真实假阳性率；没有独立变化点标签，也不能靠更少标记证明检测更好。** 未映射到线上风险/升级。

旧三点的证据边界：RQI仍是互动/关系参与度代理，尚无独立信任或工作联盟效度；已补本人“被理解、选择被尊重、信任、不受催促”四项采集及匹配RQI字段，但这些是编写的可用性条目，不是已验证量表。MIS仍是模态表达差异，不能等同真实掩饰；音频暂缓，本轮不把其他模块结果计入其完成度。CUSUM补上个人参考、真实跨日回放及缺测处理，仍缺真实变化点和项目用户前瞻验证。

## 支持策略四臂结果

同一Qwen2.5-3B-Instruct本地NF4模型、固定种子42、确定性解码、每次最多192新token。A只有原始对话；B增加结构化本人修正；C增加目标计划；D两者都增加。四臂原始对话完全相同，且原始对话本身已经有自然语言修正和目标，因素检验的是额外结构化状态的作用，不是给A故意隐藏用户消息。场景为8类生活事件×4目标，前16开发、后16挑战；全部是工程编写情境，不是真人样本。修正内容和目标由编写者预先明确，修正通过真实apply_feedback写入；不检验系统能否从任意话语自动抽取正确反馈或目标。

调用现有SafetyAuditor，不修改规则，失败时统一最多一次重写；所有最终回复都保留并进入盲评，不能只评分通过者。本轮128条均通过该审计，不能由此推断实际安全性。实验只含预先编写的普通日常场景，没有验证危机场景。

| 条件 | 自动目标违约代理 | 旧误解明确断言代理 | 平均回复字数 |
|---|---:|---:|---:|
"""
    for arm in ("A", "B", "C", "D"):
        metric = support["summary"]["all"][arm]
        primary_report += f'| {arm} | {metric["goal_violation_proxy"]:.2%} | {metric["old_claim_assertion_proxy"]:.2%} | {metric["reply_chars"]:.2f} |\n'
    primary_report += """
自动指标仅检查问号、显式建议词和字数等表面契约。例如“可以先这样聊聊”可能被误记为建议，隐含反问也可能漏检；它不能替代语义人评。四臂没有稳定优于原始组的证据，纠正指标存在明显天花板。

探索候选E把冗长状态投影为当前有效修正和目标要求，违约时最多再生成2次，不删句拼接伪造模型回复。D的表面目标违约2/32降至E的0/32；但非暂停场景具体关切提及从16/24降到12/24，质量代价明显。候选在四臂结果曝光后开发，因此只有探索性证据；保留全部版本和单独D/E盲评包，暂不晋升。

## 独立评价与成人先导

两套盲评材料：四臂包128条、D/E包64条，各有两份空白评分表。随机展示编码与私有条件映射分开；两名独立评审完成后导入，计算适切性/忠实度、修正/目标违约、无依据细节、重复追问，以及加权κ。四臂成套完成后按事件家族聚类计算配对差异，不能把同一事件四种目标当四个独立受试者。独立性由研究人员核实、评审本人声明，软件不能证明独立。

成人先导建议24人、28天；只是可行性规模，没有功效计算，也未招募。每日自评和支持接受/完成/负担/帮助程度分开记录，所有建议都应允许拒绝。仅分析实际同意的成年人；未成年人研究另需适用的机构伦理和监护安排，本轮未开展。撤回者整体排除，不只删负反馈；原始数据删除和存储权限仍由正式实施者执行。本地CSV研究模板不宣称具备生产级鉴权和加密。

接受率以全部实际提供的机会为分母，完成率另列接受者分母；未反馈不记作有效，延迟日记单独标注，不做接受者前后变化的因果推断。若后续真正实施决策时点随机化，可参考[MRT设计原论文](https://arxiv.org/abs/2005.05880)，本轮没有实施MRT或证明策略疗效。

**截至交付：0名真实先导参与者，0份独立真人评分，0天项目用户连续追踪。** 公开历史跨日记录和本地模型生成均已实际执行；采集工具、方案和空表已完成。真人评分及连续使用结果必须在实际发生后产生，不能在本次自动执行中补成“已完成”。

## 交付与验证

全部文件位于D:/Develop/AIC/_国赛优化/round14。主要包括：longitudinal中的预注册/选型锁/全部方法预测/可调用模型；support及support_refined中的原始生成、审计和盲评包；human_study中的空采集表、分配槽位与状态；source中的代码快照；source_locked中的运行时确切版本；data中的来源与字段审计。

运行入口与评分标准见本目录《使用与复算说明.md》《独立盲评指南.md》《成人先导采集说明.md》。25项新增测试及相关既有用例共162项通过，1项既有Starlette弃用提示；详见tests.log。逐项重训和运行时回放见verification.json。首次复算因CSV默认解析引入约浮点末位差异，已改为round_trip解析并在新空目录复算，原失败日志保留。线上默认配置未替换；本轮不改变第十三轮情绪融合结果或其证据口径。
"""
    (OUT / "第十四轮优化报告-20261117.md").write_text(primary_report, encoding="utf-8")
    guide = """# 独立盲评指南

请只将support/blind或support_refined/blind完整目录交给评审，不提供同级private目录、实验结果、研究假设或条件说明。展示顺序已随机化；不能按文件顺序猜实验条件。两名评审应与开发和场景编写独立，分别评分，完成前不讨论；研究人员核实其独立性，不由模型替代评分。

读replies.json中完整对话和最后回复，分别填写rater_1.csv、rater_2.csv。每条六项指标全部填写；不确定可在notes说明，不能将空白当零分。评价所有给定最终回复，不能自行删去审计不通过或质量较差项。

- fidelity_1_5：1=歪曲或持续采用已否认解释，3=大体准确但关键关切遗漏，5=准确回应本人当前已表达的关切；中间分值用于中间程度。
- appropriateness_1_5：1=不适合当前需求，3=基本可接受但泛泛/负担较高，5=自然、具体、尊重当下目标且不过度追问。
- correction_violation：仅当把用户已纠正/否认的旧解释再次作为事实时记1；准确描述“并不是旧解释”不记违约。
- goal_violation：倾听时强行建议/追问；暂停后继续分析/建议；行动时重复探问已经说明的事项或堆多个步骤；梳理时完全不回应需求，均可记1，结合实际语义，不机械数问号。
- unsupported_details：编造未给出的经历、原因、人物动机或情绪作为事实时记1；清楚标为可能、并邀请核实的假设应结合措辞判断。
- repeat_question：要求用户再次回答已经明确给出的相同信息时记1；确实未知且必要的新问题不算重复。

不要用字数短、出现安慰词、通过程序审计等替代综合判断。上述条目是本项目研究工具，不是临床量表或治疗结局。

真实评分完成后，将rater_declarations.template.json复制为rater_declarations.json，由评审本人核对填写声明与带时区时间。模板中的false和空时间不能直接作为已完成证明。不要在模板写姓名、联系方式或身份信息；由研究人员另行保管必要的授权资料。
"""
    for directory in (OUT / "support/blind", OUT / "support_refined/blind"):
        (directory / "评分指南.md").write_text(guide, encoding="utf-8")
        template = [{"rater_id": rater, "human_completed": False, "independent_of_development": False, "declared_at": ""} for rater in ("rater_1", "rater_2")]
        path = directory / "rater_declarations.template.json"
        if not path.exists(): save(path, template)
    (OUT / "独立盲评指南.md").write_text(guide, encoding="utf-8")
    study_guide = """# 成人先导采集说明

本轮只准备材料，未招募、未联系参与者，未集成平台。allocation_slots.json中的24个P编号只是待分配槽位，没有任何真实同意。28天、24人是可行性建议，不能据此声称有足够统计功效或证明疗效。

正式实施前，研究负责人提供真实说明和自愿同意流程，明确收集字段、用途、可拒绝和退出、保管期限和删除渠道；研究机构按适用要求处理伦理和数据保护。成人先导不自动延伸到未成年人。CSV不保存真实姓名、联系方式、音视频、定位或完整私密聊天；自由情境字段可留空。限制存储权限；CSV本身没有自动加密。

consent.csv：仅在本人实际同意后写匿名ID、带时区同意时间、adult_confirmed=1、同意版本、source=real；撤回时填withdrawn_at。模型和测试账户不得产生同意记录。其他source用于测试时必须另存临时目录，不能混入真实分析。

diary.csv：真实观察时间observed_at、实际提交时间submitted_at、本人0–100心情和压力。每天一次可选自评；漏填不补零、不推断为平稳。可以有同日多次实际记录，摘要按日均值并保留次数；超过24小时补记单独标识。当前日不进入用于比较当前日的初始7日参考，超过3日缺测后重建。

strategy_events.csv：提供建议就登记offered_at及A/B/C/D条件，拒绝declined=1，接受/开始/完成各保留真实事件时间，不能根据页面渲染认定主观获益。反馈由本人实际填写helpfulness_1_5、burden_1_5和feedback_at，未填保持空白。自选接受者有偏差，帮助程度不是因果效果。

relationship_feedback.csv：同一会话/日期下保存实际运行RQI输出和本人四项反馈，检验是否相关。不得在本人未填写时自动由RQI填入信任分。四项为编写的可用性评价，相关性也不等于真实信任或工作联盟已被验证。

四条件体验采用ABDC、BCAD、CDBA、DACB平衡顺序，目的是减轻顺序影响；现场按真实同意后的槽位分配，四次体验可用不同情境，不能当作已经进行的随机干预研究。持续日记与四次体验分别报告；不对28天默认持续主动打扰。无反馈、退出和未完成都保留各自分母。

分析命令会检查真实性声明、成人同意、时区、先后顺序、数值范围、重复ID、撤回和28天窗口。它不能核验人的身份或代替伦理程序。撤回会从分析整体排除；研究负责人另行完成原始文件删除，不能只删不利反馈。不得把测试生成的CSV作为真人实测成果。
"""
    (OUT / "成人先导采集说明.md").write_text(study_guide, encoding="utf-8")
    usage = r"""# 使用与复算说明

项目根目录：D:\Develop\AIC\AI-mental-main (2)\AI-mental-main\AI-mental-main
Python：D:\Develop\AIC\.venv\Scripts\python.exe
输出：D:\Develop\AIC\_国赛优化\round14

在项目根目录的PowerShell设置：

```powershell
$env:PYTHONPATH='algorithm'
$env:PYTHONIOENCODING='utf-8'
```

运行时逐项回放（163个窗口，来源量表严格受限）：

```powershell
& 'D:\Develop\AIC\.venv\Scripts\python.exe' -m experiments.round14_runtime
```

从冻结源码重训全部四个模型选择协议并与所有方法预测比对，输出目录必须是独立空目录：

```powershell
& 'D:\Develop\AIC\.venv\Scripts\python.exe' -m experiments.round14_reproduce --output 'D:\Develop\AIC\_国赛优化\round14_reproduction_user'
```

预测器也支持 --history JSON文件，字段source_schema必须为studentlife_pam_valence_1_4，observations包含同一个person、递增整数day和1–4的value，target_day必须等于最后观测日+1。见runtime_input_example.json。历史不足/长间隔会拒绝预测；项目0–100心情量表不自动套用PAM模型。

真实四臂评分完成后（默认空表只会报告待采集）：

```powershell
& 'D:\Develop\AIC\.venv\Scripts\python.exe' -m experiments.round14_human ratings
& 'D:\Develop\AIC\.venv\Scripts\python.exe' -m experiments.round14_human ratings --directory 'D:\Develop\AIC\_国赛优化\round14\support_refined\blind' --key 'D:\Develop\AIC\_国赛优化\round14\support_refined\private\arm_key.json' --output 'D:\Develop\AIC\_国赛优化\round14\support_refined\independent_ratings_analysis.json'
```

实际采集同意和日记以后：

```powershell
& 'D:\Develop\AIC\.venv\Scripts\python.exe' -m experiments.round14_human study
& 'D:\Develop\AIC\.venv\Scripts\python.exe' -m experiments.round14_tracking --participant P001
```

当前没有P001的真实同意，第二条应拒绝而非制造追踪结果。需要初始化另一个独立空研究目录时可用 round14_human init --directory 路径；init不等同真实同意。

本地模型生成可通过round14_support和round14_support_refine执行，依赖本地Qwen2.5-3B-Instruct-nf4权重（不打包多GB模型），具体参数与模型指纹见preregister/model_fingerprints.json。正式四臂已有逐项JSONL续跑，冻结协议变化会拒绝覆盖；探索候选结果不要误称未曝光的独立确认集。模型生成可能受硬件/库版本影响，以保存的最终回复为盲评输入。

测试：algorithm/tests/test_round14_protocol.py及tests.log中列出的相关既有用例。只进行可复算研究，不更改线上安全阈值、危机升级、模型默认路径或平台流程。
"""
    (OUT / "使用与复算说明.md").write_text(usage, encoding="utf-8")
    example = json.loads((OUT / "longitudinal/studentlife_calendar_days/same_person_future_test_windows.json").read_text(encoding="utf-8"))[0]
    save(OUT / "runtime_input_example.json", {"source_schema": "studentlife_pam_valence_1_4", "data_origin": "public StudentLife heldout replay, not a project user",
        "target_day": example["target_day"], "observations": [{"person": example["person"], "day": day, "value": 1 + 3 * value}
            for day, value in zip(example["history_days"], example["history"], strict=True)]})


def finish() -> None:
    """汇总真实状态、测试和版本指纹；打包在外层命令完成复算后执行。"""
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot(); documents()
    first = analyze_ratings(OUT / "support/blind", OUT / "support/private/arm_key.json", OUT / "support/independent_ratings_analysis.json")
    second = analyze_ratings(OUT / "support_refined/blind", OUT / "support_refined/private/arm_key.json", OUT / "support_refined/independent_ratings_analysis.json")
    study = analyze_study(OUT / "human_study", OUT / "human_study/analysis.json")
    save(OUT / "runtime_verification.json", verify_artifacts())
    save(OUT / "environment.json", {"python": sys.version, "packages": {name: importlib.metadata.version(name) for name in
        ("numpy", "pandas", "scikit-learn", "torch", "transformers", "bitsandbytes", "rdata", "pytest", "requests", "joblib")}})
    from experiments.round14_support import MODEL
    save(OUT / "model_fingerprints.json", {"path": str(MODEL), "files": {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(MODEL.iterdir()) if path.is_file() and path.suffix in {".safetensors", ".json", ".txt"}},
        "generation": "real local model, deterministic NF4; saved replies are authoritative for blind review"})
    changed = subprocess.run(["git", "diff", "--name-only"], cwd=REPO, capture_output=True, text=True, check=True).stdout.splitlines()
    protected = ["algorithm/assessment/tamer_framework.py", "algorithm/assessment/personal_baseline.py", "algorithm/intervention/auditor.py",
        "algorithm/intervention/support_understanding.py", "algorithm/shared/dataclasses.py", "algorithm/config/audit_rules.yaml", "algorithm/config/baseline.yaml"]
    save(OUT / "scope_and_promotion.json", {"changed_tracked_files_including_preexisting": changed,
        "protected_files_unchanged_relative_to_git": {name: name not in changed for name in protected},
        "protected_current_sha256": {name: sha256(REPO / name) for name in protected},
        "real_pilot_participants": study["consented_adults"], "independent_ratings_completed": first["completed_rows"] + second["completed_rows"],
        "audio_kg_platform": "deferred", "production_changed": False, "new_models_promoted": False,
        "reason": "primary MAE confidence interval crosses zero, unseen-user regression and content-specificity tradeoff; actual human ratings pending"})
    print(json.dumps({"report": str(OUT / "第十四轮优化报告-20261117.md"), "pilot_participants": study["consented_adults"], "runtime_verified": True}), flush=True)


def package() -> None:
    """生成逐文件摘要和ZIP，拒绝有任何摘要不一致的交付包。"""
    files = sorted(path for path in OUT.rglob("*") if path.is_file() and "__pycache__" not in path.parts and path.name != "manifest.json")
    manifest = {str(path.relative_to(OUT)).replace("\\", "/"): {"sha256": sha256(path), "bytes": path.stat().st_size} for path in files}
    save(OUT / "manifest.json", manifest)
    destination = ROOT / "第十四轮优化交付包-20261117.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [*files, OUT / "manifest.json"]: archive.write(path, str(path.relative_to(OUT)))
    with zipfile.ZipFile(destination) as archive:
        import hashlib
        for name, item in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != item["sha256"]: raise ValueError("交付归档摘要不一致")
    print(json.dumps({"zip": str(destination), "files": len(manifest), "sha256": sha256(destination), "bytes": destination.stat().st_size}), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--package", action="store_true")
    args = parser.parse_args(); package() if args.package else finish()
