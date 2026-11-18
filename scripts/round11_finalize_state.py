"""将最终交付事实写到交接记录顶部，旧阶段状态保留为历史。"""
from __future__ import annotations
from datetime import datetime
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
from experiments.round11_common import OUTPUT


def main() -> None:
    """只在真实结果、部署基准、重放都存在后写最终状态。"""
    results={name:json.loads((OUTPUT/name).read_text("utf-8")) for name in
        ("historical_diagnostic/results.json","external_m3/results.json","runtime_benchmark/results.json",
         "runtime_staged/results.json","forecast_replay/results.json","validation/final_replay.json","quality_strata.json")}
    main_result=results["historical_diagnostic/results.json"]
    path=OUTPUT/"RUN_STATE.md";old=path.read_text("utf-8")
    header=f'''# 第十一轮最终交付状态（覆盖下方历史阶段记录）

{datetime.now().astimezone().isoformat(timespec="seconds")}：完整实验、锁定测试、实际推理成本、独立重训/原媒体重放均已完成。最终导航为REVIEW-PACKAGE.md、完整优化与独立验证报告及答辩讲稿，不再按下方过时“待做”清单启动新搜索。

- 原821五元头种子均值：第十轮{main_result['round10_mean_macro_f1']:.8f}→第十一轮7B{main_result['7b']['mean_macro_f1']:.8f}；相对变化{main_result['main_relative_gain_vs_round10']:+.2%}。
- 最初10%目标参照第九轮0.6388；累计变化与相对第十轮新增变化分开汇报，不用未来M7替代M3主验收。
- 外部M3的1929条、未来M7全部6个设置均已锁定后评测，禁止再依据这些测试成绩循环选型。
- 全2876真实波形SNR20/10及干净重算、缺失/错配/低质量分层已保存。
- 7B/3B/light五元头种子模型、逐条概率、固定正负案例、三档真实媒体成本、独立重训与无标签重放均有工件；第十轮126项摘要不变。
- 依照用户“一口气全部完成”执行；下方仅为历史档案。

---

'''
    if not old.startswith("# 第十一轮最终交付状态"):path.write_text(header+old,"utf-8")
    live=OUTPUT/"LIVE-DELIVERY-STATE.md"
    if live.exists():
        archive=OUTPUT/"source/runtime_handoffs/live-20261117-0008.md"
        archive.parent.mkdir(parents=True,exist_ok=True)
        if not archive.exists():archive.write_text(live.read_text("utf-8"),"utf-8")
        live.write_text("# 最终运行记录导航\n\n计算与验证已结束。最终报告、完成度和模型结果以 [REVIEW-PACKAGE.md](REVIEW-PACKAGE.md) 为准。历史运行中快照归档于source/runtime_handoffs。\n","utf-8")


if __name__=="__main__":main()
