"""可直接插入答辩页的研究流程图，明确不同任务与训练边界。"""
from __future__ import annotations
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.font_manager import FontProperties
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
from experiments.round11_common import OUTPUT


def main() -> None:
    """框图只表示实际实现的路径，不放未经测量的示意分数。"""
    font=FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
    fig,ax=plt.subplots(figsize=(15,7.2),layout="constrained");ax.set_xlim(0,15);ax.set_ylim(0,7.2);ax.axis("off")
    def box(x: float,y: float,w: float,h: float,text: str,color: str = "#EAF2FA",size: int = 13) -> None:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.07,rounding_size=0.1",fc=color,ec="#3B689A",lw=1.4))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontproperties=font,fontsize=size,color="#173A61",linespacing=1.55)
    def arrow(x1: float,y1: float,x2: float,y2: float,color: str = "#3B689A") -> None:
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=16,color=color,lw=1.6))
    ax.text(.1,6.85,"从可靠观测到可重放的当前识别与未来预测",fontproperties=font,fontsize=24,color="#173A61",weight="bold")
    ax.text(.15,6.2,"主任务 M3：当前整体负向 / 中性 / 正向",fontproperties=font,fontsize=16,color="#245C94")
    box(.2,3.5,2.5,2.2,"真实观测\n转录 T · 音轨 A · 图像 V\n解码时间戳 / 质量 / 缺失",size=12)
    for j,label in enumerate(("整体专家  p_M","文字专家  p_T","语音专家  p_A","视觉专家  p_V")):
        box(3.3,5.12-j*.54,2.9,.4,label,size=12)
    ax.text(4.75,3.12,"各自官方监督 → 同一极性概率空间",ha="center",fontproperties=font,fontsize=11,color="#55718B")
    box(6.9,3.7,3.,1.8,"40维可核验证据\n概率 · 强度 · 冲突 · 可用性\n来源内层OOF训练融合器",size=12)
    box(10.65,3.7,3.7,1.8,"非线性证据融合 + 固定决策\n输出当前 M3\n缺字时退化到可用 A / V",color="#E5F4EF",size=12)
    arrow(2.8,4.6,3.2,4.6);arrow(6.25,4.6,6.8,4.6);arrow(10.,4.6,10.55,4.6)
    ax.text(.15,2.7,"时序任务 M7：未来同说话人七类情绪（跨度1 / 3 / 5）",fontproperties=font,fontsize=16,color="#1A8077")
    box(.2,.8,2.6,1.4,"当下 T + A 感知\n七类识别概率\n训练历史由来源OOF生成",color="#E5F4EF",size=11)
    box(3.4,.8,3.,1.4,"只使用已完成历史\n上下文5 / 10 / 20\n过去情绪是模型观测",color="#E5F4EF",size=11)
    box(7.,.8,2.9,1.4,"截止线与未来起点分离\n真实音轨时长 + 发布时标\n提前余量0.25 / 1.25秒",color="#FFF3DA",size=11)
    box(10.65,.8,3.7,1.4,"开发锁定历史模型 / GRU / 投票\n未来 M7 概率\n逐窗口结果与原媒体重放",color="#E5F4EF",size=11)
    arrow(2.9,1.5,3.3,1.5,"#1A8077");arrow(6.5,1.5,6.9,1.5,"#1A8077");arrow(10.,1.5,10.55,1.5,"#1A8077")
    ax.text(.2,.2,"评测边界：开发选型与测试评分分离；M3 / A3 / 未来M7单列；保留失败对照和原始样本。",fontproperties=font,fontsize=11,color="#55718B")
    folder=OUTPUT/"figures";folder.mkdir(exist_ok=True)
    fig.savefig(folder/"evidence_chain_architecture.png",dpi=220);fig.savefig(folder/"evidence_chain_architecture.svg");plt.close(fig)


if __name__=="__main__":main()
