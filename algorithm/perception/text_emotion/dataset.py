"""
扩充版中文青少年心理健康情感数据集

覆盖 5 类情绪：anxiety / depression / anger / neutral / positive
每类 80+ 条多样化样本，模拟青少年真实表达风格。
总计 420+ 条样本。

使用方式：
    from perception.text_emotion.dataset import generate_fake_data
    path = generate_fake_data(num_per_class=80)
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from perception.text_emotion.config import (
    DATA_DIR,
    LABEL2ID,
    MAX_LENGTH,
    MODEL_NAME,
    SEED,
)


# ============================================================
# 扩充版情感语料库（青少年视角）
# ============================================================

_FAKE_SAMPLES = {
    "anxiety": [
        "我最近总是感到很不安，担心各种事情",
        "明天就要考试了，我好紧张好焦虑",
        "总觉得有什么不好的事情要发生",
        "心跳好快，手心都在出汗",
        "晚上翻来覆去睡不着，脑子里全是各种担忧",
        "工作压力太大了，每天都提心吊胆的",
        "感觉自己控制不了自己的情绪，好害怕",
        "最近总是莫名地紧张，不知道怎么办",
        "下周要演讲了，一想到就紧张得不行",
        "妈妈又要来学校了，我好怕她跟老师说什么",
        "最近成绩一直在下降，好怕爸妈失望",
        "总觉得同学们在我背后说我坏话",
        "每次上课被点名回答问题就心跳加速",
        "晚上总是做噩梦，醒来一身冷汗",
        "手机一响就紧张，怕是坏消息",
        "感觉呼吸有点困难，是不是身体出了问题",
        "明天要见新同学了，好怕融入不进去",
        "每次考试前都会肚子疼",
        "总是忍不住检查门锁好了没有",
        "上课的时候突然感觉头晕，好怕晕倒",
        "好朋友最近跟我疏远了，是不是我做错了什么",
        "感觉事情总是在往坏的方向发展",
        "每次出门前都要反复检查好几遍",
        "心跳突然变得很快，感觉自己要猝死",
        "总是担心家人出事",
        "一到学校就感觉浑身不自在",
        "最近总是咬指甲，停不下来",
        "想到未来就很迷茫很焦虑",
        "上课注意力完全集中不了，一直在走神",
        "感觉脑子里有好多声音在吵",
        "每次跟陌生人说话就紧张到手抖",
        "晚上不敢一个人待着",
        "总觉得有人在监视我",
        "最近开始掉头发了，好担心",
        "每次坐电梯都害怕电梯会掉下去",
        "感觉胸口闷闷的，喘不上气",
        "好怕自己得了什么严重的病",
        "一闭眼就想到那些可怕的事情",
        "最近特别怕黑，以前不会这样的",
        "总是担心说错话得罪人",
    ],
    "depression": [
        "我觉得活着没什么意思",
        "今天心情很低落，什么都不想做",
        "感觉自己很没用，什么都做不好",
        "好想一个人待着，谁都不想见",
        "每天都觉得很疲惫，没有精力",
        "对什么都提不起兴趣了",
        "眼泪控制不住地流下来",
        "感觉世界都是灰色的",
        "不想上学，不想见任何人",
        "以前喜欢的事情现在都不想做了",
        "感觉心里空空的，什么感觉都没有",
        "每天都在假装开心，好累",
        "觉得自己是多余的",
        "不想吃饭，什么都吃不下",
        "上课的时候一直在发呆",
        "感觉没有人真正理解我",
        "有时候会想如果我消失了会有人在意吗",
        "好想把所有的社交软件都删掉",
        "感觉自己像一个行尸走肉",
        "总是觉得自己不够好",
        "连起床都觉得好困难",
        "不想跟任何人说话",
        "感觉被困住了，看不到出路",
        "最近经常一个人偷偷哭",
        "对未来的事情完全不期待",
        "感觉自己拖累了所有人",
        "好想找个人倾诉但又不知道说什么",
        "每天都在数着日子过",
        "感觉快乐离我很遥远",
        "以前开心的回忆现在想起来只会更难过",
        "不想努力了，反正也没什么用",
        "感觉全世界都抛弃了我",
        "连最喜欢的游戏都不想玩了",
        "总是觉得自己不配拥有好的东西",
        "感觉身体很沉重，像灌了铅一样",
        "不想出门，外面的世界跟我没关系",
        "有时候觉得睡着了比醒着好",
        "感觉自己的存在没有任何意义",
        "好想像小说里那样消失掉",
        "连微笑都觉得好费力",
    ],
    "anger": [
        "他们凭什么这样对我，太过分了",
        "我真的受够了，每次都这样",
        "气死我了，怎么可以这样不公平",
        "我再也不想理他了，太让人生气",
        "为什么总是我吃亏，太气人了",
        "这简直不可理喻",
        "我的忍耐是有限度的",
        "太欺负人了，我要讨个说法",
        "凭什么他可以我不行，太不公平了",
        "我爸又不分青红皂白就骂我",
        "老师偏心偏得太明显了",
        "好朋友居然在背后说我坏话，气死了",
        "他们把我的东西弄坏了还不道歉",
        "为什么大人总觉得是小孩的错",
        "我真的想打人",
        "每次都让我让步，凭什么",
        "太过分了，我要举报",
        "他们根本不尊重我",
        "说了多少次了就是不听，气死了",
        "感觉被利用了，好愤怒",
        "凭什么我要承受这些",
        "我真的想摔东西",
        "他们把我的秘密说出去了",
        "太虚伪了，当面一套背后一套",
        "明明是他的错还要我道歉",
        "我感觉被背叛了",
        "为什么没人站在我这边",
        "我真的忍无可忍了",
        "他们以为我好欺负吗",
        "每次都是我在付出，没人感激",
        "太恶心了，这种人怎么不去死",
        "我真的好恨他们",
        "为什么世界这么不公平",
        "我恨不得把所有东西都砸了",
        "他们根本不把我当回事",
        "凭什么限制我的自由",
        "我受够了被当成小孩对待",
        "他们永远不理解我",
        "真的好想离家出走",
        "为什么受伤的总是我",
    ],
    "neutral": [
        "今天天气还不错",
        "我早上吃了一碗面",
        "明天打算去图书馆看书",
        "这个周末没什么特别的安排",
        "刚看完一部电影",
        "下午开了个会",
        "最近在学一门新课程",
        "今天和平常差不多",
        "中午吃了食堂的红烧肉",
        "放学回家了",
        "今天上了六节课",
        "作业还没写完",
        "周末打算整理一下房间",
        "最近在看一本小说",
        "今天公交车比平时晚了几分钟",
        "同桌换了个新发型",
        "晚上准备早点睡",
        "下个月有期中考试",
        "今天去了趟超市",
        "手机快没电了",
        "明天要交数学作业",
        "最近在追一个综艺",
        "今天体育课跑了八百米",
        "晚上吃了饺子",
        "周末可能去逛街",
        "最近换了一个新的笔袋",
        "今天老师布置了很多作业",
        "放学跟同学一起走的",
        "准备洗个澡然后睡觉",
        "今天没什么特别的事",
        "早上被闹钟吵醒了",
        "中午在教室午休",
        "下午有一节自习课",
        "今天气温降了几度",
        "最近想买一双新鞋",
        "晚上跟妈妈打了电话",
        "今天食堂的菜还行",
        "准备去接水喝",
        "明天记得带伞",
        "今天作业写完了",
    ],
    "positive": [
        "今天心情特别好，阳光很温暖",
        "终于完成了项目，好开心",
        "和朋友出去玩了一整天，太愉快了",
        "感觉自己进步了很多，很有成就感",
        "生活越来越好了，充满希望",
        "收到了好消息，开心得不得了",
        "感恩身边的一切，好幸福",
        "今天被表扬了，继续努力",
        "考试进步了十名，太棒了",
        "交到新朋友了，好开心",
        "今天吃到了超好吃的蛋糕",
        "终于把难题解出来了，有成就感",
        "周末去了一个很好玩的地方",
        "今天被同学夸了，好害羞好开心",
        "感觉最近状态特别好",
        "新学的歌终于学会了",
        "今天天气好适合出去走走",
        "追的剧更新了，好期待",
        "感觉自己越来越自信了",
        "今天帮了一个同学，感觉很好",
        "爸妈今天带我吃了火锅",
        "终于把房间收拾干净了，心情舒畅",
        "今天体育课拿了第一名",
        "收到了一份意外的礼物",
        "感觉未来充满了可能性",
        "今天笑了很多次",
        "和好朋友聊了很久，很开心",
        "发现了一家很好吃的店",
        "今天学到了新东西，好满足",
        "感觉自己被爱包围着",
        "今天一切都刚刚好",
        "心情美美的，哼着歌回家",
        "今天被老师点名表扬了",
        "终于把欠的作业补完了，轻松",
        "今天阳光照进教室，好舒服",
        "和同桌吵完架又和好了",
        "感觉自己越来越好了",
        "今天运气特别好",
        "买到了想要的东西",
        "今天过得很充实",
    ],
}


def generate_fake_data(num_per_class: int = 80, output_path: Optional[str] = None) -> str:
    """生成扩充版情感训练数据

    通过模板变体 + 随机组合生成更多样化的样本。

    Args:
        num_per_class: 每个类别生成的样本数量
        output_path: 输出 JSON 文件路径

    Returns:
        生成的文件路径
    """
    rng = random.Random(SEED)
    samples = []

    # 前缀/后缀修饰词，增加多样性
    prefixes = ["", "嗯...", "哎，", "唉，", "说实话，", "其实，", "跟你说，", "不知道为什么，"]
    suffixes = ["", "。", "...", "。", "。", "。"]
    intensifiers = ["很", "非常", "特别", "超级", "好", "真的", "太"]

    for label, base_texts in _FAKE_SAMPLES.items():
        # 先加入所有原始样本
        for text in base_texts:
            samples.append({"text": text, "label": label})

        # 通过变体生成更多样本
        current_count = len(base_texts)
        needed = num_per_class - current_count

        for _ in range(max(0, needed)):
            base = rng.choice(base_texts)
            prefix = rng.choice(prefixes)
            suffix = rng.choice(suffixes)

            # 随机替换强度词
            if rng.random() < 0.3:
                for old_word in ["很", "好", "非常", "特别"]:
                    if old_word in base:
                        new_word = rng.choice(intensifiers)
                        base = base.replace(old_word, new_word, 1)
                        break

            variant = prefix + base + suffix
            samples.append({"text": variant, "label": label})

    rng.shuffle(samples)

    if output_path is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(DATA_DIR / "fake_train.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    return output_path


# ============================================================
# EmotionDataset
# ============================================================

class EmotionDataset(Dataset):
    """情感分类数据集

    从 JSON 文件加载数据，使用 RoBERTa tokenizer 进行编码。

    Args:
        data_path: JSON 数据文件路径
        tokenizer: HuggingFace tokenizer 实例
        max_length: 最大序列长度
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = MAX_LENGTH,
    ) -> None:
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.texts = [item["text"] for item in raw_data]
        self.labels = [LABEL2ID[item["label"]] for item in raw_data]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def load_tokenizer() -> AutoTokenizer:
    """加载 RoBERTa 分词器"""
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def compute_class_weights(dataset: EmotionDataset) -> torch.Tensor:
    """计算类别权重，用于加权 CrossEntropyLoss 处理类别不均衡

    权重公式：weight_i = total_samples / (num_classes * count_i)

    Args:
        dataset: 训练数据集

    Returns:
        形状为 (num_classes,) 的权重张量
    """
    from collections import Counter

    label_counts = Counter(dataset.labels)
    total = len(dataset.labels)
    num_classes = len(LABEL2ID)

    weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 1)
        weights.append(total / (num_classes * count))

    return torch.tensor(weights, dtype=torch.float)
