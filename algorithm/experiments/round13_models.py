"""保留联合语义的可用性专用分类器，知识特征与OOF软蒸馏使用独立入口。"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any
import numpy as np
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from experiments.round13_data import DATA

NAMES = ("T", "A", "V")
CONDITIONS = ("clean", "missing_A", "missing_V", "missing_T", "text_only", "audio_mismatch", "all_missing")


class SentimentLexicon:
    """NTUSD原词典的最长词匹配、显式否定和转折特征，不是个体因果知识。"""

    def __init__(self, directory: Path = DATA / "ntusd") -> None:
        mapping = {}
        for line in (directory / "TSCharacters.txt").read_text("utf-8-sig").splitlines():
            if line and not line.startswith("#") and "\t" in line:
                left, right = line.split("\t", 1)
                mapping[left] = right.split()[0]
        self.translation = str.maketrans(mapping)
        terms: dict[str, list[int]] = {}
        for file, value in (("positive.txt",1),("negative.txt",-1)):
            for word in (directory/file).read_bytes().decode("big5").splitlines():
                word = word.strip().translate(self.translation)
                if len(word) >= 2:
                    terms.setdefault(word, []).append(value)
        self.words = {k: int(np.sign(sum(v))) for k,v in terms.items() if sum(v) != 0}
        self.length = min(16, max(map(len, self.words)))

    def encode(self, texts: np.ndarray) -> np.ndarray:
        """生成固定12维词典证据，完全不拟合数据集情绪标签。"""
        output = np.zeros((len(texts), 12), np.float32)
        negations = ("不", "没", "无", "未", "别", "非", "莫")
        for i, text in enumerate(texts.astype(str)):
            text = text.translate(self.translation)
            pivot = max([text.rfind(w) for w in ("但是", "可是", "然而", "不过", "却")] + [-1])
            pos = 0
            while pos < len(text):
                found = None
                for n in range(min(self.length, len(text)-pos), 1, -1):
                    word = text[pos:pos+n]
                    if word in self.words:
                        found = word
                        break
                if found is None:
                    pos += 1
                    continue
                polarity = self.words[found]
                output[i, 0 if polarity > 0 else 1] += 1
                prefix = text[max(0, pos-3):pos]
                negated = sum(prefix.count(w) for w in negations) % 2
                corrected = polarity * (-1 if negated else 1)
                output[i, 2 if corrected > 0 else 3] += 1
                output[i, 4] += negated
                if pivot >= 0 and pos > pivot:
                    output[i, 5 if corrected > 0 else 6] += 1
                output[i, 7] += len(found)
                pos += len(found)
            output[i, 7] /= max(1, len(text))
            output[i, 8] = output[i, 2] - output[i, 3]
            output[i, 9] = float(output[i, :2].sum() > 0)
            output[i, 10] = float(pivot >= 0)
            output[i, 11] = np.log1p(len(text)) / 5
        return output


def normalized_texts(texts: np.ndarray) -> np.ndarray:
    """仅用于重复核查，不改变模型原文输入。"""
    return np.array(["".join(c.casefold() for c in str(t) if c.isalnum()) for t in texts])


def observed(data: dict[str, np.ndarray]) -> np.ndarray:
    """特征存在和媒体质量共同决定真实可用性。"""
    mask = np.column_stack([np.isfinite(data[n]).all(1) & np.any(np.nan_to_num(data[n]) != 0,axis=1) for n in NAMES])
    if "mask" in data:
        mask &= data["mask"]
    return mask


def modify(data: dict[str, np.ndarray], condition: str, donors: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """条件改变的是观测；原金标留在独立评价入口。"""
    result = {k:v.copy() for k,v in data.items()}
    result["mask"] = observed(data)
    for j in {"missing_T":(0,),"missing_A":(1,),"missing_V":(2,),"text_only":(1,2),"all_missing":(0,1,2)}.get(condition,()):
        result["mask"][:,j] = False
        result[NAMES[j]][:] = 0
    if condition == "audio_mismatch":
        if donors is None:
            raise ValueError("错配必须提供固定供体")
        result["A"] = data["A"][donors].copy()
        result["mask"][:,1] = observed(data)[donors,1]
    return result


def donor_indices(groups: np.ndarray) -> np.ndarray:
    """固定种子只按不同来源选择供体，不查看类别。"""
    rng = np.random.default_rng(42)
    result = np.empty(len(groups),int)
    for i in range(len(groups)):
        candidates = np.flatnonzero(groups != groups[i])
        if not len(candidates):
            raise ValueError("缺少不同来源音轨供体")
        result[i] = rng.choice(candidates)
    return result


class Projection:
    """各模态仅在训练观测拟合标准化与PCA，保留联合而非仅概率输入。"""

    def fit(self, data: dict[str, np.ndarray]) -> Projection:
        self.members = {}
        mask = observed(data)
        for j, name in enumerate(NAMES):
            active = mask[:,j]
            if not active.any():
                self.members[name] = None
                continue
            value = np.asarray(data[name][active], np.float64)
            scaler = StandardScaler().fit(value)
            standardized = scaler.transform(value)
            pca = None
            if value.shape[1] > 96:
                pca = PCA(n_components=min(96,len(value)-1), svd_solver="randomized",random_state=42).fit(standardized)
            self.members[name] = (scaler,pca)
        return self

    def transform(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """缺失位置先用训练均值替代，投影后严格归零。"""
        mask = observed(data)
        result = {}
        for j, name in enumerate(NAMES):
            member = self.members[name]
            if member is None:
                result[name] = np.zeros((len(mask),1))
                continue
            scaler,pca = member
            value = scaler.transform(np.where(mask[:,j:j+1],data[name],scaler.mean_))
            if pca is not None:
                value = pca.transform(value)
            result[name] = np.where(mask[:,j:j+1],value,0.) / np.sqrt(scaler.n_features_in_)
        result["mask"] = mask
        return result


def configurations() -> list[dict[str, Any]]:
    """有限预定义选型，知识及打乱对照同维同预算。"""
    return [dict(id=name, kind=kind, C=c, knowledge=k, shuffled=s) for name,kind,c,k,s in (
        ("linear_1","linear",1.,False,False), ("rbf_1","rbf",1.,False,False),
        ("rbf_5","rbf",5.,False,False), ("boost","boost",1.,False,False),
        ("rbf_knowledge","rbf",5.,True,False), ("rbf_knowledge_shuffle","rbf",5.,True,True),
        ("boost_knowledge","boost",1.,True,False), ("boost_knowledge_shuffle","boost",1.,True,True))]


def matrix(projected: dict[str, np.ndarray], pattern: int, knowledge: np.ndarray | None) -> np.ndarray:
    """一个可用性分支只消费该分支确实可见的模态。"""
    parts = [projected[n] for j,n in enumerate(NAMES) if pattern & (1<<j)]
    parts.append(projected["mask"].astype(float)[:, [j for j in range(3) if pattern & (1<<j)]])
    if knowledge is not None and pattern & 1:
        parts.append(knowledge / 5)
    return np.column_stack(parts)


def classifier(config: dict[str, Any]) -> Any:
    """统一固定种子与类别平衡，返回可序列化的整体M3分类头。"""
    if config["kind"] == "linear":
        return LogisticRegression(C=config["C"],max_iter=1000,class_weight="balanced",random_state=42)
    if config["kind"] == "rbf":
        return SVC(C=config["C"],kernel="rbf",class_weight="balanced",probability=True,random_state=42)
    return CatBoostClassifier(iterations=350,depth=4,learning_rate=.04,loss_function="MultiClass",
        auto_class_weights="SqrtBalanced",random_seed=42,thread_count=2,verbose=False,allow_writing_files=False)


def fit_heads(projected: dict[str,np.ndarray], labels: np.ndarray, config: dict[str,Any],
              knowledge: np.ndarray, patterns: list[int] | None = None) -> dict[str, Any]:
    """为所有有意义的可用性模式训练监督M头，不把T/A/V单模态金标混用。"""
    mask = projected["mask"]
    full = sum(1<<j for j in range(3) if mask[:,j].any())
    patterns = list(range(1,full+1)) if patterns is None else patterns
    models = {}
    k = knowledge if config["knowledge"] else None
    if config["shuffled"]:
        k = knowledge[np.random.default_rng(42).permutation(len(knowledge))]
    for pattern in patterns:
        if pattern & ~full:
            continue
        x = matrix(projected,pattern,k)
        models[pattern] = classifier(config).fit(x,labels)
    return dict(models=models,full=full,config=config,prior=np.bincount(labels,minlength=3)/len(labels))


def probabilities(head: Any, x: np.ndarray) -> np.ndarray:
    """显式排列负/中/正类别，避免某类缺席造成隐式列错位。"""
    p = np.zeros((len(x),3))
    p[:,np.asarray(head.classes_,int)] = head.predict_proba(x)
    return p


def predict_heads(bundle: dict[str,Any], projected: dict[str,np.ndarray], knowledge: np.ndarray,
                  specialized: bool = True) -> np.ndarray:
    """选择真实可用分支，全缺失返回训练先验；无需金标或来源ID。"""
    masks = projected["mask"]
    pattern = masks @ np.array([1,2,4])
    result = np.tile(bundle["prior"],(len(masks),1))
    k = knowledge if bundle["config"]["knowledge"] else None
    if bundle["config"]["shuffled"]:
        k = knowledge[np.random.default_rng(42).permutation(len(knowledge))]
    for bits in np.unique(pattern):
        if not bits:
            continue
        active = pattern == bits
        chosen = int(bits) if specialized else bundle["full"]
        x = matrix(projected,chosen,k)
        result[active] = probabilities(bundle["models"][chosen],x[active])
    return result
