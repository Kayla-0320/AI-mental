"""明确标记合成的多用户支持状态与时间序列压力实验；不代替真人研究。"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"algorithm"))
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from intervention.support_understanding import prepare_support,apply_feedback,parse_state
from shared import SupportFeedback
from perception.knowledge_moe.features import save_json,sha256_file


def support_simulation(count: int=120,turns: int=30) -> dict[str,Any]:
    """对实际支持状态实现运行生成交互、修正、退出、重试和用户隔离。"""
    rng=np.random.default_rng(42);checks=0;failures=[];traces=[]
    phrases={"listen":["先听我说","不用给我建议","只想被倾听"],
             "understand":["帮我理清这件事","我想被理解","一起梳理一下"],
             "act":["帮我想办法","我想知道怎么做","给我一个具体建议"]}
    for user in range(count):
        sid=f"SYNTHETIC-user-{user:04d}";state=None;history=[]
        for turn in range(turns):
            goal=list(phrases)[int(rng.integers(3))]
            request=phrases[goal][int(rng.integers(3))]
            quoted='朋友说“帮我想办法，不想说了”，' if turn%4==0 else ''
            pause=turn%13==12
            text=f"今天考试后我有些担心，{quoted}{request}"+('，但先暂停一下。' if pause else '。')
            source=f"{sid}/message-{turn}";old=asdict(state) if state else None
            state=prepare_support(sid,source,text,history,old)
            checks+=1
            if state.goal!=goal:failures.append({"user":sid,"turn":turn,"kind":"goal","expected":goal,"actual":state.goal})
            allowed=({"respect_pause"} if pause else {"listen"} if state.declined or goal=="listen" else
                     {"reflect_check","explore"} if goal=="understand" else {"collaborate_action"})
            checks+=1
            if state.plan.action not in allowed:failures.append({"user":sid,"turn":turn,"kind":"plan","actual":state.plan.action,"allowed":sorted(allowed)})
            history.append({"id":source,"role":"user","content":text})
            again=prepare_support(sid,source,text,history,asdict(state));checks+=1
            if again.revision!=state.revision:failures.append({"user":sid,"turn":turn,"kind":"idempotence"})
            for item in state.items:
                if item.origin=="user_message":
                    original=next((row["content"] for row in history if row["id"]==item.source_id),None)
                    checks+=1
                    if original is None or original[item.start:item.end]!=item.quote:failures.append({"user":sid,"turn":turn,"kind":"provenance"})
            if turn%7==0 and state.items:
                target=next((item for item in state.items if item.status!="rejected"),None)
                if target:
                    state=apply_feedback(state,SupportFeedback(f"{sid}/f-{turn}",state.revision,"correct",target.id,"我只是想确认朋友是否看到了消息"))
            if turn%11==9:state=apply_feedback(state,SupportFeedback(f"{sid}/decline-{turn}",state.revision,"decline"))
            if turn%11==10:state=apply_feedback(state,SupportFeedback(f"{sid}/resume-{turn}",state.revision,"resume"))
            checks+=1
            try:parse_state(asdict(state),sid+"-other")
            except ValueError:pass
            else:failures.append({"user":sid,"turn":turn,"kind":"cross_user_isolation"})
            if turn in [0,turns-1]:traces.append({"synthetic":True,"user":sid,"turn":turn,"goal":state.goal,"plan":state.plan.action})
    return {"synthetic":True,"users":count,"turns":count*turns,"checks":checks,"failures":failures,"sample_trace":traces,
            "implementation":"actual intervention.support_understanding, not a simulated replacement planner"}


def forecast_simulation(output: Path,count: int=120,length: int=180) -> dict[str,Any]:
    """不同合成动态中评测过去8步预测下一步；按人隔离，禁止未来插补。"""
    rng=np.random.default_rng(42);series=[];mask=[];regimes=[]
    for user in range(count):
        regime=user%4;mu=float(rng.uniform(-.4,.4));value=float(rng.normal(mu,.15));trajectory=[]
        for t in range(length):
            cycle=.2*np.sin(2*np.pi*t/12) if regime==1 else 0.
            event=.35 if regime==2 and 65<t<90 else 0.
            coefficient=[.85,.55,.75,.15][regime]
            value=coefficient*value+(1-coefficient)*(mu+cycle+event)+rng.normal(0,.08 if regime!=3 else .22)
            trajectory.append(float(np.clip(value,-1,1)))
        series.append(trajectory);mask.append(rng.random(length)>.15);regimes.append(regime)
    series=np.array(series);mask=np.array(mask);features=[];targets=[];users=[];persistence=[];regime_rows=[]
    for user in range(count):
        filled=[];last=0.
        for t in range(length):
            if mask[user,t]:last=series[user,t]
            filled.append(last)
            if t>=7 and t+1<length and mask[user,t+1]:
                features.append([*filled[-8:],*mask[user,t-7:t+1].astype(float),np.sin(2*np.pi*t/12),np.cos(2*np.pi*t/12)])
                targets.append(series[user,t+1]);users.append(user);persistence.append(last);regime_rows.append(regimes[user])
    x=np.array(features);y=np.array(targets);users=np.array(users);last=np.array(persistence)
    order=np.random.default_rng(42).permutation(count)
    train_end=int(count*.7);valid_end=int(count*.85)
    train=np.isin(users,order[:train_end]);valid=np.isin(users,order[train_end:valid_end]);test=np.isin(users,order[valid_end:])
    candidates={"ridge":make_pipeline(StandardScaler(),Ridge(alpha=10.)),
                "hgb":HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=15,l2_regularization=2.,random_state=42)}
    scores={};predictions={}
    for name,model in candidates.items():
        model.fit(x[train],y[train]);scores[name]=float(mean_absolute_error(y[valid],model.predict(x[valid])))
        predictions[name]=model.predict(x[test])
    winner=min(scores,key=scores.get)
    np.savez_compressed(output/"synthetic_forecast.npz",synthetic=np.array(True),series=series,observed=mask,
                        test_truth=y[test],test_users=users[test],test_persistence=last[test],**predictions)
    return {"synthetic":True,"users":count,"observations_generated":int(series.size),"observed":int(mask.sum()),
        "participant_split":[train_end,valid_end-train_end,count-valid_end],"windows":{"train":int(train.sum()),"validation":int(valid.sum()),"test":int(test.sum())},
        "winner":winner,"validation_mae":scores,"test_mae":{"persistence":float(mean_absolute_error(y[test],last[test])),
                    **{name:float(mean_absolute_error(y[test],pred)) for name,pred in predictions.items()}},
        "target":"next generated valence observation, arbitrary units; not questionnaire/clinical score",
        "regimes":["autocorrelated","cyclic","transient_event","weakly_predictable_noise"],
        "missingness":"15% independent synthetic missing; forward-fill uses past only, missing masks included",
        "claim_limit":"tests pipeline under assumed generator, no evidence about actual people or treatment"}


def run(output: Path) -> dict[str,Any]:
    """运行两类模拟，所有结果显式标记合成。"""
    output.mkdir(parents=True,exist_ok=True)
    protocol={"seed":42,"synthetic":True,"human_participants":0,"clinical_efficacy_tested":False,
              "support_users":120,"support_turns_per_user":30,"forecast_users":120,"forecast_steps":180,"code_sha256":sha256_file(Path(__file__))}
    save_json(output/"simulation_protocol.json",protocol)
    result={"protocol":protocol,"support":support_simulation(),"forecast":forecast_simulation(output)}
    save_json(output/"simulation_results.json",result)
    print(json.dumps({"support_checks":result["support"]["checks"],"support_failures":len(result["support"]["failures"]),"forecast":result["forecast"]}),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args().output)
