"""准备官方ECISA2021独立保留集，只做来源匹配与文本去重，不用于调参。"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import numpy as np
from round5_prepare_external import normalize


def run(archive: Path,external: Path,bundle: Path,output: Path) -> dict:
    """严格按Doc/Sentence ID匹配官方答案，不将未标注的混淆句当中性。"""
    output.mkdir(parents=True,exist_ok=True)
    labels={};duplicate_answers=0
    answer=archive/'SMP2021-ECISA_Test_答案.txt'
    for line in answer.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():continue
        fields=line.strip().split();key=fields[0];label=fields[1].rstrip('$')
        if label not in {'0','1','2'}:raise ValueError("未知答案标签")
        if key in labels:
            if labels[key]!=label:raise ValueError("同ID答案冲突")
            duplicate_answers+=1
            continue
        labels[key]=label
    seen=set()
    for name in ['train','external_valid']:
        for line in (external/(name+'.jsonl')).read_text(encoding='utf-8').splitlines():seen.add(normalize(json.loads(line)['text']))
    for split in ['train','selection','calibration','test']:
        with np.load(bundle/(split+'.npz'),allow_pickle=False) as saved:seen.update(normalize(str(t)) for t in saved['raw_text'])
    grams={t[i:i+12] for t in seen for i in range(max(0,len(t)-11))}
    rows=[];matched=set();removed=Counter();mapping={'0':1,'1':2,'2':0}
    source=archive/'SMP2021-ECISA_Test.xml'
    # 上游正文有未转义的&和<；只修复正文转义，不丢弃任何文档或改答案。
    xml=source.read_text(encoding='utf-8-sig')
    xml,escaped_ampersands=re.subn(r'&(?!#\d+;|#x[0-9a-fA-F]+;|amp;|lt;|gt;|quot;|apos;)','&amp;',xml)
    xml,escaped_literal_lt=re.subn(r'<(?!/?(?:Doc|Sentence|SMP2021-ECISA)\b|\?xml\b)','&lt;',xml)
    for doc in ET.fromstring(xml).iter('Doc'):
        for sentence in doc.iter('Sentence'):
            key=doc.get('ID')+'-'+sentence.get('ID')
            if key not in labels:continue
            if key in matched:raise ValueError("源样本ID重复")
            matched.add(key);text=''.join(sentence.itertext()).strip();normal=normalize(text)
            if not normal or normal in seen:removed['exact_overlap_or_empty']+=1;continue
            if any(normal[i:i+12] in grams for i in range(max(0,len(normal)-11))):removed['12char_overlap']+=1;continue
            seen.add(normal)
            rows.append({'id':'ecisa2021/'+key,'group':'ecisa2021/'+doc.get('ID'),'text':text,'label':mapping[labels[key]],
                'source':'official_ECISA2021_test','split':'fresh_external_holdout','synthetic':False})
    if matched!=set(labels):raise ValueError(f"未匹配答案{len(set(labels)-matched)}条")
    destination=output/'holdout.jsonl'
    destination.write_text('\n'.join(json.dumps(row,ensure_ascii=False) for row in rows)+'\n',encoding='utf-8')
    report={'source':'https://github.com/sxu-nlp/ECISA2021','official_answer_count':len(labels),'identical_duplicate_answers_collapsed':duplicate_answers,'retained':len(rows),
        'counts':dict(Counter(row['label'] for row in rows)),'removed':dict(removed),
        'xml_repairs':{'unescaped_ampersands':escaped_ampersands,'literal_lt_in_body':escaped_literal_lt},
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'answer_sha256':hashlib.sha256(answer.read_bytes()).hexdigest(),
        'prepared_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
        'selection_policy':'no model choice, threshold, prompt example, checkpoint or fitting may use this holdout',
        'construct':'Chinese implicit sentence sentiment, not multimodal mental health','license_scope':'academic research/technology evaluation; no commercial use without authorization'}
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['archive','external','bundle','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();run(args.archive,args.external,args.bundle,args.output)
