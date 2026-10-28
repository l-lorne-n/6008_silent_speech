"""Five seen 2 kHz subjects, whole-round 80/20, fixed MCU-friendly CNN."""
from collections import Counter
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from semg import lda


def main():
    data=ROOT/'recordings/goodone_plus'
    subjects=['S011','S012','S013','S014','S015']
    selections=[]
    for subject in subjects:
        paths=sorted((data/subject.lower()).rglob('metadata.json'))
        assert paths,subject
        for p in paths:
            meta=json.loads(p.read_text(encoding='utf-8-sig'))
            assert meta['subject_id']==subject
            assert (meta['sampling_rate_hz'],meta['status'],meta['condition'],meta['articulation_mode'])==(2000,'completed','silent','silent_mouthing')
            selections.append(dict(path=str(p.parent.resolve()),role='train'))
    assert len(selections)==49
    out=ROOT/'models'/('cnn_goodone_plus_h7_'+datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    out.mkdir()
    for p in [Path(__file__),ROOT/'experiments/train_small_cnn.py',ROOT/'semg/lda.py']:
        shutil.copy2(p,out/p.name)
    print('OUTPUT',out,flush=True)
    baseline=lda.train(selections,out/'lda',strategy='within_recording',train_percent=80,seed=20260924,
                       variant='notch50_bp500',feature_settings=dict(segments=6,metrics=['log_mean_abs_diff','log_mav','zc_count_zero','ssc_count_zero']),
                       channels=[0,1],progress=lambda s:print(s,flush=True))
    report=baseline['report']
    counts=[]
    for subject in subjects:
        tr=[r for r in report['train'] if r['subject']==subject]
        te=[r for r in report['test'] if r['subject']==subject]
        assert {r['word'] for r in tr}=={r['word'] for r in te}==set(report['classes'])
        counts.append(dict(subject=subject,train=len(tr),test=len(te),
                           train_rounds=len({(r['recording'],r['block']) for r in tr}),
                           test_rounds=len({(r['recording'],r['block']) for r in te}),
                           train_by_word=dict(Counter(r['word'] for r in tr)),test_by_word=dict(Counter(r['word'] for r in te))))
    for path in selections:
        tr=[r for r in report['train'] if r['path']==path['path']]
        te=[r for r in report['test'] if r['path']==path['path']]
        assert tr and te and not {r['block'] for r in tr}&{r['block'] for r in te}
    protocol=dict(data_root=str(data),baseline=baseline['folder'],subjects=subjects,counts=counts,
                  split_seed=20260924,train_percent=80,architecture='h7_fixed_pool8_conv2d_v1',
                  scope='One shared model; each recording contributes disjoint training/test whole rounds.',
                  excluded=report['excluded'],test_used_for_selection=False,
                  caveat='goodone_plus is previously inspected/selected data, including S015 selected by test accuracy. Exploratory seen-person performance, not independent final evaluation.')
    lda.write_json(out/'protocol.json',protocol)
    print('COUNTS',counts,flush=True)
    env=dict(os.environ,PYTHONIOENCODING='utf-8')
    with (out/'training.log').open('w',encoding='utf-8') as log:
        child=subprocess.Popen([sys.executable,'-u',str(ROOT/'experiments/train_small_cnn.py'),'--baseline',baseline['folder'],'--output',str(out/'cnn'),'--h7'],
                               cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',env=env)
        for line in child.stdout:
            print(line,end='',flush=True);log.write(line);log.flush()
        assert child.wait()==0,'CNN failed; see training.log'
    result=json.loads((out/'cnn/report.json').read_text(encoding='utf-8'))
    assert result['protocol']['train_subjects']==result['protocol']['test_subjects']==subjects
    assert result['model_roundtrip_verified'] and result['input_hashes_unchanged']
    lda.write_json(out/'report.json',dict(protocol=protocol,cnn=result,lda_metrics=report['metrics']))
    print('COMPLETE',out,flush=True)


if __name__=='__main__':main()
