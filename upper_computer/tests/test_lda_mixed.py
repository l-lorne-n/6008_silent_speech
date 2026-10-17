import json

import numpy as np
import pytest

from semg import lda
from test_vocabulary import make_recording


@pytest.mark.parametrize('values,zc,ssc', [
    ([1,-1,1,-1],3,2), ([1,2,3,4],0,0),
    ([0,0,0,0],0,0), ([-1,0,1],0,0),
    ([1,2,2,1],0,0), ([1,2,1,2],0,2),
])
def test_zero_threshold_counts_and_flat_convention(values,zc,ssc):
    x=np.array(values,dtype=float)
    for scale in (1, .000001, 100):
        assert lda.segment_metric(x*scale,'zc_count_zero') == zc
        assert lda.segment_metric(x*scale,'ssc_count_zero') == ssc


def test_mixed_split_keeps_every_batch_on_both_sides_and_rounds_intact():
    rows=[]
    for subject,batch,rounds in [('A',0,10),('A',1,3),('B',0,2),('B',1,7)]:
        for block in range(rounds):
            for word in ['open','close']:
                rows.append(dict(subject=subject,source_hash=subject+str(batch),block=block,word=word))
    tr,te=lda.split_records(rows,'within_recording',80,123)
    identity=lambda r:(r['subject'],r['source_hash'],r['block'])
    assert {identity(rows[i]) for i in tr}.isdisjoint(identity(rows[i]) for i in te)
    assert sorted(list(tr)+list(te))==list(range(len(rows)))
    for subject,source in {(r['subject'],r['source_hash']) for r in rows}:
        selected=[r for r in rows if (r['subject'],r['source_hash'])==(subject,source)]
        nrounds=len({r['block'] for r in selected})
        testrows=[rows[i] for i in te if (rows[i]['subject'],rows[i]['source_hash'])==(subject,source)]
        assert len(testrows)==2*int(np.ceil(nrounds*.2))
        assert 0<len(testrows)<len(selected)
    tr2,te2=lda.split_records(rows,'within_recording',80,123)
    np.testing.assert_array_equal(tr,tr2);np.testing.assert_array_equal(te,te2)
    # Do not silently drop or send an entire small recording to just one side.
    incomplete=rows+[dict(subject='C',source_hash='C0',block=0,word='open',recording='short')]
    with pytest.raises(ValueError,match='short'):
        lda.split_records(incomplete,'within_recording',80,123)


def test_group_scores_distinguish_trial_weighting_from_subject_weighting():
    rows=[dict(subject='A',source_hash='a',recording='a',word='open') for _ in range(9)]
    rows+=[dict(subject='B',source_hash='b',recording='b',word='close')]
    predictions=[0]*10
    m=lda.metrics(rows,predictions,['open','close'])
    grouped=lda.group_metrics([],rows,predictions,['open','close'])
    assert m['accuracy']==.9
    assert grouped['subject_mean_accuracy']==.5
    assert sum(g['metrics']['correct'] for g in grouped['subject'])==m['correct']
    assert sum(g['test_total'] for g in grouped['recording'])==m['total']
    only_train=[dict(subject='C',source_hash='c',recording='c',word='open')]
    grouped=lda.group_metrics(only_train,rows,predictions,['open','close'])
    assert grouped['subject'][-1]['metrics'] is None
    assert grouped['subject_mean_accuracy']==.5


def test_multi_person_multi_batch_training_and_reload(tmp_path):
    selections=[]
    for i,(subject,batch) in enumerate([('A',0),('A',1),('B',0),('B',1)]):
        folder=make_recording(tmp_path/f'{subject}_{batch}',list(lda.WORDS),seed=123+i)
        meta=json.loads((folder/'metadata.json').read_text())
        meta['subject_id']=subject
        (folder/'metadata.json').write_text(json.dumps(meta))
        selections.append(dict(path=str(folder)))
    settings=dict(segments=4,metrics=['log_mav','log_mean_abs_diff','zc_count_zero','ssc_count_zero'])
    run=lda.train(selections,tmp_path/'models',strategy='within_recording',train_percent=50,feature_settings=settings)
    report=run['report'];model=lda.load_model(run['model_path'])
    assert model['training_subjects']==['A','B']
    assert len(model['scaler_mean'])==16
    np.testing.assert_allclose(model['scaler_mean'],np.mean([r['features'] for r in report['train']],axis=0))
    assert len(report['group_metrics']['subject'])==2
    assert len(report['group_metrics']['recording'])==4
    for group in report['group_metrics']['recording']:
        assert group['train_total']==group['test_total']==8
    testing=[dict(path=s['path'],trial_ids=[r['trial_id'] for r in report['test'] if r['path']==s['path']]) for s in selections]
    loaded=lda.evaluate(run['model_path'],testing,tmp_path/'models')
    assert loaded['report']['metrics']==report['metrics']
    assert loaded['report']['group_metrics']['subject_mean_accuracy']==report['group_metrics']['subject_mean_accuracy']
    with pytest.raises(ValueError,match='已用于训练'):
        lda.evaluate(run['model_path'],selections,tmp_path/'models')
