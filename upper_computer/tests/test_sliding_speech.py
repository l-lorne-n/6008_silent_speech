from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import numpy as np
from sliding_speech import (CommandDecoder, REST, balanced_indices, training_label,
                            split_runs, window_ends, event_metrics)


def probabilities(label):
    x=np.full(11,.005)
    x[label]=.95
    return x/x.sum()


def record():
    return dict(recording='r',subject='S',first=0,annotations=[
        dict(trial_id=1,prepare=0,action=3000,rest=7000,context_end=11000,split='test',word='open'),
        dict(trial_id=2,prepare=11000,action=14000,rest=18000,context_end=22000,split='test',word='close'),
        dict(trial_id=3,prepare=22000,action=25000,rest=29000,context_end=33000,split='train',word='open'),
        dict(trial_id=4,prepare=33000,action=36000,rest=40000,context_end=44000,split='test',word='open')])


def test_rest_majority_cannot_dominate_sampler():
    rows=[]
    for c in range(11):
        for s in ['a','b']:
            for t in range(2):
                for w in range(100 if c==REST else 3):
                    rows.append(dict(label=c,subject=s,recording=s,trial_id=t))
    ids=balanced_indices(rows,np.arange(len(rows)),40,123,True)
    for c in range(11):
        assert sum(rows[i]['label']==c for i in ids)==40
        assert sum(rows[i]['label']==c and rows[i]['subject']=='a' for i in ids)==20
    unique=balanced_indices(rows,np.arange(len(rows)),40,123,False)
    assert len(unique)==len(set(unique))


def test_transition_windows_are_not_mislabeled_as_rest():
    t=record()['annotations'][0]
    assert training_label(t,4000,5600,['open','close'])==0
    assert training_label(t,7600,9200,['open','close'])==REST
    assert training_label(t,2400,4000,['open','close']) is None
    assert training_label(t,6500,8100,['open','close']) is None
    assert training_label(t,-10,1590,['open','close']) is None


def test_decoder_needs_rest_and_confirmation_and_suppresses_repeats():
    d=CommandDecoder(probability=.65,confirm=2,rest_frames=2)
    outputs=[]
    sequence=[0,0,REST,REST,0,0,0,0,REST,REST,REST,1,1]
    for i,c in enumerate(sequence):
        event=d.feed(i*200,probabilities(c))
        if event:outputs.append(event)
    assert [e['label'] for e in outputs]==[0,1]
    assert outputs[0]['emitted']==1000


def test_always_rest_cannot_get_high_command_accuracy():
    r=record();m,rows=event_metrics([], [r], 'test')
    assert m['trials']==3 and m['end_to_end_recall']==0 and m['no_response_trials']==3


def test_repeated_correct_predictions_are_not_multiple_successes():
    r=record()
    events=[dict(recording='r',emitted=end,candidate='open') for end in [4000,4200,8000]]
    m,_=event_metrics(events,[r],'test')
    assert m['correct_unique_trials']==1 and m['clean_correct_trials']==0
    assert m['duplicate_outputs']==1 and m['rest_proxy_outputs']==1
    assert m['trials']==3


def test_split_runs_do_not_reset_every_trial_or_cross_training():
    r=record();runs=split_runs(r,'test')
    assert runs==[[0,22000],[33000,44000]]
    for start,end in runs:
        for pos in window_ends(r,start,end,2800):
            assert start<=pos-2800<pos<=end
            assert not (pos-2800<33000 and pos>22000)


def test_preaction_and_late_correct_labels_are_false_outputs():
    r=record()
    ev=[dict(recording='r',emitted=2000,candidate='open'),
        dict(recording='r',emitted=7600,candidate='open')]
    m,_=event_metrics(ev,[r],'test')
    assert m['correct_unique_trials']==0 and m['outside_response_outputs']==2


def test_decoder_does_not_use_future_predictions():
    prefix=[REST,REST,0,0]
    def run(seq):
        d=CommandDecoder();result=[]
        for i,c in enumerate(seq):
            e=d.feed(i*200,probabilities(c))
            if e:result.append(e)
        return result
    a=run(prefix)
    b=run(prefix+[1,1,REST,REST])
    assert a==[e for e in b if e['emitted']<=600]
