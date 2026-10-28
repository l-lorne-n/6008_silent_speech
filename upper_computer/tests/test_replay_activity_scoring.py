"""Event metrics must penalize duplicates, missed trials and background outputs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from replay_activity_cnn import metrics


def test_duplicate_and_wrong_outputs_cannot_inflate_clean_success():
    trials=[dict(trial_id=i,prepare=i*10000,action=i*10000+3000,rest=i*10000+7000,
                 context_end=(i+1)*10000,split='test',word='open') for i in range(3)]
    records=[dict(recording='r',subject='S',annotations=trials)]
    def event(owner,match,word='open',active_start=None):
        return dict(recording='r',subject='S',owner_trial=owner,matched_trial=match,matched_word='open' if match is not None else None,
                    kind='segment',candidate=word,probability=.8,margin=.5,
                    active_start=active_start if active_start is not None else owner*10000+3000,
                    active_end=active_start+500 if active_start is not None else owner*10000+4000)
    # Two correct repetitions for trial 0; a wrong trial 1; trial 2 missed; one rest output.
    ev=[event(0,0),event(0,0),event(1,1,'close'),event(2,None,active_start=28000)]
    m=metrics(ev,records,'test')
    assert m['correct_unique_trials']==1 and m['clean_correct_trials']==0
    assert m['duplicate_outputs']==1 and m['wrong_label_outputs']==1
    assert m['missed_or_rejected_trials']==1 and m['outside_cue_outputs']==1
    assert m['rest_proxy_outputs']==1 and m['event_precision']==.25
    assert metrics(ev,records,'test',probability=.9)['outputs']==0
    # A correct command followed by a background output in the same context is not clean.
    extra=[event(0,0),event(0,None,active_start=8000)]
    assert metrics(extra,records,'test')['clean_correct_trials']==0
