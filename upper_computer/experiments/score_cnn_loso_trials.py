"""Predeclared cue-assisted ten-word diagnostic from frozen LOSO CNN traces.

This is a secondary offline classifier metric, not the online command decoder.
One prediction per trial: arithmetic mean of the original 11-class softmax
probabilities over cue-valid action windows, then argmax among the ten words.
No renormalization per frame, target fitting, threshold search or best-window pick.
"""
import argparse
from pathlib import Path
import run_cnn_current_loso as loso

np, io, base, sweep = loso.np, loso.io, loso.base, loso.sweep


def declare(out):
    assert not (out/'all_models_frozen_sha256.json').exists(), 'Declare before final models/test'
    assert not list(out.glob('S*/mid64_temporal3/test/*')), 'No test results may precede declaration'
    policy=dict(name='cue_assisted_ten_word_mean_probability_v1', primary=False,
        script_sha256=io.digest(Path(__file__)),
        rule='Mean original 11-class probabilities on cue-valid action windows; argmax of first ten entries.',
        windows='1400ms,100ms stride; >=75% action overlap; action+150ms <= window end <= rest cue; inside trial context.',
        rest='Rest probability is excluded only from the final ten-word argmax. No per-frame renormalization.',
        caveat='Uses known cue boundaries and forces a word; not online success or a new ten-output network.',
        tuning=False)
    io.write(out/'ten_word_diagnostic_policy.json',policy)
    sweep.seal(out,'diagnostic_policy_sha256.json',[out/'ten_word_diagnostic_policy.json',Path(__file__).resolve()])


def score(out):
    base.check_seal(out,'diagnostic_policy_sha256.json')
    assert io.read(out/'completed.json')['complete']
    classes=io.read(out/'protocol.json')['classes'];assert classes[-1]=='rest' and len(classes)==11
    results={};table=[];files=[]
    for kind in ['fp32','int8']:
        trials=[]
        for subject in sweep.SUBJECTS:
            fold=out/subject;folder=fold/loso.current.architecture()['name']/'test'
            base.check_seal(folder,'test_sha256.json')
            with np.load(folder/f'{kind}_trace.npz') as z:
                trace={k:z[k] for k in z.files}
            for r in io.read(fold/'recordings.json'):
                if r['subject']!=subject:continue
                rec_ix=np.flatnonzero(trace['recording']==r['recording'])
                rec_ends=trace['end'][rec_ix]
                for t in r['annotations']:
                    if t['split']!='test':continue
                    # Selection depends on cue boundaries, never the true word identity.
                    starts=rec_ends-2800
                    overlap=np.maximum(0,np.minimum(rec_ends,t['rest'])-np.maximum(starts,t['action']))
                    keep=(starts>=t['prepare']) & (rec_ends<=t['context_end']) & (overlap>=2100) & \
                         (rec_ends>=t['action']+300) & (rec_ends<=t['rest'])
                    ix=rec_ix[keep];assert len(ix)>0, (subject,r['recording'],t['trial_id'])
                    mean=trace['probabilities'][ix].astype(np.float64).mean(axis=0)
                    pred=classes[int(np.argmax(mean[:10]))]
                    trials.append(dict(subject=subject,recording=r['recording'],trial_id=t['trial_id'],
                        word=t['word'],prediction=pred,correct=pred==t['word'],windows=len(ix),
                        mean_rest_probability=float(mean[10]),
                        **{'prob_'+c:float(mean[i]) for i,c in enumerate(classes[:10])}))
        assert len(trials)==io.read(out/'protocol.json')['test_trials']
        assert len({(t['recording'],t['trial_id']) for t in trials})==len(trials)
        group={}
        for subject in sweep.SUBJECTS+['ALL']:
            part=trials if subject=='ALL' else [t for t in trials if t['subject']==subject]
            cm=np.zeros((10,10),dtype=int)
            for t in part:cm[classes.index(t['word']),classes.index(t['prediction'])]+=1
            n=len(part);correct=sum(t['correct'] for t in part)
            group[subject]=dict(trials=n,correct=correct,accuracy=correct/n,
                macro_recall=float(np.mean(cm.diagonal()/cm.sum(axis=1))),confusion_matrix=cm.tolist())
            table.append(dict(format=kind,subject=subject,trials=n,correct=correct,accuracy=correct/n))
        results[kind]=group
        path=out/f'{kind}_ten_word_trials.csv';io.write_csv(path,trials);files.append(path)
    io.write(out/'ten_word_comparison.json',results);io.write_csv(out/'ten_word_comparison.csv',table)
    files += [out/'ten_word_comparison.json',out/'ten_word_comparison.csv']
    sweep.seal(out,'ten_word_results_sha256.json',files)
    for row in table:print(row)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--declare',action='store_true')
    args=parser.parse_args()
    if args.declare:declare(args.output.resolve())
    else:score(args.output.resolve())
