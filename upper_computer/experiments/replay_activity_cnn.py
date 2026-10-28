"""Validate a label-blind activity gate + frozen CNN on continuous recordings.

No retraining, device access, source mutation, or test-driven parameter selection.
Cue times are scoring proxies, NOT manually annotated muscle onset/offset.
"""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'.cnn-deps'), str(ROOT/'.onnx-deps')]
import numpy as np
import onnxruntime as ort
import torch
from semg.activity_cnn import ActivityGate, CausalRMSFrontend, GateConfig

MODEL = ROOT/'models/cnn_goodone_plus_h7_w32_20261021_152727_4b7b5c/cnn'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def manifests():
    with (MODEL/'split_manifest.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    for r in rows:
        for k in ['trial_id','block','prepare','action','rest']:r[k]=int(r[k])
    return rows


def cues(folder, rows, end):
    """Scoring-only annotations, kept out of frontend/gate interfaces."""
    events=[json.loads(s) for s in (folder/'events.jsonl').read_text(encoding='utf-8-sig').splitlines() if s]
    ack={e['event_id']:e['device_sample_idx'] for e in events if e['kind']=='marker_ack'}
    trials={}
    for e in events:
        if e['kind']=='phase' and 'trial_id' in e and e['phase'] in ['prepare','action','rest']:
            t=trials.setdefault(e['trial_id'],dict(trial_id=e['trial_id'],word=e['word'],block=e['block'],split='excluded'))
            if e['event_id'] in ack:t[e['phase']]=ack[e['event_id']]
    lookup={r['trial_id']:r for r in rows}
    out=sorted((t for t in trials.values() if 'prepare' in t),key=lambda t:t['prepare'])
    for i,t in enumerate(out):
        t['context_end']=out[i+1]['prepare'] if i+1<len(out) else end
        if t['trial_id'] in lookup:
            r=lookup[t['trial_id']]
            assert all(t[k]==r[k] for k in ['prepare','action','rest','word','block'])
            t['split']=r['split']
    assert len(lookup)==sum(t['split']!='excluded' for t in out)
    return out


def prepare(out, rows):
    cache=out/'cache';cache.mkdir(exist_ok=True)
    contract=read(MODEL/'deployment_calibrated/input_contract.json')
    sources=read(MODEL/'protocol.json')['source_hashes']
    records=[];hashes={}
    for source in sources:
        folder=Path(source['path']);name=folder.name
        for fn,h in source['hashes'].items():
            assert digest(folder/fn)==h, f'Source changed: {folder/fn}'
            hashes[str(folder/fn)]=h
        meta=read(folder/'metadata.json')
        assert meta['sampling_rate_hz']==2000 and meta['durations']['baseline']>=5
        data=np.loadtxt(folder/'samples.csv',delimiter=',',skiprows=1,usecols=(1,4,7),dtype=np.int64)
        assert np.all(np.diff(data[:,0])==1)
        first=int(data[0,0]);end=int(data[-1,0])+1
        rr=[r for r in rows if r['recording']==name]
        annotation=cues(folder,rr,end)
        assert min(t['prepare'] for t in annotation)-first>=10000
        frontend=CausalRMSFrontend(contract['filter_sos'])
        wave=[];rms=[];ends=[];clipped=[]
        for start in range(0,len(data),20000):
            for frame in frontend.feed(data[start:start+20000,1:],first+start):
                wave.append(frame['filtered']);rms.append(frame['rms']);ends.append(frame['end']);clipped.append(frame['clipped'])
        y=np.concatenate(wave).astype(np.float32)
        np.savez(cache/(name+'.npz'),filtered=y,rms=np.asarray(rms),ends=np.asarray(ends),clipped=np.asarray(clipped))
        records.append(dict(recording=name,subject=meta['subject_id'],path=str(folder),first=first,end=end,
                            samples=len(data),seconds=len(data)/2000,annotations=annotation,
                            clipped_frames=int(sum(clipped)),tail_unprocessed_samples=len(data)-len(y)))
        print('PREPARED',meta['subject_id'],name,round(len(data)/2000,2),'s',flush=True)
    for p in [MODEL/'model.pt',MODEL/'selection_checkpoint.pt',MODEL/'split_manifest.csv',
              MODEL/'deployment_calibrated/model_int8_qdq.onnx',MODEL/'deployment/model_fp32.onnx',
              MODEL/'deployment_calibrated/percentile_99.9_development.onnx']:
        hashes[str(p)]=digest(p)
    write(out/'source_sha256.json',hashes)
    write(out/'recordings.json',records)
    return records


def detect(record, cache, config):
    gate=ActivityGate(config);events=[]
    for level,end,clip in zip(cache['rms'],cache['ends'],cache['clipped']):
        e=gate.feed(level,int(end),bool(clip))
        if e:events.append(e)
    return events,dict(state=gate.state,on=gate.on.tolist(),off=gate.off.tolist(),
                       calibration_excluded_frames=gate.calibration_excluded_frames)


def session(path):
    o=ort.SessionOptions();o.intra_op_num_threads=2;o.inter_op_num_threads=1
    return ort.InferenceSession(str(path),sess_options=o,providers=['CPUExecutionProvider'])


def attach_scores(events, record, cache, split, runtime, norm, classes):
    annotations=record['annotations'];starts=np.array([t['prepare'] for t in annotations])
    actions=[t for t in annotations if 'action' in t and 'rest' in t]
    rows=[]
    for n,event in enumerate(events):
        # A candidate belongs to the trial context in which its activity starts.
        # Crossing a split boundary never grants access to the other split's waveform.
        owner_ix=int(np.searchsorted(starts,event.get('active_start',event['start']),side='right'))-1
        if owner_ix<0:continue
        owner=annotations[owner_ix]
        if owner['split']!=split:continue
        e=dict(event,event_id=n,recording=record['recording'],subject=record['subject'],
               owner_trial=owner['trial_id'],split=split,matched_trial=None,matched_word=None,
               candidate=None,probability=None,margin=None)
        if e['kind']=='segment':
            a,b=e['active_start'],e['active_end']
            hits=[(max(0,min(b,t['rest'])-max(a,t['action'])),t) for t in actions]
            overlap,t=max(hits,key=lambda v:v[0])
            if overlap>=200 and overlap/max(1,b-a)>=.5 and t['trial_id']==owner['trial_id']:
                e.update(matched_trial=t['trial_id'],matched_word=t['word'])
                e['emitted_minus_cue_end_ms']=(e['emitted']-t['rest'])/2
            touched=[t for t in annotations if t['prepare']<e['end'] and t['context_end']>e['start']]
            if any(t['split']!=split for t in touched):
                e.update(kind='rejected',reason='cross_split_boundary')
            elif not 400<=e['end']-e['start']<=6144 or e['start']<record['first']:
                e.update(kind='rejected',reason='input_length')
            else:
                assert e['end']<=e['emitted']
                y=cache['filtered'][e['start']-record['first']:e['end']-record['first']]
                x=np.zeros((1,2,1,6144),dtype=np.float32)
                x[0,:,0,:len(y)]=((y-np.array(norm['mean']))/np.array(norm['scale'])).T
                logits=runtime.run(None,{'input':x})[0][0]
                p=np.exp(logits-logits.max());p/=p.sum();order=np.argsort(p)
                e.update(candidate=classes[int(order[-1])],probability=float(p[order[-1]]),
                         margin=float(p[order[-1]]-p[order[-2]]),length_samples=len(y))
        rows.append(e)
    return rows


def metrics(events, records, split, probability=0., margin=0., subject=None):
    recs=[r for r in records if subject is None or r['subject']==subject]
    truth={(r['recording'],t['trial_id']):t for r in recs for t in r['annotations'] if t['split']==split}
    ev=[e for e in events if subject is None or e['subject']==subject]
    predicted=[e for e in ev if e['candidate'] is not None and e['probability']>=probability and e['margin']>=margin]
    detected=[e for e in ev if e['candidate'] is not None and e['matched_trial'] is not None]
    dkeys={(e['recording'],e['matched_trial']) for e in detected}
    counts=Counter((e['recording'],e['matched_trial']) for e in predicted if e['matched_trial'] is not None)
    correct_keys=set();correct_events=0;wrong=0;outside=0
    first_keys=set();first_correct=0
    for e in predicted:
        key=(e['recording'],e['matched_trial'])
        if e['matched_trial'] is None:outside+=1;continue
        if key not in first_keys:
            first_keys.add(key)
            first_correct+=e['candidate']==e['matched_word']
        if e['candidate']==e['matched_word']:
            correct_keys.add(key);correct_events+=1
        else:wrong+=1
    # Only exactly one output, carrying the correct command, counts as a clean success.
    owner_counts=Counter((e['recording'],e['owner_trial']) for e in predicted)
    clean=sum(counts[k]==1 and owner_counts[k]==1 for k in correct_keys)
    duration=sum((t['context_end']-t['prepare'])/2000 for t in truth.values())
    rest_s=sum(max(0,t['context_end']-t['rest']-600)/2000 for t in truth.values())
    # Guard the first 300 ms of cued rest; these are still cue-based proxies.
    rest_events=0
    for e in predicted:
        t=truth[(e['recording'],e['owner_trial'])]
        a,b=e['active_start'],e['active_end']
        if max(0,min(b,t['context_end'])-max(a,t['rest']+600))>=.5*(b-a):rest_events+=1
    tp=len(correct_keys);fp=len(predicted)-tp;fn=len(truth)-tp
    return dict(trials=len(truth),duration_minutes=duration/60,scored_rest_minutes=rest_s/60,
                detected_trials=len(dkeys),detection_recall=len(dkeys)/len(truth),
                candidate_events=sum(e['candidate'] is not None for e in ev),outputs=len(predicted),
                correct_unique_trials=tp,end_to_end_recall=tp/len(truth),
                clean_correct_trials=clean,clean_success_rate=clean/len(truth),
                first_output_correct_trials=first_correct,first_output_success_rate=first_correct/len(truth),
                event_precision=tp/max(1,len(predicted)),event_f1=2*tp/max(1,2*tp+fp+fn),
                missed_or_rejected_trials=len(truth)-len(counts),duplicate_outputs=sum(max(0,n-1) for n in counts.values()),
                wrong_label_outputs=wrong,outside_cue_outputs=outside,
                rest_proxy_outputs=rest_events,rest_proxy_outputs_per_min=rest_events/max(1e-9,rest_s/60),
                rejected_events=dict(Counter(e.get('reason') for e in ev if e['kind']=='rejected')))


def run(out, records, cfg, split, runtime, norm, classes):
    events=[];states=[]
    for record in records:
        with np.load(out/'cache'/(record['recording']+'.npz')) as cache:
            detected,state=detect(record,cache,cfg)
            events+=attach_scores(detected,record,cache,split,runtime,norm,classes)
            states.append(dict(recording=record['recording'],**state))
    return events,states


def cue_control(out, records, runtime, norm, classes):
    """Diagnostic ONLY: original cue cuts with continuous causal filtering.

    This is not an automatic detector result and is never used for selection.
    """
    predictions=[]
    for record in records:
        with np.load(out/'cache'/(record['recording']+'.npz')) as cache:
            y=cache['filtered']
            for t in record['annotations']:
                if t['split']!='test':continue
                clip=y[t['action']+300-record['first']:t['rest']-300-record['first']]
                assert 400<=len(clip)<=6144
                x=np.zeros((1,2,1,6144),dtype=np.float32)
                x[0,:,0,:len(clip)]=((clip-np.array(norm['mean']))/np.array(norm['scale'])).T
                pred=classes[int(runtime.run(None,{'input':x})[0][0].argmax())]
                predictions.append(dict(recording=record['recording'],subject=record['subject'],trial_id=t['trial_id'],
                                        word=t['word'],prediction=pred,correct=pred==t['word']))
    return predictions


def trial_diagnostics(events, records, probability):
    result=[]
    for r in records:
        for t in r['annotations']:
            if t['split']!='test':continue
            ev=[e for e in events if e['recording']==r['recording'] and e['owner_trial']==t['trial_id']]
            candidates=[e for e in ev if e['candidate'] is not None]
            accepted=[e for e in candidates if e['probability']>=probability]
            matched=[e for e in accepted if e['matched_trial']==t['trial_id']]
            result.append(dict(subject=r['subject'],recording=r['recording'],trial_id=t['trial_id'],word=t['word'],
                               cue_action_ms=(t['rest']-t['action'])/2,
                               candidates=len(candidates),outputs=len(accepted),matched_outputs=len(matched),
                               clean_correct=len(accepted)==1 and len(matched)==1 and matched[0]['candidate']==t['word'],
                               predictions=';'.join(e['candidate'] for e in accepted),
                               segment_lengths_ms=';'.join(str(e['length_samples']/2) for e in candidates),
                               rejected=';'.join(e.get('reason','') for e in ev if e['kind']=='rejected')))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--phase',choices=['prepare','select','test','all'],default='all')
    args=p.parse_args()
    out=args.output or ROOT/'analysis'/('activity_cnn_replay_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    if args.phase in ['prepare','all']:
        out.mkdir(parents=True,exist_ok=False)
        shutil.copy2(__file__,out/Path(__file__).name)
        shutil.copy2(ROOT/'semg/activity_cnn.py',out/'activity_cnn.py')
        protocol=dict(model=str(MODEL),method='2: causal RMS activity gate then existing frozen 10-command CNN',
                      no_retraining=True,no_board_access=True,fs=2000,
                      selection='9 detector settings x 4 confidence settings; maximize validation clean single-output correct-trial rate, tie-break by event F1 then fewer outputs',
                      grid=dict(on_ratio=[2.,3.,4.],release_ms=[160,320,480],min_probability=[0.,.6,.75,.9]),
                      scoring='Cue proxy: >=100ms overlap and >=50% of detected active span inside the owner action cue; unique event matching; strict one-output correctness also reported.',
                      rest_scoring='Cued rest after 300ms guard, excluding prepare and excluded trials; not manually confirmed physiological rest.',
                      threshold_calibration='First 5 seconds of every recording, discard initial 500ms and clipping+200ms settling; require >=3s clean calibration; no cue labels; fixed thresholds afterward.',
                      limitations=['Previously inspected/selected seen-person data, not blind or LOSO.',
                                   'Cue times are not muscle onset/offset ground truth; rest outputs are proxies.',
                                   'Continuous filter and detector segments differ from cue-trained CNN inputs.',
                                   'State evolves continuously through all splits; scoring/inference restricted to target split; cross-split segments rejected.',
                                   'No unknown-action labels; no measured STM32 runtime or live electrode robustness.'])
        write(out/'protocol.json',protocol)
        print('OUTPUT',out,flush=True)
        records=prepare(out,manifests())
        if args.phase=='prepare':return
    else:records=read(out/'recordings.json')
    classes=read(MODEL/'deployment_calibrated/input_contract.json')['classes']
    if args.phase in ['select','all']:
        assert not (out/'selected.json').exists()
        cp=torch.load(MODEL/'selection_checkpoint.pt',map_location='cpu',weights_only=True)
        runtime=session(MODEL/'deployment_calibrated/percentile_99.9_development.onnx')
        candidates=[]
        for on in [2.,3.,4.]:
            for release in [160,320,480]:
                cfg=GateConfig(on_ratio=on,off_ratio=1+(on-1)*.4,release_ms=release)
                name=f'on{on:g}_release{release}'
                event_path=out/(name+'_validation_events.json')
                if event_path.exists():
                    events=read(event_path)
                else:
                    events,states=run(out,records,cfg,'validation',runtime,cp['normalization'],classes)
                    write(event_path,events)
                for prob in [0.,.6,.75,.9]:
                    m=metrics(events,records,'validation',prob)
                    candidates.append(dict(name=name,config=asdict(cfg),min_probability=prob,min_margin=0.,metrics=m))
                print('VALIDATION',name,[(c['min_probability'],round(c['metrics']['clean_success_rate'],4),round(c['metrics']['rest_proxy_outputs_per_min'],3)) for c in candidates[-4:]],flush=True)
                write(out/'validation_candidates.json',candidates)
        chosen=max(candidates,key=lambda c:(c['metrics']['clean_success_rate'],c['metrics']['event_f1'],-c['metrics']['outputs']))
        write(out/'selected.json',chosen)
        print('FROZEN',chosen['name'],chosen['min_probability'],chosen['metrics'],flush=True)
        if args.phase=='select':return
    if args.phase in ['test','all']:
        assert not (out/'test_report.json').exists()
        chosen=read(out/'selected.json');cfg=GateConfig(**chosen['config'])
        norm=read(MODEL/'deployment_calibrated/input_contract.json')['normalization']
        outputs={}
        for kind,path in [('int8',MODEL/'deployment_calibrated/model_int8_qdq.onnx'),('fp32',MODEL/'deployment/model_fp32.onnx')]:
            runtime=session(path)
            events,states=run(out,records,cfg,'test',runtime,norm,classes)
            write(out/(kind+'_test_events.json'),events);write_csv(out/(kind+'_test_events.csv'),events)
            write(out/(kind+'_recording_states.json'),states)
            write_csv(out/(kind+'_trial_diagnostics.csv'),trial_diagnostics(events,records,chosen['min_probability']))
            control=cue_control(out,records,runtime,norm,classes)
            write_csv(out/(kind+'_cue_control.csv'),control)
            outputs[kind]=dict(without_confidence_rejection=metrics(events,records,'test'),
                               selected_rejection=metrics(events,records,'test',chosen['min_probability']),
                               cue_cut_continuous_filter_control=dict(trials=len(control),correct=sum(r['correct'] for r in control),
                                                                     accuracy=sum(r['correct'] for r in control)/len(control)),
                               by_subject={s:metrics(events,records,'test',chosen['min_probability'],subject=s) for s in sorted({r['subject'] for r in records})})
            print('TEST',kind,outputs[kind]['selected_rejection'],flush=True)
        for path,h in read(out/'source_sha256.json').items():assert digest(path)==h
        report=dict(selected=chosen,results=outputs,recordings=len(records),total_recorded_minutes=sum(r['seconds'] for r in records)/60,
                    sources_and_models_unchanged=True,scoring_is_cue_proxy=True)
        write(out/'test_report.json',report)
        shutil.copy2(__file__,out/'replay_activity_cnn_final.py')
        shutil.copy2(ROOT/'semg/activity_cnn.py',out/'activity_cnn_final.py')
        print('COMPLETE',out,flush=True)


if __name__=='__main__':main()
