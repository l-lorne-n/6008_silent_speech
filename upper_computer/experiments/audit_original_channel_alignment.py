"""Read-only signal and channel-pattern audit, always in original physical order."""
from pathlib import Path
import sys, json, io, hashlib
from zipfile import ZipFile
from itertools import combinations
import numpy as np
from scipy.signal import sosfilt, sosfilt_zi, welch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from semg import lda
OUT=ROOT/'analysis/20261016_original_channel_alignment'
OUT.mkdir(exist_ok=False)
baseline=json.loads((ROOT/'models/lda_loso_3p_20261016_165457_187939/report.json').read_text(encoding='utf-8'))
fold=json.loads((Path(baseline['folds'][0]['folder'])/'report.json').read_text(encoding='utf-8'))
records=fold['train']+fold['test']
subjects=['S011','S012','S015']; words=fold['classes']
profiles={s:[] for s in subjects}; recording_names={s:[] for s in subjects}
quality=[]; trial_signals=[]
backup=ZipFile(ROOT/'history/data_corrections/20261016_S012_channel_swap/original_S012.zip')
sos=lda.filter_sos('notch50_bp500')
for si,src in enumerate(fold['sources']):
    folder=Path(src['path']); subject=src['subject']
    def original_bytes(name):
        if subject=='S012':
            rel=folder.relative_to(ROOT/'recordings/S012')
            data=backup.read((rel/name).as_posix())
        else: data=(folder/name).read_bytes()
        assert hashlib.sha256(data).hexdigest()==src['hashes'][name]
        return data
    meta=json.loads(original_bytes('metadata.json'))
    for name in ('events.jsonl','trials.jsonl'): original_bytes(name)
    data=np.loadtxt(io.BytesIO(original_bytes('samples.csv')),delimiter=',',skiprows=1,usecols=(1,4,7),dtype=np.int64)
    assert np.all(np.diff(data[:,0])==1)
    raw=data[:,1:]; offset=int(data[0,0]); x=raw.astype(float)-32768
    filtered=sosfilt(sos,x,axis=0,zi=sosfilt_zi(sos)[:,:,None]*x[0])[0]
    rows=sorted([r for r in records if r['recording']==folder.name],key=lambda r:r['trial_id'])
    events=[json.loads(line) for line in original_bytes('events.jsonl').splitlines() if line]
    acks={e['event_id']:e for e in events if e.get('kind')=='marker_ack'}
    base_event=next(e for e in events if e.get('kind')=='phase' and e.get('phase')=='baseline')
    bstart=acks[base_event['event_id']]['device_sample_idx']-offset+1000
    bend=min(r['prepare'] for r in rows)-offset
    assert 0<=bstart<bend<=len(raw)
    baseline_rms=np.sqrt(np.mean(filtered[bstart:bend]**2,axis=0))
    action_rms=[]; rest_rms=[]; signal_corr=[]
    for r in rows:
        start=r['action']-offset+300; end=r['rest']-offset-300
        action=filtered[start:end]
        rms=np.sqrt(np.mean(action**2,axis=0)); action_rms.append(rms)
        # The final 0.5 s of preparation is a consistent pre-action reference.
        rest=filtered[r['action']-offset-1000:r['action']-offset]
        rest_rms.append(np.sqrt(np.mean(rest**2,axis=0)))
        signal_corr.append(float(np.corrcoef(action.T)[0,1]))
        trial_signals.append(dict(subject=subject,recording=folder.name,trial_id=r['trial_id'],word=r['word'],action_rms=rms.tolist(),prepare_rms=rest_rms[-1].tolist()))
    freq,psd=welch(x[:min(len(x),120000)],fs=2000,nperseg=2048,axis=0)
    band=(freq>=20)&(freq<=500); mains=(freq>=48)&(freq<=52)
    median_frequency=[float(freq[band][np.searchsorted(np.cumsum(psd[band,c]),psd[band,c].sum()/2)]) for c in range(2)]
    entry=dict(subject=subject,session=folder.parent.name,recording=folder.name,valid_trials=len(rows),
        raw_median=np.median(raw,axis=0).tolist(),raw_min=raw.min(0).tolist(),raw_max=raw.max(0).tolist(),
        rail_fraction=np.mean((raw<=1)|(raw>=65534),axis=0).tolist(),
        baseline_rms=baseline_rms.tolist(),median_action_rms=np.median(action_rms,axis=0).tolist(),
        median_prepare_rms=np.median(rest_rms,axis=0).tolist(),
        median_action_to_prepare_ratio=np.median(np.array(action_rms)/np.maximum(rest_rms,1e-12),axis=0).tolist(),
        median_action_raw_band_correlation=float(np.median(signal_corr)),
        first_60s_unnotched_band_median_hz=median_frequency,
        first_60s_50hz_power_fraction=(psd[mains].sum(0)/psd[band].sum(0)).tolist())
    quality.append(entry)
    # word x physical channel x metric x six time segments, equal weight per recording.
    profile=np.array([np.mean([r['features'] for r in rows if r['word']==w],axis=0) for w in words]).reshape(10,2,4,6)
    profiles[subject].append(profile);recording_names[subject].append(folder.name)
    print('RAW',si+1,'/',len(fold['sources']),subject,folder.name,flush=True)
backup.close()
profiles={s:np.array(v) for s,v in profiles.items()}

def similarity(a,b):
    # Remove word-independent level and scale at each time bin. Compare word selectivity,
    # not raw voltage, after per-person gain differences are removed.
    a=a-a.mean(0);b=b-b.mean(0)
    a=a/np.maximum(a.std(0),1e-12);b=b/np.maximum(b.std(0),1e-12)
    # A separate 2x2 matching matrix for each of the four metrics.
    return np.array([[[float(np.mean(a[:,i,m,:]*b[:,j,m,:])) for j in range(2)] for i in range(2)] for m in range(4)])

def difference(m):
    return ((m[...,0,0]+m[...,1,1])-(m[...,0,1]+m[...,1,0]))/2

rng=np.random.default_rng(20261016); pairs=[]
for a,b in combinations(subjects,2):
    sim=similarity(profiles[a].mean(0),profiles[b].mean(0))
    boots=[]
    for i in range(1000):
        pa=profiles[a][rng.integers(len(profiles[a]),size=len(profiles[a]))].mean(0)
        pb=profiles[b][rng.integers(len(profiles[b]),size=len(profiles[b]))].mean(0)
        boots.append(difference(similarity(pa,pb)).mean())
    per_metric=difference(sim)
    pairs.append(dict(subjects=[a,b],metric_names=fold['feature_config']['metrics'],
        channel_similarity_by_metric=sim.tolist(),same_minus_swapped_by_metric=per_metric.tolist(),
        same_minus_swapped_mean=float(per_metric.mean()),
        recording_bootstrap_95_interval=np.quantile(boots,[.025,.975]).tolist(),
        bootstrap_fraction_same_order=float(np.mean(np.array(boots)>0))))
within=[]
for s in subjects:
    for i,p in enumerate(profiles[s]):
        ref=np.delete(profiles[s],i,axis=0).mean(0)
        sim=similarity(p,ref)
        within.append(dict(subject=s,recording=recording_names[s][i],same_minus_swapped_by_metric=difference(sim).tolist(),same_minus_swapped_mean=float(difference(sim).mean())))
summaries=[]
for s in subjects:
    q=[e for e in quality if e['subject']==s]
    summaries.append(dict(subject=s,recordings=len(q),**{k:np.median([e[k] for e in q],axis=0).tolist() for k in ('raw_median','baseline_rms','median_action_rms','median_prepare_rms','median_action_to_prepare_ratio','median_action_raw_band_correlation','first_60s_unnotched_band_median_hz','first_60s_50hz_power_fraction')}))
result=dict(original_order=True,source='Original 3-person baseline source hashes verified; S012 read from original backup, not current swapped CSV.',
    note='Channel similarity is exploratory evidence of relative matching, not proof of anatomical electrode identity. Bootstrap resamples recordings within each person; no population confidence claim.',
    subjects=summaries,recording_quality=quality,trial_signals=trial_signals,pairwise_word_selectivity=pairs,within_subject_recording_consistency=within,
    current_files='S011 original; S012 swapped; S015 original. This analysis makes no changes to recordings.',
    limitations=['No physically verified anatomical channel anchor.', 'Recordings from same person/day are not independent subjects.', 'Temporal alignment, articulation and contact differences can change similarity.', 'Stopped S012 recording excluded; analysis uses same 1500 valid trials as baseline.'])
(OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('SUMMARY',json.dumps(summaries),flush=True)
print('PAIRWISE',json.dumps(pairs),flush=True)
print('WITHIN',json.dumps(within),flush=True)
