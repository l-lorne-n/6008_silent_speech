"""Read-only original-order old/new comparisons for user-confirmed same people."""
from pathlib import Path
from zipfile import ZipFile
from collections import Counter
from itertools import combinations
import sys,json,io,hashlib
import numpy as np
from scipy.signal import resample_poly
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from semg import lda
OUT=ROOT/'analysis/20261016_same_person_channel_alignment'
OUT.mkdir(exist_ok=False)
PAIRS=[('S01','S011','s1'),('S02','S012','s2'),('S05','S015','s5')]
CONFIG=dict(segments=6,metrics=['log_mean_abs_diff','log_mav','zc_count_zero','ssc_count_zero'])
WORDS=['open','close','next','music','louder','photo','previous','quieter','start','stop']
SOS=lda.filter_sos('notch50')
backup=ZipFile(ROOT/'history/data_corrections/20261016_S012_channel_swap/original_S012.zip')
base=json.loads((ROOT/'models/lda_loso_3p_20261016_165457_187939/report.json').read_text(encoding='utf-8'))
basefold=json.loads((Path(base['folds'][0]['folder'])/'report.json').read_text(encoding='utf-8'))
new_hashes={Path(s['path']).name:s['hashes'] for s in basefold['sources']}
expected_new={(r['subject'],r['recording'],r['trial_id']) for r in basefold['train']+basefold['test']}
all_profiles=[]; exclusions=[]; source_audit=[]; valid_new=set(); seen_signals=set(); all_rows=[]
good_hashes={}
for old,new,alias in PAIRS:
    good_hashes[old]={hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'recordings/goodone'/alias).rglob('samples.csv')}
sources=[]
for old,new,alias in PAIRS:
    for subject in (old,new):
        for mp in sorted((ROOT/'recordings'/subject).rglob('metadata.json')):
            m=json.loads(mp.read_text(encoding='utf-8'))
            if m.get('status')=='completed' and m.get('channel_count')==2 and m.get('articulation_mode')=='silent_mouthing' and m.get('condition')=='silent':
                sources.append((subject,mp.parent))
for i,(subject,folder) in enumerate(sources):
    snapshots={}
    for name in ('metadata.json','samples.csv','events.jsonl','trials.jsonl'):
        if subject=='S012': data=backup.read((folder.relative_to(ROOT/'recordings/S012')/name).as_posix())
        else: data=(folder/name).read_bytes()
        snapshots[name]=data
    hashes={name:hashlib.sha256(data).hexdigest() for name,data in snapshots.items()}
    if subject in ('S011','S012','S015'): assert hashes==new_hashes[folder.name]
    meta=json.loads(snapshots['metadata.json']);fs=meta['sampling_rate_hz'];assert fs in (1000,2000)
    assert meta.get('selected_channels')==[0,1] and not meta.get('channel_corrections')
    raw=np.loadtxt(io.BytesIO(snapshots['samples.csv']),delimiter=',',skiprows=1,usecols=(1,4,7),dtype=np.int64)
    assert np.all(np.diff(raw[:,0])==1)
    offset=int(raw[0,0]);events=[json.loads(x) for x in snapshots['events.jsonl'].splitlines() if x]
    trials=[json.loads(x) for x in snapshots['trials.jsonl'].splitlines() if x]
    acks={e['event_id']:e for e in events if e.get('kind')=='marker_ack'}
    assert len(acks)==sum(e.get('kind')=='marker_ack' for e in events)
    rows=[]
    for t in trials:
        try:
            assert t.get('status')=='completed' and t['word'] in WORDS,'Incomplete/rejected/non-shared label'
            phases=[e for e in events if e.get('kind')=='phase' and e.get('trial_id')==t['trial_id'] and e.get('phase') in ('prepare','action','rest')]
            assert len(phases)==3 and len({e['phase'] for e in phases})==3,'Incomplete phase events'
            marks={e['phase']:acks[e['event_id']]['device_sample_idx'] for e in phases}
            prep,action,rest=[marks[k] for k in ('prepare','action','rest')]
            assert offset<=prep<action<rest<=int(raw[-1,0])+1 and action-prep>=fs,'Invalid boundary'
            values=raw[prep-offset:rest-offset,1:]
            assert not np.any((values<=1)|(values>=65534)),'ADC rail in preparation/action'
            signal=hashlib.sha256(values.astype('<u2').tobytes()+str(action-prep).encode()).hexdigest()
            assert signal not in seen_signals,'Duplicate signal across included recordings'
            seen_signals.add(signal)
            if fs==2000:
                assert (action-prep)%2==0
                values=resample_poly(values.astype(float)-32768,1,2,axis=0)+32768
                action_offset=(action-prep)//2
            else: action_offset=action-prep
            feats=np.concatenate([lda.feature_vector(values[:,ch],action_offset,SOS,CONFIG,fs=1000) for ch in range(2)])
            rows.append(dict(subject=subject,recording=folder.name,trial_id=t['trial_id'],word=t['word'],features=feats.tolist()))
            if fs==2000: valid_new.add((subject,folder.name,t['trial_id']))
        except (AssertionError,KeyError,ValueError) as e:
            if str(e)=='Duplicate signal across included recordings': raise
            exclusions.append(dict(subject=subject,recording=folder.name,trial_id=t.get('trial_id'),reason=str(e)))
    counts=Counter(r['word'] for r in rows)
    assert all(counts[w]>=2 for w in WORDS),(folder,counts)
    profile=np.array([np.mean([r['features'] for r in rows if r['word']==w],axis=0) for w in WORDS]).reshape(10,2,4,6)
    all_profiles.append(dict(subject=subject,recording=folder.name,session=folder.parent.name,date=folder.name[:8],
        in_goodone=hashes['samples.csv'] in good_hashes.get(subject,set()),valid_trials=len(rows),profile=profile.tolist()))
    all_rows.extend(rows)
    source_audit.append(dict(subject=subject,path=str(folder),original_fs=fs,hashes=hashes))
    print('EXTRACT',i+1,'/',len(sources),subject,folder.name,'valid',len(rows),flush=True)
backup.close()
assert valid_new==expected_new
for src in source_audit:
    if src['subject']=='S012': continue
    for name,h in src['hashes'].items(): assert hashlib.sha256((Path(src['path'])/name).read_bytes()).hexdigest()==h

def similarity(a,b):
    a=a-a.mean(0);b=b-b.mean(0)
    a=a/np.maximum(a.std(0),1e-12);b=b/np.maximum(b.std(0),1e-12)
    return np.array([[[np.mean(a[:,c,m,:]*b[:,d,m,:]) for d in range(2)] for c in range(2)] for m in range(4)])
def delta(s): return (s[...,0,0]+s[...,1,1]-s[...,0,1]-s[...,1,0])/2
def test_pair(a,b,label):
    aa=np.array([x['profile'] for x in a]);bb=np.array([x['profile'] for x in b]);sim=similarity(aa.mean(0),bb.mean(0))
    rng=np.random.default_rng(20261016);boot=[]
    for _ in range(1000):
        sa=aa[rng.integers(len(aa),size=len(aa))].mean(0);sb=bb[rng.integers(len(bb),size=len(bb))].mean(0)
        boot.append(float(delta(similarity(sa,sb)).mean()))
    record_comparisons=[dict(recording=x['recording'],session=x['session'],same_minus_swapped=float(delta(similarity(np.array(x['profile']),bb.mean(0))).mean())) for x in a]
    return dict(comparison=label,old_subject=a[0]['subject'],new_subject=b[0]['subject'],old_recordings=len(a),new_recordings=len(b),
        old_trials=sum(x['valid_trials'] for x in a),new_trials=sum(x['valid_trials'] for x in b),
        similarity_by_metric=sim.tolist(),same_minus_swapped_by_metric=delta(sim).tolist(),same_minus_swapped_mean=float(delta(sim).mean()),
        bootstrap95=np.quantile(boot,[.025,.975]).tolist(),recording_comparisons=record_comparisons)
results=[]
for old,new,_ in PAIRS:
    a=[x for x in all_profiles if x['subject']==old];b=[x for x in all_profiles if x['subject']==new]
    results.append(test_pair(a,b,'all_eligible_old_recordings'))
    results.append(test_pair([x for x in a if x['in_goodone']],b,'goodone_old_subset'))
    dates=sorted({x['date'] for x in a})
    if len(dates)>1:
        for date in dates: results.append(test_pair([x for x in a if x['date']==date],b,'old_date_'+date))
    print('PAIR',old,new,json.dumps([r for r in results if r['old_subject']==old]),flush=True)
within=[]
for subject in sorted({x['subject'] for x in all_profiles}):
    p=[x for x in all_profiles if x['subject']==subject]
    for i,x in enumerate(p):
        ref=np.mean([y['profile'] for j,y in enumerate(p) if j!=i],axis=0)
        within.append(dict(subject=subject,recording=x['recording'],session=x['session'],same_minus_swapped=float(delta(similarity(np.array(x['profile']),ref)).mean())))
output=dict(identity_pairs=PAIRS,metric_names=CONFIG['metrics'],words=WORDS,results=results,within_subject=within,
    source_audit=source_audit,exclusions=exclusions,profiles=all_profiles,
    trials_by_subject=dict(Counter(r['subject'] for r in all_rows)),
    methodology='All channels in original capture order. Old 1kHz unchanged; new 2kHz resample_poly downsampled to 1kHz for this analysis only, using anti-alias filtering. Common causal 20-400Hz and 50Hz Q30 notch, preparation warmup, trim 150ms, six segments/four features. Equal weight per recording; compare word-selectivity correlations after within-channel centering/scaling per metric/time segment. Positive delta favors same order; negative favors swapped. 1000 recording bootstrap replicates are descriptive, not physical-wiring probabilities.',
    limitations=['Same person does not imply unchanged electrode position/contact/articulation.', 'Previous 3.3V operation and current 5V can alter transfer characteristics, not only gain.', 'No independently verified anatomical channel anchor in either collection.', 'Repeated sessions within a person are not independent subjects; exploratory matching statistic is not a validated wiring detector.'],
    current_files='Unmodified: S011 original, S012 remains swapped, S015 original. S012 original read from backup.',new_trial_set_matches_original_baseline=True)
(OUT/'audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(OUT/'trial_features.json').write_text(json.dumps(all_rows,ensure_ascii=False)+'\n',encoding='utf-8')
print('COUNTS',output['trials_by_subject'],'EXCLUDED',len(exclusions),flush=True)
print('OUT',OUT,flush=True)
