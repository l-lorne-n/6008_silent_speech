"""Offline audit of quiet baseline and disjoint rests; no concatenation."""
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from semg.rhythm import inspect_quiet, indicator_metadata

OUT=Path(__file__).resolve().parent/'rhythm'
SOURCE=ROOT/'recordings/S05/session_001/20260924_123453_5ddcee'


def main():
    OUT.mkdir(exist_ok=True)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.iterdir() if p.is_file()}
    data=np.genfromtxt(SOURCE/'samples.csv',delimiter=',',names=True)
    raw=data['raw_ch0']; origin=int(data['device_sample_idx'][0])
    events=[json.loads(s) for s in (SOURCE/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    acks={e['event_id']:int(e['device_sample_idx']) for e in events if e['kind']=='marker_ack'}
    phases=[e for e in events if e['kind']=='phase']
    start=acks[phases[0]['event_id']]-origin
    x=raw[start:start+10000]
    result=inspect_quiet(x)
    sos=butter(3,[5,30],fs=1000,btype='bandpass',output='sos')
    y=sosfiltfilt(sos,x)
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False})
    fig,axes=plt.subplots(2,1,figsize=(14,7),layout='constrained')
    t=np.arange(len(x))/1000
    axes[0].plot(t,x,lw=.5); axes[0].set_title('连续静息原始 ADC（未改动）')
    axes[1].plot(t,y,lw=.8,color='#26845c')
    for peak in result.peaks: axes[1].axvline(peak,color='#bb7428',alpha=.6,lw=.8)
    axes[1].set_title(f'独立 5–30 Hz 分支：疑似心搏节律 {result.rate:.1f} 次/分；脉冲峰峰值中位数 {result.amplitude:.0f} counts')
    for ax in axes: ax.set(xlabel='相对静息开始 / s',ylabel='ADC counts',xlim=(0,10)); ax.grid(alpha=.15)
    fig.suptitle('S05 12:34 · 静息节律分析；没有同步 ECG，不能确认心脏来源')
    fig.savefig(OUT/'quiet_rhythm.png',dpi=150);plt.close(fig)
    rests=[];pairs=[];traces=[]
    for i,e in enumerate(phases[:-1]):
        if e['phase']!='rest':continue
        a=acks[e['event_id']]-origin;b=acks[phases[i+1]['event_id']]-origin
        segment=raw[a:b];z=sosfiltfilt(sos,segment)
        env=np.sqrt(np.maximum(0,uniform_filter1d(z*z,60)))
        mid=env[300:-300]; floor=float(np.median(mid)); spread=float(np.median(np.abs(mid-floor)))
        peaks,_=find_peaks(env,distance=400,prominence=max(20,5*spread,2*floor),height=max(30,3*floor))
        peaks=peaks[(peaks>=300)&(peaks<len(z)-300)]
        intervals=np.diff(peaks)/1000
        pairs.extend(intervals.tolist())
        rests.append(dict(trial_id=e['trial_id'],word=e['word'],duration_s=len(segment)/1000,
                          candidate_times_s=(peaks/1000).tolist(),adjacent_intervals_s=intervals.tolist()))
        traces.append((e,z,peaks))
    fig,axes=plt.subplots(4,2,figsize=(12,10),layout='constrained')
    # First eight rests, selected by time rather than appearance.
    for ax,(e,z,peaks) in zip(axes.flat,traces[:8]):
        t=np.arange(len(z))/1000;ax.plot(t,z,lw=.7,color='#26845c')
        for p in peaks: ax.axvline(p/1000,color='#bb7428',alpha=.6)
        ax.set(title=f"试次 {e['trial_id']} {e['word']} 后休息",xlabel='该休息段内 / s',ylabel='5–30 Hz / counts',ylim=(-2000,2000));ax.grid(alpha=.15)
    fig.suptitle('最早八个休息段（统一量程）；各段独立，短段不输出稳定心率')
    fig.savefig(OUT/'rest_examples.png',dpi=150);plt.close(fig)
    report=dict(source=str(SOURCE),algorithm=indicator_metadata(),baseline=asdict(result),
                baseline_interval_range_s=[float(min(np.diff(result.peaks))),float(max(np.diff(result.peaks)))],
                rest_segments=len(rests),rests_with_candidates=sum(bool(r['candidate_times_s']) for r in rests),
                rests_with_two_or_more=sum(len(r['candidate_times_s'])>=2 for r in rests),
                within_rest_intervals=pairs,rests=rests,source_sha256=hashes,
                user_update='User reports some Dupont wire connections were previously loose/disconnected; checked connections before this recording. Causal explanation is plausible but not a controlled comparison.',
                limitation='No synchronous ECG/PPG ground truth; this is pulse rhythm, not proven cardiac origin or verified contact.')
    for name,digest in hashes.items():assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==digest
    (OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['rests','source_sha256']},ensure_ascii=True))


if __name__=='__main__':main()
