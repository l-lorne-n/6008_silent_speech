"""Inspect uninterrupted quiet intervals; never concatenate disjoint rests."""
import json
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent/'rhythm'
OUT.mkdir(exist_ok=True)
SOURCE=OUT.parents[2]/'recordings/S05/session_001/20260924_123453_5ddcee'
data=np.genfromtxt(SOURCE/'samples.csv',delimiter=',',names=True)
raw=data['raw_ch0']; idx=data['device_sample_idx']; fs=1000
events=[json.loads(s) for s in (SOURCE/'events.jsonl').read_text(encoding='utf-8').splitlines()]
ack={e['event_id']:int(e['device_sample_idx']) for e in events if e['kind']=='marker_ack'}
phases=[e for e in events if e['kind']=='phase']
base=ack[phases[0]['event_id']]-int(idx[0]); end=ack[phases[1]['event_id']]-int(idx[0])
print('baseline',base,end,(end-base)/fs)
fig,ax=plt.subplots(4,1,figsize=(15,10),layout='constrained')
x=raw[base:end];t=np.arange(len(x))/fs
ax[0].plot(t,x,lw=.5);ax[0].set_title('Initial quiet baseline / raw ADC counts')
for a,band in zip(ax[1:],[(5,30),(1,15),(8,35)]):
    y=sosfiltfilt(butter(3,band,btype='bandpass',fs=fs,output='sos'),x)
    a.plot(t,y,lw=.7);a.set_title(f'{band} Hz / counts')
    print('band',band,'RMS',np.std(y[500:-500]))
    for polarity in [1,-1]:
        peaks,props=find_peaks(polarity*y,prominence=2*np.std(y[500:-500]),distance=350)
        peaks=peaks[(peaks>500)&(peaks<len(x)-500)]
        print(polarity, np.round(peaks/fs,3).tolist(),'intervals',np.round(np.diff(peaks)/fs,3).tolist())
        a.plot(t[peaks],y[peaks],'o',ms=3)
for a in ax:a.grid(alpha=.2);a.set_xlim(0,t[-1])
ax[-1].set_xlabel('Seconds from quiet baseline start')
fig.savefig(OUT/'baseline_exploration.png',dpi=150)
plt.close(fig)
