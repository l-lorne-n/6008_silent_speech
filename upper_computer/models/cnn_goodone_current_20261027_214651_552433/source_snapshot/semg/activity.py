"""Causal, sample-clocked activity segmentation. Never receives cue labels."""
from collections import deque
from dataclasses import asdict, dataclass
import numpy as np
from scipy.signal import sosfilt, sosfilt_zi
from .lda import filter_sos


@dataclass(frozen=True)
class ActivityConfig:
    calibration_ms: int = 5000
    onset_ms: int = 60
    release_ms: int = 350
    rearm_ms: int = 500
    pre_ms: int = 150
    post_ms: int = 150
    min_ms: int = 200
    max_ms: int = 3000
    on_ratio: float = 3.0
    off_ratio: float = 1.6

    def __post_init__(self):
        for name in ('calibration_ms','onset_ms','release_ms','rearm_ms','pre_ms','post_ms','min_ms','max_ms'):
            value = getattr(self, name)
            if type(value) is not int or value < 0 or value > 30000:
                raise ValueError('活动检测时间参数无效')
        if not (1000 <= self.calibration_ms and 0 < self.onset_ms <= self.min_ms < self.max_ms
                and self.release_ms >= self.post_ms and self.rearm_ms >= self.release_ms
                and self.pre_ms <= 1000 and self.post_ms <= 1000):
            raise ValueError('活动检测时间关系无效')
        if not 1 < self.off_ratio < self.on_ratio <= 20:
            raise ValueError('活动阈值比例无效')


def segment_features(filtered):
    """Same causal filtered segment and padding at training and inference."""
    x = np.asarray(filtered, dtype=float)
    if x.ndim == 2 and x.shape[1] in (1,2):
        return [v for c in range(x.shape[1]) for v in segment_features(x[:,c])]
    if x.ndim != 1 or len(x) < 200 or not np.isfinite(x).all():
        raise ValueError('活动片段不足200点或包含无效数值')
    qs = np.array_split(x, 4)
    return [float(np.log(np.sqrt(np.mean(q*q))+1e-8)) for q in qs] + [
        float(np.log(np.mean(np.abs(np.diff(q)))+1e-8)) for q in qs]


class ActivityDetector:
    def __init__(self, config=None, channels=1):
        if channels not in (1,2):raise ValueError('活动检测支持1或2个通道')
        self.channels = channels
        self.config = config or ActivityConfig()
        self.sos = filter_sos('notch50')
        self.reset()

    def reset(self):
        self.expected = None
        self.pending = np.empty(0) if self.channels == 1 else np.empty((0,self.channels))
        self.pending_first = None
        self.zi = None
        self.energy = deque(maxlen=5)  # 100 ms trailing RMS, 20 ms steps
        self.history = deque(maxlen=(self.config.max_ms+2000)//20+5)
        self.calibration = []
        self.elapsed = 0
        self.state = 'calibrating'
        self.on = self.off = None
        self.loud_ms = self.quiet_ms = 0
        self.start = self.last_active = None
        self.level = 0.0

    def status(self):
        return dict(state=self.state, level=float(np.max(self.level)), on=None if self.on is None else float(np.max(self.on)), off=None if self.off is None else float(np.max(self.off)),
                    channel_levels=np.atleast_1d(self.level).tolist(),channel_on=None if self.on is None else np.atleast_1d(self.on).tolist(),
                    calibration_remaining_s=max(0,(self.config.calibration_ms-self.elapsed)/1000))

    def feed(self, values, first):
        x = np.asarray(values, dtype=float)
        if self.channels == 1 and x.ndim == 2 and x.shape[1] == 1:x=x[:,0]
        valid_shape = x.ndim == 1 if self.channels == 1 else x.ndim == 2 and x.shape[1] == self.channels
        if not valid_shape or not np.isfinite(x).all() or np.any((x<0)|(x>65535)):
            raise ValueError('在线输入维度不匹配或不是有效16位ADC数据')
        if not len(x): return []
        result=[]
        if self.expected is not None and first != self.expected:
            self.reset()
            result.append(dict(kind='reset', reason='sample_gap', emitted=first))
        self.expected=first+len(x)
        if self.pending_first is None: self.pending_first=first
        self.pending=np.concatenate((self.pending,x))
        while len(self.pending)>=20:
            frame=self.pending[:20];self.pending=self.pending[20:]
            pos=self.pending_first;self.pending_first+=20
            if np.any((frame<=1)|(frame>=65534)):
                expected=self.expected;pending=self.pending.copy();pending_first=self.pending_first
                self.reset();self.expected=expected;self.pending=pending;self.pending_first=pending_first
                result.append(dict(kind='reset',reason='clipping',emitted=pos+20))
                continue
            if self.zi is None:
                self.zi=sosfilt_zi(self.sos)*(frame[0]-32768) if self.channels==1 else sosfilt_zi(self.sos)[:,:,None]*(frame[0]-32768)
            y,self.zi=sosfilt(self.sos,frame-32768,zi=self.zi,axis=0)
            self.history.append((pos,y))
            self.energy.append(np.mean(y*y,axis=0))
            self.level=np.sqrt(np.mean(self.energy,axis=0))
            end=pos+20;cfg=self.config;self.elapsed+=20
            if self.state=='calibrating':
                if self.elapsed>500:self.calibration.append(self.level)
                if self.elapsed>=cfg.calibration_ms:
                    med=np.median(self.calibration,axis=0);mad=np.median(np.abs(np.array(self.calibration)-med),axis=0)
                    self.on=np.maximum(20.,np.maximum(med*cfg.on_ratio,med+6*mad))
                    self.off=np.minimum(np.maximum(10.,np.maximum(med*cfg.off_ratio,med+3*mad)),self.on*.9)
                    self.state='idle'
                    result.append(dict(kind='calibrated',on=self.on.tolist(),off=self.off.tolist(),emitted=end))
                continue
            if self.state=='idle':
                self.loud_ms=self.loud_ms+20 if np.any(self.level>self.on) else 0
                if self.loud_ms>=cfg.onset_ms:
                    self.start=end-self.loud_ms-cfg.pre_ms
                    self.last_active=end;self.quiet_ms=0;self.state='active'
                continue
            if self.state=='refractory':
                self.quiet_ms=self.quiet_ms+20 if np.all(self.level<self.off) else 0
                if self.quiet_ms>=cfg.rearm_ms:
                    self.state='idle';self.loud_ms=0;self.quiet_ms=0
                continue
            if np.any(self.level>=self.off):
                self.last_active=end;self.quiet_ms=0
            else: self.quiet_ms+=20
            # Waiting for the release interval is not part of the active duration.
            if self.last_active-(self.start+cfg.pre_ms)>cfg.max_ms:
                result.append(dict(kind='rejected',reason='activity_too_long',start=self.start,end=end,emitted=end))
                self.state='refractory';self.quiet_ms=0
            elif self.quiet_ms>=cfg.release_ms:
                stop=self.last_active+cfg.post_ms
                duration=self.last_active-(self.start+cfg.pre_ms)
                if duration<cfg.min_ms:
                    result.append(dict(kind='rejected',reason='activity_too_short',start=self.start,end=stop,emitted=end))
                else:
                    chunks=[(p,v) for p,v in self.history if p+20>self.start and p<stop]
                    start0=chunks[0][0];data=np.concatenate([v for _,v in chunks])
                    data=data[max(0,self.start-start0):stop-start0]
                    result.append(dict(kind='segment',start=self.start,end=stop,emitted=end,
                                       features=segment_features(data)))
                self.state='refractory'  # release interval already counts toward rearm
        return result
