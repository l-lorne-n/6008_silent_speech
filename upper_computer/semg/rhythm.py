"""Exploratory quiet-pulse indicator, not ECG or contact verification.

Uses an independent preview branch. Never alters acquisition or sEMG filters.
Only uninterrupted 1 kHz quiet windows may be supplied; disjoint rests are not
joined. Rates are pulse repetition estimates, not validated heart rates.
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt

WINDOW = 10000
ALGORITHM = "quiet_pulse_v1"


@dataclass
class RhythmResult:
    status: str
    detail: str
    rate: float | None = None
    amplitude: float | None = None
    interval_cv: float | None = None
    morphology: float | None = None
    peaks: list = field(default_factory=list)


def inspect_quiet(raw):
    """Evaluate 10 continuous seconds. Thresholds are exploratory, not clinical."""
    x = np.asarray(raw, dtype=float)
    if len(x) < WINDOW:
        return RhythmResult("waiting", f"请保持静息 {len(x)/1000:.1f}/10 秒")
    x = x[-WINDOW:]
    if not np.isfinite(x).all() or np.any((x <= 1) | (x >= 65534)):
        return RhythmResult("artifact", "波形无效或削顶，请检查接触和接线")
    # Large slow baseline shifts suggest movement or an unstable connection.
    blocks = x.reshape(20, 500)
    medians = np.median(blocks, axis=1)
    local_spread = np.median(np.abs(blocks-medians[:, None]))
    if np.ptp(medians) > max(1000, 8*local_spread):
        return RhythmResult("artifact", "基线明显移动，放松并检查接触")
    y = sosfiltfilt(butter(3, [5, 30], fs=1000, btype="bandpass", output="sos"), x)
    # A short energy envelope merges the positive/negative lobes of one pulse.
    envelope = np.sqrt(np.maximum(0, uniform_filter1d(y*y, size=60)))
    middle = envelope[500:-500]
    floor = float(np.median(middle))
    spread = float(np.median(np.abs(middle-floor)))
    prominence = max(20., 5*spread, 2*floor)
    peaks, _ = find_peaks(envelope, distance=400, prominence=prominence,
                          height=max(30., 3*floor))
    peaks = peaks[(peaks >= 500) & (peaks < len(x)-500)]
    empty = RhythmResult("uncertain", "未检出稳定节律；不等于接线异常")
    if len(peaks) < 6:
        return empty
    intervals = np.diff(peaks)/1000
    median_interval = float(np.median(intervals))
    cv = float(np.std(intervals)/np.mean(intervals))
    rate = 60/median_interval
    # Reject missing/doubled pulses and isolated movements. The supported range
    # is deliberately limited; outside it we make no physiological assertion.
    if not (45 <= rate <= 140) or cv > .15 or np.any(np.abs(intervals/median_interval-1) > .3):
        return empty
    if peaks[0] > 500+1.5*median_interval*1000 or peaks[-1] < 9500-1.5*median_interval*1000:
        return empty
    waves = np.array([y[p-120:p+120] for p in peaks])
    centered = waves-waves.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = centered/np.maximum(norms, 1e-12)
    template = np.median(normalized, axis=0)
    correlations = normalized@template/max(float(np.linalg.norm(template)), 1e-12)
    similarity = float(np.median(correlations))
    amplitudes = np.ptp(waves, axis=1)
    if (similarity < .75 or np.std(amplitudes)/max(np.mean(amplitudes), 1) > .5
            or max(amplitudes) > 3*np.median(amplitudes)):
        return empty
    return RhythmResult("detected", "检测到稳定的疑似心搏节律", rate,
                        float(np.median(amplitudes)), cv, similarity,
                        (peaks/1000).tolist())


class QuietRhythmMonitor:
    """Continuity and freshness guard for disposable GUI previews."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.raw = deque(maxlen=WINDOW)
        self.expected = None
        self.last_rx_ns = None
        self.last_eval_ns = 0
        self.result = RhythmResult("waiting", "请保持静息 0.0/10 秒")

    def feed(self, values, first, rx_ns, now_ns):
        if first is None or now_ns-rx_ns > 1_500_000_000:
            self.reset()
            return
        if self.expected is not None and first != self.expected:
            self.reset()
        self.raw.extend(values)
        self.expected = (first+len(values)) & 0xFFFFFFFF
        self.last_rx_ns = rx_ns

    def evaluate(self, now_ns):
        if self.last_rx_ns is None or now_ns-self.last_rx_ns > 1_500_000_000:
            self.reset()
            return RhythmResult("waiting", "等待连续新数据，请保持放松")
        if len(self.raw) < WINDOW:
            return RhythmResult("waiting", f"请保持静息 {len(self.raw)/1000:.1f}/10 秒")
        if not self.last_eval_ns or now_ns-self.last_eval_ns >= 1_000_000_000:
            self.result = inspect_quiet(self.raw)
            self.last_eval_ns = now_ns
        return self.result


def indicator_metadata():
    return dict(algorithm=ALGORITHM, exploratory=True, fs_hz=1000,
                window_seconds=10, bandpass_hz=[5, 30],
                supported_rate_per_min=[45, 140],
                continuous_quiet_only=True, verifies_contact=False,
                validated_heart_rate=False, raw_recording_modified=False)
