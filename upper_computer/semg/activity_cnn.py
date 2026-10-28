"""Causal 2 kHz activity front end for replay experiments (no cue/word inputs).

This is deliberately separate from the legacy 1 kHz activity/LDA GUI pipeline.
Time is expressed in device samples; no event is flushed at end of file.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.signal import sosfilt, sosfilt_zi


@dataclass(frozen=True)
class GateConfig:
    fs: int = 2000
    step_ms: int = 20
    envelope_ms: int = 60
    calibration_ms: int = 5000
    onset_ms: int = 60
    release_ms: int = 320
    rearm_ms: int = 500
    pre_ms: int = 150
    post_ms: int = 150
    min_ms: int = 200
    max_ms: int = 2700
    on_ratio: float = 3.0
    off_ratio: float = 1.8

    def __post_init__(self):
        if not (self.fs == 2000 and self.step_ms == 20
                and self.envelope_ms % self.step_ms == 0
                and self.calibration_ms >= 1000
                and 0 < self.onset_ms <= self.min_ms < self.max_ms
                and self.release_ms >= self.post_ms
                and self.rearm_ms >= self.release_ms
                and 1 < self.off_ratio < self.on_ratio):
            raise ValueError('Invalid activity gate configuration')


class CausalRMSFrontend:
    """Chunk-invariant filtering and trailing RMS. Gaps require a fresh instance."""
    def __init__(self, sos, config=GateConfig()):
        self.sos = np.asarray(sos, dtype=float)
        self.cfg = config
        self.step = config.fs * config.step_ms // 1000
        self.pending = np.empty((0, 2), dtype=float)
        self.pending_first = self.expected = self.origin = None
        self.zi = None
        self.energy = deque(maxlen=config.envelope_ms // config.step_ms)

    def feed(self, raw, first):
        x = np.asarray(raw, dtype=float)
        if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all():
            raise ValueError('Expected finite two-channel samples')
        if ((x < 0) | (x > 65535)).any():
            raise ValueError('Expected 16-bit ADC values')
        if not len(x):
            return []
        if self.expected is not None and first != self.expected:
            raise ValueError('Sample gap: start a new frontend and recalibrate')
        if self.origin is None:
            self.origin = self.pending_first = first
        self.expected = first + len(x)
        self.pending = np.concatenate((self.pending, x))
        frames = []
        n = len(self.pending) // self.step * self.step
        if not n:
            return frames
        data, self.pending = self.pending[:n], self.pending[n:]
        centered = data - 32768
        if self.zi is None:
            self.zi = sosfilt_zi(self.sos)[:, :, None] * centered[0]
        y, self.zi = sosfilt(self.sos, centered, axis=0, zi=self.zi)
        for offset in range(0, n, self.step):
            part = y[offset:offset+self.step]
            self.energy.append(np.mean(part * part, axis=0))
            frames.append(dict(end=self.pending_first + offset + self.step,
                               filtered=part,
                               rms=np.sqrt(np.mean(self.energy, axis=0)),
                               clipped=bool(((data[offset:offset+self.step] <= 1) |
                                             (data[offset:offset+self.step] >= 65534)).any())))
        self.pending_first += n
        return frames


class ActivityGate:
    """Adaptive-to-initial-rest thresholds, then fixed for the recording.

    Receives only causal RMS/clipping frames. No labels, cues or future samples.
    A clipped frame cancels the active event and requires a quiet rearm period.
    """
    def __init__(self, config=GateConfig()):
        self.cfg = config
        self.origin = self.last_end = None
        self.state = 'calibrating'
        self.calibration = []
        self.on = self.off = None
        self.loud = self.quiet = 0
        self.start = self.last_active = None
        self.calibration_excluded_frames = 0
        self.calibration_settle_until = 0

    def feed(self, rms, end, clipped=False):
        c = self.cfg
        step = c.fs * c.step_ms // 1000
        if self.last_end is not None and end != self.last_end + step:
            raise ValueError('Non-contiguous RMS frames')
        if self.origin is None:
            self.origin = end - step
        self.last_end = end
        level = np.asarray(rms, dtype=float)
        if level.shape != (2,) or not np.isfinite(level).all():
            raise ValueError('Invalid RMS frame')
        elapsed = (end-self.origin)*1000/c.fs
        if self.state == 'calibrating':
            if clipped:
                self.calibration_settle_until = elapsed + 200
            valid = not clipped and elapsed > self.calibration_settle_until
            if elapsed > 500 and valid:
                self.calibration.append(level)
            elif elapsed > 500:
                self.calibration_excluded_frames += 1
            if elapsed >= c.calibration_ms:
                if len(self.calibration)*c.step_ms < 3000:
                    raise ValueError('Less than 3 seconds of clean initial calibration')
                med = np.median(self.calibration, axis=0)
                mad = np.median(np.abs(np.asarray(self.calibration)-med), axis=0)
                self.on = np.maximum(20., np.maximum(med*c.on_ratio, med+6*mad))
                self.off = np.minimum(np.maximum(10., np.maximum(med*c.off_ratio, med+3*mad)), self.on*.9)
                self.state = 'idle'
            return None
        if clipped:
            event = dict(kind='rejected', reason='clipping', emitted=end,
                         start=self.start if self.state == 'active' else end-step, end=end)
            self.state = 'refractory'
            self.loud = self.quiet = 0
            return event
        if self.state == 'idle':
            self.loud = self.loud+c.step_ms if np.any(level > self.on) else 0
            if self.loud >= c.onset_ms:
                self.start = end - self.loud*c.fs//1000
                self.last_active = end
                self.quiet = 0
                self.state = 'active'
            return None
        if self.state == 'refractory':
            self.quiet = self.quiet+c.step_ms if np.all(level < self.off) else 0
            if self.quiet >= c.rearm_ms:
                self.state = 'idle'
                self.loud = self.quiet = 0
            return None
        if np.any(level >= self.off):
            self.last_active = end
            self.quiet = 0
        else:
            self.quiet += c.step_ms
        duration_ms = (self.last_active-self.start)*1000/c.fs
        if duration_ms > c.max_ms or self.quiet >= c.release_ms:
            event = dict(kind='segment', active_start=self.start, active_end=self.last_active,
                         start=self.start-c.pre_ms*c.fs//1000,
                         end=self.last_active+c.post_ms*c.fs//1000, emitted=end)
            if duration_ms > c.max_ms:
                event.update(kind='rejected', reason='too_long', end=end)
            elif duration_ms < c.min_ms:
                event.update(kind='rejected', reason='too_short')
            self.state = 'refractory'
            if event['kind'] == 'segment':
                assert event['end'] <= end
            return event
        return None
