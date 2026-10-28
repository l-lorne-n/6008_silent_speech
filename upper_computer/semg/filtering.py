"""Rate-aware display filters; raw recordings remain untouched."""
import numpy as np
from scipy.signal import butter, iirnotch, tf2sos


def display_band(fs=1000):
    if fs not in (1000, 2000):
        raise ValueError('仅支持1000或2000 Hz完整采样')
    return [20, 500 if fs == 2000 else 400]


def display_sos(fs=1000):
    bandpass = butter(4, display_band(fs), btype="bandpass", fs=fs, output="sos")
    b, a = iirnotch(50, Q=30, fs=fs)
    return np.vstack((bandpass, tf2sos(b, a)))


def display_filter_metadata(mode, fs=1000):
    if mode == "legacy":
        return dict(source="device_columns", host_filter_applied=False,
                    notch_hz=None, raw_recording_modified=False)
    return dict(source="host", host_filter_applied=True, fs_hz=fs,
                bandpass_hz=display_band(fs), order=8, notch_hz=50, notch_q=30,
                notch_order=2, causal=True, envelope_ms=200,
                envelope_method="moving_mean_absolute",
                raw_recording_modified=False)
