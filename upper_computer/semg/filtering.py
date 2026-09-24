"""Shared 1 kHz display filter design; raw recordings remain untouched."""
import numpy as np
from scipy.signal import butter, iirnotch, tf2sos


def display_sos():
    bandpass = butter(4, [20, 400], btype="bandpass", fs=1000, output="sos")
    b, a = iirnotch(50, Q=30, fs=1000)
    return np.vstack((bandpass, tf2sos(b, a)))


def display_filter_metadata(mode):
    if mode == "legacy":
        return dict(source="device_columns", host_filter_applied=False,
                    notch_hz=None, raw_recording_modified=False)
    return dict(source="host", host_filter_applied=True, fs_hz=1000,
                bandpass_hz=[20, 400], order=8, notch_hz=50, notch_q=30,
                notch_order=2, causal=True, envelope_ms=200,
                envelope_method="moving_mean_absolute",
                raw_recording_modified=False)
