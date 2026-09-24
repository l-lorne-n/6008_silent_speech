import numpy as np
from scipy.signal import sosfilt

from semg.filtering import display_sos, display_filter_metadata


def test_notch_rejects_mains_preserves_other_signal_and_chunk_continuity():
    # A streamed two-tone signal must lose mains, retain 120 Hz, and have no
    # discontinuities from serial chunk boundaries.
    t = np.arange(10000) / 1000
    raw = 32768 + 1000 * np.sin(2*np.pi*50*t) + 300 * np.sin(2*np.pi*120*t)
    original = raw.copy()
    sos = display_sos()
    zi = np.zeros((len(sos), 2))
    chunks = []
    for batch in np.array_split(raw, 317):
        output, zi = sosfilt(sos, batch-32768, zi=zi)
        chunks.append(output)
    actual = np.concatenate(chunks)
    np.testing.assert_allclose(actual, sosfilt(sos, raw-32768), atol=1e-9)
    np.testing.assert_array_equal(raw, original)
    steady = actual[2000:]
    def amplitude(hz):
        return 2 * abs(np.mean(steady * np.exp(-2j*np.pi*hz*t[2000:])))
    assert amplitude(50) < 1
    assert 285 < amplitude(120) < 315
    # A fresh connection resets the complete cascade, including the notch.
    reset, _ = sosfilt(sos, raw[:20]-32768, zi=np.zeros((len(sos), 2)))
    np.testing.assert_allclose(reset, actual[:20])


def test_metadata_does_not_claim_notch_for_legacy():
    assert display_filter_metadata("binary")["notch_hz"] == 50
    assert display_filter_metadata("simulation")["causal"] is True
    assert display_filter_metadata("legacy")["host_filter_applied"] is False
