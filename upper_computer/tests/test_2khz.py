"""New 2 kHz acquisition must preserve physical time and old model semantics."""
import csv
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import sosfreqz

from semg import lda
from semg.acquisition import Acquisition, Simulator
from semg.filtering import display_sos, display_filter_metadata
from semg.protocol import encode, INFO, DATA, Decoder
from semg.session import schedule
from export_trials import export


def make_2k_recording(folder, words=None, repetitions=5):
    words = words or ['open', 'close']
    folder.mkdir(parents=True)
    trials = schedule(repetitions, 31, words)
    fs = 2000
    n = 4000+len(trials)*7000
    raw = np.full((n, 2), 32768., dtype=float)
    raw += np.random.default_rng(43).normal(0, 4, raw.shape)
    events = []
    for i, trial in enumerate(trials):
        start = 3000+i*7000
        k = words.index(trial['word'])
        t = np.arange(3600)/fs
        raw[start+2400:start+6000, 0] += (150+100*k)*np.sin(2*np.pi*(120+30*k)*t)
        raw[start+2400:start+6000, 1] += (200+50*k)*np.sin(2*np.pi*(440-25*k)*t)
        for phase, offset in [('prepare', 0), ('action', 2400), ('rest', 6000)]:
            eid = f'{i}_{phase}'
            events += [dict(kind='phase',phase=phase,trial_id=trial['trial_id'],event_id=eid),
                       dict(kind='marker_ack',event_id=eid,device_sample_idx=start+offset,rtt_ms=2)]
    metadata = dict(mode='binary', status='completed', subject_id='SYNTHETIC_2K', session_id=1,
                    synthetic_test_fixture=True, sampling_rate_hz=fs, channel_count=2, selected_channels=[0,1],
                    words=words, condition='silent', articulation_mode='silent_mouthing', trial_order=trials)
    (folder/'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
    with (folder/'samples.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f); writer.writerow(['device_sample_idx', 'raw_ch0', 'raw_ch1'])
        writer.writerows((i, *v) for i,v in enumerate(np.rint(raw).astype(int)))
    for name, rows in [('events.jsonl', events), ('trials.jsonl', [dict(t,status='completed') for t in trials])]:
        (folder/name).write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
    return folder


def test_500hz_is_in_band_at_2khz_and_legacy_filter_unchanged():
    np.testing.assert_array_equal(display_sos(2000), lda.filter_sos('notch50_bp500'))
    np.testing.assert_array_equal(display_sos(1000), lda.filter_sos('notch50'))
    _, h = sosfreqz(display_sos(2000), worN=[10, 50, 120, 450, 500, 800], fs=2000)
    magnitude = np.abs(h)
    assert magnitude[0] < .1 and magnitude[1] < 1e-9
    assert magnitude[2] > .98 and magnitude[3] > .85
    assert .70 < magnitude[4] < .71 and magnitude[5] < .02
    assert display_filter_metadata('binary', 2000)['bandpass_hz'] == [20, 500]
    with pytest.raises(ValueError): display_sos(1500)


def test_info_rate_change_rejected_and_dual_raw_order_preserved(tmp_path):
    a = Acquisition('binary', '', tmp_path, [0,1])
    a.consume(encode(INFO,0,0,struct.pack('<IHHI',2000,2,16,0),version=2),0)
    assert a.info['sampling_rate_hz'] == 2000
    a.consume(encode(DATA,1,0,struct.pack('<4H',100,300,200,400),2,version=2),1)
    assert a.preview.get()['rows'] == [(100,'','',300),(200,'','',400)]
    with pytest.raises(RuntimeError, match='采样率'):
        a.consume(encode(INFO,2,2,struct.pack('<IHHI',1000,2,16,0),version=2),2)
    b = Acquisition('binary', '', tmp_path, [0])
    with pytest.raises(RuntimeError, match='不支持'):
        b.consume(encode(INFO,0,0,struct.pack('<IHHI',1500,1,16,0)),0)


def test_simulator_2khz_reports_rate_and_contiguous_samples():
    sim = Simulator(2, 2000)
    sim.start -= .1
    frames = Decoder().feed(sim.read(8192))
    assert struct.unpack('<IHHI', frames[0].payload)[:3] == (2000,2,16)
    data = [f for f in frames if f.kind == DATA]
    assert sum(f.count for f in data) >= 200
    assert all(b.first == a.first+a.count for a,b in zip(data,data[1:]))


def test_2khz_lda_training_reload_export_and_wrong_rate_guard(tmp_path):
    path = make_2k_recording(tmp_path/'data')
    from semg.streaming import recording_stream
    with pytest.raises(ValueError, match='仅支持1000'):
        recording_stream(path)
    catalog, errors = lda.scan_recordings(path)
    assert not errors and not catalog[0]['disabled_reason'] and catalog[0]['sampling_rate_hz'] == 2000
    run = lda.train([dict(path=str(path))], tmp_path/'models', strategy='within_recording',
                    variant='notch50_bp500', channels=[0,1])
    model = lda.load_model(run['model_path'])
    assert model['preprocessing'] == dict(fs=2000,center=32768,trim_samples=300,prepare_min_samples=2000,trial_isolated=True)
    test = [dict(path=str(path),trial_ids=[r['trial_id'] for r in run['report']['test']])]
    assert lda.evaluate(run['model_path'],test,tmp_path/'models')['report']['metrics'] == run['report']['metrics']
    with pytest.raises(ValueError, match='采样率'):
        lda.extract_recording(dict(path=str(path)), 'notch50', channels=[0,1])
    model['preprocessing']['fs'] = 1000
    with pytest.raises(ValueError, match='预处理'): lda.validate_model(model)
    manifest = export(path)
    assert all(r['exported'] and r['samples']==3600 for r in manifest)
    saved = np.load(next((path/'cue_windows').glob('*.npz')))
    assert saved['fs'] == 2000 and saved['raw'].shape == (3600,2)


def test_feature_crop_and_frequency_use_seconds_at_2khz():
    fs=2000; t=np.arange(5600)/fs
    raw=32768+300*np.sin(2*np.pi*450*t)
    features=lda.feature_vector(raw,2400,lda.filter_sos('notch50_bp500'),dict(segments=4,metrics=['mean_frequency_hz']),fs=fs)
    np.testing.assert_allclose(features,450,atol=1)
    # 600 ms action - 300 ms trimming is too short (at either sample rate).
    with pytest.raises(ValueError,match='400 ms'):
        lda.feature_vector(raw[:3600],2400,lda.filter_sos('notch50_bp500'),fs=fs)


def test_firmware_source_rate_and_wire_budget_agree():
    root=Path(__file__).resolve().parents[1]
    stream=(root/'firmware/stream_v2.inc').read_bytes()
    assert stream == (root/'firmware/Test/Core/Src/stream_v2.inc').read_bytes()
    assert b'#define SAMPLE_RATE_HZ 2000U' in stream
    assert b'put32(payload,SAMPLE_RATE_HZ)' in stream
    main=(root/'firmware/Test/Core/Src/main.c').read_bytes()
    assert b'htim3.Init.Prescaler = 239;' in main
    assert b'htim3.Init.Period = 1000000U / SAMPLE_RATE_HZ - 1U;' in main
    assert b'huart1.Init.BaudRate = 230400;' in main
    # 100-byte dual frames, 20 time points each, plus generous control allowance.
    assert (2000/20*100+200)*10 < 230400*.5
