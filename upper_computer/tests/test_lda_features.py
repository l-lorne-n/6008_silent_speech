import copy
import json
import numpy as np
import pytest
from scipy.signal import sosfilt,sosfilt_zi
from semg import lda
from test_vocabulary import make_recording, TEN


def test_defaults_exactly_preserve_legacy_features():
    rng=np.random.default_rng(123);raw=32768+rng.normal(0,100,3500);sos=lda.filter_sos('notch50')
    y,_=sosfilt(sos,raw-32768,zi=sosfilt_zi(sos)*(raw[0]-32768))
    parts=np.array_split(y[1650:-150],4)
    old=np.array([np.log(np.sqrt(np.mean(q*q))+1e-8) for q in parts]+[np.log(np.mean(np.abs(np.diff(q)))+1e-8) for q in parts])
    np.testing.assert_array_equal(lda.feature_vector(raw,1500,sos),old)
    assert lda.feature_names()==lda.FEATURES


def test_spectral_centroid_distinguishes_frequency_not_amplitude():
    t=np.arange(300)/1000
    for hz in [60,120,240]:
        x=100*np.sin(2*np.pi*hz*t)
        assert lda.segment_metric(x,'mean_frequency_hz')==pytest.approx(hz,abs=.1)
        assert lda.segment_metric(x*3,'mean_frequency_hz')==pytest.approx(hz,abs=.1)
    assert lda.segment_metric(np.zeros(300),'mean_frequency_hz')==0


def test_time_domain_features_known_values_and_scaling():
    x = np.array([-3., 4., -3., 0., 4.])
    expected = {'rms': np.sqrt(10), 'mav': 2.8, 'zcr_zero': .5, 'waveform_length': 21.}
    for metric, value in expected.items():
        assert lda.segment_metric(x, metric) == pytest.approx(value)
        assert lda.segment_metric(-2*x, metric) == pytest.approx(value if metric == 'zcr_zero' else 2*value)
        assert lda.segment_metric(np.zeros(5), metric) == 0
    assert lda.segment_metric([1.], 'zcr_zero') == 0
    assert lda.segment_metric([-1., 1., -1.], 'zcr_zero') == 1


def test_requested_features_roundtrip(tmp_path):
    config = dict(segments=6, metrics=['rms', 'mav', 'zcr_zero', 'waveform_length'])
    test_variable_features_roundtrip_and_train_only_scaling(tmp_path, config)


@pytest.mark.parametrize('config',[dict(segments=6,metrics=list(lda.FEATURE_LABELS)),dict(segments=1,metrics=['log_mav']),dict(segments=12,metrics=['mean_frequency_hz','log_rms'])])
def test_variable_features_roundtrip_and_train_only_scaling(tmp_path,config):
    folder=make_recording(tmp_path/'source',TEN)
    run=lda.train([dict(path=str(folder))],tmp_path/'models',train_percent=50,feature_settings=config)
    model=lda.load_model(run['model_path']);report=run['report'];d=config['segments']*len(config['metrics'])
    assert len(model['scaler_mean'])==d and model['feature_config']==config
    assert model['feature_names']==lda.feature_names(config)
    np.testing.assert_allclose(model['scaler_mean'],np.mean([r['features'] for r in report['train']],axis=0))
    test=lda.evaluate(run['model_path'],[dict(path=str(folder),trial_ids=[r['trial_id'] for r in report['test']])],tmp_path/'models')
    assert test['report']['metrics']==report['metrics'] and test['report']['feature_config']==config
    # Raw trial identity is independent of feature settings.
    old,_,_=lda.extract_recording(dict(path=str(folder)),'notch50')
    new,_,_=lda.extract_recording(dict(path=str(folder)),'notch50',config)
    assert [r['fingerprint'] for r in old]==[r['fingerprint'] for r in new]


@pytest.mark.parametrize('schema',['semg_lda_v1','semg_lda_v2'])
def test_legacy_models_without_config_load_and_test(tmp_path,schema):
    folder=make_recording(tmp_path/'source',list(lda.WORDS))
    run=lda.train([dict(path=str(folder))],tmp_path/'models',train_percent=50)
    m=lda.load_model(run['model_path']);m['schema']=schema;m.pop('feature_config')
    path=tmp_path/'old.json';lda.write_json(path,m)
    assert lda.model_feature_config(lda.load_model(path))==lda.DEFAULT_FEATURE_CONFIG
    tested=lda.evaluate(path,[dict(path=str(folder),trial_ids=[r['trial_id'] for r in run['report']['test']])],tmp_path/'models')
    assert tested['report']['metrics']==run['report']['metrics']


@pytest.mark.parametrize('config',[dict(segments=0,metrics=['log_rms']),dict(segments=13,metrics=['log_rms']),dict(segments=True,metrics=['log_rms']),dict(segments=6,metrics=[]),dict(segments=6,metrics=['foo']),dict(segments=6,metrics=['log_rms','log_rms'])])
def test_invalid_feature_config(config):
    with pytest.raises(ValueError):lda.feature_config(config)


def test_too_short_segments_rejected():
    with pytest.raises(ValueError,match='每段至少50'):
        lda.feature_vector(np.ones(1800)*32768,1000,lda.filter_sos('notch50'),dict(segments=12,metrics=['log_rms']))


def test_model_config_tampering_rejected(tmp_path):
    folder=make_recording(tmp_path/'source',TEN)
    run=lda.train([dict(path=str(folder))],tmp_path/'models',train_percent=50,feature_settings=dict(segments=6,metrics=list(lda.FEATURE_LABELS)))
    m=lda.load_model(run['model_path'])
    mutations=[lambda v:v.pop('feature_config'),lambda v:v['feature_config'].update(segments=4),lambda v:v['feature_config']['metrics'].reverse(),lambda v:v.update(schema='semg_lda_v2'),lambda v:v['coef'][0].pop()]
    for mutation in mutations:
        damaged=copy.deepcopy(m);mutation(damaged)
        with pytest.raises(ValueError):lda.validate_model(damaged)
