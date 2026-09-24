import copy
import csv
import json

import numpy as np
import pytest

from semg import lda
from semg.acquisition import Simulator
from semg.session import Recorder, TrialEngine, schedule, WORDS
from semg.vocabulary import load_vocabulary, save_vocabulary, validate_words

TEN = [*WORDS, 'stop', 'start', 'yes', 'no', 'up', 'down']


def make_recording(folder, words, seed=123, rounds=4):
    """Synthetic independent trials, with valid preparation and action marks."""
    folder.mkdir()
    rng = np.random.default_rng(seed)
    meta = dict(mode='binary', sampling_rate_hz=1000, status='completed',
                subject_id='SYNTHETIC', words=words)
    (folder/'metadata.json').write_text(json.dumps(meta), encoding='utf-8')
    trials, events = [], []
    with (folder/'samples.csv').open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['device_sample_idx', 'raw_ch0'])
        for n, trial in enumerate(schedule(rounds, seed, words)):
            tid = trial['trial_id']; start = n*2500
            t = np.arange(2500)/1000
            k = words.index(trial['word'])
            raw = np.rint(32768 + np.where(t>=1.2, 100+40*k, 20)
                         * np.sin(2*np.pi*(70+17*k)*t) + rng.normal(0, 7, len(t))).astype(int)
            writer.writerows(zip(start+np.arange(2500), raw))
            trials.append(dict(trial, status='completed'))
            for phase, offset in [('prepare',0), ('action',1200), ('rest',2400)]:
                eid = f'{tid}_{phase}'
                events.extend([dict(kind='phase', phase=phase, trial_id=tid, event_id=eid),
                               dict(kind='marker_ack', event_id=eid, device_sample_idx=start+offset, rtt_ms=5)])
    for name, values in [('trials.jsonl', trials), ('events.jsonl', events)]:
        (folder/name).write_text('\n'.join(json.dumps(v) for v in values), encoding='utf-8')
    return folder


def test_vocabulary_persistence_validation_and_recording_labels(tmp_path):
    path = tmp_path/'vocabulary.json'
    assert load_vocabulary(path) == (list(WORDS), list(WORDS))
    selected = ['stop', 'open', 'yes']
    save_vocabulary(path, TEN, selected)
    assert load_vocabulary(path) == (TEN, selected)
    for words in [[], ['open','open'], ['../bad'], ['rest'], ['OPEN']]:
        with pytest.raises(ValueError): validate_words(words)
    old = path.read_bytes()
    with pytest.raises(ValueError): save_vocabulary(path, TEN, ['missing'])
    assert path.read_bytes() == old
    rec = Recorder(tmp_path, dict(subject_id='TEST', session_id=1, mode='simulation', words=selected))
    rec.close('completed', {})
    meta = json.loads((rec.path/'metadata.json').read_text())
    assert meta['words'] == selected
    assert meta['label_map'] == {'0':'rest', '1':'stop', '2':'open', '3':'yes'}


def test_ten_word_schedule_repeats_and_simulator():
    trials = schedule(2, 17, TEN)
    assert len(trials) == 20
    for start in (0, 10): assert {t['word'] for t in trials[start:start+10]} == set(TEN)
    assert all(t['label_id'] == TEN.index(t['word'])+1 for t in trials)
    events = []
    engine = TrialEngine(trials, dict(baseline=1, prepare=1, action=1, rest=1), events.append)
    engine.start(0); engine.tick(1_000_000_000)
    original = engine.current.copy()
    engine.pause(1_100_000_000)
    assert engine.trials[-1]['word'] == original['word']
    assert engine.trials[-1]['label_id'] == original['label_id']
    assert engine.trials[-1]['repeat_of'] == original['trial_id']
    sim = Simulator(); sim.word = 'stop'; sim.phase = 'action'; sim.start -= .03
    assert sim.read(8192)


@pytest.mark.parametrize('words', [TEN, ['stop','yes']])
def test_dynamic_model_roundtrip_and_subset_evaluation(tmp_path, words):
    folder = make_recording(tmp_path/'recording', words)
    before = {p.name:p.read_bytes() for p in folder.iterdir()}
    catalog, errors = lda.scan_recordings(folder)
    assert not errors and len(catalog[0]['trials']) == len(words)*4
    result = lda.train([dict(path=str(folder))], tmp_path/'models', train_percent=50)
    model = lda.load_model(result['model_path'])
    assert set(model['classes']) == set(words) and model['schema'] == 'semg_lda_v2'
    report = result['report']
    ids = [r['trial_id'] for r in report['test']]
    loaded = lda.evaluate(result['model_path'], [dict(path=str(folder), trial_ids=ids)], tmp_path/'models')
    assert loaded['report']['metrics'] == report['metrics']
    subset = [r for r in report['test'] if r['word'] == words[0]]
    limited = lda.evaluate(result['model_path'], [dict(path=str(folder), trial_ids=[r['trial_id'] for r in subset])], tmp_path/'models')
    assert limited['report']['metrics']['unknown_total'] == 0
    assert set(limited['report']['metrics']['missing_model_classes']) == set(words[1:])
    with pytest.raises(ValueError, match='已用于训练'):
        lda.evaluate(result['model_path'], [dict(path=str(folder))], tmp_path/'models')
    damaged = copy.deepcopy(model); damaged['classes'][1] = damaged['classes'][0]
    with pytest.raises(ValueError): lda.validate_model(damaged)
    assert {p.name:p.read_bytes() for p in folder.iterdir()} == before


def test_legacy_model_accepts_untrained_words_without_hiding_errors(tmp_path):
    folder = make_recording(tmp_path/'four', list(WORDS))
    result = lda.train([dict(path=str(folder))], tmp_path/'models', train_percent=50)
    model = lda.load_model(result['model_path']); model['schema'] = 'semg_lda_v1'
    old_path = tmp_path/'legacy.json'; lda.write_json(old_path, model)
    assert lda.load_model(old_path)['classes'] == list(WORDS)
    new_folder = make_recording(tmp_path/'ten', TEN, seed=234)
    result = lda.evaluate(old_path, [dict(path=str(new_folder))], tmp_path/'models')
    m = result['report']['metrics']
    assert m['known_total'] == 16 and m['unknown_total'] == 24
    assert set(m['unknown_words']) == set(TEN)-set(WORDS)
    assert np.shape(m['confusion_matrix']) == (10, 4)
    assert m['accuracy'] == m['correct']/40 and m['known_accuracy'] == m['correct']/16
    with open(result['folder']+'/predictions.csv', encoding='utf-8-sig') as f:
        predictions = list(csv.DictReader(f))
    unknown = [r for r in predictions if r['known_to_model'] == 'False']
    assert len(unknown) == 24 and all(r['correct'] == '0' and r['prediction'] in WORDS for r in unknown)
    unknown_ids = [int(r['trial_id']) for r in unknown]
    only_unknown = lda.evaluate(old_path, [dict(path=str(new_folder), trial_ids=unknown_ids)], tmp_path/'models')
    assert only_unknown['report']['metrics']['known_accuracy'] is None
    assert only_unknown['report']['metrics']['accuracy'] == 0
    # Manual training derives its vocabulary only from the training recordings.
    manual = lda.train([dict(path=str(folder),role='train'), dict(path=str(new_folder),role='test')],
                       tmp_path/'models', strategy='manual')
    assert manual['report']['classes'] == list(WORDS)
    assert manual['report']['metrics']['unknown_total'] == 24
