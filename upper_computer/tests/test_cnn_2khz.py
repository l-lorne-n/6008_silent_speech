"""Synthetic end-to-end CNN smoke test; no performance claim or real data."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.cnn-deps'))
sys.path.insert(0,str(ROOT/'experiments'))
torch=pytest.importorskip('torch')
import train_small_cnn as cnn
from semg import lda
from test_2khz import make_2k_recording


def test_2khz_cnn_preprocessing_training_reload_and_test(tmp_path, monkeypatch):
    words=['open','close','next','music','photo','previous','louder','quieter','start','stop']
    source=make_2k_recording(tmp_path/'synthetic',words=words,repetitions=5)
    baseline=lda.train([dict(path=str(source))],tmp_path/'lda',strategy='within_recording',
                       variant='notch50_bp500',channels=[0,1])
    report=baseline['report']
    rows,split=cnn.split_data(report)
    raw,lengths=cnn.extract_waveforms(rows,report)
    assert raw.shape==(50,2,4096) and (lengths==3000).all()
    normalized,scale=cnn.normalize(raw,lengths,split['train'])
    assert torch.count_nonzero(normalized[:,:,3000:])==0
    assert scale['fit_trials']==len(split['train'])
    out=tmp_path/'cnn'
    monkeypatch.setattr(cnn,'MAX_EPOCHS',2)
    monkeypatch.setattr(cnn,'SEEDS',[20261016])
    monkeypatch.setattr(cnn,'LENGTH',2048)
    monkeypatch.setattr(sys,'argv',['train_small_cnn.py','--baseline',baseline['folder'],'--output',str(out)])
    cnn.main()
    result=json.loads((out/'report.json').read_text(encoding='utf-8'))
    checkpoint=torch.load(out/'model.pt',map_location='cpu',weights_only=True)
    assert checkpoint['input_length']==4096
    assert result['protocol']['sampling_rate_hz']==2000
    assert result['protocol']['bandpass_hz']==[20,500]
    assert result['final_train_trials']==40 and result['test_trials']==10
    assert result['model_roundtrip_verified'] and result['input_hashes_unchanged']
    assert result['baseline_metrics']==report['metrics']
    assert set(checkpoint['training_fingerprints'])=={r['fingerprint'] for r in report['train']}
