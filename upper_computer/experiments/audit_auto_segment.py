"""Independently verify trained membership, exports, and replay event counts."""
import csv
from pathlib import Path
import sys
import train_auto_segment_cnn as exp
from train_auto_segment_cnn import replay, np, torch, cnn, onnx


def main():
    out=Path(sys.argv[1]).resolve();rows=replay.read(out/'segments.json')
    dev=torch.load(out/'selection_checkpoint.pt',map_location='cpu',weights_only=True)
    final=torch.load(out/'model.pt',map_location='cpu',weights_only=True)
    expected_dev={r['fingerprint'] for r in rows if r['split']=='train'}
    expected_final={r['fingerprint'] for r in rows if r['split'] in ['train','validation']}
    test={r['fingerprint'] for r in rows if r['split']=='test'}
    assert set(dev['training_fingerprints'])==expected_dev
    assert set(final['training_fingerprints'])==expected_final
    assert not expected_final&test
    with np.load(out/'segments.npz') as z:
        for target,allowed in [(dev,expected_dev),(final,expected_final)]:
            ix=np.array([i for i,r in enumerate(rows) if r['fingerprint'] in allowed])
            _,norm=cnn.normalize(z['raw'],z['lengths'],ix)
            assert norm==target['normalization']
    old=torch.load(exp.OLD/'model.pt',map_location='cpu',weights_only=True)
    assert {k:tuple(v.shape) for k,v in old['state_dict'].items()}=={k:tuple(v.shape) for k,v in final['state_dict'].items()}
    q=replay.read(out/'export/quantization_selection.json')
    assert q['selected']==min(q['candidates'],key=lambda c:c['validation_loss'])
    assert set(q['calibration_fingerprints'])<=expected_dev
    assert set(replay.read(out/'export/final_calibration.json')['calibration_fingerprints'])<=expected_final
    g=onnx.load(out/'export/model_int8_qdq.onnx');onnx.checker.check_model(g)
    initializers={v.name:v for v in g.graph.initializer};producers={o:n for n in g.graph.node for o in n.output}
    weight_types=[]
    for n in g.graph.node:
        if n.op_type in ['Conv','Gemm']:
            dq=producers[n.input[1]];assert dq.op_type=='DequantizeLinear'
            typ=initializers[dq.input[0]].data_type
            assert typ==onnx.TensorProto.INT8;weight_types.append(typ)
            activation=producers[n.input[0]];assert activation.op_type=='DequantizeLinear'
            assert initializers[activation.input[2]].data_type==onnx.TensorProto.INT8
    assert len(weight_types)==4
    report=replay.read(out/'comparison.json');records=replay.read(exp.REPLAY/'recordings.json')
    truth={(r['recording'],t['trial_id']):t for r in records for t in r['annotations'] if t['split']=='test'}
    audits={}
    for name in ['old_fp32','old_int8','new_fp32','new_int8']:
        events=replay.read(out/(name+'_events.json'));pred=[e for e in events if e['candidate'] is not None]
        correct=clean=detected=rest=0
        for key,t in truth.items():
            owned=[e for e in pred if (e['recording'],e['owner_trial'])==key]
            matched=[e for e in owned if e['matched_trial']==t['trial_id']]
            hits=[e for e in matched if e['candidate']==t['word']]
            correct+=bool(hits);clean+=len(owned)==1 and bool(hits);detected+=bool(matched)
            for e in owned:
                overlap=max(0,min(e['active_end'],t['context_end'])-max(e['active_start'],t['rest']+600))
                rest+=overlap>=.5*(e['active_end']-e['active_start'])
        expected=report['results'][name]['overall']
        assert (correct,clean,detected,rest)==(expected['correct_unique_trials'],expected['clean_correct_trials'],expected['detected_trials'],expected['rest_proxy_outputs'])
        audits[name]=dict(correct=correct,clean=clean,detected=detected,rest_outputs=rest,outputs=len(pred))
    for path,h in replay.read(out/'source_sha256.json').items():assert replay.digest(path)==h
    selected=replay.read(out/'selected_training.json')
    losses=[e['validation_loss'] for h in replay.read(out/'selection_history.json') for e in h['epochs']]
    assert selected['validation_loss']<=min(losses)+1e-4
    assert replay.digest(out/'segments.npz')==replay.read(out/'dataset_audit.json')['segments_npz_sha256']
    result=dict(development_training_segments=len(expected_dev),final_training_segments=len(expected_final),
                training_and_calibration_membership_verified=True,normalization_train_only_verified=True,
                architecture_shapes_identical=True,quantized_conv_and_dense_layers=4,
                selected_validation_losses_verified=True,independent_event_counts=audits,
                original_sources_models_and_cache_unchanged=True,segment_dataset_unchanged=True)
    replay.write(out/'independent_audit.json',result)
    print(result,flush=True)


if __name__=='__main__':main()
