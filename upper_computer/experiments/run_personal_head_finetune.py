"""User-run classifier-only adaptation, paired with the completed full-weight run.

Reuse exact source episodes, epoch window order, evaluation runs and hyperparameters.
Never edit sealed prior scripts or models. All five heads start from the base model.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import shutil
import sys
import tempfile
import traceback

import run_personal_sliding_finetune as previous

base = previous.base
np, torch, nn, io = previous.np, previous.torch, previous.nn, previous.io
SUBJECTS = previous.SUBJECTS
DEFAULT_REFERENCE = base.ROOT/'models/personal_sliding_20261022_121427_504367'


def freeze_features(net):
    if not isinstance(net.layers[-1], nn.Linear):
        raise ValueError('Expected a final linear classifier')
    net.requires_grad_(False)
    net.layers[-1].requires_grad_(True)
    names = [n for n,p in net.named_parameters() if p.requires_grad]
    assert names == ['layers.23.weight', 'layers.23.bias']
    assert sum(p.numel() for p in net.parameters() if p.requires_grad) == 363
    return names


def schedule_indices(history, rows, plan, epochs):
    """Validate and reuse the actual prior batches, not merely the same RNG seed."""
    lookup = {r['fingerprint']: i for i,r in enumerate(rows)}
    assert len(lookup) == len(rows)
    assert len(history) == epochs
    membership = {}
    for e,episode in enumerate(plan['episodes']):
        for i in episode['indices']:
            assert i not in membership
            membership[i] = e
    result = []
    for epoch,entry in enumerate(history,1):
        assert entry['epoch'] == epoch
        ix = [lookup[f] for f in entry['window_fingerprints']]
        assert len(ix)==22 and len(set(ix))==22
        assert Counter(rows[i]['label'] for i in ix) == {c:2 for c in range(11)}
        assert all(i in membership for i in ix)
        assert Counter(membership[i] for i in ix) == {e:1 for e in range(22)}
        result.append(ix)
    return result


def verify_trace_grid(trace, records, window_ms):
    expected = [(r['recording'],run,end) for r in records
                for run,(a,b) in enumerate(base.split_runs(r,'test'))
                for end in base.window_ends(r,a,b,window_ms*2)]
    actual = list(zip(trace['recording'].tolist(),trace['run'].tolist(),trace['end'].tolist()))
    assert actual == expected, 'Replay windows differ from the frozen evaluation partition'
    assert trace['probabilities'].shape == (len(expected),11)
    assert np.isfinite(trace['probabilities']).all()


def prepare(out, reference):
    assert io.read(reference/'completed.json')['complete']
    hashes = {}
    for marker in ['inputs_sha256.json','partition_sha256.json','training_sha256.json','export_sha256.json']:
        base.check_seal(reference,marker)
        hashes.update(io.read(reference/marker))
        hashes[str(reference/marker)] = io.digest(reference/marker)
    old = io.read(reference/'protocol.json')
    source = Path(old['base'])
    assert io.digest(source/'model.pt') == old['base_checkpoint_sha256']
    cp = torch.load(source/'model.pt',map_location='cpu',weights_only=True)
    rows = io.read(source/f'windows_{cp["window_ms"]}.json')
    records = io.read(reference/'evaluation_records.json')
    plans = io.read(reference/'personal_calibration.json')
    previous.audit_partition(records,rows,plans,cp['classes'])
    assert sorted(plans) == SUBJECTS
    remaining = Counter(r['subject'] for r in records for t in r['annotations'] if t['split']=='test')
    assert dict(remaining) == old['remaining_test'] and sum(remaining.values()) == 399
    for key in ['classes','normalization','window_ms']:
        assert cp[key] == old[key]
    ptq = io.read(reference/'ptq_indices.json')
    assert all(rows[i]['split'] in ['train','validation'] for i in ptq)
    assert [rows[i]['fingerprint'] for i in ptq] == io.read(source/'export/calibration_manifest.json')['calibration_fingerprints']
    assert old['decoder'] == io.read(source/'decoder_selection.json')['selected']['config']
    assert old['percentile'] == io.read(source/'export/quantization_selection.json')['selected']['percentile']
    net = base.restore(cp)
    names = freeze_features(net)
    config = deepcopy(old)
    config.update(schema='personal_head_two_shot_v1',reference=str(reference),
        trainable_parameters=363,trainable_names=names,
        training='Only final Linear weight/bias; exact prior epoch batches/order, Adam, LR, seed and epochs. '
                 'BN parameters AND buffers frozen. Same training dropout as full-weight comparison. '
                 'No feature cache or test-based selection. Each subject starts from original base.',
        comparison='Base/full-model predictions reused from the completed reference, re-decoded and rescored; '
                   'head-only model replayed on the identical window grid. FP32/INT8 separately.',
        limitations=old['limitations']+' Head-only experiment was proposed after inspecting full-weight results; '
                    'this is exploratory follow-up on the same holdout, not an independent confirmation.')
    schedules = {}
    extra = [reference/'completed.json',reference/'comparison.json',Path(__file__).resolve()]
    old_results = io.read(reference/'comparison.json')
    accumulated = defaultdict(list)
    for s in SUBJECTS:
        history_path = reference/s/'training_history.json'
        history = io.read(history_path)
        schedules[s] = schedule_indices(history,rows,plans[s],old['epochs'])
        full = torch.load(reference/s/'model.pt',map_location='cpu',weights_only=True)
        meta = full['personal_adaptation']
        assert meta['subject']==s and meta['base_checkpoint_sha256']==old['base_checkpoint_sha256']
        assert all(meta[k]==old[k] for k in ['epochs','learning_rate','seed'])
        assert meta['calibration_fingerprints'] == [rows[i]['fingerprint'] for i in sorted({i for ix in schedules[s] for i in ix})]
        own = [r for r in records if r['subject']==s]
        for kind in ['fp32','int8']:
            for mode in ['base','personal']:
                name = f'{mode}_{kind}'
                path = reference/s/f'{name}_trace.npz'
                with np.load(path) as trace:
                    verify_trace_grid(trace,own,cp['window_ms'])
                    events = base.decode(trace,old['decoder'],cp['classes'])
                metrics,_,_ = previous.score(events,own,cp['classes'])
                assert metrics == old_results['by_subject'][s][name]
                accumulated[name].extend(events)
                extra.append(path)
        extra.append(history_path)
        print(f'PRECHECK {s}: same 22 sources, {old["epochs"]} exact batches, {remaining[s]} test trials',flush=True)
    for name,events in accumulated.items():
        assert previous.score(events,records,cp['classes'])[0] == old_results['overall'][name]
    for p in extra:
        hashes[str(p)] = io.digest(p)
    io.write(out/'inputs_sha256.json',hashes)
    io.write(out/'protocol.json',config)
    for name in ['personal_calibration.json','evaluation_records.json','ptq_indices.json','calibration_windows.csv']:
        shutil.copy2(reference/name,out/name)
        assert io.digest(reference/name)==io.digest(out/name)
    io.write(out/'epoch_window_indices.json',schedules)
    snapshot = out/'source_snapshot';snapshot.mkdir()
    for p in [Path(__file__),Path(previous.__file__),Path(base.__file__)]:
        shutil.copy2(p,snapshot/p.name)
    base.seal(out,'partition_sha256.json',[out/n for n in ['protocol.json','personal_calibration.json',
        'evaluation_records.json','ptq_indices.json','epoch_window_indices.json']])
    return cp,rows,records,plans,config,schedules


def train_person(cp, rows, waves, plan, subject, config, schedule, target):
    base.cnn.set_seed(config['seed'])
    net = base.restore(cp)
    trainable = freeze_features(net)
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=config['learning_rate'])
    history=[]
    for epoch,ix in enumerate(schedule,1):
        x = base.batch(rows,ix,waves,cp['normalization'])
        y = [rows[i]['label'] for i in ix]
        # The prior step keeps training dropout, freezes BN statistics and clips
        # gradients. Frozen parameters have no gradient, including BN affine.
        loss,accuracy = previous.adaptation_step(net,optimizer,x,y)
        history.append(dict(epoch=epoch,loss=loss,calibration_batch_accuracy=accuracy,
                            window_fingerprints=[rows[i]['fingerprint'] for i in ix]))
        if epoch==1 or epoch%5==0 or epoch==len(schedule):
            print(f'HEAD ONLY {subject} {epoch}/{len(schedule)} loss={loss:.5f}',flush=True)
    frozen = [n for n in cp['state_dict'] if n not in trainable]
    assert all(torch.equal(net.state_dict()[n],cp['state_dict'][n]) for n in frozen)
    assert all(p.grad is None for n,p in net.named_parameters() if n not in trainable)
    changed = [n for n in trainable if not torch.equal(net.state_dict()[n],cp['state_dict'][n])]
    assert changed, 'Classifier did not update'
    used = sorted({i for ix in schedule for i in ix})
    personal = deepcopy(cp)
    personal['state_dict'] = net.state_dict()
    personal['personal_adaptation'] = dict(subject=subject,mode='classifier_only',
        base_checkpoint_sha256=config['base_checkpoint_sha256'],epochs=len(schedule),
        learning_rate=config['learning_rate'],seed=config['seed'],trainable_names=trainable,
        calibration_fingerprints=[rows[i]['fingerprint'] for i in used],source_episodes=plan['episodes'],
        remaining_test_used=False)
    personal['training_fingerprints'] = cp['training_fingerprints'] + personal['personal_adaptation']['calibration_fingerprints']
    torch.save(personal,target/'model.pt')
    reloaded=torch.load(target/'model.pt',map_location='cpu',weights_only=True)
    assert all(torch.equal(v,reloaded['state_dict'][n]) for n,v in net.state_dict().items())
    io.write(target/'training_history.json',history)
    io.write(target/'freeze_audit.json',dict(trainable_parameters=363,trainable_names=trainable,
        changed_names=changed,frozen_state_entries=len(frozen),all_frozen_bitwise_identical=True))
    return personal


def compare(out, cp, records, config):
    reference = Path(config['reference'])
    all_events=defaultdict(list);results={}
    for s in SUBJECTS:
        own=[r for r in records if r['subject']==s]
        target=out/s; waves=base.load_waves(own)
        head=torch.load(target/'model.pt',map_location='cpu',weights_only=True)
        group={}
        for kind in ['fp32','int8']:
            trials_by_mode={}
            for mode in ['base','full','head']:
                name=f'{mode}_{kind}'
                if mode=='head':
                    filename='model_fp32.onnx' if kind=='fp32' else 'model_int8_qdq.onnx'
                    trace=base.infer_runs(target/filename,head,own,waves,'test')
                    verify_trace_grid(trace,own,cp['window_ms'])
                    np.savez_compressed(target/f'{name}_trace.npz',**trace)
                    events=base.decode(trace,config['decoder'],cp['classes'])
                else:
                    prior_mode='personal' if mode=='full' else 'base'
                    with np.load(reference/s/f'{prior_mode}_{kind}_trace.npz') as trace:
                        verify_trace_grid(trace,own,cp['window_ms'])
                        events=base.decode(trace,config['decoder'],cp['classes'])
                metrics,trials,confusion=previous.score(events,own,cp['classes'])
                assert metrics['trials']==config['remaining_test'][s]
                group[name]=metrics;trials_by_mode[mode]=trials
                io.write(target/f'{name}_events.json',events)
                io.write_csv(target/f'{name}_trials.csv',trials)
                io.write_csv(target/f'{name}_confusion.csv',confusion)
                all_events[name].extend(events)
            for against in ['base','full']:
                # Column names say reference/head rather than the prior helper's
                # base/personal terminology, to make both comparisons unambiguous.
                paired=previous.paired_trials(trials_by_mode[against],trials_by_mode['head'])
                paired=[{k.replace('base_','reference_').replace('personal_','head_'):v for k,v in t.items()} for t in paired]
                io.write_csv(target/f'paired_{against}_vs_head_{kind}.csv',paired)
        results[s]=group;io.write(target/'comparison.json',group)
        del waves
    overall={n:previous.score(ev,records,cp['classes'])[0] for n,ev in all_events.items()}
    io.write(out/'comparison.json',dict(by_subject=results,overall=overall))
    groups=list(results.items())+[('ALL',overall)]
    io.write_csv(out/'comparison.csv',[dict(subject=s,model=n,**m) for s,g in groups for n,m in g.items()])
    deltas=[]
    for s,g in groups:
        for kind in ['fp32','int8']:
            for against in ['base','full']:
                a,b=g[f'{against}_{kind}'],g[f'head_{kind}']
                deltas.append(dict(subject=s,format=kind,reference=against,
                    correct_change=b['correct_unique_trials']-a['correct_unique_trials'],
                    accuracy_change_pp=100*(b['end_to_end_recall']-a['end_to_end_recall']),
                    strict_change_pp=100*(b['clean_success_rate']-a['clean_success_rate']),
                    wrong_output_change=b['wrong_label_outputs']-a['wrong_label_outputs'],
                    no_response_change=b['no_response_trials']-a['no_response_trials']))
    io.write_csv(out/'head_deltas.csv',deltas)
    lines=['三方对比：base原基模 / full全模型微调 / head只微调分类层',
           '同一399条剩余测试；精确复用上次校准试次、每轮窗口顺序、轮数和学习率。',
           '模型          正确命令            严格成功率   错词输出/试次   无响应   静息输出/分钟']
    for s,g in groups:
        lines.append('\n'+s)
        for n,m in g.items():
            lines.append(f'{n:13s} {m["correct_unique_trials"]}/{m["trials"]} ({m["end_to_end_recall"]:.2%}) '
                f'{m["clean_success_rate"]:.2%}   {m["wrong_label_outputs"]}/{m["wrong_response_trials"]} '
                f'  {m["no_response_trials"]}   {m["rest_proxy_outputs_per_min"]:.3f}')
    lines+=['','只训练363个参数；卷积、BN参数及统计逐项确认与基模完全相同。',
             'base/full复用已完成实验的连续预测，按冻结规则重新解码计分；head独立回放。',
             '这是查看前次结果后的探索性对照，不是新的盲测或跨日验证；无部署或板端计时。']
    report='\n'.join(lines)
    (out/'results.txt').write_text(report,encoding='utf-8-sig');print(report,flush=True)


def run_all(out, cp, rows, records, plans, config, schedules):
    base.check_seal(out,'partition_sha256.json')
    original={n:v.clone() for n,v in cp['state_dict'].items()}
    for s in SUBJECTS:
        target=out/s;target.mkdir()
        waves=base.load_waves([r for r in records if r['subject']==s])
        train_person(cp,rows,waves,plans[s],s,config,schedules[s],target)
        assert all(torch.equal(v,cp['state_dict'][n]) for n,v in original.items())
        del waves
    base.seal(out,'training_sha256.json',[out/s/n for s in SUBJECTS for n in ['model.pt','training_history.json','freeze_audit.json']])
    ix=io.read(out/'ptq_indices.json');names={rows[i]['recording'] for i in ix}
    waves=base.load_waves([r for r in records if r['recording'] in names])
    xcal=base.batch(rows,ix,waves,cp['normalization']);del waves
    for s in SUBJECTS:
        target=out/s
        head=torch.load(target/'model.pt',map_location='cpu',weights_only=True)
        pre=base.export_fp(base.restore(head),target/'model_fp32.onnx',xcal[:16])
        base.auto.quantize(pre,target/'model_int8_qdq.onnx',xcal,np.arange(len(xcal)),config['percentile'])
        base.previous.check_quantized_graph(target/'model_int8_qdq.onnx')
        contract=io.read(Path(config['base'])/'export/input_contract.json')
        contract.update(subject=s,adaptation='classifier_only',parent_checkpoint_sha256=config['base_checkpoint_sha256'])
        io.write(target/'input_contract.json',contract)
        io.write(target/'resource_estimate.json',base.resource_estimate(base.restore(head)))
        print('EXPORT COMPLETE',s,flush=True)
    del xcal
    base.seal(out,'export_sha256.json',[p for s in SUBJECTS for p in (out/s).glob('*.onnx')])
    base.verify(out)
    compare(out,cp,records,config)
    for marker in ['partition_sha256.json','training_sha256.json','export_sha256.json']:
        base.check_seal(out,marker)
    base.seal(out,'results_sha256.json',[out/n for n in ['comparison.json','comparison.csv','head_deltas.csv','results.txt']])
    io.write(out/'completed.json',dict(complete=True,no_deployment=True,remaining_test_trials=399))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=DEFAULT_REFERENCE)
    parser.add_argument('--output',type=Path,help='NEW directory only; no overwrite or resume')
    parser.add_argument('--phase',choices=['all','prepare'],default='all')
    args=parser.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    out=(args.output or base.ROOT/'models'/('personal_head_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    out.mkdir(parents=True,exist_ok=False)
    tmp=out/'temporary';tmp.mkdir();tempfile.tempdir=str(tmp)
    with (out/'run.log').open('w',encoding='utf-8') as log:
        with redirect_stdout(base.previous.Tee(sys.stdout,log)),redirect_stderr(base.previous.Tee(sys.stderr,log)):
            try:
                print('OUTPUT',out,flush=True)
                data=prepare(out,args.reference.resolve())
                if args.phase=='all':run_all(out,*data)
                print('PHASE COMPLETE',args.phase,out,flush=True)
            except Exception:
                traceback.print_exc()
                raise


if __name__=='__main__':
    main()
