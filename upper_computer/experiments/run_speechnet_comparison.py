"""User-run SpeechNet adaptation: validation selection, refit, PTQ, full replay.

Runs locally on CPU. Never accesses devices or modifies earlier data/models.
The --phase check path uses synthetic data only, and never starts full training.
"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import gc
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback

import train_auto_segment_cnn as auto
from train_small_cnn import np, torch, nn
from speechnet_two_channel import SpeechNet2, ExportModel, restore, resources, VARIANTS, LENGTH

cnn, replay = auto.cnn, auto.replay
ROOT = cnn.ROOT
SOURCE = ROOT / 'models/cnn_auto_segment_w32_20261021'
SEEDS = [20261010, 20261011, 20261012]


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log
    def write(self, value):
        self.console.write(value)
        self.log.write(value)
        self.log.flush()
        return len(value)
    def flush(self):
        self.console.flush()
        self.log.flush()


def verify_hashes(hashes):
    for name, expected in hashes.items():
        if replay.digest(name) != expected:
            raise RuntimeError(f'Input changed: {name}')


def dependencies():
    return [Path(__file__), ROOT/'experiments/speechnet_two_channel.py',
            ROOT/'experiments/train_auto_segment_cnn.py', ROOT/'experiments/train_small_cnn.py',
            ROOT/'experiments/replay_activity_cnn.py', ROOT/'experiments/calibrate_h7_cnn.py',
            ROOT/'experiments/export_h7_cnn.py', ROOT/'semg/activity_cnn.py']


def verify_inputs(out, initial=False):
    if initial:
        delivered = replay.read(SOURCE/'delivery_sha256.json')
        needed = ['segments.npz', 'segments.json', 'protocol.json', 'comparison.json',
                  'old_fp32_events.json', 'old_int8_events.json',
                  'new_fp32_events.json', 'new_int8_events.json', 'source_sha256.json',
                  'export/input_contract.json']
        hashes = {str(SOURCE/name): delivered[name] for name in needed}
        hashes.update(replay.read(SOURCE/'source_sha256.json'))
        hashes.update({str(p): replay.digest(p) for p in dependencies()})
        verify_hashes(hashes)
        replay.write(out/'inputs_sha256.json', hashes)
        snapshot = out/'source_snapshot'
        snapshot.mkdir()
        for p in dependencies():
            shutil.copy2(p, snapshot/p.name)
        for p in (ROOT/'experiments/third_party').glob('SilentWear-*'):
            shutil.copy2(p, snapshot/p.name)
    else:
        verify_hashes(replay.read(out/'inputs_sha256.json'))
    print('INPUT HASHES VERIFIED', flush=True)


def data_and_labels():
    rows, data, split = auto.load_data(SOURCE)
    classes = replay.read(SOURCE/'protocol.json')['classes']
    assert {k: len(v) for k, v in split.items()} == dict(train=1340, validation=440, test=437)
    groups = {k: {(rows[i]['recording'], rows[i]['block']) for i in ix} for k, ix in split.items()}
    assert not groups['train'] & groups['validation']
    assert not (groups['train'] | groups['validation']) & groups['test']
    y = torch.tensor([classes.index(r['word']) for r in rows])
    return rows, data, split, classes, y


def checkpoint(net, norm, selected, rows, ix, classes):
    return dict(schema='semg_speechnet2_v1', state_dict=net.state_dict(), variant=net.variant,
                normalization=norm, selected=selected, classes=classes, input_length=LENGTH,
                training_fingerprints=[rows[i]['fingerprint'] for i in ix])


def fit(out):
    rows, data, split, classes, y = data_and_labels()
    x, norm = cnn.normalize(data['raw'], data['lengths'], split['train'])
    histories, best, selected = [], float('inf'), None
    began = time.monotonic()
    for variant in VARIANTS:
        for seed in SEEDS:
            cnn.set_seed(seed)
            net = SpeechNet2(variant, len(classes))
            optimizer = torch.optim.Adam(net.parameters(), lr=.001, weight_decay=.0001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=.1, patience=2)
            history, local, stale = [], float('inf'), 0
            for epoch in range(1, 101):
                lr = float(optimizer.param_groups[0]['lr'])
                tl, ta = auto.train_epoch(net, optimizer, data, y, split['train'], norm)
                vl, va, _ = cnn.evaluate(net, x, y, split['validation'])
                if not np.isfinite(tl + vl):
                    raise RuntimeError('Non-finite loss; stopping without exporting a model.')
                history.append(dict(epoch=epoch, lr=lr, train_loss=tl, train_accuracy=ta,
                                    validation_loss=vl, validation_accuracy=va))
                if vl < local:
                    local, stale = vl, 0
                else:
                    stale += 1
                if vl < best:
                    best = vl
                    selected = dict(variant=variant, seed=seed, epoch=epoch, validation_loss=vl,
                                    validation_accuracy=va, learning_rates=[r['lr'] for r in history])
                    torch.save(checkpoint(net, norm, selected, rows, split['train'], classes),
                               out/'selection_checkpoint.pt')
                scheduler.step(vl)
                print(f'SELECT {variant} seed={seed} epoch={epoch}/100 '
                      f'lr={lr:.2g} train={ta:.3%} val={va:.3%} CE={vl:.5f}', flush=True)
                if stale >= 10:
                    break
            histories.append(dict(variant=variant, seed=seed, epochs=history))
            replay.write(out/'selection_history.json', histories)
            del net, optimizer, scheduler
            gc.collect()
    replay.write(out/'selected_training.json', selected)
    print('SELECTION FROZEN BEFORE TEST', selected, flush=True)
    final_ix = np.sort(np.r_[split['train'], split['validation']])
    x, norm = cnn.normalize(data['raw'], data['lengths'], final_ix)
    cnn.set_seed(selected['seed'])
    net = SpeechNet2(selected['variant'], len(classes))
    optimizer = torch.optim.Adam(net.parameters(), lr=.001, weight_decay=.0001)
    history = []
    # Replay the selected validation-derived LR trajectory; do not monitor test/refit validation.
    for epoch, lr in enumerate(selected['learning_rates'], 1):
        for group in optimizer.param_groups:
            group['lr'] = lr
        loss, acc = auto.train_epoch(net, optimizer, data, y, final_ix, norm)
        if not np.isfinite(loss):
            raise RuntimeError('Non-finite refit loss')
        history.append(dict(epoch=epoch, lr=lr, loss=loss, accuracy=acc))
        print(f'REFIT {epoch}/{selected["epoch"]} train={acc:.3%}', flush=True)
    torch.save(checkpoint(net, norm, selected, rows, final_ix, classes), out/'model.pt')
    replay.write(out/'refit_history.json', history)
    net.eval()
    reloaded = restore(torch.load(out/'model.pt', map_location='cpu', weights_only=True)).eval()
    with torch.inference_mode():
        torch.testing.assert_close(net(x[final_ix[:4]]), reloaded(x[final_ix[:4]]), rtol=0, atol=0)
    replay.write(out/'training_report.json', dict(selected=selected, final_train_segments=len(final_ix),
                 elapsed_seconds=time.monotonic()-began, parameters=sum(p.numel() for p in net.parameters()),
                 test_used_for_selection=False, checkpoint_roundtrip=True))
    replay.write(out/'training_sha256.json', {str(out/name): replay.digest(out/name) for name in
                 ['model.pt', 'selection_checkpoint.pt', 'selected_training.json',
                  'selection_history.json', 'training_report.json', 'protocol.json']})


def export_fp32(net, target, x, indices):
    net.eval()
    torch.onnx.export(ExportModel(net).eval(), torch.zeros(1, 2, 1, LENGTH), str(target),
                      opset_version=17, dynamo=False, input_names=['input'], output_names=['logits'],
                      do_constant_folding=True)
    auto.onnx.checker.check_model(auto.onnx.load(target))
    actual = auto.infer(target, x, indices)
    with torch.inference_mode():
        expected = np.concatenate([net(torch.from_numpy(x[part])).numpy()
                                   for part in np.array_split(indices, max(1, int(np.ceil(len(indices)/32))))])
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
    pre = target.with_name(target.stem+'_preprocessed.onnx')
    auto.quant_pre_process(str(target), str(pre), skip_optimization=True)
    return pre


def check_quantized_graph(path):
    graph = auto.onnx.load(path).graph
    initializers = {p.name: p for p in graph.initializer}
    producers = {o: n for n in graph.node for o in n.output}
    checked = []
    for node in graph.node:
        if node.op_type not in ('Conv', 'Gemm'):
            continue
        for name in node.input[:2]:
            dq = producers.get(name)
            assert dq is not None and dq.op_type == 'DequantizeLinear', node.name
            assert dq.input[2] in initializers
            assert initializers[dq.input[2]].data_type == auto.onnx.TensorProto.INT8
        checked.append(node.op_type)
    assert checked.count('Conv') == 5 and checked.count('Gemm') == 1
    return checked


def export_and_quantize(out):
    verify_hashes(replay.read(out/'training_sha256.json'))
    rows, data, split, classes, y = data_and_labels()
    target = out/'export'
    target.mkdir(exist_ok=True)
    cp = torch.load(out/'selection_checkpoint.pt', map_location='cpu', weights_only=True)
    x, norm = cnn.normalize(data['raw'], data['lengths'], split['train'])
    assert norm == cp['normalization']
    assert cp['training_fingerprints'] == [rows[i]['fingerprint'] for i in split['train']]
    x = x.numpy()
    pre = export_fp32(restore(cp), target/'development_fp32.onnx', x, split['validation'])
    ix = auto.representative(rows, split['train'])
    assert set(ix) <= set(split['train'])
    candidates = []
    for percentile in [None, 99.99, 99.9, 99.5]:
        name = 'minmax' if percentile is None else f'percentile_{percentile}'
        path = target/(name+'_development.onnx')
        auto.quantize(pre, path, x, ix, percentile)
        logits = auto.infer(path, x, split['validation'])
        candidates.append(dict(name=name, percentile=percentile,
            validation_loss=float(nn.functional.cross_entropy(torch.from_numpy(logits), y[split['validation']])),
            validation_accuracy=float(np.mean(logits.argmax(1) == y[split['validation']].numpy()))))
        print('QUANT VALIDATION', candidates[-1], flush=True)
    selected = min(candidates, key=lambda r: r['validation_loss'])
    replay.write(target/'quantization_selection.json', dict(selected=selected, candidates=candidates,
                 calibration_fingerprints=[rows[i]['fingerprint'] for i in ix], test_used=False))
    cp = torch.load(out/'model.pt', map_location='cpu', weights_only=True)
    final_ix = np.sort(np.r_[split['train'], split['validation']])
    assert cp['training_fingerprints'] == [rows[i]['fingerprint'] for i in final_ix]
    assert not set(cp['training_fingerprints']) & {rows[i]['fingerprint'] for i in split['test']}
    x, norm = cnn.normalize(data['raw'], data['lengths'], final_ix)
    assert norm == cp['normalization']
    x = x.numpy()
    net = restore(cp)
    pre = export_fp32(net, target/'model_fp32.onnx', x, final_ix[:100])
    ix = auto.representative(rows, final_ix)
    assert set(ix) <= set(final_ix)
    auto.quantize(pre, target/'model_int8_qdq.onnx', x, ix, selected['percentile'])
    quantized = check_quantized_graph(target/'model_int8_qdq.onnx')
    replay.write(target/'final_calibration.json', dict(selected=selected,
                 calibration_fingerprints=[rows[i]['fingerprint'] for i in ix], test_used=False,
                 quantized_layers=quantized))
    report = resources(net)
    replay.write(target/'resource_estimate.json', report)
    assert report['int8_inventory_plus_reserve_bytes'] < 1024*1024
    contract = replay.read(SOURCE/'export/input_contract.json')
    contract.update(normalization=norm, architecture='SpeechNet2', variant=net.variant,
                    input_shape=[1, 2, 1, LENGTH], classes=classes,
                    requires='Frozen causal 2kHz activity segments; transpose to B,1,2,T inside ONNX. '
                             'Right pad after normalization. No rest class. Not tested on STM32.')
    replay.write(target/'input_contract.json', contract)
    replay.write(out/'export_sha256.json', {str(p): replay.digest(p) for p in target.iterdir() if p.is_file()})


class CheckedSession:
    def __init__(self, path, net):
        self.runtime = replay.session(path)
        self.net = net.eval()
        self.checked = 0
    def run(self, output_names, feeds):
        got = self.runtime.run(output_names, feeds)
        with torch.inference_mode():
            expected = self.net(torch.from_numpy(feeds['input'][:, :, 0, :])).numpy()
        np.testing.assert_allclose(got[0], expected, rtol=1e-4, atol=1e-4)
        self.checked += 1
        return got


def score_and_report(out):
    verify_hashes(replay.read(out/'training_sha256.json'))
    verify_hashes(replay.read(out/'export_sha256.json'))
    cp = torch.load(out/'model.pt', map_location='cpu', weights_only=True)
    records = replay.read(auto.REPLAY/'recordings.json')
    config = auto.GateConfig(**replay.read(SOURCE/'protocol.json')['detector'])
    reference = replay.read(SOURCE/'comparison.json')['results']
    results = {name: reference[key] for name, key in [
        ('original_fp32', 'old_fp32'), ('original_int8', 'old_int8'),
        ('auto32_fp32', 'new_fp32'), ('auto32_int8', 'new_int8')]}
    metadata = ['recording', 'event_id', 'kind', 'start', 'end', 'emitted', 'matched_trial', 'matched_word', 'reason']
    checks = 0
    for kind in ['fp32', 'int8']:
        model_path = out/'export'/('model_fp32.onnx' if kind == 'fp32' else 'model_int8_qdq.onnx')
        runtime = CheckedSession(model_path, restore(cp)) if kind == 'fp32' else replay.session(model_path)
        events, states = replay.run(auto.REPLAY, records, config, 'test', runtime,
                                   cp['normalization'], cp['classes'])
        old_events = replay.read(SOURCE/f'new_{kind}_events.json')
        assert [[r.get(k) for k in metadata] for r in events] == [[r.get(k) for k in metadata] for r in old_events]
        assert sum(r['candidate'] is not None for r in events) == 465
        if kind == 'fp32':
            checks = runtime.checked
            assert checks == 465
        replay.write(out/f'{kind}_events.json', events)
        replay.write_csv(out/f'{kind}_events.csv', events)
        replay.write_csv(out/f'{kind}_trials.csv', replay.trial_diagnostics(events, records, 0.))
        overall = replay.metrics(events, records, 'test')
        assert overall['trials'] == 499 and overall['detected_trials'] == 437
        # Independent correctness tally before accepting the shared metric helper's result.
        correct = {(e['recording'], e['matched_trial']) for e in events
                   if e['candidate'] is not None and e['matched_trial'] is not None
                   and e['candidate'] == e['matched_word']}
        assert len(correct) == overall['correct_unique_trials']
        results['speechnet_'+kind] = dict(overall=overall, by_subject={
            s: replay.metrics(events, records, 'test', subject=s)
            for s in sorted({r['subject'] for r in records})})
        print('TEST', kind, overall, flush=True)
    verify_inputs(out)
    result = dict(results=results, test_targets=499, same_detector_events=True,
                  fp32_candidates_checked=checks, no_deployment=True, sources_unchanged=True,
                  target=.85, target_met_fp32=results['speechnet_fp32']['overall']['end_to_end_recall'] >= .85,
                  target_met_int8=results['speechnet_int8']['overall']['end_to_end_recall'] >= .85,
                  all_test_targets_retained=True)
    replay.write(out/'comparison.json', result)
    table = []
    for name, group in results.items():
        for subject, metrics in [('ALL', group['overall'])] + list(group['by_subject'].items()):
            table.append(dict(model=name, subject=subject, **metrics))
    replay.write_csv(out/'comparison.csv', table)
    selected = replay.read(out/'selected_training.json')
    r = replay.read(out/'export/resource_estimate.json')
    lines = ['SpeechNet双通道适配：连续回放结果', '',
             '全部499条测试命令作分母，包括62次检测未覆盖；不使用提示切段或仅成功候选分母替代端到端成绩。',
             f'验证集选择：{selected["variant"]}，seed={selected["seed"]}，epoch={selected["epoch"]}。',
             '模型                        正确命令         严格一次正确且无额外输出']
    for name, group in results.items():
        m = group['overall']
        lines.append(f'{name:23} {m["correct_unique_trials"]}/499 ({m["end_to_end_recall"]:.2%})    {m["clean_success_rate"]:.2%}')
    lines += ['', '逐人INT8：原提示训练模型 / 自动分段32模型 / SpeechNet（均为连续回放）']
    for subject in results['speechnet_int8']['by_subject']:
        values = [results[k]['by_subject'][subject]['end_to_end_recall'] for k in
                  ['original_int8', 'auto32_int8', 'speechnet_int8']]
        lines.append(f'{subject}: ' + ' / '.join(f'{v:.2%}' for v in values))
    q = results['speechnet_int8']['overall']
    lines += ['', f'INT8是否达到85%：{result["target_met_int8"]}（此项不代表板上/新佩戴验证通过）。',
              f'提示静息输出：{q["rest_proxy_outputs_per_min"]:.2f}次/分钟；检测覆盖{q["detection_recall"]:.2%}。',
              f'参数{r["parameters"]}，MAC {r["macs"]}；INT8张量清单+256KiB规划预留约{r["int8_inventory_plus_reserve_bytes"]/1024:.2f}KiB。',
              '资源是估算，未用ST工具链测arena或测板上延迟。未部署。',
              '该实验改变网络与训练优化设置，不是每项改动的独立消融；两个核方案/三个种子由验证损失选择。',
              '采用五人混合、十词、2kHz、自动片段与因果滤波，不是SilentWear原论文的14通道、按人模型、九类、500Hz、提示窗口实验。',
              '没有增加静息类或更改检测器，因此不能改善漏检及背景输出次数。',
              '训练/验证/测试整轮划分不变；旧测试此前已查看，属于探索性已见人员对照，不是新盲测。',
              '训练/标准化/校准成员无测试；原始来源与缓存哈希、检测事件及465条浮点推理已核验。']
    (out/'结果汇总.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


def preflight(out):
    """Small synthetic forward/backward, export and quantization compatibility check."""
    cnn.set_seed(20261021)
    x = np.random.default_rng(42).normal(size=(8, 2, LENGTH)).astype(np.float32)
    checks = []
    for variant in VARIANTS:
        net = SpeechNet2(variant)
        opt = torch.optim.Adam(net.parameters(), lr=.001)
        loss = nn.functional.cross_entropy(net(torch.from_numpy(x[:2])), torch.tensor([0, 1]))
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        opt.step()
        pre = export_fp32(net, out/(variant+'.onnx'), x, np.arange(2))
        q = out/(variant+'_int8.onnx')
        auto.quantize(pre, q, x, np.arange(8), 99.9)
        assert np.isfinite(auto.infer(q, x, np.arange(2))).all()
        check_quantized_graph(q)
        r = resources(net)
        assert r['int8_inventory_plus_reserve_bytes'] < 1024*1024
        replay.write(out/(variant+'_resources.json'), r)
        checks.append(dict(variant=variant, export_numeric_parity=True, backward_finite=True,
                           int8_executable=True, parameters=r['parameters']))
    replay.write(out/'preflight.json', dict(synthetic_only=True, full_training_started=False, checks=checks))
    print('PREFLIGHT PASSED. Synthetic checks only; no dataset training or board access.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['all', 'train', 'export', 'replay', 'check'], default='all')
    parser.add_argument('--output', type=Path, help='New directory for all/train/check; existing run for export/replay.')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    if args.phase in ['export', 'replay'] and args.output is None:
        parser.error('--output is required for export/replay')
    out = (args.output or ROOT/'models'/('speechnet2_'+datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    if args.phase in ['all', 'train', 'check']:
        out.mkdir(parents=True, exist_ok=False)
    elif not out.is_dir():
        parser.error(f'Run directory missing: {out}')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    temp = out/'temporary'
    temp.mkdir(exist_ok=True)
    tempfile.tempdir = str(temp)
    with (out/'run.log').open('a', encoding='utf-8') as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            print('OUTPUT', out, 'PHASE', args.phase, flush=True)
            try:
                if args.phase == 'check':
                    preflight(out)
                    return
                if args.phase in ['all', 'train']:
                    verify_inputs(out, initial=True)
                    protocol = dict(source=str(SOURCE), variants=VARIANTS, seeds=SEEDS,
                        classes=replay.read(SOURCE/'protocol.json')['classes'],
                        fs=2000, input_length=LENGTH, max_epochs=100, patience=10, batch=32,
                        optimizer='Adam', lr=.001, weight_decay=.0001, dropout=.5,
                        scheduler='ReduceLROnPlateau factor0.1 patience2; selected LR trajectory reused for fresh refit',
                        selection='minimum validation cross entropy over 2 variants x 3 seeds; no test selection',
                        preprocessing='same cached causal20-500Hz+50Hz notch, same frozen gate, training-only normalization',
                        augmentation='same +/-100ms boundary jitter, +/-40ms translation and channel gain0.9-1.1 as auto32',
                        paper='https://arxiv.org/abs/2603.02847',
                        upstream='https://github.com/pulp-bio/SilentWear',
                        changes='14->2 sensor channels; spatial7x1,7x1->2x1,1x1; 9->10 classes; '
                                '500->2000Hz, variable gate cuts padded6144; mixed subjects; causal filtering; '
                                'fixed full average pool for deployment. Original temporal kernels and 4x kernels compared.',
                        no_deployment=True, original_test_previously_viewed=True)
                    replay.write(out/'protocol.json', protocol)
                    fit(out)
                    verify_inputs(out)
                else:
                    verify_inputs(out)
                if args.phase in ['all', 'export']:
                    if (out/'export/input_contract.json').exists():
                        raise RuntimeError('Export already completed. Use --phase replay, or a new run directory.')
                    export_and_quantize(out)
                    verify_inputs(out)
                if args.phase in ['all', 'replay']:
                    if (out/'comparison.json').exists():
                        raise RuntimeError('Comparison already exists; refusing to overwrite completed test results.')
                    score_and_report(out)
                    replay.write(out/'completed.json', dict(no_deployment=True, complete=True))
                print('PHASE COMPLETE', args.phase, out, flush=True)
            except BaseException:
                traceback.print_exc()
                raise


if __name__ == '__main__':
    main()
