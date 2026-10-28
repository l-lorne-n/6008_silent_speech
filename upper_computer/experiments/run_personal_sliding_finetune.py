"""Two-shot personal adaptation of the completed sliding/rest model; user-run.

Original test rounds become calibration, never remaining evaluation. Each person
starts from the same FP32 checkpoint. No test-driven checkpoint/decoder selection.
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

import run_sliding_rest as base

np, torch, nn, io = base.np, base.torch, base.nn, base.replay
DEFAULT_BASE = base.ROOT / 'models/sliding_rest_20261022_103406'
SUBJECTS = ['S011', 'S012', 'S013', 'S014', 'S015']


def trial_key(recording, trial):
    return recording, trial['trial_id']


def select_calibration(records, rows, classes):
    """First two eligible complete test rounds, ordered by recording then block.

One source episode per word per round, and one post-action rest episode from
each round. Window multiplicity does not increase the number of repetitions.
"""
    index = defaultdict(list)
    for i, row in enumerate(rows):
        if row['split'] == 'test':
            index[(row['recording'], row['trial_id'], row['label'])].append(i)
    plans = {}
    for subject in sorted({r['subject'] for r in records}):
        rounds = defaultdict(list)
        for record in records:
            if record['subject'] == subject:
                for t in record['annotations']:
                    if t['split'] == 'test':
                        rounds[(record['recording'], t['block'])].append(t)
        chosen, episodes = [], []
        for (rec, block), trials in sorted(rounds.items()):
            if Counter(t['word'] for t in trials) != Counter(classes[:-1]):
                continue
            word_episodes = []
            for t in sorted(trials, key=lambda t: t['prepare']):
                label = classes.index(t['word'])
                indices = index[(rec, t['trial_id'], label)]
                if not indices:
                    break
                word_episodes.append(dict(recording=rec, trial_id=t['trial_id'],
                                          label=label, word=t['word'], indices=indices))
            if len(word_episodes) != len(classes)-1:
                continue
            rest_episode = None
            for t in sorted(trials, key=lambda t: t['prepare']):
                # Only one uninterrupted post-action rest episode, not both
                # preparation and rest under the same trial ID.
                ix = [i for i in index[(rec, t['trial_id'], len(classes)-1)]
                      if rows[i]['start'] >= t['rest'] + 1200]
                if ix:
                    rest_episode = dict(recording=rec, trial_id=t['trial_id'],
                                        label=len(classes)-1, word='rest', indices=ix)
                    break
            if rest_episode is None:
                continue
            chosen.append(dict(recording=rec, block=block,
                               trial_ids=[t['trial_id'] for t in trials]))
            episodes.extend(word_episodes + [rest_episode])
            if len(chosen) == 2:
                break
        if len(chosen) != 2:
            raise ValueError(f'{subject}: fewer than two complete eligible test rounds')
        plans[subject] = dict(rounds=chosen, episodes=episodes)

    heldout = deepcopy(records)
    excluded = {(g['recording'], tid) for p in plans.values()
                for g in p['rounds'] for tid in g['trial_ids']}
    for record in heldout:
        for t in record['annotations']:
            if trial_key(record['recording'], t) in excluded:
                if t['split'] != 'test':
                    raise ValueError('Calibration was not originally test')
                t['split'] = 'personal_calibration'
    audit_partition(heldout, rows, plans, classes)
    return plans, heldout


def audit_partition(records, rows, plans, classes):
    lookup = {trial_key(r['recording'], t): (r, t)
              for r in records for t in r['annotations']}
    for subject, plan in plans.items():
        assert Counter(e['word'] for e in plan['episodes']) == {c: 2 for c in classes}
        assert len({(e['recording'], e['trial_id'], e['label'])
                    for e in plan['episodes']}) == 2*len(classes)
        for episode in plan['episodes']:
            rec, t = lookup[(episode['recording'], episode['trial_id'])]
            assert rec['subject'] == subject and t['split'] == 'personal_calibration'
            for i in episode['indices']:
                row = rows[i]
                assert row['subject'] == subject and row['label'] == episode['label']
                assert (row['recording'], row['trial_id']) == (rec['recording'], t['trial_id'])
                assert t['prepare'] <= row['start'] < row['end'] <= t['context_end']
                if episode['word'] == 'rest':
                    assert row['start'] >= t['rest'] + 1200
                # Every replay window is inside a remaining contiguous test run.
                assert all(row['end'] <= a or row['start'] >= b
                           for a, b in base.split_runs(rec, 'test'))


def prepare(out, source, args):
    if not io.read(source/'completed.json')['complete']:
        raise ValueError('Base experiment is incomplete')
    for name in ['inputs_sha256.json', 'dataset_sha256.json',
                 'training_sha256.json', 'export_sha256.json']:
        base.check_seal(source, name)
    cp = torch.load(source/'model.pt', map_location='cpu', weights_only=True)
    rows_path = source/f'windows_{cp["window_ms"]}.json'
    rows, records = io.read(rows_path), base.read_records()
    assert cp['classes'][-1] == 'rest' and len(cp['classes']) == 11
    assert sorted({r['subject'] for r in records}) == SUBJECTS
    plans, heldout = select_calibration(records, rows, cp['classes'])
    original = Counter(r['subject'] for r in records for t in r['annotations'] if t['split']=='test')
    remaining = Counter(r['subject'] for r in heldout for t in r['annotations'] if t['split']=='test')
    assert all(remaining[s] == original[s]-20 for s in SUBJECTS)
    assert sum(original.values()) == 499 and sum(remaining.values()) == 399
    ptq = io.read(source/'export/calibration_manifest.json')['calibration_fingerprints']
    by_fp = {r['fingerprint']: i for i, r in enumerate(rows)}
    ptq_ix = [by_fp[f] for f in ptq]
    assert all(rows[i]['split'] in ['train', 'validation'] for i in ptq_ix)
    assert set(ptq).issubset(set(cp['training_fingerprints']))
    config = dict(schema='personal_sliding_two_shot_v1', base=str(source),
                  base_checkpoint_sha256=io.digest(source/'model.pt'),
                  epochs=args.epochs, learning_rate=args.lr, seed=args.seed,
                  classes=cp['classes'], normalization=cp['normalization'],
                  window_ms=cp['window_ms'], step_ms=100,
                  decoder=io.read(source/'decoder_selection.json')['selected']['config'],
                  percentile=io.read(source/'export/quantization_selection.json')['selected']['percentile'],
                  original_test=dict(original), remaining_test=dict(remaining),
                  training='Independent base copy; all weights; Adam; frozen BN running statistics; '
                           'one random window per source episode per epoch; one batch of 22; '
                           'fixed final epoch, no validation/checkpoint selection; no augmentation.',
                  selection='First two complete eligible original test rounds per person in recording/block order; '
                            'first eligible post-action rest episode per selected round; no model predictions used.',
                  evaluation='Same remaining contiguous test runs and decoder reset boundaries for base/personal; '
                             'full calibration contexts excluded, including unused windows and transitions.',
                  limitations='Retrospective adaptation on previously inspected, seen-person recordings; '
                              'not a fresh new-day or unseen-person study. Cue-derived labels. '
                              'Existing continuous causal filtering retained. No hardware deployment/latency claim.',
                  ptq='Fixed original base final-training calibration windows; no remaining test inputs.')
    io.write(out/'protocol.json', config)
    io.write(out/'personal_calibration.json', plans)
    io.write(out/'evaluation_records.json', heldout)
    io.write(out/'ptq_indices.json', ptq_ix)
    io.write_csv(out/'calibration_windows.csv', [{**rows[i], 'subject':s}
                 for s,p in plans.items() for e in p['episodes'] for i in e['indices']])
    # Carry forward the complete raw-data/source provenance, plus this script.
    hashes = {}
    for name in ['inputs_sha256.json', 'dataset_sha256.json', 'training_sha256.json', 'export_sha256.json']:
        hashes.update(io.read(source/name))
        hashes[str(source/name)] = io.digest(source/name)
    files = [Path(__file__).resolve(), source/'completed.json']
    for p in files:
        hashes[str(p)] = io.digest(p)
    io.write(out/'inputs_sha256.json', hashes)
    shutil.copy2(__file__, out/Path(__file__).name)
    base.seal(out, 'partition_sha256.json', [out/n for n in
              ['protocol.json', 'personal_calibration.json', 'evaluation_records.json', 'ptq_indices.json']])
    for s in SUBJECTS:
        print(f'{s}: 20 command repetitions + 2 rest episodes; remaining test {remaining[s]}', flush=True)
    return cp, rows, heldout, plans, config


def adaptation_step(net, optimizer, x, y):
    net.train()
    for module in net.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()  # Keep base population statistics with only two shots.
    optimizer.zero_grad(set_to_none=True)
    logits = net(torch.from_numpy(x))
    loss = nn.functional.cross_entropy(logits, torch.tensor(y, dtype=torch.long))
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite personal training loss')
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), float((logits.argmax(1)==torch.tensor(y)).float().mean())


def train_person(cp, rows, waves, plan, subject, config, target):
    base.cnn.set_seed(config['seed'])
    net = base.restore(cp)  # Fresh weights and optimizer for EACH person.
    optimizer = torch.optim.Adam(net.parameters(), lr=config['learning_rate'])
    rng = np.random.default_rng(config['seed'])
    history, used = [], set()
    for epoch in range(1, config['epochs']+1):
        indices = [int(rng.choice(e['indices'])) for e in plan['episodes']]
        rng.shuffle(indices)
        x = base.batch(rows, indices, waves, cp['normalization'])
        y = [rows[i]['label'] for i in indices]
        assert Counter(y) == {i: 2 for i in range(len(cp['classes']))}
        loss, accuracy = adaptation_step(net, optimizer, x, y)
        used.update(indices)
        history.append(dict(epoch=epoch, loss=loss, calibration_batch_accuracy=accuracy,
                            window_fingerprints=[rows[i]['fingerprint'] for i in indices]))
        if epoch==1 or epoch%5==0 or epoch==config['epochs']:
            print(f'FINETUNE {subject} {epoch}/{config["epochs"]} loss={loss:.5f} '
                  f'calibration_batch_accuracy={accuracy:.3f}', flush=True)
    for name, value in net.state_dict().items():
        if 'running_' in name or 'num_batches_tracked' in name:
            assert torch.equal(value, cp['state_dict'][name])
    personal = deepcopy(cp)
    personal['state_dict'] = net.state_dict()
    personal['personal_adaptation'] = dict(subject=subject,
        base_checkpoint_sha256=config['base_checkpoint_sha256'], epochs=config['epochs'],
        learning_rate=config['learning_rate'], seed=config['seed'],
        calibration_fingerprints=[rows[i]['fingerprint'] for i in sorted(used)],
        source_episodes=plan['episodes'], remaining_test_used=False)
    personal['training_fingerprints'] = cp['training_fingerprints'] + personal['personal_adaptation']['calibration_fingerprints']
    torch.save(personal, target/'model.pt')
    restored = base.restore(torch.load(target/'model.pt', map_location='cpu', weights_only=True))
    for k,v in net.state_dict().items():
        assert torch.equal(v, restored.state_dict()[k])
    io.write(target/'training_history.json', history)
    return personal


def score(events, records, classes):
    metrics, trials = base.event_metrics(events, records, 'test')
    by_trial = defaultdict(list)
    confusion = Counter()
    for r in records:
        local = [e for e in events if e['recording']==r['recording']]
        for t in r['annotations']:
            if t['split'] != 'test':
                continue
            response = [e for e in local if t['action'] <= e['emitted'] < min(t['rest']+600,t['context_end'])]
            by_trial[(r['recording'],t['trial_id'])] = response
            for e in response:
                confusion[(t['word'],e['candidate'])] += 1
    for t in trials:
        response = by_trial[(t['recording'],t['trial_id'])]
        t['wrong_response_outputs'] = sum(e['candidate']!=t['word'] for e in response)
        t['has_wrong_response'] = t['wrong_response_outputs'] > 0
    metrics['wrong_response_trials'] = sum(t['has_wrong_response'] for t in trials)
    metrics['wrong_response_trial_rate'] = metrics['wrong_response_trials']/max(1,len(trials))
    metrics['wrong_fraction_of_response_outputs'] = metrics['wrong_label_outputs']/max(1,sum(t['response_outputs'] for t in trials))
    return metrics, trials, [dict(actual=a, predicted=p, count=n) for (a,p),n in sorted(confusion.items())]


def paired_trials(before, after):
    key = lambda t: (t['recording'],t['trial_id'])
    lookup = {key(t):t for t in after}
    assert set(map(key,before)) == set(lookup)
    return [dict(recording=t['recording'], subject=t['subject'], trial_id=t['trial_id'], word=t['word'],
                 base_correct=t['correct'], personal_correct=lookup[key(t)]['correct'],
                 base_clean=t['clean'], personal_clean=lookup[key(t)]['clean'],
                 base_wrong=t['has_wrong_response'], personal_wrong=lookup[key(t)]['has_wrong_response'],
                 base_predictions=t['predictions'], personal_predictions=lookup[key(t)]['predictions']) for t in before]


def run_all(out, cp, rows, records, plans, config):
    base.check_seal(out, 'partition_sha256.json')
    source = Path(config['base'])
    original_state = {k:v.clone() for k,v in cp['state_dict'].items()}
    # Finish every person's adaptation before examining any remaining test result.
    for subject in SUBJECTS:
        target = out/subject
        target.mkdir()
        own = [r for r in records if r['subject']==subject]
        waves = base.load_waves(own)
        train_person(cp, rows, waves, plans[subject], subject, config, target)
        del waves
        assert all(torch.equal(v, cp['state_dict'][k]) for k,v in original_state.items())
    base.seal(out, 'training_sha256.json', [out/s/'model.pt' for s in SUBJECTS])
    ptq_ix = io.read(out/'ptq_indices.json')
    ptq_recs = {rows[i]['recording'] for i in ptq_ix}
    waves = base.load_waves([r for r in records if r['recording'] in ptq_recs])
    xcal = base.batch(rows, ptq_ix, waves, cp['normalization'])
    del waves
    for subject in SUBJECTS:
        target = out/subject
        personal = torch.load(target/'model.pt', map_location='cpu', weights_only=True)
        pre = base.export_fp(base.restore(personal), target/'model_fp32.onnx', xcal[:16])
        base.auto.quantize(pre, target/'model_int8_qdq.onnx', xcal, np.arange(len(xcal)), config['percentile'])
        base.previous.check_quantized_graph(target/'model_int8_qdq.onnx')
        # A runnable contract travels with each person's weights.
        contract = io.read(source/'export/input_contract.json')
        contract.update(subject=subject, parent_checkpoint_sha256=config['base_checkpoint_sha256'])
        io.write(target/'input_contract.json', contract)
        io.write(target/'resource_estimate.json', base.resource_estimate(base.restore(personal)))
        print('EXPORT COMPLETE', subject, flush=True)
    del xcal
    base.seal(out, 'export_sha256.json', [p for s in SUBJECTS for p in (out/s).glob('*.onnx')])
    base.verify(out)
    results, all_events = {}, defaultdict(list)
    for subject in SUBJECTS:
        target = out/subject
        own = [r for r in records if r['subject']==subject]
        waves = base.load_waves(own)
        personal = torch.load(target/'model.pt', map_location='cpu', weights_only=True)
        subject_results, trial_tables = {}, {}
        for kind in ['fp32','int8']:
            filename = 'model_fp32.onnx' if kind=='fp32' else 'model_int8_qdq.onnx'
            for mode, checkpoint, path in [('base',cp,source/'export'/filename), ('personal',personal,target/filename)]:
                name = f'{mode}_{kind}'
                trace = base.infer_runs(path, checkpoint, own, waves, 'test')
                np.savez_compressed(target/f'{name}_trace.npz', **trace)
                events = base.decode(trace, config['decoder'], cp['classes'])
                metrics, trials, confusion = score(events, own, cp['classes'])
                assert metrics['trials'] == config['remaining_test'][subject]
                io.write(target/f'{name}_events.json', events)
                io.write_csv(target/f'{name}_trials.csv', trials)
                io.write_csv(target/f'{name}_confusion.csv', confusion)
                subject_results[name], trial_tables[name] = metrics, trials
                all_events[name].extend(events)
            io.write_csv(target/f'paired_{kind}.csv', paired_trials(trial_tables[f'base_{kind}'],trial_tables[f'personal_{kind}']))
        results[subject] = subject_results
        io.write(target/'comparison.json', subject_results)
        del waves
    overall = {name:score(events,records,cp['classes'])[0] for name,events in all_events.items()}
    io.write(out/'comparison.json', dict(by_subject=results, overall=overall))
    summary = [dict(subject=s,model=name,**m) for s,group in list(results.items())+[('ALL',overall)] for name,m in group.items()]
    io.write_csv(out/'comparison.csv', summary)
    lines = ['个人两遍校准实验（原测试集划分；不能直接与原499条成绩比较）',
             '模型                    正确命令        严格成功率     错词输出/错词试次    无响应    静息输出/分钟']
    for s, group in list(results.items())+[('ALL',overall)]:
        lines.append('\n'+s)
        for name,m in group.items():
            lines.append(f'{name:20s} {m["correct_unique_trials"]}/{m["trials"]} ({m["end_to_end_recall"]:.2%}) '
                         f'{m["clean_success_rate"]:.2%}   {m["wrong_label_outputs"]}/{m["wrong_response_trials"]} '
                         f'   {m["no_response_trials"]}   {m["rest_proxy_outputs_per_min"]:.3f}')
    lines.extend(['', 'rest为静息，只有20条动作+2段静息参与每人的梯度更新。',
                  '固定轮数/学习率/解码规则，未按剩余测试成绩选择模型；微调可能提高也可能降低成绩。',
                  '基模已见过这些人，原测试也曾查看；这是回顾性个体校准实验，不是新日期实测。',
                  '错词输出只统计响应区间；区间外输出、重复、静息误触发详见comparison.csv。',
                  '静息误触发是提示静息代理指标，不等于吞咽/走路等真实负例。没有部署或板端计时。'])
    report = '\n'.join(lines)
    (out/'results.txt').write_text(report,encoding='utf-8-sig')
    print(report, flush=True)
    base.check_seal(out,'training_sha256.json'); base.check_seal(out,'export_sha256.json')
    io.write(out/'completed.json', dict(complete=True, no_deployment=True, remaining_test_trials=399))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=DEFAULT_BASE)
    parser.add_argument('--output', type=Path, help='Must be a NEW directory; never overwrites a run')
    parser.add_argument('--phase', choices=['all','prepare'], default='all')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=20261022)
    args = parser.parse_args()
    if args.epochs < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error('epochs and learning rate must be positive')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    out = (args.output or base.ROOT/'models'/('personal_sliding_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    out.mkdir(parents=True, exist_ok=False)
    tmp = out/'temporary'; tmp.mkdir(); tempfile.tempdir = str(tmp)
    with (out/'run.log').open('w',encoding='utf-8') as log:
        with redirect_stdout(base.previous.Tee(sys.stdout,log)), redirect_stderr(base.previous.Tee(sys.stderr,log)):
            print('OUTPUT',out,flush=True)
            prepared = prepare(out,args.base.resolve(),args)
            if args.phase == 'all':
                run_all(out,*prepared)
            print('PHASE COMPLETE',args.phase,out,flush=True)


if __name__ == '__main__':
    main()
