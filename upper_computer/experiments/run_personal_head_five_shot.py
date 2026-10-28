"""User-run five-shot classifier-only adaptation; base replay on the same remainder.

Keep the prior two-shot sources, add three complete original-test rounds per person.
New rest episodes use the base training label's 300ms post-action guard. Never
count overlapping windows as separate repetitions. No hardware access.
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

import run_personal_head_finetune as head

base, previous = head.base, head.previous
np, torch, io = head.np, head.torch, head.io
SUBJECTS = head.SUBJECTS
DEFAULT_REFERENCE = base.ROOT/'models/personal_head_20261022_122442_600468'
SHOTS = 5


def expand_calibration(records, rows, classes, old_plans):
    """Append three eligible rounds without consulting any model prediction."""
    index=defaultdict(list)
    for i,w in enumerate(rows):
        if w['split']=='test':index[(w['recording'],w['trial_id'],w['label'])].append(i)
    plans=deepcopy(old_plans)
    for s,plan in plans.items():
        assert len(plan['rounds'])==2 and len(plan['episodes'])==22
        selected={(g['recording'],g['block']) for g in plan['rounds']}
        groups=defaultdict(list)
        for r in records:
            if r['subject']==s:
                for t in r['annotations']:
                    if t['split']=='test':groups[(r['recording'],t['block'])].append(t)
        for (rec,block),trials in sorted(groups.items()):
            if (rec,block) in selected or Counter(t['word'] for t in trials)!=Counter(classes[:-1]):
                continue
            episodes=[]
            for t in sorted(trials,key=lambda t:t['prepare']):
                label=classes.index(t['word']);ix=index[(rec,t['trial_id'],label)]
                if not ix:break
                episodes.append(dict(recording=rec,trial_id=t['trial_id'],label=label,word=t['word'],indices=ix))
            if len(episodes)!=10:continue
            rest=None
            for t in sorted(trials,key=lambda t:t['prepare']):
                # Rows already passed the original label, clipping and context checks.
                ix=[i for i in index[(rec,t['trial_id'],10)] if rows[i]['start']>=t['rest']+600]
                if ix:
                    rest=dict(recording=rec,trial_id=t['trial_id'],label=10,word='rest',indices=ix)
                    break
            if rest is None:continue
            plan['rounds'].append(dict(recording=rec,block=block,trial_ids=[t['trial_id'] for t in trials]))
            plan['episodes'].extend(episodes+[rest])
            if len(plan['rounds'])==SHOTS:break
        if len(plan['rounds'])!=SHOTS:
            raise ValueError(f'{s}: insufficient complete rounds for five shots; no window duplication fallback')
        assert plan['rounds'][:2]==old_plans[s]['rounds']
        assert plan['episodes'][:22]==old_plans[s]['episodes']
    heldout=deepcopy(records)
    removed={(g['recording'],tid) for p in plans.values() for g in p['rounds'] for tid in g['trial_ids']}
    for r in heldout:
        for t in r['annotations']:
            if (r['recording'],t['trial_id']) in removed:
                assert t['split']=='test'
                t['split']='personal_calibration'
    audit_partition(heldout,rows,classes,plans)
    return plans,heldout


def audit_partition(records, rows, classes, plans):
    lookup={(r['recording'],t['trial_id']):(r,t) for r in records for t in r['annotations']}
    for s,p in plans.items():
        assert len(p['rounds'])==SHOTS and len({(g['recording'],g['block']) for g in p['rounds']})==SHOTS
        assert Counter(e['word'] for e in p['episodes'])=={c:SHOTS for c in classes}
        assert len({(e['recording'],e['trial_id'],e['label']) for e in p['episodes']})==55
        contexts={(g['recording'],tid) for g in p['rounds'] for tid in g['trial_ids']}
        assert len(contexts)==50
        actual={(r['recording'],t['trial_id']) for r in records if r['subject']==s
                for t in r['annotations'] if t['split']=='personal_calibration'}
        assert contexts==actual
        for e in p['episodes']:
            assert (e['recording'],e['trial_id']) in contexts and e['indices']
            r,t=lookup[(e['recording'],e['trial_id'])]
            assert r['subject']==s and t['split']=='personal_calibration'
            for i in e['indices']:
                w=rows[i]
                assert (w['recording'],w['trial_id'],w['label'])==(e['recording'],e['trial_id'],e['label'])
                assert w['subject']==s and w['split']=='test'
                assert t['prepare']<=w['start']<w['end']<=t['context_end']
                assert base.training_label(t,w['start'],w['end'],classes)==e['label']
                if e['word']=='rest':assert w['start']>=t['rest']+600
                assert all(w['end']<=a or w['start']>=b for a,b in base.split_runs(r,'test'))


def make_schedule(plan, rows, epochs, seed):
    rng=np.random.default_rng(seed);schedule=[]
    for _ in range(epochs):
        ix=[int(rng.choice(e['indices'])) for e in plan['episodes']]
        rng.shuffle(ix)
        assert len(ix)==55 and len(set(ix))==55
        assert Counter(rows[i]['label'] for i in ix)=={c:SHOTS for c in range(11)}
        schedule.append(ix)
    return schedule


def prepare(out, reference):
    assert io.read(reference/'completed.json')['complete']
    hashes={}
    for marker in ['inputs_sha256.json','partition_sha256.json','training_sha256.json','export_sha256.json']:
        base.check_seal(reference,marker)
        hashes.update(io.read(reference/marker));hashes[str(reference/marker)]=io.digest(reference/marker)
    old=io.read(reference/'protocol.json');source=Path(old['base'])
    cp=torch.load(source/'model.pt',map_location='cpu',weights_only=True)
    assert io.digest(source/'model.pt')==old['base_checkpoint_sha256']
    rows=io.read(source/f'windows_{cp["window_ms"]}.json')
    old_plans=io.read(reference/'personal_calibration.json')
    previous.audit_partition(io.read(reference/'evaluation_records.json'),rows,old_plans,cp['classes'])
    original=base.read_records()
    assert sorted(old_plans)==SUBJECTS
    plans,records=expand_calibration(original,rows,cp['classes'],old_plans)
    counts=Counter(r['subject'] for r in records for t in r['annotations'] if t['split']=='test')
    assert dict(counts)==dict(S011=50,S012=50,S013=50,S014=49,S015=50)
    for k in ['normalization','window_ms','classes']:assert cp[k]==old[k]
    assert old['decoder']==io.read(source/'decoder_selection.json')['selected']['config']
    assert old['percentile']==io.read(source/'export/quantization_selection.json')['selected']['percentile']
    ptq=io.read(reference/'ptq_indices.json')
    assert all(rows[i]['split'] in ['train','validation'] for i in ptq)
    assert [rows[i]['fingerprint'] for i in ptq]==io.read(source/'export/calibration_manifest.json')['calibration_fingerprints']
    config={k:deepcopy(old[k]) for k in ['base','base_checkpoint_sha256','classes','normalization','window_ms',
        'step_ms','epochs','learning_rate','seed','decoder','percentile','original_test']}
    config.update(schema='personal_head_five_shot_v1',two_shot_reference=str(reference),shots=SHOTS,
        remaining_test=dict(counts),trainable_parameters=363,
        training='Independent original base initialization; classifier weight/bias only; inherited Adam/LR/epochs/seed/dropout; '
                 'one window per source episode, 55 windows in one step per epoch; frozen BN parameters/statistics.',
        selection='Keep exactly the prior two rounds and 22 episodes; append first three additional complete eligible '
                  'original-test rounds in recording/block order. One distinct post-action rest episode per new round.',
        rest_policy='Original two rest episodes/window sets unchanged (600ms guard). New three rest episodes use '
                    'the original base training label: 300ms after cue rest, ending at least 100ms before context end.',
        evaluation='Both base and five-shot head replay from scratch on the same remaining contiguous test runs. '
                   'Exclude all 50 calibration command contexts per person, including rest/transition/unused windows.',
        ptq='Original base training-side calibration windows and frozen percentile. No remaining test calibration.',
        limitations='Exploratory seen-person follow-up after inspecting prior test results, not a new blind/new-day test. '
                    'The denominator and rest selection policy differ from the two-shot experiment; do not directly '
                    'compare its 399-trial accuracy with this 249-trial accuracy. No deployment or board latency claim.')
    schedules={s:make_schedule(plans[s],rows,config['epochs'],config['seed']) for s in SUBJECTS}
    io.write(out/'protocol.json',config);io.write(out/'personal_calibration.json',plans)
    io.write(out/'evaluation_records.json',records);io.write(out/'ptq_indices.json',ptq)
    io.write(out/'epoch_window_indices.json',schedules)
    io.write_csv(out/'calibration_windows.csv',[rows[i] for p in plans.values() for e in p['episodes'] for i in e['indices']])
    for p in [Path(__file__).resolve(),Path(head.__file__).resolve(),reference/'completed.json']:
        hashes[str(p)]=io.digest(p)
    io.write(out/'inputs_sha256.json',hashes)
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for p in [Path(__file__),Path(head.__file__),Path(previous.__file__),Path(base.__file__)]:shutil.copy2(p,snapshot/p.name)
    base.seal(out,'partition_sha256.json',[out/n for n in ['protocol.json','personal_calibration.json',
        'evaluation_records.json','ptq_indices.json','epoch_window_indices.json']])
    for s in SUBJECTS:print(f'PRECHECK {s}: 50 commands + 5 distinct rest episodes; remaining test {counts[s]}',flush=True)
    return cp,rows,records,plans,config,schedules


def compare(out, cp, records, config):
    results={};all_events=defaultdict(list)
    for s in SUBJECTS:
        target=out/s;own=[r for r in records if r['subject']==s];waves=base.load_waves(own)
        personal=torch.load(target/'model.pt',map_location='cpu',weights_only=True)
        group={}
        for kind in ['fp32','int8']:
            filename='model_fp32.onnx' if kind=='fp32' else 'model_int8_qdq.onnx'
            trials_by_mode={}
            for mode,checkpoint,path in [('base',cp,Path(config['base'])/'export'/filename),('head5',personal,target/filename)]:
                name=f'{mode}_{kind}'
                trace=base.infer_runs(path,checkpoint,own,waves,'test')
                head.verify_trace_grid(trace,own,cp['window_ms'])
                np.savez_compressed(target/f'{name}_trace.npz',**trace)
                events=base.decode(trace,config['decoder'],cp['classes'])
                metrics,trials,confusion=previous.score(events,own,cp['classes'])
                assert metrics['trials']==config['remaining_test'][s]
                group[name]=metrics;trials_by_mode[mode]=trials;all_events[name].extend(events)
                io.write(target/f'{name}_events.json',events)
                io.write_csv(target/f'{name}_trials.csv',trials);io.write_csv(target/f'{name}_confusion.csv',confusion)
            paired=previous.paired_trials(trials_by_mode['base'],trials_by_mode['head5'])
            io.write_csv(target/f'paired_base_vs_head5_{kind}.csv',[
                {k.replace('personal_','head5_'):v for k,v in t.items()} for t in paired])
        results[s]=group;io.write(target/'comparison.json',group);del waves
    overall={n:previous.score(ev,records,cp['classes'])[0] for n,ev in all_events.items()}
    io.write(out/'comparison.json',dict(by_subject=results,overall=overall))
    groups=list(results.items())+[('ALL',overall)]
    io.write_csv(out/'comparison.csv',[dict(subject=s,model=n,**m) for s,g in groups for n,m in g.items()])
    deltas=[]
    for s,g in groups:
        for kind in ['fp32','int8']:
            a,b=g[f'base_{kind}'],g[f'head5_{kind}']
            deltas.append(dict(subject=s,format=kind,correct_change=b['correct_unique_trials']-a['correct_unique_trials'],
                accuracy_change_pp=100*(b['end_to_end_recall']-a['end_to_end_recall']),
                strict_change_pp=100*(b['clean_success_rate']-a['clean_success_rate']),
                wrong_output_change=b['wrong_label_outputs']-a['wrong_label_outputs'],
                no_response_change=b['no_response_trials']-a['no_response_trials']))
    io.write_csv(out/'head_deltas.csv',deltas)
    count=sum(config['remaining_test'].values())
    lines=[f'每类五遍、只微调分类层：base与head5在同一{count}条剩余测试上重新回放。',
        '模型           正确命令          严格成功率   错词输出/试次   无响应   静息输出/分钟']
    for s,g in groups:
        lines.append('\n'+s)
        for n,m in g.items():
            lines.append(f'{n:14s} {m["correct_unique_trials"]}/{m["trials"]} ({m["end_to_end_recall"]:.2%}) '
                f'{m["clean_success_rate"]:.2%}   {m["wrong_label_outputs"]}/{m["wrong_response_trials"]} '
                f'  {m["no_response_trials"]}   {m["rest_proxy_outputs_per_min"]:.3f}')
    lines+=['','每人50条动作+5段独立rest；旧两遍保留，新增三段rest沿用基模300ms等待规则。',
        '只训练363参数；完整校准上下文已移出测试。基模和个人模型均重新连续推理。',
        '不能与此前399条上的准确率直接比较；这是探索性已见人员实验，没有部署或板端测时。']
    report='\n'.join(lines);(out/'results.txt').write_text(report,encoding='utf-8-sig');print(report,flush=True)


def run_all(out, cp, rows, records, plans, config, schedules):
    base.check_seal(out,'partition_sha256.json')
    original={n:v.clone() for n,v in cp['state_dict'].items()}
    for s in SUBJECTS:
        target=out/s;target.mkdir();waves=base.load_waves([r for r in records if r['subject']==s])
        head.train_person(cp,rows,waves,plans[s],s,config,schedules[s],target)
        assert all(torch.equal(v,cp['state_dict'][n]) for n,v in original.items())
        del waves
    base.seal(out,'training_sha256.json',[out/s/n for s in SUBJECTS for n in ['model.pt','training_history.json','freeze_audit.json']])
    ix=io.read(out/'ptq_indices.json');names={rows[i]['recording'] for i in ix}
    waves=base.load_waves([r for r in records if r['recording'] in names])
    xcal=base.batch(rows,ix,waves,cp['normalization']);del waves
    for s in SUBJECTS:
        target=out/s;personal=torch.load(target/'model.pt',map_location='cpu',weights_only=True)
        pre=base.export_fp(base.restore(personal),target/'model_fp32.onnx',xcal[:16])
        base.auto.quantize(pre,target/'model_int8_qdq.onnx',xcal,np.arange(len(xcal)),config['percentile'])
        base.previous.check_quantized_graph(target/'model_int8_qdq.onnx')
        contract=io.read(Path(config['base'])/'export/input_contract.json')
        contract.update(subject=s,adaptation='classifier_only_five_shot',parent_checkpoint_sha256=config['base_checkpoint_sha256'])
        io.write(target/'input_contract.json',contract)
        io.write(target/'resource_estimate.json',base.resource_estimate(base.restore(personal)))
        print('EXPORT COMPLETE',s,flush=True)
    del xcal
    base.seal(out,'export_sha256.json',[p for s in SUBJECTS for p in (out/s).glob('*.onnx')])
    base.verify(out);compare(out,cp,records,config)
    for marker in ['partition_sha256.json','training_sha256.json','export_sha256.json']:base.check_seal(out,marker)
    base.seal(out,'results_sha256.json',[out/n for n in ['comparison.json','comparison.csv','head_deltas.csv','results.txt']])
    io.write(out/'completed.json',dict(complete=True,no_deployment=True,remaining_test_trials=sum(config['remaining_test'].values())))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=DEFAULT_REFERENCE)
    parser.add_argument('--output',type=Path,help='NEW directory only; no overwrite or resume')
    parser.add_argument('--phase',choices=['all','prepare'],default='all')
    args=parser.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    out=(args.output or base.ROOT/'models'/('personal_head5_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    out.mkdir(parents=True,exist_ok=False);tmp=out/'temporary';tmp.mkdir();tempfile.tempdir=str(tmp)
    with (out/'run.log').open('w',encoding='utf-8') as log:
        with redirect_stdout(base.previous.Tee(sys.stdout,log)),redirect_stderr(base.previous.Tee(sys.stderr,log)):
            try:
                print('OUTPUT',out,flush=True);data=prepare(out,args.reference.resolve())
                if args.phase=='all':run_all(out,*data)
                print('PHASE COMPLETE',args.phase,out,flush=True)
            except Exception:
                traceback.print_exc();raise


if __name__=='__main__':main()
