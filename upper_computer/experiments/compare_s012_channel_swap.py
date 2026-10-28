"""Verify and summarize the explicitly requested S012 channel-swap experiment."""
import argparse
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    args = parser.parse_args()
    old, new = read(args.before/'report.json'), read(args.after/'report.json')
    for k in ('subjects', 'inventory', 'feature_config', 'variant', 'channels', 'preprocessing', 'estimator'):
        assert old['protocol'][k] == new['protocol'][k], k
    assert len(old['protocol']['excluded_recordings']) == 1
    assert Path(old['protocol']['excluded_recordings'][0]['path']).name == '20261016_125109_fe651f'
    assert new['protocol']['excluded_recordings'] == []
    assert not (Path(__file__).resolve().parents[1]/'recordings/S012/session_003/20261016_125109_fe651f').exists()
    old_fold = read(Path(old['folds'][0]['folder'])/'report.json')
    new_fold = read(Path(new['folds'][0]['folder'])/'report.json')
    key = lambda r: (r['subject'], r['recording'], r['trial_id'])
    original = {key(r): r for r in old_fold['train']+old_fold['test']}
    swapped = {key(r): r for r in new_fold['train']+new_fold['test']}
    assert original.keys() == swapped.keys() and len(original) == 1500
    assert [(key(r), r['reason']) for r in old['excluded']] == [(key(r), r['reason']) for r in new['excluded']]
    for identity, r in swapped.items():
        before = original[identity]
        assert r['recording'] != '20261016_125109_fe651f'
        for field in ('word', 'prepare', 'action', 'rest', 'boundary_rtt_ms'):
            assert before[field] == r[field]
        expected = before['features']
        if r['subject'] == 'S012':
            assert len(expected) == 48
            expected = expected[24:]+expected[:24]
            assert r['signal_fingerprints'] == before['signal_fingerprints'][::-1]
        else:
            assert r['fingerprint'] == before['fingerprint']
        np.testing.assert_array_equal(expected, r['features'])
    rows = []
    for a, b in zip(old['folds'], new['folds']):
        assert a['held_out_subject'] == b['held_out_subject']
        assert a['train_count'] == b['train_count'] == 1000
        assert a['test_count'] == b['test_count'] == 500
        rows.append(dict(subject=a['held_out_subject'], before_correct=a['correct'], after_correct=b['correct'],
                         test_count=500, before_accuracy=a['accuracy'], after_accuracy=b['accuracy'],
                         delta_percentage_points=100*(b['accuracy']-a['accuracy'])))
    notes = ('This is a user-requested exploratory test of suspected reversed S012 channels after viewing baseline results. '
             'Only S012 input channel order changed. It does not establish physical wiring reversal. '
             'Current S012 CSV files remain swapped; complete original backup and mapping are in '
             'history/data_corrections/20261016_S012_channel_swap. Original transport bytes remain unchanged.')
    result = dict(before=str(args.before.resolve()), after=str(args.after.resolve()), folds=rows,
                  before_accuracy=old['pooled_metrics']['accuracy'], after_accuracy=new['pooled_metrics']['accuracy'],
                  delta_percentage_points=100*(new['pooled_metrics']['accuracy']-old['pooled_metrics']['accuracy']),
                  identical_trial_set_verified=True, only_s012_feature_channel_permutation_verified=True,
                  s012_stopped_recording_excluded=True, notes=notes)
    (args.after/'channel_swap_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for p in [args.after/'protocol.json', args.after/'report.json']:
        data=read(p)
        data['channel_swap_experiment']=dict(subject='S012',mapping=[1,0],baseline=str(args.before.resolve()),notes=notes)
        p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['S012 双通道交换前后：相同29份录制、1500次动作、三折留人测试。',
           'S012中止录制20261016_125109_fe651f整组排除；S011、S015通道保持原样。', '']
    for r in rows:
        lines.append(f"{r['subject']}：{r['before_correct']}/500 ({r['before_accuracy']:.2%}) → {r['after_correct']}/500 ({r['after_accuracy']:.2%})；变化 {r['delta_percentage_points']:+.2f} 个百分点。")
    lines += ['', f"总体：{result['before_accuracy']:.2%} → {result['after_accuracy']:.2%}，变化 {result['delta_percentage_points']:+.2f} 个百分点。",
              '已逐动作核对：S012的48维特征只交换前后两组24维；其余两人的特征完全一致。动作标签、时间边界、训练设置与有效动作集合一致。',
              '本次基于已看过基线成绩后的接反假设进行探索性对照，不能仅凭准确率确认实际接线。',
              '当前S012的CSV仍为交换后的版本，元数据已记录映射；原始串口字节流保持原样。完整原件备份在history/data_corrections/20261016_S012_channel_swap/original_S012.zip。']
    (args.after/'通道交换对照.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True))


if __name__ == '__main__':
    main()
