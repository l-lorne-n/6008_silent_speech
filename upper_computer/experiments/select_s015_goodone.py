"""Audit the frozen 14-recording result and copy ten selected recordings."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.signal import sosfilt, sosfilt_zi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from semg import lda
from semg.protocol import Decoder, DATA

BASE = ROOT/'models/lda_per_recording_20261021_141934_7a23bb'
AUDIT = ROOT/'analysis/20261021_S015_goodone_selection'
DEST = ROOT/'recordings/S015/goodone'
FILES = ['metadata.json', 'samples.csv', 'events.jsonl', 'trials.jsonl', 'frames.jsonl', 'transport.bin']


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def audit():
    original = json.loads((BASE/'results.json').read_text(encoding='utf-8'))
    assert len(original['recordings']) == 14
    rows = []
    for old in original['recordings']:
        folder = Path(old['path'])
        hashes = {name: sha(folder/name) for name in FILES}
        assert all(hashes[k] == v for k, v in old['source_hashes'].items())
        meta = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        report = json.loads((Path(old['model_folder'])/'report.json').read_text(encoding='utf-8'))
        trials = [json.loads(s) for s in (folder/'trials.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
        samples = np.loadtxt(folder/'samples.csv', delimiter=',', skiprows=1, usecols=(1,4,7), dtype=np.int64)
        assert np.all(np.diff(samples[:,0]) == 1)
        decoder = Decoder()
        decoded = []
        flags = 0
        with (folder/'transport.bin').open('rb') as f:
            while chunk := f.read(65536):
                for frame in decoder.feed(chunk):
                    if frame.kind == DATA:
                        assert frame.version == 2
                        values = np.frombuffer(frame.payload, dtype='<u2').reshape(-1,2)
                        decoded.append(np.column_stack((np.arange(frame.first,frame.first+frame.count),values)))
                        flags += int(frame.flags != 0)
        decoded = np.concatenate(decoded)
        decoded = decoded[(decoded[:,0] >= samples[0,0]) & (decoded[:,0] <= samples[-1,0])]
        assert np.array_equal(decoded, samples), 'CSV/transport mismatch: '+folder.name
        assert decoder.crc_errors == 0 and flags == 0
        accepted = report['train'] + report['test']
        assert len(accepted) == 50
        ratios = []
        sos = lda.filter_sos('notch50_bp500')
        for row in accepted:
            p,a,r = [int(row[k]-samples[0,0]) for k in ('prepare','action','rest')]
            raw = samples[p:r,1:].astype(float)-32768
            filtered = sosfilt(sos,raw,axis=0,zi=sosfilt_zi(sos)[:,:,None]*raw[0])[0]
            offset = a-p
            preparation = filtered[offset-1000:offset]
            action = filtered[offset+300:-300]
            ratios.append(np.sqrt(np.mean(action**2,axis=0))/np.maximum(np.sqrt(np.mean(preparation**2,axis=0)),1e-8))
        medians = np.median(ratios,axis=0)
        counters = ['missing','crc_errors','device_dropped','marker_timeouts','frame_gaps']
        delta = {k: meta['stats'].get(k,0)-meta['connection_stats_at_start'].get(k,0) for k in counters}
        assert all(v == 0 for v in delta.values())
        completed = [t for t in trials if t['status']=='completed']
        assert len(completed)==50 and set(Counter(t['word'] for t in completed).values())=={5}
        item = dict(recording=folder.name, session=old['session'], source=str(folder),
                    accuracy=old['accuracy'], correct=old['correct'], test_count=old['test_count'],
                    hashes=hashes, completed=50, trial_status_counts=dict(Counter(t['status'] for t in trials)),
                    accepted_trials=50, excluded=report['excluded'], csv_transport_equal=True,
                    stats_delta=delta, crc_errors=decoder.crc_errors, flagged_frames=flags,
                    decoder_discarded_bytes=decoder.discarded, decoder_tail_bytes=len(decoder.buffer),
                    clipped_samples_per_channel=np.sum((samples[:,1:]<=1)|(samples[:,1:]>=65534),axis=0).tolist(),
                    action_to_preparation_rms_median=medians.tolist(),
                    weaker_channel_ratio=float(min(medians)),
                    marker_rtt_median_ms=float(np.median([r['boundary_rtt_ms'] for r in accepted])),
                    marker_rtt_max_ms=max(r['boundary_rtt_ms'] for r in accepted))
        rows.append(item)
        print(folder.name, f"{item['accuracy']:.0%}", 'clipping', item['clipped_samples_per_channel'],
              'action/preparation', np.round(medians,3).tolist(), flush=True)
    # Primary: accuracy. Ties: fewer clipped samples, stronger weaker-channel
    # action/preparation contrast, lower marker latency, then recording ID.
    rows.sort(key=lambda r:(-r['accuracy'],sum(r['clipped_samples_per_channel']),
                            -r['weaker_channel_ratio'],r['marker_rtt_median_ms'],r['recording']))
    for i,row in enumerate(rows):
        row.update(rank=i+1, selected=i<10)
    data = dict(created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                baseline=str(BASE), destination=str(DEST),
                rule='All recordings must pass completeness and CSV/transport integrity. Rank accuracy descending; ties use fewer full-record ADC clipped samples, higher minimum of the two channel median action/preparation RMS ratios, lower median marker RTT, then recording ID.',
                ratio_definition='Per accepted trial: causal 20-500 Hz + 50 Hz notch; preparation last 0.5 s, action trimmed 0.15 s each end. Median ratio per channel, then use the weaker channel. This is activity contrast, not a calibrated SNR or proof of correct execution.',
                caveat='Selected using previously observed test accuracy and signal data. Development subset, not an independent final test set. Never import originals and copies together.',
                recordings=rows)
    AUDIT.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__, AUDIT/Path(__file__).name)
    lda.write_json(AUDIT/'selection_audit.json',data)
    print('SELECTED', [r['recording'] for r in rows[:10]], flush=True)


def copy():
    data = json.loads((AUDIT/'selection_audit.json').read_text(encoding='utf-8'))
    assert not DEST.exists() or not any(DEST.iterdir()), 'Destination must be empty'
    DEST.mkdir(parents=True,exist_ok=True)
    selected = [r for r in data['recordings'] if r['selected']]
    assert len(selected)==10
    for row in selected:
        src = Path(row['source'])
        assert all(sha(src/n)==h for n,h in row['hashes'].items())
        dst = DEST/row['recording']
        shutil.copytree(src,dst)
        assert all(sha(dst/n)==h and sha(src/n)==h for n,h in row['hashes'].items())
    assert len(list(DEST.glob('*/metadata.json')))==10
    # Also ensure the four excluded originals remain intact.
    for row in data['recordings']:
        assert all(sha(Path(row['source'])/n)==h for n,h in row['hashes'].items())
    data['copy_verification'] = dict(recordings=10, core_files=60, completed_trials=500,
                                     all_copy_hashes_match=True, all_84_original_file_hashes_unchanged=True)
    lda.write_json(DEST/'selection_manifest.json',data)
    lda.write_json(AUDIT/'selection_manifest.json',data)
    lines = ['# S015 goodone：十组筛选记录','',
             '从原始 14 组中选取 10 组，复制保存全部原始文件；原 14 组均保留。共 500 个完成动作、每词 50 次。','',
             '先检查文件、试次完整性、设备采样连续性、CRC、逐点 CSV/transport 一致性。合格组按既有逐组 LDA 准确率排序；同分依次比较全程 ADC 削顶点数（少优先）、较弱通道的动作/准备期 RMS 比中位数（大优先）、标记延迟（小优先）。','',
             '动作/准备期 RMS 比反映活动相对背景的对比程度，不是严格信噪比，也不能证明贴位或动作一定正确。所用配置及全部指标见 selection_manifest.json。','',
             '| 原录制 | LDA准确率 | CH0动作/准备比 | CH1动作/准备比 | 入选 |',
             '|---|---:|---:|---:|---|']
    for r in data['recordings']:
        a,b=r['action_to_preparation_rms_median']
        lines.append(f"| {r['recording']} | {r['accuracy']:.0%} | {a:.3f} | {b:.3f} | {'是' if r['selected'] else '否'} |")
    lines += ['', '已逐点核对 14 组 CSV 与 transport；复制的 60 个核心文件 SHA256 全部一致，原始 84 个文件保持不变。', '',
              '这是按已观察测试成绩筛选的开发数据子集，不能以筛选后的分数证明独立泛化性能。后续只扫描本 goodone 目录，不能同时导入其原始副本；扫描 S015 根目录会同时发现原件和副本。', '',
              '本次没有合并重训 LDA，也没有删除未入选原件。']
    text='\n'.join(lines)+'\n'
    (DEST/'筛选说明.md').write_text(text,encoding='utf-8')
    (ROOT/'history/2026-10-21_S015_goodone十组筛选.md').write_text(text,encoding='utf-8')
    print(json.dumps(data['copy_verification']))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--copy',action='store_true')
    args=parser.parse_args()
    copy() if args.copy else audit()
