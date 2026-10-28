"""Independent event-count audit, readable report, and replay diagnostic figure."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    out=Path(sys.argv[1]).resolve()
    report=read(out/'test_report.json');records=read(out/'recordings.json')
    chosen=read(out/'selected.json');audits={}
    truth={(r['recording'],t['trial_id']):t for r in records for t in r['annotations'] if t['split']=='test'}
    for kind in ['int8','fp32']:
        events=read(out/(kind+'_test_events.json'))
        accepted=[e for e in events if e['candidate'] is not None and e['probability']>=chosen['min_probability']]
        correct=clean=detected=duplicates=rest_outputs=0
        for key,t in truth.items():
            owned=[e for e in accepted if (e['recording'],e['owner_trial'])==key]
            matching=[e for e in owned if e['matched_trial']==t['trial_id']]
            hits=[e for e in matching if e['candidate']==t['word']]
            correct+=bool(hits)
            clean+=len(owned)==1 and bool(hits)
            detected+=bool([e for e in events if (e['recording'],e['matched_trial'])==key and e['candidate'] is not None])
            duplicates+=max(0,len(matching)-1)
            for e in owned:
                overlap=max(0,min(e['active_end'],t['context_end'])-max(e['active_start'],t['rest']+600))
                rest_outputs+=overlap>=.5*(e['active_end']-e['active_start'])
        m=report['results'][kind]['selected_rejection']
        assert (correct,clean,detected,duplicates,rest_outputs)==(m['correct_unique_trials'],m['clean_correct_trials'],m['detected_trials'],m['duplicate_outputs'],m['rest_proxy_outputs'])
        assert all(e['end']<=e['emitted'] and e['length_samples']<=6144 for e in accepted)
        diagnostics=list(csv.DictReader((out/(kind+'_trial_diagnostics.csv')).open(encoding='utf-8-sig')))
        assert sum(r['clean_correct']=='True' for r in diagnostics)==clean
        control=list(csv.DictReader((out/(kind+'_cue_control.csv')).open(encoding='utf-8-sig')))
        assert sum(r['correct']=='True' for r in control)==report['results'][kind]['cue_cut_continuous_filter_control']['correct']
        audits[kind]=dict(trials=len(truth),correct=correct,clean=clean,detected=detected,rest_outputs=rest_outputs,
                          candidate_causality_verified=True,independent_counts_match=True)
    candidates=read(out/'validation_candidates.json')
    expected=max(candidates,key=lambda c:(c['metrics']['clean_success_rate'],c['metrics']['event_f1'],-c['metrics']['outputs']))
    assert expected==chosen
    # Retain the initial scoring implementation for transparency; the correction
    # was completed before test inference and did not change the selected gate.
    prior=read(out/'selected_before_scoring_audit.json')
    assert (prior['config'],prior['min_probability'])==(chosen['config'],chosen['min_probability'])
    protocol=read(out/'protocol.json')
    protocol['threshold_calibration']='First 5s, exclude first 500ms, clipped frames and next 200ms; require >=3s usable calibration; fixed thresholds after calibration.'
    protocol['scoring_audit']='Before test: corrected clean success to penalize any extra output in the same trial context, including background outputs. Gate selection unchanged. Earlier scoring files preserved.'
    (out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    int8=report['results']['int8']['selected_rejection'];fp32=report['results']['fp32']['selected_rejection']
    lines=[
        '方法二：连续活动检测 + 现有 CNN 回放验证（2026-10-21）',
        '',
        '结论：现有完整录制足够进行第一阶段连续回放，无需板子和电极。本轮未重新训练、未连接设备、未部署。',
        '当前直接拼接活动检测器和原整词 CNN 的效果尚不足以稳定实时使用；优先解决分段/训练输入对齐，而非继续扩大网络。',
        '',
        f'数据：goodone_plus 五人49份录制，共 {report["total_recorded_minutes"]:.2f} 分钟。每份含开头5秒静息及词间准备/静息。',
        '沿用原整轮划分：开发训练1507、验证490、最终测试499。CNN没有重训；选择检测参数时使用仅开发训练训练的INT8开发模型。',
        '检测器在完整时间序列连续运行，不读取提示、词语、trial、block、split。评分器事后读取标记，按原测试轮次评分。',
        '信号：双通道2kHz；20–500Hz带通+50Hz陷波持续因果滤波；60ms尾随RMS，每20ms更新。',
        '静息校准：每份开头5秒，跳过首500ms及削顶后200ms。3份开头各有1帧削顶，已剔除污染校准帧；要求至少3秒有效校准。',
        '',
        '参数选择：只在验证集比较9种检测配置 × 4种置信度阈值，最大化每次试次仅输出一次正确词的比例；同分看事件F1。',
        '选中 on_ratio=4、off_ratio=2.2；启动持续60ms，结束静息320ms，重新触发静息500ms；前/后各保留150ms；活动200–2700ms。',
        '逐通道阈值还结合初始基线MAD及绝对下限。任一通道超过启动阈值可触发；两通道均低于结束阈值才累积静息。',
        '置信度阈值选中0，即本轮保留所有合格候选。更严格的置信度在验证集同时排除了大量正确命令，不能只看误报降低。',
        '这是上述候选中的验证集最优，并不意味着全局最佳检测器，也未按200ms总响应约束优化。',
        '',
        '固定499次测试动作结果（提示时间代理评分）：',
        '流程                                      FP32             INT8',
        '原提示分段                                412/499 82.57%   395/499 79.16%',
        '连续滤波 + 原提示分段（诊断对照）           412/499 82.57%   395/499 79.16%',
        f'自动分段后，至少一次识别正确               {fp32["correct_unique_trials"]}/499 {100*fp32["end_to_end_recall"]:.2f}%   {int8["correct_unique_trials"]}/499 {100*int8["end_to_end_recall"]:.2f}%',
        f'自动分段后，仅输出一次正确词且无额外输出   {fp32["clean_correct_trials"]}/499 {100*fp32["clean_success_rate"]:.2f}%   {int8["clean_correct_trials"]}/499 {100*int8["clean_success_rate"]:.2f}%',
        '',
        f'可用候选覆盖437/499次动作（87.58%）；62次未获得匹配候选。INT8输出465次，其中匹配动作437次（274对、163错），另有28次未匹配提示动作。',
        f'评分时间{int8["duration_minutes"]:.2f}分钟，其中提示静息（去除起始300ms保护区）{int8["scored_rest_minutes"]:.2f}分钟，出现19次输出，即{int8["rest_proxy_outputs_per_min"]:.2f}次/静息分钟。',
        '上述19次是提示静息代理误触发，不等于人工确认完全静息时的假阳性；28次未匹配输出还包括准备期或跨边界活动。',
        '没有同一提示动作的重复匹配输出，但存在正确词之外的额外背景输出，因此严格成功率低于至少识别正确一次的比例。',
        '',
        '各人 INT8：人员 / 测试动作 / 有匹配候选 / 至少一次正确 / 无额外输出的正确试次 / 提示静息输出',
    ]
    for s,m in report['results']['int8']['by_subject'].items():
        lines.append(f'{s}: {m["trials"]} / {m["detected_trials"]} / {m["correct_unique_trials"]} / {m["clean_correct_trials"]} / {m["rest_proxy_outputs"]}')
    lines += [
        '',
        '解释：连续滤波的对照分数与原分数完全相同。自动候选片段的中位长度约1.36秒，原训练片段通常约1.7秒，起止位置也不同。',
        '这支持优先调查自动分段与训练输入的分布差异，以及检测漏检/误触发；不能仅据此断言每个错误都由长度导致。',
        '下一步建议：仅用训练侧自动分段片段及边界扰动训练/微调适配模型，验证侧选择参数；保留连续因果预处理，再用新录连续数据验收。',
        '不建议从这次测试错误反复挑参数后继续把同一499条称为盲测；这些数据及旧成绩本来就被查看过。',
        '',
        '限制与延迟：',
        '1. 标记是提示起止而非真实动作起止。匹配规则：候选活动主体至少100ms、且至少50%落在其所属试次的动作提示区间。',
        '2. 自动候选不得跨训练/验证/测试边界用于推理；1个测试候选跨界被拒绝。检测/滤波状态仍连续演化，属于同录制混合轮次评估。',
        '3. 原质量排除的4个动作继续不作有效目标；无标签的未完成试次不当成静息负例。',
        '4. 结束等待320ms是在检测能量低于阈值后累积，并非真实肌肉动作结束时刻；本轮不提供板上推理耗时或真实端到端延迟。',
        '5. 保留最多6144点；过长活动拒绝，不偷偷截断。文件末尾未完成动作不会强行输出。',
        '6. 现有数据缺少系统标注的吞咽、咬牙、走动等负样本；不能据本次回放保证这些场景可靠。',
        '7. 本实验未添加静息类、未训练新检测神经网络、未重新训练原CNN，仍是方法二的第一轮基线。',
        '',
        '验证：17项测试通过（含旧活动流程）；分块不变性、因果前缀、静息不输出、文件末尾不强行输出、缺样/削顶、过长活动、评分去重和背景输出惩罚。',
        '独立复核逐事件/逐试次计数与对照预测一致；原始来源和已有模型哈希保持不变。',
        '复现：.runtime/Scripts/python.exe experiments/replay_activity_cnn.py --output <全新输出目录>',
        '查看 test_report.json、selected.json、validation_candidates.json、int8_test_events.csv、int8_trial_diagnostics.csv、replay_examples.png。',
    ]
    (out/'验证结果.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'independent_audit.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
    print(json.dumps(audits),flush=True)
    plot(out,records)


def plot(out,records):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    e=read(out/'int8_test_events.json');states={r['recording']:r for r in read(out/'int8_recording_states.json')}
    rec={r['recording']:r for r in records}
    examples=[next(x for x in e if x['subject']=='S011' and x['candidate'] and x['candidate']==x['matched_word']),
              next(x for x in e if x['subject']=='S014' and x['candidate'] and x['matched_word'] and x['candidate']!=x['matched_word']),
              next(x for x in e if x['subject']=='S015' and x['candidate'] and x['matched_trial'] is None
                   and x['active_start']>next(t['rest']+600 for t in rec[x['recording']]['annotations'] if t['trial_id']==x['owner_trial'])
                   and x['emitted']<=next(t['context_end'] for t in rec[x['recording']]['annotations'] if t['trial_id']==x['owner_trial']))]
    fig,axes=plt.subplots(3,1,figsize=(12,9),layout='constrained')
    titles=['Matched command, correct classification','Matched command, incorrect classification','Output during cued rest (proxy false trigger)']
    for ax,x,title in zip(axes,examples,titles):
        r=rec[x['recording']];t=next(t for t in r['annotations'] if t['trial_id']==x['owner_trial'])
        with np.load(out/'cache'/(x['recording']+'.npz')) as cache:
            lower=t['prepare'];upper=t['context_end'];mask=(cache['ends']>=lower)&(cache['ends']<=upper)
            secs=(cache['ends'][mask]-lower)/2000
            for ch,color in [(0,'#2563eb'),(1,'#d97706')]:
                ax.plot(secs,cache['rms'][mask,ch],color=color,lw=1.3,label=f'CH{ch} RMS')
                ax.axhline(states[x['recording']]['on'][ch],color=color,ls='--',lw=.8,label=f'CH{ch} onset threshold')
        ax.axvspan((t['action']-lower)/2000,(t['rest']-lower)/2000,color='#22c55e',alpha=.13,label='Action cue (not true onset)')
        ax.axvspan((x['start']-lower)/2000,(x['end']-lower)/2000,color='#9333ea',alpha=.14,label='CNN segment')
        ax.axvline((x['emitted']-lower)/2000,color='#dc2626',ls=':',label='Decision time (no MCU runtime)')
        ax.set_title(f"{title} | {x['subject']} trial {t['trial_id']} | cue: {t['word']} / output: {x['candidate']}",fontsize=11,loc='left')
        ax.set_xlabel('Seconds from preparation cue');ax.set_ylabel('Filtered RMS (ADC counts)')
        ax.set_xlim(0,(upper-lower)/2000);ax.grid(alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=9)
    fig.suptitle('Causal activity detection + frozen INT8 CNN: continuous replay examples',fontsize=14)
    fig.savefig(out/'replay_examples.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':main()
