"""Write the auto-segment CNN comparison and a standalone scientific figure."""
import json
from pathlib import Path
import sys


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    out=Path(sys.argv[1]).resolve();r=read(out/'comparison.json');m=r['results']
    audit=read(out/'independent_audit.json');data=read(out/'dataset_summary.json')
    train=read(out/'training_report.json');selected=train['selected']
    quant=read(out/'export/quantization_selection.json')['selected']
    resource=read(out/'export/resource_estimate.json')
    lines=['自动分段训练32/64/128：新旧模型连续回放对比','',
           '本轮只在电脑上训练、导出和回放，没有连接或部署到STM32。原始录制与旧模型均保留且哈希不变。',
           '比较对象：同一RMS活动检测器，同样的完整连续录制，同一499次测试命令，同一0置信度阈值，仅替换CNN及其训练侧标准化参数。',
           '网络仍为3层卷积32/64/128，卷积核15/9/7、步长2、池化2/2/24、Dropout0.35；87,434参数。没有增加静息/未知类。',
           '',
           '主要结果（分母始终包括全部499次测试命令）：',
           '类型        原模型正确命令           新模型正确命令           提升',
    ]
    for kind in ['fp32','int8']:
        a=m['old_'+kind]['overall'];b=m['new_'+kind]['overall']
        delta=b['correct_unique_trials']-a['correct_unique_trials']
        lines.append(f'{kind:8} {a["correct_unique_trials"]}/499 ({a["end_to_end_recall"]:.2%})       {b["correct_unique_trials"]}/499 ({b["end_to_end_recall"]:.2%})       {delta:+d}条，{100*(b["end_to_end_recall"]-a["end_to_end_recall"]):+.2f}个百分点')
        lines.append(f'         无额外输出且仅一次正确：{a["clean_correct_trials"]}/499 ({a["clean_success_rate"]:.2%}) → {b["clean_correct_trials"]}/499 ({b["clean_success_rate"]:.2%})')
        lines.append(f'         同一437次匹配候选的分类正确率：{a["correct_unique_trials"]}/437 ({a["correct_unique_trials"]/437:.2%}) → {b["correct_unique_trials"]}/437 ({b["correct_unique_trials"]/437:.2%})；此分母不能代替端到端指标。')
        p=m['paired_'+kind]
        lines.append(f'         配对：新模型独对{p["new_only_correct"]}、旧模型独对{p["old_only_correct"]}、两者均对{p["both_correct"]}、两者均错或漏{p["both_wrong_or_missed"]}。')
    a=m['old_int8']['overall'];b=m['new_int8']['overall']
    lines+=['',f'检测覆盖均为{b["detected_trials"]}/499，{b["missed_or_rejected_trials"]}次没有匹配候选。检测器未改，CNN无法补救前级漏检。',
            f'两模型均输出{b["outputs"]}次，其中{b["outside_cue_outputs"]}次未匹配提示动作；提示静息输出均为{b["rest_proxy_outputs"]}次/{b["scored_rest_minutes"]:.2f}分钟，即{b["rest_proxy_outputs_per_min"]:.2f}次/静息分钟。',
            '固定十词输出且没有置信度拒识，因此本轮不会减少误触发的数量；静息误触发可能被分成不同词。此项不是新模型失效，而是当前对照设计的边界。',
            '', '各人INT8，正确命令数/全部测试命令：']
    for s,new in m['new_int8']['by_subject'].items():
        old=m['old_int8']['by_subject'][s]
        lines.append(f'{s}: {old["correct_unique_trials"]}/{old["trials"]} ({old["end_to_end_recall"]:.2%}) → {new["correct_unique_trials"]}/{new["trials"]} ({new["end_to_end_recall"]:.2%})；变化{100*(new["end_to_end_recall"]-old["end_to_end_recall"]):+.2f}个百分点')
    newfp=m['new_fp32']['overall'];newq=m['new_int8']['overall']
    lines+=['','数据与训练：',
            f'原整轮划分不变：开发训练1507、验证490、测试499。自动分段后可明确用于监督学习的片段为{data["usable_by_split"]["train"]}/{data["usable_by_split"]["validation"]}/{data["usable_by_split"]["test"]}。',
            '检测器不读取提示标签；事后用已有标签匹配。多候选、无明确词标签、跨试次上下文、削顶等片段不强制当作某个词训练。背景片段仍参与完整回放评分。',
            '标准化只拟合训练侧有效片段。训练从头进行，三个固定种子20261010/11/12，AdamW lr0.001、weight_decay0.001、batch32、最多120轮、早停20。',
            '新增边界增强：在同试次、且不晚于检测器输出时刻的连续波形中，起止各独立扰动最多±100ms。保留原±40ms零填充平移和0.9–1.1通道幅值增强。测试不增强。',
            f'依据验证片段交叉熵选中seed={selected["seed"]}、epoch={selected["epoch"]}，验证片段准确率{selected["validation_accuracy"]:.2%}、损失{selected["validation_loss"]:.6f}。',
            f'随后从头使用全部{train["final_training_segments"]}个train+validation自动片段重训{selected["epoch"]}轮；训练选择及最终重训合计{train["elapsed_seconds"]:.1f}秒。',
            '最终测试始终保留499个目标，不因自动分段失败而降低分母。没有用本轮测试成绩选择种子、轮数、量化方法或阈值。',
            '', 'INT8与资源：',
            f'静态PTQ QDQ，权重逐通道、激活逐张量；用训练侧代表片段校准，验证集比较MinMax/99.99/99.9/99.5，选中{quant["name"]}。没有使用测试样本校准。',
            f'新模型浮点到INT8端到端正确命令比例变化：{newfp["end_to_end_recall"]:.2%} → {newq["end_to_end_recall"]:.2%}，{100*(newq["end_to_end_recall"]-newfp["end_to_end_recall"]):+.2f}个百分点。',
            f'结构未变：{resource["training_parameters"]:,}参数，{resource["macs"]:,}次乘加，INT8权重及偏置{resource["int8_weight_plus_int32_bias_bytes"]/1024:.2f}KiB；张量清单+256KiB规划预留仍约{resource["int8_tensor_inventory_plus_reserve_bytes"]/1024:.2f}KiB。',
            '资源仍是估算，不是ST编译器arena或板上实测。检测结束等待仍为320ms，未改善这一等待时间。',
            '', '证据与解释范围：',
            '新旧检测事件逐项相同，旧FP32/INT8回放指标完整复现上一轮；所有465个测试候选的FP32 ONNX与PyTorch输出逐项数值对齐。',
            '独立审计核查训练/校准成员、标准化、网络形状、INT8卷积及全连接、逐事件/试次计数、数据/模型/缓存哈希；详细见independent_audit.json。',
            '这是自动分段数据、重新拟合标准化和边界增强这一整套适配方案的对照，不是将每种改动单独消融后的因果证明。',
            '录制和旧测试成绩此前已经查看，检测器也是上一轮验证选择的；本次属于已见五人的探索性比较，不是新受试者或全新盲测。',
            '提示时段不是人工标注的实际肌肉动作边界；静息误触发为提示代理指标。尚未测试新的佩戴状态、吞咽/咬牙/走动等系统负例，未测板上时延。',
            '', '文件：',
            'model.pt：最终自动分段CNN；selection_checkpoint.pt：只用开发训练数据的选择阶段模型。',
            'export/model_fp32.onnx、export/model_int8_qdq.onnx、export/input_contract.json：电脑验证后的模型与输入契约。',
            'comparison.json、comparison.png：对比；old/new_*_events.csv、old/new_*_trials.csv：事件和试次证据。',
            'segment_manifest.csv、excluded_segment_candidates.csv、protocol.json、训练/量化日志和曲线：复现依据。',
            '不要用旧按提示切片的预处理契约加载新模型；GUI/固件尚未接入此模型。',
    ]
    (out/'连续回放对比结果.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    plot(out,m)
    print('\n'.join(lines[:17]),flush=True)


def plot(out,m):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,3,figsize=(16,4.7),layout='constrained')
    colors=['#94a3b8','#2563eb']
    for ax,key,title in [(axes[0],'end_to_end_recall','Correct command / all 499 targets'),(axes[1],'clean_success_rate','One correct output, no extra output')]:
        x=np.arange(2)
        for j,prefix in enumerate(['old','new']):
            vals=[100*m[prefix+'_'+kind]['overall'][key] for kind in ['fp32','int8']]
            bars=ax.bar(x+(j-.5)*.34,vals,width=.34,color=colors[j],label=['Original CNN','Auto-segment trained CNN'][j])
            ax.bar_label(bars,fmt='%.1f%%',padding=3,fontsize=9)
        ax.set_xticks(x,['FP32','INT8']);ax.set_ylim(0,100);ax.set_ylabel('Percent');ax.set_title(title,fontsize=11)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax=axes[2];subjects=list(m['new_int8']['by_subject']);x=np.arange(len(subjects))
    for j,prefix in enumerate(['old','new']):
        vals=[100*m[prefix+'_int8']['by_subject'][s]['end_to_end_recall'] for s in subjects]
        bars=ax.bar(x+(j-.5)*.34,vals,.34,color=colors[j]);ax.bar_label(bars,fmt='%.0f',padding=3,fontsize=8)
    ax.set_xticks(x,subjects);ax.set_ylim(0,100);ax.set_title('INT8 by subject: all targets included',fontsize=11);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2)
    fig.suptitle('Frozen detector, same continuous replay: retraining on automatic segments',fontsize=14)
    fig.savefig(out/'comparison.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
