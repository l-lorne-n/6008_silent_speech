"""Audit and summarize saved five-person H7 CNN results without retraining."""
import argparse
import hashlib
import shutil
from export_h7_cnn import *


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    out=parser.parse_args().run.resolve();cnn=out/'cnn_6144';deploy=cnn/'deployment_calibrated'
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    report=read(cnn/'report.json');quant=read(deploy/'report.json');validation=read(deploy/'validation_candidates.json')
    protocol=read(out/'protocol.json');base=read(Path(protocol['baseline'])/'report.json')
    manifest=list(csv.DictReader((cnn/'split_manifest.csv').open(encoding='utf-8-sig')))
    devtrain={r['fingerprint'] for r in manifest if r['split']=='train'}
    finaltrain={r['fingerprint'] for r in base['train']};test={r['fingerprint'] for r in base['test']}
    assert set(validation['calibration_fingerprints'])<=devtrain
    assert set(quant['calibration_fingerprints'])<=finaltrain
    assert not set(quant['calibration_fingerprints'])&test
    assert quant['selected']==min(validation['candidates'],key=lambda r:r['validation_loss'])
    rows=list(csv.DictReader((deploy/'predictions.csv').open(encoding='utf-8-sig')))
    identity=lambda r:(r['subject'],r['recording'],int(r['trial_id']))
    assert {identity(r):r['word'] for r in rows}=={identity(r):r['word'] for r in base['test']}
    assert len(rows)==len(test)==499
    pred=[report['protocol']['classes'].index(r['int8']) for r in rows]
    assert lda.metrics(rows,pred,report['protocol']['classes'])==quant['int8']
    assert sum(int(r['correct']) for r in rows)==381
    for source in base['sources']:
        for name,h in source['hashes'].items():assert digest(Path(source['path'])/name)==h
    model=onnx.load(deploy/'model_int8_qdq.onnx');onnx.checker.check_model(model)
    producers={name:n for n in model.graph.node for name in n.output}
    initializers={v.name:v for v in model.graph.initializer}
    core=[n for n in model.graph.node if n.op_type in ('Conv','Gemm')]
    assert len(core)==4
    for n in core:
        assert producers[n.input[0]].op_type=='DequantizeLinear'
        weight=producers[n.input[1]]
        assert weight.op_type=='DequantizeLinear'
        assert initializers[weight.input[0]].data_type==onnx.TensorProto.INT8
    resources=read(deploy/'resource_estimate.json')
    audit=dict(status='passed',validation_calibration_training_only=True,final_calibration_training_only=True,
               selection_minimizes_validation_loss=True,heldout_predictions_recomputed=True,
               all_core_layers_have_static_int8_weights_and_quantized_inputs=True,
               source_files_rehashed=196,source_hashes_unchanged=True,onnx_checker_passed=True,
               st_compiler_and_board_verified=False)
    write_json(out/'quantization_audit.json',audit)
    write_json(out/'report.json',dict(protocol=protocol,cnn=report,lda_metrics=base['metrics'],quantization=quant,resources=resources))
    fp_groups={g['subject']:g for g in report['group_metrics']['subject']}
    q_groups={g['subject']:g for g in quant['group_metrics']['subject']}
    lda_groups={g['subject']:g for g in base['group_metrics']['subject']}
    table=['| 人员 | 训练 | 测试 | LDA | CNN FP32 | CNN INT8 |','|---|---:|---:|---:|---:|---:|']
    for subject in protocol['subjects']:
        g=fp_groups[subject];q=q_groups[subject]
        table.append(f"| {subject} | {g['train_total']} | {g['test_total']} | {lda_groups[subject]['metrics']['accuracy']:.2%} | {g['metrics']['correct']}/{g['test_total']} = {g['metrics']['accuracy']:.2%} | {q['metrics']['correct']}/{q['test_total']} = {q['metrics']['accuracy']:.2%} |")
    table.append('| 合计 | 1997 | 499 | 216/499 = 43.29% | 380/499 = 76.15% | 381/499 = 76.35% |')
    lines=['# goodone_plus 五人混合 CNN 与 H7 量化结果','',
           '日期：2026-10-21，Asia/Singapore。使用 goodone_plus 内全部五人、49份录制；五人共用一个模型，每人各有训练和测试。原始数据、固件保持不变。','',
           *table,'',
           'CNN FP32 宏 F1 0.7600，INT8 宏 F1 0.7636。量化后仅多对1次，不解释为确定的性能提升；FP32/INT8预测一致率93.79%。','',
           '## 数据、划分与训练','',
           '2500个completed动作中，沿用现有提取规则排除S014的4次准备/动作削顶，剩2496次。S015另有1条rejected记录也排除，不计入2500。每份录制整轮80/20，种子20260924，各人40轮训练、10轮测试。S014因削顶实际397训练/99测试，其余各400/100。','',
           '1997次训练侧内部划分为1507开发训练/490验证；测试499次固定。三个预定初始化20261010/11/12，每个最多120轮，早停耐心20，按验证交叉熵选择。选中20261012第120轮，验证准确率74.90%；随后从头用全部1997次重训120轮。测试不参与训练侧标准化或CNN模型选择。','',
           '两路2kHz、20–500Hz+50Hz陷波，逐动作准备期预热，首尾各裁150ms。实际有效长度3380–5400点，中位3460点。初次4096容量在读取S011的20261016_150956_4ad099 trial40 previous（3秒动作）时停止，尚未训练；改为6144点固定容量以完整保留动作，其余右补零，无重采样。初次失败目录cnn和training.log保留，最终目录是cnn_6144。','',
           '## 网络与资源','',
           'H7SmallCNN：Conv2d 2→16→32→64，核1×15/1×9/1×7，时间步长2，BN/ReLU，前两层平均池化2，末端固定平均池化24得到8个时间格；Dropout0.35、Flatten、十分类。输入ONNX NCHW [1,2,1,6144]，不含动态形状或自适应池化。24778个训练参数，约777万MAC。CNN直接使用波形；LDA对照仍是六段四特征48维。','',
           f"- 融合BN后的FP32参数约{resources['fp32_parameter_bytes']/1024:.2f} KiB。",
           f"- INT8权重+INT32偏置约{resources['int8_weight_plus_int32_bias_bytes']/1024:.2f} KiB；最终ONNX容器{quant['onnx_bytes']}字节（{quant['onnx_bytes']/1024:.2f} KiB），包含图与量化参数。",
           f"- INT8输入和全部算子输出均保留的张量清单约{resources['tensor_inventory_int8_bytes']/1024:.2f} KiB；加256KiB规划预留，共{resources['int8_tensor_inventory_plus_reserve_bytes']/1024:.2f} KiB，低于1MiB。",
           '- 以上为网络张量计算和规划预算，256KiB不是实测工作区；权重按放Flash估计。不是ST编译器RAM报告，也不是板上运行证明。',
           '- 当前工程H743VG为1MB Flash、约1MB分区RAM；stream.sct只安排512KiB Flash、128KiB普通RAM和128KiB D2。实际部署必须合理使用AXI SRAM等区域，并核对DMA、工作区、栈和采集缓冲；不能直接把整个1MB当连续RAM。','',
           '## INT8校准及失败记录','',
           '直接用1997次最终训练动作做MinMax校准，初次测试仅263/499=52.71%，保留在deployment，不能用作推荐量化模型。输入量化步长约0.30个标准化单位，提示大动态范围损失小幅信号。','',
           '看到初次MinMax退化后，使用保存的开发检查点，在开发训练侧每人每词固定随机抽最多10次，共500次校准；只在490次开发验证上比较MinMax、99.99/99.9/99.5百分位。按验证交叉熵选择99.9（验证359/490=73.27%）。再用同样规则从最终1997次训练中抽500次，重新校准最终模型，最后评估499次测试，得到381/499。所有候选及选择记录保留；没有按最终测试成绩在候选中择优。','',
           '权重按通道对称int8、激活按张量非对称int8，静态QDQ，FP32输入/输出接口。正式使用deployment_calibrated/model_int8_qdq.onnx，输入预处理/类别顺序见同目录input_contract.json。该流程符合ST推荐的固定维度、静态QDQ导出路线，但本机未找到ST转换CLI，未生成ST C模型、未测试板端延迟、未刷写固件。','',
           '## 文件和验证','',
           '- 最终浮点权重：cnn_6144/model.pt（schema semg_h7_cnn_v1，使用H7SmallCNN加载）。',
           '- 最终INT8模型：cnn_6144/deployment_calibrated/model_int8_qdq.onnx。',
           '- 浮点ONNX：cnn_6144/deployment/model_fp32.onnx。',
           '- 逐次预测：cnn_6144/predictions.csv和deployment_calibrated/predictions.csv。',
           '- 划分、原始哈希、训练历史、校准候选、内存预算、输入契约、脚本快照均保留。',
           '- 模型重载、PyTorch/FP32 ONNX数值一致性、ONNX格式、四个核心层静态量化权重/输入、整轮隔离、训练成员、校准不含测试、混淆矩阵与分人成绩核验通过；196个源文件哈希复核不变。','',
           '这是已见五人、已见录制内的提示窗口分类。goodone_plus为已整理数据，尤其S015按此前单组测试成绩筛选过；且已看到初次MinMax测试后进行量化修复，因此属于探索性开发评估，不是新人、跨日重贴或独立盲测。连续自动起止、静息/未知词拒识和端到端上板演示仍未完成。','',
           '参考：[ST H743VG规格](https://estore.st.com/en/products/microcontrollers-microprocessors/stm32-32-bit-arm-cortex-mcus/stm32-high-performance-mcus/stm32h7-series/stm32h743-753/stm32h743vg.html)、[ST静态QDQ量化说明](https://dev.st.com/edge-ai-docs/st-edge-ai-core/4.0.1/en/docs/user-manual/core-quantization.html)。','']
    text='\n'.join(lines)
    (out/'结果汇总.md').write_text(text,encoding='utf-8')
    history=f'结果目录：[{out.name}](../models/{out.name}/结果汇总.md)。\n\n'+text
    (ROOT/'history/2026-10-21_goodone_plus五人混合CNN与H7量化.md').write_text(history,encoding='utf-8')
    (cnn/'deployment/初次MinMax结果说明.txt').write_text('此目录的model_int8_qdq.onnx为初次MinMax量化，测试263/499（52.71%），不推荐使用。最终量化模型在相邻deployment_calibrated目录；浮点model_fp32.onnx仍有效。\n',encoding='utf-8')
    guide='''# H7 部署入口

使用 model_int8_qdq.onnx；这是经开发验证集选择的99.9%百分位静态INT8 QDQ模型。
输入/滤波/归一化/补零/输出类别顺序以 input_contract.json 为准。

安装ST Edge AI Core后可先分析：

```powershell
stedgeai analyze --model model_int8_qdq.onnx --target stm32h7
```

检查实际权重、激活、工作区和算子映射，再生成C模型并做同输入数值验证。ST默认可能优化成int8输入接口，需要按生成模型的输入量化比例转换；不要盲目把float32数组传入int8接口。

模型当前仍需要提示动作边界和准备期滤波预热。使用真实H743VG的RAM分区与链接脚本，测完整固件占用和板上延迟后才能确认实际部署。没有把PC ONNX推理时间当作H7速度。
'''
    (deploy/'部署说明.md').write_text(guide,encoding='utf-8')
    outputs=[cnn/'model.pt',cnn/'report.json',deploy/'model_int8_qdq.onnx',deploy/'input_contract.json',deploy/'report.json',out/'saved_outputs_audit.json',out/'quantization_audit.json']
    write_json(out/'delivery_sha256.json',{str(p.relative_to(out)):digest(p) for p in outputs})
    shutil.copy2(__file__,out/Path(__file__).name)
    print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
