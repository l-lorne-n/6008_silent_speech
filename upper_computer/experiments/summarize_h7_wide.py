"""Audit and compare completed wide/small CNN runs, without model selection."""
import argparse, csv, hashlib, json, shutil, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.cnn-deps'))
sys.path.append(str(ROOT/'.onnx-deps'))
import onnx

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvrows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def ident(r):return (r['subject'],r['recording'],str(r['trial_id']))
def save(p,r):p.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    folder=parser.parse_args().run.resolve();new=folder/'cnn'
    protocol=read(folder/'protocol.json');old=Path(protocol['comparison_reference'])/'cnn_6144'
    nr=read(new/'report.json');nq=read(new/'deployment_calibrated/report.json')
    sr=read(old/'report.json');sq=read(old/'deployment_calibrated/report.json')
    resource=read(new/'deployment_calibrated/resource_estimate.json')
    baseline=read(Path(protocol['baseline'])/'report.json')
    assert nr['protocol']['parameters']==87434
    differences={k for k in set(nr['protocol'])|set(sr['protocol']) if nr['protocol'].get(k)!=sr['protocol'].get(k)}
    assert differences=={'architecture','parameters','cnn_widths'},differences
    assert resource['training_parameters']==87434 and resource['macs']==28125184
    assert (new/'split_manifest.csv').read_bytes()==(old/'split_manifest.csv').read_bytes()
    assert nr['final_normalization']==sr['final_normalization']
    manifest=csvrows(new/'split_manifest.csv')
    sets={k:{r['fingerprint'] for r in manifest if r['split']==k} for k in ['train','validation','test']}
    cal=read(new/'deployment_calibrated/validation_candidates.json')
    assert set(cal['calibration_fingerprints'])<=sets['train']
    assert set(nq['calibration_fingerprints'])<=sets['train']|sets['validation']
    assert not set(nq['calibration_fingerprints'])&sets['test']
    assert len(nq['calibration_fingerprints'])==len(set(nq['calibration_fingerprints']))==500
    assert min(cal['candidates'],key=lambda c:c['validation_loss'])==nq['selected']
    assert nr['model_roundtrip_verified'] and nr['input_hashes_unchanged']
    assert read(new/'deployment/report.json')['fp32_onnx_matches_pytorch']
    checked=0
    for source in baseline['sources']:
        for name,h in source['hashes'].items():
            assert sha(Path(source['path'])/name)==h
            checked+=1
    for name,h in read(folder/'comparison_reference_sha256.json').items():assert sha(Path(name))==h
    onnx_file=new/'deployment_calibrated/model_int8_qdq.onnx'
    model=onnx.load(onnx_file);onnx.checker.check_model(model)
    initializers={v.name:v for v in model.graph.initializer}
    producers={o:n for n in model.graph.node for o in n.output}
    core=[]
    for node in model.graph.node:
        if node.op_type not in ('Conv','Gemm'):continue
        activation=producers[node.input[0]];weight=producers[node.input[1]]
        assert activation.op_type==weight.op_type=='DequantizeLinear'
        tensor=initializers[weight.input[0]]
        assert tensor.data_type==onnx.TensorProto.INT8
        assert initializers[activation.input[2]].data_type==onnx.TensorProto.INT8
        core.append(dict(name=node.name,op=node.op_type,weight_shape=list(tensor.dims),weight_dtype='INT8'))
    assert len(core)==4
    series=[('small_fp32',sr['metrics'],csvrows(old/'predictions.csv'),'prediction'),
            ('small_int8',sq['int8'],csvrows(old/'deployment_calibrated/predictions.csv'),'int8'),
            ('wide_fp32',nr['metrics'],csvrows(new/'predictions.csv'),'prediction'),
            ('wide_int8',nq['int8'],csvrows(new/'deployment_calibrated/predictions.csv'),'int8')]
    labels={ident(r):r['word'] for r in baseline['test']};classes=nr['protocol']['classes']
    correctness={};subjects=protocol['subjects'];subject_metrics={}
    for name,metric,rows,key in series:
        assert len(rows)==len(labels)==499
        assert {ident(r):r['word'] for r in rows}==labels
        correct={ident(r):(r[key]==r['word']) for r in rows};correctness[name]=correct
        matrix=[[0]*10 for _ in range(10)]
        for r in rows:matrix[classes.index(r['word'])][classes.index(r[key])]+=1
        assert sum(correct.values())==metric['correct'] and matrix==metric['confusion_matrix']
        f1=sum(2*matrix[i][i]/(sum(matrix[i])+sum(row[i] for row in matrix)) for i in range(10))/10
        assert abs(f1-metric['macro_f1'])<1e-12
        subject_metrics[name]={s:dict(correct=sum(v for k,v in correct.items() if k[0]==s),total=sum(k[0]==s for k in correct)) for s in subjects}
    pairs={}
    for dtype in ['fp32','int8']:
        a=correctness['small_'+dtype];b=correctness['wide_'+dtype]
        pairs[dtype]=dict(wide_only_correct=sum(b[k] and not a[k] for k in labels),
                         small_only_correct=sum(a[k] and not b[k] for k in labels),
                         both_correct=sum(a[k] and b[k] for k in labels),
                         both_wrong=sum(not a[k] and not b[k] for k in labels))
    audit=dict(status='passed',protocol_differences_only=sorted(differences),same_split_manifest_bytes=True,whole_round_split_audit=read(folder/'saved_outputs_audit.json'),
               same_training_normalization=True,source_files_rehashed_unchanged=checked,
               reference_models_unchanged=True,calibration_train_only=True,calibration_selected_by_validation_loss=True,
               source_predictions_and_metrics_recomputed=True,onnx_checker_passed=True,quantized_core_layers=core,
               fp32_onnx_matches_pytorch=True,st_compiler_verified=False,board_deployed=False,board_latency_measured=False)
    save(folder/'comparison_audit.json',audit)
    summary=dict(protocol=protocol,small=dict(fp32=sr['metrics'],int8=sq['int8'],selected=sr['selected']),
                 wide=dict(fp32=nr['metrics'],int8=nq['int8'],selected=nr['selected'],selected_quantization=nq['selected']['name']),
                 subjects=subject_metrics,paired=pairs,resources=resource,
                 int8_file_bytes=onnx_file.stat().st_size,pc_fp32_latency_ms=dict(median=nr['inference_ms_median'],p95=nr['inference_ms_p95']),audit=audit)
    save(folder/'comparison.json',summary)
    lines=['# 32/64/128 CNN 对比结果','',
           '本次只将三层卷积通道从16/32/64扩大到32/64/128，未部署或修改板上固件。', '',
           '| 模型 | FP32准确率 | INT8准确率 | FP32宏F1 | INT8宏F1 |',
           '|---|---:|---:|---:|---:|']
    for name,r,q in [('原16/32/64',sr,sq),('新32/64/128',nr,nq)]:
        lines.append(f'| {name} | {r["metrics"]["correct"]}/499 = {r["metrics"]["accuracy"]:.2%} | {q["int8"]["correct"]}/499 = {q["int8"]["accuracy"]:.2%} | {r["metrics"]["macro_f1"]:.4f} | {q["int8"]["macro_f1"]:.4f} |')
    delta=(nq['int8']['accuracy']-sq['int8']['accuracy'])*100
    lines+=['',f'INT8较原模型变化 {delta:+.2f} 个百分点，同一499条测试中净变化 {nq["int8"]["correct"]-sq["int8"]["correct"]:+d} 条正确预测。',
            '这是一轮固定结构实验；已有测试集曾被查看，不将单次变化解释为统计显著或新受试者泛化提升。','',
            '| 人员 | 测试数 | 原FP32 | 新FP32 | 原INT8 | 新INT8 |','|---|---:|---:|---:|---:|---:|']
    for subject in subjects:
        groups=[subject_metrics[name][subject] for name in ['small_fp32','wide_fp32','small_int8','wide_int8']]
        lines.append('| '+subject+' | '+str(groups[0]['total'])+' | '+' | '.join(f'{g["correct"]}/{g["total"]} = {g["correct"]/g["total"]:.2%}' for g in groups)+' |')
    lines+=['',f'新模型量化后相对其FP32下降{(nr["metrics"]["accuracy"]-nq["int8"]["accuracy"])*100:.2f}个百分点，少对{nr["metrics"]["correct"]-nq["int8"]["correct"]}条。各人量化变化需分别看表，不能把总体提升解释成人人改善。',
            '本次结果支持保留加宽浮点模型继续研究；下一步优先验证减少量化损失的方法，而非直接继续扩大网络。']
    lines+=['','## 训练与划分','',
            '沿用goodone_plus五人49份录制。有效2496条；每份录制按整轮80/20固定，种子20260924。开发训练1507条、验证490条；选定后以全部1997条训练从头重训，测试499条。新旧split_manifest.csv逐字节一致，各人都有训练和测试。',
            '双通道2kHz，20–500Hz带通+50Hz陷波；准备期滤波预热，动作首尾各裁150ms，6144点容量，标准化后右补零；无重采样或静默截断。训练侧均值/标准差与原模型完全相同。',
            '三次预定初始化20261010/11/12，最多120轮、早停耐心20，AdamW lr=0.001、weight_decay=0.001、batch32、dropout0.35；训练增强为共用±80点平移、每通道0.9–1.1幅值变化。按验证交叉熵选择种子和轮数。',
            f'新模型选中种子{nr["selected"]["seed"]}、第{nr["selected"]["epoch"]}轮，验证准确率{nr["selected"]["validation_accuracy"]:.2%}、验证损失{nr["selected"]["validation_loss"]:.6f}。原模型对应{sr["selected"]["validation_accuracy"]:.2%}、{sr["selected"]["validation_loss"]:.6f}。',
            f'训练选择与重训耗时{nr["elapsed_seconds"]:.1f}秒（不含此前数据提取、之后量化）。','',
            '## INT8量化','',
            '静态QDQ，权重逐通道对称INT8、激活逐张量非对称INT8。只在开发训练侧按每人每词抽10条（500条）校准开发模型，在490条验证上比较MinMax、99.99/99.9/99.5百分位，按验证交叉熵选择。然后同规则从最终1997条训练侧抽500条校准最终模型，最后评估测试。没有用测试数据校准，也没有先跑本轮MinMax测试再调参。',
            f'选中的校准方法：{nq["selected"]["name"]}；FP32/INT8预测一致率{nq["fp32_int8_prediction_agreement"]:.2%}。',
            'ONNX模型外部输入/输出仍为float32，内部卷积和全连接的权重与量化激活已核验为INT8；这不等于ST后端已验证所有算子均无浮点回退。','',
            '## 资源与板上适用性','',
            f'- 参数：87,434；MAC：{resource["macs"]:,}/次。',
            f'- INT8权重+INT32偏置：{resource["int8_weight_plus_int32_bias_bytes"]/1024:.2f} KiB；ONNX文件：{onnx_file.stat().st_size/1024:.2f} KiB。',
            f'- INT8输入与各物化层输出清单：{resource["tensor_inventory_int8_bytes"]/1024:.2f} KiB；加256 KiB规划预留：{resource["int8_tensor_inventory_plus_reserve_bytes"]/1024:.2f} KiB。',
            '- 张量清单可以通过内存复用减少，但未含真实后端scratch/布局转换；上述不是实测RAM。应让ST工具生成真实arena/Flash报告，并正确分配512KiB AXI SRAM与其他RAM区域。',
            f'- 电脑FP32单样本推理中位{nr["inference_ms_median"]:.3f} ms、P95 {nr["inference_ms_p95"]:.3f} ms；不是STM32延迟，不能用它承诺板上<200ms。',
            '- 本次没有接入调试器、烧录模型或改固件；此前只读核查已确认1MiB Flash，现有固件约31.4KiB。','',
            '## 解释范围和交付文件','',
            '数据曾经筛选/查看；当前是已见五人的离线提示分段十词任务，不是留一人测试，也未验证自主动作分段、静息/未知拒识或连续在线运行。扩大后是否值得部署，应综合验证结果、各人效果和实际板上资源/延迟，不只看汇总准确率。',
            '- cnn/model.pt：最终PyTorch模型；cnn/selection_checkpoint.pt：只用开发训练的模型。',
            '- cnn/deployment/model_fp32.onnx：经PyTorch对齐核验的浮点导出。',
            '- cnn/deployment_calibrated/model_int8_qdq.onnx：验证集选择校准方式后的INT8模型。',
            '- cnn/deployment_calibrated/input_contract.json：预处理系数、标准化、形状和类别顺序。',
            '- comparison.json / comparison_audit.json：机器可读对比与独立复核。',
            '- comparison.png：准确率、验证曲线、混淆矩阵。',
            '- training.log、selection_history.json、split_manifest.csv、predictions.csv：完整训练与逐条证据。','']
    (folder/'结果对比.md').write_text('\n'.join(lines),encoding='utf-8')
    shutil.copy2(__file__,folder/Path(__file__).name)
    files=[folder/'结果对比.md',folder/'comparison.json',folder/'comparison_audit.json',new/'model.pt',new/'deployment/model_fp32.onnx',onnx_file,new/'deployment_calibrated/input_contract.json']
    save(folder/'delivery_sha256.json',{str(p.relative_to(folder)):sha(p) for p in files})
    print(json.dumps(dict(fp32=nr['metrics']['accuracy'],int8=nq['int8']['accuracy'],delta_percentage_points=delta,resource_kib=resource['int8_tensor_inventory_plus_reserve_bytes']/1024,output=str(folder)),ensure_ascii=False))

if __name__=='__main__':main()
