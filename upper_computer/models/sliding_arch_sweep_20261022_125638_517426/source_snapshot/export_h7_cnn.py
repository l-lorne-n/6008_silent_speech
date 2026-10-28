"""Export fixed H7 CNN, training-only static INT8 calibration, heldout check."""
import argparse
import copy
import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from train_small_cnn import H7SmallCNN, h7_from_checkpoint, extract_waveforms, split_data, normalize, write_json
import numpy as np
import torch
from torch import nn
from semg import lda
# Keep training NumPy/Torch imports fixed; export dependencies are isolated.
sys.path.append(str(ROOT/'.onnx-deps'))
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process


def fused_layers(net):
    layers=copy.deepcopy(net.layers).eval()
    for conv,bn in [(0,1),(4,5),(8,9)]:
        layers[conv]=torch.nn.utils.fuse_conv_bn_eval(layers[conv],layers[bn])
        layers[bn]=nn.Identity()
    return layers


def resource_report(net):
    model=fused_layers(net)
    x=torch.zeros(1,2,1,6144)
    records=[];macs=0
    with torch.inference_mode():
        for i,layer in enumerate(model):
            incoming=x.numel();x=layer(x)
            if isinstance(layer,nn.Conv2d):
                macs+=x.numel()*layer.in_channels*int(np.prod(layer.kernel_size))//layer.groups
            elif isinstance(layer,nn.Linear):
                macs+=layer.in_features*layer.out_features
            records.append(dict(index=i,type=type(layer).__name__,shape=list(x.shape),elements=x.numel(),
                                adjacent_int8_tensor_bytes=incoming+x.numel()))
    # A conservative tensor inventory: retain every arithmetic-layer output,
    # even though an embedded planner should reuse most of these buffers.
    materialized=[r for r in records if r['type'] not in ('Identity','Flatten','Dropout')]
    tensors=12288+sum(r['elements'] for r in materialized)
    weight_elements=sum(p.numel() for n,p in model.named_parameters() if n.endswith('weight'))
    bias_elements=sum(p.numel() for n,p in model.named_parameters() if n.endswith('bias'))
    return dict(training_parameters=sum(p.numel() for p in net.parameters()),
                macs=macs,fp32_parameter_bytes=sum(p.numel()*4 for p in model.parameters()),
                int8_weight_plus_int32_bias_bytes=weight_elements+4*bias_elements,
                tensor_inventory_int8_bytes=tensors,tensor_inventory_fp32_bytes=4*tensors,
                max_adjacent_int8_tensor_bytes=max(r['adjacent_int8_tensor_bytes'] for r in records),
                planning_reserve_bytes=256*1024,
                int8_tensor_inventory_plus_reserve_bytes=tensors+256*1024,
                fp32_tensor_inventory_plus_reserve_bytes=4*tensors+256*1024,
                reserve_note='256 KiB planning allowance for kernel workspace, acquisition/preprocessing buffers, stack and other firmware; an allowance, not a measured requirement. Weights assumed in Flash.',
                layers=records,st_compiler_verified=False,board_verified=False,
                limitation='Tensor counts and MACs are calculated, not ST-generated RAM/Flash totals. Existing linker only allocates 128 KiB general RAM + 128 KiB D2; deployment must place arena/buffers in suitable RAM banks including AXI SRAM.')


class Reader(CalibrationDataReader):
    def __init__(self,x,indices):
        self.iterator=iter(indices);self.x=x
    def get_next(self):
        i=next(self.iterator,None)
        return None if i is None else {'input':self.x[i:i+1,:,None,:]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--widths',nargs=3,type=int,default=[16,32,64],help='Random preflight model widths; trained exports infer widths from checkpoint.')
    parser.add_argument('--fp32-only',action='store_true',help='Prepare checked FP32 export for validation-selected quantization; do not run MinMax test.')
    args=parser.parse_args();out=args.folder.resolve()
    torch.set_num_threads(2)
    net=H7SmallCNN(widths=tuple(args.widths)).eval()
    if args.preflight:
        out.mkdir(parents=True,exist_ok=True)
    else:
        checkpoint=torch.load(out/'model.pt',map_location='cpu',weights_only=True)
        assert checkpoint['schema']=='semg_h7_cnn_v1'
        net=h7_from_checkpoint(checkpoint).eval()
    target=out/'deployment';target.mkdir(exist_ok=False)
    temporary=target/'temporary';temporary.mkdir()
    tempfile.tempdir=str(temporary.resolve())
    shutil.copy2(__file__,target/Path(__file__).name)
    resources=resource_report(net)
    assert resources['int8_tensor_inventory_plus_reserve_bytes'] < 1024*1024
    write_json(target/'resource_estimate.json',resources)
    layers=fused_layers(net)
    fp=target/'model_fp32.onnx'
    torch.onnx.export(layers,torch.zeros(1,2,1,6144),str(fp),input_names=['input'],output_names=['logits'],
                      opset_version=17,dynamo=False,do_constant_folding=True)
    onnx.checker.check_model(onnx.load(fp))
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(fp),options,providers=['CPUExecutionProvider'])
    sample=np.random.default_rng(20260924).normal(size=(1,2,1,6144)).astype(np.float32)
    with torch.inference_mode(): expected=net(torch.from_numpy(sample[:,:,0,:])).numpy()
    np.testing.assert_allclose(session.run(None,{'input':sample})[0],expected,rtol=1e-4,atol=1e-5)
    if args.preflight:
        # Check one real backward/update with the actual shape before relying on training.
        net.train();optim=torch.optim.AdamW(net.parameters(),lr=.001)
        loss=nn.functional.cross_entropy(net(torch.from_numpy(np.repeat(sample[:,:,0,:],2,axis=0))),torch.tensor([0,1]))
        loss.backward();assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        optim.step()
        write_json(target/'preflight.json',dict(export_checked=True,onnx_matches_pytorch=True,finite_training_gradients=True,random_untrained_model=True))
        print(json.dumps(resources));return
    report=json.loads((out/'report.json').read_text(encoding='utf-8'))
    baseline=json.loads((Path(report['protocol']['baseline'])/'report.json').read_text(encoding='utf-8'))
    rows,split=split_data(baseline)
    train=np.sort(np.r_[split['train'],split['validation']]);test=split['test']
    raw,lengths=extract_waveforms(rows,baseline,input_length=6144)
    x,scaler=normalize(raw,lengths,train);x=x.numpy()
    assert scaler==checkpoint['normalization']
    assert [rows[i]['fingerprint'] for i in train]==checkpoint['training_fingerprints']
    write_json(target/'calibration_manifest.json',dict(source='All final training trials; no test data used for quantization calibration.',
               fingerprints=[rows[i]['fingerprint'] for i in train],method='MinMax',per_channel=True,activation='signed asymmetric int8',weight='signed symmetric int8'))
    pre=target/'model_preprocessed.onnx'
    quant_pre_process(str(fp),str(pre),skip_optimization=True)
    write_json(target/'input_contract.json',dict(classes=checkpoint['classes'],normalization=scaler,
               sampling_rate_hz=2000,channels=[0,1],filter_sos=lda.filter_sos('notch50_bp500').tolist(),
               preprocessing=lda.preprocessing_config('notch50_bp500'),input_shape=[1,2,1,6144],
               padding='Right-side zero padding after normalization; no time resampling.',
               requires='Cue-defined action boundaries and preparation filter warmup; automatic segmentation is not implemented.'))
    if args.fp32_only:
        fp_logits=np.concatenate([session.run(None,{'input':x[i:i+1,:,None,:]})[0] for i in test])
        net.eval()
        with torch.inference_mode():
            pt_logits=np.concatenate([net(torch.from_numpy(x[part])).numpy() for part in np.array_split(test,20)])
        np.testing.assert_allclose(fp_logits,pt_logits,rtol=1e-4,atol=1e-4)
        assert lda.metrics([rows[i] for i in test],fp_logits.argmax(1),checkpoint['classes'])==report['metrics']
        write_json(target/'report.json',dict(fp32_onnx_matches_pytorch=True,fp32=report['metrics'],
                   resources=resources,onnx_bytes=fp.stat().st_size,quantization='Deferred to development-validation selection; no MinMax test evaluated.'))
        write_json(target/'calibration_manifest.json',dict(status='No quantization performed in this export stage; calibration chosen later using development validation.'))
        print('FP32_EXPORT_VERIFIED',fp,flush=True)
        return
    quant=target/'model_int8_qdq.onnx'
    quantize_static(str(pre),str(quant),Reader(x,train),quant_format=QuantFormat.QDQ,
                    per_channel=True,activation_type=QuantType.QInt8,weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax,
                    op_types_to_quantize=['Conv','Gemm','AveragePool','Flatten'],
                    extra_options={'ActivationSymmetric':False,'WeightSymmetric':True})
    onnx.checker.check_model(onnx.load(quant))
    qs=ort.InferenceSession(str(quant),options,providers=['CPUExecutionProvider'])
    fp_logits=np.concatenate([session.run(None,{'input':x[i:i+1,:,None,:]})[0] for i in test])
    q_logits=np.concatenate([qs.run(None,{'input':x[i:i+1,:,None,:]})[0] for i in test])
    net.eval()
    with torch.inference_mode(): pt_logits=np.concatenate([net(torch.from_numpy(x[part])).numpy() for part in np.array_split(test,20)])
    np.testing.assert_allclose(fp_logits,pt_logits,rtol=1e-4,atol=1e-4)
    fp_pred=fp_logits.argmax(1);q_pred=q_logits.argmax(1)
    test_rows=[rows[i] for i in test];train_rows=[rows[i] for i in train]
    assert lda.metrics(test_rows,fp_pred,checkpoint['classes'])==report['metrics']
    qm=lda.metrics(test_rows,q_pred,checkpoint['classes'])
    groups=lda.group_metrics(train_rows,test_rows,q_pred,checkpoint['classes'])
    result=dict(fp32=report['metrics'],int8=qm,group_metrics=groups,
                fp32_int8_prediction_agreement=float(np.mean(fp_pred==q_pred)),
                calibration_trials=len(train),test_trials=len(test),test_used_for_calibration=False,
                fp32_onnx_matches_pytorch=True,resources=resources,
                onnx_bytes=dict(fp32=fp.stat().st_size,int8=quant.stat().st_size),
                onnx_version=onnx.__version__,onnxruntime_version=ort.__version__,
                input='float32 [1,2,1,6144], normalized causal filtered signal, padded after normalization with zeros',
                output='ten logits, argmax in saved classes order',classes=checkpoint['classes'])
    write_json(target/'report.json',result)
    with (target/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['subject','recording','trial_id','word','fp32','int8','int8_correct'])
        for r,p,q in zip(test_rows,fp_pred,q_pred):
            writer.writerow([r['subject'],r['recording'],r['trial_id'],r['word'],checkpoint['classes'][p],checkpoint['classes'][q],int(r['word']==checkpoint['classes'][q])])
    write_json(target/'input_contract.json',dict(classes=checkpoint['classes'],normalization=scaler,
               sampling_rate_hz=2000,channels=[0,1],filter_sos=lda.filter_sos('notch50_bp500').tolist(),
               preprocessing=lda.preprocessing_config('notch50_bp500'),input_shape=[1,2,1,6144],
               padding='Right-side zero padding after normalization; no time resampling.',
               requires='Cue-defined action boundaries and preparation filter warmup; automatic segmentation is not implemented.'))
    print('QUANTIZED',json.dumps(dict(fp32=report['metrics']['accuracy'],int8=qm['accuracy'],size=result['onnx_bytes'])),flush=True)


if __name__=='__main__':main()
