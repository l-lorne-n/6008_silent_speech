"""Export the frozen sweep QDQ graph. No training or calibration is performed."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / '.onnx-deps'), str(ROOT / '.cnn-deps')]
import numpy as np
import onnx
from onnx import numpy_helper
from scipy.signal import sosfilt_zi

from semg.board_models import NAMES, model_source


def source(name):
    return model_source(name, ROOT)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(name, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    folder = source(name)
    path = folder / 'model_int8_qdq.onnx'
    contract = json.loads((folder / 'input_contract.json').read_text(encoding='utf-8'))
    assert contract['input_shape'] == [1, 2, 1, 2800]
    assert contract['fs'] == 2000 and contract['step_samples'] == 200
    assert contract['classes'] == ['open','close','next','music','louder','photo','previous','quieter','start','stop','rest']
    assert contract['decoder'] == dict(probability=.5, confirm=2, rest_frames=2, margin=0.)
    graph = onnx.load(str(path)).graph
    arrays = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    convs = [n for n in graph.node if n.op_type == 'Conv']
    assert len(convs) == 5
    header = ['/* Generated from frozen ONNX. Do not edit. */']

    def arr(cname, x, ctype='float'):
        x = np.asarray(x).ravel()
        fmt = (lambda v: float(v).hex() + 'f') if ctype == 'float' else (lambda v: str(int(v)))
        header.append(f'static const {ctype} {cname}[{len(x)}] = {{\n' +
                      ',\n'.join(','.join(fmt(v) for v in x[i:i+24]) for i in range(0, len(x), 24)) + '\n};')

    def scalar(n):
        return arrays[n].item()

    total_weights = 0
    maxfeat = 5600
    maxcol = 0
    configs = []
    height, width, inc = 2, 2800, 1
    ins = scalar('/Transpose_output_0_scale')
    inz = scalar('/Transpose_output_0_zero_point')
    first_scale = ins
    for layer, idx in enumerate([0, 4, 8, 12, 16]):
        node = convs[layer]
        att = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        kh, kw = att['kernel_shape']; ph, pw, ph2, pw2 = att['pads']
        assert att['group'] == 1 and att['strides'] == [1, 1] and att['dilations'] == [1, 1]
        assert ph == ph2 == 0 and pw == pw2
        weights = arrays[f'layers.{idx}.weight_quantized']
        oc = weights.shape[0]
        assert oc <= 128
        assert weights.shape == (oc, inc, kh, kw)
        assert not np.any(arrays[f'layers.{idx}.weight_zero_point'])
        cs = scalar(f'/layers/layers.{idx}/Conv_output_0_scale')
        cz = scalar(f'/layers/layers.{idx}/Conv_output_0_zero_point')
        pool = [8, 4, 4, 1, 1][layer]
        oh, convw = height-kh+1, width+2*pw-kw+1
        ow = convw//pool
        # QDQ before ReLU must remain: saturating before pooling is significant.
        if layer < 4:
            outs = scalar(f'/layers/layers.{idx+3}/MaxPool_output_0_scale')
            outz = scalar(f'/layers/layers.{idx+3}/MaxPool_output_0_zero_point')
        else:
            outs, outz = cs, cz
        lut = np.clip(np.rint(np.maximum((np.arange(-128,128)-cz)*cs,0)/outs)+outz,-128,127)
        arr(f'w{layer}', weights, 'int8_t')
        arr(f'b{layer}', arrays[f'layers.{idx}.bias_quantized'], 'int32_t')
        arr(f'm{layer}', np.asarray(ins*arrays[f'layers.{idx}.weight_scale']/cs, dtype=np.float32))
        arr(f'lut{layer}', lut, 'int8_t')
        ncol = inc*kh*kw
        assert ncol % 2 == 0
        configs.append('{'+f'{inc},{oc},{height},{width},{oh},{ow},{kh},{kw},{pw},{pool},{inz},{cz},'
                       f'{total_weights},w{layer},b{layer},m{layer},lut{layer}'+'}')
        total_weights += weights.size
        maxfeat = max(maxfeat, oc*oh*ow)
        maxcol = max(maxcol, ncol)
        height, width, inc, ins, inz = oh, ow, oc, outs, outz
    avg = next(n for n in graph.node if n.op_type == 'AveragePool')
    avgatt = {a.name: onnx.helper.get_attribute_value(a) for a in avg.attribute}
    avgwidth = avgatt['kernel_shape'][1]
    bins = width//avgwidth
    avgscale = scalar('/layers/layers.20/AveragePool_output_0_scale')
    avgzero = scalar('/layers/layers.20/AveragePool_output_0_zero_point')
    assert avgscale == scalar('/layers/layers.21/Flatten_output_0_scale')
    assert avgzero == scalar('/layers/layers.21/Flatten_output_0_zero_point')
    fw = arrays['layers.23.weight_quantized']
    assert fw.shape == (11, inc*bins) and not np.any(arrays['layers.23.weight_zero_point'])
    arr('fc_w', fw, 'int8_t')
    arr('fc_b', arrays['layers.23.bias_quantized'], 'int32_t')
    logscale = scalar('logits_scale'); logzero = scalar('logits_zero_point')
    arr('fc_m', np.asarray(avgscale*arrays['layers.23.weight_scale']/logscale, dtype=np.float32))
    arr('filter_sos', contract['filter_sos'])
    arr('filter_zi', sosfilt_zi(contract['filter_sos']))
    arr('norm_mean', contract['normalization']['mean'])
    arr('norm_scale', contract['normalization']['scale'])
    hash_ = digest(path)
    constants = dict(MODEL_FEATURES=maxfeat, MODEL_COLUMN=maxcol, MODEL_WEIGHTS=total_weights,
                     MODEL_TAIL_CHANNELS=inc, MODEL_TAIL_WIDTH=width, MODEL_POOL_BINS=bins,
                     MODEL_AVG_WIDTH=avgwidth, MODEL_FC_INPUT=inc*bins, MODEL_INPUT_ZERO=scalar('/Transpose_output_0_zero_point'),
                     MODEL_AVG_ZERO=avgzero, MODEL_TAIL_ZERO=inz, MODEL_LOGIT_ZERO=logzero)
    for k,v in constants.items(): header.append(f'#define {k} {v}')
    for k,v in dict(MODEL_INPUT_SCALE=first_scale, MODEL_AVG_MULT=ins/avgwidth/avgscale,
                    MODEL_LOGIT_SCALE=logscale).items(): header.append(f'#define {k} {float(v).hex()}f')
    header += [f'#define MODEL_NAME "{name}"', f'#define MODEL_HASH "{hash_[:16]}"',
               'static const Layer model_layers[5] = {' + ',\n'.join(configs) + '};']
    (dest / 'model_data.h').write_text('\n'.join(header)+'\n', encoding='utf-8')
    manifest = dict(name=name, onnx=str(path), onnx_sha256=hash_, contract=contract,
                    contract_sha256=digest(folder/'input_contract.json'), constants=constants,
                    runtime='QDQ INT8 weights/activations; int32 sums; Cortex-M7 DSP int16-expanded weights')
    (dest/'model.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest


if __name__ == '__main__':
    export(sys.argv[1], sys.argv[2])
