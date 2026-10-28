"""Reproducible 32/64/128 experiment against the existing frozen small-CNN split."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib, json, os, shutil, subprocess, sys, uuid

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'models/cnn_goodone_plus_h7_20261021_143705_309702'

def main():
    out=ROOT/'models'/('cnn_goodone_plus_h7_w32_'+datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    out.mkdir()
    previous=json.loads((OLD/'protocol.json').read_text(encoding='utf-8'))
    protocol=dict(previous,comparison_reference=str(OLD),architecture='h7_fixed_pool8_conv2d_v1_width32_64_128',
                  cnn_widths=[32,64,128],cnn_subdir='cnn',final_cnn_subdir='cnn',
                  experiment='Change only convolution widths from 16/32/64 to 32/64/128. Same input, optimizer, dropout, augmentation, 3 seeds, 120 epoch cap and whole-round split.',
                  quantization_selection='MinMax, percentile99.99/99.9/99.5 selected using development validation loss; no initial MinMax final-test experiment.',
                  deployment_authorized=False)
    # Do not inherit an old input-adjustment note as a change made in this run.
    for key in ['input_length_adjustment','input_adjustment','initial_attempt_directory','initial_attempt_log']:
        protocol.pop(key,None)
    (out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    scripts=['run_h7_wide_comparison.py','train_small_cnn.py','export_h7_cnn.py','calibrate_h7_cnn.py','verify_cnn_mixed_outputs.py']
    for name in scripts:shutil.copy2(ROOT/'experiments'/name,out/name)
    reference_files=[OLD/'cnn_6144/model.pt',OLD/'cnn_6144/report.json',OLD/'cnn_6144/deployment_calibrated/report.json',OLD/'cnn_6144/split_manifest.csv']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in reference_files}
    (out/'comparison_reference_sha256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print('OUTPUT',out,flush=True)
    env=dict(os.environ,PYTHONIOENCODING='utf-8')
    stages=[('preflight','export_h7_cnn.py',[str(out/'preflight'),'--preflight','--widths','32','64','128']),
            ('training','train_small_cnn.py',['--baseline',previous['baseline'],'--output',str(out/'cnn'),'--h7','--h7-widths','32','64','128']),
            ('fp32_export','export_h7_cnn.py',[str(out/'cnn'),'--fp32-only']),
            ('quantization','calibrate_h7_cnn.py',[str(out/'cnn')]),
            ('audit','verify_cnn_mixed_outputs.py',[str(out)])]
    for stage,script,args in stages:
        print('STAGE',stage,flush=True)
        with (out/(stage+'.log')).open('w',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,'-u',str(ROOT/'experiments'/script),*args],cwd=ROOT,
                                   stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',env=env)
            for line in child.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
            if child.wait()!=0:raise RuntimeError(f'{stage} failed; see {out/(stage+".log")}')
    for p,h in hashes.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    assert (out/'cnn/split_manifest.csv').read_bytes()==(OLD/'cnn_6144/split_manifest.csv').read_bytes()
    print('COMPLETE',out,flush=True)

if __name__=='__main__':main()
