"""Actual UART observations and on-device parity checks. No simulated success."""
import argparse
import binascii
from datetime import datetime
import json
from pathlib import Path
import struct
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import serial
from semg.board_protocol import BoardDecoder,result,model_info,envelope
from semg.serial_settings import BAUDRATES, DEFAULT_BAUDRATE


def collect(port,seconds,command='N'):
    decoder=BoardDecoder();packets=[]
    port.write((command+'\nI\n').encode())
    started=time.monotonic()
    while time.monotonic()-started<seconds:
        packets.extend(decoder.feed(port.read(4096)))
    return decoder,packets


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',default='COM7');p.add_argument('--seconds',type=float,default=20)
    p.add_argument('--parity',action='store_true');p.add_argument('--raw-parity',action='store_true');p.add_argument('--model',default='mid64_temporal3')
    p.add_argument('--baud',type=int,choices=BAUDRATES,default=DEFAULT_BAUDRATE)
    a=p.parse_args()
    out=ROOT/'board/logs'/datetime.now().strftime('%Y%m%d_%H%M%S_check');out.mkdir(parents=True)
    s=serial.Serial(port=None,baudrate=a.baud,timeout=.1,write_timeout=3);s.dtr=False;s.rts=False;s.port=a.port
    report=dict(port=a.port,model=a.model,baudrate=a.baud)
    try:
        s.open()
        decoder,packets=collect(s,a.seconds)
        # Discard any packet already in flight before our mode-change INFO.
        first_info=next((i for i,f in enumerate(packets) if f.kind==5),len(packets))
        rows=[result(f) for f in packets[first_info:] if f.kind==4]
        infos=[model_info(f) for f in packets if f.kind==5]
        report.update(crc_errors=decoder.crc_errors,infos=infos,results=len(rows),flags=sorted({f.flags for f in packets}))
        (out/'live.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        if not rows:raise RuntimeError('No actual board inference packets received')
        sys.path[:0]=[str(ROOT/'.onnx-deps'),str(ROOT/'.cnn-deps'),str(ROOT/'board')]
        import numpy as np
        envelopes=[f for f in packets[first_info:] if f.kind==8]
        if infos[-1].get('env_hz')==100:
            values=np.array([envelope(f) for f in envelopes])
            assert len(values)>100, 'Missing live board envelopes'
            gaps=sum(b.end-a.end!=20 for a,b in zip(envelopes,envelopes[1:]))
            report['envelope']=dict(packets=len(values),gaps=gaps,min=values.min(axis=0).tolist(),max=values.max(axis=0).tolist())
            assert gaps==0, 'Envelope packet gap'
        for field in ['inference_us','pipeline_us','filter_us','age_samples']:
            x=np.array([r[field] for r in rows]);report[field]=dict(min=float(x.min()),median=float(np.median(x)),p95=float(np.percentile(x,95)),max=float(x.max()))
        report['last']=rows[-1]
        report['non_100ms_steps']=sum(b['end']-a['end']!=200 for a,b in zip(rows,rows[1:]))
        if a.parity or a.raw_parity:
            import onnxruntime as ort
            import onnx
            from onnx import numpy_helper
            from export_model import source
            model=source(a.model)/'model_int8_qdq.onnx'
            graph=onnx.load(str(model)).graph
            init={t.name:numpy_helper.to_array(t) for t in graph.initializer}
            scale=init['/Transpose_output_0_scale'].item();zero=init['/Transpose_output_0_zero_point'].item()
            sess=ort.InferenceSession(str(model),providers=['CPUExecutionProvider'])
            data=np.load(ROOT/'board/validation_windows.npz')
            cases=[]
            for i,x in enumerate(data['inputs']):
                q=np.clip(np.rint(x/scale)+zero,-128,127).astype(np.int8)
                # Model input is [1,2,1,2800]; transpose only inserts singleton.
                payload=q.tobytes();payload+=struct.pack('<H',binascii.crc_hqx(payload,0xffff))
                s.reset_input_buffer();d=BoardDecoder();s.write(b'V\n');s.write(payload)
                response=None;start=time.monotonic()
                while response is None and time.monotonic()-start<4:
                    for frame in d.feed(s.read(4096)):
                        if frame.kind==6:response=frame
                if response is None:raise RuntimeError(f'No validation response for window {i}')
                us=struct.unpack_from('<I',response.payload)[0]
                actual=np.array(struct.unpack_from('<11f',response.payload,4))
                # Dequantized input avoids attributing host input rounding to MACs.
                reference=sess.run(None,{'input':((q.astype(np.float32)-zero)*scale).reshape(1,2,1,2800)})[0].ravel()
                delta=float(np.max(np.abs(actual-reference)))
                cases.append(dict(index=i,inference_us=us,reference=reference.tolist(),actual=actual.tolist(),max_abs_logit_delta=delta,
                                  same_top=int(actual.argmax())==int(reference.argmax())))
                print('parity',i,'max_logit_delta',delta,'top',int(actual.argmax()),int(reference.argmax()),flush=True)
            report['parity']=cases
            assert all(c['same_top'] and c['max_abs_logit_delta']<.11 for c in cases), 'Network parity failed'
            if a.raw_parity:
                from scipy.signal import sosfilt,sosfilt_zi
                contract=json.loads((source(a.model)/'input_contract.json').read_text(encoding='utf-8'))
                records=json.loads((ROOT/'models/sliding_arch_sweep_20261022_125638_517426/recordings.json').read_text(encoding='utf-8'))
                rawcases=[]
                for subject in ['S011','S012','S013','S014','S015']:
                    rec=next(r for r in records if r['subject']==subject)
                    raw=np.loadtxt(Path(rec['path'])/'samples.csv',delimiter=',',skiprows=1,max_rows=28000,usecols=(4,7),dtype=np.uint16)
                    centered=raw.astype(float)-32768
                    sos=np.array(contract['filter_sos'])
                    y,_=sosfilt(sos,centered,axis=0,zi=sosfilt_zi(sos)[:,:,None]*centered[0])
                    s.write(b'R\n');time.sleep(.1)
                    for offset in range(0,len(raw)-2799,2800):
                        payload=raw[offset:offset+2800].astype('<u2').tobytes()
                        payload+=struct.pack('<H',binascii.crc_hqx(payload,0xffff))
                        s.reset_input_buffer();d=BoardDecoder();s.write(b'F\n');s.write(payload)
                        response=None;start=time.monotonic()
                        while response is None and time.monotonic()-start<4:
                            for frame in d.feed(s.read(4096)):
                                if frame.kind==6:response=frame
                        if response is None:raise RuntimeError('Raw-filter validation timeout')
                        actual=np.array(struct.unpack_from('<11f',response.payload,4))
                        x=((y[offset:offset+2800]-np.array(contract['normalization']['mean']))/np.array(contract['normalization']['scale'])).T.astype(np.float32)
                        reference=sess.run(None,{'input':x.reshape(1,2,1,2800)})[0].ravel()
                        delta=float(np.max(np.abs(actual-reference)))
                        rawcases.append(dict(subject=subject,offset=offset,max_abs_logit_delta=delta,same_top=int(actual.argmax())==int(reference.argmax()),
                                             reference=reference.tolist(),actual=actual.tolist()))
                    print('raw parity',subject,'max delta',max(c['max_abs_logit_delta'] for c in rawcases if c['subject']==subject),flush=True)
                report['raw_parity']=rawcases
                assert all(c['same_top'] and c['max_abs_logit_delta']<.21 for c in rawcases), 'Raw pipeline parity failed'
            s.write(b'N\nI\n')
        assert infos and infos[-1]['model']==a.model and report['flags']==[0] and not report['crc_errors']
        assert not report['non_100ms_steps'] and not rows[-1]['skipped_windows'] and not rows[-1]['tx_dropped']
        assert 0<report['pipeline_us']['max']<100000, 'Pipeline cannot keep up with 100 ms stride'
        print(json.dumps({k:v for k,v in report.items() if k not in ('parity','raw_parity','last')},indent=2),flush=True)
    except Exception as exc:
        report['error']=str(exc);raise
    finally:
        s.close();(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('REPORT',out,flush=True)


if __name__=='__main__':main()
