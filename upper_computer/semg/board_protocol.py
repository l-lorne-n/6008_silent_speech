"""Board-only inference packets, separate from legacy capture decoders."""
import binascii
from dataclasses import dataclass
import json
import math
import struct

CLASSES = ['open','close','next','music','louder','photo','previous','quieter','start','stop','rest']
HEADER = struct.Struct('<2sBBHIIHH')
RESULT = struct.Struct('<9I4B11f2I')


@dataclass
class Packet:
    kind: int
    seq: int
    end: int
    count: int
    flags: int
    payload: bytes


class BoardDecoder:
    def __init__(self):
        self.buffer=bytearray();self.crc_errors=0;self.discarded=0

    def feed(self,data):
        self.buffer.extend(data);out=[]
        while len(self.buffer)>=2:
            pos=self.buffer.find(b'\xa5\x5a')
            if pos<0:
                keep=int(self.buffer[-1]==0xa5);self.discarded+=len(self.buffer)-keep
                self.buffer[:]=self.buffer[-1:] if keep else b'';break
            if pos:self.discarded+=pos;del self.buffer[:pos]
            if len(self.buffer)<HEADER.size:break
            _,version,kind,n,seq,end,count,flags=HEADER.unpack_from(self.buffer)
            valid=version==2 and kind in range(1,9) and n<=236
            valid &= (kind!=1 or n==count*4 and 0<count<=59)
            valid &= (kind!=2 or n==12 and count==0)
            valid &= (kind!=3 or n==8 and count==0)
            valid &= (kind not in (4,6) or n==92 and count==0)
            valid &= (kind!=7 or n==1)
            valid &= (kind!=8 or n==8 and count==1)
            if not valid:del self.buffer[0];self.discarded+=1;continue
            size=20+n
            if len(self.buffer)<size:break
            if binascii.crc_hqx(self.buffer[:size-2],0xffff)!=struct.unpack_from('<H',self.buffer,size-2)[0]:
                self.crc_errors+=1;del self.buffer[0];continue
            out.append(Packet(kind,seq,end,count,flags,bytes(self.buffer[18:size-2])))
            del self.buffer[:size]
        return out


def result(packet):
    if packet.kind!=4:raise ValueError('Not a live board result')
    a=RESULT.unpack(packet.payload)
    keys=['uptime_ms','inference_us','pipeline_us','filter_us','skipped_windows','max_pipeline_us',
          'over_100ms','over_200ms','tx_dropped','mode','top','emitted','armed']
    r=dict(zip(keys,a[:13]));r['probabilities']=list(a[13:24]);r['clipped_samples']=a[24];r['age_samples']=a[25]
    if r['top']>=11 or r['emitted'] not in (*range(10),255):raise ValueError('Invalid class ID')
    if any(not math.isfinite(x) or x<0 or x>1 for x in r['probabilities']) or abs(sum(r['probabilities'])-1)>.001:
        raise ValueError('Invalid board probabilities')
    if r['inference_us']==0 or r['pipeline_us']<r['inference_us']:
        raise ValueError('Invalid board timing counter')
    r.update(end=packet.end,flags=packet.flags,seq=packet.seq)
    return r


def model_info(packet):
    if packet.kind!=5:raise ValueError('Not model info')
    m=json.loads(packet.payload)
    if m.get('fw')!=1 or m.get('fs')!=2000 or m.get('win')!=2800 or m.get('step')!=200:
        raise ValueError('Unsupported board contract')
    return m


def envelope(packet):
    if packet.kind!=8:raise ValueError('Not a board envelope packet')
    values=struct.unpack('<2f',packet.payload)
    if any(not math.isfinite(v) or v<0 for v in values):
        raise ValueError('Invalid board envelope')
    return values


def score_trials(trials,outputs):
    """Board sample timestamps from cue ACKs; no predictions inferred from cues."""
    rows=[]
    for t in trials:
        if not all(k in t for k in ('prepare_idx','action_idx','rest_idx','end_idx')):
            rows.append(dict(word=t['word'],valid=False,reason='missing cue ACK'));continue
        ev=[e for e in outputs if t['prepare_idx']<=e['end']<t['end_idx']]
        response=[e for e in ev if t['action_idx']<=e['end']<min(t['rest_idx']+600,t['end_idx'])]
        good=[e for e in response if e['word']==t['word']]
        rows.append(dict(word=t['word'],valid=True,correct=bool(good),strict=len(ev)==1 and len(good)==1,
                         wrong=sum(e['word']!=t['word'] for e in response),miss=not response,
                         outputs=[e['word'] for e in ev],
                         cue_to_correct_ms=(good[0]['end']-t['action_idx'])/2 if good else None))
    return rows
