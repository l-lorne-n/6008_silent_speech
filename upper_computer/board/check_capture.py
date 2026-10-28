"""Twelve-second actual capture/ACK compatibility check after mode flashing."""
from datetime import datetime
import json
from pathlib import Path
import struct
import sys
import time
import serial
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from semg.protocol import Decoder,Continuity,DATA,INFO,ACK
s=serial.Serial(port=None,baudrate=230400,timeout=.1);s.dtr=False;s.rts=False;s.port=sys.argv[1] if len(sys.argv)>1 else 'COM7'
d=Decoder();c=Continuity();frames=[];started=time.monotonic()
with s:
    s.write(b'I\nM,7654321\n')
    while time.monotonic()-started<12:frames.extend(d.feed(s.read(4096)))
for f in frames:
    if f.kind==DATA:c.accept(f.first,f.count)
report=dict(crc_errors=d.crc_errors,missing_samples=c.missing,frames=len(frames),
            sample_pairs=sum(f.count for f in frames if f.kind==DATA),flags=sorted({f.flags for f in frames}),
            infos=[struct.unpack('<IHHI',f.payload) for f in frames if f.kind==INFO],
            ack_ok=any(f.kind==ACK and struct.unpack_from('<I',f.payload)[0]==7654321 for f in frames))
out=ROOT/'board/logs'/('capture_check_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.json')
out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));print(out)
assert report['sample_pairs']>20000 and not report['missing_samples'] and not report['crc_errors'] and report['flags']==[0] and report['ack_ok']
