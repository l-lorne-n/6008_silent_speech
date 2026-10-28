import struct
import pytest
from semg.protocol import encode
from semg.board_protocol import BoardDecoder, RESULT, result, score_trials, envelope


def frame(end=2800):
    p=RESULT.pack(1400,61000,61500,450,0,61500,0,0,0,1,10,255,1,*([0.0]*10+[1.0]),0,120)
    return encode(4,7,end,p,version=2)


def test_fragmented_board_result_and_crc_recovery():
    d=BoardDecoder();data=frame();out=[]
    damaged=bytearray(data);damaged[55]^=1
    for byte in b'noise'+damaged+data:out+=d.feed(bytes([byte]))
    assert len(out)==1 and d.crc_errors==1
    r=result(out[0]);assert r['inference_us']==61000 and r['top']==10 and r['emitted']==255


def test_board_parser_keeps_capture_and_marker_packets():
    d=BoardDecoder()
    packets=d.feed(encode(1,1,0,bytes(80),count=20,version=2)+encode(3,2,20,struct.pack('<II',123,10),version=2))
    assert [p.kind for p in packets]==[1,3]
    assert packets[0].count==20


def test_bad_result_length_does_not_desynchronize():
    d=BoardDecoder()
    assert len(d.feed(encode(4,1,0,b'x',version=2)+frame()))==1


def test_locked_timer_cannot_be_reported_as_zero_latency():
    packet=BoardDecoder().feed(frame())[0]
    p=bytearray(packet.payload);struct.pack_into('<I',p,4,0);packet.payload=bytes(p)
    with pytest.raises(ValueError,match='timing'):result(packet)


def test_nan_probability_is_rejected():
    packet=BoardDecoder().feed(frame())[0]
    p=bytearray(packet.payload);struct.pack_into('<f',p,40,float('nan'));packet.payload=bytes(p)
    with pytest.raises(ValueError,match='probabilities'):result(packet)


def test_envelope_and_inference_interleave_without_losing_frames():
    data=encode(8,6,2780,struct.pack('<2f',123.5,640.25),count=1,version=2)+frame()
    d=BoardDecoder();packets=[]
    for i in range(0,len(data),7):packets.extend(d.feed(data[i:i+7]))
    assert [p.kind for p in packets]==[8,4] and d.crc_errors==0
    assert envelope(packets[0])==(123.5,640.25)  # >500 remains data, never silently clamped


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1])
def test_invalid_envelope_is_rejected(value):
    packet=BoardDecoder().feed(encode(8,1,20,struct.pack('<2f',value,20),count=1,version=2))[0]
    with pytest.raises(ValueError,match='envelope'):envelope(packet)


def test_scoring_wrong_miss_strict_and_boundary():
    trial=dict(word='open',prepare_idx=2000,action_idx=5000,rest_idx=9000,end_idx=13000)
    events=[dict(end=5200,word='close'),dict(end=5400,word='open'),dict(end=9600,word='open')]
    row=score_trials([trial],events)[0]
    assert row['correct'] and not row['strict'] and row['wrong']==1 and not row['miss']
    assert row['cue_to_correct_ms']==200
    assert score_trials([trial],[dict(end=9600,word='open')])[0]['miss']


def test_missing_ack_invalidates_trial_instead_of_counting_as_miss():
    assert not score_trials([dict(word='open')],[])[0]['valid']


def test_cue_does_not_turn_wrong_output_into_success():
    trial=dict(word='open',prepare_idx=0,action_idx=3000,rest_idx=7000,end_idx=11000)
    row=score_trials([trial],[dict(end=4000,word='close')])[0]
    assert not row['correct'] and row['wrong']==1 and not row['miss']
