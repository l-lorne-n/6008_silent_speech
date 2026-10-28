"""Inspect BT24 GATT and optionally listen to board packets without flashing."""
import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semg.board_protocol import BoardDecoder, model_info, result, envelope
from semg.board_models import model_source
import hashlib


async def inspect(args):
    from bleak import BleakClient, BleakScanner
    report = dict(devices=[], connected=False, services=[], bytes=0, packets={})
    try:
        print('Scanning BLE (8 seconds)...', flush=True)
        found = await asyncio.wait_for(BleakScanner.discover(timeout=8, return_adv=True), timeout=15)
        candidates = []
        for device, adv in found.values():
            name = adv.local_name or device.name or ''
            if (args.address and device.address.lower() == args.address.lower()) or (not args.address and 'BT24' in name.upper()):
                candidates.append(device)
                report['devices'].append(dict(name=name, address=device.address, rssi=adv.rssi))
        print(json.dumps(report['devices'], ensure_ascii=False), flush=True)
        if not args.connect:
            return report
        if len(candidates) != 1:
            raise ValueError('Expected exactly one BT24; use --address if multiple devices are found.')
        async with BleakClient(candidates[0], timeout=15) as client:
            report['connected'] = True
            for service in client.services:
                report['services'].append(dict(uuid=service.uuid, characteristics=[dict(uuid=c.uuid, properties=c.properties) for c in service.characteristics]))
            print(json.dumps(report['services'], ensure_ascii=False), flush=True)
            char = client.services.get_characteristic('0000ffe1-0000-1000-8000-00805f9b34fb')
            if char is None or 'notify' not in char.properties:
                raise ValueError('No BT24 FFE1 notify characteristic')
            decoder = BoardDecoder()
            counts = Counter()
            sample = bytearray()
            rows = []
            sequence_gaps = 0
            last_seq = None
            env_gaps = 0
            last_env = None
            envelopes = []
            def received(sender, data):
                nonlocal sequence_gaps, last_seq, env_gaps, last_env
                report['bytes'] += len(data)
                sample.extend(data[:max(0, 256-len(sample))])
                for p in decoder.feed(data):
                    if last_seq is not None and p.seq != (last_seq+1)&0xffffffff:sequence_gaps += 1
                    last_seq = p.seq
                    counts[p.kind] += 1
                    if p.kind == 5:
                        report['model'] = model_info(p)
                    elif p.kind == 4:rows.append(result(p))
                    elif p.kind == 8:
                        envelopes.append(envelope(p))
                        if last_env is not None and p.end-last_env != 20:env_gaps += 1
                        last_env = p.end
            await client.start_notify(char, received)
            # Passive by default. --request-info sends only the existing INFO query.
            if args.request_info:
                await client.write_gatt_char(char, b'I\n', response='write' in char.properties)
            await asyncio.sleep(args.seconds)
            report.update(packets=dict(counts), crc_errors=decoder.crc_errors, discarded=decoder.discarded,
                          sample_hex=sample.hex(), sample_text=sample.decode('utf-8', errors='replace'),
                          sequence_gaps=sequence_gaps, envelope_gaps=env_gaps)
            if rows:
                report.update(results=len(rows), non_100ms_steps=sum(b['end']-a['end']!=200 for a,b in zip(rows,rows[1:])),
                              flags=sorted({r['flags'] for r in rows}), last=rows[-1],
                              inference_ms=dict(min=min(r['inference_us'] for r in rows)/1000,max=max(r['inference_us'] for r in rows)/1000),
                              pipeline_ms=dict(min=min(r['pipeline_us'] for r in rows)/1000,max=max(r['pipeline_us'] for r in rows)/1000))
            if args.model:
                info = report.get('model', {})
                digest = hashlib.sha256((model_source(args.model)/'model_int8_qdq.onnx').read_bytes()).hexdigest()[:16]
                report['identity_verified'] = info.get('model') == args.model and info.get('sha') == digest and info.get('mode') == 1
                if not report['identity_verified'] or not rows:
                    raise ValueError('No verified model identity/live results')
                if decoder.crc_errors or sequence_gaps or env_gaps or report['non_100ms_steps'] or report['flags'] != [0] or rows[-1]['tx_dropped'] or rows[-1]['skipped_windows']:
                    raise ValueError('Live stream health check failed; inspect report')
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if args.output:
            Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--address')
    parser.add_argument('--connect', action='store_true')
    parser.add_argument('--request-info', action='store_true')
    parser.add_argument('--seconds', type=float, default=5)
    parser.add_argument('--output')
    parser.add_argument('--model', help='Require live binary results and this frozen model identity')
    options = parser.parse_args()
    outcome = asyncio.run(inspect(options))
    sys.exit(1 if outcome.get('error') else 0)
