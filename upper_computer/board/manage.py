"""Build/flash locally. Capture uses the archived, validated v0.6.0 firmware."""
from pathlib import Path
import argparse
from datetime import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from semg.serial_settings import BAUDRATES, DEFAULT_BAUDRATE, validate_baudrate
from semg.board_models import model_source


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_dir(name, baudrate=DEFAULT_BAUDRATE):
    model_source(name)  # Reject unknown IDs before using them as path components.
    validate_baudrate(baudrate)
    return HERE/'build'/name/f'baud_{baudrate}'


def uart_source(main, baudrate):
    validate_baudrate(baudrate)
    main, count = re.subn(r'(huart1\.Init\.BaudRate\s*=\s*)\d+(\s*;)',
                          lambda m: m[1] + str(baudrate) + m[2], main)
    if count != 1:
        raise ValueError('Expected exactly one USART1 baud-rate assignment')
    return main


def built_image(name, baudrate=DEFAULT_BAUDRATE):
    folder = build_dir(name, baudrate)
    if not (folder/'build_manifest.json').exists() and baudrate == DEFAULT_BAUDRATE:
        folder = HERE/'build'/name  # Existing 230400 artifacts remain usable.
    m = json.loads((folder/'build_manifest.json').read_text(encoding='utf-8'))
    if m['model'] != name or m.get('baudrate', DEFAULT_BAUDRATE) != baudrate:
        raise RuntimeError('Built firmware model/baud rate does not match the request')
    image = folder/'Objects/realtime.hex'
    if image.resolve() != Path(m['hex']).resolve() or sha(image) != m['hex_sha256']:
        raise RuntimeError('Built image path/checksum changed')
    return image


def build(name, baudrate=DEFAULT_BAUDRATE):
    from export_model import export
    dst = build_dir(name, baudrate)
    model = export(name, dst)
    original = ROOT/'firmware/Test'
    project = original/'MDK-ARM/Test.uvprojx'
    main = (original/'Core/Src/main.c').read_text(encoding='utf-8-sig')
    main = uart_source(main, baudrate)
    main = main.replace('#include "stream_v2.inc"', '#include "realtime.inc"')
    main = main.replace('  MPU_Config();', '  MPU_Config();\n  SCB_EnableICache();\n  SCB_EnableDCache();')
    main = main.replace('    stream_poll();', '    stream_poll();\n    realtime_poll();')
    anchor = '  /* Enables the MPU */'
    assert anchor in main
    main = main.replace(anchor, '''  /* DMA1 SRAM: normal, non-cacheable, shareable (TEX=1 C=0 B=0). */
  MPU_InitStruct.Number = MPU_REGION_NUMBER1;
  MPU_InitStruct.BaseAddress = 0x30000000;
  MPU_InitStruct.Size = MPU_REGION_SIZE_128KB;
  MPU_InitStruct.SubRegionDisable = 0;
  MPU_InitStruct.TypeExtField = MPU_TEX_LEVEL1;
  MPU_InitStruct.AccessPermission = MPU_REGION_FULL_ACCESS;
  MPU_InitStruct.DisableExec = MPU_INSTRUCTION_ACCESS_DISABLE;
  MPU_InitStruct.IsShareable = MPU_ACCESS_SHAREABLE;
  MPU_InitStruct.IsCacheable = MPU_ACCESS_NOT_CACHEABLE;
  MPU_InitStruct.IsBufferable = MPU_ACCESS_NOT_BUFFERABLE;
  HAL_MPU_ConfigRegion(&MPU_InitStruct);
''' + anchor)
    (dst/'main.c').write_text(main,encoding='utf-8')
    startup = (original/'MDK-ARM/startup_stm32h743xx.s').read_text()
    startup = re.sub(r'(Stack_Size\s+EQU\s+)0x400\b',r'\g<1>0x2000',startup)
    (dst/'startup_stm32h743xx.s').write_text(startup)
    (dst/'board.sct').write_text('''LR_IROM1 0x08000000 0x00080000 {
 ER_IROM1 0x08000000 0x00080000 {
  *.o (RESET, +First)
  *(InRoot$$Sections)
  .ANY (+RO)
  .ANY (+XO)
 }
 RW_D2 0x30000000 0x00020000 { * (.RAM_D2) }
 RW_AXI 0x24000000 0x00080000 { * (.AXI) }
 RW_DTCM 0x20000000 0x00020000 { .ANY (+RW +ZI) }
}
''')
    tree = ET.parse(project)
    for element in tree.findall('.//FilePath'):
        p = (project.parent/element.text).resolve()
        if p.name == 'main.c': p = dst/'main.c'
        elif p.name == 'startup_stm32h743xx.s': p = dst/'startup_stm32h743xx.s'
        element.text = str(p)
    for element in tree.findall('.//Cads/VariousControls/IncludePath'):
        paths = [str((project.parent/p).resolve()) for p in element.text.split(';')]
        element.text = ';'.join([str(dst),str(HERE)]+paths)
    replacements = {'.//OutputDirectory':'Objects\\', './/OutputName':'realtime',
                    './/ListingPath':'Objects\\', './/ScatterFile':str(dst/'board.sct'),
                    './/Cads/Optim':'4', './/Cads/VariousControls/MiscControls':'-ffp-contract=off',
                    './/AfterMake/RunUserProg2':'0'}
    for query,value in replacements.items(): tree.find(query).text=value
    (dst/'Objects').mkdir(exist_ok=True)
    target = dst/'realtime.uvprojx';tree.write(target,encoding='utf-8',xml_declaration=True)
    uv = Path(os.environ.get('SEMG_UV4',r'D:\keil\keil_v5\UV4\UV4.exe'))
    if not uv.is_file(): raise RuntimeError('Set SEMG_UV4 to the Keil UV4.exe path')
    log = dst/'build.log'
    if log.exists(): log.unlink()
    result = subprocess.run([str(uv),'-r',str(target),'-j0','-o',str(log)],timeout=180)
    text = log.read_text(encoding='utf-8-sig',errors='replace') if log.exists() else ''
    print(text,flush=True)
    image = dst/'Objects/realtime.hex'
    if result.returncode not in (0,1) or '0 Error(s)' not in text or not image.is_file():
        raise RuntimeError(f'Build failed, exit={result.returncode}: {log}')
    files = [HERE/'manage.py',HERE/'export_model.py',ROOT/'semg/board_models.py',ROOT/'semg/serial_settings.py',HERE/'nn_runtime.h',HERE/'realtime.inc',
             dst/'model_data.h',dst/'main.c',dst/'board.sct',target,image]
    manifest = dict(model=name,baudrate=baudrate,onnx_sha256=model['onnx_sha256'],hex=str(image),hex_sha256=sha(image),
                    created=datetime.now().astimezone().isoformat(),sources={str(p):sha(p) for p in files})
    (dst/'build_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'BUILT {image}',flush=True)
    return image


def flash(image, label):
    sys.path.insert(0,str(ROOT/'.probe-deps'))
    from pyocd.core.session import Session
    from pyocd.core.helpers import ConnectHelper
    from pyocd.flash.file_programmer import FileProgrammer
    from intelhex import IntelHex
    Session.get_current().options['cmsis_dap.prefer_v1']=True
    probes=ConnectHelper.get_all_connected_probes(blocking=False)
    if len(probes)!=1: raise RuntimeError(f'Expected one CMSIS-DAP probe; found {len(probes)}')
    expected=IntelHex(str(image))
    addresses=expected.addresses()
    if not addresses or min(addresses)!=0x08000000 or max(addresses)>=0x08080000:
        raise ValueError('Firmware must fit the actual H743VG first 512 KiB bank')
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    out=HERE/'logs'/f'{stamp}_{label}';out.mkdir(parents=True)
    opts=dict(target_override='stm32h743xx',connect_mode='attach',auto_unlock=False,
              frequency=100000,dap_protocol='swd',no_config=True,resume_on_disconnect=False,
              **{'cmsis_dap.prefer_v1':True,'cmsis_dap.deferred_transfers':False,'cmsis_dap.limit_packets':True})
    report=dict(image=str(image),sha256=sha(image),label=label,probe=probes[0].unique_id)
    session=ConnectHelper.session_with_chosen_probe(unique_id=probes[0].unique_id,options=opts)
    try:
        with session:
            target=session.target
            target.halt()
            size=target.read16(0x1ff1e880);device=target.read32(0x5c001000)&0xfff
            if size!=1024 or device!=0x450: raise RuntimeError(f'Unexpected MCU: size={size}, ID={device:x}')
            print('Backing up first 512 KiB flash bank before programming...',flush=True)
            backup=bytearray()
            for offset in range(0,512*1024,4096):backup.extend(target.read_memory_block8(0x08000000+offset,4096))
            (out/'bank1_before.bin').write_bytes(backup)
            report['backup_sha256']=hashlib.sha256(backup).hexdigest()
            # Sector erase only; never mass erase or alter option bytes / bank 2.
            FileProgrammer(session,chip_erase='sector',smart_flash=True).program(str(image))
            target.halt()
            mismatch=0
            for start,end in expected.segments():
                for addr in range(start,end,4096):
                    n=min(4096,end-addr)
                    actual=bytes(target.read_memory_block8(addr,n))
                    desired=bytes(expected.tobinarray(start=addr,size=n))
                    mismatch+=sum(a!=b for a,b in zip(actual,desired))
            report['verified_bytes']=len(addresses);report['mismatches']=mismatch
            if mismatch: raise RuntimeError(f'Flash verification failed: {mismatch} bytes')
            target.reset()
            report['state_after']=str(target.get_state())
            print(f'FLASH VERIFIED: {len(addresses)} bytes. Reset complete.',flush=True)
    except Exception as exc:
        report['error']=str(exc)
        raise
    finally:
        (out/'flash.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(f'Flash log: {out}',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['build','flash','build-flash'])
    p.add_argument('--model',default='mid64_temporal3')
    p.add_argument('--capture',action='store_true')
    p.add_argument('--baud',type=int,choices=BAUDRATES,default=DEFAULT_BAUDRATE,
                   help='USART1 baud rate; does not change ADC sampling or the model')
    a=p.parse_args()
    if a.capture:
        if a.baud != DEFAULT_BAUDRATE:
            p.error('Archived 2 kHz capture firmware uses 230400; 115200 is available for monitoring builds.')
        image=ROOT/'releases/v0.6.0-2khz/semg_dual_2khz_v0.6.0.hex'
        if sha(image)!='3120757f7864d9282ba847ffbe22ea1f1d46f3c989a078555c04f98575847290':
            raise RuntimeError('Archived capture image hash changed')
    elif a.action in ('build','build-flash'): image=build(a.model,a.baud)
    else:
        image=built_image(a.model,a.baud)
    if a.action in ('flash','build-flash'):flash(image,'capture' if a.capture else f'{a.model}_baud_{a.baud}')


if __name__=='__main__':main()
