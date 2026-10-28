"""Create an isolated Keil project from the inspected original. Never flash."""
from pathlib import Path
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "code/test_lvbo/Test/Test"
DEST = HERE / "Test"


def main():
    if DEST.exists():
        raise SystemExit(f"Refusing to overwrite existing project: {DEST}")
    DEST.mkdir(parents=True)
    for path in ("Core", "Drivers/STM32H7xx_HAL_Driver", "Drivers/CMSIS/Include", "Drivers/CMSIS/Device/ST"):
        shutil.copytree(SOURCE / path, DEST / path)
    (DEST / "MDK-ARM").mkdir()
    for name in ("Test.uvprojx", "Test.uvoptx", "startup_stm32h743xx.s"):
        shutil.copy2(SOURCE / "MDK-ARM" / name, DEST / "MDK-ARM" / name)
    shutil.copy2(SOURCE / "Test.ioc", DEST / "Test.ioc")
    original = (SOURCE / "Core/Src/main.c").read_bytes()
    # Existing source has GBK comments; preserve them without lossy decoding.
    content = original.decode("gb18030")
    def replace_user(tag, body):
        nonlocal content
        start = f"/* USER CODE BEGIN {tag} */"
        end = f"/* USER CODE END {tag} */"
        a, b = content.index(start) + len(start), content.index(end)
        content = content[:a] + "\n" + body + "\n" + content[b:]
    replace_user("PV", '#include "stream_v1.inc"')
    replace_user("PFP", "")
    # Original contains a duplicated USER CODE BEGIN 0; replace the whole region.
    a = content.index("/* USER CODE BEGIN 0 */")
    b = content.index("/* USER CODE END 0 */", a) + len("/* USER CODE END 0 */")
    content = content[:a] + content[b:]
    replace_user("2", "  stream_start();")
    a = content.index("  while (1)")
    b = content.index("    /* USER CODE END WHILE */", a)
    content = content[:a] + "  while (1)\n  {\n    stream_poll();\n" + content[b:]
    replace_user("4", "")
    (DEST / "Core/Src/main.c").write_bytes(content.encode("gb18030"))
    shutil.copy2(HERE / "stream_v1.inc", DEST / "Core/Src/stream_v1.inc")
    scatter = '''LR_IROM1 0x08000000 0x00080000 {
  ER_IROM1 0x08000000 0x00080000 {
    *.o (RESET, +First)
    *(InRoot$$Sections)
    .ANY (+RO)
    .ANY (+XO)
  }
  RW_D2 0x30000000 0x00020000 {
    * (.RAM_D2)
  }
  RW_IRAM1 0x20000000 0x00020000 {
    .ANY (+RW +ZI)
  }
}
'''
    (DEST / "MDK-ARM/stream.sct").write_text(scatter, encoding="ascii")
    project = DEST / "MDK-ARM/Test.uvprojx"
    tree = ET.parse(project)
    defines = tree.find(".//Cads/VariousControls/Define")
    defines.text += ",DATA_IN_D2_SRAM"
    ld = tree.find(".//LDads")
    ld.find("umfTarg").text = "0"
    ld.find("useFile").text = "1"
    ld.find("ScatterFile").text = "stream.sct"
    tree.write(project, encoding="UTF-8", xml_declaration=True)
    manifest = {"source": str(SOURCE), "source_main_sha256": hashlib.sha256(original).hexdigest(),
                "protocol_version": 1, "auto_flash": False,
                "note": "Test.ioc is reference only; CubeMX regeneration would replace custom acquisition logic."}
    (HERE / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(project)


if __name__ == "__main__":
    main()
