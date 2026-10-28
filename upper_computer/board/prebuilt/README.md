# 预构建固件归档

两个模型各有 `baud_115200/realtime.hex` 与 `baud_230400/realtime.hex`，共四份。当前实机验证使用 `mid64_temporal3_20261027/baud_115200/realtime.hex`。配套 `model_data.h`、`model.json`、生成的 `main.c`、构建日志与原 manifest 用于追溯；两档波特率不改变对应模型参数或2 kHz采样。

执行 `python upper_computer/board/verify_assets.py` 核验文件。原manifest含作者本机绝对路径，因此此目录是可加载HEX的存档，不是 `manage.py flash` 所需的本机 `board/build` 目录。使用 Keil/其他匹配的SWD工具加载已核验的HEX，或安装本地工具链后执行 `board/manage.py build --model <ID> --baud <速率>` 再按需烧录。勿将文件改名混用，不要通过BLE尝试现有SWD烧录流程。

当前UART故障锁存和BLE长测问题仍待修复，这些固件没有包含未完成的修复。正常运行不需要再次烧录。
