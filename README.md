# 6008 Silent Speech — App、部署模型、CNN研究与训练数据

独立交付分支 **`integration-app-data-20261029`**，承接 2026-10-28 的 `integration-bt24-models-20261028`（`95cce21`）。包含当前上位机、媒体 App、训练代码、两版 CNN、四份预构建 HEX，以及指定的完整训练录制。保留原 `firmware/` 和 `tools/bt24_receiver/`；本次没有更新 `main`。

本分支包括最新桌面上位机、照片/音乐App、训练代码、两版CNN模型，以及两模型×两UART速率的四份预构建HEX。**最新模型已烧录及短时运行验证；BLE长时间稳定性、UART故障恢复和真人十词控制仍待验收。**

| 入口 | 用途 |
| --- | --- |
| [完整交接](upper_computer/history/2026-10-28_完整交接_BT24双波特率_双模型与GitHub发布.md) | 当前状态、方案、改动、产物、证据、限制与后续工作 |
| [上位机](upper_computer/README.md) | 采集/LDA与板端实时监测 |
| [媒体 App](app/使用说明.txt) | 本地网页照片/音乐控制，Python后端桥接 |
| [当前训练入口](upper_computer/experiments/run_cnn_goodone_plus_current.py) | 新数据固定结构重训；本分支已附对应原始数据 |
| [模型资产](upper_computer/models/README.md) | `mid64_temporal3`、`mid64_temporal3_20261027`及哈希 |
| [固件归档](upper_computer/board/prebuilt/README.md) | 115200/230400 HEX及原构建证据 |
| [发布检查](upper_computer/history/publish_20261028/validation.json) | 软件测试结果与范围 |

## 2026-10-30 CNN 跨人与单人模型

新增当前 `mid64_temporal3` 架构的两组训练代码、**10 个最终研究模型**（每个均有 PyTorch、FP32 ONNX、INT8 ONNX）、精确划分、训练历史、逐试次结果与连续概率轨迹。原有两版正式部署模型、固件及 BT24 接入保持不变；这十个新权重没有烧录或替换 App 模型。

| 实验 | FP32 十词分类 | FP32 连续正确命令率 | FP32 严格成功率 |
| --- | ---: | ---: | ---: |
| 四人训练一人测试，五折跨人，无个人校准 | 1006/2496，40.30% | 37.94% | 37.06% |
| 五个独立个人模型，各自整轮 80/20 | 464/500，92.80% | 80.80% | 78.80% |

十词分类利用动作提示边界汇总概率，不等于实时触发成功率。两组训练人员、训练量和测试集合不同；个人同录制整轮留出仍可能存在同次佩戴相关性，不能将分数差异当作纯架构改进或新佩戴保证。

- [十个模型、结果与复现说明](upper_computer/model_releases/cnn_20261030/README.md)
- [跨人完整训练入口](upper_computer/experiments/run_cnn_current_loso.py) / [Windows 启动](upper_computer/run_cnn_cross_subject.cmd)
- [个人训练入口](upper_computer/experiments/run_cnn_individual_80_20.py) / [Windows 启动](upper_computer/run_cnn_individual_80_20.cmd)
- [更新后的中文技术报告](reports/20261030_S3_S5_CNN/EE6008_S3-S5_完整技术报告_中文审阅版_CNN跨人与单人补充版.docx) / [Markdown 阅读版](reports/20261030_S3_S5_CNN/report.md)
- [发布校验记录](upper_computer/model_releases/cnn_20261030/verification.json) / [完成交接](upper_computer/history/2026-10-30_CNN跨人与单人模型_GitHub及报告交接.md)

从仓库根目录运行 `python upper_computer/model_releases/cnn_20261030/verify.py --data` 可校验发布文件及 196 个核心原始文件；安装训练依赖后加 `--models` 可检查模型数值、训练分区并复算保存的测试结果。新实验输出写入本机 `models/`，发布的最终资产独立放在 `model_releases/`，不把精简发布包作为历史训练缓存续跑。

## 2026-10-29 更新

- 新增两种离线 LDA 标签校准方案、固定划分和可复现入口，见下方“LDA 校准实验”。
- App 顶部静息状态框：黄色等待，绿色可开始下一个命令；照片全屏同步可见。
- App 场景过滤开关：照片页过滤 start/stop/quieter/louder，音乐页过滤 open/close；默认开启并记住选择，全屏也可切换。
- 发布 `upper_computer/recordings/goodone_plus` 全部 49 组，以及 `upper_computer/recordings/S013/session_001` 全部 10 组，共 354 个原始文件，约 2.41 GiB。
- **旧模型的 S013 使用 session_001；新模型使用当前 goodone_plus/s013（四组替换后）。** 两版各 196 个训练输入核心文件均与历史哈希匹配。详细列表与使用方法见 [数据与两模型对应说明](upper_computer/recordings/README.md)。
- 新功能与数据交付见 [本次交接](upper_computer/history/2026-10-29_GitHub新分支_App双模型与原始数据交接.md)，软件验证见 [发布检查](upper_computer/history/publish_20261029/validation.json)。

## 启动

```powershell
git clone --branch integration-app-data-20261029 https://github.com/l-lorne-n/6008_silent_speech.git
cd 6008_silent_speech\upper_computer
.\setup.cmd
.\start_monitor.cmd
# 或启动媒体 App：
..\app\start.cmd
```

已验证环境：Windows 10/11、64位Python 3.12。当前板子选 **BT24/BLE、115200、新版模型**；App通过板端身份自动选模型。蓝牙不用Windows配对。请勿让多个接收器同时占用BT24。没有型号匹配、无故障数据和板端确认命令时，连接成功也不会驱动媒体。

本分支包含上述两处指定原始录制；不含其他录制、Python 环境、个人照片/音乐、训练缓存或烧录前整片备份。自行放置媒体到 `app/photo`、`app/music`；没有本地媒体时清单为空。原始采集仍用 `upper_computer/start.cmd`、230400与对应采集固件。

从仓库根目录执行 `python upper_computer/board/verify_assets.py`，校验两版模型与四份HEX的203个相关文件。历史训练/构建清单保留原绝对路径；发布清单使用仓库相对路径。`.gitattributes`保留原文件字节，避免换行转换破坏哈希。

数据可用 `python upper_computer/recordings/verify_recordings.py` 校验全部原始文件和两版历史训练输入。旧交接中“不上传原始录制”的描述仅适用于 2026-10-28 分支。

## 原 main 固件说明（历史）

下文是本分支基底的原README，描述队友原固件链路；本次模型固件与BLE接入以顶部交接为准。

# 6008 Silent Speech

<!-- TODO: 一句话说明项目目标，例如：采集 XXX 信号，识别无声语音 / subvocal speech -->

> **当前状态**：固件侧的采集链路已可运行——1 kHz 采样、20–400 Hz 带通、包络提取，并通过串口以 100 行/秒输出 CSV。识别与上位机部分待补充。

## 仓库结构

| 路径 | 说明 |
| --- | --- |
| `firmware/` | STM32H743VGT6 采集固件，STM32CubeMX 生成 + HAL |
| `firmware/Test.ioc` | CubeMX 工程配置。**改引脚或外设一律改这里，再重新生成代码** |

<!-- TODO: 之后加入的 Python / 模型 / 数据集 / 文档目录补到这里 -->

## BT24 电脑接收工具

已加入 [BT24 蓝牙接收检查工具](tools/bt24_receiver/README.md)。队友在支持 BLE 的 Windows 10/11 电脑上安装 Python 3.12，下载并解压本仓库后，双击 `tools/bt24_receiver/start.cmd` 即可使用；首次启动自动安装依赖，不需要 Codex。

工具通过 BLE 直接扫描和连接 BT24，显示双通道固件输出的六列 CSV 数据及接收计数。使用前关闭手机蓝牙，无需在 Windows 设置中配对。此工具用于链路检查；正式上位机的波形、录制和事件标记不包含在内。

## 硬件

| 项目 | 配置 |
| --- | --- |
| MCU | STM32H743VGT6（Cortex-M7，LQFP100） |
| 主频 | SYSCLK 480 MHz / HCLK 240 MHz / APB 120 MHz |
| 外部晶振 | HSE 25 MHz（PH0-OSC_IN、PH1-OSC_OUT） |
| ADC 时钟 | 30 MHz（PLL2P） |
| 信号输入 | PA7 → `ADC1_INP7`，16 bit，16× 过采样 |
| 串口 | PB14 = `USART1_TX`，PB15 = `USART1_RX`，115200 8N1 |
| 其他引脚 | PC13 = 数字输出（上电置低），PA1 = 数字输入 |
| 调试接口 | SWD（PA13 = SWDIO，PA14 = SWCLK） |

<!-- TODO: 补上传感器型号、供电方式、原理图链接、实际接线图 -->

## 固件采集链路

```
TIM3 TRGO @ 1 kHz
      │
      ▼
ADC1 (PA7, 16 bit, 16× 过采样，硬件触发)
      │
      ▼
DMA1_Stream0 循环搬运 → adc_buffer[1000] @ .RAM_D2 (0x30000000)
      │
      ▼
半满 / 全满中断置标志 → 主循环一次处理 500 点
      │
      ▼
8 阶巴特沃斯带通 20–400 Hz（fs = 1 kHz）
      │
      ▼
整流 + 200 点滑动平均包络（1 kHz ⇒ 200 ms 窗口）
      │
      ▼
每 10 点 printf 一行 → USART1 (DMA1_Stream1)
```

关键参数都在 [firmware/Core/Src/main.c](firmware/Core/Src/main.c) 里以宏定义形式给出：`ADC_BUF_LEN`、`ENV_WIN`、`PRINT_EVERY`、`VERIFY_MODE`。

### 采样率怎么来的

TIM3 时钟 240 MHz，`Prescaler = 239`、`Period = 999`，得到 `240 MHz / 240 / 1000 = 1 kHz`，这个 1 kHz 同时是 ADC 的硬件触发源和后续所有频率计算的基准（包络窗口、带通系数的 `fs`）。**改动 TIM3 参数必须同步重算带通系数。**

## 串口输出协议

ASCII CSV，每行以 `\r\n` 结尾，输出速率 100 行/秒（1 kHz 每 10 点一行）。字段由 `VERIFY_MODE` 决定：

| `VERIFY_MODE` | 每行格式 | 用途 |
| --- | --- | --- |
| `0` | `raw` | 只看原始 ADC 值 |
| `1` | `raw,envelope` | 原始值 + 包络 |
| `2`（当前） | `raw,filtered,envelope` | 全链路验证 |

ADC 为 16 bit，代码里以 `32768`（即 1.65 V 中点）为零点做去偏置，所以 `raw` 是无符号值，`filtered` / `envelope` 是相对中点、取整后的有符号值。

<!-- TODO: 如果上位机按固定列名解析，把解析用的 Python 脚本路径和列定义补进来 -->

## 构建与烧录

需要 STM32CubeCLT，或用 STM32Cube 的 VS Code 扩展自带工具链（`arm-none-eabi-gcc` + Ninja）。

```bash
cd firmware
cmake --preset Debug
cmake --build --preset Debug
```

用 Keil MDK-ARM 的话，直接打开 `firmware/MDK-ARM/Test.uvprojx`。

<!-- TODO: 补上烧录方式（STM32CubeProgrammer / ST-Link / openocd 命令） -->

## 开发约定

- 外设、引脚、时钟的改动走 `firmware/Test.ioc`，用 CubeMX 重新生成，不要手改生成区域的代码。
- 自己的代码写在 `/* USER CODE BEGIN xxx */` 和 `/* USER CODE END xxx */` 之间，否则重新生成时会被覆盖。
- 编译产物（`build/`、`MDK-ARM/Test/`、Keil 的 `.uvguix.*` 等）已在 `.gitignore` 里排除，不要提交。
- `firmware/Drivers/CMSIS/DSP`、`NN`、`RTOS*` 目前被 `.gitignore` 排除，因为这个工程没有引用它们。**如果之后要启用 DSP 库或 FreeRTOS，先删掉 `.gitignore` 里对应的行。**

## 已知问题 / 待办

- [ ] 源文件目前是 **GBK 编码**，GitHub 网页和 macOS / Linux 上会显示成乱码，建议统一转成 UTF-8。
- [ ] 补上项目简介、传感器型号与实物接线。
- [ ] 上位机接收与解析脚本。
- [ ] 识别算法部分。

<!-- TODO: 补上团队成员分工与联系方式 -->

## LDA 校准实验

本次补充的两种方法均在电脑上离线运行，不需要 STM32。五人分别作为目标人，其他四人数据参与训练；重新拟合全局 StandardScaler 和 LDA(lsqr, shrinkage=auto)。没有修改 App、BT24 或已发布的两版 CNN。

| 方法 | 每人真实校准量 | 校准复制次数 | 总正确数/测试数 | 准确率 |
| --- | ---: | ---: | ---: | ---: |
| 固定50条×40＋其他四人 | 50条，每词5条 | 40 | 915/2246 | 40.74% |
| K=3按录制取样后按人汇总×10＋其他四人 | 270或300条 | 10 | 686/1026 | 66.86% |

固定50方案保持之前的校准集和450/446条测试集不变；K=3在每份录制每个词取最早3条有效动作，汇总目标人的校准，再测试剩余动作。**两种方法的校准预算与测试集不同，不能直接比较分数高低。** 后者还共享录制，属于回顾性适应。重复样本不增加独立动作数量；两者都没有额外启用按人归一化，也不是CNN微调。

安装依赖后，在仓库根目录运行：

```powershell
# 默认运行两种方法；检查源文件哈希、精确划分及所有历史预测
.\upper_computer\.runtime\Scripts\python.exe -X utf8 .\upper_computer\experiments\run_lda_calibration_comparison.py
# 可选：先从49份原始录制重新提取并核对特征，再运行
.\upper_computer\.runtime\Scripts\python.exe -X utf8 .\upper_computer\experiments\run_lda_calibration_comparison.py --reextract
# 单独运行时追加 --method fixed50_x40 或 --method pooled_k3_x10
```

每次输出到新的 `upper_computer/models/lda_calibration_release_...`；成功标记为 `completed.json` 的 `passed: true`。发布前已从该发布目录中的原始录制重新提取2496×48特征、训练10个模型，两种方法的每一条测试预测均与原结果一致。环境为 Python 3.12、NumPy 2.5.3、SciPy 1.18.1、scikit-learn 1.9.1。

- [可移植运行代码](upper_computer/experiments/run_lda_calibration_comparison.py)
- [冻结划分 特征与参考结果](upper_computer/experiments/calibration_data_20261029/README.md)
- [本轮完整交接和解释](upper_computer/history/2026-10-29_LDA校准实验汇总与两方案报告交接.md)
- [发布复现核验](upper_computer/history/calibration_publish_20261029/validation.json)

当前数据零校准五折LOSO为446/2496（17.87%）。上述40.74%和66.86%使用了目标人标签，均为离线提示动作分类，不能作为未见人零校准、独立新佩戴或实时App控制成功率。重复倍数在查看前序结果后继续探索，尚未进行独立参数验证。
