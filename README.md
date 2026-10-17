# 6008 Silent Speech

本分支 `upper_computer_v0.6.0` 提供双通道 sEMG 无声口型采集上位机：每路 2 kHz、20–500 Hz 显示滤波、完整原始数据保存、离线 LDA 及独立 CNN/LSTM 实验脚本。

## 安装与启动

Windows 10/11，64位 Python 3.12。克隆后进入应用目录：

```powershell
git clone --branch upper_computer_v0.6.0 https://github.com/l-lorne-n/6008_silent_speech.git
cd 6008_silent_speech\upper_computer
.\setup.cmd
.\start.cmd
```

首次安装需网络。可先用模拟演练检查界面；真实设备使用本机实际串口，波特率 **230400**。详细说明见 [上位机 README](upper_computer/README.md)。运行采集与LDA无需PyTorch；神经网络实验另装可选依赖。

## 三人原始数据与固件

[下载 v0.6.0 发布附件](https://github.com/l-lorne-n/6008_silent_speech/releases/tag/v0.6.0-2khz)：

- S011：10份录制，500次完整动作。
- S012：9份录制，500次完整动作；中止录制不包含。
- S015：10份录制，500次完整动作。

三个ZIP均保留原始采集通道顺序、完整六类录制文件；实验交换已撤回。通道的解剖位置未独立确认。解压到 `upper_computer/` 得到 `recordings/S011` 等路径，再在LDA扫描；不要覆盖本机已有同名录制。`recordings_manifest.json` 和 `SHA256SUMS.txt` 用于核验。数据作为Release附件提供，不放入Git历史。

配套固件为 [upper_computer/firmware/Test](upper_computer/firmware/Test)，每路2 kHz、双通道原始二进制v2、230400/8N1；发布附件含已烧录校验的HEX。数字滤波在上位机，传感器模拟带宽未由软件改变。旧1 kHz连接选择115200，按原20–400Hz处理。新2 kHz自动活动切段/在线识别尚不支持。

2026-10-17软件验证：92项pytest通过。实机采集链路2026-10-16检查约12秒、24140对样本，CRC/缺样/设备fault均0；这不代表识别准确率或模拟带宽已达标。

## 历史根目录固件参考

下面保留的是main原有根目录 `firmware/` 工程说明，属于旧文本联调链路。新版上位机应使用 `upper_computer/firmware/Test/`，不要用下方旧配置替代。

## 仓库结构

| 路径 | 说明 |
| --- | --- |
| `firmware/` | STM32H743VGT6 采集固件，STM32CubeMX 生成 + HAL |
| `firmware/Test.ioc` | CubeMX 工程配置。**改引脚或外设一律改这里，再重新生成代码** |

<!-- TODO: 之后加入的 Python / 模型 / 数据集 / 文档目录补到这里 -->

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
