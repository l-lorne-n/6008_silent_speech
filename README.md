# 6008 Silent Speech

采集喉部单通道表面肌电（sEMG），研究闭唇、不出声、内部尝试发音时的词语分类。

> **当前分支：`upper_computer_v0.4.1`**。已加入Windows上位机：完整1 kHz串口接收、提词与原始数据保存、可配置词表、离线LDA训练及模型测试。默认open / close / next / music，可添加词语扩展到十词。CNN和串口实时分类尚未实现。

## 上位机快速开始

推荐Windows 10/11及64位Python 3.12。安装Python时启用Python launcher或加入PATH，然后：

```powershell
git clone --branch upper_computer_v0.4.1 https://github.com/l-lorne-n/6008_silent_speech.git
cd 6008_silent_speech\upper_computer
.\setup.cmd
.\start.cmd
```

首次安装需要网络。可先使用“模拟演练”检查界面；真实采集须连接已烧录完整采样固件的板子，并选择这台电脑实际串口。详细依赖、词表、数据与模型操作见 [上位机README](upper_computer/README.md) 和 [LDA使用说明](upper_computer/LDA使用说明.md)。

源码不依赖作者的L盘目录。Python环境、实验录制、模型、个人词表和编译HEX不包含在此源码分支，需要在本机安装或另外分享。克隆后没有历史录制和训练模型是正常情况。

**注意两份固件的区别：** 本分支保留根目录`firmware/`作为原三列文本联调工程（串口约100行/秒）。上位机正式采集对应的完整1 kHz二进制固件位于 [`upper_computer/firmware/Test/`](upper_computer/firmware/Test/)，Keil入口为 [`Test.uvprojx`](upper_computer/firmware/Test/MDK-ARM/Test.uvprojx)。其协议、20点块及双向标记见 [PROTOCOL.md](upper_computer/firmware/PROTOCOL.md)。不要将根目录旧固件当作完整采样固件，也不要用CubeMX直接覆盖完整采样工程的自定义逻辑。

## 仓库结构

| 路径 | 说明 |
| --- | --- |
| `upper_computer/` | v0.4.1上位机、运行依赖、模拟采集及离线LDA |
| `upper_computer/firmware/Test/` | 与上位机正式模式配套的完整1 kHz二进制固件 |
| `firmware/` | 原STM32H743VGT6三列文本联调固件，STM32CubeMX生成 + HAL |
| `firmware/Test.ioc` | 原文本固件的CubeMX工程配置；不适用于覆盖上面的完整采样工程 |

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

## 原文本固件采集链路（根目录firmware）

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

## 原文本固件串口输出协议

ASCII CSV，每行以 `\r\n` 结尾，输出速率 100 行/秒（1 kHz 每 10 点一行）。字段由 `VERIFY_MODE` 决定：

| `VERIFY_MODE` | 每行格式 | 用途 |
| --- | --- | --- |
| `0` | `raw` | 只看原始 ADC 值 |
| `1` | `raw,envelope` | 原始值 + 包络 |
| `2`（当前） | `raw,filtered,envelope` | 全链路验证 |

ADC 为 16 bit，代码里以 `32768`（即 1.65 V 中点）为零点做去偏置，所以 `raw` 是无符号值，`filtered` / `envelope` 是相对中点、取整后的有符号值。

<!-- TODO: 如果上位机按固定列名解析，把解析用的 Python 脚本路径和列定义补进来 -->

## 原文本固件构建与烧录

需要 STM32CubeCLT，或用 STM32Cube 的 VS Code 扩展自带工具链（`arm-none-eabi-gcc` + Ninja）。

```bash
cd firmware
cmake --preset Debug
cmake --build --preset Debug
```

用 Keil MDK-ARM 的话，直接打开 `firmware/MDK-ARM/Test.uvprojx`。

<!-- TODO: 补上烧录方式（STM32CubeProgrammer / ST-Link / openocd 命令） -->

## 原文本固件开发约定

- 外设、引脚、时钟的改动走 `firmware/Test.ioc`，用 CubeMX 重新生成，不要手改生成区域的代码。
- 自己的代码写在 `/* USER CODE BEGIN xxx */` 和 `/* USER CODE END xxx */` 之间，否则重新生成时会被覆盖。
- 编译产物（`build/`、`MDK-ARM/Test/`、Keil 的 `.uvguix.*` 等）已在 `.gitignore` 里排除，不要提交。
- `firmware/Drivers/CMSIS/DSP`、`NN`、`RTOS*` 目前被 `.gitignore` 排除，因为这个工程没有引用它们。**如果之后要启用 DSP 库或 FreeRTOS，先删掉 `.gitignore` 里对应的行。**

## 已知问题 / 待办

- [ ] 源文件目前是 **GBK 编码**，GitHub 网页和 macOS / Linux 上会显示成乱码，建议统一转成 UTF-8。
- [ ] 补上项目简介、传感器型号与实物接线。
- [x] 上位机接收、提词、保存和可配置词表，见upper_computer。
- [x] 离线LDA训练、分组评估、模型保存及加载测试。
- [ ] 跨录制稳定性验证、串口实时识别及未知词拒识。

<!-- TODO: 补上团队成员分工与联系方式 -->
