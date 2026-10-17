# sEMG binary protocol v1 / v2

## Current v0.6.0 firmware: 2 kHz per channel (2026-10-16)

The current `Test/` sources use TIM3 prescaler 239 and ARR `1000000 / 2000 - 1 = 499` with the existing nominal 240 MHz timer clock. UART is **230400, 8N1**. INFO is `(2000,2,16,dropped)`. Protocol version remains 2 (two interleaved channels), with the same 20-pair/100-byte DATA frame. Frames arrive nominally every 10 ms; DATA traffic is 10000 bytes/s, below the 23040 bytes/s nominal UART capacity. ACK block quantization is 20 sample pairs = 10 ms, not 20 ms.

ADC pin order and reference voltage are unchanged. MCU sends raw ADC values; host selects 20–500 Hz + 50 Hz notch at 2000 Hz. No MCU digital filter or decimation. Source and `.ioc` timer/UART settings agree, but CubeMX regeneration still requires preserving custom stream logic and linker/DMA settings.

The updated host also accepts old 1000 Hz INFO with the old display/model definitions (select 115200 for old firmware). It rejects a sample-rate change during a connection. Build log `build_v0.6.0.log`: 0 errors, 0 warnings. Flashed and verified on 2026-10-16 via CMSIS-DAP, with automatic reset/run enabled. COM5 at 230400 delivered 24140 pairs in about 12 seconds, INFO=(2000,2,16,0), no CRC errors, missing pairs or fault flags, and working marker ACKs. ACK sample-index/tick differences give 2000 Hz/channel (not an independent absolute clock calibration). CH1 had 5 ADC samples equal to 0; analogue signal quality remains to be checked. Evidence: `../releases/v0.6.0-2khz/flash_run_20261016.log` and `hardware_check_20261016.json`. The remaining sections describe the historical v0.5.0 / v1 behavior.

## v2 dual-channel extension (application v0.5.0)

The current `Test/` project includes `stream_v2.inc`. It scans ADC1 rank 1 PA7/INP7 (CH0), then rank 2 PC5/INP8 (CH1) per 1 kHz TIM3 trigger. These are sequential conversions in one trigger cycle, not simultaneously sampled ADCs. No digital filtering, envelope or sample decimation runs on the MCU.

Header version byte is 2. DATA count and first index refer to sample PAIRS (time points), not total uint16 values. Payload layout is interleaved little-endian `ch0[t], ch1[t], ch0[t+1], ch1[t+1], ...`; payload size is count*4. Each 20-pair frame has 80 payload bytes and 100 total bytes. INFO reports `(1000,2,16,dropped)`. ACK, CRC, sequence and fault semantics remain as below. A version change during a connection is rejected. v1 still always means one channel; v2 always means two.

DMA ADC buffer is 80 halfwords: each half holds 20 pairs. Frame slot grows to 128 bytes (32 slots). Nominal DATA traffic is 5000 bytes/s, plus small INFO/ACK traffic, within 115200 8N1 nominal 11520 bytes/s capacity. DMA buffers remain in D2 SRAM; cache coherency assumptions are unchanged. Hardware throughput and timing require board validation.

Teammate reference: https://github.com/l-lorne-n/6008_silent_speech/commit/3ca6a226fba117b0b176590d11226f7ee22e40f0 . Only dual ADC/pin configuration is integrated; its six-column 100-line/s diagnostic CSV is not the v2 protocol. Build log `build_v0.5.0.log` confirms compilation, not flashing or hardware validation.

## Original v1 specification (retained for old devices)

UART: 115200 baud, 8 data bits, no parity, 1 stop bit, no flow control. All integers little-endian. Version 1 supports one physical ADC channel at nominal 1000 samples/s, unsigned 16-bit ADC counts.

## Device to host frame

| Offset | Bytes | Field |
|---|---:|---|
| 0 | 2 | Magic A5 5A |
| 2 | 1 | Version = 1 |
| 3 | 1 | Type: DATA=1, INFO=2, ACK=3 |
| 4 | 2 | Payload byte length |
| 6 | 4 | Frame sequence, increments per enqueue attempt, wraps uint32 |
| 10 | 4 | First sample index for DATA; latest completed block boundary for INFO/ACK |
| 14 | 2 | Sample count, DATA only; otherwise zero |
| 16 | 2 | Sticky fault flags |
| 18 | variable | Payload |
| 18+length | 2 | CRC16/CCITT-FALSE, little-endian |

CRC covers the entire 18-byte header INCLUDING magic, then payload. Polynomial 0x1021, init 0xFFFF, no reflection, xorout 0. Standard check: ASCII `123456789` produces 0x29B1. Python equivalent: `binascii.crc_hqx(header_plus_payload, 0xFFFF)`.

DATA payload: sample_count consecutive uint16 values. Firmware sends 20 values / 40 payload bytes / 60 total bytes approximately every 20 ms. Index 0 is the first ADC output after starting this boot. Samples are indexed from ADC completed blocks even when transmission fails, not from successful UART delivery. Queue overflow increments dropped count and sets sticky fault bit 0.

INFO payload: `<IHHI>` = nominal_fs_hz, channel_count, adc_bits, dropped_frame_attempts. Expected values 1000, 1, 16, count. INFO sent once on startup and upon command I. No unique boot UUID; host treats backward/repeated frame sequence or sample index as a discontinuity and ends the connection. Do not stitch reconnects into one batch.

ACK payload: `<II>` = token, HAL_GetTick milliseconds captured while parsing the completed command. Header index is the next sample index at the latest completed 20-sample boundary. It can lag actual ADC progress by approximately 0–20 ms plus interrupt servicing delay. It is not a timestamp of physical screen presentation.

Fault flags: bit 0 TX queue overflow; bit 1 ADC error; bit 2 UART error. Sticky until reset. Host stops acquisition when nonzero rather than silently accepting compromised data. ADC global IRQ is enabled so HAL can surface ADC errors.

## Host to device commands

ASCII commands terminated by LF (optional CR before LF):

```text
I\n
M,123\n
```

`I` requests INFO. `M,<uint32 token>` requests ACK capturing the current block boundary. GUI phase marks and periodic link checks use separate token/event mappings in events.jsonl. Non-numeric or overflowing tokens and overlong lines are ignored; maximum accepted line length is 31 bytes excluding LF.

Legacy mode sends NO commands, only receives existing `raw,filtered,envelope\r\n` output. Binary mode requires valid samples, INFO and a matching ACK before recording is enabled. Host periodically repeats I and M every 2 seconds; no fitted clock-drift model is used in v1.

## Firmware structure and limitations

`stream_v1.inc` implements bounded TX ring, CRC, ADC callbacks and UART callbacks. ADC IRQ copies the finished half-buffer into a queue slot before DMA can reuse it. UART DMA completion releases the queue slot. Queue mutations are protected from concurrent ISR/main updates; buffers being transmitted are not reused.

`prepare_firmware.py` creates `Test/` from the original firmware without overwriting an existing copy. It records a source hash, changes main.c user regions, adds a scatter file for D2 SRAM and defines DATA_IN_D2_SRAM so SystemInit enables its clocks before C runtime memory initialization. The source `.ioc` is retained as reference; automatic CubeMX regeneration is not part of this build workflow.

No D-cache is enabled in the original startup or new main. If cache is enabled later, DMA coherency must be revisited. Nominal sampling rate derives from the original HSE/PLL/TIM3 configuration and requires actual board verification. This firmware has been compiled but not flashed or hardware-timed in this delivery.
