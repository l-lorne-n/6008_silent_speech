# sEMG binary protocol v1

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
