# Bite Force Estimation via Bioacoustic Sensing

## Wiring

### Load Cell (HX711)
| Signal | GPIO |
|--------|------|
| DOUT   | 19   |
| SCK    | 18   |
| VCC    | 3v3  |
| GND    | GND  |

### MEMS Microphone
| Signal | GPIO |
|--------|------|
| OUT    | 34   |
| VCC    | 3V3  |
| GND    | GND  |

## Architecture

```
ESP32 Core 0                    ESP32 Core 1                  Laptop (Python)
+--------------+                +--------------+              +------------------+
| MEMS task    |                | loop()       |    USB       | Device.read_line |
| I2S DMA ADC  |--+  serial     | HX711 poll   |---460800---> | +- 0xABCD -> CSV |
| 22050 Hz     |  +- mutex -->  | FSR reads    |   baud       | +- 0xABCE -> .bin|
| GPIO 34      |--+             | slow packets |              | + ring buffer    |
+--------------+                +--------------+              +------------------+
```

- **Core 0**: I2S ADC DMA samples MEMS mic at 22,050 Hz in hardware (zero jitter)
- **Core 1**: Polls HX711 load cell, sends slow data packets
- **Serial mutex** prevents interleaved writes from both cores
- **Baud rate**: 460800 (~46 KB/s throughput; MEMS uses ~22 KB/s)

## Binary Protocol

Two packet types share a common header:

| Offset | Size | Field |
|--------|------|-------|
| 0      | 2    | Magic (`0xABCD` slow, `0xABCE` MEMS) |
| 2      | 2    | Payload count (uint16) |
| 4      | 4    | Timestamp in ms (uint32) |
| 8+     | ...  | Payload |
| last   | 1    | XOR checksum |

**Slow packet (0xABCD)** - per reading (10 bytes):
- `int32` loadcell, `uint16` fsr0, `uint16` fsr1, `uint16` delta_ms

**MEMS packet (0xABCE)** - 512 x `uint8` samples (12-bit ADC >> 4)

## Usage

1. Flash `src/main.cpp` to the ESP32
2. Run `python gather/main.py`
3. Data is logged to `data/gathered_data/YYYYMMDD_HHMMSS/weights.csv` and `data/gathered_data/YYYYMMDD_HHMMSS/mems.bin`
