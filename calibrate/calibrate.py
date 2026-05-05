import struct
import time
from pathlib import Path

import serial
import serial.tools.list_ports


BAUD_RATE = 460800
NUM_SAMPLES = 20
SCALE_FILE = Path(__file__).resolve().parents[1] / "data" / "scale.bin"

KNOWN_DEVICES = (
    {"vid": 0x10C4, "pid": 0xEA60},
    {"vid": 0x1A86, "pid": 0x7523},
    {"vid": 0x1A86, "pid": 0x55D4},
    {"vid": 0x0403},
    {"vid": 0x2341},
    {"vid": 0x2A03},
)


def find_device() -> str:
    for port in serial.tools.list_ports.comports():
        for device in KNOWN_DEVICES:
            vid_matches = port.vid == device.get("vid")
            pid_matches = "pid" not in device or port.pid == device["pid"]
            if vid_matches and pid_matches:
                return port.device
    raise RuntimeError("No serial device found.")


def read_samples(connection: serial.Serial, count: int = NUM_SAMPLES) -> list[int]:
    samples = []
    connection.reset_input_buffer()

    while len(samples) < count:
        line = connection.readline().decode(errors="ignore").strip()
        if not line:
            continue
        samples.append(int(line))

    return samples


def average(values: list[int]) -> float:
    return sum(values) / len(values)


def write_scale_file(factor: float, tare: float) -> None:
    SCALE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SCALE_FILE.open("wb") as file:
        file.write(struct.pack("ff", float(factor), float(tare)))


def main() -> None:
    port = find_device()
    print(f"Connected to {port}")
    with serial.Serial(port, BAUD_RATE, timeout=2) as connection:
        time.sleep(2)

        input(f"Remove all weight, then press Enter to tare ({NUM_SAMPLES} samples).")
        tare_samples = read_samples(connection)
        tare = average(tare_samples)
        print(f"Tare value: {tare:.2f}")

        known_weight = float(input("Place known weight on scale. Enter grams: "))
        input(f"Press Enter to measure ({NUM_SAMPLES} samples).")
        raw_average = average(read_samples(connection))

        factor = (raw_average - tare) / known_weight
        write_scale_file(factor, tare)

        print(f"\nScale factor: {factor:.6f}")
        print(f"Tare value:   {tare:.2f}")
        print(f"Written to {SCALE_FILE}")

        input("\nPress Enter to take one test reading.")
        test_average = average(read_samples(connection))
        measured = (test_average - tare) / factor
        print(f"Test reading: {measured:.2f} g")


if __name__ == "__main__":
    main()
