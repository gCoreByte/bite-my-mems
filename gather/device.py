import datetime as dt
import struct

import numpy as np
import serial
import serial.tools.list_ports


BAUD_RATE = 460800
SLOW_PACKET = 0xABCD
MEMS_PACKET = 0xABCE
MAX_SLOW_COUNT = 100
MAX_MEMS_COUNT = 1024

KNOWN_DEVICES = (
    {"vid": 0x10C4, "pid": 0xEA60},  # CP2102
    {"vid": 0x1A86, "pid": 0x7523},  # CH340G
    {"vid": 0x1A86, "pid": 0x55D4},  # CH343
    {"vid": 0x0403},                 # FTDI
    {"vid": 0x2341},                 # Arduino LLC
    {"vid": 0x2A03},                 # Arduino.org
)


class Device:
    """Serial reader for slow force packets and fast MEMS packets."""

    def __init__(self, *sensors, mems_sensor, baud_rate: int = BAUD_RATE, verbose: bool = False) -> None:
        self.sensors = list(sensors)
        self.mems_sensor = mems_sensor
        self.timestamps: list[dt.datetime] = []
        self.log_path = None
        self.start_time = None
        self.verbose = verbose

        for sensor in self.sensors:
            sensor.timestamps = self.timestamps

        port = self.find_device()
        if port is None:
            raise RuntimeError("No serial device found.")
        self.serial = serial.Serial(port, baud_rate, timeout=2)
        print(f"Connected to {port}")

    def read_line(self) -> int:
        """Read one packet from serial. Returns the magic value."""
        magic = self._read_magic()
        if magic == SLOW_PACKET:
            self._read_slow_packet()
        else:
            self._read_mems_packet()
        return magic

    def _read_magic(self) -> int:
        previous = self._read_exact(1)
        while True:
            current = self._read_exact(1)
            magic = struct.unpack("<H", previous + current)[0]
            if magic in (SLOW_PACKET, MEMS_PACKET):
                return magic
            previous = current

    def _read_slow_packet(self) -> None:
        for sensor in self.sensors:
            sensor.last_read.clear()
        self.timestamps.clear()

        count = self._read_uint16()
        if count > MAX_SLOW_COUNT:
            raise RuntimeError(f"Slow packet count {count} exceeds limit {MAX_SLOW_COUNT}.")

        elapsed_ms = self._read_uint16()
        self.start_time = dt.datetime.now() - dt.timedelta(milliseconds=elapsed_ms)

        for _ in range(count):
            for sensor in self.sensors:
                sensor.save_data(self._read_int32())

            sample_ms = self._read_int32()
            self.timestamps.append(self.start_time + dt.timedelta(milliseconds=sample_ms))

        if self.verbose:
            self.print_readings()
        self.log()

    def _read_mems_packet(self) -> None:
        count = self._read_uint16()
        if count > MAX_MEMS_COUNT:
            raise RuntimeError(f"MEMS packet count {count} exceeds limit {MAX_MEMS_COUNT}.")

        self._read_uint32()  # base timestamp, unused
        payload = self._read_exact(count * 2)
        checksum = self._read_exact(1)
        if checksum[0] != self._checksum(payload):
            raise RuntimeError("MEMS packet checksum mismatch.")

        samples = np.frombuffer(payload, dtype=np.uint16).astype(np.float32)
        self.mems_sensor.append_samples(samples)
        self.mems_sensor.log(samples)

    def initialize_logging(self, log_path: str, mems_path: str) -> None:
        self.log_path = log_path
        with open(log_path, "w") as file:
            header = ",".join(sensor.ID for sensor in self.sensors)
            file.write(f"{header},timestamp\n")

        self.mems_sensor.initialize_logging(mems_path)

    def log(self) -> None:
        rows = []
        for index, moment in enumerate(self.timestamps):
            values = [str(sensor.last_read[index]) for sensor in self.sensors]
            rows.append(f"{','.join(values)},{moment}\n")

        with open(self.log_path, "a") as file:
            file.writelines(rows)

    def print_readings(self) -> None:
        print("-" * 30)
        for sensor in self.sensors:
            print(f"ID: {sensor.ID} | Data: {sensor.get_smoothed()}")

    def close(self) -> None:
        self.mems_sensor.close()
        self.serial.close()

    def _read_exact(self, count: int) -> bytes:
        data = self.serial.read(count)
        if len(data) != count:
            raise RuntimeError(f"Serial read timed out: wanted {count} bytes, got {len(data)}.")
        return data

    def _read_uint16(self) -> int:
        return struct.unpack("<H", self._read_exact(2))[0]

    def _read_uint32(self) -> int:
        return struct.unpack("<I", self._read_exact(4))[0]

    def _read_int32(self) -> int:
        return struct.unpack("<i", self._read_exact(4))[0]

    @staticmethod
    def _checksum(payload: bytes) -> int:
        checksum = 0
        for byte in payload:
            checksum ^= byte
        return checksum

    @staticmethod
    def find_device() -> str | None:
        for port in serial.tools.list_ports.comports():
            for device in KNOWN_DEVICES:
                vid_matches = port.vid == device.get("vid")
                pid_matches = "pid" not in device or port.pid == device["pid"]
                if vid_matches and pid_matches:
                    return port.device
        return None
