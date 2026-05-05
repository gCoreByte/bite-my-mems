import struct
from pathlib import Path

from sensor import Sensor


SCALE_FILE = Path(__file__).resolve().parents[1] / "data" / "scale.bin"


class LoadCell(Sensor):
    """Load cell channel that applies tare and scale factor from scale.bin."""

    def __init__(self, sensor_id: str = "weight", scale_file: Path | None = None) -> None:
        super().__init__(sensor_id)
        self.scale_file = Path(scale_file) if scale_file is not None else SCALE_FILE
        self.load_settings()

    def load_settings(self) -> None:
        with self.scale_file.open("rb") as file:
            self.factor, self.tare_value = struct.unpack("ff", file.read(8))

        print(f"[LoadCell] Factor: {self.factor}")
        print(f"[LoadCell] Tare:   {self.tare_value}")

    def save_data(self, raw_value: float) -> None:
        weight = round((raw_value - self.tare_value) / self.factor, 2)
        super().save_data(weight)
