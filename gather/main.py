import json
import threading
import time
from datetime import datetime
from pathlib import Path

from device import Device
from load_cell import LoadCell
from mems_sensor import MEMSSensor


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "gathered_data"


def build_device() -> Device:
    load_cell = LoadCell("weight")
    mems = MEMSSensor()
    return Device(load_cell, mems_sensor=mems, verbose=False)


def save_marks(path: Path, start_epoch: float, marks: list[float]) -> None:
    with open(path, "w") as file:
        json.dump({"t0_epoch": start_epoch, "marks_sec": marks}, file, indent=2)


def read_device_loop(device: Device, stop: threading.Event) -> None:
    while not stop.is_set():
        device.read_line()
        if device.mems_sensor.signal_lost.is_set():
            stop.set()
            print("\nMEMS signal lost. Stopping recording.")
            return


def main() -> None:
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    recording_dir = DATA_DIR / suffix
    recording_dir.mkdir(parents=True, exist_ok=True)

    log_path = recording_dir / "weights.csv"
    mems_path = recording_dir / "mems.bin"
    marks_path = recording_dir / "marks.json"

    device = build_device()
    device.initialize_logging(log_path, mems_path)

    start_epoch = time.time()
    marks: list[float] = []
    stop = threading.Event()
    reader = threading.Thread(target=read_device_loop, args=(device, stop), daemon=True)
    reader.start()

    print(f"Logging to {log_path}")
    print("Press Enter to mark a bite. Press Ctrl+C to stop.")

    try:
        while not stop.is_set():
            input()
            mark = time.time() - start_epoch
            marks.append(mark)
            print(f"MARK #{len(marks)} at {mark:.2f}s")
    except KeyboardInterrupt:
        print("\nStopping recording.")
    finally:
        stop.set()
        reader.join(timeout=3)
        save_marks(marks_path, start_epoch, marks)
        device.close()
        print(f"Saved {len(marks)} marks to {marks_path}")


if __name__ == "__main__":
    main()
