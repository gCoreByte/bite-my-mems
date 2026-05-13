import argparse
import json
import statistics
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from device import Device, SLOW_PACKET
from load_cell import LoadCell
from mems_sensor import MEMSSensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from recordings import mic_calibration_factor  # noqa: E402


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "gathered_data"
MAX_CONSECUTIVE_STUCK_READINGS = 3
STUCK_READING_TIMEOUT = 1
BASELINE_DURATION_SEC = 1.0


def build_device(verbose: bool = False) -> Device:
    load_cell = LoadCell("weight")
    mems = MEMSSensor()
    return Device(load_cell, mems_sensor=mems, verbose=verbose)


def save_marks(path: Path, start_epoch: float, marks: list[float], mic_calibration: float | None) -> None:
    payload: dict = {"t0_epoch": start_epoch, "marks_sec": marks}
    if mic_calibration is not None:
        payload["mic_calibration"] = mic_calibration
    with open(path, "w") as file:
        json.dump(payload, file, indent=2)


def collect_baseline(device: Device, duration: float = BASELINE_DURATION_SEC) -> float:
    """Read packets for `duration` seconds and return the median weight reading."""
    load_cell = device.sensors[0]
    samples: list[float] = []
    end = time.monotonic() + duration
    while time.monotonic() < end:
        magic = device.read_line()
        if device.mems_sensor.signal_lost.is_set():
            raise RuntimeError("MEMS signal lost during baseline collection.")
        if magic == SLOW_PACKET:
            samples.extend(load_cell.last_read)

    if not samples:
        raise RuntimeError("No load cell samples received during baseline collection.")
    return statistics.median(samples)


def read_device_loop(
    device: Device,
    stop: threading.Event,
    weight_lost: threading.Event,
    weight_lost_reason: list[str],
) -> None:
    load_cell = device.sensors[0]
    last_weight_time = time.monotonic()
    recent_weights: deque[float] = deque(maxlen=3)

    while not stop.is_set():
        magic = device.read_line()
        if device.mems_sensor.signal_lost.is_set():
            stop.set()
            print("\nMEMS signal lost. Stopping recording.")
            return

        if magic == SLOW_PACKET:
            last_weight_time = time.monotonic()
            for weight in load_cell.last_read:
                recent_weights.append(weight)
                if len(recent_weights) == MAX_CONSECUTIVE_STUCK_READINGS and len(set(recent_weights)) == 1:
                    weight_lost_reason.append(f"Load cell stuck at {weight} for {MAX_CONSECUTIVE_STUCK_READINGS} consecutive readings.")
                    weight_lost.set()
                    stop.set()
                    return
        elif time.monotonic() - last_weight_time > STUCK_READING_TIMEOUT:
            weight_lost_reason.append(f"No weight data received from HX711 for {STUCK_READING_TIMEOUT} seconds.")
            weight_lost.set()
            stop.set()
            return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--type", choices=["soft", "medium", "hard", "none"], default=None)
    args = parser.parse_args()

    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.type is not None:
        suffix = f"{suffix}_{args.type}"
    recording_dir = DATA_DIR / suffix
    recording_dir.mkdir(parents=True, exist_ok=True)

    log_path = recording_dir / "weights.csv"
    mems_path = recording_dir / "mems.bin"
    marks_path = recording_dir / "marks.json"

    device = build_device(verbose=args.verbose)

    print("\nRestarting device...")
    device.reset()
    time.sleep(5)
    device.resync()

    print(f"\nCollecting baseline for {BASELINE_DURATION_SEC:.1f}s — keep the jaw still and unloaded.")
    baseline = collect_baseline(device)
    load_cell = device.sensors[0]
    load_cell.baseline = baseline
    load_cell._history.clear()
    print(f"[LoadCell] Baseline: {baseline:.2f}")

    baseline_audio = device.mems_sensor.get_last_samples(int(MEMSSensor.SAMPLE_RATE * BASELINE_DURATION_SEC))
    baseline_audio = baseline_audio - baseline_audio.mean()
    try:
        mic_calibration = mic_calibration_factor(baseline_audio, baseline_sec=BASELINE_DURATION_SEC)
        print(f"[MEMS] Mic calibration (band-pass envelope median): {mic_calibration:.2f}")
    except ValueError as error:
        print(f"[MEMS] Skipping calibration: {error}")
        mic_calibration = None

    device.initialize_logging(log_path, mems_path)

    start_epoch = time.time()
    marks: list[float] = []
    stop = threading.Event()
    weight_lost = threading.Event()
    weight_lost_reason: list[str] = []
    reader = threading.Thread(
        target=read_device_loop,
        args=(device, stop, weight_lost, weight_lost_reason),
        daemon=True,
    )
    reader.start()

    print(f"\nLogging to {log_path}")
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
        save_marks(marks_path, start_epoch, marks, mic_calibration)
        device.close()
        print(f"Saved {len(marks)} marks to {marks_path}")
        if weight_lost.is_set():
            raise RuntimeError(weight_lost_reason[0])


if __name__ == "__main__":
    main()
