import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, sosfilt


MEMS_SAMPLE_RATE = 22050
WINDOW_SEC = 1.0
ENV_WINDOW_MS = 10
ENV_SAMPLE_COUNT = int(2 * WINDOW_SEC * (1000 / ENV_WINDOW_MS))
LC_SATURATION = 8388607
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "gathered_data"
WEIGHTS_FILE = "weights.csv"
MEMS_FILE = "mems.bin"
MARKS_FILE = "marks.json"

BANDS = (
    ("broad", 300, 5000),
    ("low", 300, 1000),
    ("mid", 1000, 3000),
    ("high", 3000, 5000),
)


@dataclass(frozen=True)
class Recording:
    csv_path: Path
    mems_path: Path
    marks_path: Path | None
    label: str


def recording_from_directory(directory: Path) -> Recording | None:
    csv_path = directory / WEIGHTS_FILE
    mems_path = directory / MEMS_FILE
    if not csv_path.exists() or not mems_path.exists():
        return None

    marks_path = directory / MARKS_FILE
    return Recording(
        csv_path=csv_path,
        mems_path=mems_path,
        marks_path=marks_path if marks_path.exists() else None,
        label=directory.name,
    )


def find_recordings(directory: str | Path = DATA_DIR) -> list[Recording]:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Recordings directory {directory} does not exist.")

    recordings = []
    for recording_dir in sorted(path for path in directory.iterdir() if path.is_dir()):
        recording = recording_from_directory(recording_dir)
        if recording is not None:
            recordings.append(recording)

    return recordings


def load_log(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    if data.empty:
        raise ValueError(f"{path} is empty.")

    data["timestamp"] = pd.to_datetime(data["timestamp"], format="ISO8601")
    start = data["timestamp"].iloc[0]
    data["t_sec"] = (data["timestamp"] - start).dt.total_seconds()
    return data


def load_mems(path: Path) -> np.ndarray:
    samples = np.fromfile(path, dtype=np.uint16).astype(np.float32)
    return samples - np.mean(samples)


def bandpass(samples: np.ndarray, low: int, high: int) -> np.ndarray:
    sos = butter(4, [low, high], btype="band", fs=MEMS_SAMPLE_RATE, output="sos")
    return sosfilt(sos, samples)


def mic_calibration_factor(
    samples: np.ndarray,
    baseline_sec: float = 2.0,
    band: tuple[int, int] = (300, 5000),
    smooth_ms: int = 30,
) -> float:
    """Median band-pass envelope over the first `baseline_sec` of audio.

    Used as a per-recording mic-gain reference: divide the rest of the recording
    by this value to put the ambient noise floor on a common scale across sessions.
    """
    baseline_samples = int(baseline_sec * MEMS_SAMPLE_RATE)
    quiet = samples[:baseline_samples]
    if len(quiet) < MEMS_SAMPLE_RATE // 4:
        raise ValueError("Not enough baseline audio to compute calibration.")

    filtered = bandpass(quiet, band[0], band[1])
    smooth_window = max(1, int(MEMS_SAMPLE_RATE * smooth_ms / 1000))
    envelope = np.sqrt(uniform_filter1d(filtered**2, size=smooth_window))
    return float(max(np.median(envelope), 1e-6))


def load_mic_calibration(recording: Recording, samples: np.ndarray | None = None) -> float:
    """Return the persisted mic calibration if present, else recompute from audio."""
    if recording.marks_path is not None:
        with recording.marks_path.open() as file:
            payload = json.load(file)
        value = payload.get("mic_calibration")
        if value is not None:
            return float(value)

    if samples is None:
        samples = load_mems(recording.mems_path)
    return mic_calibration_factor(samples)


def compute_envelopes(samples: np.ndarray, time_scale: float) -> tuple[np.ndarray, list[np.ndarray]]:
    window = int(MEMS_SAMPLE_RATE * ENV_WINDOW_MS / 1000)
    envelopes = []

    for _, low, high in BANDS:
        filtered = bandpass(samples, low, high)
        envelopes.append(np.sqrt(uniform_filter1d(filtered**2, size=window)))

    time_axis = np.arange(len(envelopes[0])) / MEMS_SAMPLE_RATE * time_scale
    return time_axis, envelopes


def detrend_load_cell(values: np.ndarray) -> np.ndarray:
    baseline = uniform_filter1d(values, size=200)
    return np.abs(values - baseline)


def detect_load_cell_peaks(times: np.ndarray, values: np.ndarray) -> tuple[list[float], np.ndarray]:
    detrended = detrend_load_cell(values)
    diffs = np.diff(times)
    sample_interval = np.median(diffs[diffs > 0]) if np.any(diffs > 0) else 1.0
    threshold = np.median(detrended) + 3.5 * np.std(detrended)
    min_distance = max(int(1.0 / sample_interval), 1)

    peaks, _ = find_peaks(detrended, height=threshold, distance=min_distance)
    valid = np.abs(values[peaks]) < LC_SATURATION
    return [float(times[index]) for index in peaks[valid]], detrended


def event_times(recording: Recording, log: pd.DataFrame) -> tuple[list[float], np.ndarray, str]:
    values = log["weight"].values.astype(float)
    if recording.marks_path is None:
        times, detrended = detect_load_cell_peaks(log["t_sec"].values, values)
        return times, detrended, "auto"

    with recording.marks_path.open() as file:
        marks = json.load(file)["marks_sec"]
    return [float(mark) for mark in marks], detrend_load_cell(values), "manual"


def force_near(time_sec: float, times: np.ndarray, forces: np.ndarray, window_sec: float = 0.5) -> float:
    mask = (times >= time_sec - window_sec) & (times <= time_sec + window_sec)
    return float(np.max(forces[mask])) if np.any(mask) else 0.0


def extract_window(time_sec: float, envelope_times: np.ndarray, envelopes: list[np.ndarray]) -> np.ndarray | None:
    start = int(np.searchsorted(envelope_times, time_sec - WINDOW_SEC))
    end = int(np.searchsorted(envelope_times, time_sec + WINDOW_SEC))
    if end >= len(envelope_times):
        return None

    channels = []
    for envelope in envelopes:
        segment = envelope[start:end]
        if len(segment) < 10:
            return None

        channels.append(
            np.interp(
                np.linspace(0, 1, ENV_SAMPLE_COUNT),
                np.linspace(0, 1, len(segment)),
                segment,
            )
        )

    return np.stack(channels, axis=0).astype(np.float32)


def load_force_windows(recording: Recording) -> tuple[list[np.ndarray], list[float]]:
    log = load_log(recording.csv_path)
    mems = load_mems(recording.mems_path)
    calibration = load_mic_calibration(recording, mems)
    time_scale = log["t_sec"].iloc[-1] / (len(mems) / MEMS_SAMPLE_RATE)
    envelope_times, envelopes = compute_envelopes(mems, time_scale)
    envelopes = [envelope / calibration for envelope in envelopes]

    events, detrended_force, source = event_times(recording, log)
    windows = []
    forces = []

    for time_sec in events:
        window = extract_window(time_sec, envelope_times, envelopes)
        if window is None:
            continue
        windows.append(window)
        forces.append(force_near(time_sec, log["t_sec"].values, detrended_force))

    print(f"  [{recording.label}] {len(windows)}/{len(events)} events ({source})")
    return windows, forces


def load_binary_windows(recording: Recording, negative_ratio: int = 2) -> tuple[list[np.ndarray], list[np.ndarray]]:
    log = load_log(recording.csv_path)
    mems = load_mems(recording.mems_path)
    calibration = load_mic_calibration(recording, mems)
    time_scale = log["t_sec"].iloc[-1] / (len(mems) / MEMS_SAMPLE_RATE)
    envelope_times, envelopes = compute_envelopes(mems, time_scale)
    envelopes = [envelope / calibration for envelope in envelopes]
    bite_times, _, source = event_times(recording, log)

    positives = [
        window
        for window in (extract_window(time_sec, envelope_times, envelopes) for time_sec in bite_times)
        if window is not None
    ]

    negatives: list[np.ndarray] = []
    if not bite_times:
        candidate = WINDOW_SEC
        end_time = float(envelope_times[-1]) - WINDOW_SEC
        while candidate <= end_time:
            window = extract_window(candidate, envelope_times, envelopes)
            if window is not None:
                negatives.append(window)
            candidate += 2 * WINDOW_SEC
    else:
        seed = int.from_bytes(hashlib.md5(recording.label.encode()).digest()[:4], "big")
        rng = np.random.default_rng(seed)
        attempts = 0
        target_negatives = len(positives) * negative_ratio
        max_attempts = max(target_negatives * 20, 1)

        while len(negatives) < target_negatives and attempts < max_attempts:
            candidate = rng.uniform(WINDOW_SEC, float(envelope_times[-1]) - WINDOW_SEC)
            far_from_bites = all(abs(candidate - bite_time) > 2 * WINDOW_SEC for bite_time in bite_times)
            if far_from_bites:
                window = extract_window(candidate, envelope_times, envelopes)
                if window is not None:
                    negatives.append(window)
            attempts += 1

    print(f"  [{recording.label}] {len(positives)} bites, {len(negatives)} non-bites ({source})")
    return positives, negatives


def normalize_windows_by_recording(windows: list[np.ndarray]) -> list[np.ndarray]:
    if not windows:
        return []

    data = np.array(windows, dtype=np.float32)
    for channel in range(data.shape[1]):
        mean = data[:, channel, :].mean()
        std = data[:, channel, :].std()
        if std > 0:
            data[:, channel, :] = (data[:, channel, :] - mean) / std

    return [data[index] for index in range(len(data))]


def train_test_mask(count: int, test_fraction: float = 0.2, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(count)
    test_count = max(1, int(count * test_fraction))
    mask = np.zeros(count, dtype=bool)
    mask[indices[:test_count]] = True
    return mask


def feature_names() -> list[str]:
    names = []
    for band, _, _ in BANDS:
        names.extend(
            [
                f"{band}_peak",
                f"{band}_mean",
                f"{band}_std",
                f"{band}_peak_above_med",
            ]
        )
    return names


def window_to_features(window: np.ndarray) -> list[float]:
    features = []
    for channel in window:
        features.extend(
            [
                float(np.max(channel)),
                float(np.mean(channel)),
                float(np.std(channel)),
                float(np.max(channel) - np.median(channel)),
            ]
        )
    return features
