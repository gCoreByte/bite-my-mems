"""3D waterfall plot of bite audio: 3 soft, 3 medium, 3 hard, 3 none with a 3 s window."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from recordings import (
    DATA_DIR,
    MEMS_FILE,
    MEMS_SAMPLE_RATE,
    MARKS_FILE,
    WEIGHTS_FILE,
    bandpass,
    load_log,
    load_mems,
    load_mic_calibration,
    recording_from_directory,
)
from scipy.ndimage import uniform_filter1d


WINDOW_SEC = 1.5
PLOT_SAMPLES = 600
PLOT_PATH = "bites_3d.png"

CATEGORIES = [
    ("soft", "20260508_151811_soft", "tab:green"),
    ("medium", "20260508_152325_medium", "tab:orange"),
    ("hard", "20260508_151630_hard", "tab:red"),
    ("none", "20260508_152628_none", "tab:gray"),
]


ENV_WINDOW_MS = 30
BAND_LOW = 300
BAND_HIGH = 5000


def extract_audio_window(directory: Path, mark_sec: float) -> tuple[np.ndarray, np.ndarray]:
    log = load_log(directory / WEIGHTS_FILE)
    mems = load_mems(directory / MEMS_FILE)
    time_scale = log["t_sec"].iloc[-1] / (len(mems) / MEMS_SAMPLE_RATE)

    recording = recording_from_directory(directory)
    calibration = load_mic_calibration(recording, samples=mems)
    filtered = bandpass(mems, BAND_LOW, BAND_HIGH)
    smooth_window = int(MEMS_SAMPLE_RATE * ENV_WINDOW_MS / 1000)
    envelope = np.sqrt(uniform_filter1d(filtered**2, size=smooth_window)) / calibration

    audio_t = mark_sec / time_scale
    start = int((audio_t - WINDOW_SEC) * MEMS_SAMPLE_RATE)
    end = int((audio_t + WINDOW_SEC) * MEMS_SAMPLE_RATE)
    if start < 0 or end > len(envelope):
        raise ValueError(f"Window out of range for mark {mark_sec} in {directory.name}")

    segment = envelope[start:end]
    times = np.linspace(-WINDOW_SEC, WINDOW_SEC, len(segment))
    return times, segment


def downsample(times: np.ndarray, values: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray]:
    if len(values) <= target:
        return times, values
    step = len(values) // target
    return times[::step][:target], values[::step][:target]


def main() -> None:
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    y_labels = []
    y_positions = []
    row = 0

    for category, recording_name, color in CATEGORIES:
        directory = base / recording_name
        marks = json.loads((directory / MARKS_FILE).read_text())["marks_sec"]
        if not marks:
            log = load_log(directory / WEIGHTS_FILE)
            duration = float(log["t_sec"].iloc[-1])
            marks = list(np.linspace(WINDOW_SEC + 1, duration - WINDOW_SEC - 1, 3))
        for index, mark in enumerate(marks[:3]):
            times, samples = extract_audio_window(directory, float(mark))
            times, samples = downsample(times, samples, PLOT_SAMPLES)
            ys = np.full_like(times, row, dtype=float)
            ax.plot(times, ys, samples, color=color, linewidth=0.7, alpha=0.9)
            y_labels.append(f"{category} #{index + 1}")
            y_positions.append(row)
            row += 1

    ax.set_xlabel("time relative to bite (s)")
    ax.set_ylabel("bite")
    ax.set_zlabel("bite envelope (× ambient noise floor)")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_title("Bite audio (±1.5 s) — 3 soft, 3 medium, 3 hard, 3 none")
    ax.view_init(elev=25, azim=-65)

    handles = [plt.Line2D([0], [0], color=color, label=category) for category, _, color in CATEGORIES]
    ax.legend(handles=handles, loc="upper left")

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.show()
    print(f"Saved {PLOT_PATH}")


if __name__ == "__main__":
    main()
