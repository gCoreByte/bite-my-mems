"""Live bite-force predictor — connects to the device and serves a browser dashboard.

Reads load cell + MEMS audio the same way as gather/main.py, detects bite events
from the load cell, and runs the trained CNN on the matching 2-second audio
window. Streams the live weight stream and predicted force to a local web page.

Usage:
    python prediction/main.py
"""

import json
import statistics
import sys
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gather"))
sys.path.insert(0, str(ROOT / "analysis"))

from device import Device, SLOW_PACKET  # noqa: E402
from load_cell import LoadCell  # noqa: E402
from mems_sensor import MEMSSensor  # noqa: E402
from recordings import (  # noqa: E402
    BANDS,
    ENV_SAMPLE_COUNT,
    ENV_WINDOW_MS,
    MEMS_SAMPLE_RATE,
    WINDOW_SEC,
    bandpass,
    find_recordings,
    load_force_windows,
    mic_calibration_factor,
    train_test_mask,
)


MODEL_PATH = ROOT / "bite_force_cnn.keras"
TEMPLATE_PATH = Path(__file__).resolve().parent / "index.html"
BASELINE_DURATION_SEC = 1.0
PORT = 8765

WEIGHT_BUFFER_SECONDS = 8
PREDICTION_HISTORY = 10
BITE_THRESHOLD_GRAMS = 1500.0
BITE_REFRACTORY_SEC = 1.2
POST_BITE_WAIT_SEC = WINDOW_SEC + 0.05
# Extra audio fed to the bandpass filter before the actual prediction window,
# so the IIR filter has time to settle. Without this pre-roll, every window's
# envelope is dominated by the filter transient and predictions collapse to y_mean.
FILTER_PREROLL_SEC = 1.0
TOTAL_AUDIO_SEC = FILTER_PREROLL_SEC + 2 * WINDOW_SEC + POST_BITE_WAIT_SEC


def compute_normalization_stats() -> tuple[np.ndarray, np.ndarray, float, float]:
    """Recompute the per-channel input stats and target stats from training data.

    Mirrors build_dataset() in analysis/train_cnn.py so live predictions get the
    same normalization the model was trained with.
    """
    recordings = find_recordings()
    if not recordings:
        raise RuntimeError("No recordings found — cannot derive normalization stats.")

    all_windows: list[np.ndarray] = []
    all_forces: list[float] = []
    for recording in recordings:
        windows, forces = load_force_windows(recording)
        all_windows.extend(windows)
        all_forces.extend(forces)

    X = np.log1p(np.array(all_windows, dtype=np.float32).transpose(0, 2, 1))
    y_raw = np.array(all_forces, dtype=np.float32)

    test_mask = train_test_mask(len(X))
    X_train = X[~test_mask]
    y_train = y_raw[~test_mask]

    x_mean = X_train.mean(axis=(0, 1), keepdims=True)
    x_std = X_train.std(axis=(0, 1), keepdims=True)
    x_std = np.where(x_std > 0, x_std, 1.0)
    y_mean = float(y_train.mean())
    y_std = float(max(y_train.std(), 1.0))
    return x_mean, x_std, y_mean, y_std


def envelope_from_audio(audio: np.ndarray, valid_start_sec: float, valid_length_sec: float) -> np.ndarray:
    """Compute the multi-band envelope window the CNN expects, shape (channels, ENV_SAMPLE_COUNT).

    `audio` is longer than the actual prediction window — we filter the full clip
    so bandpass transients live in the discarded pre-roll, then slice out the
    centered window before resampling.
    """
    smooth = int(MEMS_SAMPLE_RATE * ENV_WINDOW_MS / 1000)
    start = int(valid_start_sec * MEMS_SAMPLE_RATE)
    length = int(valid_length_sec * MEMS_SAMPLE_RATE)
    channels = []
    for _, low, high in BANDS:
        filtered = bandpass(audio, low, high)
        envelope = np.sqrt(uniform_filter1d(filtered**2, size=smooth))
        segment = envelope[start : start + length]
        channels.append(
            np.interp(
                np.linspace(0, 1, ENV_SAMPLE_COUNT),
                np.linspace(0, 1, len(segment)),
                segment,
            )
        )
    return np.stack(channels, axis=0).astype(np.float32)


class Predictor:
    def __init__(
        self,
        model: keras.Model,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: float,
        y_std: float,
        mic_calibration: float,
    ) -> None:
        self.model = model
        self.x_mean = x_mean[0]
        self.x_std = x_std[0]
        self.y_mean = y_mean
        self.y_std = y_std
        self.mic_calibration = mic_calibration

    def predict(self, audio: np.ndarray, valid_start_sec: float, valid_length_sec: float) -> float:
        envelope = envelope_from_audio(audio, valid_start_sec, valid_length_sec) / self.mic_calibration
        X_log = np.log1p(envelope.T)
        X = (X_log - self.x_mean) / self.x_std
        normalized = float(self.model.predict(X[np.newaxis, ...], verbose=0)[0])
        force = normalized * self.y_std + self.y_mean
        print(
            f"[predictor] env(raw): max={envelope.max():.2f} mean={envelope.mean():.2f} | "
            f"log1p: max={X_log.max():.2f} mean={X_log.mean():.2f} | "
            f"norm: min={X.min():.2f} max={X.max():.2f} | "
            f"z_pred={normalized:+.3f} -> {force:.0f}g"
        )
        return force


class State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.weights: deque[tuple[float, float]] = deque()
        self.predictions: deque[dict] = deque(maxlen=PREDICTION_HISTORY)
        self.last_bite_time: float | None = None
        self.connection_lost = False
        self.lost_reason: str | None = None

    def add_weights(self, samples: list[tuple[float, float]]) -> None:
        with self.lock:
            self.weights.extend(samples)
            cutoff = samples[-1][0] - WEIGHT_BUFFER_SECONDS
            while self.weights and self.weights[0][0] < cutoff:
                self.weights.popleft()

    def add_prediction(self, t: float, predicted: float, measured: float) -> None:
        with self.lock:
            self.predictions.appendleft(
                {"t": t, "predicted": round(predicted, 1), "measured": round(measured, 1)}
            )

    def mark_lost(self, reason: str) -> None:
        with self.lock:
            self.connection_lost = True
            self.lost_reason = reason

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "weights": list(self.weights),
                "predictions": list(self.predictions),
                "buffer_seconds": WEIGHT_BUFFER_SECONDS,
                "connection_lost": self.connection_lost,
                "lost_reason": self.lost_reason,
            }

    def weights_snapshot(self) -> list[tuple[float, float]]:
        with self.lock:
            return list(self.weights)


def detrended_peak_force(weights: list[tuple[float, float]], peak_t: float, half_window: float = 0.5) -> float:
    """max(|weight - rolling_mean|) within ±half_window of peak_t — matches training labels."""
    if len(weights) < 10:
        return 0.0
    times = np.array([t for t, _ in weights])
    values = np.array([w for _, w in weights])
    rolling = uniform_filter1d(values, size=min(200, len(values)))
    detrended = np.abs(values - rolling)
    mask = (times >= peak_t - half_window) & (times <= peak_t + half_window)
    return float(np.max(detrended[mask])) if mask.any() else 0.0


def device_loop(
    device: Device,
    predictor: Predictor,
    state: State,
    stop: threading.Event,
) -> None:
    load_cell: LoadCell = device.sensors[0]
    start = time.monotonic()
    pending_bite: dict | None = None

    while not stop.is_set():
        try:
            magic = device.read_line()
        except RuntimeError as error:
            state.mark_lost(str(error))
            stop.set()
            return

        if device.mems_sensor.signal_lost.is_set():
            state.mark_lost("MEMS signal lost.")
            stop.set()
            return

        now_rel = time.monotonic() - start

        if magic == SLOW_PACKET:
            samples: list[tuple[float, float]] = []
            for index, weight in enumerate(load_cell.last_read):
                # Spread the batch evenly between previous read and now.
                offset = (index + 1) / len(load_cell.last_read) * 0.05
                samples.append((now_rel - 0.05 + offset, float(weight)))
            if samples:
                state.add_weights(samples)
                peak_t, peak_w = max(samples, key=lambda pair: abs(pair[1]))
                if abs(peak_w) >= BITE_THRESHOLD_GRAMS:
                    if (
                        state.last_bite_time is None
                        or peak_t - state.last_bite_time >= BITE_REFRACTORY_SEC
                    ):
                        state.last_bite_time = peak_t
                        pending_bite = {"t": peak_t, "wait_until": peak_t + POST_BITE_WAIT_SEC}

        if pending_bite is not None and now_rel >= pending_bite["wait_until"]:
            target_samples = int(MEMS_SAMPLE_RATE * TOTAL_AUDIO_SEC)
            audio = device.mems_sensor.get_last_samples(target_samples)
            if len(audio) >= int(target_samples * 0.95):
                audio = audio - audio.mean()
                # Audio buffer covers [now - TOTAL_AUDIO_SEC, now]. The bite was at
                # peak_t ≈ now - POST_BITE_WAIT_SEC, so the centered 2 sec prediction
                # window starts FILTER_PREROLL_SEC into the buffer.
                try:
                    predicted = predictor.predict(audio, FILTER_PREROLL_SEC, 2 * WINDOW_SEC)
                    measured = detrended_peak_force(state.weights_snapshot(), pending_bite["t"])
                    state.add_prediction(pending_bite["t"], predicted, measured)
                except Exception as error:  # noqa: BLE001
                    print(f"[predictor] prediction failed: {error}")
            pending_bite = None


def make_handler(state: State, template: bytes) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args, **_kwargs) -> None:  # silence default logging
            pass

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(template)))
                self.end_headers()
                self.wfile.write(template)
                return
            if self.path == "/data":
                body = json.dumps(state.snapshot()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(404)
            self.end_headers()

    return Handler


def collect_baseline(device: Device, duration: float = BASELINE_DURATION_SEC) -> float:
    load_cell: LoadCell = device.sensors[0]
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


def main() -> None:
    print(f"Loading model from {MODEL_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}.")
    model = keras.models.load_model(MODEL_PATH)

    print("Recomputing training normalization stats...")
    x_mean, x_std, y_mean, y_std = compute_normalization_stats()
    print(f"  y_mean={y_mean:.1f}, y_std={y_std:.1f}")
    print(f"  x_mean(per channel)={x_mean[0, 0].tolist()}")
    print(f"  x_std(per channel)={x_std[0, 0].tolist()}")

    template = TEMPLATE_PATH.read_bytes()

    load_cell = LoadCell("weight")
    mems = MEMSSensor()
    device = Device(load_cell, mems_sensor=mems, verbose=False)

    print("\nRestarting device...")
    device.reset()
    time.sleep(5)
    device.resync()

    print(f"\nCollecting baseline for {BASELINE_DURATION_SEC:.1f}s — keep the jaw still and unloaded.")
    baseline = collect_baseline(device)
    load_cell.baseline = baseline
    load_cell._history.clear()
    print(f"[LoadCell] Baseline: {baseline:.2f}")

    baseline_audio = mems.get_last_samples(int(MEMSSensor.SAMPLE_RATE * BASELINE_DURATION_SEC))
    baseline_audio = baseline_audio - baseline_audio.mean()
    mic_calibration = mic_calibration_factor(baseline_audio, baseline_sec=BASELINE_DURATION_SEC)
    print(f"[MEMS] Mic calibration: {mic_calibration:.2f} (training values were ~15-22)")

    predictor = Predictor(model, x_mean, x_std, y_mean, y_std, mic_calibration)
    state = State()
    stop = threading.Event()

    reader = threading.Thread(target=device_loop, args=(device, predictor, state, stop), daemon=True)
    reader.start()

    server = ThreadingHTTPServer(("127.0.0.1", PORT), make_handler(state, template))
    url = f"http://127.0.0.1:{PORT}/"
    print(f"\nServing dashboard at {url}")
    threading.Timer(0.7, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop.set()
        server.server_close()
        reader.join(timeout=2)
        device.close()


if __name__ == "__main__":
    main()
