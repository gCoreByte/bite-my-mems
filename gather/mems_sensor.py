import threading

import numpy as np


class MEMSSensor:
    """Ring buffer and binary logger for MEMS audio samples."""

    SAMPLE_RATE = 22050
    SIGNAL_LOST_THRESHOLD = 1.0

    def __init__(self, buffer_seconds: int = 10) -> None:
        self.ring_buffer = np.zeros(self.SAMPLE_RATE * buffer_seconds, dtype=np.float32)
        self.write_pos = 0
        self.total_samples = 0
        self.lock = threading.Lock()
        self.log_file = None
        self.signal_lost = threading.Event()

    def append_samples(self, samples: np.ndarray) -> None:
        if np.mean(samples) < self.SIGNAL_LOST_THRESHOLD:
            self.signal_lost.set()
            return

        with self.lock:
            count = len(samples)
            end = self.write_pos + count
            size = len(self.ring_buffer)

            if end <= size:
                self.ring_buffer[self.write_pos:end] = samples
            else:
                first = size - self.write_pos
                self.ring_buffer[self.write_pos:] = samples[:first]
                self.ring_buffer[: count - first] = samples[first:]

            self.write_pos = (self.write_pos + count) % size
            self.total_samples += count

    def get_last_samples(self, count: int) -> np.ndarray:
        with self.lock:
            count = min(count, self.total_samples, len(self.ring_buffer))
            start = (self.write_pos - count) % len(self.ring_buffer)

            if start + count <= len(self.ring_buffer):
                return self.ring_buffer[start:start + count].copy()

            first = len(self.ring_buffer) - start
            return np.concatenate((self.ring_buffer[start:], self.ring_buffer[: count - first]))

    def initialize_logging(self, path: str) -> None:
        self.log_file = open(path, "wb")

    def log(self, samples: np.ndarray) -> None:
        if self.log_file is None:
            return
        self.log_file.write(samples.astype(np.uint16).tobytes())

    def close(self) -> None:
        self.log_file.flush()
        self.log_file.close()
        self.log_file = None
