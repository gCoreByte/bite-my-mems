from collections import deque


class Sensor:
    """Simple in-memory buffer for one slow sensor channel."""

    def __init__(self, sensor_id: str) -> None:
        self.ID = sensor_id
        self.last_read = []
        self.timestamps = []
        self._history = deque(maxlen=3)

    def save_data(self, value: float) -> None:
        self.last_read.append(value)

    def get_average(self) -> float | None:
        if not self.last_read:
            return None
        return round(sum(self.last_read) / len(self.last_read), 2)

    def get_smoothed(self) -> float | None:
        average = self.get_average()
        if average is not None:
            self._history.append(average)

        if not self._history:
            return None
        return round(sum(self._history) / len(self._history), 2)
