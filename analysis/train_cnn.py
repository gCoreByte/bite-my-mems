"""
Train a compact 1D CNN to predict bite force from MEMS envelope windows.

Usage:
    python train_cnn.py
    python train_cnn.py ./data/gathered_data/20260505_140101
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from recordings import (
    BANDS,
    DATA_DIR,
    find_recordings,
    load_force_windows,
    normalize_windows_by_recording,
    train_test_mask,
)


BATCH_SIZE = 16
EPOCHS = 300
LEARNING_RATE = 1e-3
PATIENCE = 40
MODEL_PATH = "bite_force_cnn.keras"
PLOT_PATH = "cnn_results.png"


def build_model(channels: int = len(BANDS), length: int | None = None) -> keras.Model:
    # Keras Conv1D is channels-last: input shape is (batch, time, channels).
    # One channel per frequency band in BANDS — the CNN learns cross-band patterns over time.
    # length=None lets the same architecture accept any window length at inference.
    return keras.Sequential(
        [
            keras.Input(shape=(length, channels)),
            # Wider kernel (7) on layer 1 captures the broader bite envelope shape.
            layers.Conv1D(16, kernel_size=7, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling1D(2),
            # Narrower kernel (5) on top of pooled features for finer local structure.
            layers.Conv1D(16, kernel_size=5, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            # Global pool collapses the time axis → fixed-size vector regardless of input length.
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.5),
            layers.Dense(1),
            # Drop the trailing size-1 axis so output is shape (batch,) and matches scalar targets.
            layers.Reshape(()),
        ]
    )


def build_dataset(directory: str = str(DATA_DIR)) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    recordings = find_recordings(directory)
    if not recordings:
        raise Exception("No recordings found.")

    print(f"Found {len(recordings)} recording(s)\n")
    all_windows = []
    all_forces = []

    for recording in recordings:
        windows, forces = load_force_windows(recording)
        windows = normalize_windows_by_recording(windows)
        if not windows:
            raise Exception(f"No events found in recording {recording.label}.")

        all_windows.extend(windows)
        all_forces.extend(forces)


    # extract_window returns (channels, time); Keras Conv1D wants (time, channels).
    X = np.array(all_windows, dtype=np.float32).transpose(0, 2, 1)
    y_raw = np.array(all_forces, dtype=np.float32)
    # Normalize targets so MSE isn't dominated by absolute force magnitude.
    # max(std, 1.0) guards a degenerate constant-target split from divide-by-zero.
    y_mean = float(y_raw.mean())
    y_std = float(max(y_raw.std(), 1.0))
    y = (y_raw - y_mean) / y_std

    print(f"\nTotal events: {len(X)}")
    print(f"Force range: {y_raw.min():.0f}-{y_raw.max():.0f}")

    test_mask = train_test_mask(len(X))
    return X[~test_mask], y[~test_mask], X[test_mask], y[test_mask], y_raw[test_mask], {"y_mean": y_mean, "y_std": y_std}


def augment(inputs: tf.Tensor, _label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    # `inputs` is one unbatched sample shaped (time, channels) — axis 0 is time.
    # tf.data calls this per-example, so each gets its own shift/noise/scale draw.
    length = tf.shape(inputs)[0]
    # Shift up to ±5% of window length — simulates small bite-onset timing jitter.
    max_shift = tf.maximum(length // 20, 1)
    shift = tf.random.uniform([], -max_shift, max_shift + 1, dtype=tf.int32)
    shifted = tf.roll(inputs, shift=shift, axis=0)

    # Additive Gaussian noise + multiplicative gain mimic mic placement / level variation.
    noise = 0.1 * tf.random.normal(tf.shape(shifted))
    scale = 0.8 + 0.4 * tf.random.uniform([])
    return (shifted + noise) * scale, _label


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_test_raw: np.ndarray,
    target_norm: dict[str, float],
) -> tuple[dict[str, list[float]], np.ndarray, np.ndarray, float, float]:
    # Augment BEFORE batch so each sample's (time, channels) tensor is unbatched,
    # matching `augment`'s assumption that axis 0 is time. Shuffle before map so
    # different examples land in different batches each epoch.
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .shuffle(len(X_train))
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = build_model(length=X_train.shape[1])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, weight_decay=1e-3),
        loss="mse",
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=15, factor=0.5),
        keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        train_ds,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )

    normalized_predictions = model.predict(X_test, verbose=0).reshape(-1)
    predicted_force = normalized_predictions * target_norm["y_std"] + target_norm["y_mean"]
    actual_force = y_test_raw
    r = float(np.corrcoef(actual_force, predicted_force)[0, 1])
    rmse = float(np.sqrt(np.mean((actual_force - predicted_force) ** 2)))

    print(f"\nTEST RESULTS (n={len(actual_force)})")
    print(f"  r:    {r:.3f}")
    print(f"  RMSE: {rmse:.0f}")
    print(f"Saved {MODEL_PATH}")

    return history.history, actual_force, predicted_force, r, rmse


def plot_results(history: dict[str, list[float]], actual: np.ndarray, predicted: np.ndarray, r: float, rmse: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(history["loss"], label="Train", linewidth=0.8)
    axes[0].plot(history["val_loss"], label="Validation", linewidth=0.8)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE loss")
    axes[0].set_title("Training curves")
    axes[0].legend()

    axes[1].scatter(actual, predicted, s=40, alpha=0.7)
    limits = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    axes[1].plot(limits, limits, "k--", linewidth=1, alpha=0.4)
    axes[1].set_title(f"Test r={r:.3f}, RMSE={rmse:.0f}")
    axes[1].set_xlabel("Actual force")
    axes[1].set_ylabel("Predicted force")

    axes[2].hist(predicted - actual, bins=max(5, len(actual) // 3), edgecolor="black", alpha=0.7)
    axes[2].axvline(0, color="red", linestyle="--")
    axes[2].set_title("Residuals")
    axes[2].set_xlabel("Prediction error")

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.show()
    print(f"Saved {PLOT_PATH}")


def main() -> None:
    directory = sys.argv[1] if len(sys.argv) > 1 else str(DATA_DIR)
    X_train, y_train, X_test, y_test, y_test_raw, target_norm = build_dataset(directory)
    history, actual, predicted, r, rmse = train_model(X_train, y_train, X_test, y_test, y_test_raw, target_norm)
    plot_results(history, actual, predicted, r, rmse)


if __name__ == "__main__":
    main()
