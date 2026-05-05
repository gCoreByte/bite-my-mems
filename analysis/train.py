"""
Train a simple force regressor from MEMS envelope features.

Usage:
    python train.py
    python train.py ./data/gathered_data/20260505_140101
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

from recordings import (
    DATA_DIR,
    feature_names,
    find_recordings,
    load_force_windows,
    normalize_windows_by_recording,
    train_test_mask,
    window_to_features,
)


MODEL_PATH = "bite_force_model.npz"
PLOT_PATH = "training_results.png"


def remove_outliers(windows: list[np.ndarray], forces: list[float]) -> tuple[list[np.ndarray], np.ndarray]:
    forces_array = np.array(forces, dtype=np.float32)
    q1, q3 = np.percentile(forces_array, [25, 75])
    iqr = q3 - q1
    keep = (forces_array > 0) & (forces_array >= q1 - 3 * iqr) & (forces_array <= q3 + 3 * iqr)

    removed = int(np.sum(~keep))
    if removed:
        kept = forces_array[keep]
        print(f"Outlier removal: dropped {removed}/{len(forces_array)} events ({kept.min():.0f}-{kept.max():.0f})")

    return [window for window, should_keep in zip(windows, keep) if should_keep], forces_array[keep]


def build_dataset(directory: str = str(DATA_DIR)) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    recordings = find_recordings(directory)
    if not recordings:
        sys.exit("No recordings found.")

    print(f"Found {len(recordings)} recording(s)\n")
    all_windows = []
    all_forces = []

    for recording in recordings:
        windows, forces = load_force_windows(recording)
        windows = normalize_windows_by_recording(windows)
        if not windows:
            print("    -> skipped (no events)")
            continue

        all_windows.extend(windows)
        all_forces.extend(forces)

    if len(all_windows) < 10:
        sys.exit(f"\nOnly {len(all_windows)} usable events. Collect at least 10.")

    all_windows, forces = remove_outliers(all_windows, all_forces)
    names = feature_names()
    X = np.array([window_to_features(window) for window in all_windows], dtype=np.float32)
    y = forces.astype(np.float32)

    print(f"\nTotal events: {len(X)}")
    print(f"Force range: {y.min():.0f}-{y.max():.0f}")
    print(f"Features: {len(names)}")

    print("\nPer-feature correlation:")
    for index, name in enumerate(names):
        correlation = np.corrcoef(X[:, index], y)[0, 1]
        print(f"  {name:25s}: {correlation:+.3f}")

    test_mask = train_test_mask(len(X))
    return X[~test_mask], y[~test_mask], X[test_mask], y[test_mask], names


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = np.where(train.std(axis=0) > 0, train.std(axis=0), 1.0)
    return (train - mean) / std, (test - mean) / std, mean, std


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    X_with_bias = np.hstack([X, np.ones((len(X), 1))])
    penalty = alpha * np.eye(X_with_bias.shape[1])
    penalty[-1, -1] = 0
    return np.linalg.solve(X_with_bias.T @ X_with_bias + penalty, X_with_bias.T @ y)


def leave_one_out_score(X: np.ndarray, y: np.ndarray, alpha: float) -> float:
    predictions = np.zeros(len(X))
    for index in range(len(X)):
        keep = np.ones(len(X), dtype=bool)
        keep[index] = False
        weights = ridge_fit(X[keep], y[keep], alpha)
        predictions[index] = np.append(X[index], 1) @ weights

    return float(np.corrcoef(y, predictions)[0, 1])


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    X_train, X_test, mean, std = standardize(X_train, X_test)

    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    scores = {alpha: leave_one_out_score(X_train, y_train, alpha) for alpha in alphas}
    best_alpha = max(scores, key=scores.get)
    print(f"\nBest alpha: {best_alpha}  LOO r: {scores[best_alpha]:.3f}")

    weights = ridge_fit(X_train, y_train, best_alpha)
    predictions = np.hstack([X_test, np.ones((len(X_test), 1))]) @ weights

    r = float(np.corrcoef(y_test, predictions)[0, 1])
    rmse = float(np.sqrt(np.mean((y_test - predictions) ** 2)))

    print("\nFeature weights:")
    for name, weight in zip(names, weights[:-1]):
        print(f"  {name:25s}: {weight:+.1f}")

    print(f"\nTEST RESULTS (n={len(y_test)})")
    print(f"  r:    {r:.3f}")
    print(f"  RMSE: {rmse:.0f}")

    np.savez(MODEL_PATH, weights=weights, mean=mean, std=std, feature_names=names)
    print(f"Saved {MODEL_PATH}")

    loo_predictions = np.zeros(len(X_train))
    for index in range(len(X_train)):
        keep = np.ones(len(X_train), dtype=bool)
        keep[index] = False
        weights_loo = ridge_fit(X_train[keep], y_train[keep], best_alpha)
        loo_predictions[index] = np.append(X_train[index], 1) @ weights_loo

    return y_test, predictions, y_train, loo_predictions, r, rmse


def plot_results(actual: np.ndarray, predicted: np.ndarray, y_train: np.ndarray, y_loo: np.ndarray, r: float, rmse: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(y_train, y_loo, s=30, alpha=0.6)
    axes[0].set_title("Train leave-one-out")
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")

    axes[1].scatter(actual, predicted, s=40, alpha=0.7, color="darkorange")
    limits = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    axes[1].plot(limits, limits, "k--", linewidth=1, alpha=0.4)
    axes[1].set_title(f"Test r={r:.3f}, RMSE={rmse:.0f}")
    axes[1].set_xlabel("Actual force")
    axes[1].set_ylabel("Predicted force")

    axes[2].hist(predicted - actual, bins=max(5, len(actual) // 3), edgecolor="black", alpha=0.7)
    axes[2].axvline(0, color="red", linestyle="--")
    axes[2].set_title("Test residuals")
    axes[2].set_xlabel("Prediction error")

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.show()
    print(f"Saved {PLOT_PATH}")


def main() -> None:
    directory = sys.argv[1] if len(sys.argv) > 1 else str(DATA_DIR)
    X_train, y_train, X_test, y_test, names = build_dataset(directory)
    actual, predicted, y_train, y_loo, r, rmse = train_model(X_train, y_train, X_test, y_test, names)
    plot_results(actual, predicted, y_train, y_loo, r, rmse)


if __name__ == "__main__":
    main()
