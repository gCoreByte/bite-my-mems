"""
Train a feature-based regressor for bite force from MEMS envelope features.

Usage:
    python train_features.py
    python train_features.py ./data/gathered_data/20260505_140101
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from recordings import (
    DATA_DIR,
    feature_names,
    find_recordings,
    load_force_windows,
    window_to_features,
)


MODEL_PATH = "bite_force_features.npz"
PLOT_PATH = "features_results.png"


def build_dataset(directory: str = str(DATA_DIR)) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    recordings = find_recordings(directory)
    if not recordings:
        sys.exit("No recordings found.")

    print(f"Found {len(recordings)} recording(s)\n")
    features: list[list[float]] = []
    forces: list[float] = []
    groups: list[str] = []

    for recording in recordings:
        windows, recording_forces = load_force_windows(recording)
        if not windows:
            print("    -> skipped (no events)")
            continue

        features.extend(window_to_features(window) for window in windows)
        forces.extend(recording_forces)
        groups.extend([recording.label] * len(windows))

    X = np.array(features, dtype=np.float32)
    y = np.array(forces, dtype=np.float32)
    g = np.array(groups)
    if len(X) == 0:
        sys.exit("No usable samples found.")

    print(f"\nTotal: {len(X)} events across {len(np.unique(g))} recordings")
    print(f"Force range: {y.min():.0f}-{y.max():.0f}")
    return X, y, g, feature_names()


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = np.where(train.std(axis=0) > 0, train.std(axis=0), 1.0)
    return (train - mean) / std, (test - mean) / std, mean, std


def fit_ridge(X_train: np.ndarray, y_train: np.ndarray) -> RidgeCV:
    model = RidgeCV(alphas=np.logspace(-3, 3, 13))
    model.fit(X_train, y_train)
    return model


def evaluate(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    if len(actual) < 2 or actual.std() == 0 or predicted.std() == 0:
        r = float("nan")
    else:
        r = float(np.corrcoef(actual, predicted)[0, 1])
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    return r, rmse


def leave_one_recording_out(X: np.ndarray, y: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.zeros_like(y)
    for label in np.unique(g):
        test_mask = g == label
        X_train_raw = X[~test_mask]
        X_test_raw = X[test_mask]
        X_train, X_test, _, _ = standardize(X_train_raw, X_test_raw)
        model = fit_ridge(X_train, y[~test_mask])
        predictions[test_mask] = model.predict(X_test)
    return y, predictions


def k_fold(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> tuple[np.ndarray, np.ndarray]:
    n_splits = min(n_splits, len(X))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    predictions = np.zeros_like(y)
    for train_idx, test_idx in kf.split(X):
        X_train, X_test, _, _ = standardize(X[train_idx], X[test_idx])
        model = fit_ridge(X_train, y[train_idx])
        predictions[test_idx] = model.predict(X_test)
    return y, predictions


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, tuple[float, float]]]:
    print("\nLeave-one-recording-out CV (Ridge):")
    actual_loro, pred_loro = leave_one_recording_out(X, y, g)
    r_loro, rmse_loro = evaluate(actual_loro, pred_loro)
    print(f"  r:    {r_loro:.3f}")
    print(f"  RMSE: {rmse_loro:.0f}")

    print("\n5-fold random CV (Ridge):")
    actual_kf, pred_kf = k_fold(X, y, n_splits=5)
    r_kf, rmse_kf = evaluate(actual_kf, pred_kf)
    print(f"  r:    {r_kf:.3f}")
    print(f"  RMSE: {rmse_kf:.0f}")

    X_std, _, mean, std = standardize(X, X)
    final = fit_ridge(X_std, y)
    print("\nFeature weights (full-fit Ridge, standardized inputs):")
    for index in np.argsort(np.abs(final.coef_))[::-1]:
        print(f"  {names[index]:25s}: {final.coef_[index]:+.3f}")

    np.savez(
        MODEL_PATH,
        coef=final.coef_,
        intercept=final.intercept_,
        mean=mean,
        std=std,
        feature_names=names,
    )
    print(f"\nSaved {MODEL_PATH}")

    metrics = {
        "loro": (r_loro, rmse_loro),
        "kfold": (r_kf, rmse_kf),
    }
    return actual_loro, pred_loro, actual_kf, pred_kf, metrics


def plot_results(
    actual_loro: np.ndarray,
    pred_loro: np.ndarray,
    actual_kf: np.ndarray,
    pred_kf: np.ndarray,
    metrics: dict[str, tuple[float, float]],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    r_loro, rmse_loro = metrics["loro"]
    axes[0].scatter(actual_loro, pred_loro, s=40, alpha=0.7)
    limits = [
        min(actual_loro.min(), pred_loro.min()),
        max(actual_loro.max(), pred_loro.max()),
    ]
    axes[0].plot(limits, limits, "k--", linewidth=1, alpha=0.4)
    axes[0].set_title(f"LORO r={r_loro:.3f}, RMSE={rmse_loro:.0f}")
    axes[0].set_xlabel("Actual force")
    axes[0].set_ylabel("Predicted force")

    r_kf, rmse_kf = metrics["kfold"]
    axes[1].scatter(actual_kf, pred_kf, s=40, alpha=0.7)
    limits = [
        min(actual_kf.min(), pred_kf.min()),
        max(actual_kf.max(), pred_kf.max()),
    ]
    axes[1].plot(limits, limits, "k--", linewidth=1, alpha=0.4)
    axes[1].set_title(f"5-fold r={r_kf:.3f}, RMSE={rmse_kf:.0f}")
    axes[1].set_xlabel("Actual force")
    axes[1].set_ylabel("Predicted force")

    residuals = pred_loro - actual_loro
    axes[2].hist(residuals, bins=max(5, len(residuals) // 3), edgecolor="black", alpha=0.7)
    axes[2].axvline(0, color="red", linestyle="--")
    axes[2].set_title("LORO residuals")
    axes[2].set_xlabel("Prediction error")

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.show()
    print(f"Saved {PLOT_PATH}")


def main() -> None:
    directory = sys.argv[1] if len(sys.argv) > 1 else str(DATA_DIR)
    X, y, g, names = build_dataset(directory)
    actual_loro, pred_loro, actual_kf, pred_kf, metrics = train_model(X, y, g, names)
    plot_results(actual_loro, pred_loro, actual_kf, pred_kf, metrics)


if __name__ == "__main__":
    main()
