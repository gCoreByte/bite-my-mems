"""
Train a bite/no-bite classifier from MEMS envelope features.

Usage:
    python train_binary.py
    python train_binary.py ./data/gathered_data/20260505_140101
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

from recordings import (
    DATA_DIR,
    feature_names,
    find_recordings,
    load_binary_windows,
    train_test_mask,
    window_to_features,
)


MODEL_PATH = "bite_classifier.npz"
PLOT_PATH = "binary_results.png"


def build_dataset(directory: str = str(DATA_DIR)) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    recordings = find_recordings(directory)
    if not recordings:
        sys.exit("No recordings found.")

    print(f"Found {len(recordings)} recording(s)\n")
    features = []
    labels = []

    for recording in recordings:
        positives, negatives = load_binary_windows(recording)
        if not positives:
            print("    -> skipped (no bites)")
            continue

        features.extend(window_to_features(window) for window in positives)
        labels.extend([1] * len(positives))
        features.extend(window_to_features(window) for window in negatives)
        labels.extend([0] * len(negatives))

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=int)
    if len(X) == 0:
        sys.exit("No usable samples found.")

    print(f"\nTotal: {len(X)} samples ({np.sum(y == 1)} bite, {np.sum(y == 0)} no-bite)")

    test_mask = train_test_mask(len(X))
    return X[~test_mask], y[~test_mask], X[test_mask], y[test_mask], feature_names()


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = np.where(train.std(axis=0) > 0, train.std(axis=0), 1.0)
    return (train - mean) / std, (test - mean) / std, mean, std


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    names: list[str],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    X_train, X_test, mean, std = standardize(X_train, X_test)
    class_counts = np.bincount(y_train, minlength=2)
    cv = min(5, int(class_counts.min()))
    if cv < 2:
        sys.exit("Need at least 2 bite and 2 no-bite samples in the training split.")

    model = LogisticRegressionCV(cv=cv, max_iter=1000, scoring="accuracy")
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)[:, 1]
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    test_auc = roc_auc_score(y_test, test_prob)

    print("\nRESULTS")
    print(f"  Train accuracy: {train_acc:.1%}")
    print(f"  Test accuracy:  {test_acc:.1%}")
    print(f"  Test AUC:       {test_auc:.3f}")
    print("\nTest classification report:")
    print(classification_report(y_test, test_pred, target_names=["no-bite", "bite"]))

    print("Feature weights:")
    for index in np.argsort(np.abs(model.coef_[0]))[::-1]:
        print(f"  {names[index]:25s}: {model.coef_[0][index]:+.3f}")

    np.savez(
        MODEL_PATH,
        coef=model.coef_,
        intercept=model.intercept_,
        mean=mean,
        std=std,
        feature_names=names,
    )
    print(f"\nSaved {MODEL_PATH}")

    return test_pred, test_prob, test_acc, test_auc


def plot_results(y_test: np.ndarray, test_pred: np.ndarray, test_prob: np.ndarray, accuracy: float, auc: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    matrix = confusion_matrix(y_test, test_pred)
    axes[0].imshow(matrix, cmap="Blues", aspect="auto")
    for row in range(2):
        for col in range(2):
            text_color = "white" if matrix[row, col] > matrix.max() / 2 else "black"
            axes[0].text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=18, color=text_color)
    axes[0].set_xticks([0, 1], ["no-bite", "bite"])
    axes[0].set_yticks([0, 1], ["no-bite", "bite"])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title(f"Confusion matrix (acc={accuracy:.1%})")

    bins = np.linspace(0, 1, 20)
    axes[1].hist(test_prob[y_test == 0], bins=bins, alpha=0.6, label="no-bite", edgecolor="black")
    axes[1].hist(test_prob[y_test == 1], bins=bins, alpha=0.6, label="bite", edgecolor="black")
    axes[1].axvline(0.5, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Predicted P(bite)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Score distribution (AUC={auc:.3f})")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.show()
    print(f"Saved {PLOT_PATH}")


def main() -> None:
    directory = sys.argv[1] if len(sys.argv) > 1 else str(DATA_DIR)
    X_train, y_train, X_test, y_test, names = build_dataset(directory)
    predictions, probabilities, accuracy, auc = train_model(X_train, y_train, X_test, y_test, names)
    plot_results(y_test, predictions, probabilities, accuracy, auc)


if __name__ == "__main__":
    main()
