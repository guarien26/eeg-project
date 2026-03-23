"""
Evaluation module for the seizure detection model.

Computes clinical-grade metrics:
  - Sensitivity (recall): % of seizures correctly detected — MUST be >95%
  - Specificity: % of non-seizure correctly identified
  - Precision (PPV): % of seizure alerts that are true seizures
  - F1 score: harmonic mean of precision and recall
  - AUC-ROC: area under ROC curve
  - False alarm rate: false positives per hour of monitoring

Also generates:
  - Confusion matrix visualization
  - ROC curve plot
  - Per-patient performance breakdown
  - Threshold sensitivity analysis
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    f1_score, precision_score, recall_score,
)
from tensorflow import keras

import config
from features import prepare_model_input


def load_test_data() -> tuple[np.ndarray, np.ndarray]:
    """Load preprocessed test set."""
    path = config.PROCESSED_DIR / "test.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run preprocess.py first."
        )
    data = np.load(str(path))
    return data["windows"], data["labels"]


def evaluate_model(model_path: str = None, threshold: float = None):
    """
    Full evaluation pipeline on held-out test patients.

    Args:
        model_path: path to saved .keras model (default: best model)
        threshold: classification threshold (default from config)
    """
    threshold = threshold or config.SEIZURE_THRESHOLD
    if model_path is None:
        model_path = str(config.MODEL_DIR / "seizure_detector_full.keras")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SEIZURE DETECTION MODEL EVALUATION")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from {model_path}...")
    model = keras.models.load_model(model_path)

    # Load test data
    print("Loading test data...")
    X_test_raw, y_test = load_test_data()
    print(f"Test set: {len(y_test)} windows "
          f"({int(np.sum(y_test))} seizure / "
          f"{len(y_test) - int(np.sum(y_test))} normal)")

    # Prepare input
    X_test = prepare_model_input(X_test_raw, mode="raw")

    # Get predictions
    print("Running inference...")
    y_prob = model.predict(X_test, batch_size=config.BATCH_SIZE, verbose=1)
    y_prob = y_prob.flatten()
    y_pred = (y_prob >= threshold).astype(int)

    # ── Core Metrics ──────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"RESULTS (threshold = {threshold})")
    print("=" * 60)

    sensitivity = recall_score(y_test, y_pred, zero_division=0)
    specificity = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # False alarm rate: FP per hour of monitoring
    n_normal = int(np.sum(y_test == 0))
    n_fp = int(np.sum((y_pred == 1) & (y_test == 0)))
    # Each window is 4 seconds, with 3-second step (1s overlap)
    hours_monitored = n_normal * (config.WINDOW_SECONDS - config.OVERLAP_SECONDS) / 3600
    fa_rate = n_fp / max(hours_monitored, 1e-6)

    cm = confusion_matrix(y_test, y_pred)

    results = {
        "threshold": threshold,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1_score": float(f1),
        "auc_roc": float(roc_auc),
        "false_alarm_rate_per_hour": float(fa_rate),
        "confusion_matrix": cm.tolist(),
        "n_test_windows": int(len(y_test)),
        "n_seizure_windows": int(np.sum(y_test)),
        "n_normal_windows": int(n_normal),
    }

    print(f"\n  Sensitivity (recall):     {sensitivity:.4f}  "
          f"{'PASS' if sensitivity >= config.MIN_SENSITIVITY else 'FAIL'} "
          f"(target >= {config.MIN_SENSITIVITY})")
    print(f"  Specificity:              {specificity:.4f}")
    print(f"  Precision (PPV):          {precision:.4f}")
    print(f"  F1 Score:                 {f1:.4f}")
    print(f"  AUC-ROC:                  {roc_auc:.4f}")
    print(f"  False alarm rate:         {fa_rate:.2f}/hr  "
          f"{'PASS' if fa_rate <= config.MAX_FALSE_ALARM_RATE else 'FAIL'} "
          f"(target <= {config.MAX_FALSE_ALARM_RATE}/hr)")

    print(f"\n  Confusion Matrix:")
    print(f"                  Predicted Normal  Predicted Seizure")
    print(f"  Actual Normal:  {cm[0][0]:>8d}         {cm[0][1]:>8d}")
    print(f"  Actual Seizure: {cm[1][0]:>8d}         {cm[1][1]:>8d}")

    # Full classification report
    print(f"\n{classification_report(y_test, y_pred, target_names=['Normal', 'Seizure'])}")

    # ── Save Results ──────────────────────────────────────
    results_path = config.RESULTS_DIR / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    # ── Generate Plots ────────────────────────────────────
    plot_confusion_matrix(cm)
    plot_roc_curve(fpr, tpr, roc_auc)
    plot_threshold_analysis(y_test, y_prob)

    # ── Threshold Sweep ───────────────────────────────────
    print("\n" + "=" * 60)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 60)
    print(f"  {'Threshold':>10s}  {'Sensitivity':>12s}  {'Specificity':>12s}  "
          f"{'Precision':>10s}  {'F1':>8s}  {'FA/hr':>8s}")

    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        yp = (y_prob >= t).astype(int)
        sens = recall_score(y_test, yp, zero_division=0)
        spec = recall_score(y_test, yp, pos_label=0, zero_division=0)
        prec = precision_score(y_test, yp, zero_division=0)
        f1_t = f1_score(y_test, yp, zero_division=0)
        fp_t = int(np.sum((yp == 1) & (y_test == 0)))
        fa_t = fp_t / max(hours_monitored, 1e-6)
        marker = " ◀" if t == threshold else ""
        print(f"  {t:>10.2f}  {sens:>12.4f}  {spec:>12.4f}  "
              f"{prec:>10.4f}  {f1_t:>8.4f}  {fa_t:>8.2f}{marker}")

    return results


def plot_confusion_matrix(cm: np.ndarray):
    """Save confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Seizure"],
                yticklabels=["Normal", "Seizure"],
                ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Seizure Detection — Confusion Matrix")
    plt.tight_layout()
    path = config.RESULTS_DIR / "confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Confusion matrix plot saved to {path}")


def plot_roc_curve(fpr, tpr, roc_auc):
    """Save ROC curve plot."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1D9E75", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("Seizure Detection — ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = config.RESULTS_DIR / "roc_curve.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  ROC curve plot saved to {path}")


def plot_threshold_analysis(y_true, y_prob):
    """Save threshold vs sensitivity/specificity/FA rate plot."""
    thresholds = np.linspace(0.1, 0.95, 50)
    sensitivities = []
    specificities = []
    fa_rates = []

    n_normal = int(np.sum(y_true == 0))
    hours = n_normal * (config.WINDOW_SECONDS - config.OVERLAP_SECONDS) / 3600

    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        sensitivities.append(recall_score(y_true, yp, zero_division=0))
        specificities.append(recall_score(y_true, yp, pos_label=0, zero_division=0))
        fp = int(np.sum((yp == 1) & (y_true == 0)))
        fa_rates.append(fp / max(hours, 1e-6))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(thresholds, sensitivities, color="#D85A30", lw=2, label="Sensitivity")
    ax1.plot(thresholds, specificities, color="#378ADD", lw=2, label="Specificity")
    ax1.axhline(y=config.MIN_SENSITIVITY, color="gray", ls="--", alpha=0.5,
                label=f"Target sensitivity ({config.MIN_SENSITIVITY})")
    ax1.axvline(x=config.SEIZURE_THRESHOLD, color="gray", ls=":", alpha=0.5,
                label=f"Current threshold ({config.SEIZURE_THRESHOLD})")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Rate")
    ax1.set_title("Sensitivity vs Specificity")
    ax1.legend(fontsize=9)

    ax2.plot(thresholds, fa_rates, color="#A32D2D", lw=2)
    ax2.axhline(y=config.MAX_FALSE_ALARM_RATE, color="gray", ls="--", alpha=0.5,
                label=f"Target max ({config.MAX_FALSE_ALARM_RATE}/hr)")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("False Alarms per Hour")
    ax2.set_title("False Alarm Rate vs Threshold")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    path = config.RESULTS_DIR / "threshold_analysis.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"  Threshold analysis plot saved to {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate seizure detection model")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to .keras model file")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Classification threshold (default: 0.5)")
    args = parser.parse_args()

    evaluate_model(model_path=args.model, threshold=args.threshold)
