"""
Training pipeline for CNN-LSTM seizure detection model.

Usage:
  # Step 1: Download CHB-MIT dataset
  #   wget -r -np https://physionet.org/files/chbmit/1.0.0/
  #   Move to data/chb-mit-scalp-eeg-database-1.0.0/

  # Step 2: Preprocess (creates train/val/test .npz files)
  python preprocess.py

  # Step 3: Train the model
  python train.py

  # Step 4: Evaluate on test set
  python evaluate.py
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime

import config
from features import prepare_model_input
from model import (
    build_cnn_lstm_model,
    build_lightweight_model,
    get_training_callbacks,
    compute_class_weights,
)


def load_processed_data(split: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load preprocessed .npz data files.

    Args:
        split: "train", "val", or "test"

    Returns:
        (windows, labels)
    """
    path = config.PROCESSED_DIR / f"{split}.npz"
    if not path.exists():
        print(f"Error: {path} not found. Run preprocess.py first.")
        sys.exit(1)

    data = np.load(str(path))
    windows = data["windows"]
    labels = data["labels"]
    print(f"Loaded {split}: {len(labels)} windows "
          f"({int(np.sum(labels))} seizure / "
          f"{len(labels) - int(np.sum(labels))} normal)")
    return windows, labels


def augment_seizure_windows(windows: np.ndarray,
                            labels: np.ndarray,
                            target_ratio: float = 0.2) -> tuple:
    """
    Augment seizure windows via time-domain transformations to
    reduce class imbalance.

    Augmentation methods:
      1. Gaussian noise addition
      2. Time shifting (circular roll)
      3. Amplitude scaling
      4. Channel dropout (zero one channel randomly)

    Args:
        windows: shape (N, n_channels, window_samples)
        labels: shape (N,)
        target_ratio: target seizure percentage after augmentation

    Returns:
        (augmented_windows, augmented_labels)
    """
    seizure_mask = labels == 1
    n_seizure = int(np.sum(seizure_mask))
    n_normal = len(labels) - n_seizure

    if n_seizure == 0:
        print("Warning: No seizure windows to augment.")
        return windows, labels

    current_ratio = n_seizure / len(labels)
    if current_ratio >= target_ratio:
        print(f"Seizure ratio {current_ratio:.1%} already >= target {target_ratio:.1%}")
        return windows, labels

    # How many augmented seizure windows do we need?
    # target_ratio = (n_seizure + n_aug) / (n_total + n_aug)
    n_aug = int((target_ratio * n_normal - n_seizure) / (1 - target_ratio))
    n_aug = max(0, n_aug)

    print(f"Augmenting: {n_seizure} seizure windows → "
          f"{n_seizure + n_aug} ({n_aug} synthetic)")

    seizure_windows = windows[seizure_mask]
    augmented = []

    rng = np.random.default_rng(42)
    for i in range(n_aug):
        # Pick a random seizure window to augment
        idx = rng.integers(0, n_seizure)
        w = seizure_windows[idx].copy()

        # Apply random augmentation(s)
        aug_type = rng.integers(0, 4)

        if aug_type == 0:
            # Gaussian noise (SNR ~20 dB)
            noise_std = np.std(w) * 0.1
            w += rng.normal(0, noise_std, w.shape)

        elif aug_type == 1:
            # Time shift (roll by ±50 samples = ±200ms)
            shift = rng.integers(-50, 50)
            w = np.roll(w, shift, axis=-1)

        elif aug_type == 2:
            # Amplitude scaling (0.8x to 1.2x)
            scale = rng.uniform(0.8, 1.2)
            w *= scale

        elif aug_type == 3:
            # Channel dropout (zero one random channel)
            ch = rng.integers(0, w.shape[0])
            w[ch] = 0.0

        augmented.append(w)

    if augmented:
        aug_windows = np.array(augmented)
        aug_labels = np.ones(n_aug, dtype=labels.dtype)

        windows = np.concatenate([windows, aug_windows], axis=0)
        labels = np.concatenate([labels, aug_labels], axis=0)

        # Shuffle
        perm = rng.permutation(len(labels))
        windows = windows[perm]
        labels = labels[perm]

    new_ratio = np.sum(labels) / len(labels)
    print(f"After augmentation: {len(labels)} windows "
          f"({int(np.sum(labels))} seizure = {new_ratio:.1%})")

    return windows, labels


def train(model_type: str = "full", use_augmentation: bool = True):
    """
    Full training pipeline.

    Args:
        model_type: "full" for CNN-LSTM, "lite" for lightweight
        use_augmentation: whether to augment seizure windows
    """
    print("=" * 60)
    print(f"SEIZURE DETECTION MODEL TRAINING")
    print(f"Model: {model_type}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)

    # Create output directories
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # ── Load data ──────────────────────────────────────────
    print("\n[1/5] Loading preprocessed data...")
    X_train_raw, y_train = load_processed_data("train")
    X_val_raw, y_val = load_processed_data("val")

    if len(X_train_raw) == 0 or len(X_val_raw) == 0:
        print("Error: Empty dataset. Check preprocessing output.")
        return

    # ── Augment training data ──────────────────────────────
    if use_augmentation:
        print("\n[2/5] Augmenting seizure windows...")
        X_train_raw, y_train = augment_seizure_windows(
            X_train_raw, y_train, target_ratio=0.15
        )
    else:
        print("\n[2/5] Skipping augmentation.")

    # ── Prepare model input ────────────────────────────────
    print("\n[3/5] Preparing model input...")
    X_train = prepare_model_input(X_train_raw, mode="raw")
    X_val = prepare_model_input(X_val_raw, mode="raw")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")

    # ── Build model ────────────────────────────────────────
    print("\n[4/5] Building model...")
    if model_type == "lite":
        model = build_lightweight_model()
    else:
        model = build_cnn_lstm_model()

    model.summary()
    print(f"\nTotal trainable parameters: {model.count_params():,}")

    # ── Compute class weights ──────────────────────────────
    class_weights = None
    if config.USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(y_train)

    # ── Train ──────────────────────────────────────────────
    print(f"\n[5/5] Training for up to {config.EPOCHS} epochs...")
    model_save_path = str(
        config.MODEL_DIR / f"seizure_detector_{model_type}.keras"
    )
    cbs = get_training_callbacks(model_path=model_save_path)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        class_weight=class_weights,
        callbacks=cbs,
        verbose=1,
    )

    # ── Save training history ──────────────────────────────
    history_path = config.RESULTS_DIR / f"training_history_{model_type}.json"
    history_data = {k: [float(v) for v in vals]
                    for k, vals in history.history.items()}
    with open(history_path, "w") as f:
        json.dump(history_data, f, indent=2)
    print(f"\nTraining history saved to {history_path}")

    # ── Final metrics ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    val_metrics = model.evaluate(X_val, y_val, verbose=0)
    metric_names = model.metrics_names
    for name, val in zip(metric_names, val_metrics):
        print(f"  val_{name}: {val:.4f}")

    print(f"\nBest model saved to: {model_save_path}")
    print(f"Run 'python evaluate.py' for full test set evaluation.")

    return model, history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train seizure detection model")
    parser.add_argument("--model", choices=["full", "lite"], default="full",
                        help="Model architecture: 'full' CNN-LSTM or 'lite'")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable seizure window augmentation")
    args = parser.parse_args()

    train(model_type=args.model, use_augmentation=not args.no_augment)
