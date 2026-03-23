"""
CNN-LSTM model architecture for EEG seizure detection.

Architecture overview:
  Input: (batch, 1000, 6) — 4-sec window, 6 channels at 250 Hz

  1D CNN layers extract local spatial/spectral features:
    Conv1D(32, k=7) → BatchNorm → ReLU → MaxPool(2)
    Conv1D(64, k=7) → BatchNorm → ReLU → MaxPool(2)
    Conv1D(128, k=7) → BatchNorm → ReLU → MaxPool(2)

  LSTM layer captures temporal dependencies across the window:
    LSTM(64, return_sequences=False)

  Dense classifier:
    Dense(32) → ReLU → Dropout(0.5) → Dense(1, sigmoid)

  Output: seizure probability (0 to 1)

This architecture follows the DWT-CNN-LSTM approach from:
  "Epileptic seizure detection from EEG signals based on
   1D CNN-LSTM deep learning model using DWT"
  (Scientific Reports, 2025) — achieving 96.94% on CHB-MIT.

We adapt it to 6 channels (matching our ADS1299-6PAG hardware)
instead of the full 23-channel clinical montage.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks

import config


def build_cnn_lstm_model(
    input_shape: tuple = None,
    cnn_filters: list[int] = None,
    kernel_size: int = None,
    pool_size: int = None,
    lstm_units: int = None,
    dense_units: int = None,
    dropout_rate: float = None,
    learning_rate: float = None,
) -> Model:
    """
    Build and compile the 1D CNN-LSTM seizure detection model.

    Args:
        input_shape: (time_steps, channels) — default (1000, 6)
        cnn_filters: list of filter counts per conv layer
        kernel_size: 1D conv kernel size
        pool_size: max pooling factor
        lstm_units: LSTM hidden state size
        dense_units: pre-output dense layer size
        dropout_rate: dropout probability
        learning_rate: Adam optimizer LR

    Returns:
        Compiled Keras model
    """
    # Defaults from config
    if input_shape is None:
        input_shape = (config.WINDOW_SAMPLES, config.N_CHANNELS)
    cnn_filters = cnn_filters or config.CNN_FILTERS
    kernel_size = kernel_size or config.CNN_KERNEL_SIZE
    pool_size = pool_size or config.CNN_POOL_SIZE
    lstm_units = lstm_units or config.LSTM_UNITS
    dense_units = dense_units or config.DENSE_UNITS
    dropout_rate = dropout_rate or config.DROPOUT_RATE
    learning_rate = learning_rate or config.LEARNING_RATE

    # Input layer
    inputs = layers.Input(shape=input_shape, name="eeg_input")
    x = inputs

    # === 1D CNN Feature Extraction ===
    # Each Conv1D layer learns increasingly abstract spectral-spatial
    # patterns from the multi-channel EEG signal.
    for i, filters in enumerate(cnn_filters):
        x = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            name=f"conv1d_{i+1}",
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = layers.ReLU(name=f"relu_{i+1}")(x)
        x = layers.MaxPooling1D(
            pool_size=pool_size,
            name=f"maxpool_{i+1}",
        )(x)
        x = layers.Dropout(
            dropout_rate * 0.5,  # lighter dropout in CNN layers
            name=f"dropout_cnn_{i+1}",
        )(x)

    # === LSTM Temporal Modeling ===
    # The LSTM sees the CNN feature maps as a sequence and learns
    # temporal dependencies across the 4-second window.
    # This is critical for detecting the rhythmic 3 Hz spike-wave
    # pattern characteristic of absent seizures.
    x = layers.LSTM(
        units=lstm_units,
        return_sequences=False,  # only final hidden state
        name="lstm",
    )(x)
    x = layers.Dropout(dropout_rate, name="dropout_lstm")(x)

    # === Dense Classifier ===
    x = layers.Dense(dense_units, name="dense_1")(x)
    x = layers.ReLU(name="relu_dense")(x)
    x = layers.Dropout(dropout_rate, name="dropout_dense")(x)

    # Output: single neuron with sigmoid for binary classification
    outputs = layers.Dense(1, activation="sigmoid", name="seizure_prob")(x)

    model = Model(inputs=inputs, outputs=outputs, name="SeizureDetector_CNN_LSTM")

    # Compile with binary crossentropy + Adam
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def build_lightweight_model(input_shape: tuple = None) -> Model:
    """
    Build a smaller model variant for faster iteration / limited data.

    Useful for:
      - Initial testing with small data subsets
      - Exploring if the pipeline works end-to-end
      - Potential future TinyML conversion (Phase 2 of project)

    Architecture: Conv1D(16) → Conv1D(32) → LSTM(32) → Dense(1)
    """
    if input_shape is None:
        input_shape = (config.WINDOW_SAMPLES, config.N_CHANNELS)

    inputs = layers.Input(shape=input_shape, name="eeg_input")
    x = inputs

    # Lighter CNN
    x = layers.Conv1D(16, 5, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(4)(x)

    x = layers.Conv1D(32, 5, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(4)(x)

    # Lighter LSTM
    x = layers.LSTM(32, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="seizure_prob")(x)

    model = Model(inputs=inputs, outputs=outputs, name="SeizureDetector_Lite")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall"),
                 keras.metrics.AUC(name="auc")],
    )

    return model


def get_training_callbacks(model_path: str = None) -> list:
    """
    Create standard training callbacks.

    Returns list of:
      - EarlyStopping: halt if val_loss doesn't improve
      - ReduceLROnPlateau: reduce LR when loss plateaus
      - ModelCheckpoint: save best model by val_recall
        (recall is prioritized — we must not miss seizures)
    """
    if model_path is None:
        model_path = str(config.MODEL_DIR / "best_model.keras")

    return [
        callbacks.EarlyStopping(
            monitor="val_auc",
            patience=config.EARLY_STOP_PATIENCE,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.LR_REDUCE_FACTOR,
            patience=config.LR_REDUCE_PATIENCE,
            min_lr=1e-6,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_recall",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]


def compute_class_weights(labels: np.ndarray) -> dict:
    """
    Compute class weights to handle seizure/non-seizure imbalance.

    Seizure windows are typically 2-5% of the dataset. Without weighting,
    the model could achieve 95%+ accuracy by always predicting non-seizure.
    We must ensure seizures are not missed.

    Args:
        labels: binary labels array

    Returns:
        dict {0: weight_normal, 1: weight_seizure}
    """
    n_total = len(labels)
    n_seizure = int(np.sum(labels))
    n_normal = n_total - n_seizure

    if n_seizure == 0 or n_normal == 0:
        return {0: 1.0, 1: 1.0}

    # Inverse frequency weighting
    weight_normal = n_total / (2 * n_normal)
    weight_seizure = n_total / (2 * n_seizure)

    print(f"Class weights: normal={weight_normal:.2f}, "
          f"seizure={weight_seizure:.2f} "
          f"(ratio: 1:{weight_seizure/weight_normal:.1f})")

    return {0: weight_normal, 1: weight_seizure}


if __name__ == "__main__":
    """Print model summaries."""
    print("=" * 60)
    print("FULL CNN-LSTM MODEL")
    print("=" * 60)
    model = build_cnn_lstm_model()
    model.summary()

    print(f"\nTotal parameters: {model.count_params():,}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

    # Test forward pass
    dummy = np.random.randn(2, config.WINDOW_SAMPLES, config.N_CHANNELS)
    pred = model.predict(dummy, verbose=0)
    print(f"Test prediction: {pred.flatten()}")

    print("\n" + "=" * 60)
    print("LIGHTWEIGHT MODEL (for testing / future TinyML)")
    print("=" * 60)
    lite = build_lightweight_model()
    lite.summary()
    print(f"\nLite parameters: {lite.count_params():,}")
