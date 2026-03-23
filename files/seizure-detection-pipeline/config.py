"""
Configuration for Seizure Detection CNN-LSTM Pipeline.

Hardware target: ADS1299-6PAG (6 channels) + ESP32-C3
Dataset: CHB-MIT Scalp EEG Database (PhysioNet)
"""

from pathlib import Path

# ============================================================
# Dataset paths
# ============================================================
# Download CHB-MIT from: https://physionet.org/content/chbmit/1.0.0/
# Place the extracted folder here:
DATA_DIR = Path("data/chb-mit-scalp-eeg-database-1.0.0")
PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

# ============================================================
# Signal acquisition parameters (matching ADS1299-6PAG config)
# ============================================================
TARGET_SFREQ = 250          # Hz — matches ADS1299 at 250 SPS
N_CHANNELS = 6              # 6-channel montage (Fp1, Fp2, C3, C4, T7, T8)
WINDOW_SECONDS = 4          # seconds per inference window
WINDOW_SAMPLES = TARGET_SFREQ * WINDOW_SECONDS  # 1000 samples
OVERLAP_SECONDS = 1         # sliding window overlap
OVERLAP_SAMPLES = TARGET_SFREQ * OVERLAP_SECONDS  # 250 samples

# ============================================================
# Preprocessing parameters
# ============================================================
BANDPASS_LOW = 0.5           # Hz — remove DC drift
BANDPASS_HIGH = 45.0         # Hz — remove muscle artifacts + line noise harmonics
NOTCH_FREQS = [60.0]        # Hz — US mains (add 50.0 for EU)
NOTCH_WIDTH = 2.0           # Hz — notch bandwidth

# CHB-MIT uses bipolar montage with 23 channels at 256 Hz.
# We select 6 channels that map to our electrode montage:
#   Fp1-F7  → approximates Fp1 (frontal left)
#   Fp2-F8  → approximates Fp2 (frontal right)
#   C3-P3   → approximates C3 (central left)
#   C4-P4   → approximates C4 (central right)
#   T7-P7   → approximates T7 (temporal left)  [or F7-T7]
#   T8-P8   → approximates T8 (temporal right)  [or F8-T8]
#
# Note: CHB-MIT uses different naming conventions per patient.
# The channel selector will try multiple name variants.
CHANNEL_MAPPING = {
    "Fp1": ["FP1-F7", "FP1-F3", "Fp1-F7", "Fp1-F3"],
    "Fp2": ["FP2-F8", "FP2-F4", "Fp2-F8", "Fp2-F4"],
    "C3":  ["C3-P3",  "F3-C3",  "C3-P3"],
    "C4":  ["C4-P4",  "F4-C4",  "C4-P4"],
    "T7":  ["F7-T7",  "T7-P7",  "F7-T7", "T7-FT9", "T3-T5"],
    "T8":  ["F8-T8",  "T8-P8",  "F8-T8", "T8-FT10", "T4-T6"],
}

# ============================================================
# DWT feature extraction parameters
# ============================================================
DWT_WAVELET = "db4"          # Daubechies-4 (standard for EEG)
DWT_LEVEL = 5                # Decomposition levels at 250 Hz:
                             #   Level 1: 62.5-125 Hz (gamma — mostly noise)
                             #   Level 2: 31.25-62.5 Hz (low gamma)
                             #   Level 3: 15.625-31.25 Hz (beta)
                             #   Level 4: 7.8125-15.625 Hz (alpha)
                             #   Level 5: 3.90625-7.8125 Hz (theta)
                             #   Approx:  0-3.90625 Hz (delta)

# Which DWT sub-bands to keep as features
# Maps to physiological EEG bands relevant for seizure detection
DWT_BANDS = {
    "delta":  "cA5",   # 0-3.9 Hz   — dominant in absent seizures (3 Hz spike-wave)
    "theta":  "cD5",   # 3.9-7.8 Hz — pre-ictal changes
    "alpha":  "cD4",   # 7.8-15.6 Hz — suppressed during seizures
    "beta":   "cD3",   # 15.6-31.3 Hz — motor cortex activity
}

# ============================================================
# Model architecture parameters
# ============================================================
CNN_FILTERS = [32, 64, 128]  # 1D conv filter counts per layer
CNN_KERNEL_SIZE = 7          # temporal kernel size
CNN_POOL_SIZE = 2            # max pooling factor
LSTM_UNITS = 64              # LSTM hidden units
DENSE_UNITS = 32             # fully connected layer before output
DROPOUT_RATE = 0.5           # dropout for regularization

# ============================================================
# Training parameters
# ============================================================
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 8     # stop if val_loss doesn't improve
LR_REDUCE_PATIENCE = 4      # reduce LR if val_loss plateaus
LR_REDUCE_FACTOR = 0.5
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# Class balance: seizure windows are rare (~2-5% of total)
# Use class weights to compensate
USE_CLASS_WEIGHTS = True

# Patient-wise split: ensures no data leakage between train/val/test
# These patient IDs are from CHB-MIT
TRAIN_PATIENTS = [
    "chb01", "chb02", "chb03", "chb04", "chb05",
    "chb06", "chb07", "chb08", "chb09", "chb10",
    "chb11", "chb12", "chb13", "chb14", "chb15",
]
VAL_PATIENTS = [
    "chb16", "chb17", "chb18",
]
TEST_PATIENTS = [
    "chb19", "chb20", "chb21", "chb22", "chb23",
]

# ============================================================
# Evaluation thresholds
# ============================================================
SEIZURE_THRESHOLD = 0.5      # probability above this → seizure detected
MIN_SENSITIVITY = 0.95       # target: catch 95% of seizures
MAX_FALSE_ALARM_RATE = 1.0   # max false alarms per hour
# Pre-ictal prediction settings
PREICTAL_SECONDS = 120       # 5 minutes before seizure = pre-ictal window
POSTICTAL_BUFFER = 30        # 30s after seizure end
PREICTAL_BUFFER = 30         # 30s gap before pre-ictal
PREDICTION_MODE = "binary"   # pre-ictal+ictal vs interictal
