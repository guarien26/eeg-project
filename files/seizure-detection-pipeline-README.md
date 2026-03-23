# Seizure Detection CNN-LSTM Training Pipeline

AI-powered seizure detection model for the Portable EEG project.
Designed to work with the **ADS1299-6PAG + ESP32-C3** hardware in a hybrid architecture
where the ESP32-C3 acquires and streams EEG data, and this server-side model performs
seizure classification.

## Architecture

```
Input: 4-second EEG window (6 channels × 1000 samples at 250 Hz)
  │
  ├─ Conv1D(32, k=7) → BatchNorm → ReLU → MaxPool(2)
  ├─ Conv1D(64, k=7) → BatchNorm → ReLU → MaxPool(2)
  ├─ Conv1D(128, k=7) → BatchNorm → ReLU → MaxPool(2)
  │
  ├─ LSTM(64)
  ├─ Dense(32) → ReLU → Dropout(0.5)
  └─ Dense(1, sigmoid) → Seizure probability [0, 1]
```

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Download the CHB-MIT dataset

The CHB-MIT Scalp EEG Database is freely available from PhysioNet:

```bash
# Option A: Using wget (downloads ~35 GB)
wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/

# Option B: Using the PhysioNet CLI
pip install wfdb
# Then download specific patients as needed

# Move/symlink to the expected directory:
mkdir -p data
mv physionet.org/files/chbmit/1.0.0 data/chb-mit-scalp-eeg-database-1.0.0
```

### 3. Preprocess the data

```bash
python preprocess.py
```

This will:
- Load all .edf files for each patient
- Select 6 channels matching our electrode montage (Fp1, Fp2, C3, C4, T7, T8)
- Resample to 250 Hz (matching ADS1299 output)
- Apply bandpass (0.5–45 Hz) and notch (60 Hz) filters
- Segment into 4-second windows with 1-second overlap
- Label windows using seizure annotations from summary files
- Save as compressed .npz files in `data/processed/`

### 4. Train the model

```bash
# Full CNN-LSTM model (recommended)
python train.py --model full

# Lightweight model (for quick testing)
python train.py --model lite
```

### 5. Evaluate on test patients

```bash
python evaluate.py
```

This generates:
- Full metrics report (sensitivity, specificity, AUC, false alarm rate)
- Confusion matrix plot
- ROC curve plot
- Threshold sensitivity analysis

## Project Structure

```
seizure-detection-pipeline/
├── config.py          # All configuration constants
├── preprocess.py      # CHB-MIT data loading, filtering, windowing
├── features.py        # DWT feature extraction
├── model.py           # CNN-LSTM model architecture
├── train.py           # Training pipeline with augmentation
├── evaluate.py        # Test evaluation with clinical metrics
├── requirements.txt   # Python dependencies
├── README.md          # This file
├── data/
│   ├── chb-mit-scalp-eeg-database-1.0.0/  # Raw dataset
│   └── processed/                          # Preprocessed .npz files
├── models/            # Saved model checkpoints
└── results/           # Evaluation results and plots
```

## Channel Mapping

Our 6-channel montage maps to CHB-MIT's bipolar channels:

| Our Electrode | ADS1299 Input | CHB-MIT Channel | Brain Region |
|--------------|---------------|-----------------|--------------|
| Fp1          | J1 Pin 1      | FP1-F7          | Frontal left |
| Fp2          | J1 Pin 2      | FP2-F8          | Frontal right |
| C3           | J1 Pin 3      | C3-P3           | Central left |
| C4           | J1 Pin 4      | C4-P4           | Central right |
| T7           | J1 Pin 5      | F7-T7           | Temporal left |
| T8           | J1 Pin 6      | F8-T8           | Temporal right |
| Cz (ref)     | J1 Pin 7      | —               | Vertex reference |
| Fpz (bias)   | J1 Pin 8      | —               | Bias drive |

## Key Design Decisions

**Why 4-second windows?** Absent seizures produce 3 Hz spike-wave discharges.
A 4-second window captures 12 full cycles — enough for the model to reliably
identify the pattern while keeping inference latency practical.

**Why CNN before LSTM?** The 1D CNN layers extract local spectral features
(spike shapes, frequency content) from the multi-channel signal. The LSTM then
models how these features evolve over the 4-second window — critical for
distinguishing the rhythmic seizure pattern from transient artifacts.

**Why patient-wise splitting?** EEG varies greatly between individuals. If we
split randomly, the model sees windows from the same patient in both train and
test sets, inflating accuracy. Patient-wise splitting ensures we test on
patients the model has never seen, giving realistic performance estimates.

**Why class weighting + augmentation?** Seizure windows are only 2–5% of the
data. Without correction, the model learns to always predict "normal" and still
achieves 95%+ accuracy. We use inverse-frequency class weights and synthetic
augmentation (noise, time-shift, scaling, channel dropout) to force the model
to actually learn seizure patterns.

## Performance Targets

| Metric | Target | Why |
|--------|--------|-----|
| Sensitivity | ≥ 95% | Must not miss seizures — patient safety |
| Specificity | ≥ 90% | Reduce alarm fatigue for caregivers |
| AUC-ROC | ≥ 0.95 | Strong overall discrimination |
| False alarm rate | ≤ 1.0/hr | Practical for continuous monitoring |
| Inference latency | < 500ms | Real-time response requirement |

## Next Steps (Phase 2+)

After training is validated:
1. **ESP32-C3 firmware** — SPI driver for ADS1299, BLE/Wi-Fi streaming
2. **Inference server** — Python WebSocket server running this model live
3. **React dashboard** — Real-time visualization matching the Figma design
4. **Integration testing** — End-to-end with real electrodes on scalp
