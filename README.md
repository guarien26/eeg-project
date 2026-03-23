# Portable AI-Enabled EEG for Seizure Prediction

AI-powered real-time EEG analysis for early seizure onset detection. Built for Professor Campisi's Impact Innovation course.

**Team:** Will Shang, Bree Choi, Guarien Barone

---

## Overview

This project is a portable Electroencephalogram (EEG) headset that uses a CNN-LSTM deep learning model to predict seizures up to 2 minutes before onset. The system acquires 6-channel EEG data via the ADS1299-6PAG analog front-end, streams it wirelessly through an ESP32-C3 microcontroller, processes it through a Python inference server, and displays real-time predictions on a React dashboard.

### Architecture

```
EEG Electrodes → ADS1299-6PAG → ESP32-C3 (WiFi) → Python Server → React Dashboard
  (6 channels)    (24-bit ADC)    (SPI + WebSocket)   (CNN-LSTM)     (visualization)
```

### Model Performance

| Metric | Within-Patient (v3) | Cross-Patient (v5) |
|--------|--------------------|--------------------|
| AUC-ROC | 99.97% | In progress |
| Pre-ictal window | 2 minutes | 2 minutes |
| Inference latency | ~50ms on M1 Mac | ~50ms on M1 Mac |

---

## Repository Structure

```
files/
├── seizure-detection-pipeline/       # ML training + inference server
│   ├── config.py                     # Hyperparameters and constants
│   ├── preprocess.py                 # EDF loading, windowing, label generation
│   ├── features.py                   # DWT feature extraction (db4 wavelet)
│   ├── model.py                      # CNN-LSTM architecture, callbacks, class weights
│   ├── train.py                      # Augmentation utilities
│   ├── evaluate.py                   # Evaluation metrics
│   ├── server.py                     # WebSocket inference server
│   ├── test_client.py                # Terminal-based test client
│   ├── start.sh                      # Single-command launcher (server + dashboard)
│   ├── requirements.txt              # Python dependencies
│   ├── models/                       # Trained .keras models (gitignored)
│   └── data/                         # CHB-MIT dataset + feature cache (gitignored)
│
├── eeg-dashboard/                    # React frontend
│   ├── src/
│   │   ├── App.jsx                   # Main app with WebSocket + CSV export
│   │   ├── components/               # Header, MetricCards, ControlBar, WaveformPanel, AnalysisPanel
│   │   └── hooks/                    # useWebSocket, useDemoData
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
│
├── Design_Project_Proposal__Portable_EEG.pdf
└── Schematic_EEG.pdf                 # ADS1299 + ESP32-C3 circuit schematic
```

---

## Getting Started (for teammates)

### Prerequisites

- **macOS** (tested on M1/M2, macOS 26+)
- **Python 3.12+**
- **Node.js 18+** and npm
- ~60GB free disk space (for dataset)

### Step 1: Clone and set up Python environment

```bash
git clone https://github.com/guarien26/eeg-project.git
cd eeg-project/files/seizure-detection-pipeline

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Download the CHB-MIT dataset

The dataset is from [PhysioNet's CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/). It's ~60GB total and gitignored. Run this to download all patients:

```bash
cd data
mkdir -p chb-mit-scalp-eeg-database-1.0.0
cd chb-mit-scalp-eeg-database-1.0.0

cat > /tmp/dl.sh << 'EOF'
p="$1"; f="$2"
if [ -f "$p/$f" ] && [ $(stat -f%z "$p/$f" 2>/dev/null || echo 0) -gt 1000 ]; then
  echo "  SKIP $f"
else
  echo "  GET  $f"
  curl -C - -s -O --output-dir "$p" "https://physionet.org/files/chbmit/1.0.0/$p/$f"
fi
EOF
chmod +x /tmp/dl.sh

for p in chb01 chb02 chb03 chb04 chb05 chb06 chb07 chb08 chb09 chb10 chb11 chb12 chb13 chb14 chb15 chb16 chb17 chb18 chb19 chb20 chb21 chb22 chb23; do
  echo "=== $p ==="
  mkdir -p "$p"
  curl -C - -s -O --output-dir "$p" "https://physionet.org/files/chbmit/1.0.0/$p/${p}-summary.txt"
  curl -s "https://physionet.org/files/chbmit/1.0.0/$p/" | grep -oE 'href="[^"]+\.edf"' | sed 's/href="//;s/"//' | \
    xargs -P 8 -I {} sh /tmp/dl.sh "$p" "{}"
  echo "  Done: $(ls "$p"/*.edf 2>/dev/null | wc -l) files"
done
```

This runs 8 parallel downloads per patient. Takes 2-4 hours depending on connection. You can check progress anytime:

```bash
for d in chb*/; do echo "$d  $(ls "$d"/*.edf 2>/dev/null | wc -l) files"; done
```

### Step 3: Cache DWT features (one-time preprocessing)

Raw EDF files are too large to load all at once. This step converts each patient's data into compact DWT feature files (~500MB-3GB each) that the trainer can stream from disk:

```bash
cd /path/to/eeg-project/files/seizure-detection-pipeline
source venv/bin/activate

python3 -c "
from pathlib import Path
import numpy as np
from preprocess import process_patient
from features import prepare_model_input
import gc

data_dir = Path('data/chb-mit-scalp-eeg-database-1.0.0')
cache_dir = Path('data/feature_cache')
cache_dir.mkdir(exist_ok=True)

all_pts = ['chb01','chb02','chb03','chb04','chb05','chb06','chb07','chb08',
           'chb09','chb10','chb11','chb12','chb13','chb14','chb15','chb16','chb18','chb20']

for pid in all_pts:
    xpath = cache_dir / f'{pid}_X.npy'
    ypath = cache_dir / f'{pid}_y.npy'
    if xpath.exists() and ypath.exists():
        print(f'  CACHED {pid}')
        continue
    try:
        result = process_patient(pid, data_dir=data_dir)
        if len(result['windows']) == 0 or int(result['labels'].sum()) == 0:
            print(f'  SKIP {pid} (no positives)')
            continue
        print(f'  {pid}: {result[\"label_counts\"]}')
        X = prepare_model_input(result['windows'], mode='hybrid')
        y = result['labels']
        np.save(xpath, X)
        np.save(ypath, y)
        print(f'  SAVED {pid}: X={X.shape}')
        del result, X, y
        gc.collect()
    except Exception as e:
        print(f'  SKIP {pid}: {e}')
        gc.collect()
"
```

This processes one patient at a time to stay within 16GB RAM. Takes 1-2 hours. Each patient's features are saved as `.npy` files in `data/feature_cache/`. If it crashes on a specific patient, just re-run — it skips already-cached patients.

### Step 4: Train the model

The training script uses a custom data generator that loads one patient at a time from the cached `.npy` files, keeping RAM usage under control:

```bash
source venv/bin/activate

python3 -c "
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from model import compute_class_weights, get_training_callbacks
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
import gc, os
os.makedirs('models', exist_ok=True)

cache_dir = Path('data/feature_cache')

train_pts = ['chb01','chb02','chb03','chb04','chb06','chb07','chb08',
             'chb09','chb10','chb11','chb13','chb16','chb20']
val_pts = ['chb05','chb12','chb14']

# Load val fully (small enough)
val_X_list, val_y_list = [], []
for pid in val_pts:
    xp = cache_dir / f'{pid}_X.npy'
    yp = cache_dir / f'{pid}_y.npy'
    if not xp.exists(): continue
    y = np.load(yp)
    if y.sum() == 0: continue
    val_X_list.append(np.load(xp))
    val_y_list.append(y)
    print(f'Val {pid}: {len(y)} windows, {int(y.sum())} pos')

X_val = np.concatenate(val_X_list); del val_X_list
y_val = np.concatenate(val_y_list); del val_y_list
gc.collect()

# See full training script in conversation history or adapt as needed
# The model architecture is CNN-LSTM with 125,665 parameters
# Training uses patient-wise split with class weighting
print(f'Val: {X_val.shape}, {int(y_val.sum())} positive')
print('Ready to train — see training commands in project docs')
"
```

The full training command with the generator-based approach is documented in the project conversation history. The key parameters are: 15 train patients, 3 val patients (patient-wise split), 3% augmentation ratio, 40 epochs with early stopping.

### Step 5: Set up the dashboard

```bash
cd /path/to/eeg-project/files/eeg-dashboard
npm install
```

### Step 6: Run everything

**Option A — Single command (recommended):**

```bash
cd /path/to/eeg-project/files/seizure-detection-pipeline
./start.sh              # live mode (waits for ESP32 hardware)
./start.sh --simulate   # demo mode (flat-line standby)
```

This launches the inference server and dashboard together, opens `http://localhost:3000` in your browser, and Ctrl+C shuts both down cleanly.

**Option B — Manual (two terminals):**

Terminal 1 — Inference server:
```bash
cd files/seizure-detection-pipeline
source venv/bin/activate
python server.py --simulate
```

Terminal 2 — Dashboard:
```bash
cd files/eeg-dashboard
npm run dev
```

Then open `http://localhost:3000`.

---

## Hardware

### Components

| Component | Part Number | Purpose |
|-----------|-------------|---------|
| ADC | ADS1299-6PAG | 6-channel, 24-bit, 250 SPS EEG analog front-end |
| MCU | ESP32-C3 (Seeed XIAO) | WiFi/BLE microcontroller for SPI + wireless streaming |
| Electrodes | Ag/AgCl dry electrodes | 6 EEG channels + reference + bias |
| ESD Protection | ESD7351XV2T1G (x8) | Input protection diodes |

### Electrode Placement (10-20 System)

6 channels mapped to standard positions: Fp1, Fp2, C3, C4, T7, T8 with Cz reference (SRB1) and Fpz bias (BIASOUT).

### Schematic

See `Schematic_EEG.pdf` for the full circuit design including input filters (4.7kΩ + 4.7nF per channel), ESD protection, decoupling capacitors, and SPI connections between the ADS1299 and ESP32-C3.

---

## Technical Details

### CNN-LSTM Model Architecture

```
Input: (1000, 6) → DWT hybrid features
→ Conv1D(32, k=7) + BatchNorm + ReLU + MaxPool(2)
→ Conv1D(64, k=7) + BatchNorm + ReLU + MaxPool(2)
→ Conv1D(128, k=7) + BatchNorm + ReLU + MaxPool(2)
→ LSTM(64) → Dense(32, ReLU) + Dropout(0.5) → Dense(1, sigmoid)
```

125,665 parameters. ~490 KB model file.

### Key Configuration

| Parameter | Value |
|-----------|-------|
| Sample rate | 250 Hz |
| Channels | 6 (Fp1, Fp2, C3, C4, T7, T8) |
| Window size | 4 seconds (1000 samples) |
| Slide step | 1 second (3s overlap) |
| Pre-ictal window | 120 seconds before seizure onset |
| Bandpass filter | 0.5–45 Hz |
| DWT wavelet | db4, level 5 |
| Prediction threshold | 0.7 |

### WebSocket Protocol

The server (`ws://localhost:8765`) sends two message types:

**prediction** — sent every 1 second during recording:
```json
{
  "type": "prediction",
  "data": {
    "probability": 0.0312,
    "risk_level": "normal",
    "threshold": 0.7,
    "latency_ms": 48.2,
    "channels": { "Fp1": {"mean": -2.1, "std": 15.3}, ... }
  }
}
```

**waveform** — raw samples for dashboard visualization:
```json
{
  "type": "waveform",
  "data": {
    "channels": { "Fp1": [1.2, -0.8, ...], ... },
    "sfreq": 250
  }
}
```

The dashboard can send: `start_recording`, `stop_recording`, `set_threshold`.

---

## Dataset

[CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/) — 23 pediatric patients with intractable seizures, recorded at Boston Children's Hospital. 844 hours of continuous EEG, 198 seizures annotated.

We use 18 patients with seizure annotations. The data is not included in the repo due to size (~60GB). See Step 2 above for download instructions.

---

## Troubleshooting

**OOM during training:** The feature cache files total ~33GB. The training script uses a generator that loads one patient at a time. If you still get killed, reduce the number of training patients or use `np.load(mmap_mode='r')` for memory-mapped access.

**`zsh: command not found: python`:** Use `python3` instead, or activate the venv first with `source venv/bin/activate`.

**Server shows `SyntaxError: global`:** The `global WS_PORT, THRESHOLD` lines in `server.py main()` need to be removed. This has been fixed in the current version.

**Parser misses seizures for chb06+:** The summary file format changed from `Seizure Start Time` to `Seizure 1 Start Time` for later patients. The parser now uses `re.search(r"Seizure.*Start", line)` to match both formats.

---

## License

Academic project — CHB-MIT data is available under the PhysioNet Credentialed Health Data License.
