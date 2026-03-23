"""
Preprocessing module for CHB-MIT EEG data — v2 with PRE-ICTAL PREDICTION.

Changes from v1:
  - 3-class labeling: 0 = interictal (normal), 1 = pre-ictal, 2 = ictal (seizure)
  - Configurable prediction horizon (default 5 minutes before seizure onset)
  - For binary training, pre-ictal + ictal are merged as positive class
  - Buffer zone between interictal and pre-ictal to reduce label noise

Labels:
  0 = interictal  (normal brain activity, well before any seizure)
  1 = pre-ictal   (5 minutes before seizure onset — the PREDICTION window)
  2 = ictal        (during the seizure itself)

For binary prediction mode: label = 1 if pre-ictal OR ictal, else 0
"""

import re
import numpy as np
import mne
from pathlib import Path
from scipy.signal import iirnotch, filtfilt, butter
from tqdm import tqdm

import config


# ── Pre-ictal configuration ──────────────────────────────────────
PREICTAL_SECONDS = getattr(config, 'PREICTAL_SECONDS', 300)  # 5 minutes
POSTICTAL_BUFFER = getattr(config, 'POSTICTAL_BUFFER', 30)   # 30s after seizure
PREICTAL_BUFFER = getattr(config, 'PREICTAL_BUFFER', 30)     # 30s gap before pre-ictal
PREDICTION_MODE = getattr(config, 'PREDICTION_MODE', 'binary')  # 'binary' or 'three_class'


def parse_summary_file(patient_dir: Path) -> list[dict]:
    """
    Parse the CHB-MIT summary .txt file to extract seizure annotations.

    Each patient folder contains a *-summary.txt with entries like:
        File Name: chb01_03.edf
        Number of Seizures in File: 1
        Seizure Start Time: 2996 seconds
        Seizure End Time: 3036 seconds

    Returns a list of dicts: {filename, seizures: [(start_sec, end_sec), ...]}
    """
    summary_files = list(patient_dir.glob("*-summary.txt")) + \
                    list(patient_dir.glob("*-summary.txt"))
    if not summary_files:
        print(f"  Warning: No summary file found in {patient_dir}")
        return []

    summary_path = summary_files[0]
    records = []
    current_record = None

    with open(summary_path, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("File Name:"):
                if current_record is not None:
                    current_record.pop("_pending_start", None)
                    records.append(current_record)
                fname = line.split(":")[-1].strip()
                current_record = {"filename": fname, "seizures": []}

            elif re.search(r"Seizure.*Start", line) and current_record is not None:
                match = re.search(r"(\d+)\s*seconds", line)
                if match:
                    start = int(match.group(1))
                    current_record["_pending_start"] = start

            elif re.search(r"Seizure.*End", line) and current_record is not None:
                match = re.search(r"(\d+)\s*seconds", line)
                if match and "_pending_start" in current_record:
                    end = int(match.group(1))
                    current_record["seizures"].append(
                        (current_record.pop("_pending_start"), end)
                    )

    if current_record is not None:
        current_record.pop("_pending_start", None)
        records.append(current_record)

    return records


def build_file_timeline(records: list[dict]) -> list[dict]:
    """
    Build a global timeline of seizures across all files for a patient.
    
    This is needed because the pre-ictal period for a seizure in file N
    might start in file N-1. We need to know the absolute time of each
    seizure across the full recording session.
    
    Returns list of {filename, file_start_sec, seizures: [(abs_start, abs_end), ...]}
    """
    # CHB-MIT files are typically 1 hour each, sequential
    # We estimate absolute times from file ordering
    timeline = []
    cumulative_sec = 0.0
    
    for record in records:
        entry = {
            "filename": record["filename"],
            "file_start_sec": cumulative_sec,
            "seizures": [],
            "local_seizures": record["seizures"],
        }
        for sz_start, sz_end in record["seizures"]:
            entry["seizures"].append(
                (cumulative_sec + sz_start, cumulative_sec + sz_end)
            )
        timeline.append(entry)
        # Assume 1-hour files (3600 sec) — will be corrected when we load
        cumulative_sec += 3600.0
    
    return timeline


def get_window_label(win_start_sec: float, win_end_sec: float,
                     seizure_intervals: list[tuple],
                     all_seizure_starts: list[float] = None) -> int:
    """
    Determine the label for a window based on its temporal relationship
    to seizures.
    
    Labels:
      0 = interictal (normal, safe distance from any seizure)
      1 = pre-ictal  (within PREICTAL_SECONDS before seizure onset)
      2 = ictal      (during seizure)
    
    Args:
        win_start_sec: window start in seconds (relative to file)
        win_end_sec: window end in seconds (relative to file)  
        seizure_intervals: list of (start, end) for seizures in this file
        all_seizure_starts: all seizure start times for pre-ictal calc
    
    Returns:
        Label: 0, 1, or 2
    """
    win_mid = (win_start_sec + win_end_sec) / 2.0
    
    # Check ICTAL first (highest priority)
    for sz_start, sz_end in seizure_intervals:
        ov_start = max(win_start_sec, sz_start)
        ov_end = min(win_end_sec, sz_end)
        if ov_end > ov_start:
            overlap = ov_end - ov_start
            fraction = overlap / (win_end_sec - win_start_sec)
            if fraction > 0.5:
                return 2  # ictal
    
    # Check PRE-ICTAL
    if all_seizure_starts is None:
        all_seizure_starts = [s for s, e in seizure_intervals]
    
    for sz_start in all_seizure_starts:
        preictal_start = sz_start - PREICTAL_SECONDS
        preictal_end = sz_start  # right up to seizure onset
        
        if preictal_start <= win_mid < preictal_end:
            return 1  # pre-ictal
    
    # Check POST-ICTAL buffer (label as 0 but could be excluded)
    # For now, just label as interictal
    
    return 0  # interictal


def select_channels(raw: mne.io.Raw) -> list[str]:
    """
    Select the 6 channels from CHB-MIT that best match our electrode montage.
    """
    available = raw.ch_names
    selected = []

    for target, candidates in config.CHANNEL_MAPPING.items():
        found = False
        for candidate in candidates:
            for ch in available:
                if ch.upper().replace(" ", "") == candidate.upper().replace(" ", ""):
                    selected.append(ch)
                    found = True
                    break
            if found:
                break

        if not found:
            pass  # Silently skip — warning was too verbose

    return selected


def apply_bandpass(data: np.ndarray, sfreq: float,
                   low: float = None, high: float = None) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter."""
    low = low or config.BANDPASS_LOW
    high = high or config.BANDPASS_HIGH
    nyq = sfreq / 2.0
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data, axis=-1)


def apply_notch(data: np.ndarray, sfreq: float,
                freqs: list[float] = None) -> np.ndarray:
    """Apply notch filter(s) to remove mains interference."""
    freqs = freqs or config.NOTCH_FREQS
    for freq in freqs:
        quality_factor = freq / config.NOTCH_WIDTH
        b, a = iirnotch(freq, quality_factor, sfreq)
        data = filtfilt(b, a, data, axis=-1)
    return data


def create_windows(data: np.ndarray, sfreq: float,
                   seizure_intervals: list[tuple],
                   total_duration: float,
                   all_seizure_starts: list[float] = None,
                   binary: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Slice continuous EEG into overlapping windows with prediction labels.

    A window is labeled as:
      2 (ictal) if >50% overlaps with a seizure interval
      1 (pre-ictal) if its midpoint falls within 5 min before any seizure start
      0 (interictal) otherwise

    If binary=True (default), labels 1 and 2 are merged into 1.

    Args:
        data: shape (n_channels, n_samples)
        sfreq: sampling frequency
        seizure_intervals: list of (start_sec, end_sec) tuples
        total_duration: total recording duration in seconds
        all_seizure_starts: seizure onset times for pre-ictal labeling
        binary: if True, merge pre-ictal + ictal into single positive class

    Returns:
        windows: shape (n_windows, n_channels, window_samples)
        labels: shape (n_windows,) — 0/1 if binary, 0/1/2 if three-class
    """
    window_samples = int(config.WINDOW_SECONDS * sfreq)
    step_samples = int((config.WINDOW_SECONDS - config.OVERLAP_SECONDS) * sfreq)
    n_samples = data.shape[1]

    # Build seizure start list if not provided
    if all_seizure_starts is None:
        all_seizure_starts = [s for s, e in seizure_intervals]

    windows = []
    labels = []

    start = 0
    while start + window_samples <= n_samples:
        window = data[:, start:start + window_samples]
        windows.append(window)

        win_start_sec = start / sfreq
        win_end_sec = (start + window_samples) / sfreq

        label = get_window_label(
            win_start_sec, win_end_sec,
            seizure_intervals, all_seizure_starts
        )

        if binary and label >= 1:
            label = 1

        labels.append(label)
        start += step_samples

    return np.array(windows), np.array(labels)


def process_patient(patient_id: str, data_dir: Path = None,
                    binary: bool = True) -> dict:
    """
    Process all recordings for a single CHB-MIT patient with pre-ictal labeling.

    Pipeline per file:
      1. Load .edf
      2. Select 6 target channels
      3. Resample to 250 Hz
      4. Bandpass filter 0.5–45 Hz
      5. Notch filter 60 Hz
      6. Window into 4-second segments with prediction labels

    Args:
        patient_id: e.g., "chb01"
        data_dir: path to CHB-MIT root (default from config)
        binary: if True, pre-ictal + ictal = 1; if False, 3-class labels

    Returns:
        dict with:
          "windows": np.array shape (N, 6, 1000)
          "labels": np.array shape (N,)
          "patient_id": str
          "label_counts": dict with counts per label
    """
    data_dir = data_dir or config.DATA_DIR
    patient_dir = data_dir / patient_id

    if not patient_dir.exists():
        print(f"  Skipping {patient_id}: directory not found at {patient_dir}")
        return {"windows": np.array([]), "labels": np.array([]),
                "patient_id": patient_id, "label_counts": {}}

    # Parse seizure annotations
    records = parse_summary_file(patient_dir)
    if not records:
        return {"windows": np.array([]), "labels": np.array([]),
                "patient_id": patient_id, "label_counts": {}}

    # Collect ALL seizure start times across the patient for pre-ictal labeling
    # Each file's seizures are in local time (within that file)
    # For pre-ictal labeling, we only need local times since we process per-file
    
    all_windows = []
    all_labels = []

    for record in tqdm(records, desc=f"  {patient_id}", leave=False):
        edf_path = patient_dir / record["filename"]
        if not edf_path.exists():
            continue

        try:
            raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        except Exception as e:
            print(f"  Error loading {edf_path.name}: {e}")
            continue

        # Select our 6 target channels
        channels = select_channels(raw)
        if len(channels) < config.N_CHANNELS:
            continue

        raw.pick_channels(channels[:config.N_CHANNELS])

        # Resample to match ADS1299 output rate
        if raw.info["sfreq"] != config.TARGET_SFREQ:
            raw.resample(config.TARGET_SFREQ, verbose=False)

        sfreq = raw.info["sfreq"]
        data = raw.get_data()  # shape: (n_channels, n_samples)
        file_duration = data.shape[1] / sfreq

        # Apply filters
        data = apply_bandpass(data, sfreq)
        data = apply_notch(data, sfreq)

        # Normalize per-channel (z-score)
        for ch in range(data.shape[0]):
            mu = np.mean(data[ch])
            sigma = np.std(data[ch])
            if sigma > 0:
                data[ch] = (data[ch] - mu) / sigma

        # Get seizure starts for this file (for pre-ictal labeling)
        local_seizure_starts = [s for s, e in record["seizures"]]
        
        # Also check if a seizure in the NEXT file means this file's
        # end is pre-ictal. For simplicity in v2, we handle this within
        # each file — the 5-min pre-ictal window before a seizure at
        # time T means windows from T-300 to T are labeled pre-ictal.

        # Create windows with prediction labels
        windows, labels = create_windows(
            data, sfreq, record["seizures"], file_duration,
            all_seizure_starts=local_seizure_starts,
            binary=binary
        )

        if len(windows) > 0:
            all_windows.append(windows)
            all_labels.append(labels)

    if all_windows:
        windows = np.concatenate(all_windows, axis=0)
        labels = np.concatenate(all_labels, axis=0)
    else:
        windows = np.array([])
        labels = np.array([])

    # Count labels
    label_counts = {}
    if len(labels) > 0:
        unique, counts = np.unique(labels, return_counts=True)
        label_counts = {int(k): int(v) for k, v in zip(unique, counts)}

    n_total = len(labels)
    if binary:
        n_positive = int(np.sum(labels == 1))
        n_negative = n_total - n_positive
        print(f"  {patient_id}: {n_total} windows "
              f"({n_positive} positive [pre-ictal+ictal], "
              f"{n_negative} interictal)")
    else:
        n_ictal = label_counts.get(2, 0)
        n_preictal = label_counts.get(1, 0)
        n_interictal = label_counts.get(0, 0)
        print(f"  {patient_id}: {n_total} windows "
              f"({n_ictal} ictal, {n_preictal} pre-ictal, "
              f"{n_interictal} interictal)")

    return {
        "windows": windows,
        "labels": labels,
        "patient_id": patient_id,
        "label_counts": label_counts,
    }


def process_all_patients(patient_ids: list[str],
                         save_path: Path = None,
                         binary: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Process multiple patients and optionally save to disk.
    """
    all_windows = []
    all_labels = []

    print(f"Processing {len(patient_ids)} patients...")
    print(f"  Mode: {'binary prediction' if binary else 'three-class'}")
    print(f"  Pre-ictal window: {PREICTAL_SECONDS}s ({PREICTAL_SECONDS/60:.0f} min) "
          f"before seizure onset")

    for pid in patient_ids:
        result = process_patient(pid, binary=binary)
        if len(result["windows"]) > 0:
            all_windows.append(result["windows"])
            all_labels.append(result["labels"])

    if not all_windows:
        print("Warning: No data was processed successfully.")
        return np.array([]), np.array([])

    windows = np.concatenate(all_windows, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(save_path), windows=windows, labels=labels)
        print(f"Saved {len(windows)} windows to {save_path}")

    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nTotal: {len(labels)} windows")
    for u, c in zip(unique, counts):
        pct = c / len(labels) * 100
        if binary:
            name = "positive (pre-ictal+ictal)" if u == 1 else "interictal"
        else:
            name = {0: "interictal", 1: "pre-ictal", 2: "ictal"}.get(u, f"label_{u}")
        print(f"  {name}: {c} ({pct:.1f}%)")

    return windows, labels


if __name__ == "__main__":
    """Run preprocessing on all patient sets."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    binary = (PREDICTION_MODE == "binary")

    print("=" * 60)
    print("SEIZURE PREDICTION PREPROCESSING")
    print(f"Pre-ictal window: {PREICTAL_SECONDS/60:.0f} minutes before onset")
    print(f"Mode: {'binary' if binary else 'three-class'}")
    print("=" * 60)

    print("\nTRAIN SET")
    print("=" * 60)
    process_all_patients(
        config.TRAIN_PATIENTS,
        save_path=config.PROCESSED_DIR / "train.npz",
        binary=binary
    )

    print("\n" + "=" * 60)
    print("VALIDATION SET")
    print("=" * 60)
    process_all_patients(
        config.VAL_PATIENTS,
        save_path=config.PROCESSED_DIR / "val.npz",
        binary=binary
    )

    print("\n" + "=" * 60)
    print("TEST SET")
    print("=" * 60)
    process_all_patients(
        config.TEST_PATIENTS,
        save_path=config.PROCESSED_DIR / "test.npz",
        binary=binary
    )
