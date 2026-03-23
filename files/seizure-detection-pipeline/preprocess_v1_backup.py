"""
Preprocessing module for CHB-MIT EEG data.

Handles:
  1. Loading .edf files and seizure annotations
  2. Channel selection (mapping CHB-MIT channels to our 6-channel montage)
  3. Resampling to 250 Hz (matching ADS1299 output)
  4. Bandpass filtering (0.5–45 Hz)
  5. Notch filtering (60 Hz mains removal)
  6. Windowing with overlap and seizure labeling
"""

import re
import numpy as np
import mne
from pathlib import Path
from scipy.signal import iirnotch, filtfilt, butter
from tqdm import tqdm

import config


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
                    records.append(current_record)
                fname = line.split(":")[-1].strip()
                current_record = {"filename": fname, "seizures": []}

            elif "Seizure Start" in line and current_record is not None:
                # Handle both "Seizure Start Time:" and "Seizure 1 Start Time:"
                match = re.search(r"(\d+)\s*seconds", line)
                if match:
                    start = int(match.group(1))
                    current_record["_pending_start"] = start

            elif "Seizure End" in line and current_record is not None:
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


def select_channels(raw: mne.io.Raw) -> list[str]:
    """
    Select the 6 channels from CHB-MIT that best match our electrode montage.

    CHB-MIT uses bipolar montage names (e.g., "FP1-F7") while our device uses
    referential electrodes. We pick the bipolar pairs that include our target
    electrode as the primary.

    Returns list of 6 channel names found in the recording.
    """
    available = raw.ch_names
    selected = []

    for target, candidates in config.CHANNEL_MAPPING.items():
        found = False
        for candidate in candidates:
            # Case-insensitive matching
            for ch in available:
                if ch.upper().replace(" ", "") == candidate.upper().replace(" ", ""):
                    selected.append(ch)
                    found = True
                    break
            if found:
                break

        if not found:
            print(f"  Warning: Could not find channel for {target}. "
                  f"Available: {available[:5]}...")

    return selected


def apply_bandpass(data: np.ndarray, sfreq: float,
                   low: float = None, high: float = None) -> np.ndarray:
    """
    Apply zero-phase Butterworth bandpass filter.

    Args:
        data: shape (n_channels, n_samples)
        sfreq: sampling frequency in Hz
        low: low cutoff (default from config)
        high: high cutoff (default from config)

    Returns:
        Filtered data, same shape.
    """
    low = low or config.BANDPASS_LOW
    high = high or config.BANDPASS_HIGH
    nyq = sfreq / 2.0
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data, axis=-1)


def apply_notch(data: np.ndarray, sfreq: float,
                freqs: list[float] = None) -> np.ndarray:
    """
    Apply notch filter(s) to remove mains interference.

    Args:
        data: shape (n_channels, n_samples)
        sfreq: sampling frequency in Hz
        freqs: list of notch frequencies (default from config)

    Returns:
        Filtered data, same shape.
    """
    freqs = freqs or config.NOTCH_FREQS
    for freq in freqs:
        quality_factor = freq / config.NOTCH_WIDTH
        b, a = iirnotch(freq, quality_factor, sfreq)
        data = filtfilt(b, a, data, axis=-1)
    return data


def create_windows(data: np.ndarray, sfreq: float,
                   seizure_intervals: list[tuple],
                   total_duration: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Slice continuous EEG into overlapping windows with seizure labels.

    A window is labeled as seizure (1) if >50% of its samples overlap
    with any seizure interval.

    Args:
        data: shape (n_channels, n_samples)
        sfreq: sampling frequency
        seizure_intervals: list of (start_sec, end_sec) tuples
        total_duration: total recording duration in seconds

    Returns:
        windows: shape (n_windows, n_channels, window_samples)
        labels: shape (n_windows,) — 0 = non-seizure, 1 = seizure
    """
    window_samples = int(config.WINDOW_SECONDS * sfreq)
    step_samples = int((config.WINDOW_SECONDS - config.OVERLAP_SECONDS) * sfreq)
    n_samples = data.shape[1]

    windows = []
    labels = []

    start = 0
    while start + window_samples <= n_samples:
        window = data[:, start:start + window_samples]
        windows.append(window)

        # Determine label: check overlap with seizure intervals
        win_start_sec = start / sfreq
        win_end_sec = (start + window_samples) / sfreq
        overlap = 0.0
        for sz_start, sz_end in seizure_intervals:
            ov_start = max(win_start_sec, sz_start)
            ov_end = min(win_end_sec, sz_end)
            if ov_end > ov_start:
                overlap += (ov_end - ov_start)

        # Label as seizure if >50% of window overlaps with seizure
        fraction = overlap / config.WINDOW_SECONDS
        labels.append(1 if fraction > 0.5 else 0)

        start += step_samples

    return np.array(windows), np.array(labels)


def process_patient(patient_id: str, data_dir: Path = None) -> dict:
    """
    Process all recordings for a single CHB-MIT patient.

    Pipeline per file:
      1. Load .edf
      2. Select 6 target channels
      3. Resample to 250 Hz
      4. Bandpass filter 0.5–45 Hz
      5. Notch filter 60 Hz
      6. Window into 4-second segments with labels

    Args:
        patient_id: e.g., "chb01"
        data_dir: path to CHB-MIT root (default from config)

    Returns:
        dict with:
          "windows": np.array shape (N, 6, 1000)
          "labels": np.array shape (N,)
          "patient_id": str
    """
    data_dir = data_dir or config.DATA_DIR
    patient_dir = data_dir / patient_id

    if not patient_dir.exists():
        print(f"  Skipping {patient_id}: directory not found at {patient_dir}")
        return {"windows": np.array([]), "labels": np.array([]),
                "patient_id": patient_id}

    # Parse seizure annotations
    records = parse_summary_file(patient_dir)
    if not records:
        return {"windows": np.array([]), "labels": np.array([]),
                "patient_id": patient_id}

    all_windows = []
    all_labels = []

    for record in tqdm(records, desc=f"  {patient_id}", leave=False):
        edf_path = patient_dir / record["filename"]
        if not edf_path.exists():
            continue

        try:
            # Load EDF (suppress MNE verbose output)
            raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        except Exception as e:
            print(f"  Error loading {edf_path.name}: {e}")
            continue

        # Select our 6 target channels
        channels = select_channels(raw)
        if len(channels) < config.N_CHANNELS:
            print(f"  Skipping {edf_path.name}: only found {len(channels)}/{config.N_CHANNELS} channels")
            continue

        raw.pick_channels(channels[:config.N_CHANNELS])

        # Resample to match ADS1299 output rate
        if raw.info["sfreq"] != config.TARGET_SFREQ:
            raw.resample(config.TARGET_SFREQ, verbose=False)

        sfreq = raw.info["sfreq"]
        data = raw.get_data()  # shape: (n_channels, n_samples)

        # Apply bandpass filter
        data = apply_bandpass(data, sfreq)

        # Apply notch filter
        data = apply_notch(data, sfreq)

        # Normalize per-channel (z-score)
        for ch in range(data.shape[0]):
            mu = np.mean(data[ch])
            sigma = np.std(data[ch])
            if sigma > 0:
                data[ch] = (data[ch] - mu) / sigma

        # Create windows with seizure labels
        duration = data.shape[1] / sfreq
        windows, labels = create_windows(
            data, sfreq, record["seizures"], duration
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

    n_seizure = int(np.sum(labels)) if len(labels) > 0 else 0
    n_total = len(labels)
    print(f"  {patient_id}: {n_total} windows ({n_seizure} seizure, "
          f"{n_total - n_seizure} non-seizure)")

    return {
        "windows": windows,
        "labels": labels,
        "patient_id": patient_id,
    }


def process_all_patients(patient_ids: list[str],
                         save_path: Path = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Process multiple patients and optionally save to disk.

    Args:
        patient_ids: list of patient IDs to process
        save_path: if provided, saves .npz file

    Returns:
        (all_windows, all_labels)
    """
    all_windows = []
    all_labels = []

    print(f"Processing {len(patient_ids)} patients...")
    for pid in patient_ids:
        result = process_patient(pid)
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

    n_seizure = int(np.sum(labels))
    print(f"\nTotal: {len(labels)} windows "
          f"({n_seizure} seizure / {len(labels) - n_seizure} non-seizure)")
    print(f"Seizure prevalence: {n_seizure/len(labels)*100:.1f}%")

    return windows, labels


if __name__ == "__main__":
    """Run preprocessing on all patient sets."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TRAIN SET")
    print("=" * 60)
    process_all_patients(
        config.TRAIN_PATIENTS,
        save_path=config.PROCESSED_DIR / "train.npz"
    )

    print("\n" + "=" * 60)
    print("VALIDATION SET")
    print("=" * 60)
    process_all_patients(
        config.VAL_PATIENTS,
        save_path=config.PROCESSED_DIR / "val.npz"
    )

    print("\n" + "=" * 60)
    print("TEST SET")
    print("=" * 60)
    process_all_patients(
        config.TEST_PATIENTS,
        save_path=config.PROCESSED_DIR / "test.npz"
    )
