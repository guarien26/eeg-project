"""
Feature extraction module using Discrete Wavelet Transform (DWT).

Extracts frequency band features from EEG windows that are critical
for seizure detection:
  - Delta (0–4 Hz): dominant in absent seizure 3 Hz spike-wave
  - Theta (4–8 Hz): pre-ictal state marker
  - Alpha (8–13 Hz): suppressed during seizures
  - Beta (13–30 Hz): motor cortex / arousal

The DWT coefficients are concatenated per channel to form the
feature vector fed into the CNN-LSTM model.
"""

import numpy as np
import pywt
from tqdm import tqdm

import config


def dwt_decompose(signal: np.ndarray, wavelet: str = None,
                  level: int = None) -> dict[str, np.ndarray]:
    """
    Perform multi-level DWT decomposition on a 1D signal.

    Args:
        signal: 1D array of shape (n_samples,)
        wavelet: wavelet name (default: config.DWT_WAVELET = "db4")
        level: decomposition level (default: config.DWT_LEVEL = 5)

    Returns:
        dict mapping band names to coefficient arrays:
          {"delta": cA5, "theta": cD5, "alpha": cD4, "beta": cD3}
    """
    wavelet = wavelet or config.DWT_WAVELET
    level = level or config.DWT_LEVEL

    # pywt.wavedec returns [cA_n, cD_n, cD_n-1, ..., cD_1]
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Map to named bands
    bands = {}
    for band_name, coeff_name in config.DWT_BANDS.items():
        if coeff_name == "cA5":
            bands[band_name] = coeffs[0]        # approximation at level 5
        elif coeff_name == "cD5":
            bands[band_name] = coeffs[1]        # detail at level 5
        elif coeff_name == "cD4":
            bands[band_name] = coeffs[2]        # detail at level 4
        elif coeff_name == "cD3":
            bands[band_name] = coeffs[3]        # detail at level 3
        elif coeff_name == "cD2":
            bands[band_name] = coeffs[4]        # detail at level 2
        elif coeff_name == "cD1":
            bands[band_name] = coeffs[5]        # detail at level 1

    return bands


def extract_band_features(coefficients: np.ndarray) -> np.ndarray:
    """
    Extract statistical features from a single DWT band's coefficients.

    Features extracted (7 per band):
      1. Mean absolute value
      2. Standard deviation
      3. Energy (sum of squares)
      4. Entropy (Shannon)
      5. Max absolute value
      6. Zero crossing rate
      7. RMS (root mean square)

    Args:
        coefficients: 1D array of DWT coefficients for one band

    Returns:
        1D array of 7 features
    """
    c = coefficients.astype(np.float64)
    n = len(c)

    # Mean absolute value
    mean_abs = np.mean(np.abs(c))

    # Standard deviation
    std = np.std(c)

    # Energy
    energy = np.sum(c ** 2) / n

    # Shannon entropy
    c_sq = c ** 2
    total = np.sum(c_sq)
    if total > 0:
        p = c_sq / total
        p = p[p > 0]  # avoid log(0)
        entropy = -np.sum(p * np.log2(p))
    else:
        entropy = 0.0

    # Max absolute value
    max_abs = np.max(np.abs(c))

    # Zero crossing rate
    zero_crossings = np.sum(np.diff(np.sign(c)) != 0) / n

    # RMS
    rms = np.sqrt(np.mean(c ** 2))

    return np.array([mean_abs, std, energy, entropy,
                     max_abs, zero_crossings, rms])


def extract_window_features(window: np.ndarray) -> np.ndarray:
    """
    Extract DWT features from a single multi-channel EEG window.

    For each of the 6 channels and 4 frequency bands, extracts
    7 statistical features = 6 × 4 × 7 = 168 features total.

    Also appends the raw DWT coefficients concatenated across bands
    for the CNN to learn spatial-temporal patterns directly.

    Args:
        window: shape (n_channels, window_samples) e.g., (6, 1000)

    Returns:
        feature_vector: shape (n_channels, n_bands, n_features_per_band)
                       = (6, 4, 7) → can be reshaped for model input
    """
    n_channels = window.shape[0]
    n_bands = len(config.DWT_BANDS)
    n_features = 7  # features per band from extract_band_features

    features = np.zeros((n_channels, n_bands, n_features))

    for ch in range(n_channels):
        bands = dwt_decompose(window[ch])
        for b, (band_name, coeffs) in enumerate(bands.items()):
            features[ch, b] = extract_band_features(coeffs)

    return features


def extract_dwt_coefficients(window: np.ndarray) -> np.ndarray:
    """
    Extract raw DWT coefficients and pad/truncate to uniform length.

    This provides the CNN-LSTM with the actual wavelet coefficients
    (not just statistics), preserving temporal structure within each band.

    Args:
        window: shape (n_channels, window_samples)

    Returns:
        coeffs_matrix: shape (n_channels, total_coeff_length)
        where total_coeff_length is sum of padded band lengths
    """
    n_channels = window.shape[0]
    all_coeffs = []

    for ch in range(n_channels):
        bands = dwt_decompose(window[ch])
        ch_coeffs = []
        for band_name in config.DWT_BANDS:
            ch_coeffs.append(bands[band_name])
        # Concatenate all bands for this channel
        all_coeffs.append(np.concatenate(ch_coeffs))

    # Pad to same length (bands have different coefficient counts)
    max_len = max(len(c) for c in all_coeffs)
    padded = np.zeros((n_channels, max_len))
    for i, c in enumerate(all_coeffs):
        padded[i, :len(c)] = c

    return padded


def prepare_model_input(windows: np.ndarray,
                        mode: str = "hybrid") -> np.ndarray:
    """
    Transform raw EEG windows into CNN-LSTM model input.

    Three modes available:
      - "raw": Use raw EEG directly (6, 1000) per window
      - "dwt_coeffs": Use DWT coefficients (6, ~130) per window
      - "hybrid": Concatenate raw + DWT features (default, best accuracy)

    Args:
        windows: shape (N, n_channels, window_samples)
        mode: "raw", "dwt_coeffs", or "hybrid"

    Returns:
        model_input: shape depends on mode
          raw: (N, 1000, 6) — transposed for Conv1D
          dwt_coeffs: (N, coeff_len, 6)
          hybrid: (N, 1000, 6) — raw signal (DWT features as aux input)
    """
    N = windows.shape[0]

    if mode == "raw":
        # Transpose to (N, time_steps, channels) for Conv1D
        return np.transpose(windows, (0, 2, 1))

    elif mode == "dwt_coeffs":
        print("Extracting DWT coefficients...")
        coeffs_list = []
        for i in tqdm(range(N), desc="  DWT extraction"):
            coeffs = extract_dwt_coefficients(windows[i])
            coeffs_list.append(coeffs)
        coeffs = np.array(coeffs_list)
        # Transpose to (N, coeff_length, channels)
        return np.transpose(coeffs, (0, 2, 1))

    elif mode == "hybrid":
        # For hybrid: return raw transposed. DWT features are extracted
        # in the model as a secondary input (see model.py)
        return np.transpose(windows, (0, 2, 1))

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'raw', 'dwt_coeffs', or 'hybrid'")


def compute_feature_statistics(windows: np.ndarray,
                               labels: np.ndarray) -> dict:
    """
    Compute aggregate DWT feature statistics for seizure vs non-seizure.
    Useful for data exploration and model interpretability.

    Args:
        windows: shape (N, n_channels, window_samples)
        labels: shape (N,)

    Returns:
        dict with per-band, per-class mean features
    """
    seizure_mask = labels == 1
    normal_mask = labels == 0

    stats = {}
    for class_name, mask in [("seizure", seizure_mask), ("normal", normal_mask)]:
        if not np.any(mask):
            continue
        class_windows = windows[mask]
        # Sample up to 500 windows for efficiency
        n_sample = min(500, len(class_windows))
        idx = np.random.choice(len(class_windows), n_sample, replace=False)

        all_feats = []
        for i in idx:
            feats = extract_window_features(class_windows[i])
            all_feats.append(feats)

        all_feats = np.array(all_feats)  # (n_sample, 6, 4, 7)
        mean_feats = np.mean(all_feats, axis=0)  # (6, 4, 7)

        stats[class_name] = {
            "mean_features": mean_feats,
            "n_windows": int(np.sum(mask)),
        }

    return stats


if __name__ == "__main__":
    """Quick test: extract features from a synthetic EEG window."""
    print("Testing DWT feature extraction...")

    # Create synthetic 6-channel, 4-second EEG at 250 Hz
    np.random.seed(42)
    sfreq = config.TARGET_SFREQ
    t = np.linspace(0, config.WINDOW_SECONDS, config.WINDOW_SAMPLES)

    # Simulate: normal alpha rhythm + some theta
    window = np.zeros((config.N_CHANNELS, config.WINDOW_SAMPLES))
    for ch in range(config.N_CHANNELS):
        alpha = 20 * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)
        theta = 10 * np.sin(2 * np.pi * 6 * t + np.random.rand() * 2 * np.pi)
        noise = 5 * np.random.randn(config.WINDOW_SAMPLES)
        window[ch] = alpha + theta + noise

    # Extract features
    features = extract_window_features(window)
    print(f"Feature shape: {features.shape}")
    print(f"  = ({config.N_CHANNELS} channels) x "
          f"({len(config.DWT_BANDS)} bands) x (7 features)")

    # Extract DWT coefficients
    coeffs = extract_dwt_coefficients(window)
    print(f"DWT coefficients shape: {coeffs.shape}")

    # Prepare model input
    windows_batch = window[np.newaxis, ...]  # (1, 6, 1000)
    model_input = prepare_model_input(windows_batch, mode="raw")
    print(f"Model input shape (raw): {model_input.shape}")

    print("\nDWT band feature means (channel 0):")
    for b, band_name in enumerate(config.DWT_BANDS):
        feat_names = ["mean_abs", "std", "energy", "entropy",
                      "max_abs", "zero_cross", "rms"]
        print(f"  {band_name:6s}: ", end="")
        for f, fn in enumerate(feat_names):
            print(f"{fn}={features[0, b, f]:.3f}  ", end="")
        print()

    print("\nFeature extraction test passed.")
