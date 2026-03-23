"""
EEG Seizure Prediction — WebSocket Inference Server

Receives real-time EEG data (from ESP32-C3 or simulator),
runs the CNN-LSTM model, and pushes predictions to the
React dashboard via WebSocket.

Architecture:
  ESP32-C3 (BLE/WiFi) → This Server → React Dashboard
       raw samples        predictions    visualization

Usage:
  # Start with simulated EEG data (no hardware needed)
  python server.py --simulate

  # Start waiting for real ESP32 data
  python server.py

  # Custom model path
  python server.py --model models/seizure_predictor_v4.keras --simulate
"""

import asyncio
import json
import time
import argparse
import logging
from pathlib import Path
from collections import deque
from dataclasses import dataclass, asdict

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eeg-server")


# ── Configuration ────────────────────────────────────────────────

SFREQ = 250              # Samples per second (matching ADS1299)
N_CHANNELS = 6           # Fp1, Fp2, C3, C4, T7, T8
WINDOW_SEC = 4           # Seconds per analysis window
STEP_SEC = 1             # Slide window every N seconds (3s overlap)
WINDOW_SAMPLES = SFREQ * WINDOW_SEC   # 1000
STEP_SAMPLES = SFREQ * STEP_SEC       # 250
THRESHOLD = 0.7          # Prediction threshold (from v3 sweep)

WS_HOST = "0.0.0.0"
WS_PORT = 8765
HTTP_PORT = 8766         # For health check / status


# ── Data Structures ──────────────────────────────────────────────

@dataclass
class Prediction:
    timestamp: float
    probability: float
    risk_level: str       # "normal", "elevated", "warning", "critical"
    threshold: float
    window_index: int
    channels_summary: dict  # per-channel stats for dashboard


@dataclass
class ServerState:
    is_recording: bool = False
    total_windows: int = 0
    total_predictions: int = 0
    seizure_warnings: int = 0
    start_time: float = 0.0
    current_risk: str = "normal"
    model_loaded: bool = False
    connected_clients: int = 0


# ── Model Wrapper ────────────────────────────────────────────────

class SeizurePredictor:
    """Wraps the trained CNN-LSTM model for real-time inference."""

    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path
        self._use_dwt = True  # v3+ uses hybrid DWT features

    def load(self):
        """Load the trained Keras model."""
        if self.model_path and Path(self.model_path).exists():
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
            self.model = tf.keras.models.load_model(self.model_path)
            log.info(f"Model loaded: {self.model_path}")
            log.info(f"  Input shape: {self.model.input_shape}")
            log.info(f"  Parameters: {self.model.count_params():,}")
            return True
        else:
            log.warning(f"Model not found at {self.model_path} — running in dummy mode")
            return False

    def predict(self, window: np.ndarray) -> float:
        """
        Run inference on a single EEG window.

        Args:
            window: shape (N_CHANNELS, WINDOW_SAMPLES) = (6, 1000)

        Returns:
            Seizure probability [0, 1]
        """
        if self.model is None:
            return np.random.uniform(0.0, 0.15)

        try:
            if self._use_dwt:
                from features import prepare_model_input
                x = prepare_model_input(
                    window[np.newaxis, :, :], mode="hybrid"
                )
            else:
                x = window[np.newaxis, :, :].transpose(0, 2, 1)

            prob = self.model.predict(x, verbose=0)[0][0]
            return float(prob)
        except Exception as e:
            log.error(f"Inference error: {e}")
            return 0.0

    @property
    def latency_estimate_ms(self) -> float:
        """Rough inference latency for the dashboard."""
        return 50.0 if self.model else 1.0


# ── Signal Processing ────────────────────────────────────────────

class SignalProcessor:
    """Buffers incoming samples and produces analysis windows."""

    def __init__(self):
        self.buffer = np.zeros((N_CHANNELS, 0))
        self.window_count = 0

    def add_samples(self, samples: np.ndarray):
        """
        Add new samples to the buffer.

        Args:
            samples: shape (N_CHANNELS, n_new_samples)
        """
        self.buffer = np.concatenate([self.buffer, samples], axis=1)

    def get_windows(self) -> list[np.ndarray]:
        """
        Extract all ready windows from the buffer.
        Returns list of (N_CHANNELS, WINDOW_SAMPLES) arrays.
        """
        windows = []
        while self.buffer.shape[1] >= WINDOW_SAMPLES:
            window = self.buffer[:, :WINDOW_SAMPLES].copy()
            windows.append(window)
            self.buffer = self.buffer[:, STEP_SAMPLES:]
            self.window_count += 1
        return windows

    def reset(self):
        self.buffer = np.zeros((N_CHANNELS, 0))
        self.window_count = 0


# ── EEG Simulator ────────────────────────────────────────────────

class EEGSimulator:
    """
    Generates realistic-looking EEG data for testing without hardware.
    Simulates normal activity with periodic pre-ictal episodes.
    """

    def __init__(self, seizure_interval_sec: int = 120):
        self.sfreq = SFREQ
        self.n_channels = N_CHANNELS
        self.seizure_interval = seizure_interval_sec
        self.time_offset = 0.0
        self.rng = np.random.default_rng(42)

    def generate_chunk(self, n_samples: int = None) -> np.ndarray:
        """
        Generate a chunk of simulated EEG.

        Returns shape (N_CHANNELS, n_samples)
        """
        if n_samples is None:
            n_samples = STEP_SAMPLES  # 250 samples = 1 second

        t = np.arange(n_samples) / self.sfreq + self.time_offset
        self.time_offset += n_samples / self.sfreq

        data = np.zeros((self.n_channels, n_samples))

        for ch in range(self.n_channels):
            alpha = 0.5 * np.sin(2 * np.pi * 10 * t + self.rng.uniform(0, 2 * np.pi))
            theta = 0.3 * np.sin(2 * np.pi * 6 * t + self.rng.uniform(0, 2 * np.pi))
            beta = 0.2 * np.sin(2 * np.pi * 20 * t + self.rng.uniform(0, 2 * np.pi))
            noise = 0.1 * self.rng.standard_normal(n_samples)
            data[ch] = alpha + theta + beta + noise

        cycle_pos = self.time_offset % self.seizure_interval
        preictal_start = self.seizure_interval - 130  # ~2 min before
        seizure_start = self.seizure_interval - 10

        if cycle_pos > preictal_start and cycle_pos < seizure_start:
            intensity = (cycle_pos - preictal_start) / (seizure_start - preictal_start)
            for ch in range(self.n_channels):
                spike = intensity * 2.0 * np.sin(2 * np.pi * 3 * t)
                data[ch] += spike * (0.5 + 0.5 * self.rng.random())

        elif cycle_pos >= seizure_start:
            for ch in range(self.n_channels):
                spike_wave = 3.0 * np.sin(2 * np.pi * 3 * t)
                data[ch] += spike_wave

        return data

    @property
    def current_phase(self) -> str:
        cycle_pos = self.time_offset % self.seizure_interval
        preictal_start = self.seizure_interval - 130
        seizure_start = self.seizure_interval - 10
        if cycle_pos >= seizure_start:
            return "ictal"
        elif cycle_pos > preictal_start:
            return "pre-ictal"
        return "normal"


# ── Risk Assessment ──────────────────────────────────────────────

class RiskAssessor:
    """
    Smooths predictions over time to reduce alarm flicker.
    Uses a rolling window of recent predictions.
    """

    def __init__(self, window_size: int = 5):
        self.history = deque(maxlen=window_size)

    def update(self, probability: float) -> str:
        self.history.append(probability)
        avg = np.mean(list(self.history))

        if avg >= 0.8:
            return "critical"
        elif avg >= 0.6:
            return "warning"
        elif avg >= 0.3:
            return "elevated"
        return "normal"

    def reset(self):
        self.history.clear()


# ── WebSocket Server ─────────────────────────────────────────────

class InferenceServer:
    """Main server that ties everything together."""

    def __init__(self, model_path: str, simulate: bool = False):
        self.predictor = SeizurePredictor(model_path)
        self.processor = SignalProcessor()
        self.risk = RiskAssessor()
        self.state = ServerState()
        self.simulate = simulate
        self.simulator = EEGSimulator() if simulate else None
        self.clients: set = set()
        self._running = False

    async def start(self):
        """Start the WebSocket server."""
        try:
            import websockets
        except ImportError:
            log.error("websockets not installed. Run: pip install websockets")
            return

        self.state.model_loaded = self.predictor.load()

        log.info("=" * 50)
        log.info("EEG Seizure Prediction Server")
        log.info("=" * 50)
        log.info(f"  WebSocket:  ws://{WS_HOST}:{WS_PORT}")
        log.info(f"  Model:      {'loaded' if self.state.model_loaded else 'dummy mode'}")
        log.info(f"  Simulate:   {self.simulate}")
        log.info(f"  Threshold:  {THRESHOLD}")
        log.info(f"  Window:     {WINDOW_SEC}s ({WINDOW_SAMPLES} samples)")
        log.info(f"  Step:       {STEP_SEC}s ({STEP_SAMPLES} samples)")
        log.info("=" * 50)

        self._running = True

        async with websockets.serve(
            self._handle_client, WS_HOST, WS_PORT,
            ping_interval=20, ping_timeout=10,
        ):
            if self.simulate:
                asyncio.create_task(self._simulation_loop())
            await asyncio.Future()  # Run forever

    async def _handle_client(self, websocket):
        """Handle a new WebSocket client connection."""
        self.clients.add(websocket)
        self.state.connected_clients = len(self.clients)
        client_addr = websocket.remote_address
        log.info(f"Client connected: {client_addr} ({self.state.connected_clients} total)")

        try:
            await websocket.send(json.dumps({
                "type": "server_info",
                "data": {
                    "model_loaded": self.state.model_loaded,
                    "simulate": self.simulate,
                    "threshold": THRESHOLD,
                    "sfreq": SFREQ,
                    "n_channels": N_CHANNELS,
                    "window_sec": WINDOW_SEC,
                    "channel_names": ["Fp1", "Fp2", "C3", "C4", "T7", "T8"],
                }
            }))

            async for message in websocket:
                await self._handle_message(message, websocket)

        except Exception as e:
            log.debug(f"Client disconnected: {client_addr} ({e})")
        finally:
            self.clients.discard(websocket)
            self.state.connected_clients = len(self.clients)
            log.info(f"Client removed: {client_addr} ({self.state.connected_clients} total)")

    async def _handle_message(self, message: str, websocket):
        """Process incoming messages from clients."""
        try:
            msg = json.loads(message)
        except json.JSONDecodeError:
            return

        msg_type = msg.get("type", "")

        if msg_type == "eeg_data":
            samples = np.array(msg["data"])  # (N_CHANNELS, n_samples)
            self.processor.add_samples(samples)
            await self._process_windows()

        elif msg_type == "start_recording":
            self.state.is_recording = True
            self.state.start_time = time.time()
            self.processor.reset()
            self.risk.reset()
            log.info("Recording started")
            await self._broadcast({"type": "recording_started"})

        elif msg_type == "stop_recording":
            self.state.is_recording = False
            log.info("Recording stopped")
            await self._broadcast({"type": "recording_stopped"})

        elif msg_type == "get_status":
            await websocket.send(json.dumps({
                "type": "status",
                "data": asdict(self.state),
            }))

        elif msg_type == "set_threshold":
            global THRESHOLD
            THRESHOLD = float(msg.get("value", 0.7))
            log.info(f"Threshold updated: {THRESHOLD}")

    async def _process_windows(self):
        """Process any ready windows through the model."""
        windows = self.processor.get_windows()

        for window in windows:
            start = time.perf_counter()
            probability = self.predictor.predict(window)
            latency = (time.perf_counter() - start) * 1000

            risk_level = self.risk.update(probability)
            self.state.total_windows += 1
            self.state.total_predictions += 1
            self.state.current_risk = risk_level

            if risk_level in ("warning", "critical"):
                self.state.seizure_warnings += 1

            ch_names = ["Fp1", "Fp2", "C3", "C4", "T7", "T8"]
            ch_stats = {}
            band_powers = {}
            band_ranges = {
                "delta": (0.5, 4),
                "theta": (4, 8),
                "alpha": (8, 13),
                "beta": (13, 30),
                "gamma": (30, 45),
            }

            for i, name in enumerate(ch_names):
                sig = window[i]
                ch_stats[name] = {
                    "mean_uv": round(float(np.mean(sig)), 2),
                    "std_uv": round(float(np.std(sig)), 2),
                    "peak_uv": round(float(np.max(np.abs(sig))), 2),
                }

                from scipy.signal import welch
                freqs, psd = welch(sig, fs=SFREQ, nperseg=min(256, len(sig)))
                ch_bands = {}
                for band_name, (lo, hi) in band_ranges.items():
                    mask = (freqs >= lo) & (freqs < hi)
                    ch_bands[band_name] = round(float(np.mean(psd[mask])) if mask.any() else 0.0, 4)
                band_powers[name] = ch_bands

            prediction = {
                "type": "prediction",
                "data": {
                    "timestamp": time.time(),
                    "probability": round(probability, 4),
                    "risk_level": risk_level,
                    "threshold": THRESHOLD,
                    "window_index": self.state.total_windows,
                    "latency_ms": round(latency, 1),
                    "channels": ch_stats,
                    "band_powers": band_powers,
                }
            }

            waveform = {
                "type": "waveform",
                "data": {
                    "timestamp": time.time(),
                    "channels": {
                        name: window[i].tolist()
                        for i, name in enumerate(ch_names)
                    },
                    "sfreq": SFREQ,
                }
            }

            await self._broadcast(prediction)
            await self._broadcast(waveform)

            level_icon = {"normal": ".", "elevated": "+", "warning": "!", "critical": "!!!"}
            log.info(
                f"Window {self.state.total_windows:>5d} | "
                f"prob={probability:.3f} | risk={risk_level:<8s} "
                f"{level_icon.get(risk_level, '')} | "
                f"{latency:.0f}ms"
            )

    async def _broadcast(self, message: dict):
        """Send a message to all connected clients."""
        if not self.clients:
            return
        data = json.dumps(message)
        dead = set()
        for ws in self.clients:
            try:
                await ws.send(data)
            except Exception:
                dead.add(ws)
        self.clients -= dead

    async def _simulation_loop(self):
        """Generate simulated EEG data in real-time."""
        log.info("Simulator started — generating EEG every 1 second")
        self.state.is_recording = True
        self.state.start_time = time.time()

        while self._running:
            chunk = self.simulator.generate_chunk()
            self.processor.add_samples(chunk)
            await self._process_windows()

            phase = self.simulator.current_phase
            if phase != "normal":
                log.info(f"  Simulator phase: {phase}")

            await asyncio.sleep(1.0)


# ── Entry Point ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EEG Seizure Prediction Inference Server"
    )
    parser.add_argument(
        "--model", type=str,
        default="models/seizure_predictor_v3.keras",
        help="Path to trained .keras model",
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Generate simulated EEG data (no hardware needed)",
    )
    parser.add_argument(
        "--port", type=int, default=WS_PORT,
        help=f"WebSocket port (default: {WS_PORT})",
    )
    parser.add_argument(
        "--threshold", type=float, default=THRESHOLD,
        help=f"Prediction threshold (default: {THRESHOLD})",
    )
    args = parser.parse_args()

    server = InferenceServer(
        model_path=args.model,
        simulate=args.simulate,
    )

    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        log.info("Server stopped.")


if __name__ == "__main__":
    main()
