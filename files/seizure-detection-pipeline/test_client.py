"""
Test client for the EEG inference server.

Connects via WebSocket, displays predictions in real-time.
Use this to verify the server works before building the React dashboard.

Usage:
  # In terminal 1: start server with simulator
  python server.py --simulate

  # In terminal 2: run this client
  python test_client.py
"""

import asyncio
import json
import sys

try:
    import websockets
except ImportError:
    print("Install websockets: pip install websockets --break-system-packages")
    sys.exit(1)


RISK_COLORS = {
    "normal": "\033[92m",    # green
    "elevated": "\033[93m",  # yellow
    "warning": "\033[91m",   # red
    "critical": "\033[95m",  # magenta
}
RESET = "\033[0m"


async def connect():
    uri = "ws://localhost:8765"
    print(f"Connecting to {uri}...")

    async with websockets.connect(uri) as ws:
        print("Connected!\n")

        prediction_count = 0
        async for message in ws:
            msg = json.loads(message)

            if msg["type"] == "server_info":
                info = msg["data"]
                print(f"Server info:")
                print(f"  Model loaded: {info['model_loaded']}")
                print(f"  Simulating:   {info['simulate']}")
                print(f"  Threshold:    {info['threshold']}")
                print(f"  Channels:     {', '.join(info['channel_names'])}")
                print(f"  Sample rate:  {info['sfreq']} Hz")
                print(f"\nWaiting for predictions...\n")
                print(f"{'#':>5s}  {'Prob':>6s}  {'Risk':<10s}  {'Latency':>8s}")
                print("-" * 38)

            elif msg["type"] == "prediction":
                d = msg["data"]
                prediction_count += 1
                risk = d["risk_level"]
                color = RISK_COLORS.get(risk, "")

                print(
                    f"{d['window_index']:>5d}  "
                    f"{d['probability']:>6.3f}  "
                    f"{color}{risk:<10s}{RESET}  "
                    f"{d['latency_ms']:>6.1f}ms"
                )

            elif msg["type"] == "recording_started":
                print("\n>>> Recording started")

            elif msg["type"] == "recording_stopped":
                print("\n>>> Recording stopped")


def main():
    try:
        asyncio.run(connect())
    except KeyboardInterrupt:
        print("\nDisconnected.")
    except ConnectionRefusedError:
        print(f"Could not connect. Is the server running?")
        print(f"  Start it with: python server.py --simulate")


if __name__ == "__main__":
    main()
