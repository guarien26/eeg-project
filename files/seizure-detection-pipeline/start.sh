#!/bin/bash
#
# EEG Seizure Prediction System — Single Start Command
#
# Usage:
#   ./start.sh              Start with real hardware (waits for ESP32)
#   ./start.sh --simulate   Start with simulated EEG data (for testing)
#
# Starts both the Python inference server and React dashboard,
# then opens the browser automatically.
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_DIR="$SCRIPT_DIR"
DASH_DIR="$(dirname "$SCRIPT_DIR")/eeg-dashboard"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   EEG Seizure Prediction System              ║${NC}"
echo -e "${GREEN}║   Portable AI-Enabled EEG for Seizure Detection ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════╝${NC}"
echo ""

# Check dependencies
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "Error: npm not found. Install Node.js: brew install node"
    exit 1
fi

if [ ! -d "$DASH_DIR/node_modules" ]; then
    echo -e "${YELLOW}First run detected — installing dashboard dependencies...${NC}"
    cd "$DASH_DIR" && npm install
    echo ""
fi

# Pass through any flags (like --simulate)
SIM_FLAG=""
if [ "$1" = "--simulate" ]; then
    SIM_FLAG="--simulate"
    echo -e "${YELLOW}Mode: Simulated EEG (no hardware required)${NC}"
else
    echo -e "${BLUE}Mode: Live hardware (waiting for ESP32 connection)${NC}"
fi
echo ""

# Activate venv and start the inference server
echo -e "${BLUE}[1/2] Starting inference server...${NC}"
cd "$SERVER_DIR"
source venv/bin/activate
python3 server.py $SIM_FLAG &
SERVER_PID=$!
sleep 2

# Check if server started
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "Error: Server failed to start"
    exit 1
fi

# Start the React dashboard
echo -e "${BLUE}[2/2] Starting dashboard...${NC}"
cd "$DASH_DIR"
npm run dev -- --open &
DASH_PID=$!
sleep 3

echo ""
echo -e "${GREEN}══════════════════════════════════════════════${NC}"
echo -e "${GREEN}  System running!${NC}"
echo ""
echo -e "  Dashboard:  ${BLUE}http://localhost:3000${NC}"
echo -e "  Server:     ${BLUE}ws://localhost:8765${NC}"
echo ""
echo -e "  1. Dashboard opens in your browser automatically"
echo -e "  2. Press ${GREEN}Start Recording${NC} to begin monitoring"
echo -e "  3. Press ${YELLOW}Ctrl+C${NC} here to stop everything"
echo -e "${GREEN}══════════════════════════════════════════════${NC}"
echo ""

# Clean shutdown on Ctrl+C
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $DASH_PID 2>/dev/null
    kill $SERVER_PID 2>/dev/null
    wait $DASH_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    echo "Done."
    exit 0
}

trap cleanup INT TERM

# Keep running until Ctrl+C
wait
