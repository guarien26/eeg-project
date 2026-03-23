#!/bin/bash
# Setup script for EEG Seizure Dashboard
# Run this once after copying the project folder

echo "=== EEG Seizure Dashboard Setup ==="
echo ""

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "Node.js not found. Install it from https://nodejs.org"
    echo "Or with Homebrew: brew install node"
    exit 1
fi

echo "Node.js: $(node --version)"
echo "npm: $(npm --version)"
echo ""

# Install dependencies
echo "Installing dependencies..."
npm install

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start the dashboard:"
echo "  npm run dev"
echo ""
echo "Then open http://localhost:3000 in your browser."
echo ""
echo "To connect to the inference server:"
echo "  1. In another terminal, run: python server.py --simulate"
echo "  2. The dashboard will auto-connect and show 'Server Connected'"
echo ""
