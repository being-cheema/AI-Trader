#!/bin/bash

# Run AI trading agent for Indian stock market (NSE/BSE - Nifty 50)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

echo "🤖 Starting AI trading agent (India stock mode)..."

python main.py configs/india_stock_config.json  # Run India stock configuration

echo "✅ AI-Trader stopped"
