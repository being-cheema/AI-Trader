#!/bin/bash

# Indian stock market (NSE/BSE) data preparation
# Fetches Nifty 50 daily price data using yfinance and merges into JSONL

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

cd data/india_stock

python get_daily_price_yfinance.py
python merge_jsonl.py

cd ../..
