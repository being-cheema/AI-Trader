"""
Fetch daily price data for Indian stock market (NSE) using yfinance.
Downloads data for Nifty 50 constituent stocks and saves individual JSON files
in the same format as the Alpha Vantage data used for other markets.
"""

import json
import os
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# Nifty 50 constituent stocks (NSE symbols with .NS suffix)
NIFTY_50_SYMBOLS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "AXISBANK.NS",
    "MARUTI.NS",
    "NTPC.NS",
    "TITAN.NS",
    "WIPRO.NS",
    "TECHM.NS",
    "ULTRACEMCO.NS",
    "TATAMOTORS.NS",
    "M&M.NS",
    "BAJFINANCE.NS",
    "ASIANPAINT.NS",
    "TATASTEEL.NS",
    "POWERGRID.NS",
    "SUNPHARMA.NS",
    "ONGC.NS",
    "COALINDIA.NS",
    "NESTLEIND.NS",
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "BAJAJ-AUTO.NS",
    "BAJAJFINSV.NS",
    "BEL.NS",
    "BPCL.NS",
    "BRITANNIA.NS",
    "CIPLA.NS",
    "DRREDDY.NS",
    "EICHERMOT.NS",
    "GRASIM.NS",
    "HCLTECH.NS",
    "HDFCLIFE.NS",
    "HEROMOTOCO.NS",
    "HINDALCO.NS",
    "INDUSINDBK.NS",
    "JSWSTEEL.NS",
    "SBILIFE.NS",
    "SHRIRAMFIN.NS",
    "TATACONSUM.NS",
    "TRENT.NS",
    "APOLLOHOSP.NS",
]


def load_existing_data(filepath: str):
    """Load existing data file if it exists."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None
    return None


def merge_data(existing_data: dict, new_data: dict) -> dict:
    """Merge new data into existing data, preserving existing dates."""
    if existing_data is None or "Time Series (Daily)" not in existing_data:
        return new_data

    existing_dates = existing_data["Time Series (Daily)"]
    new_dates = new_data["Time Series (Daily)"]

    merged_dates = existing_dates.copy()
    for date in new_dates:
        if date not in merged_dates:
            merged_dates[date] = new_dates[date]

    # Sort descending (latest first)
    sorted_dates = OrderedDict(sorted(merged_dates.items(), key=lambda x: x[0], reverse=True))

    merged_data = existing_data.copy()
    merged_data["Time Series (Daily)"] = sorted_dates
    if sorted_dates:
        merged_data["Meta Data"]["3. Last Refreshed"] = list(sorted_dates.keys())[0]

    return merged_data


def get_daily_price(symbol: str, start_date: str = "2025-01-01") -> None:
    """
    Fetch daily OHLCV data for a single NSE symbol using yfinance and save as JSON.

    Args:
        symbol: NSE ticker symbol with .NS suffix (e.g. "RELIANCE.NS")
        start_date: Start date for data fetch in YYYY-MM-DD format
    """
    output_dir = Path(__file__).parent / "india_stock_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"daily_prices_{symbol}.json"

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, interval="1d")

        if df.empty:
            print(f"⚠️  No data returned for {symbol}")
            return

        # Remove timezone info from index for consistent date formatting
        df.index = df.index.tz_localize(None) if df.index.tzinfo is None else df.index.tz_convert(None)

        # Get stock name from yfinance metadata
        try:
            stock_info = ticker.info
            stock_name = stock_info.get("longName") or stock_info.get("shortName") or symbol
        except Exception:
            stock_name = symbol

        latest_date = df.index[-1].strftime("%Y-%m-%d")

        # Build time series in Alpha Vantage-compatible format
        time_series = OrderedDict()
        for ts, row in df.iterrows():
            date_str = ts.strftime("%Y-%m-%d")
            if date_str == latest_date:
                # For latest date, only include buy price to avoid future leakage
                time_series[date_str] = {"1. buy price": f"{row['Open']:.2f}"}
            else:
                time_series[date_str] = {
                    "1. buy price": f"{row['Open']:.2f}",
                    "2. high": f"{row['High']:.2f}",
                    "3. low": f"{row['Low']:.2f}",
                    "4. sell price": f"{row['Close']:.2f}",
                    "5. volume": str(int(row["Volume"])) if row["Volume"] == row["Volume"] else "0",
                }

        # Sort descending (latest first)
        sorted_series = OrderedDict(sorted(time_series.items(), key=lambda x: x[0], reverse=True))

        new_data = {
            "Meta Data": {
                "1. Information": "Daily Prices (buy price, high, low, sell price) and Volumes",
                "2. Symbol": symbol,
                "2.1. Name": stock_name,
                "3. Last Refreshed": latest_date,
                "4. Output Size": "Compact",
                "5. Time Zone": "Asia/Kolkata",
            },
            "Time Series (Daily)": sorted_series,
        }

        # Merge with existing data
        existing_data = load_existing_data(str(output_file))
        merged = merge_data(existing_data, new_data)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=4)

        print(f"✅ {symbol} ({stock_name}): {len(sorted_series)} trading days saved")

    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")


if __name__ == "__main__":
    start_date = os.getenv("INDIA_STOCK_START_DATE", "2025-01-01")
    print(f"📊 Fetching Nifty 50 daily prices from {start_date} ...")
    print(f"📁 Output: data/india_stock/india_stock_data/")
    print("=" * 60)

    for i, symbol in enumerate(NIFTY_50_SYMBOLS):
        get_daily_price(symbol, start_date=start_date)
        # Small delay to be polite to the API
        if i < len(NIFTY_50_SYMBOLS) - 1:
            time.sleep(0.5)

    print("=" * 60)
    print(f"✅ Done. Fetched data for {len(NIFTY_50_SYMBOLS)} Nifty 50 stocks.")
