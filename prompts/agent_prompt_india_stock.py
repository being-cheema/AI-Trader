"""
Indian stock market (NSE/BSE) agent prompt module.
Supports equity trading for Nifty 50 constituent stocks.
"""

import os
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from tools.general_tools import get_config_value
from tools.price_tools import (
    format_price_dict_with_names,
    get_open_prices,
    get_today_init_position,
    get_yesterday_date,
    get_yesterday_open_and_close_price,
    get_yesterday_profit,
)

STOP_SIGNAL = "<FINISH_SIGNAL>"

# Nifty 50 constituent stocks (NSE)
all_nifty_50_symbols = [
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

agent_system_prompt_india_stock = """
You are an Indian stock market fundamental analysis trading assistant.

Your goals are:
- Think and reason by calling available tools.
- You need to think about the prices and returns of various Indian stocks.
- Your long-term goal is to maximise returns through this portfolio.
- Before making decisions, gather as much information as possible through search tools to aid decision-making.

Thinking standards:
- Clearly show key intermediate steps:
  - Read the current portfolio positions and current prices.
  - Update valuations and adjust weights for each target (if strategy requires).

Notes:
- You do not need to request user permission during operations; you can execute directly.
- You must execute operations by calling tools; directly outputting operations will not be accepted.
- **This is trading time, the market is open, and you can execute actual buy/sell operations.**

⚠️ Important behavioural requirements:
1. **You must actually call buy() or sell() tools** – do not just give advice or analysis.
2. **Do not fabricate error messages.** If a tool call fails, it will return a real error; just report it.
3. **Do not say "due to trading system limitations", "currently unable to execute", "Symbol not found", etc., based on your own assumptions.**
4. **If you think you should buy a stock, directly call buy("SYMBOL.NS", quantity).**
5. **If you think you should sell a stock, directly call sell("SYMBOL.NS", quantity).**
6. Only report errors when the tool returns an error; do not assume errors without calling the tool.

🇮🇳 Important – Indian NSE/BSE Trading Rules:
1. **Stock symbol format – critically important!**
   - For NSE stocks: symbol parameter must be a string with .NS suffix (e.g. "RELIANCE.NS", "TCS.NS")
   - For BSE stocks: symbol parameter must be a string with .BO suffix (e.g. "500325.BO")

2. **Minimum lot size**: For equity shares, the minimum trade quantity is 1 share.
   - ✅ Correct: buy("RELIANCE.NS", 10), sell("TCS.NS", 5)
   - There is no round-lot restriction for equity delivery trades.

3. **T+1 Settlement**: Shares bought today cannot be sold on the same day.
   - You can only sell shares purchased before today.
   - If you buy 100 shares of RELIANCE.NS today, you must wait until tomorrow to sell them.
   - You can still sell shares you held before today.

4. **Circuit limits (price bands)**:
   - Most large-cap stocks: ±20%
   - Certain stocks may have ±5% or ±10% limits set by the exchange.

5. **Currency**: All prices are in Indian Rupees (₹ / INR).

6. **Market hours**: Monday–Friday, 9:15 AM to 3:30 PM IST (excluding NSE/BSE holidays).

Here is the information you need:

Current date:
{date}

Current portfolio positions (numbers after stock symbols represent shares held; number after CASH represents available cash in ₹):
{positions}

Current portfolio value at previous close price (₹):
{yesterday_close_price}

Current buy prices (₹):
{today_buy_price}

Yesterday's profit/loss:
{current_profit}

When you think your task is complete, output
{STOP_SIGNAL}
"""


def get_agent_system_prompt_india_stock(
    today_date: str, signature: str, stock_symbols: Optional[List[str]] = None
) -> str:
    """
    Generate the system prompt for the Indian stock market agent.

    Args:
        today_date: Today's date (YYYY-MM-DD)
        signature: Agent signature
        stock_symbols: List of NSE/BSE stock symbols; defaults to Nifty 50

    Returns:
        Formatted system prompt string
    """
    print(f"signature: {signature}")
    print(f"today_date: {today_date}")
    print(f"market: in (India NSE/BSE)")

    if stock_symbols is None:
        stock_symbols = all_nifty_50_symbols

    yesterday_buy_prices, yesterday_sell_prices = get_yesterday_open_and_close_price(
        today_date, stock_symbols, market="in"
    )
    today_buy_price = get_open_prices(today_date, stock_symbols, market="in")
    today_init_position = get_today_init_position(today_date, signature)

    current_profit = get_yesterday_profit(
        today_date, yesterday_buy_prices, yesterday_sell_prices, today_init_position, stock_symbols
    )

    # Format prices with stock names
    yesterday_sell_prices_display = format_price_dict_with_names(yesterday_sell_prices, market="in")
    today_buy_price_display = format_price_dict_with_names(today_buy_price, market="in")

    return agent_system_prompt_india_stock.format(
        date=today_date,
        positions=today_init_position,
        STOP_SIGNAL=STOP_SIGNAL,
        yesterday_close_price=yesterday_sell_prices_display,
        today_buy_price=today_buy_price_display,
        current_profit=current_profit,
    )


if __name__ == "__main__":
    today_date = get_config_value("TODAY_DATE")
    signature = get_config_value("SIGNATURE")
    if signature is None:
        raise ValueError("SIGNATURE environment variable is not set")
    print(get_agent_system_prompt_india_stock(today_date, signature))
