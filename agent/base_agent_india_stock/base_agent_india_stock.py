"""
BaseAgentIndiaStock class - Indian stock market (NSE/BSE) trading agent base class
Supports NSE equity trading for Nifty 50 constituent stocks.
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


class DeepSeekChatOpenAI(ChatOpenAI):
    """Custom ChatOpenAI wrapper for DeepSeek API compatibility."""

    def _generate(self, messages: list, stop: Optional[list] = None, **kwargs):
        result = super()._generate(messages, stop, **kwargs)
        for generation in result.generations:
            for gen in generation:
                if hasattr(gen, "message") and hasattr(gen.message, "additional_kwargs"):
                    tool_calls = gen.message.additional_kwargs.get("tool_calls")
                    if tool_calls:
                        for tool_call in tool_calls:
                            if "function" in tool_call and "arguments" in tool_call["function"]:
                                args = tool_call["function"]["arguments"]
                                if isinstance(args, str):
                                    try:
                                        tool_call["function"]["arguments"] = json.loads(args)
                                    except json.JSONDecodeError:
                                        pass
        return result

    async def _agenerate(self, messages: list, stop: Optional[list] = None, **kwargs):
        result = await super()._agenerate(messages, stop, **kwargs)
        for generation in result.generations:
            for gen in generation:
                if hasattr(gen, "message") and hasattr(gen.message, "additional_kwargs"):
                    tool_calls = gen.message.additional_kwargs.get("tool_calls")
                    if tool_calls:
                        for tool_call in tool_calls:
                            if "function" in tool_call and "arguments" in tool_call["function"]:
                                args = tool_call["function"]["arguments"]
                                if isinstance(args, str):
                                    try:
                                        tool_call["function"]["arguments"] = json.loads(args)
                                    except json.JSONDecodeError:
                                        pass
        return result


from prompts.agent_prompt_india_stock import (STOP_SIGNAL,
                                               get_agent_system_prompt_india_stock)
from tools.general_tools import (extract_conversation, extract_tool_messages,
                                  get_config_value, write_config_value)
from tools.price_tools import add_no_trade_record

load_dotenv()


class BaseAgentIndiaStock:
    """
    Indian stock market (NSE/BSE) trading agent base class.

    Supports:
    - NSE equity trading for Nifty 50 constituent stocks
    - T+1 settlement rules
    - Indian market trading hours (9:15 AM - 3:30 PM IST)
    """

    # Default Nifty 50 constituent stocks (NSE)
    DEFAULT_NIFTY50_SYMBOLS = [
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

    def __init__(
        self,
        signature: str,
        basemodel: str,
        stock_symbols: Optional[List[str]] = None,
        mcp_config: Optional[Dict[str, Dict[str, Any]]] = None,
        log_path: Optional[str] = None,
        max_steps: int = 10,
        max_retries: int = 3,
        base_delay: float = 0.5,
        openai_base_url: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        initial_cash: float = 1000000.0,  # Default ₹10,00,000 (10 lakh INR)
        init_date: str = "2025-10-09",
        market: str = "in",  # Accept but always uses "in"
    ):
        """
        Initialize BaseAgentIndiaStock.

        Args:
            signature: Agent signature/name
            basemodel: Base model name
            stock_symbols: List of stock symbols (NSE format, e.g. RELIANCE.NS), defaults to Nifty 50
            mcp_config: MCP tool configuration
            log_path: Log path, defaults to ./data/agent_data_india_stock
            max_steps: Maximum reasoning steps
            max_retries: Maximum retry attempts
            base_delay: Base delay time for retries
            openai_base_url: OpenAI API base URL
            openai_api_key: OpenAI API key
            initial_cash: Initial cash amount in INR (default: ₹10,00,000)
            init_date: Initialization date
            market: Market type (always uses "in" for Indian market)
        """
        self.signature = signature
        self.basemodel = basemodel
        self.market = "in"  # Hard-coded to Indian market

        # Default to Nifty 50
        if stock_symbols is None:
            self.stock_symbols = self.DEFAULT_NIFTY50_SYMBOLS
        else:
            self.stock_symbols = stock_symbols

        self.max_steps = max_steps
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.initial_cash = initial_cash
        self.init_date = init_date

        # Set MCP configuration
        self.mcp_config = mcp_config or self._get_default_mcp_config()

        # Set log path
        self.base_log_path = log_path or "./data/agent_data_india_stock"

        # Set OpenAI configuration
        if openai_base_url is None:
            self.openai_base_url = os.getenv("OPENAI_API_BASE")
        else:
            self.openai_base_url = openai_base_url
        if openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        else:
            self.openai_api_key = openai_api_key

        # Initialize components
        self.client: Optional[MultiServerMCPClient] = None
        self.tools: Optional[List] = None
        self.model: Optional[ChatOpenAI] = None
        self.agent: Optional[Any] = None

        # Data paths
        self.data_path = os.path.join(self.base_log_path, self.signature)
        self.position_file = os.path.join(self.data_path, "position", "position.jsonl")

    def _get_default_mcp_config(self) -> Dict[str, Dict[str, Any]]:
        """Get default MCP configuration."""
        return {
            "math": {
                "transport": "streamable_http",
                "url": f"http://localhost:{os.getenv('MATH_HTTP_PORT', '8000')}/mcp",
            },
            "stock_local": {
                "transport": "streamable_http",
                "url": f"http://localhost:{os.getenv('GETPRICE_HTTP_PORT', '8003')}/mcp",
            },
            "search": {
                "transport": "streamable_http",
                "url": f"http://localhost:{os.getenv('SEARCH_HTTP_PORT', '8004')}/mcp",
            },
            "trade": {
                "transport": "streamable_http",
                "url": f"http://localhost:{os.getenv('TRADE_HTTP_PORT', '8002')}/mcp",
            },
        }

    async def initialize(self) -> None:
        """Initialize MCP client and AI model."""
        print(f"🚀 Initializing India stock agent: {self.signature}")

        if not self.openai_api_key:
            raise ValueError(
                "❌ OpenAI API key not set. Please configure OPENAI_API_KEY in environment or config file."
            )
        if not self.openai_base_url:
            print("⚠️  OpenAI base URL not set, using default")

        try:
            self.client = MultiServerMCPClient(self.mcp_config)
            self.tools = await self.client.get_tools()
            if not self.tools:
                print("⚠️  Warning: No MCP tools loaded. MCP services may not be running.")
                print(f"   MCP configuration: {self.mcp_config}")
            else:
                print(f"✅ Loaded {len(self.tools)} MCP tools")
        except Exception as e:
            raise RuntimeError(
                f"❌ Failed to initialize MCP client: {e}\n"
                f"   Please ensure MCP services are running at the configured ports.\n"
                f"   Run: python agent_tools/start_mcp_services.py"
            )

        try:
            if "deepseek" in self.basemodel.lower():
                self.model = DeepSeekChatOpenAI(
                    model=self.basemodel,
                    base_url=self.openai_base_url,
                    api_key=self.openai_api_key,
                    max_retries=3,
                    timeout=30,
                )
            else:
                self.model = ChatOpenAI(
                    model=self.basemodel,
                    base_url=self.openai_base_url,
                    api_key=self.openai_api_key,
                    max_retries=3,
                    timeout=30,
                )
        except Exception as e:
            raise RuntimeError(f"❌ Failed to initialize AI model: {e}")

        print(f"✅ India stock agent {self.signature} initialization completed")

    def _setup_logging(self, today_date: str) -> str:
        """Set up log file path."""
        log_path = os.path.join(self.base_log_path, self.signature, "log", today_date)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        return os.path.join(log_path, "log.jsonl")

    def _log_message(self, log_file: str, new_messages) -> None:
        """Log messages to log file."""
        log_entry = {"timestamp": datetime.now().isoformat(), "signature": self.signature, "new_messages": new_messages}
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    async def _ainvoke_with_retry(self, message: List[Dict[str, str]]) -> Any:
        """Agent invocation with retry."""
        for attempt in range(1, self.max_retries + 1):
            try:
                return await self.agent.ainvoke({"messages": message}, {"recursion_limit": 100})
            except Exception as e:
                if attempt == self.max_retries:
                    raise e
                print(f"⚠️ Attempt {attempt} failed, retrying after {self.base_delay * attempt} seconds...")
                print(f"Error details: {e}")
                await asyncio.sleep(self.base_delay * attempt)

    async def run_trading_session(self, today_date: str) -> None:
        """
        Run single day trading session for Indian market.

        Args:
            today_date: Trading date (YYYY-MM-DD)
        """
        print(f"📈 Starting India stock trading session: {today_date}")

        log_file = self._setup_logging(today_date)

        self.agent = create_agent(
            self.model,
            tools=self.tools,
            system_prompt=get_agent_system_prompt_india_stock(today_date, self.signature, self.stock_symbols),
        )

        user_query = [{"role": "user", "content": f"Please analyse and update today's ({today_date}) portfolio positions."}]
        message = user_query.copy()

        self._log_message(log_file, user_query)

        current_step = 0
        while current_step < self.max_steps:
            current_step += 1
            print(f"🔄 Step {current_step}/{self.max_steps}")

            try:
                response = await self._ainvoke_with_retry(message)
                agent_response = extract_conversation(response, "final")

                if STOP_SIGNAL in agent_response:
                    print("✅ Received stop signal, trading session ended")
                    print(agent_response)
                    self._log_message(log_file, [{"role": "assistant", "content": agent_response}])
                    break

                tool_msgs = extract_tool_messages(response)
                tool_response = "\n".join([msg.content for msg in tool_msgs])

                new_messages = [
                    {"role": "assistant", "content": agent_response},
                    {"role": "user", "content": f"Tool results: {tool_response}"},
                ]

                message.extend(new_messages)
                self._log_message(log_file, new_messages[0])
                self._log_message(log_file, new_messages[1])

            except Exception as e:
                print(f"❌ Trading session error: {str(e)}")
                raise

        await self._handle_trading_result(today_date)

    async def _handle_trading_result(self, today_date: str) -> None:
        """Handle trading results."""
        if_trade = get_config_value("IF_TRADE")
        if if_trade:
            write_config_value("IF_TRADE", False)
            print("✅ Trading completed")
        else:
            print("📊 No trading, maintaining positions")
            try:
                add_no_trade_record(today_date, self.signature)
            except NameError as e:
                print(f"❌ NameError: {e}")
                raise
            write_config_value("IF_TRADE", False)

    def register_agent(self) -> None:
        """Register new agent, create initial positions."""
        if os.path.exists(self.position_file):
            print(f"⚠️ Position file {self.position_file} already exists, skipping registration")
            return

        position_dir = os.path.join(self.data_path, "position")
        if not os.path.exists(position_dir):
            os.makedirs(position_dir)
            print(f"📁 Created position directory: {position_dir}")

        init_position = {symbol: 0 for symbol in self.stock_symbols}
        init_position["CASH"] = self.initial_cash

        init_date_str = self.init_date
        if " " in init_date_str:
            try:
                datetime.strptime(init_date_str, "%Y-%m-%d %H:%M:%S")
            except Exception:
                try:
                    date_part, time_part = init_date_str.split(" ", 1)
                    hh, mm, ss = time_part.split(":")
                    init_date_str = f"{date_part} {hh.zfill(2)}:{mm}:{ss}"
                except Exception:
                    pass

        with open(self.position_file, "w") as f:
            f.write(json.dumps({"date": init_date_str, "id": 0, "positions": init_position}) + "\n")

        print(f"✅ India stock agent {self.signature} registration completed")
        print(f"📁 Position file: {self.position_file}")
        print(f"💰 Initial cash: ₹{self.initial_cash:,.2f}")
        print(f"📊 Number of stocks: {len(self.stock_symbols)}")

    def get_trading_dates(self, init_date: str, end_date: str) -> List[str]:
        """
        Get trading date list for Indian market (NSE).

        Args:
            init_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            List of trading dates excluding weekends and market holidays
        """
        from tools.price_tools import is_trading_day

        max_date = None

        if not os.path.exists(self.position_file):
            self.register_agent()
            max_date = init_date
        else:
            with open(self.position_file, "r") as f:
                for line in f:
                    doc = json.loads(line)
                    current_date = doc["date"]
                    if max_date is None:
                        max_date = current_date
                    else:
                        current_date_obj = datetime.strptime(current_date, "%Y-%m-%d")
                        max_date_obj = datetime.strptime(max_date, "%Y-%m-%d")
                        if current_date_obj > max_date_obj:
                            max_date = current_date

        max_date_obj = datetime.strptime(max_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

        if end_date_obj <= max_date_obj:
            return []

        trading_dates = []
        current_date = max_date_obj + timedelta(days=1)

        while current_date <= end_date_obj:
            date_str = current_date.strftime("%Y-%m-%d")
            if is_trading_day(date_str, market="in"):
                trading_dates.append(date_str)
            current_date += timedelta(days=1)

        return trading_dates

    async def run_with_retry(self, today_date: str) -> None:
        """Run method with retry."""
        for attempt in range(1, self.max_retries + 1):
            try:
                print(f"🔄 Attempting to run {self.signature} - {today_date} (Attempt {attempt})")
                await self.run_trading_session(today_date)
                print(f"✅ {self.signature} - {today_date} run successful")
                return
            except Exception as e:
                print(f"❌ Attempt {attempt} failed: {str(e)}")
                if attempt == self.max_retries:
                    print(f"💥 {self.signature} - {today_date} all retries failed")
                    raise
                else:
                    wait_time = self.base_delay * attempt
                    print(f"⏳ Waiting {wait_time} seconds before retry...")
                    await asyncio.sleep(wait_time)

    async def run_date_range(self, init_date: str, end_date: str) -> None:
        """
        Run all trading days in date range.

        Args:
            init_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        print(f"📅 Running India stock date range: {init_date} to {end_date}")

        trading_dates = self.get_trading_dates(init_date, end_date)

        if not trading_dates:
            print(f"ℹ️ No trading days to process")
            return

        print(f"📊 Trading days to process: {trading_dates}")

        for date in trading_dates:
            print(f"🔄 Processing {self.signature} - Date: {date}")

            write_config_value("TODAY_DATE", date)
            write_config_value("SIGNATURE", self.signature)

            try:
                await self.run_with_retry(date)
            except Exception as e:
                print(f"❌ Error processing {self.signature} - Date: {date}")
                print(e)
                raise

        print(f"✅ {self.signature} processing completed")

    def get_position_summary(self) -> Dict[str, Any]:
        """Get position summary."""
        if not os.path.exists(self.position_file):
            return {"error": "Position file does not exist"}

        positions = []
        with open(self.position_file, "r") as f:
            for line in f:
                positions.append(json.loads(line))

        if not positions:
            return {"error": "No position records"}

        latest_position = positions[-1]
        return {
            "signature": self.signature,
            "latest_date": latest_position.get("date"),
            "positions": latest_position.get("positions", {}),
            "total_records": len(positions),
        }

    def __str__(self) -> str:
        return (
            f"BaseAgentIndiaStock(signature='{self.signature}', basemodel='{self.basemodel}', "
            f"market='in', stocks={len(self.stock_symbols)})"
        )

    def __repr__(self) -> str:
        return self.__str__()
