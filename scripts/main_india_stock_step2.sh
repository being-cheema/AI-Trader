#!/bin/bash

# Start MCP services for Indian stock market trading

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

echo "🔧 Starting MCP services..."
cd agent_tools
python start_mcp_services.py
cd ..
