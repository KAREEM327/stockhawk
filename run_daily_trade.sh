#!/usr/bin/env bash
# Stock Hawk — daily live paper trading runner
# Runs at 9:25 AM ET weekdays via LaunchAgent
# Logs to: ~/Library/Logs/stockhawk/trade_YYYY-MM-DD.log

set -euo pipefail

TRADER_DIR="/Users/blackstarr/CLAUDE COWORK/alpaca-trader"
PYTHON="$TRADER_DIR/.venv/bin/python"
LOG_DIR="$HOME/Library/Logs/stockhawk"
DATE=$(date +%Y-%m-%d)
LOG_FILE="$LOG_DIR/trade_$DATE.log"

mkdir -p "$LOG_DIR"

# Only run on weekdays (Mon=1 .. Fri=5)
DOW=$(date +%u)
if [ "$DOW" -gt 5 ]; then
    echo "[$(date '+%H:%M:%S')] Weekend — skipping." >> "$LOG_FILE"
    exit 0
fi

echo "[$(date '+%H:%M:%S')] ── Stock Hawk daily trade ── $DATE" >> "$LOG_FILE"
echo "[$(date '+%H:%M:%S')] Python: $PYTHON" >> "$LOG_FILE"

cd "$TRADER_DIR"
PYTHONUNBUFFERED=1 "$PYTHON" -u main.py trade >> "$LOG_FILE" 2>&1
EXIT=$?

echo "[$(date '+%H:%M:%S')] Exit code: $EXIT" >> "$LOG_FILE"

# Keep last 30 logs, remove older ones
find "$LOG_DIR" -name "trade_*.log" -mtime +30 -delete 2>/dev/null || true

exit $EXIT
