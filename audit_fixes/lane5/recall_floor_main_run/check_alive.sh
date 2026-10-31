#!/bin/bash
# Lane 5 -- liveness check for the 24-seed harness. Added after the run
# died silently for ~3.75h with no visible trace until someone checked
# `ps` by hand. Usage: ./check_alive.sh [pid]
#   pid (optional): the lane5_harness.py PID to check. If omitted, tries
#   to find it via pgrep.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HEARTBEAT_FILE="$HERE/logs/heartbeat.txt"

PID="$1"
if [ -z "$PID" ]; then
    PID=$(pgrep -f "[l]ane5_harness.py" | head -1)
fi

echo "=== Lane 5 liveness check ($(date '+%Y-%m-%d %H:%M:%S')) ==="

if [ -z "$PID" ]; then
    echo "PID: NOT FOUND (no 'python3 lane5_harness.py' process running)"
else
    if kill -0 "$PID" 2>/dev/null; then
        echo "PID $PID: ALIVE"
    else
        echo "PID $PID: NOT ALIVE (stale/incorrect PID)"
    fi
fi

if [ ! -f "$HEARTBEAT_FILE" ]; then
    echo "Heartbeat file: NOT FOUND ($HEARTBEAT_FILE) -- run has not written one yet"
    exit 0
fi

HB_CONTENT=$(cat "$HEARTBEAT_FILE")
HB_EPOCH=$(echo "$HB_CONTENT" | awk '{print $1}')
NOW_EPOCH=$(date +%s)
AGE=$(( NOW_EPOCH - ${HB_EPOCH%.*} ))

echo "Heartbeat: $HB_CONTENT"
echo "Heartbeat age: ${AGE}s"

if [ "$AGE" -gt 600 ]; then
    echo "WARNING: heartbeat is over 10 minutes old -- the run may be dead or hung."
elif [ "$AGE" -gt 120 ]; then
    echo "NOTE: heartbeat is over 2 minutes old -- normal only if mid-control-batch on a slow depth; check again shortly."
else
    echo "OK: heartbeat is recent."
fi
