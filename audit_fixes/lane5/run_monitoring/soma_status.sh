#!/bin/bash

PROJECT_DIR="<REPO_ROOT>"

echo "=== SOMA status ($(date '+%Y-%m-%d %H:%M:%S')) ==="

show_process() {
    label="$1"
    pattern="$2"
    pids=$(pgrep -fl "$pattern" | awk '$0 !~ /caffeinate/ {print $1}' || true)

    if [ -n "$pids" ]; then
        printf '%s: RUNNING (PID%s %s)\n' "$label" "$(printf '%s\n' "$pids" | wc -l | tr -d ' ')" "$(printf '%s\n' "$pids" | tr '\n' ' ')"
    else
        printf '%s: NOT RUNNING\n' "$label"
    fi
}

show_process "Lane 5 harness" "[l]ane5_harness.py"
show_process "Lane 5 chain runner" "[r]un_chain.py"
show_process "ResNet final sweep" "[d]egradation_aware_resnet_final_sweep.py"

echo
if [ -f "$PROJECT_DIR/audit_fixes/lane5/recall_floor_main_run/check_alive.sh" ]; then
    bash "$PROJECT_DIR/audit_fixes/lane5/recall_floor_main_run/check_alive.sh"
else
    echo "Lane 5 heartbeat checker not found under $PROJECT_DIR"
fi