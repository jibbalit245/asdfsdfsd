#!/bin/bash
# monitor.sh — live multi-instance dashboard for all running coupled_field jobs
#
# Reads status.json written by each instance into its snap_dir, then draws
# a compact per-instance panel refreshed every second.
#
# Usage:
#   ./deploy/monitor.sh [refresh_seconds]
#
# Each coupled_field instance writes <snap_dir>/status.json atomically.
# Single-GPU runs that use the current directory write ./status.json.

REFRESH=${1:-1}

# ANSI helpers
BOLD='\033[1m'
DIM='\033[2m'
CYAN='\033[36m'
GREEN='\033[32m'
YELLOW='\033[33m'
RED='\033[31m'
RESET='\033[0m'
CLEAR='\033[2J\033[H'

# Require jq or python3 for JSON parsing; prefer jq, fall back to python3
if command -v jq &>/dev/null; then
    get() { jq -r ".$1 // \"\"" "$2" 2>/dev/null; }
elif command -v python3 &>/dev/null; then
    get() {
        python3 -c "
import sys, json
try:
    d = json.load(open('$2'))
    v = d.get('$1', '')
    print('' if v is None else v)
except: print('')
" 2>/dev/null
    }
else
    echo "monitor.sh requires jq or python3" >&2
    exit 1
fi

fmt_num() {
    # Compact number formatter: 1234567 → 1.23M
    python3 -c "
v=float('${1:-0}')
if v>=1e9: print('%.2fB'%(v/1e9))
elif v>=1e6: print('%.2fM'%(v/1e6))
elif v>=1e3: print('%.2fK'%(v/1e3))
else: print('%.4g'%v)
" 2>/dev/null
}

bar() {
    # bar <val> <max> <width>
    local val=$1 max=$2 width=$3
    local filled=0
    if [ "$max" -gt 0 ] 2>/dev/null; then
        filled=$(python3 -c "print(min(int(float('$val')/float('$max')*$width), $width))" 2>/dev/null)
    fi
    local empty=$(( width - filled ))
    printf '%s' "${GREEN}"
    printf '%0.s█' $(seq 1 $filled 2>/dev/null) 2>/dev/null
    printf '%s' "${DIM}"
    printf '%0.s·' $(seq 1 $empty 2>/dev/null) 2>/dev/null
    printf '%s' "${RESET}"
}

draw_panel() {
    local f="$1"       # path to status.json
    local label="$2"   # instance label (e.g. "GPU 0 — frames_gpu0_random")
    local pid_file="${f%status.json}pid"

    local tick       ; tick=$(get tick "$f")
    local goal       ; goal=$(get ticks_goal "$f")
    local active     ; active=$(get active_cells "$f")
    local wave_e     ; wave_e=$(get wave_energy "$f")
    local delta_e    ; delta_e=$(get delta_energy "$f")
    local edges      ; edges=$(get edge_count "$f")
    local natural    ; natural=$(get natural_count "$f")
    local snaps      ; snaps=$(get snapshots "$f")
    local nca_loss   ; nca_loss=$(get nca_loss "$f")
    local nca_res    ; nca_res=$(get nca_residual "$f")
    local grad_norm  ; grad_norm=$(get grad_norm "$f")
    local tick_ms    ; tick_ms=$(get tick_ms "$f")
    local alerts     ; alerts=$(get alerts "$f")
    local paused     ; paused=$(get paused "$f")
    local device     ; device=$(get device "$f")

    # Staleness check — warn if file is >30s old
    local stale=""
    if [ -f "$f" ]; then
        local age
        age=$(python3 -c "import os,time; print(int(time.time()-os.path.getmtime('$f')))" 2>/dev/null)
        [ "${age:-0}" -gt 30 ] 2>/dev/null && stale=" ${YELLOW}(stale ${age}s)${RESET}"
    fi

    # PID alive check
    local alive=""
    if [ -f "$pid_file" ]; then
        local pid; pid=$(cat "$pid_file")
        kill -0 "$pid" 2>/dev/null && alive="${GREEN}[running PID $pid]${RESET}" || alive="${RED}[dead PID $pid]${RESET}"
    fi

    local pct=0
    [ "${goal:-0}" -gt 0 ] 2>/dev/null && pct=$(python3 -c "print('%.1f'%($tick/$goal*100))" 2>/dev/null)

    local cells_total=2073600  # 1920*1080
    local act_pct=0
    [ "${active:-0}" -gt 0 ] 2>/dev/null && act_pct=$(python3 -c "print('%.1f'%($active/$cells_total*100))" 2>/dev/null)

    echo -e "  ${BOLD}${CYAN}── $label${RESET}  $alive$stale"
    echo -e "  ${DIM}$device${RESET}"

    # Progress bar
    printf "  ${BOLD}Tick${RESET}  %s / %s  (%s%%)\n  " "$tick" "$goal" "$pct"
    bar "${tick:-0}" "${goal:-1}" 40
    echo

    # Active cells
    local active_fmt; active_fmt=$(fmt_num "${active:-0}")
    printf "  ${BOLD}Active${RESET}  %s (%.1f%%)  ${BOLD}Edges${RESET}  %s  ${BOLD}Nat.${RESET}  %s  ${BOLD}Snaps${RESET}  %s\n" \
        "$active_fmt" "$act_pct" "${edges:-0}" "${natural:-0}" "${snaps:-0}"

    # Energies
    local we_fmt; we_fmt=$(fmt_num "${wave_e:-0}")
    local de_fmt; de_fmt=$(fmt_num "${delta_e:-0}")
    printf "  ${BOLD}Wave energy${RESET}  %-10s  ${BOLD}Delta energy${RESET}  %s\n" "$we_fmt" "$de_fmt"

    # NCA loss
    local loss_fmt; loss_fmt=$(fmt_num "${nca_loss:-0}")
    local res_fmt;  res_fmt=$(fmt_num "${nca_res:-0}")
    local grad_col="$DIM"
    [ "$(python3 -c "print(1 if float('${grad_norm:-0}')>1 else 0)" 2>/dev/null)" = "1" ] && grad_col="$YELLOW"
    printf "  ${BOLD}NCA loss${RESET}  %-10s  ${BOLD}Residual${RESET}  %-10s  ${grad_col}grad %.3f${RESET}\n" \
        "$loss_fmt" "$res_fmt" "${grad_norm:-0}"

    # Speed
    if [ "${tick_ms:-0}" != "0" ] && [ -n "${tick_ms:-}" ]; then
        local tps; tps=$(python3 -c "print('%.0f'%(1000.0/float('$tick_ms')))" 2>/dev/null)
        printf "  ${BOLD}Speed${RESET}  %.1f ms/tick  (~%s ticks/sec)\n" "$tick_ms" "$tps"
    fi

    # Alerts
    local alert_int=${alerts:-0}
    [ "$alert_int" -eq 0 ] 2>/dev/null && echo -e "  ${GREEN}✓  field stable${RESET}" || {
        [ $(( alert_int & 1 )) -ne 0 ] 2>/dev/null && echo -e "  ${YELLOW}⚠  delta collapse — field quiet${RESET}"
        [ $(( alert_int & 2 )) -ne 0 ] 2>/dev/null && echo -e "  ${RED}⚠  wave divergence >1e10${RESET}"
    }
    [ "${paused:-0}" != "0" ] && echo -e "  ${YELLOW}PAUSED${RESET}"

    echo ""
}

# ── main loop ──────────────────────────────────────────────────────────────────
cd "$(dirname "$0")/.."

while true; do
    printf '%b' "$CLEAR"

    echo -e "${BOLD}${CYAN}  ╔════════════════════════════════════════════════╗"
    echo -e "  ║     COUPLED FIELD MULTI-INSTANCE MONITOR      ║"
    echo -e "  ╚════════════════════════════════════════════════╝${RESET}"
    echo -e "  ${DIM}$(date '+%Y-%m-%d %H:%M:%S')   refresh every ${REFRESH}s   Ctrl+C to exit${RESET}\n"

    # Best NCA loss from shared model
    if [ -f best_nca.bin ]; then
        local_best=$(python3 -c "
import struct, sys
try:
    with open('best_nca.bin','rb') as f:
        magic,n,t = struct.unpack('<III', f.read(12))
    print('adam_step=%d' % t)
except: print('?')
" 2>/dev/null)
        echo -e "  ${BOLD}best_nca.bin${RESET}  ${GREEN}present${RESET}  ($local_best)"
    else
        echo -e "  ${DIM}best_nca.bin  not yet created${RESET}"
    fi
    echo ""

    # GPU hardware summary (if nvidia-smi available)
    if command -v nvidia-smi &>/dev/null; then
        echo -e "  ${BOLD}GPU Hardware${RESET}"
        nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total \
            --format=csv,noheader 2>/dev/null | while IFS=, read idx name temp util mem_used mem_total; do
            printf "  GPU %-2s  %-28s  %s°C  util:%-5s  mem:%s/%s\n" \
                "$idx" "$name" "$temp" "$util" "$mem_used" "$mem_total"
        done
        echo ""
    fi

    # Find all status.json files (snap_dir instances + current dir fallback)
    found=0
    for f in frames_gpu*/status.json frames_*/status.json status.json; do
        [ -f "$f" ] || continue
        dir="${f%/status.json}"
        [ "$dir" = "status.json" ] && dir="." && label="default" || label="$dir"
        draw_panel "$f" "$label"
        found=$(( found + 1 ))
    done

    if [ "$found" -eq 0 ]; then
        echo -e "  ${DIM}No status.json files found yet."
        echo -e "  Waiting for instances to start...${RESET}"
        echo ""
        echo -e "  ${DIM}Expected locations:${RESET}"
        echo -e "    frames_gpu0_random/status.json"
        echo -e "    frames_gpu1_quantum/status.json  etc."
    fi

    sleep "$REFRESH"
done
