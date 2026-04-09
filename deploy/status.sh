#!/bin/bash
# status.sh — quick health check on all running instances

echo "=== STSC Status ==="
echo ""

# GPU temps and utilization
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader | while IFS=, read idx name temp util mem_used mem_total; do
    printf "  GPU %s %-24s  %s°C  util:%s  mem:%s/%s\n" \
        "$idx" "$name" "$temp" "$util" "$mem_used" "$mem_total"
done

echo ""
echo "=== Frame counts ==="
for d in frames_gpu*/ frames0/ frames1/ frames2/ frames/; do
    [ -d "$d" ] || continue
    COUNT=$(ls "$d"*.png 2>/dev/null | wc -l)
    PID=$(cat "$d/pid" 2>/dev/null || echo "?")
    ALIVE=""
    [ "$PID" != "?" ] && kill -0 "$PID" 2>/dev/null && ALIVE=" [running]"
    printf "  %-30s %6d frames  PID=%s%s\n" "$d" "$COUNT" "$PID" "$ALIVE"
done

echo ""
echo "=== Recent log (run.log / run0.log) ==="
for f in frames_gpu0/run.log run0.log; do
    [ -f "$f" ] || continue
    grep -E "NCA loss|Active|Speed|grad norm" "$f" | tail -4
    break
done
