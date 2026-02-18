#!/usr/bin/env bash
# =============================================================================
#  run_benchmark_knapsack.sh
#  Compares Serial (WP1) vs OpenMP (WP2) 0/1 Knapsack implementations.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."

SERIAL_BIN="$ROOT/WP1_serial/knapsack_serial"
OPENMP_BIN="$ROOT/WP2_openmp/knapsack_openmp"
DATA_DIR="$ROOT/data/knapsack"
RESULTS_FILE="$SCRIPT_DIR/results_knapsack.csv"

NUM_THREADS=$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)

echo "=== Building WP1 (serial) ==="
(cd "$ROOT/WP1_serial" && make -s knapsack_serial)

echo "=== Building WP2 (OpenMP, $NUM_THREADS threads) ==="
(cd "$ROOT/WP2_openmp" && make -s knapsack_openmp)

# Generate instances if needed
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
    echo "Generating knapsack instances..."
    python3 "$ROOT/data/generate_knapsack.py"
fi

KS_FILES=()
while IFS= read -r f; do
    KS_FILES+=("$f")
done < <(find "$DATA_DIR" -name "*.txt" | sort)

if [ ${#KS_FILES[@]} -eq 0 ]; then
    echo "No .txt files found in $DATA_DIR"
    exit 1
fi

echo "instance,n,W,serial_value,serial_time_s,openmp_value,openmp_time_s,speedup,threads" \
    > "$RESULTS_FILE"

echo ""
echo "================================================================================="
printf "%-24s %6s %8s  %12s %8s  %12s %8s  %7s\n" \
    "Instance" "n" "W" "Opt.Value" "Serial(s)" "Opt.Value" "OMP(s)" "Speedup"
echo "================================================================================="

for KS_FILE in "${KS_FILES[@]}"; do
    INSTANCE=$(basename "$KS_FILE" .txt)

    # Serial
    SERIAL_OUT=$("$SERIAL_BIN" "$KS_FILE" 2>&1)
    SERIAL_CSV=$(echo "$SERIAL_OUT" | grep "^CSV," | head -1)
    S_N=$(    echo "$SERIAL_CSV" | cut -d',' -f2)
    S_W=$(    echo "$SERIAL_CSV" | cut -d',' -f3)
    S_VAL=$(  echo "$SERIAL_CSV" | cut -d',' -f4)
    S_TIME=$( echo "$SERIAL_CSV" | cut -d',' -f5)

    # OpenMP
    OPENMP_OUT=$("$OPENMP_BIN" "$KS_FILE" "$NUM_THREADS" 2>&1)
    OPENMP_CSV=$(echo "$OPENMP_OUT" | grep "^CSV," | head -1)
    O_VAL=$( echo "$OPENMP_CSV" | cut -d',' -f5)
    O_TIME=$(echo "$OPENMP_CSV" | cut -d',' -f6)

    # Both are exact algorithms → values must match
    MATCH="OK"
    [ "$S_VAL" != "$O_VAL" ] && MATCH="MISMATCH!"

    SPEEDUP=$(python3 -c "print(f'{float($S_TIME)/float($O_TIME):.2f}')" 2>/dev/null || echo "N/A")

    printf "%-24s %6s %8s  %12s %8s  %12s %8s  %7s %s\n" \
        "$INSTANCE" "$S_N" "$S_W" "$S_VAL" "$S_TIME" \
        "$O_VAL" "$O_TIME" "${SPEEDUP}x" "$MATCH"

    echo "$INSTANCE,$S_N,$S_W,$S_VAL,$S_TIME,$O_VAL,$O_TIME,$SPEEDUP,$NUM_THREADS" \
        >> "$RESULTS_FILE"
done

echo "================================================================================="
echo ""
echo "Results saved to: $RESULTS_FILE"
