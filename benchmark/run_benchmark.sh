#!/usr/bin/env bash
# =============================================================================
#  run_benchmark.sh
#  Compares Serial (WP1) vs OpenMP (WP2) TSP implementations.
#
#  Usage:
#    chmod +x run_benchmark.sh
#    ./run_benchmark.sh                      # benchmark all instances in ../data/
#    ./run_benchmark.sh ../data/berlin52.tsp  # benchmark a specific file
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."

SERIAL_BIN="$ROOT/WP1_serial/tsp_serial"
OPENMP_BIN="$ROOT/WP2_openmp/tsp_openmp"
DATA_DIR="$ROOT/data"
RESULTS_FILE="$SCRIPT_DIR/results.csv"

INIT_TEMP=1000.0
COOLING_RATE=0        # 0 = auto (recommended: each program computes optimal rate for its n)
NUM_THREADS=$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)

# Iteration budget per instance (scales with problem size, capped for tractability)
# Adjust MAX_ITER_CAP to trade off runtime vs solution quality
MAX_ITER_CAP=10000000   # 10M iterations per run maximum

# ── Build both binaries ───────────────────────────────────────────────────────
echo "=== Building WP1 (serial) ==="
(cd "$ROOT/WP1_serial" && make -s)

echo "=== Building WP2 (OpenMP, $NUM_THREADS threads) ==="
(cd "$ROOT/WP2_openmp" && make -s)

# ── Collect TSP files ─────────────────────────────────────────────────────────
if [ $# -ge 1 ]; then
    TSP_FILES=("$@")
else
    # Compatible with both bash and zsh on macOS
    TSP_FILES=()
    while IFS= read -r f; do
        TSP_FILES+=("$f")
    done < <(find "$DATA_DIR" -name "*.tsp" | sort)
fi

if [ ${#TSP_FILES[@]} -eq 0 ]; then
    echo "No .tsp files found. Run: python3 $DATA_DIR/generate_instances.py"
    exit 1
fi

# ── CSV header ────────────────────────────────────────────────────────────────
echo "instance,n,serial_cost,serial_time_s,openmp_cost,openmp_time_s,speedup,threads" \
    > "$RESULTS_FILE"

echo ""
echo "=================================================================="
printf "%-22s %6s  %12s %8s  %12s %8s  %7s\n" \
    "Instance" "n" "Serial cost" "Time(s)" "OpenMP cost" "Time(s)" "Speedup"
echo "=================================================================="

for TSP_FILE in "${TSP_FILES[@]}"; do
    INSTANCE=$(basename "$TSP_FILE" .tsp)

    # Determine iteration count: n*n*10, capped at MAX_ITER_CAP
    N_CITIES=$(grep "^DIMENSION" "$TSP_FILE" | awk '{print $NF}')
    MAX_ITER=$(python3 -c "print(min(int($N_CITIES)**2*10, $MAX_ITER_CAP))")

    # Compute cooling rate: T decays from INIT_TEMP to ~1e-9 over MAX_ITER steps
    COOL_SERIAL=$(python3 -c "import math; print(f'{math.exp(math.log(1e-9/$INIT_TEMP)/$MAX_ITER):.10f}')")
    ITERS_PER_THREAD=$(python3 -c "print(max(1, $MAX_ITER // $NUM_THREADS))")
    COOL_OPENMP=$(python3 -c "import math; print(f'{math.exp(math.log(1e-9/$INIT_TEMP)/$ITERS_PER_THREAD):.10f}')")

    # -- Serial run --
    SERIAL_OUT=$("$SERIAL_BIN" "$TSP_FILE" "$MAX_ITER" "$INIT_TEMP" "$COOL_SERIAL" 2>&1)
    SERIAL_CSV=$(echo "$SERIAL_OUT" | grep "^CSV," | head -1)
    SERIAL_N=$(    echo "$SERIAL_CSV" | cut -d',' -f2)
    SERIAL_COST=$( echo "$SERIAL_CSV" | cut -d',' -f3)
    SERIAL_TIME=$( echo "$SERIAL_CSV" | cut -d',' -f4)

    # -- OpenMP run (same total iterations, P-thread cooling schedule) --
    OPENMP_OUT=$("$OPENMP_BIN" "$TSP_FILE" "$MAX_ITER" "$INIT_TEMP" "$COOL_OPENMP" "$NUM_THREADS" 2>&1)
    OPENMP_CSV=$(echo "$OPENMP_OUT" | grep "^CSV," | head -1)
    OPENMP_COST=$(  echo "$OPENMP_CSV" | cut -d',' -f4)
    OPENMP_TIME=$(  echo "$OPENMP_CSV" | cut -d',' -f5)

    # -- Speedup --
    SPEEDUP=$(python3 -c "print(f'{float($SERIAL_TIME)/float($OPENMP_TIME):.2f}')" 2>/dev/null || echo "N/A")

    printf "%-22s %6s  %12s %8s  %12s %8s  %7s\n" \
        "$INSTANCE" "$SERIAL_N" "$SERIAL_COST" "$SERIAL_TIME" \
        "$OPENMP_COST" "$OPENMP_TIME" "${SPEEDUP}x"

    echo "$INSTANCE,$SERIAL_N,$SERIAL_COST,$SERIAL_TIME,$OPENMP_COST,$OPENMP_TIME,$SPEEDUP,$NUM_THREADS" \
        >> "$RESULTS_FILE"
done

echo "=================================================================="
echo ""
echo "Results saved to: $RESULTS_FILE"
