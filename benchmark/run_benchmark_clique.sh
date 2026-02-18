#!/usr/bin/env bash
# =============================================================================
#  run_benchmark_clique.sh
#  Compares Serial (WP1) vs OpenMP (WP2) Maximum Clique implementations.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."

SERIAL_BIN="$ROOT/WP1_serial/max_clique_serial"
OPENMP_BIN="$ROOT/WP2_openmp/max_clique_openmp"
DATA_DIR="$ROOT/data/graphs"
RESULTS_FILE="$SCRIPT_DIR/results_clique.csv"

NUM_THREADS=$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)

echo "=== Building WP1 (serial) ==="
(cd "$ROOT/WP1_serial" && make -s max_clique_serial)

echo "=== Building WP2 (OpenMP, $NUM_THREADS threads) ==="
(cd "$ROOT/WP2_openmp" && make -s max_clique_openmp)

# Generate instances if needed
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
    echo "Generating graph instances..."
    python3 "$ROOT/data/generate_graphs.py"
fi

# Collect files
GRAPH_FILES=()
while IFS= read -r f; do
    GRAPH_FILES+=("$f")
done < <(find "$DATA_DIR" -name "*.dimacs" | sort)

if [ ${#GRAPH_FILES[@]} -eq 0 ]; then
    echo "No .dimacs files found in $DATA_DIR"
    exit 1
fi

echo "instance,n,m,serial_clique,serial_time_s,openmp_clique,openmp_time_s,serial_nodes,openmp_nodes,speedup,threads" \
    > "$RESULTS_FILE"

echo ""
echo "================================================================================="
printf "%-28s %5s %7s  %6s %8s  %6s %8s  %7s\n" \
    "Instance" "n" "m" "Clique" "Serial(s)" "Clique" "OMP(s)" "Speedup"
echo "================================================================================="

for GRAPH_FILE in "${GRAPH_FILES[@]}"; do
    INSTANCE=$(basename "$GRAPH_FILE" .dimacs)

    # Serial
    SERIAL_OUT=$("$SERIAL_BIN" "$GRAPH_FILE" 2>&1)
    SERIAL_CSV=$(echo "$SERIAL_OUT" | grep "^CSV," | head -1)
    S_N=$(     echo "$SERIAL_CSV" | cut -d',' -f2)
    S_M=$(     echo "$SERIAL_CSV" | cut -d',' -f3)
    S_CLIQUE=$(echo "$SERIAL_CSV" | cut -d',' -f4)
    S_TIME=$(  echo "$SERIAL_CSV" | cut -d',' -f5)
    S_NODES=$( echo "$SERIAL_CSV" | cut -d',' -f6)

    # OpenMP
    OPENMP_OUT=$("$OPENMP_BIN" "$GRAPH_FILE" "$NUM_THREADS" 2>&1)
    OPENMP_CSV=$(echo "$OPENMP_OUT" | grep "^CSV," | head -1)
    O_CLIQUE=$(echo "$OPENMP_CSV" | cut -d',' -f5)
    O_TIME=$(  echo "$OPENMP_CSV" | cut -d',' -f6)
    O_NODES=$( echo "$OPENMP_CSV" | cut -d',' -f7)

    # Verify both find the same clique size
    MATCH="OK"
    [ "$S_CLIQUE" != "$O_CLIQUE" ] && MATCH="MISMATCH!"

    SPEEDUP=$(python3 -c "print(f'{float($S_TIME)/float($O_TIME):.2f}')" 2>/dev/null || echo "N/A")

    printf "%-28s %5s %7s  %6s %8s  %6s %8s  %7s %s\n" \
        "$INSTANCE" "$S_N" "$S_M" "$S_CLIQUE" "$S_TIME" \
        "$O_CLIQUE" "$O_TIME" "${SPEEDUP}x" "$MATCH"

    echo "$INSTANCE,$S_N,$S_M,$S_CLIQUE,$S_TIME,$O_CLIQUE,$O_TIME,$S_NODES,$O_NODES,$SPEEDUP,$NUM_THREADS" \
        >> "$RESULTS_FILE"
done

echo "================================================================================="
echo ""
echo "Results saved to: $RESULTS_FILE"
