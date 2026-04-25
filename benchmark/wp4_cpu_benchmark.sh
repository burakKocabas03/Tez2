#!/usr/bin/env bash
# =============================================================================
#  WP4 — Systematic CPU Benchmark
#  Thread scaling (1,2,4,8) × 5 repeats × all problem sizes
#  Produces CSV files consumed by wp4_analyze.py
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."
RESULTS_DIR="$SCRIPT_DIR/wp4_results"
mkdir -p "$RESULTS_DIR"

REPEATS=5
THREAD_COUNTS=(1 2 4 8)

INIT_TEMP=1000.0
MAX_ITER_CAP=10000000

MAX_THREADS=$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 8)

# ── Build ─────────────────────────────────────────────────────────────────────
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║           WP4 — Systematic CPU Benchmark                    ║"
echo "║  Repeats: $REPEATS   Thread counts: ${THREAD_COUNTS[*]}              ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

echo "=== Building WP1 (serial) ==="
(cd "$ROOT/WP1_serial" && make -s)

echo "=== Building WP2 (OpenMP) ==="
(cd "$ROOT/WP2_openmp" && make -s)

# ═══════════════════════════════════════════════════════════════════════════════
#  TSP BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════
TSP_CSV="$RESULTS_DIR/tsp_results.csv"
echo "instance,n,impl,threads,run,cost,time_s" > "$TSP_CSV"

TSP_FILES=(
    "$ROOT/data/berlin52.tsp"
    "$ROOT/data/random100.tsp"
    "$ROOT/data/random500.tsp"
    "$ROOT/data/random1000.tsp"
)

echo ""
echo "████████████████████████████████████████████████████████████████"
echo "  TSP BENCHMARK"
echo "████████████████████████████████████████████████████████████████"

for TSP_FILE in "${TSP_FILES[@]}"; do
    [ ! -f "$TSP_FILE" ] && continue
    INST=$(basename "$TSP_FILE" .tsp)
    N=$(grep "^DIMENSION" "$TSP_FILE" | awk '{print $NF}')
    MAX_ITER=$(python3 -c "print(min(int($N)**2*10, $MAX_ITER_CAP))")
    COOL_SERIAL=$(python3 -c "import math; print(f'{math.exp(math.log(1e-9/$INIT_TEMP)/$MAX_ITER):.12f}')")

    echo ""
    echo "  ── $INST (n=$N, iters=$MAX_ITER) ──"

    # Serial (5 repeats)
    printf "    Serial:  "
    for r in $(seq 1 $REPEATS); do
        OUT=$("$ROOT/WP1_serial/tsp_serial" "$TSP_FILE" "$MAX_ITER" "$INIT_TEMP" "$COOL_SERIAL" "$r" 2>&1)
        CSV_LINE=$(echo "$OUT" | grep "^CSV," | head -1)
        COST=$(echo "$CSV_LINE" | cut -d',' -f3)
        TIME=$(echo "$CSV_LINE" | cut -d',' -f4)
        echo "$INST,$N,serial,1,$r,$COST,$TIME" >> "$TSP_CSV"
        printf "%.4f " "$TIME"
    done
    echo ""

    # OpenMP with different thread counts
    for T in "${THREAD_COUNTS[@]}"; do
        [ "$T" -gt "$MAX_THREADS" ] && continue
        ITERS_PER_THREAD=$(python3 -c "print(max(1, $MAX_ITER // $T))")
        COOL_OMP=$(python3 -c "import math; print(f'{math.exp(math.log(1e-9/$INIT_TEMP)/$ITERS_PER_THREAD):.12f}')")
        printf "    OMP-%dT:  " "$T"
        for r in $(seq 1 $REPEATS); do
            OUT=$("$ROOT/WP2_openmp/tsp_openmp" "$TSP_FILE" "$MAX_ITER" "$INIT_TEMP" "$COOL_OMP" "$T" 2>&1)
            CSV_LINE=$(echo "$OUT" | grep "^CSV," | head -1)
            COST=$(echo "$CSV_LINE" | cut -d',' -f4)
            TIME=$(echo "$CSV_LINE" | cut -d',' -f5)
            echo "$INST,$N,openmp,$T,$r,$COST,$TIME" >> "$TSP_CSV"
            printf "%.4f " "$TIME"
        done
        echo ""
    done
done

# ═══════════════════════════════════════════════════════════════════════════════
#  MAX CLIQUE BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════
MCP_CSV="$RESULTS_DIR/clique_results.csv"
echo "instance,n,density,impl,threads,run,clique_size,time_s" > "$MCP_CSV"

MCP_FILES=(
    "$ROOT/data/graphs/rand_n50_d50.dimacs"
    "$ROOT/data/graphs/rand_n50_d90.dimacs"
    "$ROOT/data/graphs/rand_n100_d50.dimacs"
    "$ROOT/data/graphs/rand_n100_d80.dimacs"
    "$ROOT/data/graphs/rand_n150_d50.dimacs"
    "$ROOT/data/graphs/rand_n200_d50.dimacs"
)

echo ""
echo "████████████████████████████████████████████████████████████████"
echo "  MAX CLIQUE BENCHMARK"
echo "████████████████████████████████████████████████████████████████"

for GRAPH_FILE in "${MCP_FILES[@]}"; do
    [ ! -f "$GRAPH_FILE" ] && continue
    INST=$(basename "$GRAPH_FILE" .dimacs)
    N_VERT=$(grep "^p " "$GRAPH_FILE" | awk '{print $3}')
    N_EDGE=$(grep "^p " "$GRAPH_FILE" | awk '{print $4}')
    DENS=$(python3 -c "n=$N_VERT; print(f'{2*$N_EDGE/(n*(n-1)):.2f}' if n>1 else '0')")

    echo ""
    echo "  ── $INST (n=$N_VERT, density=$DENS) ──"

    # Serial
    printf "    Serial:  "
    for r in $(seq 1 $REPEATS); do
        OUT=$("$ROOT/WP1_serial/max_clique_serial" "$GRAPH_FILE" 2>&1)
        CSV_LINE=$(echo "$OUT" | grep "^CSV," | head -1)
        SZ=$(echo "$CSV_LINE" | cut -d',' -f4)
        TIME=$(echo "$CSV_LINE" | cut -d',' -f5)
        echo "$INST,$N_VERT,$DENS,serial,1,$r,$SZ,$TIME" >> "$MCP_CSV"
        printf "%.6f " "$TIME"
    done
    echo ""

    # OpenMP
    for T in "${THREAD_COUNTS[@]}"; do
        [ "$T" -gt "$MAX_THREADS" ] && continue
        printf "    OMP-%dT:  " "$T"
        for r in $(seq 1 $REPEATS); do
            OUT=$("$ROOT/WP2_openmp/max_clique_openmp" "$GRAPH_FILE" "$T" 2>&1)
            CSV_LINE=$(echo "$OUT" | grep "^CSV," | head -1)
            SZ=$(echo "$CSV_LINE" | cut -d',' -f5)
            TIME=$(echo "$CSV_LINE" | cut -d',' -f6)
            echo "$INST,$N_VERT,$DENS,openmp,$T,$r,$SZ,$TIME" >> "$MCP_CSV"
            printf "%.6f " "$TIME"
        done
        echo ""
    done
done

# ═══════════════════════════════════════════════════════════════════════════════
#  KNAPSACK BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════
KS_CSV="$RESULTS_DIR/knapsack_results.csv"
echo "instance,n,W,dp_cells,impl,threads,run,opt_value,time_s" > "$KS_CSV"

KS_FILES=(
    "$ROOT/data/knapsack/ks_n500_W50000.txt"
    "$ROOT/data/knapsack/ks_n1000_W50000.txt"
    "$ROOT/data/knapsack/ks_n1000_W100000.txt"
    "$ROOT/data/knapsack/ks_n2000_W50000.txt"
    "$ROOT/data/knapsack/ks_n2000_W100000.txt"
    "$ROOT/data/knapsack/ks_n5000_W50000.txt"
)

echo ""
echo "████████████████████████████████████████████████████████████████"
echo "  KNAPSACK BENCHMARK"
echo "████████████████████████████████████████████████████████████████"

for KS_FILE in "${KS_FILES[@]}"; do
    [ ! -f "$KS_FILE" ] && continue
    INST=$(basename "$KS_FILE" .txt)
    N_ITEMS=$(head -1 "$KS_FILE" | awk '{print $1}')
    W_CAP=$(head -1 "$KS_FILE" | awk '{print $2}')
    DP_CELLS=$(python3 -c "print($N_ITEMS * ($W_CAP + 1))")

    echo ""
    echo "  ── $INST (n=$N_ITEMS, W=$W_CAP, cells=$DP_CELLS) ──"

    # Serial
    printf "    Serial:  "
    for r in $(seq 1 $REPEATS); do
        OUT=$("$ROOT/WP1_serial/knapsack_serial" "$KS_FILE" 2>&1)
        CSV_LINE=$(echo "$OUT" | grep "^CSV," | head -1)
        VAL=$(echo "$CSV_LINE" | cut -d',' -f4)
        TIME=$(echo "$CSV_LINE" | cut -d',' -f5)
        echo "$INST,$N_ITEMS,$W_CAP,$DP_CELLS,serial,1,$r,$VAL,$TIME" >> "$KS_CSV"
        printf "%.4f " "$TIME"
    done
    echo ""

    # OpenMP
    for T in "${THREAD_COUNTS[@]}"; do
        [ "$T" -gt "$MAX_THREADS" ] && continue
        printf "    OMP-%dT:  " "$T"
        for r in $(seq 1 $REPEATS); do
            OUT=$("$ROOT/WP2_openmp/knapsack_openmp" "$KS_FILE" "$T" 2>&1)
            CSV_LINE=$(echo "$OUT" | grep "^CSV," | head -1)
            VAL=$(echo "$CSV_LINE" | cut -d',' -f5)
            TIME=$(echo "$CSV_LINE" | cut -d',' -f6)
            echo "$INST,$N_ITEMS,$W_CAP,$DP_CELLS,openmp,$T,$r,$VAL,$TIME" >> "$KS_CSV"
            printf "%.4f " "$TIME"
        done
        echo ""
    done
done

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  BENCHMARK COMPLETE                                         ║"
echo "║  Results:                                                   ║"
echo "║    $TSP_CSV"
echo "║    $MCP_CSV"
echo "║    $KS_CSV"
echo "╚═══════════════════════════════════════════════════════════════╝"
