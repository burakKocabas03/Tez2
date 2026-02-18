#!/usr/bin/env bash
# =============================================================================
#  run_benchmark_all.sh
#  Compares Serial (WP1) vs OpenMP (WP2) vs CUDA (WP3) for all 3 problems.
#
#  Usage:
#    ./run_benchmark_all.sh            # full benchmark
#    ./run_benchmark_all.sh --no-cuda  # skip CUDA (no GPU available)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."
RESULTS="$SCRIPT_DIR/results_all.csv"

HAS_CUDA=true
[ "${1:-}" = "--no-cuda" ] && HAS_CUDA=false

NUM_THREADS=$(sysctl -n hw.logicalcpu 2>/dev/null || nproc 2>/dev/null || echo 4)

# ── Build ─────────────────────────────────────────────────────────────────────
echo "=== Building WP1 (serial) ==="
(cd "$ROOT/WP1_serial" && make -s)

echo "=== Building WP2 (OpenMP, $NUM_THREADS threads) ==="
(cd "$ROOT/WP2_openmp" && make -s)

if $HAS_CUDA; then
    echo "=== Building WP3 (CUDA) ==="
    (cd "$ROOT/WP3_cuda" && make -s)
fi

# ── Generate data if needed ───────────────────────────────────────────────────
[ ! -f "$ROOT/data/random100.tsp" ]                  && python3 "$ROOT/data/generate_instances.py"
[ ! -d "$ROOT/data/graphs" ]                         && python3 "$ROOT/data/generate_graphs.py"
[ ! -d "$ROOT/data/knapsack" ]                       && python3 "$ROOT/data/generate_knapsack.py"

echo "problem,instance,n,serial_result,serial_time,openmp_result,openmp_time,cuda_result,cuda_time,speedup_omp,speedup_cuda,threads" > "$RESULTS"

# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "████████████████████████████████████████████████████████████████"
echo "  PROBLEM 1: TRAVELING SALESMAN PROBLEM (TSP)"
echo "████████████████████████████████████████████████████████████████"

for f in "$ROOT/data/berlin52.tsp" "$ROOT/data/random100.tsp" "$ROOT/data/random500.tsp" "$ROOT/data/random1000.tsp"; do
    [ ! -f "$f" ] && continue
    INST=$(basename "$f" .tsp)
    N=$(grep "^DIMENSION" "$f" | awk '{print $NF}')
    ITER=$(python3 -c "print(min(int($N)**2*10, 10000000))")
    TEMP=1000.0

    # Serial
    S_CSV=$("$ROOT/WP1_serial/tsp_serial" "$f" "$ITER" "$TEMP" | grep "^CSV")
    S_COST=$(echo "$S_CSV" | cut -d, -f3)
    S_TIME=$(echo "$S_CSV" | cut -d, -f4)

    # OpenMP
    O_CSV=$("$ROOT/WP2_openmp/tsp_openmp" "$f" "$ITER" "$TEMP" "" "$NUM_THREADS" 2>/dev/null \
            || "$ROOT/WP2_openmp/tsp_openmp" "$f" "$ITER" "$TEMP" | grep "^CSV")
    # Handle auto cooling rate
    O_CSV=$(echo "$O_CSV" | grep "^CSV" | head -1)
    O_COST=$(echo "$O_CSV" | cut -d, -f4)
    O_TIME=$(echo "$O_CSV" | cut -d, -f5)

    # CUDA
    C_COST="N/A"; C_TIME="N/A"; SP_C="N/A"
    if $HAS_CUDA; then
        C_CSV=$("$ROOT/WP3_cuda/tsp_cuda" "$f" "$ITER" "$TEMP" 256 | grep "^CSV")
        C_COST=$(echo "$C_CSV" | cut -d, -f4)
        C_TIME=$(echo "$C_CSV" | cut -d, -f5)
        SP_C=$(python3 -c "print(f'{float($S_TIME)/float($C_TIME):.2f}')")
    fi

    SP_O=$(python3 -c "print(f'{float($S_TIME)/float($O_TIME):.2f}')")

    printf "  %-14s n=%-5s  Serial: %10s (%ss)  OMP: %10s (%ss) %sx" \
        "$INST" "$N" "$S_COST" "$S_TIME" "$O_COST" "$O_TIME" "$SP_O"
    $HAS_CUDA && printf "  CUDA: %10s (%ss) %sx" "$C_COST" "$C_TIME" "$SP_C"
    echo ""

    echo "TSP,$INST,$N,$S_COST,$S_TIME,$O_COST,$O_TIME,$C_COST,$C_TIME,$SP_O,$SP_C,$NUM_THREADS" >> "$RESULTS"
done

# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "████████████████████████████████████████████████████████████████"
echo "  PROBLEM 2: MAXIMUM CLIQUE PROBLEM (MCP)"
echo "████████████████████████████████████████████████████████████████"

for f in "$ROOT/data/graphs/rand_n100_d50.dimacs" \
         "$ROOT/data/graphs/rand_n150_d50.dimacs" \
         "$ROOT/data/graphs/rand_n200_d50.dimacs" \
         "$ROOT/data/graphs/rand_n50_d90.dimacs"; do
    [ ! -f "$f" ] && continue
    INST=$(basename "$f" .dimacs)

    S_CSV=$("$ROOT/WP1_serial/max_clique_serial" "$f" | grep "^CSV")
    S_SZ=$(echo "$S_CSV" | cut -d, -f4)
    S_TIME=$(echo "$S_CSV" | cut -d, -f5)

    O_CSV=$("$ROOT/WP2_openmp/max_clique_openmp" "$f" "$NUM_THREADS" | grep "^CSV")
    O_SZ=$(echo "$O_CSV" | cut -d, -f5)
    O_TIME=$(echo "$O_CSV" | cut -d, -f6)

    C_SZ="N/A"; C_TIME="N/A"; SP_C="N/A"
    if $HAS_CUDA; then
        C_CSV=$("$ROOT/WP3_cuda/max_clique_cuda" "$f" | grep "^CSV")
        C_SZ=$(echo "$C_CSV" | cut -d, -f4)
        C_TIME=$(echo "$C_CSV" | cut -d, -f5)
        SP_C=$(python3 -c "print(f'{float($S_TIME)/float($C_TIME):.2f}')")
    fi

    SP_O=$(python3 -c "print(f'{float($S_TIME)/float($O_TIME):.2f}')")

    printf "  %-20s  Serial: clique=%s (%ss)  OMP: clique=%s (%ss) %sx" \
        "$INST" "$S_SZ" "$S_TIME" "$O_SZ" "$O_TIME" "$SP_O"
    $HAS_CUDA && printf "  CUDA: clique=%s (%ss) %sx" "$C_SZ" "$C_TIME" "$SP_C"
    echo ""

    echo "MCP,$INST,$S_SZ,$S_SZ,$S_TIME,$O_SZ,$O_TIME,$C_SZ,$C_TIME,$SP_O,$SP_C,$NUM_THREADS" >> "$RESULTS"
done

# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "████████████████████████████████████████████████████████████████"
echo "  PROBLEM 3: 0/1 KNAPSACK PROBLEM"
echo "████████████████████████████████████████████████████████████████"

for f in "$ROOT/data/knapsack/ks_n1000_W50000.txt" \
         "$ROOT/data/knapsack/ks_n2000_W100000.txt" \
         "$ROOT/data/knapsack/ks_n5000_W50000.txt"; do
    [ ! -f "$f" ] && continue
    INST=$(basename "$f" .txt)

    S_CSV=$("$ROOT/WP1_serial/knapsack_serial" "$f" | grep "^CSV")
    S_VAL=$(echo "$S_CSV" | cut -d, -f4)
    S_TIME=$(echo "$S_CSV" | cut -d, -f5)

    O_CSV=$("$ROOT/WP2_openmp/knapsack_openmp" "$f" "$NUM_THREADS" | grep "^CSV")
    O_VAL=$(echo "$O_CSV" | cut -d, -f5)
    O_TIME=$(echo "$O_CSV" | cut -d, -f6)

    C_VAL="N/A"; C_TIME="N/A"; SP_C="N/A"
    if $HAS_CUDA; then
        C_CSV=$("$ROOT/WP3_cuda/knapsack_cuda" "$f" | grep "^CSV")
        C_VAL=$(echo "$C_CSV" | cut -d, -f4)
        C_TIME=$(echo "$C_CSV" | cut -d, -f5)
        SP_C=$(python3 -c "print(f'{float($S_TIME)/float($C_TIME):.2f}')")
    fi

    SP_O=$(python3 -c "print(f'{float($S_TIME)/float($O_TIME):.2f}')")

    printf "  %-24s  Serial: %s (%ss)  OMP: %s (%ss) %sx" \
        "$INST" "$S_VAL" "$S_TIME" "$O_VAL" "$O_TIME" "$SP_O"
    $HAS_CUDA && printf "  CUDA: %s (%ss) %sx" "$C_VAL" "$C_TIME" "$SP_C"
    echo ""

    echo "KS,$INST,,$S_VAL,$S_TIME,$O_VAL,$O_TIME,$C_VAL,$C_TIME,$SP_O,$SP_C,$NUM_THREADS" >> "$RESULTS"
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Results saved to: $RESULTS"
echo "════════════════════════════════════════════════════════════════"
