#!/usr/bin/env bash
# ============================================================================
#  Benchmark Runner — TSP / MCP / Knapsack (Serial, OpenMP, CPU-Opt)
#  macOS compatible (no 'timeout' dependency)
# ============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS="$ROOT/benchmark/results"
DATA="$ROOT/data/public_datas"

mkdir -p "$RESULTS"

echo "================================================================"
echo "  Benchmark Runner — $(date)"
echo "  Root: $ROOT"
echo "================================================================"

# ============================================================================
#  TSP BENCHMARKS
# ============================================================================
echo ""
echo "================================================================"
echo "  TSP Benchmarks"
echo "================================================================"

TSP_DIR="$ROOT/TSP_SOLUTIONS"
TSP_DATA="$DATA/TSP"
TSP_CSV="$RESULTS/tsp_results.csv"

echo "problem,variant,dataset,n,cost,time_s" > "$TSP_CSV"

TSP_DATASETS=("berlin52.tsp" "lin105.tsp" "pcb442.tsp" "rw1621.tsp")

for ds in "${TSP_DATASETS[@]}"; do
    dsname="${ds%.tsp}"
    dsfile="$TSP_DATA/$ds"
    [[ ! -f "$dsfile" ]] && echo "  [SKIP] $ds" && continue

    echo "  --- $ds ---"

    # --- Serial ---
    echo -n "    serial ... "
    tmpout=$(mktemp)
    if "$TSP_DIR/tsp_serial" "$dsfile" > "$tmpout" 2>&1; then
        csv_line=$(grep "^CSV," "$tmpout" | head -1)
        if [[ -n "$csv_line" ]]; then
            IFS=',' read -r _ n cost time_s _ _ <<< "$csv_line"
            echo "TSP,serial,$dsname,$n,$cost,$time_s" >> "$TSP_CSV"
            echo "OK (cost=$cost, time=${time_s}s)"
        else
            echo "NO CSV"
        fi
    else
        echo "FAIL"
    fi
    rm -f "$tmpout"

    # --- OpenMP ---
    echo -n "    openmp ... "
    tmpout=$(mktemp)
    if "$TSP_DIR/tsp_openmp" file "$dsfile" > "$tmpout" 2>&1; then
        csv_line=$(grep "^CSV," "$tmpout" | head -1)
        if [[ -n "$csv_line" ]]; then
            IFS=',' read -r _ n threads cost time_s _ _ <<< "$csv_line"
            echo "TSP,openmp,$dsname,$n,$cost,$time_s" >> "$TSP_CSV"
            echo "OK (cost=$cost, time=${time_s}s)"
        else
            echo "NO CSV"
        fi
    else
        echo "FAIL"
    fi
    rm -f "$tmpout"

    # --- CPU-Opt ---
    echo -n "    cpu_opt ... "
    tmpout=$(mktemp)
    if "$TSP_DIR/tsp_cpu_opt" file "$dsfile" > "$tmpout" 2>&1; then
        csv_line=$(grep "^CSV," "$tmpout" | head -1)
        if [[ -n "$csv_line" ]]; then
            IFS=',' read -r _ n threads cost time_s <<< "$csv_line"
            echo "TSP,cpu_opt,$dsname,$n,$cost,$time_s" >> "$TSP_CSV"
            echo "OK (cost=$cost, time=${time_s}s)"
        else
            echo "NO CSV"
        fi
    else
        echo "FAIL"
    fi
    rm -f "$tmpout"
done

echo ""
echo "  TSP results:"
cat "$TSP_CSV"

# ============================================================================
#  MCP BENCHMARKS
# ============================================================================
echo ""
echo "================================================================"
echo "  MCP Benchmarks"
echo "================================================================"

MCP_DIR="$ROOT/MCP_SOLUTIONS"
MCP_DATA="$DATA/MCP"
MCP_CSV="$RESULTS/mcp_results.csv"

echo "problem,variant,dataset,n,m,clique_size,time_s,nodes" > "$MCP_CSV"

MCP_DATASETS=("C125.9.clq" "keller4.clq" "brock400_2.clq")

for ds in "${MCP_DATASETS[@]}"; do
    dsname="${ds%.clq}"
    dsfile="$MCP_DATA/$ds"
    [[ ! -f "$dsfile" ]] && echo "  [SKIP] $ds" && continue

    echo "  --- $ds ---"

    # --- Serial ---
    echo -n "    serial ... "
    tmpout=$(mktemp)
    if "$MCP_DIR/max_clique_serial" "$dsfile" > "$tmpout" 2>&1; then
        csv_line=$(grep "^CSV," "$tmpout" | head -1)
        if [[ -n "$csv_line" ]]; then
            IFS=',' read -r _ n m clique_size time_s nodes <<< "$csv_line"
            echo "MCP,serial,$dsname,$n,$m,$clique_size,$time_s,$nodes" >> "$MCP_CSV"
            echo "OK (clique=$clique_size, time=${time_s}s)"
        else
            echo "NO CSV"
        fi
    else
        echo "FAIL"
    fi
    rm -f "$tmpout"

    # --- OpenMP ---
    echo -n "    openmp ... "
    tmpout=$(mktemp)
    if "$MCP_DIR/max_clique_openmp" file "$dsfile" > "$tmpout" 2>&1; then
        csv_line=$(grep "^CSV," "$tmpout" | head -1)
        if [[ -n "$csv_line" ]]; then
            IFS=',' read -r _ n m threads clique_size time_s nodes <<< "$csv_line"
            echo "MCP,openmp,$dsname,$n,$m,$clique_size,$time_s,$nodes" >> "$MCP_CSV"
            echo "OK (clique=$clique_size, time=${time_s}s)"
        else
            echo "NO CSV"
        fi
    else
        echo "FAIL"
    fi
    rm -f "$tmpout"

    # --- CPU-Opt ---
    echo -n "    cpu_opt ... "
    tmpout=$(mktemp)
    if "$MCP_DIR/max_clique_cpu_opt" "$dsfile" > "$tmpout" 2>&1; then
        csv_line=$(grep "^CSV," "$tmpout" | head -1)
        if [[ -n "$csv_line" ]]; then
            IFS=',' read -r _ n m threads clique_size time_s nodes <<< "$csv_line"
            echo "MCP,cpu_opt,$dsname,$n,$m,$clique_size,$time_s,$nodes" >> "$MCP_CSV"
            echo "OK (clique=$clique_size, time=${time_s}s)"
        else
            echo "NO CSV"
        fi
    else
        echo "FAIL"
    fi
    rm -f "$tmpout"
done

echo ""
echo "  MCP results:"
cat "$MCP_CSV"

# ============================================================================
#  KNAPSACK BENCHMARKS
# ============================================================================
echo ""
echo "================================================================"
echo "  Knapsack Benchmarks"
echo "================================================================"

KNAP_DIR="$ROOT/KNAPSACK_SOLUTIONS"
KNAP_DATA="$DATA/KNAPSACK/Datas"
KNAP_SOL="$DATA/KNAPSACK/Solutions"
KNAP_CSV="$RESULTS/knapsack_results.csv"

echo "problem,variant,dataset,n,W,optimal_value,time_s,known_optimal" > "$KNAP_CSV"

KNAP_DATASETS=(
    "knapPI_1_100_1000_1"
    "knapPI_1_200_1000_1"
    "knapPI_1_500_1000_1"
    "knapPI_1_1000_1000_1"
    "knapPI_1_2000_1000_1"
    "knapPI_1_5000_1000_1"
    "knapPI_1_10000_1000_1"
)

for ds in "${KNAP_DATASETS[@]}"; do
    dsfile="$KNAP_DATA/$ds"
    solfile="$KNAP_SOL/$ds"
    known_opt=""
    [[ -f "$solfile" ]] && known_opt=$(cat "$solfile" | tr -d '[:space:]')

    [[ ! -f "$dsfile" ]] && echo "  [SKIP] $ds" && continue

    echo "  --- $ds ---"

    # --- Serial ---
    echo -n "    serial ... "
    tmpout=$(mktemp)
    if "$KNAP_DIR/knapsack_serial" "$dsfile" > "$tmpout" 2>&1; then
        csv_line=$(grep "^CSV," "$tmpout" | head -1)
        if [[ -n "$csv_line" ]]; then
            IFS=',' read -r _ n W opt time_s dp_cells <<< "$csv_line"
            echo "KNAPSACK,serial,$ds,$n,$W,$opt,$time_s,$known_opt" >> "$KNAP_CSV"
            echo "OK (opt=$opt, known=$known_opt, time=${time_s}s)"
        else
            echo "NO CSV"
        fi
    else
        echo "FAIL"
    fi
    rm -f "$tmpout"

    # --- OpenMP ---
    echo -n "    openmp ... "
    tmpout=$(mktemp)
    if "$KNAP_DIR/knapsack_openmp" file "$dsfile" > "$tmpout" 2>&1; then
        csv_line=$(grep "^CSV," "$tmpout" | head -1)
        if [[ -n "$csv_line" ]]; then
            IFS=',' read -r _ n W threads opt time_s dp_cells <<< "$csv_line"
            echo "KNAPSACK,openmp,$ds,$n,$W,$opt,$time_s,$known_opt" >> "$KNAP_CSV"
            echo "OK (opt=$opt, known=$known_opt, time=${time_s}s)"
        else
            echo "NO CSV"
        fi
    else
        echo "FAIL"
    fi
    rm -f "$tmpout"

    # --- CPU-Opt ---
    echo -n "    cpu_opt ... "
    tmpout=$(mktemp)
    if "$KNAP_DIR/knapsack_cpu_opt" "$dsfile" > "$tmpout" 2>&1; then
        csv_line=$(grep "^CSV," "$tmpout" | head -1)
        if [[ -n "$csv_line" ]]; then
            IFS=',' read -r _ n W threads opt time_s throughput <<< "$csv_line"
            echo "KNAPSACK,cpu_opt,$ds,$n,$W,$opt,$time_s,$known_opt" >> "$KNAP_CSV"
            echo "OK (opt=$opt, known=$known_opt, time=${time_s}s)"
        else
            echo "NO CSV"
        fi
    else
        echo "FAIL"
    fi
    rm -f "$tmpout"
done

echo ""
echo "  Knapsack results:"
cat "$KNAP_CSV"

# ============================================================================
echo ""
echo "================================================================"
echo "  All benchmarks complete!"
echo "  Results in: $RESULTS/"
echo "================================================================"
