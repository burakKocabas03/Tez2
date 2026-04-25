# Graduation Thesis – Work Packages 1 & 2

**Thesis:** A Comparison of NP-hard Problems through Parallel Algorithms Utilizing CPU and GPU-Based Solutions  
**Authors:** Burak Kocabaş & Emin Özgür Elmalı  
**Advisor:** Prof. Dr. Hasan BULUT  

---

## Three Representative NP-hard Problems

| # | Problem | Algorithm | WP1 (Serial) | WP2 (OpenMP) | WP3 (CUDA) |
|---|---------|-----------|:---:|:---:|:---:|
| 1 | **Traveling Salesman (TSP)** | Simulated Annealing + 2-opt | ✅ | ✅ | ➡ WP3 |
| 2 | **Maximum Clique (MCP)** | Branch-and-Bound Bron-Kerbosch | ✅ | ✅ | ➡ WP3 |
| 3 | **0/1 Knapsack** | Dynamic Programming | ✅ | ✅ | ➡ WP3 |

Each problem is chosen to represent a different parallelism type:
- **TSP** → Heuristic search, independent chains (embarrassingly parallel)
- **MCP** → Exact combinatorial search, dynamic load balancing
- **Knapsack** → Exact DP, row-level data parallelism (memory-bound)

---

## Repository Structure

```
Tez2/
├── WP1_serial/
│   ├── tsp_serial.cpp           ← TSP serial (SA + 2-opt)
│   ├── max_clique_serial.cpp    ← MCP serial (B&B Bron-Kerbosch)
│   ├── knapsack_serial.cpp      ← Knapsack serial (DP 2-row rolling)
│   └── Makefile
├── WP2_openmp/
│   ├── tsp_openmp.cpp           ← TSP parallel (island-model SA)
│   ├── max_clique_openmp.cpp    ← MCP parallel (parallel BK + dynamic schedule)
│   ├── knapsack_openmp.cpp      ← Knapsack parallel (row-level DP)
│   └── Makefile
├── data/
│   ├── berlin52.tsp             ← Classic 52-city TSP (optimal = 7542)
│   ├── generate_instances.py    ← Random TSP generator
│   ├── generate_graphs.py       ← Random graph generator (DIMACS format)
│   ├── generate_knapsack.py     ← Random knapsack generator
│   ├── graphs/                  ← Generated graph instances
│   └── knapsack/                ← Generated knapsack instances
└── benchmark/
    ├── run_benchmark.sh         ← TSP comparison
    ├── run_benchmark_clique.sh  ← MCP comparison
    └── run_benchmark_knapsack.sh← Knapsack comparison
```

---

## Build

### Prerequisites (macOS)

```bash
brew install libomp   # required for WP2 OpenMP on Apple Clang
```

### Compile all

```bash
# All serial binaries
cd WP1_serial && make

# All OpenMP binaries
cd WP2_openmp && make
```

---

## Problem 1: Traveling Salesman Problem (TSP)

**NP-hard.** Find the shortest tour through N cities visiting each exactly once.

**Algorithm:** Simulated Annealing with 2-opt neighbourhood moves.

**Parallel strategy (WP2):** Island Model — P independent SA chains, each starting
from a different city with a unique random seed, for `maxIter/P` iterations each.
No synchronisation during search → near-linear speedup.

### Run

```bash
./WP1_serial/tsp_serial  <tsp_file> [max_iter] [init_temp] [cooling_rate]
./WP2_openmp/tsp_openmp  <tsp_file> [max_iter] [init_temp] [cooling_rate] [threads]
```

### Benchmark Results (8 threads, Apple M-series)

| Instance | n | Serial (s) | OpenMP (s) | Speedup | Solution |
|----------|---|-----------|-----------|---------|---------|
| berlin52 | 52 | 0.026 | 0.009 | **2.9×** | ~7544 (optimal: 7542) |
| random100 | 100 | 0.087 | 0.015 | **5.8×** | improved ~16% over NN |
| random500 | 500 | 0.44 | 0.05 | **8.8×** | improved ~9% over NN |
| random1000 | 1000 | 5.95 | 1.38 | **4.3×** | improved ~14% over NN |

---

## Problem 2: Maximum Clique Problem (MCP)

**NP-hard.** Find the largest subset of vertices where every pair is connected by an edge.

**Algorithm:** Branch-and-Bound Bron-Kerbosch with cardinality pruning and
degree-based vertex ordering.

**Parallel strategy (WP2):** Vertex-level parallelism — P threads each explore
independent subtrees starting from different vertices. `schedule(dynamic,1)` is
essential because subtree sizes vary wildly (hard instances near the front).

**Key insight: Superlinear Speedup.**  
Because threads share a global best clique size as a pruning bound, later threads
benefit from the best solution already found by earlier threads, pruning more branches
than a single thread would. This allows speedup > P (superlinear).

### Run

```bash
./WP1_serial/max_clique_serial  <graph.dimacs>
./WP2_openmp/max_clique_openmp  <graph.dimacs> [threads]
```

### Generate instances

```bash
python3 data/generate_graphs.py   # creates data/graphs/*.dimacs
```

### Benchmark Results (8 threads, Apple M-series)

| Instance | n | m | Serial (s) | OpenMP (s) | Speedup | Clique |
|----------|---|---|-----------|-----------|---------|--------|
| rand_n50_d50 | 50 | 603 | 0.000 | 0.002 | — (too small) | 8 |
| rand_n100_d50 | 100 | 2492 | 0.011 | 0.005 | **2.2×** | 9 |
| rand_n150_d50 | 150 | 5539 | 0.042 | 0.013 | **3.2×** | 10 |
| rand_n200_d50 | 200 | 9903 | 0.425 | 0.052 | **8.1×** ⭐ | 11 |
| rand_n50_d90 | 50 | 1101 | 0.158 | 0.089 | **1.8×** | 22 |

⭐ Superlinear speedup: parallel pruning reduces total nodes explored below serial count.

---

## Problem 3: 0/1 Knapsack Problem

**NP-hard.** Select items to maximise total value without exceeding capacity W.

**Algorithm:** Dynamic Programming (exact).  
`dp[i][c] = max(dp[i-1][c], dp[i-1][c-w[i]] + v[i])`  
Each row of the DP table depends only on the previous row.

**Parallel strategy (WP2):** Row-level parallelism — all W+1 cells of a row are
computed in parallel (`#pragma omp for`) since they have no intra-row dependencies.
One persistent thread team with one barrier per row (double-buffer avoids `omp single`).

### Run

```bash
./WP1_serial/knapsack_serial   <instance.txt>
./WP2_openmp/knapsack_openmp   <instance.txt> [threads]
```

### Generate instances

```bash
python3 data/generate_knapsack.py   # creates data/knapsack/*.txt
```

### Benchmark Results (8 threads, Apple M-series)

| Instance | n | W | Serial (s) | OpenMP (s) | Speedup |
|----------|---|---|-----------|-----------|---------|
| ks_n200_W1000000 | 200 | 1,000,000 | 0.40 | 0.36 | **1.1×** |
| ks_n500_W1000000 | 500 | 1,000,000 | 0.99 | 1.06 | ~1× |
| ks_n1000_W50000 | 1000 | 50,000 | 0.12 | 0.45 | 0.3× |

### Why CPU speedup is limited for Knapsack DP

The knapsack DP is **memory-bandwidth-bound** rather than compute-bound:

1. **Sequential row dependency**: Row `i` must complete before row `i+1` begins.
   Every item requires one barrier synchronisation.
   
2. **Apple Silicon unified memory**: All CPU cores share a single memory controller.
   Multiple threads competing for the same memory bus provide no bandwidth increase
   (unlike server CPUs with multi-channel DDR5).

3. **E-core + P-core heterogeneity**: Apple M chips have 4 fast P-cores and 4 slow
   E-cores. With static scheduling, P-core threads wait for E-cores at each barrier.

**GPU advantage (WP3):** The CUDA implementation is expected to achieve high speedup because:
- GPU dedicated memory (GDDR6/HBM) provides 5–10× the bandwidth of CPU memory
- Thousands of homogeneous CUDA cores with no E/P heterogeneity
- Warp-level synchronisation replaces expensive CPU barriers

This demonstrates a key HPC principle: **not all problems are equally suited for CPU parallelisation**.
The Roofline model predicts that memory-bound kernels benefit more from GPU (bandwidth) than CPU (compute).

---

## Running All Benchmarks

```bash
# Generate all data
cd data
python3 generate_instances.py        # TSP: random100.tsp … random5000.tsp
python3 generate_graphs.py           # Graphs: rand_n*.dimacs
python3 generate_knapsack.py         # Knapsack: ks_n*_W*.txt

# Run benchmarks
cd benchmark
./run_benchmark.sh               # TSP
./run_benchmark_clique.sh        # Max Clique (skip large dense instances)
./run_benchmark_knapsack.sh      # Knapsack

# Results
cat results.csv
cat results_clique.csv
cat results_knapsack.csv
```

---

## Key Metrics

| Metric | TSP | Max Clique | Knapsack |
|--------|-----|-----------|---------|
| Solution quality | Tour cost (↓ better) | Clique size (↑ better) | Optimal value (exact) |
| Execution time | Wall-clock seconds | Wall-clock seconds | Wall-clock seconds |
| Speedup | `T_serial / T_openmp` | `T_serial / T_openmp` | `T_serial / T_openmp` |
| Notes | Heuristic (approx.) | Exact (optimal) | Exact (optimal) |

---

## Work Package Timeline

| WP | Task | Period | Owner |
|----|------|--------|-------|
| **WP1** | Serial baseline (all 3 problems) | 01.11.2025–20.12.2025 | Burak & Emin |
| **WP2** | CPU OpenMP (all 3 problems) | 20.12.2025–01.03.2026 | Burak |
| **WP3** | GPU CUDA (all 3 problems) | 20.12.2025–01.03.2026 | Emin |
| **WP4** | Systematic Benchmarking | 01.03.2026–01.04.2026 | Both |
| **WP5** | Comparative Analysis & Report | 01.04.2026–25.04.2026 | Both |
