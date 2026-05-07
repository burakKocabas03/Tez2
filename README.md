# Graduation Thesis – NP-hard Problem Comparison

**Thesis:** A Comparison of NP-hard Problems through Parallel Algorithms Utilizing CPU and GPU-Based Solutions  
**Authors:** Burak Kocabaş & Emin Özgür Elmalı  
**Advisor:** Prof. Dr. Hasan BULUT  

---

## Three Representative NP-hard Problems

| # | Problem | Algorithm | Serial | OpenMP | CUDA | CPU-Opt | GPU-Opt |
|---|---------|-----------|:---:|:---:|:---:|:---:|:---:|
| 1 | **Traveling Salesman (TSP)** | SA + 2-opt / EAX GA | ✅ | ✅ | ✅ | ✅ | ✅ |
| 2 | **Maximum Clique (MCP)** | Branch-and-Bound Bron-Kerbosch | ✅ | ✅ | ✅ | ✅ | ✅ |
| 3 | **0/1 Knapsack** | Dynamic Programming | ✅ | ✅ | ✅ | ✅ | ✅ |

Each problem is chosen to represent a different parallelism type:
- **TSP** → Heuristic search, independent chains (embarrassingly parallel)
- **MCP** → Exact combinatorial search, dynamic load balancing
- **Knapsack** → Exact DP, row-level data parallelism (memory-bound)

---

## Repository Structure

```
Tez2/
├── TSP_SOLUTIONS/                ← All TSP implementations
│   ├── tsp_serial.cpp            ← WP1: Serial SA + 2-opt
│   ├── tsp_openmp.cpp            ← WP2: Island-model SA (OpenMP)
│   ├── tsp_cuda.cu              ← WP3: CUDA SA
│   ├── tsp_cpu_opt.cpp           ← WP4: EAX-inspired GA (OpenMP)
│   ├── tsp_gpu_opt.cu           ← WP4: GPU-optimized SA
│   └── Makefile
├── MCP_SOLUTIONS/                ← All Maximum Clique implementations
│   ├── max_clique_serial.cpp     ← WP1: Serial B&B Bron-Kerbosch
│   ├── max_clique_openmp.cpp     ← WP2: Parallel BK (OpenMP)
│   ├── max_clique_cuda.cu       ← WP3: CUDA BK
│   ├── max_clique_cpu_opt.cpp    ← WP4: Bitset-optimized BK (OpenMP)
│   ├── max_clique_gpu_opt.cu    ← WP4: GPU-optimized BK
│   └── Makefile
├── KNAPSACK_SOLUTIONS/           ← All Knapsack implementations
│   ├── knapsack_serial.cpp       ← WP1: Serial DP (2-row rolling)
│   ├── knapsack_openmp.cpp       ← WP2: Row-level parallel DP (OpenMP)
│   ├── knapsack_cuda.cu         ← WP3: CUDA row-parallel DP
│   ├── knapscak_cpu_opt.cpp      ← WP4: Sparse DP / Hybrid (OpenMP)
│   ├── knapsack_gpu_opt.cu      ← WP4: GPU-optimized DP
│   └── Makefile
├── WP1_serial/                   ← (Legacy) Serial-only build
│   └── Makefile
├── WP2_openmp/                   ← (Legacy) OpenMP-only build
│   └── Makefile
├── WP3_cuda/                     ← (Legacy) CUDA-only build
│   └── Makefile
├── WP4_optimized/                ← (Legacy) Optimized build
│   └── Makefile
├── data/
│   ├── berlin52.tsp              ← Classic 52-city TSP (optimal = 7542)
│   ├── generate_instances.py     ← Random TSP generator
│   ├── generate_graphs.py        ← Random graph generator (DIMACS format)
│   ├── generate_knapsack.py      ← Random knapsack generator
│   ├── graphs/                   ← Generated graph instances
│   └── knapsack/                 ← Generated knapsack instances
├── benchmark/
│   ├── run_benchmark.sh          ← TSP comparison
│   ├── run_benchmark_clique.sh   ← MCP comparison
│   ├── run_benchmark_knapsack.sh ← Knapsack comparison
│   └── run_benchmark_all.sh      ← All problems
└── figures/                      ← Report figures
```

---

## Build

### Prerequisites

#### macOS (Apple Silicon / Intel)

```bash
brew install libomp   # required for OpenMP on Apple Clang
```

#### Linux (x86_64)

```bash
# Ubuntu / Debian
sudo apt install g++ make

# OpenMP is built into GCC — no extra package needed

# For CUDA (WP3 / GPU targets)
# Install NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
```

> **Note:** Makefile'lar işletim sistemini otomatik algılar (`uname -s`).  
> macOS'ta Homebrew libomp, Linux'ta GCC'nin dahili OpenMP desteği kullanılır.

### Compile by problem

Each problem has its own directory with a self-contained Makefile:

```bash
# TSP — all CPU targets (serial + openmp + cpu_opt)
cd TSP_SOLUTIONS && make

# MCP — all CPU targets
cd MCP_SOLUTIONS && make

# Knapsack — all CPU targets
cd KNAPSACK_SOLUTIONS && make
```

### Makefile targets

Each Makefile supports the following targets:

| Target | Description |
|--------|-------------|
| `make` / `make cpu` | Build all CPU targets (serial + openmp + cpu_opt) |
| `make gpu` | Build all GPU targets (cuda + gpu_opt) — requires `nvcc` |
| `make serial` | Build only the serial baseline |
| `make openmp` | Build only the OpenMP version |
| `make cuda` | Build only the CUDA version |
| `make cpu_opt` | Build only the CPU-optimized version |
| `make gpu_opt` | Build only the GPU-optimized version |
| `make clean` | Remove all compiled binaries |

### CUDA architecture

Default: `-arch=sm_75` (Turing). Change in the Makefile to match your GPU:
- `sm_70` = Volta (V100)
- `sm_80` = Ampere (A100, RTX 3000)
- `sm_86` = Ampere (RTX 3060+)
- `sm_89` = Ada Lovelace (RTX 4000)

---

## Problem 1: Traveling Salesman Problem (TSP)

**NP-hard.** Find the shortest tour through N cities visiting each exactly once.

| Version | Algorithm | Key Feature |
|---------|-----------|-------------|
| Serial | Simulated Annealing + 2-opt | Single-threaded baseline |
| OpenMP | Island-model SA | P independent chains, near-linear speedup |
| CUDA | GPU SA | Massive parallel neighbourhood evaluation |
| CPU-Opt | EAX-inspired Genetic Algorithm | AB-cycle crossover + 2-opt refinement |
| GPU-Opt | GPU-optimized SA | Tuned kernel for bandwidth utilization |

### Run

```bash
./TSP_SOLUTIONS/tsp_serial   <tsp_file> [max_iter] [init_temp] [cooling_rate]
./TSP_SOLUTIONS/tsp_openmp   <mode> <arg> [max_iter] [init_temp] [cooling_rate] [threads]
./TSP_SOLUTIONS/tsp_cpu_opt  <mode> <arg> [pop_size] [max_gen] [threads]
```

---

## Problem 2: Maximum Clique Problem (MCP)

**NP-hard.** Find the largest subset of vertices where every pair is connected by an edge.

| Version | Algorithm | Key Feature |
|---------|-----------|-------------|
| Serial | B&B Bron-Kerbosch | Degree ordering + cardinality pruning |
| OpenMP | Parallel BK | Vertex-level parallelism, dynamic scheduling |
| CUDA | GPU BK | GPU-parallel branch exploration |
| CPU-Opt | Bitset-optimized BK | 64-bit bitsets, greedy colouring bounds |
| GPU-Opt | GPU-optimized BK | Tuned GPU kernel |

### Run

```bash
./MCP_SOLUTIONS/max_clique_serial   <graph.dimacs>
./MCP_SOLUTIONS/max_clique_openmp   <graph.dimacs> [threads]
./MCP_SOLUTIONS/max_clique_cpu_opt  <graph.dimacs> [threads]
```

### Generate instances

```bash
python3 data/generate_graphs.py   # creates data/graphs/*.dimacs
```

---

## Problem 3: 0/1 Knapsack Problem

**NP-hard.** Select items to maximise total value without exceeding capacity W.

| Version | Algorithm | Key Feature |
|---------|-----------|-------------|
| Serial | DP (2-row rolling) | O(nW) time, O(W) space |
| OpenMP | Row-level parallel DP | `#pragma omp for` on capacity loop |
| CUDA | Row-parallel DP | One kernel per item, massive parallelism |
| CPU-Opt | Sparse DP / Hybrid | Auto-selects dense vs sparse based on W |
| GPU-Opt | GPU-optimized DP | Same kernel as WP3 (already optimal) |

### Run

```bash
./KNAPSACK_SOLUTIONS/knapsack_serial   <instance.txt>
./KNAPSACK_SOLUTIONS/knapsack_openmp   random <num_items> <capacity> [threads]
./KNAPSACK_SOLUTIONS/knapscak_cpu_opt  <input_file> [threads]
./KNAPSACK_SOLUTIONS/knapscak_cpu_opt  random <num_items> <capacity> [seed] [threads]
```

### Generate instances

```bash
python3 data/generate_knapsack.py   # creates data/knapsack/*.txt
```

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
./run_benchmark_clique.sh        # Max Clique
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
| Speedup | `T_serial / T_parallel` | `T_serial / T_parallel` | `T_serial / T_parallel` |
| Notes | Heuristic (approx.) | Exact (optimal) | Exact (optimal) |

---

## Work Package Timeline

| WP | Task | Period | Owner |
|----|------|--------|-------|
| **WP1** | Serial baseline (all 3 problems) | 01.11.2025–20.12.2025 | Burak & Emin |
| **WP2** | CPU OpenMP (all 3 problems) | 20.12.2025–01.03.2026 | Burak |
| **WP3** | GPU CUDA (all 3 problems) | 20.12.2025–01.03.2026 | Emin |
| **WP4** | CPU & GPU Optimizations | 01.03.2026–01.04.2026 | Both |
| **WP5** | Comparative Analysis & Report | 01.04.2026–25.04.2026 | Both |
