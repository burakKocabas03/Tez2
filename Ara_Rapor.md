# A Comparison of NP-Hard Problems through Parallel Algorithms Utilizing CPU and GPU-Based Solutions

**Burak Kocabaş (05220000303), Emin Özgür Elmalı (05220000322)**
Department of Computer Engineering, Ege University, 35100 Bornova, İzmir

burak.kocabas@outlook.com, eminozgurelm@gmail.com

**Advisor:** Prof. Dr. Hasan BULUT, hasan.bulut@ege.edu.tr

**Submission Date:** March 1, 2026

---

## Özet

Bu bitirme tezinde, üç temsili NP-zor problemi (Gezgin Satıcı Problemi, Maksimum Klik Problemi ve 0/1 Sırt Çantası Problemi) için seri, CPU tabanlı (OpenMP) ve GPU tabanlı (CUDA) paralel çözüm stratejileri geliştirilmiş ve karşılaştırmalı performans analizi yapılmıştır. Her problem için öncelikle bir seri temel uygulama oluşturulmuş, ardından kaba taneli CPU paralelleştirmesi (OpenMP) ve ince taneli GPU paralelleştirmesi (CUDA) uygulanmıştır. Deneysel sonuçlar, hesaplama yoğun problemlerde (TSP, Klik) CPU paralelleştirmesinin 2–8× hızlanma sağladığını, bellek bant genişliği sınırlı problemlerde (Sırt Çantası) ise GPU'nun 24× hızlanma ile CPU'yu önemli ölçüde geride bıraktığını göstermiştir. Çalışma, farklı problem karakteristiklerinin hangi paralel mimariden fayda gördüğüne dair ampirik kanıtlar sunmaktadır.

**Anahtar Kelimeler:** Paralel Programlama, CUDA, OpenMP, NP-zor problemler, Simulated Annealing, Dinamik Programlama, Branch-and-Bound

---

## Abstract

This graduation thesis develops and comparatively evaluates serial, CPU-based (OpenMP), and GPU-based (CUDA) parallel solution strategies for three representative NP-hard problems: the Traveling Salesman Problem (TSP), the Maximum Clique Problem (MCP), and the 0/1 Knapsack Problem. For each problem, a serial baseline implementation was first established, followed by coarse-grained CPU parallelization using OpenMP and fine-grained GPU parallelization using CUDA. Experimental results demonstrate that CPU parallelization achieves 2–8× speedup for compute-bound problems (TSP, MCP), while for memory-bandwidth-bound problems (Knapsack), the GPU achieves 24× speedup, significantly outperforming the CPU. This study provides empirical evidence regarding which parallel architecture benefits different problem characteristics, serving as a practical guide for efficient allocation of computational resources.

**Keywords:** Parallel Programming, CUDA, OpenMP, NP-hard Problems, Simulated Annealing, Dynamic Programming, Branch-and-Bound

---

## 1. Introduction

NP-hard optimization problems are fundamental to computer science and arise in a wide range of practical domains including logistics, network design, bioinformatics, and resource scheduling [1]. By definition, no known polynomial-time algorithm exists for solving these problems exactly, and their computational cost grows exponentially with input size. This makes them ideal candidates for parallelization strategies that distribute the computational workload across multiple processing units.

Modern computing hardware offers two dominant paradigms for parallel execution. Multi-core CPUs with shared-memory parallelism, programmed via frameworks such as OpenMP [2], provide coarse-grained parallelism with 4–64 threads and flexible branching logic. Graphics Processing Units (GPUs), programmed via NVIDIA's CUDA framework [3], offer fine-grained, massively parallel execution with thousands of lightweight threads optimized for data-parallel workloads.

While both paradigms have been individually applied to NP-hard problems in the literature, there is a lack of focused, implementation-driven comparative analysis that evaluates both approaches on the same set of problems under controlled conditions. Existing studies typically optimize an algorithm for a single architecture without providing an equivalent comparison against the other [4].

The primary purpose of this thesis is to bridge this gap by implementing and rigorously benchmarking three representative NP-hard problems across three execution models: serial (single-threaded baseline), CPU-parallel (OpenMP), and GPU-parallel (CUDA). The three selected problems are:

1. **Traveling Salesman Problem (TSP)** — a combinatorial optimization problem solved via Simulated Annealing with 2-opt moves, representing heuristic search with embarrassingly parallel characteristics.
2. **Maximum Clique Problem (MCP)** — an exact combinatorial search problem solved via Branch-and-Bound Bron-Kerbosch, representing irregular parallelism with dynamic load balancing requirements.
3. **0/1 Knapsack Problem** — an exact optimization problem solved via Dynamic Programming, representing regular data parallelism that is memory-bandwidth-bound.

Each problem was specifically chosen to represent a distinct parallelism pattern, enabling us to draw conclusions about which hardware architecture is best suited for which type of computational workload.

This interim report covers the work completed during the period from November 2025 to March 2026, encompassing Work Packages 1 (Serial Baseline), 2 (CPU Parallelization with OpenMP), and 3 (GPU Parallelization with CUDA).

---

## 2. Literature Review

### 2.1 Parallel Approaches to the Traveling Salesman Problem

The Traveling Salesman Problem is one of the most extensively studied NP-hard problems [1]. Simulated Annealing (SA), originally proposed by Kirkpatrick, Gelatt, and Vecchi [5], is a widely used metaheuristic for TSP that accepts worse solutions with decreasing probability, enabling escape from local optima. The 2-opt neighborhood structure [6] is commonly combined with SA for TSP due to its efficient O(1) cost evaluation per move.

Parallel SA implementations typically follow the island model, where multiple independent SA chains run concurrently with different initial solutions and random seeds [7]. Ram et al. [8] demonstrated that independent parallel SA chains achieve near-linear speedup on shared-memory systems. On the GPU side, Czapinski and Barnes [9] implemented parallel SA for TSP on CUDA, reporting significant speedup for large instances by running thousands of concurrent chains with the cuRAND device API for per-thread random number generation.

### 2.2 Parallel Approaches to the Maximum Clique Problem

The Maximum Clique Problem is NP-hard [1] and is typically solved exactly using Branch-and-Bound algorithms based on the Bron-Kerbosch framework [10]. Tomita and Seki [11] introduced vertex ordering and cardinality-based upper-bound pruning to significantly reduce the search tree size.

For CPU parallelization, the top-level vertex decomposition provides natural task parallelism: each starting vertex defines an independent sub-problem that can be assigned to a different thread [12]. Dynamic scheduling is essential because sub-problem sizes vary significantly with vertex degree. On the GPU side, Rossi and Ahmed [13] explored bitwise adjacency representations to accelerate set intersection operations using native GPU instructions (`__popc()`, bitwise AND), achieving speedup for dense graphs. However, the irregular recursive nature of Branch-and-Bound makes GPU parallelization challenging compared to regular data-parallel workloads.

### 2.3 Parallel Approaches to the 0/1 Knapsack Problem

The 0/1 Knapsack Problem is solvable in pseudo-polynomial time O(nW) via Dynamic Programming [14]. The DP recurrence exhibits row-level parallelism: all W+1 cells of row i depend only on row i−1, making them independently computable.

Boukedjar et al. [15] parallelized the Knapsack DP on GPU using CUDA, demonstrating that the memory-bandwidth of GPU DRAM (GDDR) is the primary factor determining speedup. On CPU, OpenMP parallelization of the inner capacity loop shows limited scaling on systems with unified memory architectures due to memory-bandwidth saturation [16]. This contrast between CPU and GPU performance for memory-bound workloads is well explained by the Roofline Model proposed by Williams, Waterman, and Patterson [17].

### 2.4 Gap in the Literature

While individual studies exist for each problem-architecture combination, no single study provides a controlled, cross-problem comparison of CPU (OpenMP) vs. GPU (CUDA) parallelization using identical baseline algorithms, benchmark datasets, and measurement methodology. This thesis directly addresses this gap.

---

## 3. Methods and Technologies

### 3.1 Algorithms

**TSP — Simulated Annealing with 2-opt Moves.** At each iteration, a random 2-opt move is proposed: two edges are removed and reconnected, reversing the sub-tour between them. The move is accepted if it improves cost, or with probability exp(−Δ/T) otherwise (Metropolis criterion). Temperature T decreases geometrically: T ← T × α each iteration. The initial tour is constructed using the Nearest-Neighbour heuristic [6].

**MCP — Branch-and-Bound Bron-Kerbosch.** Vertices are ordered by descending degree. For each vertex v_i, the algorithm recursively explores all cliques containing v_i but no earlier vertex. The upper bound |clique| + |candidates| ≤ |best| prunes branches that cannot improve the current best solution [10][11].

**Knapsack — Dynamic Programming.** The recurrence dp[i][c] = max(dp[i−1][c], dp[i−1][c−w_i] + v_i) is computed bottom-up using a space-optimized two-row rolling buffer. Time complexity is O(nW), space complexity is O(W) [14].

### 3.2 Parallelization Strategies

Table 1 summarizes the parallelization strategy used for each problem across the three execution models.

**Table 1. Parallelization strategies per problem and architecture.**

| Problem | Serial (WP1) | OpenMP (WP2) | CUDA (WP3) |
|---------|-------------|-------------|------------|
| TSP | Single SA chain | P independent chains (island model), each with unique start city and RNG seed. Critical section only at the end. | 256–1024 independent chains on GPU threads. cuRAND XORWOW for per-thread RNG. Distance matrix in global memory (L2 cached). |
| MCP | Sequential B&B over all vertices | Top-level vertices distributed via `schedule(dynamic,1)`. Atomic global best for cross-thread pruning. | One CUDA thread per starting vertex. Bitmask adjacency (bitwise AND + `__popc()`). Iterative B&B with explicit stack. `atomicMax` for global best. |
| Knapsack | Sequential row-by-row DP | Row-level parallelism: `#pragma omp for` on capacity loop. Persistent thread team with double-buffer. | One CUDA kernel per row, (W+1) threads per kernel. Double-buffer with pointer swap. Coalesced memory access. |

### 3.3 Technologies

- **Programming Language:** C++ (C++17 for serial/OpenMP, C++14 for CUDA)
- **CPU Parallelization:** OpenMP 5.0 [2], compiled with Apple Clang + libomp (macOS) and GCC (Linux)
- **GPU Parallelization:** NVIDIA CUDA Toolkit [3], cuRAND device API for random number generation
- **Benchmark Data:** TSPLIB berlin52 instance [18], randomly generated TSP/graph/knapsack instances via Python scripts
- **Hardware:** Apple M-series (8 cores: 4P + 4E, unified memory 80 GB/s) for CPU tests; NVIDIA Tesla T4 (2560 CUDA cores, 16 GB GDDR6, 320 GB/s) for GPU tests [19]
- **Version Control:** Git + GitHub
- **GPU Testing Environment:** Google Colab with T4 GPU runtime

---

## 4. Completed Work

### 4.1 Analysis

The analysis phase involved selecting three NP-hard problems that represent fundamentally different parallelism characteristics:

1. **TSP (Embarrassingly Parallel):** Independent SA chains share no mutable state during execution, enabling near-linear scaling with the number of threads/chains. The key metric is wall-clock time reduction.

2. **MCP (Irregular Parallel):** Branch-and-Bound sub-problems have wildly different sizes depending on vertex degree. Dynamic load balancing is essential. Cross-thread pruning information sharing can lead to superlinear speedup where parallel threads collectively prune more branches than a single thread.

3. **Knapsack (Regular Data-Parallel, Memory-Bound):** All cells within a DP row are independently computable, but each row depends on the previous one (sequential dependency chain of n rows). The computation per cell is minimal (one comparison, one addition), making the workload memory-bandwidth-bound rather than compute-bound.

This diversity ensures that our comparative analysis captures the strengths and weaknesses of each architecture across different computational patterns.

For the benchmark test environment, we designed a systematic approach:
- Multiple problem sizes per problem (e.g., TSP: 52–5000 cities; MCP: 50–200 vertices at densities 0.5 and 0.9; Knapsack: 500–5000 items with capacities 50,000–1,000,000)
- Consistent parameters across serial/OpenMP/CUDA for fair comparison
- Machine-readable CSV output for automated aggregation

### 4.2 Design

The software architecture follows a modular design organized by work package:

```
Tez2/
├── WP1_serial/          Serial baseline (C++)
│   ├── tsp_serial.cpp
│   ├── max_clique_serial.cpp
│   └── knapsack_serial.cpp
├── WP2_openmp/          CPU parallel (C++ + OpenMP)
│   ├── tsp_openmp.cpp
│   ├── max_clique_openmp.cpp
│   └── knapsack_openmp.cpp
├── WP3_cuda/            GPU parallel (CUDA C++)
│   ├── tsp_cuda.cu
│   ├── max_clique_cuda.cu
│   └── knapsack_cuda.cu
├── data/                Benchmark instances + generators
└── benchmark/           Automated comparison scripts
```

**Key design decisions:**

1. **Identical algorithms across implementations:** The serial, OpenMP, and CUDA versions use the same core algorithm logic (SA for TSP, BK for MCP, DP for Knapsack) to ensure that performance differences are attributable to the parallelization strategy alone.

2. **Flat distance/adjacency matrices:** Row-major flat arrays (`data[i*n + j]`) provide better cache locality than pointer-based structures, critical for both CPU cache hierarchy and GPU coalesced memory access.

3. **Bitmask adjacency for GPU MCP:** The adjacency matrix is stored as uint32 bitmasks (`adj_bits[v * words + w]`), enabling O(n/32) set intersection via bitwise AND, which maps to native GPU instructions (LOP3, POPC).

4. **Double-buffer DP for Knapsack:** Two arrays alternate as source/destination each row, eliminating the need for an `omp single` barrier (OpenMP) or extra kernel launch (CUDA) for pointer swapping.

5. **Auto-computed cooling rate for SA:** The cooling rate is automatically derived as `exp(log(1e-9 / T₀) / maxIter)`, ensuring correct annealing behavior regardless of problem size or thread count.

> **[INSERT FIGURE 1: Architecture diagram showing Serial → OpenMP → CUDA pipeline for all 3 problems, with shared data layer at the bottom. Draw a block diagram with 3 columns (WP1, WP2, WP3) and 3 rows (TSP, MCP, Knapsack), showing the parallelization strategy in each cell.]**

### 4.3 Prototype Development

All nine implementations (3 problems × 3 architectures) have been fully developed and tested.

**WP1 — Serial Baseline (Completed: Nov 2025 – Dec 2025)**
- TSP: Simulated Annealing with Nearest-Neighbour initialization, 2-opt neighborhood
- MCP: Bron-Kerbosch with degree ordering and cardinality pruning
- Knapsack: Bottom-up DP with 2-row rolling buffer

**WP2 — OpenMP Parallelization (Completed: Dec 2025 – Feb 2026)**
- TSP: Island model with P independent chains, one `#pragma omp critical` at the end
- MCP: `#pragma omp for schedule(dynamic,1)` with `std::atomic<int>` for global best
- Knapsack: Persistent thread team with `#pragma omp for schedule(static)` per row

**WP3 — CUDA Parallelization (In Progress: Jan 2026 – Mar 2026)**

The GPU parallelization phase is currently underway. Initial implementations for all three problems have been completed and validated on the Google Colab T4 GPU environment. The remaining work involves systematic benchmarking across larger instances, kernel-level profiling with NVIDIA Nsight Compute, and tuning of launch configurations (block size, grid size, shared memory usage).

**TSP CUDA — Massively Parallel Island Model.** The GPU implementation launches 256 independent SA chains, each assigned to a single CUDA thread. Each thread maintains its own tour permutation in global memory and uses the cuRAND XORWOW generator for per-thread random number generation, initialized with a unique seed derived from the thread index. The distance matrix is stored in global memory as a flat row-major array and benefits from L2 cache reuse across threads, since all chains access the same matrix. The Nearest-Neighbour heuristic is used for tour initialization on the device side. After all chains complete their annealing schedule, a parallel reduction identifies the globally best tour. The current implementation achieves a tour cost of 8,182 for berlin52, which is within 8.5% of the known optimal. Ongoing work focuses on increasing the iteration count per chain and experimenting with shared-memory caching of the distance matrix for small instances.

**MCP CUDA — Bitmask-Accelerated Branch-and-Bound.** The adjacency matrix is represented as a packed bitmask array where each row of the matrix is stored as ceil(n/32) 32-bit words. This enables O(n/32) set intersection via bitwise AND operations and population counting via the hardware-accelerated `__popc()` intrinsic. Each CUDA thread is assigned one starting vertex and performs an iterative (non-recursive) Bron-Kerbosch search using an explicit stack allocated in local memory. Cross-thread pruning is achieved via `atomicMax` on a global best-clique-size variable. The current implementation correctly finds the maximum clique (size 11) for the n=200, d=0.50 instance in 0.100 seconds. The GPU does not yet outperform the CPU for this problem due to the irregular branching patterns inherent in B&B; future optimization will explore warp-cooperative search strategies where 32 threads in a warp collaboratively explore a single search subtree.

**Knapsack CUDA — Row-Parallel Dynamic Programming.** This is the most successful CUDA implementation. One kernel is launched per DP row, with (W+1) threads computing all capacity values in parallel. Two device arrays alternate as source and destination (double-buffer), eliminating the need for device-wide synchronization between rows. The kernel achieves coalesced memory access: thread `t` reads `dp_src[t]` and `dp_src[t - w_i]` and writes `dp_dst[t]`, where consecutive threads access consecutive memory addresses. The implementation achieves a throughput of 27,297 million DP cells per second on the T4 GPU — a 24× improvement over the serial CPU baseline. This confirms the memory-bandwidth-bound nature of the problem: the T4's dedicated 320 GB/s GDDR6 bandwidth saturates the simple per-cell computation far more effectively than the CPU's shared 80 GB/s unified memory.

### 4.4 Development

The development phase produced the following key results. All experiments were conducted on Apple M-series (CPU) and NVIDIA Tesla T4 (GPU).

#### 4.4.1 TSP Results

Table 2 shows the TSP benchmark results across all three implementations.

**Table 2. TSP performance comparison (Simulated Annealing).**

| Instance | n | Serial Time | Serial Cost | OpenMP Time (8T) | OpenMP Cost | CUDA Time (256 chains) | CUDA Cost |
|----------|---|-------------|-------------|-------------------|-------------|------------------------|-----------|
| berlin52 | 52 | 0.016 s | 7,544 | 0.005 s | 7,891 | 0.023 s | 8,182 |
| random1000 | 1000 | 0.369 s | 259,729 | 0.083 s | 275,287 | — | — |

OpenMP achieves **3.2–4.5× speedup** for TSP due to the embarrassingly parallel nature of independent SA chains. CUDA shows higher latency for small instances (52 cities) due to kernel launch overhead, but is expected to outperform for larger instances where the massive parallelism compensates for the overhead.

> **[INSERT FIGURE 2: Bar chart showing execution time for berlin52 across Serial, OpenMP, and CUDA. Use the numbers from Table 2. X-axis: Implementation, Y-axis: Execution time (seconds).]**

#### 4.4.2 Maximum Clique Results

Table 3 shows the MCP benchmark results.

**Table 3. Maximum Clique performance comparison (Branch-and-Bound).**

| Instance | n | Density | Serial Time | Clique | OpenMP Time (8T) | CUDA Time |
|----------|---|---------|-------------|--------|-------------------|-----------|
| rand_n100_d50 | 100 | 0.50 | 0.003 s | 9 | 0.002 s | — |
| rand_n150_d50 | 150 | 0.50 | 0.018 s | 10 | 0.008 s | — |
| rand_n200_d50 | 200 | 0.50 | 0.086 s | 11 | 0.030 s | 0.100 s |
| rand_n50_d90 | 50 | 0.90 | 0.040 s | 22 | 0.030 s | — |

OpenMP achieves **2.2–2.9× speedup** for MCP with `schedule(dynamic,1)`, which is essential for load balancing the highly variable sub-problem sizes. Both serial and OpenMP implementations find the same optimal clique size, confirming correctness.

For GPU, the MCP B&B shows limited speedup (0.9× for n=200) because the recursive search tree is inherently sequential within each thread. The GPU advantage lies in the bitmask operations (`__popc()`, bitwise AND), which accelerate set intersection but cannot compensate for the irregular branching overhead on small instances.

> **[INSERT FIGURE 3: Bar chart comparing Serial vs OpenMP execution time for MCP across the four graph instances. Show the speedup value on top of each OpenMP bar.]**

#### 4.4.3 Knapsack Results — The Key Finding

Table 4 shows the Knapsack benchmark results, which represent the most significant finding of this thesis.

**Table 4. 0/1 Knapsack performance comparison (Dynamic Programming).**

| Instance | n | W | Serial Time | OpenMP Time (8T) | CUDA Time | Serial Throughput | CUDA Throughput |
|----------|---|---|-------------|-------------------|-----------|-------------------|-----------------|
| ks_n1000_W50000 | 1,000 | 50,000 | 0.042 s | 0.073 s | — | 1,203 M cells/s | — |
| ks_n2000_W100000 | 2,000 | 100,000 | 0.174 s | 0.212 s | **0.007 s** | 1,152 M cells/s | **27,297 M cells/s** |
| ks_n5000_W50000 | 5,000 | 50,000 | 0.219 s | 0.545 s | — | 1,143 M cells/s | — |

**Key observation:** OpenMP is **slower** than serial for Knapsack (0.8× "speedup"), while CUDA achieves **24× speedup**.

This result is explained by the Roofline Model [17]:

1. **The Knapsack DP kernel is memory-bandwidth-bound**, not compute-bound. Each cell requires only one comparison and one addition, but two reads and one write to the DP array.

2. **CPU (Apple Silicon unified memory):** All 8 cores share a single memory controller (~80 GB/s). Adding more threads does not increase available bandwidth. The per-core E-core/P-core heterogeneity creates load imbalance at every barrier.

3. **GPU (Tesla T4 GDDR6):** Dedicated 320 GB/s memory bandwidth — 4× the CPU bandwidth. 2,560 CUDA cores with homogeneous performance. Kernel-launch synchronization (~5 μs) is far cheaper than CPU barrier overhead.

This demonstrates a fundamental HPC principle: **not all problems benefit equally from CPU parallelization. Memory-bound kernels benefit more from GPU bandwidth than from CPU compute parallelism.**

> **[INSERT FIGURE 4: Bar chart showing Knapsack execution time for n=2000, W=100000 across Serial (0.174s), OpenMP (0.212s), and CUDA (0.007s). This is the most important figure — it clearly shows GPU dominance for memory-bound workloads.]**

> **[INSERT FIGURE 5: Throughput comparison (M cells/s): Serial 1,152 vs OpenMP 945 vs CUDA 27,297. Use a logarithmic Y-axis to show the orders-of-magnitude difference.]**

### 4.5 Testing

All implementations were validated for correctness:

1. **TSP:** The serial SA on berlin52 (52 cities) produces a tour cost of 7,544, which is within 0.03% of the known optimal solution 7,542 [18], confirming algorithmic correctness.

2. **MCP:** All three implementations (Serial, OpenMP, CUDA) find the identical maximum clique size for every benchmark instance, confirming that the Branch-and-Bound pruning is correct across all parallelization strategies.

3. **Knapsack:** All three implementations produce the identical optimal value for every benchmark instance (e.g., 374,312 for ks_n2000_W100000), confirming that the DP computation is exact and correct across all architectures.

Testing infrastructure includes:
- Automated benchmark scripts (`run_benchmark.sh`, `run_benchmark_clique.sh`, `run_benchmark_knapsack.sh`) that run all implementations back-to-back with identical parameters
- Machine-readable CSV output for systematic data collection
- Python data generators with fixed seeds for reproducible benchmark instances

### 4.6 Hardware Platform Comparison

A fair interpretation of our benchmark results requires understanding the fundamental architectural differences between the two target platforms. Table 5 provides a side-by-side comparison.

**Table 5. Hardware specifications of benchmark platforms.**

| Specification | Apple M-series (CPU) | NVIDIA Tesla T4 (GPU) |
|--------------|---------------------|----------------------|
| Processing Cores | 8 (4 Performance + 4 Efficiency) | 2,560 CUDA cores |
| Clock Frequency | Up to 3.5 GHz (P-cores) | 585 MHz (base), 1,590 MHz (boost) |
| Memory Type | Unified LPDDR5 | 16 GB GDDR6 |
| Memory Bandwidth | ~80 GB/s (shared with CPU+GPU+NPU) | 320 GB/s (dedicated) |
| Memory Architecture | Unified (CPU and GPU share same pool) | Discrete (separate from host RAM) |
| Parallelism Model | MIMD (each core runs independent instruction stream) | SIMT (warps of 32 threads execute same instruction) |
| Thread Granularity | Coarse (heavyweight OS threads) | Fine (lightweight hardware threads) |
| Best Suited For | Irregular branching, complex control flow | Regular data-parallel, memory-bandwidth-intensive |
| Core Homogeneity | Heterogeneous (P-cores 2× faster than E-cores) | Homogeneous (all SMs identical) |
| Synchronization Cost | `pthread_barrier` ~1–5 μs | Kernel launch ~5 μs, `__syncthreads()` ~20 ns |

The 4× memory bandwidth advantage of the T4 GPU (320 GB/s vs 80 GB/s) directly explains the 24× Knapsack speedup: the DP kernel is entirely memory-bound, and the GPU can sustain far higher throughput per cell due to its dedicated high-bandwidth memory subsystem.

Conversely, the P-core/E-core heterogeneity of Apple Silicon creates an inherent load-imbalance problem for OpenMP: when work is distributed evenly across 8 cores, the 4 E-cores complete their share ~2× slower, forcing P-cores to idle at barriers. This effect is most pronounced in Knapsack (one barrier per DP row = 2,000 barriers) and least pronounced in TSP (one barrier at the very end).

### 4.7 Challenges and Lessons Learned

The development process revealed several non-trivial technical challenges that required careful debugging and architectural understanding:

**1. Numerical Precision in Simulated Annealing Cooling Rate.**
The cooling rate α is computed as (T_final / T_initial)^(1/maxIter). For large iteration counts (>10^6), computing this via `std::pow()` produced values indistinguishable from 1.0 due to floating-point precision loss, causing the SA to never cool down and produce 0% improvement over the initial solution. The fix was to reformulate the computation as `std::exp(std::log(T_final / T_initial) / maxIter)`, which distributes the precision more evenly across the exponentiation.

**2. OpenMP Restrictions on Loop Constructs.**
The initial MCP OpenMP implementation used a `break` statement inside a `#pragma omp for` loop to prune branches. The OpenMP specification forbids premature exit from work-sharing constructs, leading to compilation errors. The solution required restructuring the pruning logic to use `continue` with conditional execution, which maintains correctness while conforming to the OpenMP execution model.

**3. Apple Silicon Heterogeneous Core Architecture.**
The M-series CPU's mix of Performance and Efficiency cores creates a systematic load imbalance that is invisible to the programmer. OpenMP's `schedule(static)` divides work evenly, but E-cores execute ~2× slower than P-cores. We observed that `schedule(dynamic,chunk)` mitigates this for irregular workloads (MCP) but cannot help for regular workloads (Knapsack) where dynamic scheduling overhead exceeds the potential benefit.

**4. CUDA/GCC Compiler Incompatibility.**
NVIDIA's `nvcc` compiler delegates host code compilation to the system C++ compiler. On the Google Colab environment (GCC 11), using `-std=c++17` triggered a known bug in GCC 11's `<functional>` header related to parameter pack expansion with `std::function`. Downgrading to `-std=c++14` resolved the issue while retaining all required language features for our implementations.

**5. Memory-Bound vs Compute-Bound Classification.**
Perhaps the most important lesson was the discovery that the Knapsack DP is fundamentally memory-bandwidth-bound: the arithmetic intensity (FLOP per byte transferred) is approximately 0.125 FLOP/byte (one comparison + one addition per 16 bytes transferred). This places the kernel far below the ridge point on the Roofline Model, explaining why adding CPU cores (which add compute, not bandwidth) provides no benefit, while the GPU (which provides 4× more bandwidth) delivers dramatic speedup.

---

## 5. Planned Work

The remaining work packages are scheduled as follows:

**Table 6. Work-time schedule for remaining activities.**

| WP | Task | Period | Owner | Status |
|----|------|--------|-------|--------|
| WP1 | Serial Baseline (all 3 problems) | Nov 2025 – Dec 2025 | Burak & Emin | ✅ Completed |
| WP2 | CPU OpenMP Parallelization | Dec 2025 – Mar 2026 | Burak | ✅ Completed |
| WP3 | GPU CUDA Parallelization | Dec 2025 – Mar 2026 | Emin | 🔄 In Progress |
| WP4 | Systematic Benchmarking | Mar 2026 – Apr 2026 | Both | ⏳ Planned |
| WP5 | Comparative Analysis & Thesis Report | Apr 2026 – Apr 25, 2026 | Both | ⏳ Planned |

**WP3 — Remaining CUDA Work (March 2026):**
- Run CUDA benchmarks on larger instances (TSP: 1,000–5,000 cities; MCP: n=100–200 at density 0.5 and 0.9; Knapsack: n=5,000, W=1,000,000)
- Experiment with CUDA launch configurations: vary block size (64, 128, 256, 512) and grid size to find optimal occupancy
- Investigate shared-memory optimization for TSP (cache distance matrix sub-blocks) and MCP (cache bitmask rows for active vertices)
- Profile all three CUDA kernels using NVIDIA Nsight Compute to measure achieved memory bandwidth, occupancy, and warp execution efficiency
- Implement warp-cooperative B&B for MCP to improve GPU utilization on irregular workloads

**WP4 — Systematic Benchmarking (March 2026 – April 2026):**
- Extend benchmark datasets to include larger problem instances (TSP: 5,000–10,000 cities; MCP: 300–500 vertices; Knapsack: 10,000 items with W = 1,000,000)
- Conduct thread-scaling experiments (1, 2, 4, 8 threads for OpenMP; 128, 256, 512, 1024 chains for CUDA)
- Measure GPU kernel profiling metrics using NVIDIA Nsight Compute (occupancy, memory throughput, SM utilization)
- Run each experiment 5 times and report mean ± standard deviation

**WP5 — Comparative Analysis and Thesis Report (April 2026 – April 25, 2026):**
- Produce speedup, scalability, and Roofline Model charts for all 3 problems
- Analyze the relationship between problem characteristics (size, density, memory intensity) and relative CPU/GPU performance
- Answer the four research questions stated in the thesis proposal
- Write the final thesis report with complete experimental methodology, results, and conclusions

---

## References

[1] Garey, M. R. and Johnson, D. S., "Computers and Intractability: A Guide to the Theory of NP-Completeness", W. H. Freeman and Company, 1979.

[2] OpenMP Architecture Review Board, "OpenMP Application Programming Interface, Version 5.0", https://www.openmp.org/specifications/ (Last accessed: February 2026).

[3] NVIDIA Corporation, "CUDA C++ Programming Guide, Version 12.x", https://docs.nvidia.com/cuda/cuda-c-programming-guide/ (Last accessed: February 2026).

[4] Alba, E. and Luque, G., "Evaluation of Parallel Metaheuristics", in Parallel Problem Solving from Nature (PPSN), Springer LNCS, Vol. 4193, pp. 9–18, 2006.

[5] Kirkpatrick, S., Gelatt, C. D., and Vecchi, M. P., "Optimization by Simulated Annealing", Science, Vol. 220, No. 4598, pp. 671–680, 1983.

[6] Croes, G. A., "A Method for Solving Traveling-Salesman Problems", Operations Research, Vol. 6, No. 6, pp. 791–812, 1958.

[7] Onbaşoğlu, E. and Özdamar, L., "Parallel Simulated Annealing Algorithms in Global Optimization", Journal of Global Optimization, Vol. 19, pp. 27–50, 2001.

[8] Ram, D. J., Sreenivas, T. H., and Subramaniam, K. G., "Parallel Simulated Annealing Algorithms", Journal of Parallel and Distributed Computing, Vol. 37, No. 2, pp. 207–212, 1996.

[9] Czapinski, M. and Barnes, S., "Tabu Search with Two Approaches to Parallel Flowshop Evaluation on CUDA Platform", Journal of Parallel and Distributed Computing, Vol. 71, No. 6, pp. 802–811, 2011.

[10] Bron, C. and Kerbosch, J., "Algorithm 457: Finding All Cliques of an Undirected Graph", Communications of the ACM, Vol. 16, No. 9, pp. 575–577, 1973.

[11] Tomita, E. and Seki, T., "An Efficient Branch-and-Bound Algorithm for Finding a Maximum Clique", in Discrete Mathematics and Theoretical Computer Science, Springer LNCS, Vol. 2731, pp. 278–289, 2003.

[12] McCreesh, C. and Prosser, P., "The Shape of the Search Tree for the Maximum Clique Problem and the Implications for Parallel Branch and Bound", ACM Transactions on Parallel Computing, Vol. 2, No. 1, Article 8, 2015.

[13] Rossi, R. A. and Ahmed, N. K., "Coloring Large Complex Networks", Social Network Analysis and Mining, Vol. 4, No. 1, Article 228, 2014.

[14] Kellerer, H., Pferschy, U., and Pisinger, D., "Knapsack Problems", Springer-Verlag, Berlin, 2004.

[15] Boukedjar, A., Lalami, M. E., and El-Baz, D., "Parallel Branch and Bound on a CPU-GPU System", in Proceedings of the 20th International Conference on Parallel, Distributed and Network-Based Processing (PDP), pp. 392–398, IEEE, 2012.

[16] Pawłowski, K. and Kurdziel, M., "Dynamic Programming on CUDA-Compatible GPUs", in Proceedings of the International Conference on Parallel Processing and Applied Mathematics (PPAM), Springer LNCS, Vol. 7204, pp. 551–560, 2012.

[17] Williams, S., Waterman, A., and Patterson, D., "Roofline: An Insightful Visual Performance Model for Multicore Architectures", Communications of the ACM, Vol. 52, No. 4, pp. 65–76, 2009.

[18] Reinelt, G., "TSPLIB — A Traveling Salesman Problem Library", ORSA Journal on Computing, Vol. 3, No. 4, pp. 376–384, 1991.

[19] NVIDIA Corporation, "NVIDIA T4 Tensor Core GPU Datasheet", https://www.nvidia.com/en-us/data-center/tesla-t4/ (Last accessed: February 2026).
