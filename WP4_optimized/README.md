# WP4 – Architecture-Optimized Implementations

This folder contains CPU-optimal and GPU-optimal algorithm implementations
for each of the three NP-hard problems. Unlike WP1-WP3 (which use the
**same algorithm** across all platforms), these implementations choose
the **best algorithm for each specific architecture**.

## Philosophy

> "Fair comparison" ≠ "same code everywhere".
> True fairness means letting each platform shine with its best tool.

## Algorithms

| Problem | CPU-Optimal | GPU-Optimal |
|---------|-------------|-------------|
| **TSP** | SA + **Or-opt** (segment relocation) | SA + **shared memory** distance cache |
| **MCP** | BK + **greedy coloring** bound + **pivot** | Iterative BK + **bitmask** + `__popc()` |
| **Knapsack** | **Single-row reverse** DP (cache-friendly) | **Shared-memory tiled** DP |

## Why Different Algorithms?

### TSP
- **CPU**: Or-opt moves involve complex branching (segment extraction, reinsertion)
  that benefits from CPU branch prediction and large caches.
- **GPU**: Thousands of independent SA chains benefit from massive parallelism;
  shared memory eliminates distance matrix bottleneck for small instances.

### Maximum Clique
- **CPU**: Greedy coloring provides a much tighter upper bound than simple |P|,
  dramatically pruning the search tree. Sequential nature of coloring is ideal for CPU.
- **GPU**: Bitmask adjacency with hardware `__popc()` enables fast set intersection.
  Each thread independently explores a subtree using an iterative stack.

### Knapsack
- **CPU**: Single-row reverse DP halves memory usage and has perfect sequential
  access pattern for CPU prefetcher. No synchronization needed.
- **GPU**: Tiled DP with shared memory reduces global memory traffic. The
  capacity dimension provides natural parallelism (each w is independent per row).

## Build

```bash
# CPU versions (requires OpenMP)
make cpu

# GPU versions (requires CUDA toolkit)
make gpu
```

## Run

```bash
# TSP
./tsp_cpu_opt ../data/berlin52.tsp 1000000 1000.0 8
./tsp_gpu_opt ../data/berlin52.tsp 1000000 1000.0 2048

# Maximum Clique
./max_clique_cpu_opt ../data/random_n100_d050.clq 8
./max_clique_gpu_opt ../data/random_n100_d050.clq

# Knapsack
./knapsack_cpu_opt ../data/knapsack_n1000_W5000.txt 8
./knapsack_gpu_opt ../data/knapsack_n1000_W5000.txt
```
