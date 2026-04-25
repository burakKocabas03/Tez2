/**
 * WP2 – CPU Parallelization (OpenMP)
 * ====================================
 * Problem   : 0/1 Knapsack Problem
 * Algorithm : Parallel Dynamic Programming
 *
 * Parallel Strategy: Row-Level Parallelism
 * -----------------------------------------
 * The DP recurrence is:
 *   dp[i][c] = max(dp[i-1][c],  dp[i-1][c - w[i]] + v[i])
 *
 * Key observation:
 *   - Row i depends ONLY on row i-1 (not on other elements of row i).
 *   - Therefore, all W+1 cells of row i can be computed in PARALLEL.
 *   - `#pragma omp parallel for` on the capacity loop gives perfect
 *     data parallelism — no race conditions, no synchronisation needed
 *     within a row.
 *   - Rows are processed sequentially (natural dependency chain), but
 *     each row's computation is fully parallel.
 *
 * Memory layout
 * -------------
 * Two full rows (prev and curr) stored as flat arrays of size W+1.
 * After each item, the rows are swapped.  Each thread works on a
 * contiguous chunk of the capacity dimension → excellent cache behaviour.
 *
 * Expected speedup and observed behaviour
 * -----------------------------------------
 * The Knapsack DP is MEMORY-BANDWIDTH-BOUND rather than compute-bound.
 *
 * On Apple Silicon (unified memory, 1 memory controller shared by all cores):
 *   - All threads compete for the same memory bandwidth — no scaling benefit.
 *   - Mixed P-core / E-core architecture causes load imbalance with static
 *     scheduling: fast P-cores finish their chunk and wait for slow E-cores.
 *   - Barrier overhead (~n barriers per run) further reduces throughput.
 *   → CPU speedup is limited or negative on this architecture.
 *
 * On a server-class x86 CPU with multi-channel DDR5 or NUMA memory:
 *   - Each NUMA node adds independent memory bandwidth → better scaling.
 *   → Moderate speedup (2–4×) with 8 homogeneous cores expected.
 *
 * On GPU (WP3, CUDA):
 *   - Dedicated GDDR6/HBM bandwidth (400–900 GB/s vs 80 GB/s for Apple M).
 *   - Thousands of homogeneous CUDA cores; no E/P core heterogeneity.
 *   - Warp-level synchronisation replaces expensive CPU barriers.
 *   → High speedup expected (8–20×).
 *
 * This result illustrates the Roofline Model: memory-bound kernels benefit
 * more from GPU (bandwidth) than CPU (FLOPS) parallelisation.
 *
 * Thesis   : A Comparison of NP-hard Problems through Parallel Algorithms
 *            Utilizing CPU and GPU-Based Solutions
 * Author   : Burak Kocabaş
 * Advisor  : Prof. Dr. Hasan BULUT
 *
 * Build    : see Makefile  (requires libomp on macOS: brew install libomp)
 * Run      : ./knapsack_openmp <instance_file> [num_threads]
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <string>
#include <iomanip>

#include <omp.h>

// ---------------------------------------------------------------------------
//  Instance
// ---------------------------------------------------------------------------

struct KnapsackInstance {
    int              n;
    long long        W;
    std::vector<int> weight;
    std::vector<int> value;
};

KnapsackInstance readInstance(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open file: " << filename << "\n";
        std::exit(EXIT_FAILURE);
    }

    KnapsackInstance inst;
    file >> inst.n >> inst.W;
    inst.weight.resize(inst.n);
    inst.value .resize(inst.n);

    for (int i = 0; i < inst.n; ++i)
        file >> inst.weight[i] >> inst.value[i];

    return inst;
}

// ---------------------------------------------------------------------------
//  Parallel DP solver
// ---------------------------------------------------------------------------

struct KnapsackResult {
    long long optimalValue;
    long long dpCells;
};

KnapsackResult solveKnapsack(const KnapsackInstance& inst, int numThreads) {
    const int       n = inst.n;
    const long long W = inst.W;

    // Double-buffer strategy: two flat arrays dp[0] and dp[1].
    // Row i reads from dp[i%2] and writes to dp[1 - i%2].
    // No pointer/vector swap needed — the parity alternates automatically.
    // This eliminates the `omp single` (and its barrier) that would otherwise
    // appear after every row, leaving only ONE implicit barrier per row
    // (at the end of `#pragma omp for`).
    //
    // The single persistent thread team (created by `#pragma omp parallel`)
    // is reused across all n rows, eliminating per-row fork/join overhead.
    //
    //  Memory layout after n items:
    //    result = dp[n % 2][W]
    std::vector<long long> dp[2];
    dp[0].assign(W + 1, 0LL);
    dp[1].assign(W + 1, 0LL);

    #pragma omp parallel num_threads(numThreads)
    {
        for (int i = 0; i < n; ++i) {
            const int src = i & 1;          // source row (already computed)
            const int dst = 1 - src;        // destination row (being computed)

            const int wi = inst.weight[i];
            const int vi = inst.value[i];

            const long long* const p = dp[src].data();
            long long* const       c = dp[dst].data();

            // Inner loop: fully parallel, zero data dependencies within a row.
            // Each thread gets a contiguous chunk of the capacity dimension
            // (static scheduling) for cache-friendly access.
            #pragma omp for schedule(static)
            for (long long cap = 0; cap <= W; ++cap) {
                const long long take = (cap >= wi) ? p[cap - wi] + vi : 0LL;
                c[cap] = (p[cap] > take) ? p[cap] : take;
            }
            // Implicit barrier at end of `omp for`:
            //   ensures all threads have written row i before any thread
            //   starts reading it as `src` in row i+1.
        }
    }

    return {dp[n & 1][W], static_cast<long long>(n) * (W + 1)};
}

// ---------------------------------------------------------------------------
//  Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <instance_file> [num_threads]\n"
                  << "  Format: first line 'N W', then N lines 'weight value'\n";
        return EXIT_FAILURE;
    }

    const auto inst = readInstance(argv[1]);
    int numThreads  = (argc > 2) ? std::stoi(argv[2]) : omp_get_max_threads();
    if (numThreads > omp_get_max_threads())
        numThreads = omp_get_max_threads();

    std::cout << "═══════════════════════════════════════════════════════\n";
    std::cout << " WP2 – 0/1 Knapsack Problem (OpenMP Parallel DP)\n";
    std::cout << "═══════════════════════════════════════════════════════\n";
    std::cout << " Instance    : " << argv[1]  << "\n";
    std::cout << " Items (n)   : " << inst.n   << "\n";
    std::cout << " Capacity (W): " << inst.W   << "\n";
    std::cout << " DP cells    : " << static_cast<long long>(inst.n) * (inst.W + 1) << "\n";
    std::cout << " Threads     : " << numThreads << "\n";
    std::cout << "───────────────────────────────────────────────────────\n";

    const auto t0  = std::chrono::high_resolution_clock::now();
    const auto res = solveKnapsack(inst, numThreads);
    const auto t1  = std::chrono::high_resolution_clock::now();

    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    const double mpps    = (res.dpCells / 1e6) / elapsed;

    std::cout << " Optimal value       : " << res.optimalValue << "\n";
    std::cout << " DP cells computed   : " << res.dpCells << "\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << " Throughput          : " << mpps << " M cells/s\n";
    std::cout << " Threads used        : " << numThreads << "\n";
    std::cout << std::setprecision(6);
    std::cout << " Execution time      : " << elapsed << " s\n";
    std::cout << "═══════════════════════════════════════════════════════\n";

    // CSV: n, W, threads, optimal_value, time_s, dp_cells
    std::cout << "CSV," << inst.n << "," << inst.W << "," << numThreads << ","
              << res.optimalValue << "," << elapsed << "," << res.dpCells << "\n";

    return EXIT_SUCCESS;
}
