/**
 * WP3 – GPU Parallelization (CUDA)
 * ==================================
 * Problem   : 0/1 Knapsack Problem
 * Algorithm : Parallel Dynamic Programming on GPU
 *
 * Parallel Strategy
 * -----------------
 * The DP recurrence is:
 *   dp[i][c] = max(dp[i-1][c],  dp[i-1][c - w[i]] + v[i])
 *
 * Row i depends only on row i-1.  All W+1 cells of row i can be computed
 * independently → launch one CUDA kernel PER ROW with (W+1) threads.
 *
 * GPU advantage over CPU (OpenMP):
 * ---------------------------------
 *   1. MEMORY BANDWIDTH:  The Knapsack DP is memory-bandwidth-bound.
 *      GPU GDDR6/HBM provides 400–900 GB/s vs ~80 GB/s for Apple Silicon
 *      unified memory or ~50 GB/s for typical desktop DDR5.
 *      This is THE key advantage for this problem.
 *
 *   2. MASSIVE PARALLELISM:  For W = 100,000, a single kernel launch
 *      dispatches 100,001 threads.  The GPU can schedule thousands of
 *      these concurrently across SMs, fully saturating memory bandwidth.
 *
 *   3. NO BARRIER OVERHEAD:  Rows are separated by kernel launches (implicit
 *      global synchronisation via cudaDeviceSynchronize or sequential launches
 *      on the same stream).  CUDA kernel launch overhead is ~5 μs — far less
 *      than an OpenMP barrier on heterogeneous (P+E) cores.
 *
 *   4. HOMOGENEOUS CORES:  All CUDA cores run at the same speed.
 *      No P-core / E-core load imbalance.
 *
 * Implementation details
 * ----------------------
 *   - Double-buffer: two device arrays (d_prev, d_curr), swapped each row.
 *   - One kernel launch per item (n launches total).
 *   - Each CUDA thread computes dp[i][c] for one capacity value c.
 *   - Block size = 256 threads; grid size = ceil((W+1) / 256).
 *
 * Thesis   : A Comparison of NP-hard Problems through Parallel Algorithms
 *            Utilizing CPU and GPU-Based Solutions
 * Author   : Emin Özgür Elmalı
 * Advisor  : Prof. Dr. Hasan BULUT
 *
 * Build    : nvcc -O3 -std=c++17 -o knapsack_cuda knapsack_cuda.cu
 * Run      : ./knapsack_cuda <instance_file>
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <string>
#include <iomanip>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::cerr << "[CUDA ERROR] " << cudaGetErrorString(err)          \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";      \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
//  Kernel: compute one row of the DP table
// ---------------------------------------------------------------------------

/**
 * Each thread handles one capacity value c ∈ [0, W].
 *
 *   prev[c]  = dp[i-1][c]         — previous row (read-only)
 *   curr[c]  = dp[i][c]           — current row  (write-only)
 *
 *   curr[c] = max(prev[c], prev[c - wi] + vi)   if c >= wi
 *           = prev[c]                             otherwise
 *
 * No shared memory needed: prev[] reads hit L2 cache since adjacent threads
 * read adjacent addresses (coalesced access).
 */
__global__ void knapsackRowKernel(const long long* __restrict__ prev,
                                  long long*       __restrict__ curr,
                                  int              wi,
                                  int              vi,
                                  long long        W)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c > W) return;

    long long keep = prev[c];
    long long take = (c >= wi) ? prev[c - wi] + vi : 0LL;
    curr[c] = (keep > take) ? keep : take;
}

// ---------------------------------------------------------------------------
//  Host: instance reader
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
        std::cerr << "[ERROR] Cannot open: " << filename << "\n";
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
//  Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <instance_file>\n";
        return EXIT_FAILURE;
    }

    auto inst = readInstance(argv[1]);
    int       n = inst.n;
    long long W = inst.W;
    long long dpCells = (long long)n * (W + 1);

    std::cout << "═══════════════════════════════════════════════════════\n"
              << " WP3 – 0/1 Knapsack Problem (CUDA Parallel DP)\n"
              << "═══════════════════════════════════════════════════════\n"
              << " Instance    : " << argv[1] << "\n"
              << " Items (n)   : " << n << "\n"
              << " Capacity (W): " << W << "\n"
              << " DP cells    : " << dpCells << "\n"
              << "───────────────────────────────────────────────────────\n";

    // ── Device allocations ───────────────────────────────────────────────
    long long* d_prev;
    long long* d_curr;

    size_t rowBytes = sizeof(long long) * (W + 1);
    CUDA_CHECK(cudaMalloc(&d_prev, rowBytes));
    CUDA_CHECK(cudaMalloc(&d_curr, rowBytes));

    // Initialise both rows to zero
    CUDA_CHECK(cudaMemset(d_prev, 0, rowBytes));
    CUDA_CHECK(cudaMemset(d_curr, 0, rowBytes));

    // ── Kernel launch config ─────────────────────────────────────────────
    int blockSize = 256;
    int gridSize  = static_cast<int>((W + 1 + blockSize - 1) / blockSize);

    // ── Run DP: one kernel per item ──────────────────────────────────────
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n; ++i) {
        knapsackRowKernel<<<gridSize, blockSize>>>(
            d_prev, d_curr, inst.weight[i], inst.value[i], W);

        // Swap pointers (no data movement — just pointer swap on host)
        std::swap(d_prev, d_curr);
    }

    // Wait for all kernels to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    // ── Read result ──────────────────────────────────────────────────────
    // After n swaps, result is in d_prev[W]
    long long optVal;
    CUDA_CHECK(cudaMemcpy(&optVal, d_prev + W, sizeof(long long), cudaMemcpyDeviceToHost));

    double mpps = (dpCells / 1e6) / elapsed;

    std::cout << " Optimal value       : " << optVal << "\n"
              << " DP cells computed   : " << dpCells << "\n"
              << std::fixed << std::setprecision(2)
              << " Throughput          : " << mpps << " M cells/s\n"
              << std::setprecision(6)
              << " Execution time (GPU): " << elapsed << " s\n"
              << "═══════════════════════════════════════════════════════\n";

    std::cout << "CSV," << n << "," << W << ","
              << optVal << "," << elapsed << "," << dpCells << "\n";

    cudaFree(d_prev);
    cudaFree(d_curr);

    return EXIT_SUCCESS;
}
