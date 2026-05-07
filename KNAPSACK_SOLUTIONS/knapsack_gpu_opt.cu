/**
 * WP4 – GPU-Optimized 0/1 Knapsack
 * ==================================
 * Algorithm: Row-parallel DP (same kernel as WP3)
 *
 * Finding: The simple row-parallel kernel IS the GPU-optimal approach.
 * Attempted optimizations (shared memory tiling) add overhead without
 * benefit because weight values (up to W/10) far exceed tile size (256),
 * causing most lookups to fall back to global memory anyway.
 *
 * The GPU advantage comes from:
 *   - Raw memory bandwidth (400-900 GB/s GDDR6/HBM vs 80 GB/s CPU)
 *   - Massive parallelism across capacity dimension
 *   - Coalesced access patterns → L2 cache efficiency
 *   - NOT from shared memory (problem is bandwidth-bound, not latency-bound)
 *
 * Build: nvcc -O3 -std=c++14 -arch=sm_75 -o knapsack_gpu_opt knapsack_gpu_opt.cu
 * Run:   ./knapsack_gpu_opt <num_items> <capacity>
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <random>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "[CUDA ERROR] " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void knapsack_row_kernel(
    const long long* __restrict__ prev,
    long long* __restrict__ curr,
    int W, int wi, int vi)
{
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w > W) return;

    if (w >= wi)
        curr[w] = max(prev[w], prev[w - wi] + (long long)vi);
    else
        curr[w] = prev[w];
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <num_items> <capacity>\n";
        return 1;
    }

    int n = std::stoi(argv[1]);
    int W = std::stoi(argv[2]);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> distW(1, std::max(1, W / 10));
    std::uniform_int_distribution<int> distV(10, 1000);

    std::vector<int> weights(n), values(n);
    for (int i = 0; i < n; ++i) {
        weights[i] = distW(rng);
        values[i] = distV(rng);
    }

    std::cout << "═══════════════════════════════════════════════════════\n"
              << " GPU-Optimized 0/1 Knapsack (Row-parallel DP)\n"
              << "═══════════════════════════════════════════════════════\n"
              << " Instance  : random n=" << n << " W=" << W << " (seed=42)\n"
              << " Items     : " << n << "   Capacity: " << W << "\n";

    long long *d_prev, *d_curr;
    CUDA_CHECK(cudaMalloc(&d_prev, sizeof(long long) * (W + 1)));
    CUDA_CHECK(cudaMalloc(&d_curr, sizeof(long long) * (W + 1)));
    CUDA_CHECK(cudaMemset(d_prev, 0, sizeof(long long) * (W + 1)));

    int blockSize = 256;
    int gridSize = (W + blockSize) / blockSize;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n; ++i) {
        knapsack_row_kernel<<<gridSize, blockSize>>>(
            d_prev, d_curr, W, weights[i], values[i]);
        std::swap(d_prev, d_curr);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    long long result;
    CUDA_CHECK(cudaMemcpy(&result, d_prev + W, sizeof(long long), cudaMemcpyDeviceToHost));

    double cells = (double)n * (double)W;
    double throughput = cells / elapsed / 1e6;

    std::cout << " Optimal   : " << result << "\n"
              << std::setprecision(6)
              << " Time      : " << elapsed << " s\n"
              << std::fixed << std::setprecision(2)
              << " Throughput: " << throughput << " Mcells/s\n"
              << "═══════════════════════════════════════════════════════\n";

    std::cout << "CSV," << n << "," << W << ","
              << result << "," << elapsed << "," << throughput << "\n";

    cudaFree(d_prev);
    cudaFree(d_curr);
    return 0;
}
