/**
 * WP4 – GPU-Optimized 0/1 Knapsack
 * ==================================
 * Algorithm: Inter-row parallel DP with shared memory tiling
 *
 * Build: nvcc -O3 -std=c++14 -arch=sm_75 -o knapsack_gpu_opt knapsack_gpu_opt.cu
 * Run:   ./knapsack_gpu_opt <num_items> <capacity> [use_tiling]
 *        e.g. ./knapsack_gpu_opt 1000 50000 1
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

__global__ void knapsack_tiled_kernel(
    const long long* __restrict__ prev,
    long long* __restrict__ curr,
    int W, int wi, int vi)
{
    extern __shared__ long long sharedPrev[];

    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int tileStart = blockIdx.x * blockDim.x;
    int tileEnd = min(tileStart + (int)blockDim.x - 1, W);

    if (w <= W)
        sharedPrev[threadIdx.x] = prev[w];
    __syncthreads();

    if (w > W) return;

    long long val = sharedPrev[threadIdx.x];

    if (w >= wi) {
        int srcW = w - wi;
        long long srcVal;
        if (srcW >= tileStart && srcW <= tileEnd)
            srcVal = sharedPrev[srcW - tileStart];
        else
            srcVal = prev[srcW];

        curr[w] = max(val, srcVal + (long long)vi);
    } else {
        curr[w] = val;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <num_items> <capacity> [use_tiling]\n"
                  << "  e.g. " << argv[0] << " 1000 50000 1\n";
        return 1;
    }

    int n = std::stoi(argv[1]);
    int W = std::stoi(argv[2]);
    bool useTiling = (argc > 3) ? std::stoi(argv[3]) : 1;

    // Generate random knapsack instance
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> distW(1, std::max(1, W / 10));
    std::uniform_int_distribution<int> distV(10, 1000);

    std::vector<int> weights(n), values(n);
    for (int i = 0; i < n; ++i) {
        weights[i] = distW(rng);
        values[i] = distV(rng);
    }

    std::cout << "═══════════════════════════════════════════════════════\n"
              << " GPU-Optimized 0/1 Knapsack (Tiled DP)\n"
              << "═══════════════════════════════════════════════════════\n"
              << " Instance  : random n=" << n << " W=" << W << " (seed=42)\n"
              << " Items     : " << n << "   Capacity: " << W << "\n"
              << " Tiling    : " << (useTiling ? "ON" : "OFF") << "\n";

    long long *d_prev, *d_curr;
    CUDA_CHECK(cudaMalloc(&d_prev, sizeof(long long) * (W + 1)));
    CUDA_CHECK(cudaMalloc(&d_curr, sizeof(long long) * (W + 1)));
    CUDA_CHECK(cudaMemset(d_prev, 0, sizeof(long long) * (W + 1)));

    int blockSize = 256;
    int gridSize = (W + blockSize) / blockSize;
    size_t sharedMem = sizeof(long long) * blockSize;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n; ++i) {
        int wi = weights[i];
        int vi = values[i];

        // Tiling only helps when wi < blockSize (lookup stays within tile)
        // For large wi, the simple kernel is faster (no sync overhead)
        if (useTiling && wi < blockSize) {
            knapsack_tiled_kernel<<<gridSize, blockSize, sharedMem>>>(
                d_prev, d_curr, W, wi, vi);
        } else {
            knapsack_row_kernel<<<gridSize, blockSize>>>(d_prev, d_curr, W, wi, vi);
        }

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
              << std::setprecision(2)
              << " Throughput: " << throughput << " Mcells/s\n"
              << "═══════════════════════════════════════════════════════\n";

    std::cout << "CSV," << n << "," << W << ","
              << result << "," << elapsed << "," << throughput << "\n";

    cudaFree(d_prev);
    cudaFree(d_curr);
    return 0;
}
