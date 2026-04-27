/**
 * WP4 – GPU-Optimized Maximum Clique
 * ====================================
 * Algorithm: Iterative BK with bitmask adjacency + atomicMax pruning
 *
 * Build: nvcc -O3 -std=c++14 -arch=sm_75 -o max_clique_gpu_opt max_clique_gpu_opt.cu
 * Run:   ./max_clique_gpu_opt <num_vertices> <density_percent>
 *        e.g. ./max_clique_gpu_opt 100 50
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <numeric>
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

static constexpr int MAX_N = 512;
static constexpr int MAX_STACK = 64;

struct StackFrame {
    int cliqueSize;
    int candStart;
    int candCount;
};

__global__ void clique_kernel(
    const uint32_t* __restrict__ adjBits,
    const int* __restrict__ vertexOrder,
    int n, int wordsPerRow,
    int* globalBestSize,
    int* globalBestClique,
    int numStartVertices)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numStartVertices) return;

    int startIdx = tid;
    if (startIdx >= n) return;

    int localBest[MAX_N];
    int localBestSz = 0;
    int clique[MAX_N];
    int candidates[MAX_N];
    int newCandidates[MAX_N];

    int gBest = atomicAdd(globalBestSize, 0);
    if (n - startIdx <= gBest) return;

    int v0 = vertexOrder[startIdx];
    clique[0] = v0;
    int cliqueSize = 1;

    int candCount = 0;
    for (int j = startIdx + 1; j < n; ++j) {
        int u = vertexOrder[j];
        int word = u / 32, bit = u % 32;
        if ((adjBits[v0 * wordsPerRow + word] >> bit) & 1u)
            candidates[candCount++] = u;
    }

    StackFrame stack[MAX_STACK];
    int stackTop = 0;

    int ci = 0;
    while (true) {
        if (ci >= candCount || cliqueSize + (candCount - ci) <= localBestSz) {
            if (candCount == 0 || ci >= candCount) {
                if (cliqueSize > localBestSz) {
                    localBestSz = cliqueSize;
                    for (int k = 0; k < cliqueSize; ++k) localBest[k] = clique[k];

                    int oldBest = atomicMax(globalBestSize, localBestSz);
                    if (localBestSz > oldBest) {
                        for (int k = 0; k < localBestSz; ++k)
                            globalBestClique[k] = localBest[k];
                    }
                }
            }

            if (stackTop == 0) break;
            --stackTop;
            cliqueSize = stack[stackTop].cliqueSize;
            ci = stack[stackTop].candStart;
            candCount = stack[stackTop].candCount;
            continue;
        }

        int gSz = atomicAdd(globalBestSize, 0);
        if (gSz > localBestSz) localBestSz = gSz;
        if (cliqueSize + (candCount - ci) <= localBestSz) {
            if (stackTop == 0) break;
            --stackTop;
            cliqueSize = stack[stackTop].cliqueSize;
            ci = stack[stackTop].candStart;
            candCount = stack[stackTop].candCount;
            continue;
        }

        int v = candidates[ci];
        ++ci;

        if (stackTop < MAX_STACK) {
            stack[stackTop].cliqueSize = cliqueSize;
            stack[stackTop].candStart = ci;
            stack[stackTop].candCount = candCount;
            ++stackTop;
        }

        int newCount = 0;
        for (int j = ci; j < candCount; ++j) {
            int u = candidates[j];
            int word = u / 32, bit = u % 32;
            if ((adjBits[v * wordsPerRow + word] >> bit) & 1u)
                newCandidates[newCount++] = u;
        }

        clique[cliqueSize] = v;
        ++cliqueSize;

        for (int j = 0; j < newCount; ++j) candidates[j] = newCandidates[j];
        candCount = newCount;
        ci = 0;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <num_vertices> <density_percent>\n"
                  << "  e.g. " << argv[0] << " 100 50\n";
        return 1;
    }

    int n = std::stoi(argv[1]);
    int densityPct = std::stoi(argv[2]);

    if (n > MAX_N) {
        std::cerr << "n=" << n << " exceeds MAX_N=" << MAX_N << "\n";
        return 1;
    }

    // Generate random graph
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> pctDist(0, 99);

    std::vector<std::vector<bool>> adj(n, std::vector<bool>(n, false));
    std::vector<int> deg(n, 0);
    int m = 0;
    for (int u = 0; u < n; ++u)
        for (int v = u + 1; v < n; ++v)
            if (pctDist(rng) < densityPct) {
                adj[u][v] = adj[v][u] = true;
                ++deg[u]; ++deg[v]; ++m;
            }

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return deg[a] > deg[b]; });

    int wordsPerRow = (n + 31) / 32;
    std::vector<uint32_t> adjBits(n * wordsPerRow, 0u);
    for (int u = 0; u < n; ++u)
        for (int v = 0; v < n; ++v)
            if (adj[u][v])
                adjBits[u * wordsPerRow + v / 32] |= (1u << (v % 32));

    uint32_t *d_adj;
    int *d_order, *d_bestSize, *d_bestClique;

    CUDA_CHECK(cudaMalloc(&d_adj, sizeof(uint32_t) * n * wordsPerRow));
    CUDA_CHECK(cudaMalloc(&d_order, sizeof(int) * n));
    CUDA_CHECK(cudaMalloc(&d_bestSize, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bestClique, sizeof(int) * n));

    CUDA_CHECK(cudaMemcpy(d_adj, adjBits.data(), sizeof(uint32_t)*n*wordsPerRow, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_order, order.data(), sizeof(int)*n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_bestSize, 0, sizeof(int)));

    int blockSize = 128;
    int gridSize = (n + blockSize - 1) / blockSize;

    std::cout << "═══════════════════════════════════════════════════════\n"
              << " GPU-Optimized Max Clique (Iterative BK + bitmask)\n"
              << "═══════════════════════════════════════════════════════\n"
              << " Instance    : random n=" << n << " density=" << densityPct << "% (seed=42)\n"
              << " Vertices    : " << n << "   Edges: " << m << "\n";

    auto t0 = std::chrono::high_resolution_clock::now();

    clique_kernel<<<gridSize, blockSize>>>(
        d_adj, d_order, n, wordsPerRow, d_bestSize, d_bestClique, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    int bestSize = 0;
    CUDA_CHECK(cudaMemcpy(&bestSize, d_bestSize, sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << " Clique size : " << bestSize << "\n"
              << std::setprecision(6)
              << " Time        : " << elapsed << " s\n"
              << "═══════════════════════════════════════════════════════\n";

    std::cout << "CSV," << n << "," << m << ","
              << bestSize << "," << elapsed << "\n";

    cudaFree(d_adj); cudaFree(d_order);
    cudaFree(d_bestSize); cudaFree(d_bestClique);
    return 0;
}
