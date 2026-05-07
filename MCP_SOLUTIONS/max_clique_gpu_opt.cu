/**
 * WP4 – GPU-Optimized Maximum Clique
 * ====================================
 * Algorithm: Iterative BK with 64-bit bitmask + __ldg + tight pruning
 *
 * Optimizations over WP3:
 *   1. 64-bit bitmask (unsigned long long): half the words per row,
 *      half the loop iterations in AND/popcnt/popLowest
 *   2. __ldg() for adjacency reads: routes through read-only/texture cache
 *   3. Tighter pruning: re-check global best after finding a new local best
 *   4. __popcll() and __ffsll(): native 64-bit GPU instructions
 *
 * Build: nvcc -O3 -std=c++14 -arch=sm_75 -o max_clique_gpu_opt max_clique_gpu_opt.cu
 * Run:   ./max_clique_gpu_opt <num_vertices> <density_percent>
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

static constexpr int MAX_N     = 1024;
static constexpr int MAX_WORDS = (MAX_N + 63) / 64;
static constexpr int MAX_DEPTH = 128;

typedef unsigned long long uint64;

__device__ inline int bitmaskPopcnt(const uint64* mask, int words) {
    int cnt = 0;
    for (int w = 0; w < words; ++w)
        cnt += __popcll(mask[w]);
    return cnt;
}

__device__ inline void bitmaskAnd_ldg(uint64* dst,
                                      const uint64* a,
                                      const uint64* __restrict__ b,
                                      int words) {
    for (int w = 0; w < words; ++w)
        dst[w] = a[w] & __ldg(&b[w]);
}

__device__ inline void bitmaskCopy(uint64* dst, const uint64* src, int words) {
    for (int w = 0; w < words; ++w)
        dst[w] = src[w];
}

__device__ inline int bitmaskPopLowest(uint64* mask, int words) {
    for (int w = 0; w < words; ++w) {
        if (mask[w] != 0ULL) {
            int bit = __ffsll(mask[w]) - 1;
            mask[w] &= ~(1ULL << bit);
            return w * 64 + bit;
        }
    }
    return -1;
}

__global__ void maxCliqueKernel(
    const uint64* __restrict__ adjBits,
    const int*    __restrict__ order,
    int n, int wordsPerRow,
    int* d_globalBestSz)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int globalBestSz = __ldg(d_globalBestSz);
    if (n - tid <= globalBestSz) return;

    int v = __ldg(&order[tid]);

    uint64 P[MAX_WORDS];
    for (int w = 0; w < wordsPerRow; ++w) P[w] = 0ULL;

    for (int j = tid + 1; j < n; ++j) {
        int u = __ldg(&order[j]);
        if (__ldg(&adjBits[v * wordsPerRow + (u / 64)]) & (1ULL << (u % 64)))
            P[u / 64] |= (1ULL << (u % 64));
    }

    int candSize = bitmaskPopcnt(P, wordsPerRow);
    if (1 + candSize <= globalBestSz) return;

    struct Frame {
        uint64 P[MAX_WORDS];
        int    cliqueSize;
    };

    Frame stack[MAX_DEPTH];
    int sp = 0;

    bitmaskCopy(stack[0].P, P, wordsPerRow);
    stack[0].cliqueSize = 1;

    int localBestSz = globalBestSz;

    while (sp >= 0) {
        Frame& frame = stack[sp];
        int pSize = bitmaskPopcnt(frame.P, wordsPerRow);

        if (frame.cliqueSize + pSize <= localBestSz) {
            --sp;
            continue;
        }

        int u = bitmaskPopLowest(frame.P, wordsPerRow);
        if (u < 0) {
            if (frame.cliqueSize > localBestSz) {
                localBestSz = frame.cliqueSize;
                // Re-read global best after finding improvement
                int gSz = __ldg(d_globalBestSz);
                if (gSz > localBestSz) localBestSz = gSz;
            }
            --sp;
            continue;
        }

        if (sp + 1 < MAX_DEPTH) {
            Frame& child = stack[sp + 1];
            bitmaskAnd_ldg(child.P, frame.P, &adjBits[u * wordsPerRow], wordsPerRow);
            child.cliqueSize = frame.cliqueSize + 1;

            int childPSize = bitmaskPopcnt(child.P, wordsPerRow);
            if (child.cliqueSize + childPSize > localBestSz) {
                if (childPSize == 0) {
                    if (child.cliqueSize > localBestSz) {
                        localBestSz = child.cliqueSize;
                        int gSz = __ldg(d_globalBestSz);
                        if (gSz > localBestSz) localBestSz = gSz;
                    }
                } else {
                    ++sp;
                    continue;
                }
            }
        }
    }

    if (localBestSz > globalBestSz)
        atomicMax(d_globalBestSz, localBestSz);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <num_vertices> <density_percent>\n";
        return 1;
    }

    int n = std::stoi(argv[1]);
    int densityPct = std::stoi(argv[2]);

    if (n > MAX_N) {
        std::cerr << "n=" << n << " exceeds MAX_N=" << MAX_N << "\n";
        return 1;
    }

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

    int wordsPerRow = (n + 63) / 64;
    std::vector<uint64> adjBits(n * wordsPerRow, 0ULL);
    for (int u = 0; u < n; ++u)
        for (int v = 0; v < n; ++v)
            if (adj[u][v])
                adjBits[u * wordsPerRow + v / 64] |= (1ULL << (v % 64));

    uint64* d_adj;
    int *d_order, *d_bestSize;

    CUDA_CHECK(cudaMalloc(&d_adj, sizeof(uint64) * n * wordsPerRow));
    CUDA_CHECK(cudaMalloc(&d_order, sizeof(int) * n));
    CUDA_CHECK(cudaMalloc(&d_bestSize, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_adj, adjBits.data(), sizeof(uint64)*n*wordsPerRow, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_order, order.data(), sizeof(int)*n, cudaMemcpyHostToDevice));
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_bestSize, &zero, sizeof(int), cudaMemcpyHostToDevice));

    int blockSize = 128;
    int gridSize = (n + blockSize - 1) / blockSize;

    std::cout << "═══════════════════════════════════════════════════════\n"
              << " GPU-Optimized Max Clique (64-bit + __ldg + fixed prune)\n"
              << "═══════════════════════════════════════════════════════\n"
              << " Instance    : random n=" << n << " density=" << densityPct << "% (seed=42)\n"
              << " Vertices    : " << n << "   Edges: " << m << "\n";

    auto t0 = std::chrono::high_resolution_clock::now();

    maxCliqueKernel<<<gridSize, blockSize>>>(
        d_adj, d_order, n, wordsPerRow, d_bestSize);
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

    cudaFree(d_adj); cudaFree(d_order); cudaFree(d_bestSize);
    return 0;
}
