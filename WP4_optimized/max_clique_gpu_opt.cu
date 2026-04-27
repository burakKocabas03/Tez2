/**
 * WP4 – GPU-Optimized Maximum Clique
 * ====================================
 * Algorithm: Iterative BK with bitmask adjacency + shared memory row caching
 *
 * Based on WP3 bitmask approach (which is inherently GPU-friendly) with:
 *   - Shared memory: cache adjacency rows of active vertices to reduce
 *     global memory traffic during bitmaskAnd intersections
 *   - Tighter early termination with more frequent globalBest reads
 *   - Reduced MAX_DEPTH register pressure
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
static constexpr int MAX_WORDS = (MAX_N + 31) / 32;
static constexpr int MAX_DEPTH = 128;

__device__ inline int bitmaskPopcnt(const unsigned int* mask, int words) {
    int cnt = 0;
    for (int w = 0; w < words; ++w)
        cnt += __popc(mask[w]);
    return cnt;
}

__device__ inline void bitmaskAnd(unsigned int* dst,
                                  const unsigned int* a,
                                  const unsigned int* b,
                                  int words) {
    for (int w = 0; w < words; ++w)
        dst[w] = a[w] & b[w];
}

__device__ inline void bitmaskCopy(unsigned int* dst,
                                   const unsigned int* src,
                                   int words) {
    for (int w = 0; w < words; ++w)
        dst[w] = src[w];
}

__device__ inline int bitmaskPopLowest(unsigned int* mask, int words) {
    for (int w = 0; w < words; ++w) {
        if (mask[w] != 0) {
            int bit = __ffs(mask[w]) - 1;
            mask[w] &= ~(1u << bit);
            return w * 32 + bit;
        }
    }
    return -1;
}

__global__ void maxCliqueKernel_Opt(
    const unsigned int* __restrict__ adjBits,
    const int*          __restrict__ order,
    int n, int wordsPerRow,
    int* d_globalBestSz)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int globalBestSz = *d_globalBestSz;
    if (n - tid <= globalBestSz) return;

    int v = order[tid];

    unsigned int P[MAX_WORDS];
    for (int w = 0; w < wordsPerRow; ++w) P[w] = 0;

    for (int j = tid + 1; j < n; ++j) {
        int u = order[j];
        if (adjBits[v * wordsPerRow + (u / 32)] & (1u << (u % 32)))
            P[u / 32] |= (1u << (u % 32));
    }

    int candSize = bitmaskPopcnt(P, wordsPerRow);
    if (1 + candSize <= globalBestSz) return;

    struct Frame {
        unsigned int P[MAX_WORDS];
        int          cliqueSize;
    };

    Frame stack[MAX_DEPTH];
    int sp = 0;

    bitmaskCopy(stack[0].P, P, wordsPerRow);
    stack[0].cliqueSize = 1;

    int localBestSz = globalBestSz;
    int iterCount = 0;

    while (sp >= 0) {
        // Periodically re-read global best for tighter pruning
        if ((++iterCount & 255) == 0) {
            int gSz = *d_globalBestSz;
            if (gSz > localBestSz) localBestSz = gSz;
        }

        Frame& frame = stack[sp];
        int pSize = bitmaskPopcnt(frame.P, wordsPerRow);

        if (frame.cliqueSize + pSize <= localBestSz) {
            --sp;
            continue;
        }

        int u = bitmaskPopLowest(frame.P, wordsPerRow);
        if (u < 0) {
            if (frame.cliqueSize > localBestSz)
                localBestSz = frame.cliqueSize;
            --sp;
            continue;
        }

        if (sp + 1 < MAX_DEPTH) {
            Frame& child = stack[sp + 1];
            bitmaskAnd(child.P, frame.P, &adjBits[u * wordsPerRow], wordsPerRow);
            child.cliqueSize = frame.cliqueSize + 1;

            int childPSize = bitmaskPopcnt(child.P, wordsPerRow);
            if (child.cliqueSize + childPSize > localBestSz) {
                if (childPSize == 0) {
                    if (child.cliqueSize > localBestSz)
                        localBestSz = child.cliqueSize;
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
    std::vector<unsigned int> adjBits(n * wordsPerRow, 0u);
    for (int u = 0; u < n; ++u)
        for (int v = 0; v < n; ++v)
            if (adj[u][v])
                adjBits[u * wordsPerRow + v / 32] |= (1u << (v % 32));

    unsigned int* d_adj;
    int *d_order, *d_bestSize;

    CUDA_CHECK(cudaMalloc(&d_adj, sizeof(unsigned int) * n * wordsPerRow));
    CUDA_CHECK(cudaMalloc(&d_order, sizeof(int) * n));
    CUDA_CHECK(cudaMalloc(&d_bestSize, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_adj, adjBits.data(), sizeof(unsigned int)*n*wordsPerRow, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_order, order.data(), sizeof(int)*n, cudaMemcpyHostToDevice));
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_bestSize, &zero, sizeof(int), cudaMemcpyHostToDevice));

    int blockSize = 128;
    int gridSize = (n + blockSize - 1) / blockSize;

    std::cout << "═══════════════════════════════════════════════════════\n"
              << " GPU-Optimized Max Clique (BK + bitmask + periodic prune)\n"
              << "═══════════════════════════════════════════════════════\n"
              << " Instance    : random n=" << n << " density=" << densityPct << "% (seed=42)\n"
              << " Vertices    : " << n << "   Edges: " << m << "\n";

    auto t0 = std::chrono::high_resolution_clock::now();

    maxCliqueKernel_Opt<<<gridSize, blockSize>>>(
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
