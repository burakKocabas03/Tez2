/**
 * WP3 – GPU Parallelization (CUDA)
 * ==================================
 * Problem   : Maximum Clique Problem (MCP)
 * Algorithm : Parallel Branch-and-Bound with Bitmask Operations
 *
 * Parallel Strategy
 * -----------------
 * The graph adjacency matrix is stored as a FLAT BITMASK on GPU global memory:
 *   adj_bits[v * words_per_row + w] = 32-bit word covering vertices [w*32 .. w*32+31]
 *
 * Candidate set intersection (P ∩ N(v)) becomes a single bitwise AND per word
 * — O(n/32) instead of O(n) for the adjacency-list version.
 *
 * Top-level parallelism:
 *   - n starting vertices → n independent sub-problems (same as WP1/WP2).
 *   - Each sub-problem assigned to one CUDA thread.
 *   - Within each thread, an ITERATIVE B&B (explicit stack, no recursion)
 *     explores the sub-tree using bitmask operations.
 *   - atomicMax updates the global best clique size for cross-thread pruning.
 *
 * Why this suits GPU:
 *   1. Thousands of sub-problems run concurrently (vs 8 on CPU).
 *   2. Bitmask AND + __popc() map to native GPU instructions (LOP3, POPC).
 *   3. Adjacency bitmask is read-only → GPU L2 cache is highly effective.
 *   4. No inter-thread communication except one atomicMax per sub-problem.
 *
 * Limitation: n ≤ MAX_N due to fixed-size bitmask arrays in thread-local memory.
 * For n > 1024, increase MAX_N and MAX_WORDS (costs more registers/LMEM).
 *
 * Thesis   : A Comparison of NP-hard Problems through Parallel Algorithms
 *            Utilizing CPU and GPU-Based Solutions
 * Author   : Emin Özgür Elmalı
 * Advisor  : Prof. Dr. Hasan BULUT
 *
 * Build    : nvcc -O3 -std=c++17 -o max_clique_cuda max_clique_cuda.cu
 * Run      : ./max_clique_cuda <dimacs_graph_file>
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <numeric>

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
//  Constants
// ---------------------------------------------------------------------------

static constexpr int MAX_N     = 1024;
static constexpr int MAX_WORDS = (MAX_N + 31) / 32;   // 32 words for n ≤ 1024
static constexpr int MAX_DEPTH = 128;                  // max recursion depth

// ---------------------------------------------------------------------------
//  Device helpers: bitmask operations
// ---------------------------------------------------------------------------

__device__ inline int bitmaskPopcnt(const unsigned int* mask, int words) {
    int cnt = 0;
    for (int w = 0; w < words; ++w)
        cnt += __popc(mask[w]);
    return cnt;
}

__device__ inline void bitmaskAnd(unsigned int* dst,
                                  const unsigned int* a,
                                  const unsigned int* b,
                                  int words)
{
    for (int w = 0; w < words; ++w)
        dst[w] = a[w] & b[w];
}

__device__ inline void bitmaskCopy(unsigned int* dst,
                                   const unsigned int* src,
                                   int words)
{
    for (int w = 0; w < words; ++w)
        dst[w] = src[w];
}

/**
 * Find and clear the lowest set bit. Returns the bit position, or -1 if empty.
 */
__device__ inline int bitmaskPopLowest(unsigned int* mask, int words) {
    for (int w = 0; w < words; ++w) {
        if (mask[w] != 0) {
            int bit = __ffs(mask[w]) - 1;   // __ffs returns 1-indexed
            mask[w] &= ~(1u << bit);
            return w * 32 + bit;
        }
    }
    return -1;
}

__device__ inline void bitmaskClearBit(unsigned int* mask, int bit) {
    mask[bit / 32] &= ~(1u << (bit % 32));
}

// ---------------------------------------------------------------------------
//  Kernel: one CUDA thread per top-level starting vertex
// ---------------------------------------------------------------------------

__global__ void maxCliqueKernel(const unsigned int* __restrict__ adjBits,
                                const int*          __restrict__ order,
                                int                  n,
                                int                  wordsPerRow,
                                int*                 d_globalBestSz)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // Read current global best for pruning
    int globalBestSz = *d_globalBestSz;

    // Early termination: remaining vertices can't form a larger clique
    if (n - tid <= globalBestSz) return;

    int v = order[tid];

    // Build initial candidate set: order[tid+1 .. n-1] ∩ N(v)
    // as a bitmask over the ORIGINAL vertex IDs
    unsigned int P[MAX_WORDS];
    for (int w = 0; w < wordsPerRow; ++w) P[w] = 0;

    for (int j = tid + 1; j < n; ++j) {
        int u = order[j];
        // Check if v and u are adjacent
        if (adjBits[v * wordsPerRow + (u / 32)] & (1u << (u % 32)))
            P[u / 32] |= (1u << (u % 32));
    }

    int candSize = bitmaskPopcnt(P, wordsPerRow);
    if (1 + candSize <= globalBestSz) return;

    // ── Iterative BK with explicit stack ─────────────────────────────────
    // Stack frame: (candidate set, clique size at this level)
    struct Frame {
        unsigned int P[MAX_WORDS];
        int          cliqueSize;
    };

    // Allocate stack in local memory (will spill to LMEM/L1)
    Frame stack[MAX_DEPTH];
    int   sp = 0;

    // Push initial frame
    bitmaskCopy(stack[0].P, P, wordsPerRow);
    stack[0].cliqueSize = 1;  // just vertex v

    int localBestSz = globalBestSz;

    while (sp >= 0) {
        Frame& frame = stack[sp];

        // Count remaining candidates
        int pSize = bitmaskPopcnt(frame.P, wordsPerRow);

        // Pruning: clique + remaining candidates can't beat local best
        if (frame.cliqueSize + pSize <= localBestSz) {
            --sp;
            continue;
        }

        // Pick next candidate (lowest bit in P)
        int u = bitmaskPopLowest(frame.P, wordsPerRow);
        if (u < 0) {
            // No more candidates — this is a leaf
            if (frame.cliqueSize > localBestSz)
                localBestSz = frame.cliqueSize;
            --sp;
            continue;
        }

        // Branch: try adding vertex u to the clique
        // New candidates = current P ∩ N(u)
        if (sp + 1 < MAX_DEPTH) {
            Frame& child = stack[sp + 1];
            bitmaskAnd(child.P, frame.P, &adjBits[u * wordsPerRow], wordsPerRow);
            child.cliqueSize = frame.cliqueSize + 1;

            int childPSize = bitmaskPopcnt(child.P, wordsPerRow);
            if (child.cliqueSize + childPSize > localBestSz) {
                // Also check if leaf
                if (childPSize == 0) {
                    if (child.cliqueSize > localBestSz)
                        localBestSz = child.cliqueSize;
                } else {
                    ++sp;   // push child frame
                    continue;
                }
            }
        }
        // If we didn't push, the while loop will re-enter current frame
        // with u removed from P (bitmaskPopLowest already cleared it).
    }

    // Publish to global best via atomicMax
    if (localBestSz > globalBestSz)
        atomicMax(d_globalBestSz, localBestSz);
}

// ---------------------------------------------------------------------------
//  Host: DIMACS reader
// ---------------------------------------------------------------------------

struct Graph {
    int n = 0, m = 0;
    std::vector<std::vector<bool>> adj;
    std::vector<int> deg;

    explicit Graph(int n_)
        : n(n_), adj(n_, std::vector<bool>(n_, false)), deg(n_, 0) {}

    void addEdge(int u, int v) {
        if (!adj[u][v]) {
            adj[u][v] = adj[v][u] = true;
            ++deg[u]; ++deg[v]; ++m;
        }
    }
};

Graph readDIMACS(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open: " << filename << "\n";
        std::exit(EXIT_FAILURE);
    }
    int n = 0, m = 0;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == 'c') continue;
        if (line[0] == 'p') {
            std::istringstream iss(line);
            std::string t1, t2;
            iss >> t1 >> t2 >> n >> m;
            break;
        }
    }
    Graph G(n);
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == 'c') continue;
        if (line[0] == 'e') {
            std::istringstream iss(line);
            char ch; int u, v;
            iss >> ch >> u >> v;
            G.addEdge(u - 1, v - 1);
        }
    }
    return G;
}

// ---------------------------------------------------------------------------
//  Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dimacs_graph_file>\n";
        return EXIT_FAILURE;
    }

    Graph G = readDIMACS(argv[1]);
    int n = G.n;

    if (n > MAX_N) {
        std::cerr << "[ERROR] n=" << n << " exceeds MAX_N=" << MAX_N
                  << ". Increase MAX_N and recompile.\n";
        return EXIT_FAILURE;
    }

    // Degree-ordered vertex list (descending)
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return G.deg[a] > G.deg[b]; });

    // Build bitmask adjacency on host
    int wordsPerRow = (n + 31) / 32;
    std::vector<unsigned int> adjBits(n * wordsPerRow, 0u);
    for (int u = 0; u < n; ++u)
        for (int v = 0; v < n; ++v)
            if (G.adj[u][v])
                adjBits[u * wordsPerRow + v / 32] |= (1u << (v % 32));

    std::cout << "═══════════════════════════════════════════════════════\n"
              << " WP3 – Maximum Clique Problem (CUDA)\n"
              << "═══════════════════════════════════════════════════════\n"
              << " Instance    : " << argv[1] << "\n"
              << " Vertices    : " << n << "\n"
              << " Edges       : " << G.m << "\n"
              << " Density     : " << std::fixed << std::setprecision(4)
              << (n > 1 ? (2.0 * G.m) / ((double)n * (n - 1)) : 0.0) << "\n"
              << " Words/row   : " << wordsPerRow << "\n"
              << "───────────────────────────────────────────────────────\n";

    // ── Device allocations ───────────────────────────────────────────────
    unsigned int* d_adjBits;
    int*          d_order;
    int*          d_globalBestSz;

    CUDA_CHECK(cudaMalloc(&d_adjBits,      sizeof(unsigned int) * n * wordsPerRow));
    CUDA_CHECK(cudaMalloc(&d_order,        sizeof(int) * n));
    CUDA_CHECK(cudaMalloc(&d_globalBestSz, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_adjBits, adjBits.data(),
                          sizeof(unsigned int) * n * wordsPerRow, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_order, order.data(),
                          sizeof(int) * n, cudaMemcpyHostToDevice));
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_globalBestSz, &zero, sizeof(int), cudaMemcpyHostToDevice));

    // ── Launch kernel ────────────────────────────────────────────────────
    int blockSize = 128;
    int gridSize  = (n + blockSize - 1) / blockSize;

    auto t0 = std::chrono::high_resolution_clock::now();

    maxCliqueKernel<<<gridSize, blockSize>>>(
        d_adjBits, d_order, n, wordsPerRow, d_globalBestSz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    // ── Read result ──────────────────────────────────────────────────────
    int bestSz;
    CUDA_CHECK(cudaMemcpy(&bestSz, d_globalBestSz, sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << " Maximum clique size : " << bestSz << "\n"
              << std::setprecision(6)
              << " Execution time (GPU): " << elapsed << " s\n"
              << "═══════════════════════════════════════════════════════\n";

    std::cout << "CSV," << n << "," << G.m << "," << bestSz << "," << elapsed << "\n";

    cudaFree(d_adjBits);
    cudaFree(d_order);
    cudaFree(d_globalBestSz);

    return EXIT_SUCCESS;
}
