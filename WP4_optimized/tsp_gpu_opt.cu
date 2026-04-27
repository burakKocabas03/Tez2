/**
 * WP4 – GPU-Optimized TSP
 * ========================
 * Algorithm: Massively Parallel SA with shared memory distance matrix caching
 *
 * Build: nvcc -O3 -std=c++14 -arch=sm_75 -o tsp_gpu_opt tsp_gpu_opt.cu
 * Run:   ./tsp_gpu_opt <num_cities> [max_iter] [init_temp] [num_chains]
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <random>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "[CUDA ERROR] " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

#define MAX_SHARED_BYTES 49152
#define MAX_SHARED_N 78  // sqrt(49152 / sizeof(double)) ≈ 78

__global__ void initRNG(curandState* states, int numChains, unsigned long long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numChains) return;
    curand_init(seed, tid, 0, &states[tid]);
}

__global__ void saKernel_Optimized(
    const double* __restrict__ distMat,
    int n, long long maxIter, double initTemp, double coolRate,
    curandState* states, double* bestCosts, int* bestTours, int numChains,
    bool useShared)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numChains) return;

    extern __shared__ double sharedDist[];

    if (useShared && n <= MAX_SHARED_N) {
        int totalElems = n * n;
        int threadsInBlock = blockDim.x;
        for (int idx = threadIdx.x; idx < totalElems; idx += threadsInBlock)
            sharedDist[idx] = distMat[idx];
        __syncthreads();
    }

    const double* dist = (useShared && n <= MAX_SHARED_N) ? sharedDist : distMat;

    curandState localState = states[tid];
    int* tour = bestTours + (long long)tid * n;

    int startCity = tid % n;
    for (int i = 0; i < n; ++i) tour[i] = i;
    tour[startCity] = 0;
    tour[0] = startCity;

    for (int step = 0; step < n - 1; ++step) {
        int curr = tour[step];
        double bestD = 1e30;
        int bestIdx = step + 1;
        for (int j = step + 1; j < n; ++j) {
            double d = dist[curr * n + tour[j]];
            if (d < bestD) { bestD = d; bestIdx = j; }
        }
        int tmp = tour[step + 1];
        tour[step + 1] = tour[bestIdx];
        tour[bestIdx] = tmp;
    }

    double cost = 0.0;
    for (int i = 0; i < n; ++i)
        cost += dist[tour[i] * n + tour[(i + 1) % n]];

    double bestCost = cost;
    double T = initTemp;

    for (long long iter = 0; iter < maxIter && T > 1e-10; ++iter) {
        int i = curand(&localState) % n;
        int j = curand(&localState) % n;
        if (i == j) continue;
        if (i > j) { int tmp = i; i = j; j = tmp; }
        if (j - i < 2 || (i == 0 && j == n - 1)) continue;

        int a = tour[i], b = tour[i+1], c = tour[j], d = tour[(j+1)%n];
        double delta = dist[a*n+c] + dist[b*n+d] - dist[a*n+b] - dist[c*n+d];

        if (delta < 0.0 || curand_uniform_double(&localState) < exp(-delta / T)) {
            int left = i + 1, right = j;
            while (left < right) {
                int tmp = tour[left]; tour[left] = tour[right]; tour[right] = tmp;
                ++left; --right;
            }
            cost += delta;
            if (cost < bestCost) bestCost = cost;
        }
        T *= coolRate;
    }

    bestCosts[tid] = bestCost;
    states[tid] = localState;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <num_cities> [max_iter] [init_temp] [num_chains]\n";
        return 1;
    }

    int n = std::stoi(argv[1]);
    long long totalIter = (argc > 2) ? std::stoll(argv[2]) : (long long)n * n * 100;
    double initTemp = (argc > 3) ? std::stod(argv[3]) : 1000.0;
    int numChains = (argc > 4) ? std::stoi(argv[4]) : 2048;

    // Generate random cities
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> coordDist(0.0, 1000.0);
    std::vector<double> cx(n), cy(n);
    for (int i = 0; i < n; ++i) { cx[i] = coordDist(rng); cy[i] = coordDist(rng); }

    long long itersPerChain = std::max(1LL, totalIter / numChains);
    double coolRate = std::exp(std::log(1e-9 / initTemp) / (double)itersPerChain);

    std::vector<double> hostDist(n * n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            double dx = cx[i] - cx[j], dy = cy[i] - cy[j];
            double d = std::sqrt(dx * dx + dy * dy);
            hostDist[i * n + j] = d;
            hostDist[j * n + i] = d;
        }

    // NN cost for reference
    double nnCost = 0.0;
    {
        std::vector<bool> vis(n, false);
        int curr = 0; vis[0] = true;
        for (int step = 1; step < n; ++step) {
            double best = 1e30; int next = -1;
            for (int j = 0; j < n; ++j)
                if (!vis[j] && hostDist[curr*n+j] < best) { best = hostDist[curr*n+j]; next = j; }
            nnCost += best; vis[next] = true; curr = next;
        }
        nnCost += hostDist[curr * n + 0];
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "═══════════════════════════════════════════════════════\n"
              << " GPU-Optimized TSP (SA + shared mem, " << numChains << " chains)\n"
              << "═══════════════════════════════════════════════════════\n"
              << " Instance       : random " << n << " cities (seed=42)\n"
              << " Iters/chain    : " << itersPerChain << "\n"
              << " NN cost        : " << nnCost << "\n";

    double *d_dist, *d_bestCosts;
    curandState *d_states;
    int *d_bestTours;

    CUDA_CHECK(cudaMalloc(&d_dist, sizeof(double) * n * n));
    CUDA_CHECK(cudaMalloc(&d_states, sizeof(curandState) * numChains));
    CUDA_CHECK(cudaMalloc(&d_bestCosts, sizeof(double) * numChains));
    CUDA_CHECK(cudaMalloc(&d_bestTours, sizeof(int) * (long long)numChains * n));
    CUDA_CHECK(cudaMemcpy(d_dist, hostDist.data(), sizeof(double)*n*n, cudaMemcpyHostToDevice));

    int blockSize = 128;
    int gridSize = (numChains + blockSize - 1) / blockSize;

    initRNG<<<gridSize, blockSize>>>(d_states, numChains, 42ULL);
    CUDA_CHECK(cudaGetLastError());

    bool useShared = (n <= MAX_SHARED_N);
    size_t sharedMem = useShared ? sizeof(double) * n * n : 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    saKernel_Optimized<<<gridSize, blockSize, sharedMem>>>(
        d_dist, n, itersPerChain, initTemp, coolRate,
        d_states, d_bestCosts, d_bestTours, numChains, useShared);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::vector<double> hostBestCosts(numChains);
    CUDA_CHECK(cudaMemcpy(hostBestCosts.data(), d_bestCosts, sizeof(double)*numChains, cudaMemcpyDeviceToHost));

    double bestCost = hostBestCosts[0];
    int bestChain = 0;
    for (int i = 1; i < numChains; ++i)
        if (hostBestCosts[i] < bestCost) { bestCost = hostBestCosts[i]; bestChain = i; }

    std::cout << " Best tour cost : " << bestCost << "\n"
              << " Best chain     : " << bestChain << "\n"
              << std::setprecision(6)
              << " Execution time : " << elapsed << " s\n"
              << (useShared ? " [Shared memory: ON]\n" : " [Shared memory: OFF (n>256)]\n")
              << "═══════════════════════════════════════════════════════\n";

    std::cout << "CSV," << n << "," << numChains << ","
              << std::setprecision(4) << bestCost << "," << elapsed << "\n";

    cudaFree(d_dist); cudaFree(d_states);
    cudaFree(d_bestCosts); cudaFree(d_bestTours);
    return 0;
}
