/**
 * WP4 – GPU-Optimized TSP
 * ========================
 * Algorithm: Ant Colony Optimization (ACO)
 *
 * Why ACO is GPU-optimal (vs SA on CPU):
 *   - Each ant builds a tour INDEPENDENTLY → embarrassingly parallel
 *   - Tour construction: probabilistic city selection → GPU RNG excels
 *   - Pheromone matrix: shared read-mostly structure → GPU L2 cache
 *   - Pheromone update: atomicAdd on global matrix → native GPU op
 *   - No complex branching (unlike Or-opt) → no warp divergence
 *
 * Why SA is CPU-optimal (vs ACO on GPU):
 *   - SA benefits from deep sequential local search (2-opt, Or-opt)
 *   - Complex move evaluation with branch prediction
 *   - Few threads, long chains → CPU excels
 *
 * Parameters:
 *   - numAnts: number of ants per iteration (= GPU threads)
 *   - numIters: number of ACO iterations (pheromone updates)
 *   - alpha: pheromone importance
 *   - beta: heuristic (1/distance) importance
 *   - rho: evaporation rate
 *
 * Build: nvcc -O3 -std=c++14 -arch=sm_75 -o tsp_gpu_opt tsp_gpu_opt.cu
 * Run:   ./tsp_gpu_opt <num_cities> [num_iters] [num_ants]
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

__global__ void initRNG(curandState* states, int numAnts, unsigned long long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numAnts) return;
    curand_init(seed, tid, 0, &states[tid]);
}

/**
 * Each ant constructs a tour using probabilistic city selection
 * based on pheromone levels and heuristic information (1/distance).
 */
__global__ void antTourKernel(
    const double* __restrict__ distMat,
    const double* __restrict__ pheromone,
    int n, double alpha, double beta,
    curandState* states, int* tours, double* tourCosts,
    int numAnts)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numAnts) return;

    curandState localState = states[tid];
    int* tour = tours + (long long)tid * n;

    // Each ant starts from a random city
    int startCity = curand(&localState) % n;

    // visited bitmask (up to 1024 cities with 32 words)
    unsigned int visited[32];
    int words = (n + 31) / 32;
    for (int w = 0; w < words; ++w) visited[w] = 0;

    tour[0] = startCity;
    visited[startCity / 32] |= (1u << (startCity % 32));

    for (int step = 1; step < n; ++step) {
        int curr = tour[step - 1];

        // Calculate selection probabilities
        double totalProb = 0.0;
        double probs[1024];

        for (int j = 0; j < n; ++j) {
            if ((visited[j / 32] >> (j % 32)) & 1u) {
                probs[j] = 0.0;
                continue;
            }
            double dist = distMat[curr * n + j];
            if (dist < 1e-10) dist = 1e-10;
            double tau = pheromone[curr * n + j];
            double eta = 1.0 / dist;

            double p = pow(tau, alpha) * pow(eta, beta);
            probs[j] = p;
            totalProb += p;
        }

        // Roulette wheel selection
        double r = curand_uniform_double(&localState) * totalProb;
        double cumulative = 0.0;
        int nextCity = -1;

        for (int j = 0; j < n; ++j) {
            cumulative += probs[j];
            if (cumulative >= r) {
                nextCity = j;
                break;
            }
        }

        // Fallback: pick first unvisited
        if (nextCity < 0) {
            for (int j = 0; j < n; ++j)
                if (!((visited[j / 32] >> (j % 32)) & 1u)) { nextCity = j; break; }
        }

        tour[step] = nextCity;
        visited[nextCity / 32] |= (1u << (nextCity % 32));
    }

    // Compute tour cost
    double cost = 0.0;
    for (int i = 0; i < n; ++i)
        cost += distMat[tour[i] * n + tour[(i + 1) % n]];

    tourCosts[tid] = cost;
    states[tid] = localState;
}

/**
 * Pheromone evaporation: tau[i][j] *= (1 - rho)
 */
__global__ void evaporateKernel(double* pheromone, int n, double evapFactor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;
    pheromone[idx] *= evapFactor;
    if (pheromone[idx] < 1e-10) pheromone[idx] = 1e-10;
}

/**
 * Pheromone deposit: for each ant's tour, add Q/cost to edges
 */
__global__ void depositKernel(
    double* pheromone, const int* tours, const double* tourCosts,
    int n, int numAnts, double Q)
{
    int antId = blockIdx.x * blockDim.x + threadIdx.x;
    if (antId >= numAnts) return;

    double cost = tourCosts[antId];
    double deposit = Q / cost;
    const int* tour = tours + (long long)antId * n;

    for (int i = 0; i < n; ++i) {
        int from = tour[i], to = tour[(i + 1) % n];
        atomicAdd(&pheromone[from * n + to], deposit);
        atomicAdd(&pheromone[to * n + from], deposit);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <num_cities> [num_iters] [num_ants]\n";
        return 1;
    }

    int n = std::stoi(argv[1]);
    int numIters = (argc > 2) ? std::stoi(argv[2]) : 100;
    int numAnts = (argc > 3) ? std::stoi(argv[3]) : 1024;

    double alpha = 1.0;   // pheromone importance
    double beta = 3.0;    // heuristic importance
    double rho = 0.1;     // evaporation rate
    double Q = 1000.0;    // deposit constant

    if (n > 1024) {
        std::cerr << "n=" << n << " exceeds max 1024 for ACO visited bitmask\n";
        return 1;
    }

    // Generate random cities
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> coordDist(0.0, 1000.0);
    std::vector<double> cx(n), cy(n);
    for (int i = 0; i < n; ++i) { cx[i] = coordDist(rng); cy[i] = coordDist(rng); }

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

    // Initialize pheromone matrix
    double tau0 = 1.0 / (n * nnCost);
    std::vector<double> hostPheromone(n * n, tau0);

    double *d_dist, *d_pheromone, *d_tourCosts;
    curandState *d_states;
    int *d_tours;

    CUDA_CHECK(cudaMalloc(&d_dist, sizeof(double) * n * n));
    CUDA_CHECK(cudaMalloc(&d_pheromone, sizeof(double) * n * n));
    CUDA_CHECK(cudaMalloc(&d_states, sizeof(curandState) * numAnts));
    CUDA_CHECK(cudaMalloc(&d_tours, sizeof(int) * (long long)numAnts * n));
    CUDA_CHECK(cudaMalloc(&d_tourCosts, sizeof(double) * numAnts));

    CUDA_CHECK(cudaMemcpy(d_dist, hostDist.data(), sizeof(double)*n*n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pheromone, hostPheromone.data(), sizeof(double)*n*n, cudaMemcpyHostToDevice));

    int antBlock = 128;
    int antGrid = (numAnts + antBlock - 1) / antBlock;
    int phBlock = 256;
    int phGrid = (n * n + phBlock - 1) / phBlock;

    initRNG<<<antGrid, antBlock>>>(d_states, numAnts, 42ULL);
    CUDA_CHECK(cudaGetLastError());

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "═══════════════════════════════════════════════════════\n"
              << " GPU-Optimized TSP (ACO, " << numAnts << " ants, " << numIters << " iters)\n"
              << "═══════════════════════════════════════════════════════\n"
              << " Instance       : random " << n << " cities (seed=42)\n"
              << " NN cost        : " << nnCost << "\n";

    double globalBestCost = 1e30;
    std::vector<double> hostTourCosts(numAnts);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < numIters; ++iter) {
        // 1. Each ant builds a tour
        antTourKernel<<<antGrid, antBlock>>>(
            d_dist, d_pheromone, n, alpha, beta,
            d_states, d_tours, d_tourCosts, numAnts);

        // 2. Evaporate pheromones
        evaporateKernel<<<phGrid, phBlock>>>(d_pheromone, n, 1.0 - rho);

        // 3. Deposit pheromones
        depositKernel<<<antGrid, antBlock>>>(
            d_pheromone, d_tours, d_tourCosts, n, numAnts, Q);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    // Find best tour across all ants in last iteration
    CUDA_CHECK(cudaMemcpy(hostTourCosts.data(), d_tourCosts, sizeof(double)*numAnts, cudaMemcpyDeviceToHost));

    double bestCost = hostTourCosts[0];
    int bestAnt = 0;
    for (int i = 1; i < numAnts; ++i)
        if (hostTourCosts[i] < bestCost) { bestCost = hostTourCosts[i]; bestAnt = i; }

    std::cout << " Best tour cost : " << bestCost << "\n"
              << " Best ant       : " << bestAnt << "\n"
              << std::setprecision(6)
              << " Execution time : " << elapsed << " s\n"
              << "═══════════════════════════════════════════════════════\n";

    std::cout << "CSV," << n << "," << numAnts << ","
              << std::setprecision(4) << bestCost << "," << elapsed << "\n";

    cudaFree(d_dist); cudaFree(d_pheromone); cudaFree(d_states);
    cudaFree(d_tours); cudaFree(d_tourCosts);
    return 0;
}
