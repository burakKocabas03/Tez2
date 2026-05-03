/**
 * WP4 – CPU-Optimized 0/1 Knapsack
 * ==================================
 * Algorithm: Single-row reverse DP (space-optimized, cache-friendly)
 *
 * Improvements over WP1/WP2:
 *   - Serial: single-row reverse DP → halves memory, improves cache utilization
 *   - Parallel: two-row forward DP with OpenMP (race-condition-free)
 *   - Sequential access pattern perfectly suits CPU cache prefetcher
 *
 * CPU advantages exploited:
 *   - Reverse iteration avoids row-swapping and synchronization complexity
 *   - Sequential access pattern perfectly suits CPU cache prefetcher
 *   - L1/L2 cache can hold entire DP row for moderate capacities
 *
 * Build: g++ -O3 -std=c++17 -fopenmp -o knapsack_cpu_opt knapsack_cpu_opt.cpp
 * Run:   ./knapsack_cpu_opt <num_items> <capacity> [num_threads]
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <random>
#include <omp.h>

struct KnapsackInstance {
    int n = 0;
    long long W = 0;
    std::vector<int> weights;
    std::vector<int> values;
};

KnapsackInstance generateRandom(int n, long long W, unsigned seed = 42) {
    KnapsackInstance inst;
    inst.n = n;
    inst.W = W;
    inst.weights.resize(n);
    inst.values.resize(n);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> distW(1, std::max(1LL, W / 10));
    std::uniform_int_distribution<int> distV(10, 1000);
    for (int i = 0; i < n; ++i) {
        inst.weights[i] = distW(rng);
        inst.values[i] = distV(rng);
    }
    return inst;
}

long long solveSingleRow_Serial(const KnapsackInstance& inst) {
    long long W = inst.W;
    int n = inst.n;
    std::vector<long long> dp(W + 1, 0);

    for (int i = 0; i < n; ++i) {
        int wi = inst.weights[i];
        int vi = inst.values[i];
        for (long long w = W; w >= wi; --w)
            dp[w] = std::max(dp[w], dp[w - wi] + vi);
    }
    return dp[W];
}

long long solveAntiDiag_Parallel(const KnapsackInstance& inst, int numThreads) {
    long long W = inst.W;
    int n = inst.n;
    std::vector<long long> prev(W + 1, 0), curr(W + 1, 0);

    omp_set_num_threads(numThreads);

    for (int i = 0; i < n; ++i) {
        int wi = inst.weights[i];
        int vi = inst.values[i];

        #pragma omp parallel for schedule(static)
        for (long long w = 0; w <= W; ++w) {
            if (w >= wi)
                curr[w] = std::max(prev[w], prev[w - wi] + vi);
            else
                curr[w] = prev[w];
        }

        std::swap(prev, curr);
    }
    return prev[W];
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <num_items> <capacity> [num_threads]\n";
        return 1;
    }

    int n_items = std::stoi(argv[1]);
    long long cap = std::stoll(argv[2]);
    KnapsackInstance inst = generateRandom(n_items, cap);
    int numThreads = (argc > 3) ? std::stoi(argv[3]) : omp_get_max_threads();

    std::cout << "═══════════════════════════════════════════════════════\n"
              << " CPU-Optimized 0/1 Knapsack\n"
              << "═══════════════════════════════════════════════════════\n"
              << " Instance  : random n=" << inst.n << " W=" << inst.W << " (seed=42)\n"
              << " Items     : " << inst.n << "   Capacity: " << inst.W << "\n"
              << " Threads   : " << numThreads << "\n";

    long long result;
    auto t0 = std::chrono::high_resolution_clock::now();

    if (numThreads == 1) {
        result = solveSingleRow_Serial(inst);
    } else {
        result = solveAntiDiag_Parallel(inst, numThreads);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    double cells = (double)inst.n * (double)inst.W;
    double throughput = cells / elapsed / 1e6;

    std::cout << " Optimal   : " << result << "\n"
              << std::setprecision(6)
              << " Time      : " << elapsed << " s\n"
              << std::setprecision(2)
              << " Throughput: " << throughput << " Mcells/s\n"
              << " Method    : " << (numThreads == 1 ? "Single-row reverse DP" : "Two-row parallel DP") << "\n"
              << "═══════════════════════════════════════════════════════\n";

    std::cout << "CSV," << inst.n << "," << inst.W << "," << numThreads << ","
              << result << "," << elapsed << "," << throughput << "\n";
    return 0;
}
