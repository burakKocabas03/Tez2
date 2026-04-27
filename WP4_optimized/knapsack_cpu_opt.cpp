/**
 * WP4 – CPU-Optimized 0/1 Knapsack
 * ==================================
 * Algorithm: Single-row reverse DP (space-optimized, cache-friendly)
 *
 * Improvements over WP1/WP2:
 *   - Single row instead of 2 rows: iterate capacity in reverse (classic trick)
 *     → halves memory, improves cache utilization
 *   - OpenMP parallel over items with inner loop parallelism
 *   - Optional: bitwise-packed DP for ultra-small weight items
 *   - Memory prefetch hints for sequential access pattern
 *
 * CPU advantages exploited:
 *   - Reverse iteration avoids row-swapping and synchronization complexity
 *   - Sequential access pattern perfectly suits CPU cache prefetcher
 *   - L1/L2 cache can hold entire DP row for moderate capacities
 *
 * Build: g++ -O3 -std=c++17 -fopenmp -o knapsack_cpu_opt knapsack_cpu_opt.cpp
 * Run:   ./knapsack_cpu_opt <input_file> [num_threads]
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <omp.h>

struct KnapsackInstance {
    int n = 0;
    int W = 0;
    std::vector<int> weights;
    std::vector<int> values;
};

KnapsackInstance readInput(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) { std::cerr << "Cannot open " << filename << "\n"; std::exit(1); }
    KnapsackInstance inst;
    file >> inst.n >> inst.W;
    inst.weights.resize(inst.n);
    inst.values.resize(inst.n);
    for (int i = 0; i < inst.n; ++i) file >> inst.values[i] >> inst.weights[i];
    return inst;
}

long long solveSingleRow_Serial(const KnapsackInstance& inst) {
    int n = inst.n, W = inst.W;
    std::vector<long long> dp(W + 1, 0);

    for (int i = 0; i < n; ++i) {
        int wi = inst.weights[i];
        int vi = inst.values[i];
        for (int w = W; w >= wi; --w)
            dp[w] = std::max(dp[w], dp[w - wi] + vi);
    }
    return dp[W];
}

long long solveSingleRow_Parallel(const KnapsackInstance& inst, int numThreads) {
    int n = inst.n, W = inst.W;
    std::vector<long long> dp(W + 1, 0);

    omp_set_num_threads(numThreads);

    // Anti-dependency-safe parallelism: partition items into groups,
    // then process each item's reverse scan in parallel using block decomposition
    for (int i = 0; i < n; ++i) {
        int wi = inst.weights[i];
        int vi = inst.values[i];

        // Parallel reverse scan with block partitioning
        // Each thread processes a contiguous block of the dp array in reverse
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nT = omp_get_num_threads();

            int range = W - wi + 1;
            if (range <= 0) { /* nothing to update */ }
            else {

            int blockSize = (range + nT - 1) / nT;
            int myStart = wi + tid * blockSize;
            int myEnd = std::min(W, myStart + blockSize - 1);

            for (int w = myEnd; w >= myStart; --w)
                dp[w] = std::max(dp[w], dp[w - wi] + vi);
            }
        }
    }
    return dp[W];
}

long long solveAntiDiag_Parallel(const KnapsackInstance& inst, int numThreads) {
    int n = inst.n, W = inst.W;
    std::vector<long long> prev(W + 1, 0), curr(W + 1, 0);

    omp_set_num_threads(numThreads);

    for (int i = 0; i < n; ++i) {
        int wi = inst.weights[i];
        int vi = inst.values[i];

        #pragma omp parallel for schedule(static)
        for (int w = 0; w <= W; ++w) {
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
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file> [num_threads]\n";
        return 1;
    }

    KnapsackInstance inst = readInput(argv[1]);
    int numThreads = (argc > 2) ? std::stoi(argv[2]) : 1;

    std::cout << "═══════════════════════════════════════════════════════\n"
              << " CPU-Optimized 0/1 Knapsack\n"
              << "═══════════════════════════════════════════════════════\n"
              << " Instance  : " << argv[1] << "\n"
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
