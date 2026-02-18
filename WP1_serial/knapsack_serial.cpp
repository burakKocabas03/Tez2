/**
 * WP1 – Baseline Serial Implementation
 * ======================================
 * Problem  : 0/1 Knapsack Problem
 * Algorithm: Dynamic Programming (bottom-up, space-optimized 2-row)
 *
 * Given n items each with weight w[i] and value v[i], and a knapsack of
 * capacity W, select a subset of items that maximises total value without
 * exceeding total weight.
 *
 * DP formulation
 * --------------
 *   dp[i][c] = maximum value achievable using items 0..i-1 with capacity c
 *
 *   dp[i][c] = dp[i-1][c]                              if c < w[i]
 *            = max(dp[i-1][c], dp[i-1][c-w[i]] + v[i]) otherwise
 *
 * Space optimisation: only two rows needed (current and previous).
 * The inner loop over capacities is independent for each row → ideal for
 * parallelisation in WP2 (OpenMP) and WP3 (CUDA).
 *
 * Time complexity : O(n * W)
 * Space complexity: O(W)   (2-row rolling)
 *
 * Input format:
 *   N W
 *   w1 v1
 *   w2 v2
 *   ...
 *
 * Thesis   : A Comparison of NP-hard Problems through Parallel Algorithms
 *            Utilizing CPU and GPU-Based Solutions
 * Authors  : Burak Kocabaş & Emin Özgür Elmalı
 * Advisor  : Prof. Dr. Hasan BULUT
 *
 * Build    : g++ -O3 -std=c++17 -o knapsack_serial knapsack_serial.cpp
 * Run      : ./knapsack_serial <instance_file>
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>

// ---------------------------------------------------------------------------
//  Instance
// ---------------------------------------------------------------------------

struct KnapsackInstance {
    int              n;       // number of items
    long long        W;       // capacity
    std::vector<int> weight;
    std::vector<int> value;
};

KnapsackInstance readInstance(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open file: " << filename << "\n";
        std::exit(EXIT_FAILURE);
    }

    KnapsackInstance inst;
    file >> inst.n >> inst.W;
    inst.weight.resize(inst.n);
    inst.value .resize(inst.n);

    for (int i = 0; i < inst.n; ++i)
        file >> inst.weight[i] >> inst.value[i];

    return inst;
}

// ---------------------------------------------------------------------------
//  DP solver
// ---------------------------------------------------------------------------

struct KnapsackResult {
    long long optimalValue;
    long long dpCells;         // total DP cells computed = n * (W+1)
};

KnapsackResult solveKnapsack(const KnapsackInstance& inst) {
    const int       n = inst.n;
    const long long W = inst.W;

    // Two-row rolling DP: prev[c] = best value with capacity c using items so far
    std::vector<long long> prev(W + 1, 0LL);
    std::vector<long long> curr(W + 1, 0LL);

    for (int i = 0; i < n; ++i) {
        const int wi = inst.weight[i];
        const int vi = inst.value[i];

        for (long long c = 0; c <= W; ++c) {
            curr[c] = prev[c];                      // don't take item i
            if (c >= wi && prev[c - wi] + vi > curr[c])
                curr[c] = prev[c - wi] + vi;        // take item i
        }

        std::swap(prev, curr);
    }

    return {prev[W], static_cast<long long>(n) * (W + 1)};
}

// ---------------------------------------------------------------------------
//  Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <instance_file>\n"
                  << "  Format: first line 'N W', then N lines 'weight value'\n";
        return EXIT_FAILURE;
    }

    const auto inst = readInstance(argv[1]);

    std::cout << "═══════════════════════════════════════════════════════\n";
    std::cout << " WP1 – 0/1 Knapsack Problem (Serial DP)\n";
    std::cout << "═══════════════════════════════════════════════════════\n";
    std::cout << " Instance    : " << argv[1] << "\n";
    std::cout << " Items (n)   : " << inst.n   << "\n";
    std::cout << " Capacity (W): " << inst.W   << "\n";
    std::cout << " DP cells    : " << static_cast<long long>(inst.n) * (inst.W + 1) << "\n";
    std::cout << "───────────────────────────────────────────────────────\n";

    const auto t0  = std::chrono::high_resolution_clock::now();
    const auto res = solveKnapsack(inst);
    const auto t1  = std::chrono::high_resolution_clock::now();

    const double elapsed = std::chrono::duration<double>(t1 - t0).count();
    const double mpps    = (res.dpCells / 1e6) / elapsed;  // million cells/sec

    std::cout << " Optimal value       : " << res.optimalValue << "\n";
    std::cout << " DP cells computed   : " << res.dpCells << "\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << " Throughput          : " << mpps << " M cells/s\n";
    std::cout << std::setprecision(6);
    std::cout << " Execution time      : " << elapsed << " s\n";
    std::cout << "═══════════════════════════════════════════════════════\n";

    // CSV: n, W, optimal_value, time_s, dp_cells
    std::cout << "CSV," << inst.n << "," << inst.W << ","
              << res.optimalValue << "," << elapsed << "," << res.dpCells << "\n";

    return EXIT_SUCCESS;
}
