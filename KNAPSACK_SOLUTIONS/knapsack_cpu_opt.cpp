/**
 * WP4 – CPU Sparse DP (SKPDP) 0/1 Knapsack — CORRECTED
 * =====================================================
 *
 * Bug fix: replaced incorrect divide-and-conquer with standard iterative
 * Pareto frontier update.  The D&C combine was doing merge(A,B) instead
 * of convolution(A,B) (pairwise sums), which silently dropped valid
 * item combinations.
 *
 * Algorithm: for each item, create shifted frontier (all states + item),
 * then merge-prune the union of old and shifted states.
 *
 * Build: g++ -O3 -std=c++17 -fopenmp -march=native -o knapsack_cpu_sparse knapsack_cpu_sparse.cpp
 * Run:   ./knapsack_cpu_sparse random <num_items> <capacity> [seed] [num_threads]
 *    or: ./knapsack_cpu_sparse <input_file> [num_threads]
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <random>
#include <cstring>
#include <omp.h>

struct KnapsackInstance {
    int n = 0;
    long long W = 0;
    std::vector<long long> weights;
    std::vector<long long> values;
};

/* ------------------------------------------------------------------ */
/*  Input                                                             */
/* ------------------------------------------------------------------ */
KnapsackInstance readInput(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) { std::cerr << "Cannot open " << filename << "\n"; std::exit(1); }
    KnapsackInstance inst;

    // Auto-detect format by reading the first line
    std::string firstLine;
    std::getline(file, firstLine);
    std::istringstream iss(firstLine);

    long long a, b;
    iss >> a;
    if (iss >> b) {
        // Two numbers → Pisinger format: "N W", then "value weight"
        inst.n = static_cast<int>(a);
        inst.W = b;
        inst.weights.resize(inst.n);
        inst.values.resize(inst.n);
        for (int i = 0; i < inst.n; ++i) file >> inst.values[i] >> inst.weights[i];
    } else {
        // Single number → test.in format: "N", then "idx value weight", last line "W"
        inst.n = static_cast<int>(a);
        inst.weights.resize(inst.n);
        inst.values.resize(inst.n);
        for (int i = 0; i < inst.n; ++i) {
            int idx;
            file >> idx >> inst.values[i] >> inst.weights[i];
        }
        file >> inst.W;
    }

    return inst;
}

KnapsackInstance generateRandom(int n, long long W, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> distW(1, static_cast<int>(std::max(1LL, W / 10)));
    std::uniform_int_distribution<int> distV(10, 1000);
    KnapsackInstance inst;
    inst.n = n; inst.W = W;
    inst.weights.resize(n);
    inst.values.resize(n);
    for (int i = 0; i < n; ++i) {
        inst.weights[i] = distW(rng);
        inst.values[i]  = distV(rng);
    }
    return inst;
}

/* ------------------------------------------------------------------ */
/*  Frontier — raw buffer                                             */
/* ------------------------------------------------------------------ */
struct Frontier {
    long long* w;           // weights  (sorted ascending)
    long long* v;           // values   (strictly increasing after prune)
    int size;
    int cap;

    explicit Frontier(int initial_cap = 16) : size(0) {
        cap = std::max(initial_cap, 4);
        w = static_cast<long long*>(std::malloc(sizeof(long long) * cap));
        v = static_cast<long long*>(std::malloc(sizeof(long long) * cap));
        if (!w || !v) { std::cerr << "OOM\n"; std::exit(1); }
    }

    ~Frontier() { std::free(w); std::free(v); }

    Frontier(const Frontier&) = delete;
    Frontier& operator=(const Frontier&) = delete;

    Frontier(Frontier&& other) noexcept
        : w(other.w), v(other.v), size(other.size), cap(other.cap) {
        other.w = nullptr; other.v = nullptr; other.size = other.cap = 0;
    }

    Frontier& operator=(Frontier&& other) noexcept {
        if (this != &other) {
            std::free(w); std::free(v);
            w = other.w; v = other.v;
            size = other.size; cap = other.cap;
            other.w = nullptr; other.v = nullptr;
            other.size = other.cap = 0;
        }
        return *this;
    }

    void reserve(int new_cap) {
        if (new_cap <= cap) return;
        cap = new_cap * 2;
        w = static_cast<long long*>(std::realloc(w, sizeof(long long) * cap));
        v = static_cast<long long*>(std::realloc(v, sizeof(long long) * cap));
        if (!w || !v) { std::cerr << "OOM\n"; std::exit(1); }
    }

    void push(long long weight, long long value) {
        reserve(size + 1);
        w[size] = weight;
        v[size] = value;
        ++size;
    }
};

/* ------------------------------------------------------------------ */
/*  FUSED MERGE-PRUNE                                                 */
/*  A, B: sorted by weight, strictly increasing values.               */
/*  Returns Pareto frontier of (A ∪ B) with weight ≤ W_cap.           */
/* ------------------------------------------------------------------ */
Frontier merge_prune(const Frontier& A, const Frontier& B, long long W_cap) {
    int na = A.size, nb = B.size;
    Frontier C(na + nb + 2);

    int i = 0, j = 0;
    long long best_v = -1;

    auto consider = [&](long long weight, long long value) {
        if (weight > W_cap) return;
        if (value > best_v) {
            C.w[C.size] = weight;
            C.v[C.size] = value;
            ++C.size;
            best_v = value;
        }
    };

    while (i < na && j < nb) {
        if (A.w[i] < B.w[j]) {
            consider(A.w[i], A.v[i]);
            ++i;
        } else if (B.w[j] < A.w[i]) {
            consider(B.w[j], B.v[j]);
            ++j;
        } else {
            consider(A.w[i], std::max(A.v[i], B.v[j]));
            ++i; ++j;
        }
    }
    while (i < na) { consider(A.w[i], A.v[i]); ++i; }
    while (j < nb) { consider(B.w[j], B.v[j]); ++j; }

    return C;
}

/* ------------------------------------------------------------------ */
/*  ITERATIVE Sparse DP — CORRECT                                     */
/* ------------------------------------------------------------------ */
long long solveSparseDP(const KnapsackInstance& inst,
                        size_t& max_frontier,
                        int num_threads)
{
    long long W = inst.W;
    int n = inst.n;

    Frontier f(2);
    f.push(0, 0);
    max_frontier = 1;

    for (int i = 0; i < n; ++i) {
        long long wi = inst.weights[i];
        long long vi = inst.values[i];

        // Build shifted frontier: every existing state + this item
        Frontier shifted(f.size + 2);
        shifted.size = f.size;

        // Parallelize shift only when frontier is large enough to pay off
        if (f.size > 4096) {
            #pragma omp parallel for num_threads(num_threads) schedule(static)
            for (int k = 0; k < f.size; ++k) {
                shifted.w[k] = f.w[k] + wi;
                shifted.v[k] = f.v[k] + vi;
            }
        } else {
            for (int k = 0; k < f.size; ++k) {
                shifted.w[k] = f.w[k] + wi;
                shifted.v[k] = f.v[k] + vi;
            }
        }

        // Merge: keep best of (don't take item) ∪ (take item)
        f = merge_prune(f, shifted, W);

        if (static_cast<size_t>(f.size) > max_frontier)
            max_frontier = f.size;
    }

    return f.v[f.size - 1];
}

/* ------------------------------------------------------------------ */
/*  Dense DP fallback (OpenMP)                                        */
/* ------------------------------------------------------------------ */
long long solve_dense_openmp(const KnapsackInstance& inst, int num_threads) {
    long long W = inst.W;
    int n = inst.n;
    std::vector<long long> prev(W + 1, 0), curr(W + 1, 0);
    omp_set_num_threads(num_threads);

    for (int i = 0; i < n; ++i) {
        long long wi = inst.weights[i];
        long long vi = inst.values[i];
        #pragma omp parallel for schedule(static)
        for (long long w = 0; w <= W; ++w) {
            curr[w] = (w >= wi) ? std::max(prev[w], prev[w - wi] + vi) : prev[w];
        }
        std::swap(prev, curr);
    }
    return prev[W];
}

/* ------------------------------------------------------------------ */
/*  HYBRID: auto-select dense vs sparse                               */
/* ------------------------------------------------------------------ */
long long solve_hybrid(const KnapsackInstance& inst,
                       int num_threads,
                       size_t& max_frontier)
{
    // Dense is better when capacity is small (frontier ≈ W anyway)
    // Also, dense is impossible when W > ~500M (memory limit)
    if (inst.W > 500000000LL) {
        // Force sparse — dense would need >4GB per row
        return solveSparseDP(inst, max_frontier, num_threads);
    }
    if (inst.W <= 50000 || inst.n <= 64) {
        max_frontier = static_cast<size_t>(inst.W);
        return solve_dense_openmp(inst, num_threads);
    }
    return solveSparseDP(inst, max_frontier, num_threads);
}

/* ------------------------------------------------------------------ */
/*  Main                                                              */
/* ------------------------------------------------------------------ */
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file> [num_threads]\n"
                  << "   or: " << argv[0] << " random <num_items> <capacity> [seed] [num_threads]\n";
        return 1;
    }

    KnapsackInstance inst;
    std::string source_desc;
    int num_threads = omp_get_max_threads();

    if (std::string(argv[1]) == "random") {
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0] << " random <num_items> <capacity> [seed] [num_threads]\n";
            return 1;
        }
        int n = std::stoi(argv[2]);
        long long W = std::stoll(argv[3]);
        unsigned seed = (argc > 4) ? static_cast<unsigned>(std::stoi(argv[4])) : 42u;
        num_threads = (argc > 5) ? std::stoi(argv[5]) : omp_get_max_threads();
        inst = generateRandom(n, W, seed);
        source_desc = "random n=" + std::to_string(n) + " W=" + std::to_string(W)
                    + " (seed=" + std::to_string(seed) + ")";
    } else {
        inst = readInput(argv[1]);
        if (argc > 2) num_threads = std::stoi(argv[2]);
        source_desc = argv[1];
    }

    std::cout << "═══════════════════════════════════════════════════════\n"
              << " CPU Sparse DP (SKPDP) — CORRECTED\n"
              << "═══════════════════════════════════════════════════════\n"
              << " Instance  : " << source_desc << "\n"
              << " Items     : " << inst.n << "   Capacity: " << inst.W << "\n"
              << " Threads   : " << num_threads << "\n";

    size_t max_frontier = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    long long result = solve_hybrid(inst, num_threads, max_frontier);
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    double cells = static_cast<double>(inst.n) * static_cast<double>(inst.W);
    double throughput = cells / elapsed / 1e6;

    std::cout << " Optimal   : " << result << "\n"
              << std::setprecision(6) << " Time      : " << elapsed << " s\n"
              << std::setprecision(2) << " Throughput: " << throughput << " Mcells/s (vs dense n*W)\n"
              << " Frontier  : max=" << max_frontier << "\n"
              << " Method    : " << (inst.W <= 50000 ? "Dense DP (auto-selected)" : "Sparse DP iterative") << "\n"
              << "═══════════════════════════════════════════════════════\n";

    std::cout << "CSV," << inst.n << "," << inst.W << "," << num_threads << ","
              << result << "," << elapsed << "," << throughput << "\n";
    return 0;
}