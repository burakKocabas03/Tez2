/**
 * WP2 – CPU Parallelization (OpenMP)
 * ====================================
 * Problem   : Maximum Clique Problem (MCP)
 * Algorithm : Parallel Branch-and-Bound Bron-Kerbosch
 *
 * Parallel Strategy
 * -----------------
 * The top-level B&B loop iterates over n starting vertices v_0, v_1, ..., v_{n-1}
 * (ordered by degree).  Each starting vertex defines a fully independent
 * sub-problem: "find the largest clique that contains v_i but not v_0..v_{i-1}".
 *
 * These sub-problems are distributed across P OpenMP threads using
 * DYNAMIC scheduling, which is essential here because sub-problem sizes vary
 * wildly (high-degree vertices near the start generate much larger search trees
 * than low-degree ones near the end).
 *
 * Cross-thread pruning
 * --------------------
 * Before each thread starts a new sub-problem it reads the current global best
 * size.  If the local best is already ≥ global, the global is not updated but
 * subsequent sub-problems benefit from tighter pruning.  This is a single
 * omp critical read — negligible overhead compared to the BK recursion.
 *
 * Expected speedup
 * ----------------
 * Near-linear for P threads when the graph has many equally hard sub-problems.
 * Speedup is slightly sub-linear due to dynamic imbalance at the very start
 * (the first few high-degree vertices generate the hardest sub-trees).
 *
 * Thesis   : A Comparison of NP-hard Problems through Parallel Algorithms
 *            Utilizing CPU and GPU-Based Solutions
 * Author   : Burak Kocabaş
 * Advisor  : Prof. Dr. Hasan BULUT
 *
 * Build    : see Makefile  (requires libomp on macOS: brew install libomp)
 * Run      : ./max_clique_openmp <dimacs_graph_file> [num_threads]
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
#include <atomic>

#include <omp.h>

// ---------------------------------------------------------------------------
//  Graph  (identical to WP1 for a fair comparison)
// ---------------------------------------------------------------------------

struct Graph {
    int n = 0, m = 0;
    std::vector<std::vector<bool>> adj;
    std::vector<int>               deg;

    explicit Graph(int n_)
        : n(n_), adj(n_, std::vector<bool>(n_, false)), deg(n_, 0) {}

    void addEdge(int u, int v) {
        if (!adj[u][v]) {
            adj[u][v] = adj[v][u] = true;
            ++deg[u]; ++deg[v];
            ++m;
        }
    }
};

// ---------------------------------------------------------------------------
//  BK kernel  (thread-local, identical logic to WP1)
// ---------------------------------------------------------------------------

static void bk(const Graph&             G,
               std::vector<int>&         clique,
               const std::vector<int>&   P,
               std::vector<int>&         best,     // thread-local best
               int                       globalBestSz,  // read-only hint for pruning
               long long&                nodes)
{
    ++nodes;

    if (P.empty()) {
        if (clique.size() > best.size())
            best = clique;
        return;
    }

    // Prune against both thread-local best and global best hint
    const int upperBound = static_cast<int>(clique.size() + P.size());
    if (upperBound <= static_cast<int>(best.size()))   return;
    if (upperBound <= globalBestSz)                    return;

    for (int i = 0; i < static_cast<int>(P.size()); ++i) {
        const int remaining = static_cast<int>(clique.size()) + static_cast<int>(P.size()) - i;
        if (remaining <= static_cast<int>(best.size()))   return;
        if (remaining <= globalBestSz)                    return;

        const int v = P[i];

        std::vector<int> newP;
        newP.reserve(P.size() - i - 1);
        for (int j = i + 1; j < static_cast<int>(P.size()); ++j)
            if (G.adj[v][P[j]])
                newP.push_back(P[j]);

        clique.push_back(v);
        bk(G, clique, newP, best, globalBestSz, nodes);
        clique.pop_back();
    }
}

// ---------------------------------------------------------------------------
//  DIMACS reader
// ---------------------------------------------------------------------------

Graph readDIMACS(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open file: " << filename << "\n";
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
        std::cerr << "Usage: " << argv[0] << " <dimacs_graph_file> [num_threads]\n";
        return EXIT_FAILURE;
    }

    const Graph G      = readDIMACS(argv[1]);
    int numThreads = (argc > 2) ? std::stoi(argv[2]) : omp_get_max_threads();
    if (numThreads > omp_get_max_threads())
        numThreads = omp_get_max_threads();
    omp_set_num_threads(numThreads);

    // Vertex ordering: descending degree (same as serial)
    const int n = G.n;
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return G.deg[a] > G.deg[b]; });

    std::cout << "═══════════════════════════════════════════════════════\n";
    std::cout << " WP2 – Maximum Clique Problem (OpenMP Parallel)\n";
    std::cout << "═══════════════════════════════════════════════════════\n";
    std::cout << " Instance    : " << argv[1] << "\n";
    std::cout << " Vertices    : " << G.n << "\n";
    std::cout << " Edges       : " << G.m << "\n";
    std::cout << " Threads     : " << numThreads << "\n";
    std::cout << "───────────────────────────────────────────────────────\n";

    // Shared mutable state: global best clique size (atomic for low-cost reads)
    std::vector<int> globalBest;
    std::atomic<int> globalBestSz{0};
    long long        totalNodes = 0;

    const auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel num_threads(numThreads)
    {
        std::vector<int> localBest;      // actual best clique found by this thread
        int              localBestSz = 0; // pruning bound (may exceed localBest.size())
        long long        localNodes  = 0;

        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < n; ++i) {
            // Snapshot global best for pruning within this sub-problem.
            // This thread-local copy avoids atomic reads inside the BK recursion.
            const int gSz = globalBestSz.load(std::memory_order_relaxed);
            if (gSz > localBestSz)
                localBestSz = gSz;

            // Early skip: remaining candidates can't beat known best
            if (n - i <= localBestSz) continue;

            const int v = order[i];

            std::vector<int> P;
            P.reserve(n - i - 1);
            for (int j = i + 1; j < n; ++j)
                if (G.adj[v][order[j]])
                    P.push_back(order[j]);

            std::vector<int> clique = {v};
            bk(G, clique, P, localBest, localBestSz, localNodes);
            localBestSz = std::max(localBestSz, static_cast<int>(localBest.size()));

            // Publish improved result to global best
            if (localBestSz > globalBestSz.load(std::memory_order_relaxed)) {
                #pragma omp critical
                {
                    if (localBest.size() > globalBest.size()) {
                        globalBest = localBest;
                        globalBestSz.store(static_cast<int>(globalBest.size()),
                                           std::memory_order_release);
                    }
                }
            }
        }

        #pragma omp critical
        {
            totalNodes += localNodes;
            if (localBest.size() > globalBest.size()) {
                globalBest = localBest;
                globalBestSz.store(static_cast<int>(globalBest.size()),
                                   std::memory_order_release);
            }
        }
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << " Maximum clique size : " << globalBest.size() << "\n";
    std::cout << " Clique vertices     : ";
    for (int v : globalBest) std::cout << v + 1 << " ";
    std::cout << "\n";
    std::cout << " Nodes explored      : " << totalNodes << "\n";
    std::cout << " Threads used        : " << numThreads << "\n";
    std::cout << std::setprecision(6);
    std::cout << " Execution time      : " << elapsed << " s\n";
    std::cout << "═══════════════════════════════════════════════════════\n";

    // CSV: n, m, threads, clique_size, time_s, nodes
    std::cout << "CSV," << G.n << "," << G.m << "," << numThreads << ","
              << globalBest.size() << "," << elapsed << "," << totalNodes
              << "\n";

    return EXIT_SUCCESS;
}
