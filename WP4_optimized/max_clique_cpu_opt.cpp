/**
 * WP4 – CPU-Optimized Maximum Clique
 * ====================================
 * Algorithm: BK + Greedy Coloring Upper Bound + Pivot Selection
 *
 * Improvements over WP1/WP2:
 *   - Greedy coloring bound: assigns colors to candidates, the number of
 *     distinct colors is a tighter upper bound than |P| for the clique size
 *   - Pivot selection: choose the vertex in P with maximum connections to
 *     other candidates, reducing the number of recursive branches
 *   - Bitset adjacency: std::vector<uint64_t> bitmask for fast set operations
 *
 * CPU advantages exploited:
 *   - Deep recursion with complex branching → CPU branch predictor excels
 *   - Coloring requires sequential greedy assignment → inherently serial
 *   - Large L2/L3 cache holds adjacency matrix for moderate graphs
 *
 * Build: g++ -O3 -std=c++17 -fopenmp -o max_clique_cpu_opt max_clique_cpu_opt.cpp
 * Run:   ./max_clique_cpu_opt <dimacs_file> [num_threads]
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
#include <cstring>
#include <omp.h>

static constexpr int WORD_BITS = 64;

struct Graph {
    int n = 0, m = 0;
    std::vector<std::vector<uint64_t>> adjBits;
    std::vector<int> deg;
    int words = 0;

    explicit Graph(int n_) : n(n_), deg(n_, 0) {
        words = (n + WORD_BITS - 1) / WORD_BITS;
        adjBits.assign(n, std::vector<uint64_t>(words, 0ULL));
    }

    void addEdge(int u, int v) {
        if (!(adjBits[u][v / WORD_BITS] & (1ULL << (v % WORD_BITS)))) {
            adjBits[u][v / WORD_BITS] |= (1ULL << (v % WORD_BITS));
            adjBits[v][u / WORD_BITS] |= (1ULL << (u % WORD_BITS));
            ++deg[u]; ++deg[v]; ++m;
        }
    }

    inline bool adjacent(int u, int v) const {
        return (adjBits[u][v / WORD_BITS] >> (v % WORD_BITS)) & 1ULL;
    }
};

static inline int popcount64(uint64_t x) { return __builtin_popcountll(x); }

// Greedy coloring: returns color count (upper bound on clique size in P)
int greedyColoring(const Graph& G, const std::vector<int>& P,
                   std::vector<int>& colorOrder) {
    int np = (int)P.size();
    std::vector<int> color(np, 0);
    int maxColor = 0;

    for (int i = 0; i < np; ++i) {
        // Find smallest color not used by already-colored neighbors in P
        std::vector<bool> usedColors(np + 1, false);
        for (int j = 0; j < i; ++j) {
            if (G.adjacent(P[i], P[j]))
                usedColors[color[j]] = true;
        }
        int c = 1;
        while (usedColors[c]) ++c;
        color[i] = c;
        if (c > maxColor) maxColor = c;
    }

    // Reorder: vertices with higher colors first (they form tighter bounds)
    std::vector<std::pair<int,int>> colorVertex(np);
    for (int i = 0; i < np; ++i)
        colorVertex[i] = {color[i], P[i]};
    std::sort(colorVertex.begin(), colorVertex.end());

    colorOrder.resize(np);
    for (int i = 0; i < np; ++i)
        colorOrder[i] = colorVertex[i].second;

    return maxColor;
}

// Pivot: select vertex in P with most connections to other P vertices
int selectPivot(const Graph& G, const std::vector<int>& P) {
    int bestV = P[0], bestConn = -1;
    for (int v : P) {
        int conn = 0;
        for (int u : P)
            if (u != v && G.adjacent(v, u)) ++conn;
        if (conn > bestConn) { bestConn = conn; bestV = v; }
    }
    return bestV;
}

static void bk_opt(const Graph& G,
                   std::vector<int>& clique,
                   std::vector<int>& P,
                   std::vector<int>& best,
                   int globalBestSz,
                   long long& nodes) {
    ++nodes;

    if (P.empty()) {
        if ((int)clique.size() > (int)best.size())
            best = clique;
        return;
    }

    // Coloring-based upper bound
    std::vector<int> colorOrder;
    int colorBound = greedyColoring(G, P, colorOrder);

    if ((int)clique.size() + colorBound <= (int)best.size()) return;
    if ((int)clique.size() + colorBound <= globalBestSz) return;

    // Pivot-based branching: only branch on vertices NOT adjacent to pivot
    int pivot = selectPivot(G, P);

    for (int idx = (int)colorOrder.size() - 1; idx >= 0; --idx) {
        int remaining = idx + 1;
        if ((int)clique.size() + remaining <= (int)best.size()) return;
        if ((int)clique.size() + remaining <= globalBestSz) return;

        int v = colorOrder[idx];

        // Skip vertices adjacent to pivot (they can't extend a maximal clique
        // that doesn't include pivot -- Bron-Kerbosch optimization)
        if (G.adjacent(v, pivot) && v != pivot) continue;

        std::vector<int> newP;
        newP.reserve(P.size());
        for (int u : P)
            if (G.adjacent(v, u)) newP.push_back(u);

        clique.push_back(v);
        bk_opt(G, clique, newP, best, globalBestSz, nodes);
        clique.pop_back();

        // Remove v from P (it's been fully explored)
        P.erase(std::find(P.begin(), P.end(), v));
    }
}

Graph readDIMACS(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) { std::cerr << "Cannot open " << filename << "\n"; std::exit(1); }
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

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dimacs_file> [num_threads]\n";
        return 1;
    }

    Graph G = readDIMACS(argv[1]);
    int numThreads = (argc > 2) ? std::stoi(argv[2]) : omp_get_max_threads();
    omp_set_num_threads(numThreads);

    int n = G.n;
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return G.deg[a] > G.deg[b]; });

    std::vector<int> globalBest;
    std::atomic<int> globalBestSz{0};
    long long totalNodes = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        std::vector<int> localBest;
        int localBestSz = 0;
        long long localNodes = 0;

        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < n; ++i) {
            int gSz = globalBestSz.load(std::memory_order_relaxed);
            if (gSz > localBestSz) localBestSz = gSz;
            if (n - i <= localBestSz) continue;

            int v = order[i];
            std::vector<int> P;
            for (int j = i + 1; j < n; ++j)
                if (G.adjacent(v, order[j])) P.push_back(order[j]);

            std::vector<int> clique = {v};
            bk_opt(G, clique, P, localBest, localBestSz, localNodes);
            localBestSz = std::max(localBestSz, (int)localBest.size());

            if (localBestSz > globalBestSz.load(std::memory_order_relaxed)) {
                #pragma omp critical
                {
                    if ((int)localBest.size() > (int)globalBest.size()) {
                        globalBest = localBest;
                        globalBestSz.store((int)globalBest.size(), std::memory_order_release);
                    }
                }
            }
        }

        #pragma omp critical
        {
            totalNodes += localNodes;
            if ((int)localBest.size() > (int)globalBest.size()) {
                globalBest = localBest;
                globalBestSz.store((int)globalBest.size(), std::memory_order_release);
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "═══════════════════════════════════════════════════════\n"
              << " CPU-Optimized Max Clique (BK + Coloring + Pivot)\n"
              << "═══════════════════════════════════════════════════════\n"
              << " Instance    : " << argv[1] << "\n"
              << " Vertices    : " << G.n << "   Edges: " << G.m << "\n"
              << " Threads     : " << numThreads << "\n"
              << " Clique size : " << globalBest.size() << "\n"
              << " Nodes       : " << totalNodes << "\n"
              << std::setprecision(6)
              << " Time        : " << elapsed << " s\n"
              << "═══════════════════════════════════════════════════════\n";

    std::cout << "CSV," << G.n << "," << G.m << "," << numThreads << ","
              << globalBest.size() << "," << elapsed << "," << totalNodes << "\n";
    return 0;
}
