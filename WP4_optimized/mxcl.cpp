/**
 * WP4 – CPU-Optimized Maximum Clique (CORRECTED TOMITA + DOUBLE DENSITY)
 * =======================================================================
 *
 * Build: g++ -O3 -std=c++17 -fopenmp -march=native -o max_clique_cpu_opt max_clique_cpu_opt.cpp
 * Run:   ./max_clique_cpu_opt <num_vertices> <density_percent> [num_threads]
 *        e.g. ./max_clique_cpu_opt 100 50.5 16
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <atomic>
#include <random>
#include <omp.h>

static constexpr int WORD_BITS = 64;

/* ------------------------------------------------------------------ */
/*  Graph                                                             */
/* ------------------------------------------------------------------ */
struct Graph {
    int n = 0, m = 0, words = 0;
    std::vector<std::vector<uint64_t>> adjBits;
    std::vector<int> deg;

    explicit Graph(int n_) : n(n_), deg(n_, 0) {
        words = (n + WORD_BITS - 1) / WORD_BITS;
        adjBits.assign(n, std::vector<uint64_t>(words, 0ULL));
    }

    void addEdge(int u, int v) {
        if (u == v) return;
        int wu = u / WORD_BITS, bu = u % WORD_BITS;
        int wv = v / WORD_BITS, bv = v % WORD_BITS;
        if (!(adjBits[u][wv] >> bv & 1ULL)) {
            adjBits[u][wv] |= (1ULL << bv);
            adjBits[v][wu] |= (1ULL << bu);
            ++deg[u]; ++deg[v]; ++m;
        }
    }

    inline bool adjacent(int u, int v) const {
        return (adjBits[u][v / WORD_BITS] >> (v % WORD_BITS)) & 1ULL;
    }
};

Graph generateRandomGraph(int n, double densityPct, unsigned seed = 42) {
    Graph G(n);
    std::mt19937 rng(seed);
    // FIX: real-valued density [0.0, 100.0)
    std::uniform_real_distribution<double> pctDist(0.0, 1.0);

    for (int u = 0; u < n; ++u)
        for (int v = u + 1; v < n; ++v)
            if (pctDist(rng) < densityPct)
                G.addEdge(u, v);
    return G;
}

/* ------------------------------------------------------------------ */
/*  Greedy coloring (O(|P|))                                         */
/* ------------------------------------------------------------------ */
int greedyColor(const Graph& G, const std::vector<int>& P,
                std::vector<int>& color, std::vector<int>& order) {
    int np = (int)P.size();
    if (np == 0) return 0;

    static thread_local std::vector<int> seen;
    static thread_local int seenGen = 0;
    if ((int)seen.size() < np + 2) seen.resize(np + 2, 0);

    int maxColor = 0;
    color.resize(np);
    order.resize(np);

    for (int i = 0; i < np; ++i) {
        int v = P[i];
        ++seenGen;
        if (seenGen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seenGen = 1;
        }
        for (int j = 0; j < i; ++j)
            if (G.adjacent(v, P[j])) seen[color[j]] = seenGen;

        int c = 1;
        while (c <= maxColor && seen[c] == seenGen) ++c;
        color[i] = c;
        if (c > maxColor) maxColor = c;
    }

    std::vector<std::pair<int,int>> tmp(np);
    for (int i = 0; i < np; ++i) tmp[i] = {color[i], P[i]};
    std::sort(tmp.begin(), tmp.end(),
              [](auto& a, auto& b) { return a.first > b.first; });

    for (int i = 0; i < np; ++i) {
        color[i] = tmp[i].first;
        order[i] = tmp[i].second;
    }
    return maxColor;
}

/* ------------------------------------------------------------------ */
/*  Recursive BK with Tomita coloring bound                           */
/* ------------------------------------------------------------------ */
void bk_tomita(const Graph& G,
               std::vector<int>& C,
               std::vector<int>& P,
               std::vector<int>& bestLocal,
               std::atomic<int>& globalBest,
               long long& nodes) {
    ++nodes;
    if (P.empty()) {
        if ((int)C.size() > (int)bestLocal.size()) bestLocal = C;
        return;
    }

    std::vector<int> color, order;
    int maxColor = greedyColor(G, P, color, order);

    int gBest = globalBest.load(std::memory_order_relaxed);
    if ((int)C.size() + maxColor <= gBest) return;
    if ((int)C.size() + maxColor <= (int)bestLocal.size()) return;

    for (int i = 0; i < (int)order.size(); ++i) {
        int v = order[i];
        int remainingColors = (i < (int)color.size()) ? color[i] : 1;
        if ((int)C.size() + remainingColors <= gBest) return;
        if ((int)C.size() + remainingColors <= (int)bestLocal.size()) return;

        std::vector<int> newP;
        newP.reserve(P.size());
        for (int u : P)
            if (u != v && G.adjacent(v, u)) newP.push_back(u);

        C.push_back(v);
        bk_tomita(G, C, newP, bestLocal, globalBest, nodes);
        C.pop_back();

        auto it = std::find(P.begin(), P.end(), v);
        if (it != P.end()) { *it = P.back(); P.pop_back(); }
    }
}

/* ------------------------------------------------------------------ */
/*  Main                                                              */
/* ------------------------------------------------------------------ */
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <num_vertices> <density_percent> [num_threads]\n"
                  << "  e.g. " << argv[0] << " 100 50.5 16\n";
        return 1;
    }

    int nVert = std::stoi(argv[1]);
    double densityPct = std::stod(argv[2]);   // FIX: double density
    int numThreads = (argc > 3) ? std::stoi(argv[3]) : omp_get_max_threads();

    Graph G = generateRandomGraph(nVert, densityPct);
    omp_set_num_threads(numThreads);

    int n = G.n;
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return G.deg[a] > G.deg[b]; });

    std::atomic<int> globalBest{0};
    long long totalNodes = 0;
    std::vector<int> globalBestClique;

    auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        std::vector<int> bestLocal;
        long long localNodes = 0;

        #pragma omp for schedule(dynamic, 1) nowait
        for (int i = 0; i < n; ++i) {
            int gBest = globalBest.load(std::memory_order_relaxed);
            if (n - i <= gBest) continue;

            int v = order[i];
            std::vector<int> P;
            for (int j = i + 1; j < n; ++j)
                if (G.adjacent(v, order[j])) P.push_back(order[j]);

            std::vector<int> C = {v};
            bk_tomita(G, C, P, bestLocal, globalBest, localNodes);

            if ((int)bestLocal.size() > gBest) {
                #pragma omp critical
                {
                    int currGlobal = globalBest.load(std::memory_order_relaxed);
                    if ((int)bestLocal.size() > currGlobal) {
                        globalBest.store((int)bestLocal.size(), std::memory_order_release);
                        globalBestClique = bestLocal;
                    }
                }
            }
        }

        #pragma omp critical
        totalNodes += localNodes;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "═══════════════════════════════════════════════════════\n"
              << " CPU-Optimized Max Clique (Tomita + Coloring)\n"
              << "═══════════════════════════════════════════════════════\n"
              << " Instance    : random n=" << G.n << " density=" << densityPct << "% (seed=42)\n"
              << " Vertices    : " << G.n << "   Edges: " << G.m << "\n"
              << " Threads     : " << numThreads << "\n"
              << " Clique size : " << globalBest.load() << "\n"
              << " Nodes       : " << totalNodes << "\n"
              << std::setprecision(6)
              << " Time        : " << elapsed << " s\n"
              << "═══════════════════════════════════════════════════════\n";

    std::cout << "CSV," << G.n << "," << G.m << "," << numThreads << ","
              << globalBest.load() << "," << elapsed << "," << totalNodes << "\n";
    return 0;
}