#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <string>
#include <iomanip>
#include <numeric>
#include <atomic>
#include <random>
#include <fstream>
#include <sstream>
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

Graph generateRandomGraph(int n, double density, unsigned seed = 42) {
    Graph G(n);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int u = 0; u < n; ++u)
        for (int v = u + 1; v < n; ++v)
            if (dist(rng) < density)
                G.addEdge(u, v);
    return G;
}

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
            char ch;
            int u, v;
            iss >> ch >> u >> v;
            G.addEdge(u - 1, v - 1);
        }
    }
    return G;
}

int greedyColoring(const Graph& G, const std::vector<int>& P,
                   std::vector<int>& colorOrder) {
    int np = (int)P.size();
    std::vector<int> color(np, 0);
    int maxColor = 0;

    std::vector<char> usedColors(np + 2, 0);

    for (int i = 0; i < np; ++i) {
        for (int j = 0; j < i; ++j) {
            if (G.adjacent(P[i], P[j]))
                usedColors[color[j]] = 1;
        }
        int c = 1;
        while (usedColors[c]) ++c;
        color[i] = c;
        if (c > maxColor) maxColor = c;

        for (int j = 0; j < i; ++j) {
            if (G.adjacent(P[i], P[j]))
                usedColors[color[j]] = 0;
        }
    }

    std::vector<std::pair<int,int>> colorVertex(np);
    for (int i = 0; i < np; ++i)
        colorVertex[i] = {color[i], P[i]};
    std::sort(colorVertex.rbegin(), colorVertex.rend());

    colorOrder.resize(np);
    for (int i = 0; i < np; ++i)
        colorOrder[i] = colorVertex[i].second;

    return maxColor;
}

int selectPivot(const Graph& G, const std::vector<int>& P) {
    int bestV = P[0], bestConn = -1;
    int words = G.words;

    std::vector<uint64_t> pMask(words, 0ULL);
    for (int v : P)
        pMask[v / WORD_BITS] |= (1ULL << (v % WORD_BITS));

    for (int v : P) {
        int conn = 0;
        for (int w = 0; w < words; ++w)
            conn += popcount64(G.adjBits[v][w] & pMask[w]);
        if (conn > bestConn) { bestConn = conn; bestV = v; }
    }
    return bestV;
}

static void bk_opt(const Graph& G,
                   std::vector<int>& clique,
                   std::vector<int>& P,
                   std::vector<int>& best,
                   const std::atomic<int>& globalBestSz,
                   long long& nodes) {
    ++nodes;

    if (P.empty()) {
        if ((int)clique.size() > (int)best.size())
            best = clique;
        return;
    }

    std::vector<int> colorOrder;
    int colorBound = greedyColoring(G, P, colorOrder);

    int gSz = globalBestSz.load(std::memory_order_relaxed);
    if ((int)clique.size() + colorBound <= (int)best.size()) return;
    if ((int)clique.size() + colorBound <= gSz) return;

    int pivot = selectPivot(G, P);

    for (int idx = 0; idx < (int)colorOrder.size(); ++idx) {
        int remaining = (int)colorOrder.size() - idx;
        gSz = globalBestSz.load(std::memory_order_relaxed);
        if ((int)clique.size() + remaining <= (int)best.size()) return;
        if ((int)clique.size() + remaining <= gSz) return;

        int v = colorOrder[idx];

        if (G.adjacent(v, pivot) && v != pivot) continue;

        std::vector<int> newP;
        newP.reserve(P.size());
        for (int u : P)
            if (G.adjacent(v, u)) newP.push_back(u);

        clique.push_back(v);
        bk_opt(G, clique, newP, best, globalBestSz, nodes);
        clique.pop_back();

        auto it = std::find(P.begin(), P.end(), v);
        if (it != P.end()) {
            std::swap(*it, P.back());
            P.pop_back();
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage:\n  " << argv[0]
                  << " random <num_vertices> <density> [num_threads]\n  "
                  << argv[0] << " file   <dimacs_file> [num_threads]\n";
        return 1;
    }

    Graph G(1);
    std::string instanceLine;
    int numThreads = 0;

    const std::string mode = argv[1];
    if (mode == "random") {
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0] << " random <num_vertices> <density> [num_threads]\n";
            return 1;
        }
        int nVert = std::stoi(argv[2]);
        double density = std::stod(argv[3]);
        G = generateRandomGraph(nVert, density);
        instanceLine = "random n=" + std::to_string(G.n) + " density=" + std::to_string(density) + " (seed=42)";
        numThreads = (argc > 4) ? std::stoi(argv[4]) : omp_get_max_threads();
    } else if (mode == "file") {
        G = readDIMACS(argv[2]);
        instanceLine = std::string(argv[2]);
        numThreads = (argc > 3) ? std::stoi(argv[3]) : omp_get_max_threads();
    } else {
        std::cerr << "[ERROR] First argument must be 'random' or 'file'.\n";
        return 1;
    }

    if (numThreads > omp_get_max_threads())
        numThreads = omp_get_max_threads();
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
            bk_opt(G, clique, P, localBest, globalBestSz, localNodes);
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
              << " Instance    : " << instanceLine << "\n"
              << " Vertices    : " << G.n << "    Edges: " << G.m << "\n"
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