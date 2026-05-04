#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <omp.h>

using u64 = uint64_t;

struct Graph {
int n = 0;
int m = 0;
int words = 0;
std::vector<std::vector<u64>> adj;
std::vector<int> deg;


Graph() = default;

explicit Graph(int n_) : n(n_), words((n_ + 63) >> 6), adj(n_, std::vector<u64>(words, 0)), deg(n_, 0) {}

bool hasEdge(int u, int v) const {
    return (adj[u][v >> 6] >> (v & 63)) & 1ULL;
}

void addEdge(int u, int v) {
    if (u == v || u < 0 || v < 0 || u >= n || v >= n) return;
    u64 mask = 1ULL << (v & 63);
    if (adj[u][v >> 6] & mask) return;
    adj[u][v >> 6] |= mask;
    adj[v][u >> 6] |= 1ULL << (u & 63);
    ++deg[u];
    ++deg[v];
    ++m;
}


};

struct BitSet {
std::vector<u64> b;


BitSet() = default;

explicit BitSet(int words) : b(words, 0) {}

void set(int i) {
    b[i >> 6] |= 1ULL << (i & 63);
}

void reset(int i) {
    b[i >> 6] &= ~(1ULL << (i & 63));
}

bool test(int i) const {
    return (b[i >> 6] >> (i & 63)) & 1ULL;
}

bool empty() const {
    for (u64 x : b) if (x) return false;
    return true;
}

int count() const {
    int s = 0;
    for (u64 x : b) s += __builtin_popcountll(x);
    return s;
}

int first() const {
    for (int i = 0; i < (int)b.size(); ++i) {
        if (b[i]) return (i << 6) + __builtin_ctzll(b[i]);
    }
    return -1;
}

void intersectWith(const BitSet& o) {
    for (int i = 0; i < (int)b.size(); ++i) b[i] &= o.b[i];
}

void subtractAdjFrom(const BitSet& adj) {
    for (int i = 0; i < (int)b.size(); ++i) b[i] &= ~adj.b[i];
}


};

struct Solver {
int n = 0;
int words = 0;
std::vector<BitSet> nbr;
std::vector<int> original;
std::atomic<int> globalBest{0};
std::vector<int> globalClique;
long long totalNodes = 0;


Solver(const Graph& g) {
    n = g.n;
    words = (n + 63) >> 6;
    std::vector<int> ord(n);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](int a, int b) {
        if (g.deg[a] != g.deg[b]) return g.deg[a] > g.deg[b];
        return a < b;
    });

    original = ord;
    std::vector<int> pos(n);
    for (int i = 0; i < n; ++i) pos[ord[i]] = i;

    nbr.assign(n, BitSet(words));
    for (int u0 = 0; u0 < n; ++u0) {
        int u = pos[u0];
        for (int w = 0; w < g.words; ++w) {
            u64 x = g.adj[u0][w];
            while (x) {
                int bit = __builtin_ctzll(x);
                int v0 = (w << 6) + bit;
                if (v0 < n) nbr[u].set(pos[v0]);
                x &= x - 1;
            }
        }
    }
}

void greedyInitial() {
    std::vector<int> clique;
    BitSet cand(words);
    for (int i = 0; i < n; ++i) cand.set(i);
    while (!cand.empty()) {
        int best = -1;
        int bestd = -1;
        for (int w = 0; w < words; ++w) {
            u64 x = cand.b[w];
            while (x) {
                int bit = __builtin_ctzll(x);
                int v = (w << 6) + bit;
                if (v < n) {
                    BitSet tmp = cand;
                    tmp.intersectWith(nbr[v]);
                    int d = tmp.count();
                    if (d > bestd) {
                        bestd = d;
                        best = v;
                    }
                }
                x &= x - 1;
            }
        }
        if (best < 0) break;
        clique.push_back(best);
        cand.intersectWith(nbr[best]);
    }
    globalClique = clique;
    globalBest.store((int)clique.size(), std::memory_order_release);
}

void colorSort(const BitSet& P, std::vector<int>& order, std::vector<int>& color) const {
    order.clear();
    color.clear();
    BitSet R = P;
    int c = 0;
    while (!R.empty()) {
        ++c;
        BitSet Q = R;
        while (!Q.empty()) {
            int v = Q.first();
            order.push_back(v);
            color.push_back(c);
            R.reset(v);
            Q.reset(v);
            Q.subtractAdjFrom(nbr[v]);
        }
    }
}

void expand(BitSet P, std::vector<int>& C, std::vector<int>& bestLocal, long long& nodes) {
    ++nodes;
    if (P.empty()) {
        if ((int)C.size() > (int)bestLocal.size()) {
            bestLocal = C;
            int s = (int)bestLocal.size();
            int cur = globalBest.load(std::memory_order_relaxed);
            while (s > cur && !globalBest.compare_exchange_weak(cur, s, std::memory_order_release, std::memory_order_relaxed)) {}
        }
        return;
    }

    std::vector<int> order;
    std::vector<int> color;
    colorSort(P, order, color);

    for (int i = (int)order.size() - 1; i >= 0; --i) {
        int bound = (int)C.size() + color[i];
        int gbest = globalBest.load(std::memory_order_relaxed);
        if (bound <= gbest || bound <= (int)bestLocal.size()) return;

        int v = order[i];
        if (!P.test(v)) continue;

        BitSet newP = P;
        newP.intersectWith(nbr[v]);

        C.push_back(v);
        expand(newP, C, bestLocal, nodes);
        C.pop_back();

        P.reset(v);
    }
}

void solve(int threads) {
    if (threads <= 0) threads = omp_get_max_threads();
    omp_set_num_threads(threads);
    greedyInitial();

    #pragma omp parallel
    {
        std::vector<int> bestLocal;
        long long localNodes = 0;

        #pragma omp for schedule(dynamic, 1) nowait
        for (int root = 0; root < n; ++root) {
            int gbest = globalBest.load(std::memory_order_relaxed);
            if (n - root <= gbest) continue;

            BitSet P(words);
            for (int j = root + 1; j < n; ++j) {
                if (nbr[root].test(j)) P.set(j);
            }

            std::vector<int> C;
            C.push_back(root);
            expand(P, C, bestLocal, localNodes);
        }

        #pragma omp critical
        {
            totalNodes += localNodes;
            if ((int)bestLocal.size() > (int)globalClique.size()) globalClique = bestLocal;
        }
    }

    if ((int)globalClique.size() < globalBest.load(std::memory_order_acquire)) {
        globalClique.clear();
    }
}

std::vector<int> cliqueOriginal() const {
    std::vector<int> r;
    r.reserve(globalClique.size());
    for (int v : globalClique) r.push_back(original[v] + 1);
    std::sort(r.begin(), r.end());
    return r;
}


};

Graph readDIMACS(const std::string& path) {
std::ifstream in(path);
if (!in) {
std::cerr << "Cannot open file: " << path << "\n";
std::exit(1);
}

std::string line;
int n = 0;
int declaredM = 0;
std::vector<std::pair<int, int>> edges;

while (std::getline(in, line)) {
    if (line.empty()) continue;
    if (line[0] == 'c') continue;
    if (line[0] == 'p') {
        std::istringstream ss(line);
        std::string p, fmt;
        ss >> p >> fmt >> n >> declaredM;
        edges.reserve(declaredM > 0 ? declaredM : 0);
    } else if (line[0] == 'e') {
        std::istringstream ss(line);
        char e;
        int u, v;
        ss >> e >> u >> v;
        edges.emplace_back(u - 1, v - 1);
    }
}

if (n <= 0) {
    std::cerr << "Invalid DIMACS file: missing problem line\n";
    std::exit(1);
}

Graph g(n);
for (auto [u, v] : edges) g.addEdge(u, v);
return g;


}

int main(int argc, char** argv) {
if (argc < 2) {
std::cerr << "Usage: " << argv[0] << " <dimacs_file> [threads]\n";
return 1;
}

std::string path = argv[1];
int threads = argc >= 3 ? std::atoi(argv[2]) : omp_get_max_threads();

Graph g = readDIMACS(path);
Solver solver(g);

auto t0 = std::chrono::high_resolution_clock::now();
solver.solve(threads);
auto t1 = std::chrono::high_resolution_clock::now();

double secs = std::chrono::duration<double>(t1 - t0).count();
int omega = solver.globalBest.load(std::memory_order_acquire);
std::vector<int> clique = solver.cliqueOriginal();

std::cout << "Instance " << path << "\n";
std::cout << "Vertices " << g.n << "\n";
std::cout << "Edges " << g.m << "\n";
std::cout << "Threads " << threads << "\n";
std::cout << "MaximumCliqueSize " << omega << "\n";
std::cout << "Nodes " << solver.totalNodes << "\n";
std::cout << std::fixed << std::setprecision(6) << "Time " << secs << "\n";
std::cout << "Clique";
for (int v : clique) std::cout << " " << v;
std::cout << "\n";
std::cout << "CSV," << g.n << "," << g.m << "," << threads << "," << omega << "," << std::setprecision(6) << secs << "," << solver.totalNodes << "\n";

return 0;


}
