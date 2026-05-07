/**
 * WP1 – Baseline Serial Implementation
 * ======================================
 * Problem  : Maximum Clique Problem (MCP)
 * Algorithm: Branch-and-Bound Bron-Kerbosch with vertex ordering and
 *            cardinality-based upper bound pruning
 *
 * Given an undirected graph G = (V, E), find the largest subset of vertices
 * such that every pair of vertices in the subset is connected by an edge.
 * MCP is NP-hard; this exact solver uses Bron-Kerbosch with pruning.
 *
 * Algorithm details
 * -----------------
 *  1. Sort vertices by degree (descending) — high-degree vertices are more
 *     likely to be in a large clique, so processing them first improves pruning.
 *  2. For each vertex v_i in the ordering, find the maximum clique that
 *     includes v_i but no earlier vertex (independent sub-problems).
 *  3. Within each sub-problem, recurse with upper-bound pruning:
 *     |current_clique| + |candidates| <= |best| → prune branch.
 *
 * Input format: DIMACS edge format
 *   p edge N M
 *   e u v          (1-indexed, undirected)
 *
 * Thesis   : A Comparison of NP-hard Problems through Parallel Algorithms
 *            Utilizing CPU and GPU-Based Solutions
 * Authors  : Burak Kocabaş & Emin Özgür Elmalı
 * Advisor  : Prof. Dr. Hasan BULUT
 *
 * Build    : g++ -O3 -std=c++17 -o max_clique_serial max_clique_serial.cpp
 * Run      : ./max_clique_serial <graph_file>
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

// ---------------------------------------------------------------------------
//  Graph representation
// ---------------------------------------------------------------------------

struct Graph {
    int n = 0, m = 0;
    std::vector<std::vector<bool>> adj;   // O(n²) adjacency matrix for fast lookup
    std::vector<int>               deg;   // degree of each vertex

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
//  Branch-and-Bound Bron-Kerbosch kernel
// ---------------------------------------------------------------------------

/**
 * Recursive B&B search.
 *
 * @param G       the graph (read-only)
 * @param clique  current clique being built (modified in place, restored on return)
 * @param P       candidate vertices (ordered; only vertices at index > current
 *                are considered to avoid duplicates)
 * @param best    best clique found so far by this search thread
 * @param nodes   counter of recursive calls (for analysis)
 */
static void bk(const Graph&             G,
               std::vector<int>&         clique,
               const std::vector<int>&   P,
               std::vector<int>&         best,
               long long&                nodes)
{
    ++nodes;

    if (P.empty()) {
        if (clique.size() > best.size())
            best = clique;
        return;
    }

    // Upper bound: even adding all remaining candidates can't beat best
    if (clique.size() + P.size() <= best.size())
        return;

    for (int i = 0; i < static_cast<int>(P.size()); ++i) {
        // Tighter incremental bound
        if (clique.size() + (P.size() - i) <= best.size())
            return;

        const int v = P[i];

        // New candidates: P[i+1 ..] that are adjacent to v
        std::vector<int> newP;
        newP.reserve(P.size() - i - 1);
        for (int j = i + 1; j < static_cast<int>(P.size()); ++j)
            if (G.adj[v][P[j]])
                newP.push_back(P[j]);

        clique.push_back(v);
        bk(G, clique, newP, best, nodes);
        clique.pop_back();
    }
}

// ---------------------------------------------------------------------------
//  Top-level solver
// ---------------------------------------------------------------------------

struct MCPResult {
    std::vector<int> clique;
    long long        nodesExplored;
};

MCPResult solveMCP(const Graph& G) {
    const int n = G.n;

    // Vertex ordering: descending degree
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return G.deg[a] > G.deg[b]; });

    std::vector<int> best;
    long long        nodes = 0;

    for (int i = 0; i < n; ++i) {
        // Early termination: remaining candidates cannot beat current best
        if (static_cast<int>(best.size()) >= n - i)
            break;

        const int v = order[i];

        // Candidates: order[i+1 ..] that are adjacent to v
        std::vector<int> P;
        P.reserve(n - i - 1);
        for (int j = i + 1; j < n; ++j)
            if (G.adj[v][order[j]])
                P.push_back(order[j]);

        std::vector<int> clique = {v};
        bk(G, clique, P, best, nodes);
    }

    return {best, nodes};
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
            std::string tmp1, tmp2;
            iss >> tmp1 >> tmp2 >> n >> m;
            break;
        }
    }

    Graph G(n);

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == 'c') continue;
        if (line[0] == 'e') {
            std::istringstream iss(line);
            char   ch;
            int    u, v;
            iss >> ch >> u >> v;
            G.addEdge(u - 1, v - 1);   // convert to 0-indexed
        }
    }
    return G;
}

// ---------------------------------------------------------------------------
//  Main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <dimacs_graph_file>\n";
        return EXIT_FAILURE;
    }

    const Graph G = readDIMACS(argv[1]);

    std::cout << "═══════════════════════════════════════════════════════\n";
    std::cout << " WP1 – Maximum Clique Problem (Serial)\n";
    std::cout << "═══════════════════════════════════════════════════════\n";
    std::cout << " Instance    : " << argv[1] << "\n";
    std::cout << " Vertices    : " << G.n << "\n";
    std::cout << " Edges       : " << G.m << "\n";
    std::cout << " Density     : " << std::fixed << std::setprecision(4)
              << (G.n > 1 ? (2.0 * G.m) / ((double)G.n * (G.n - 1)) : 0.0)
              << "\n";
    std::cout << "───────────────────────────────────────────────────────\n";

    const auto t0  = std::chrono::high_resolution_clock::now();
    const auto res = solveMCP(G);
    const auto t1  = std::chrono::high_resolution_clock::now();

    const double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << " Maximum clique size : " << res.clique.size() << "\n";
    std::cout << " Clique vertices     : ";
    for (int v : res.clique) std::cout << v + 1 << " ";    // back to 1-indexed
    std::cout << "\n";
    std::cout << " Nodes explored      : " << res.nodesExplored << "\n";
    std::cout << std::setprecision(6);
    std::cout << " Execution time      : " << elapsed << " s\n";
    std::cout << "═══════════════════════════════════════════════════════\n";

    // CSV: n, m, clique_size, time_s, nodes
    std::cout << "CSV," << G.n << "," << G.m << ","
              << res.clique.size() << "," << elapsed << "," << res.nodesExplored
              << "\n";

    return EXIT_SUCCESS;
}
