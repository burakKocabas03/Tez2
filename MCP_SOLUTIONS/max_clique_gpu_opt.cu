%%writefile mxcl.cu
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifndef MAX_LOCAL_CAND
#define MAX_LOCAL_CAND 512
#endif

static constexpr int THREADS = 256;
static constexpr int LOCAL_WORDS = (MAX_LOCAL_CAND + 31) / 32;

struct CSRGraph {
    int n = 0;
    long long m = 0;
    std::vector<int> row;
    std::vector<int> col;
    std::vector<int> deg;
};

static void die(const std::string& s) {
    std::cerr << s << "\n";
    std::exit(1);
}

static void cuda_check(cudaError_t e) {
    if (e != cudaSuccess) die(cudaGetErrorString(e));
}

static bool is_number_token(const std::string& s) {
    if (s.empty()) return false;
    size_t i = 0;
    if (s[0] == '-' || s[0] == '+') i = 1;
    if (i >= s.size()) return false;
    for (; i < s.size(); ++i) {
        if (s[i] < '0' || s[i] > '9') return false;
    }
    return true;
}

static CSRGraph read_graph_dimacs(const std::string& path) {
    std::ifstream in(path);
    if (!in) die("cannot open input file");

    int n = 0;
    long long m_decl = 0;
    std::string line;

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        size_t p = 0;
        while (p < line.size() && (line[p] == ' ' || line[p] == '\t' || line[p] == '\r')) ++p;
        if (p >= line.size()) continue;
        if (line[p] == 'c') continue;
        if (line[p] == 'p') {
            std::istringstream ss(line);
            std::string ptag, fmt;
            ss >> ptag >> fmt >> n >> m_decl;
            break;
        }
    }

    if (n <= 0) die("invalid dimacs header");

    std::vector<int> deg(n, 0);
    long long edge_lines = 0;

    in.clear();
    in.seekg(0, std::ios::beg);

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        size_t p = 0;
        while (p < line.size() && (line[p] == ' ' || line[p] == '\t' || line[p] == '\r')) ++p;
        if (p >= line.size()) continue;
        char c = line[p];
        if (c == 'c' || c == 'p') continue;
        if (c == 'e' || c == 'a') {
            std::istringstream ss(line);
            char etag;
            long long uu, vv;
            ss >> etag >> uu >> vv;
            if (!ss) continue;
            int u = (int)(uu - 1);
            int v = (int)(vv - 1);
            if (u < 0 || v < 0 || u >= n || v >= n || u == v) continue;
            ++deg[u];
            ++deg[v];
            ++edge_lines;
        }
    }

    std::vector<int> row(n + 1, 0);
    long long total = 0;
    for (int i = 0; i < n; ++i) {
        total += deg[i];
        if (total > (long long)std::numeric_limits<int>::max()) die("too many adjacency entries for int CSR");
        row[i + 1] = (int)total;
    }

    std::vector<int> col(row[n], 0);
    std::vector<int> cur = row;

    in.clear();
    in.seekg(0, std::ios::beg);

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        size_t p = 0;
        while (p < line.size() && (line[p] == ' ' || line[p] == '\t' || line[p] == '\r')) ++p;
        if (p >= line.size()) continue;
        char c = line[p];
        if (c == 'c' || c == 'p') continue;
        if (c == 'e' || c == 'a') {
            std::istringstream ss(line);
            char etag;
            long long uu, vv;
            ss >> etag >> uu >> vv;
            if (!ss) continue;
            int u = (int)(uu - 1);
            int v = (int)(vv - 1);
            if (u < 0 || v < 0 || u >= n || v >= n || u == v) continue;
            col[cur[u]++] = v;
            col[cur[v]++] = u;
        }
    }

    std::vector<int> new_row(n + 1, 0);
    std::vector<int> new_col;
    new_col.reserve(col.size());

    for (int v = 0; v < n; ++v) {
        int l = row[v];
        int r = row[v + 1];
        std::sort(col.begin() + l, col.begin() + r);

        new_row[v] = (int)new_col.size();
        int last = -1;
        for (int i = l; i < r; ++i) {
            int u = col[i];
            if (u != last) {
                new_col.push_back(u);
                last = u;
            }
        }
    }
    new_row[n] = (int)new_col.size();

    CSRGraph g;
    g.n = n;
    g.row.swap(new_row);
    g.col.swap(new_col);
    g.deg.assign(n, 0);
    for (int v = 0; v < n; ++v) g.deg[v] = g.row[v + 1] - g.row[v];
    g.m = (long long)g.col.size() / 2;
    return g;
}

static CSRGraph read_graph_edges_csv(const std::string& path) {
    std::ifstream in(path);
    if (!in) die("cannot open input file");

    std::string line;
    long long declared_n = 0;
    long long max_id = std::numeric_limits<long long>::min();
    long long min_id = std::numeric_limits<long long>::max();

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        size_t p = 0;
        while (p < line.size() && (line[p] == ' ' || line[p] == '\t' || line[p] == '\r')) ++p;
        if (p >= line.size()) continue;

        if (line[p] == '%' || line[p] == '#') {
            std::string s = line.substr(p + 1);
            for (char& c : s) if (c == ',' || c == '\t') c = ' ';
            std::istringstream hs(s);
            long long a, b, c;
            if (hs >> a) {
                declared_n = std::max(declared_n, a);
                if (hs >> b) declared_n = std::max(declared_n, b);
                if (hs >> c) declared_n = std::max(declared_n, c);
            }
            continue;
        }

        std::string s = line.substr(p);
        for (char& c : s) if (c == ',' || c == ';' || c == '\t') c = ' ';
        std::istringstream ss(s);
        long long u, v;
        if (!(ss >> u >> v)) continue;
        if (u < 0 || v < 0) continue;
        if (u == v) continue;
        min_id = std::min(min_id, std::min(u, v));
        max_id = std::max(max_id, std::max(u, v));
    }

    if (max_id == std::numeric_limits<long long>::min()) die("no edges found");

    bool zero_based = (min_id == 0);
    long long n_from_edges = zero_based ? (max_id + 1) : max_id;
    long long n_ll = std::max(declared_n, n_from_edges);
    if (n_ll <= 0 || n_ll > std::numeric_limits<int>::max()) die("vertex count out of int range");
    int n = (int)n_ll;

    std::vector<int> deg(n, 0);

    in.clear();
    in.seekg(0, std::ios::beg);

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        size_t p = 0;
        while (p < line.size() && (line[p] == ' ' || line[p] == '\t' || line[p] == '\r')) ++p;
        if (p >= line.size()) continue;
        if (line[p] == '%' || line[p] == '#') continue;

        std::string s = line.substr(p);
        for (char& c : s) if (c == ',' || c == ';' || c == '\t') c = ' ';
        std::istringstream ss(s);
        long long uu, vv;
        if (!(ss >> uu >> vv)) continue;

        int u = zero_based ? (int)uu : (int)(uu - 1);
        int v = zero_based ? (int)vv : (int)(vv - 1);

        if (u < 0 || v < 0 || u >= n || v >= n || u == v) continue;

        ++deg[u];
        ++deg[v];
    }

    std::vector<int> row(n + 1, 0);
    long long total = 0;
    for (int i = 0; i < n; ++i) {
        total += deg[i];
        if (total > (long long)std::numeric_limits<int>::max()) die("too many adjacency entries for int CSR");
        row[i + 1] = (int)total;
    }

    std::vector<int> col(row[n], 0);
    std::vector<int> cur = row;

    in.clear();
    in.seekg(0, std::ios::beg);

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        size_t p = 0;
        while (p < line.size() && (line[p] == ' ' || line[p] == '\t' || line[p] == '\r')) ++p;
        if (p >= line.size()) continue;
        if (line[p] == '%' || line[p] == '#') continue;

        std::string s = line.substr(p);
        for (char& c : s) if (c == ',' || c == ';' || c == '\t') c = ' ';
        std::istringstream ss(s);
        long long uu, vv;
        if (!(ss >> uu >> vv)) continue;

        int u = zero_based ? (int)uu : (int)(uu - 1);
        int v = zero_based ? (int)vv : (int)(vv - 1);

        if (u < 0 || v < 0 || u >= n || v >= n || u == v) continue;

        col[cur[u]++] = v;
        col[cur[v]++] = u;
    }

    std::vector<int> new_row(n + 1, 0);
    std::vector<int> new_col;
    new_col.reserve(col.size());

    for (int v = 0; v < n; ++v) {
        int l = row[v];
        int r = row[v + 1];
        std::sort(col.begin() + l, col.begin() + r);

        new_row[v] = (int)new_col.size();
        int last = -1;
        for (int i = l; i < r; ++i) {
            int u = col[i];
            if (u != last) {
                new_col.push_back(u);
                last = u;
            }
        }
    }
    new_row[n] = (int)new_col.size();

    CSRGraph g;
    g.n = n;
    g.row.swap(new_row);
    g.col.swap(new_col);
    g.deg.assign(n, 0);
    for (int v = 0; v < n; ++v) g.deg[v] = g.row[v + 1] - g.row[v];
    g.m = (long long)g.col.size() / 2;
    return g;
}

static CSRGraph read_graph(const std::string& path) {
    std::ifstream in(path);
    if (!in) die("cannot open input file");

    std::string line;
    bool dimacs = false;
    bool csv = false;

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        size_t p = 0;
        while (p < line.size() && (line[p] == ' ' || line[p] == '\t' || line[p] == '\r')) ++p;
        if (p >= line.size()) continue;
        char c = line[p];
        if (c == 'c') continue;
        if (c == 'p' || c == 'e' || c == 'a') dimacs = true;
        if (line.find(',') != std::string::npos) csv = true;
        break;
    }

    if (dimacs) return read_graph_dimacs(path);
    if (csv) return read_graph_edges_csv(path);
    return read_graph_dimacs(path);
}

static bool has_edge_cpu(const CSRGraph& g, int u, int v) {
    return std::binary_search(g.col.begin() + g.row[u], g.col.begin() + g.row[u + 1], v);
}

static std::vector<int> core_decomposition(const CSRGraph& g) {
    int n = g.n;
    std::vector<int> deg = g.deg;
    int max_deg = 0;

    for (int d : deg) max_deg = std::max(max_deg, d);

    std::vector<int> bin(max_deg + 1, 0);
    std::vector<int> pos(n);
    std::vector<int> vert(n);

    for (int d : deg) ++bin[d];

    int start = 0;

    for (int d = 0; d <= max_deg; ++d) {
        int num = bin[d];
        bin[d] = start;
        start += num;
    }

    for (int v = 0; v < n; ++v) {
        pos[v] = bin[deg[v]];
        vert[pos[v]] = v;
        ++bin[deg[v]];
    }

    for (int d = max_deg; d >= 1; --d) bin[d] = bin[d - 1];
    bin[0] = 0;

    for (int i = 0; i < n; ++i) {
        int v = vert[i];

        for (int e = g.row[v]; e < g.row[v + 1]; ++e) {
            int u = g.col[e];

            if (deg[u] > deg[v]) {
                int du = deg[u];
                int pu = pos[u];
                int pw = bin[du];
                int w = vert[pw];

                if (u != w) {
                    vert[pu] = w;
                    pos[w] = pu;
                    vert[pw] = u;
                    pos[u] = pw;
                }

                ++bin[du];
                --deg[u];
            }
        }
    }

    return deg;
}

static int greedy_initial_clique(const CSRGraph& g, const std::vector<int>& order, const std::vector<int>& core) {
    int n = g.n;

    if (n == 0) return 0;
    if (g.m == 0) return 1;

    int best = 2;
    int tries = std::min(n, 8192);

    std::vector<int> clique;
    std::vector<int> cand;

    for (int t = 0; t < tries; ++t) {
        int root = order[t];

        clique.clear();
        cand.clear();

        clique.push_back(root);

        for (int e = g.row[root]; e < g.row[root + 1]; ++e) cand.push_back(g.col[e]);

        std::sort(cand.begin(), cand.end(), [&](int a, int b) {
            if (core[a] != core[b]) return core[a] > core[b];
            if (g.deg[a] != g.deg[b]) return g.deg[a] > g.deg[b];
            return a < b;
        });

        for (int v : cand) {
            bool ok = true;

            for (int u : clique) {
                if (!has_edge_cpu(g, u, v)) {
                    ok = false;
                    break;
                }
            }

            if (ok) clique.push_back(v);
        }

        best = std::max(best, (int)clique.size());
    }

    return best;
}

static void greedy_color_cpu(const CSRGraph& g, const std::vector<int>& P, std::vector<int>& order, std::vector<int>& color) {
    std::vector<int> U = P;
    std::vector<int> next;
    std::vector<int> cls;

    order.clear();
    color.clear();

    int c = 0;

    while (!U.empty()) {
        ++c;
        next.clear();
        cls.clear();

        for (int v : U) {
            bool ok = true;

            for (int u : cls) {
                if (has_edge_cpu(g, u, v)) {
                    ok = false;
                    break;
                }
            }

            if (ok) {
                cls.push_back(v);
                order.push_back(v);
                color.push_back(c);
            } else {
                next.push_back(v);
            }
        }

        U.swap(next);
    }
}

static void cpu_expand(const CSRGraph& g, const std::vector<int>& P, int csize, int& best) {
    if (P.empty()) {
        if (csize > best) best = csize;
        return;
    }

    if (csize + (int)P.size() <= best) return;

    std::vector<int> order;
    std::vector<int> color;

    greedy_color_cpu(g, P, order, color);

    for (int i = (int)order.size() - 1; i >= 0; --i) {
        if (csize + color[i] <= best) return;

        int v = order[i];
        std::vector<int> newP;
        newP.reserve(i);

        for (int j = 0; j < i; ++j) {
            int u = order[j];

            if (has_edge_cpu(g, v, u)) newP.push_back(u);
        }

        cpu_expand(g, newP, csize + 1, best);
    }
}

__device__ int d_best_size;

__device__ __forceinline__ int bit_count_words(const uint32_t* bits) {
    int s = 0;

    for (int i = 0; i < LOCAL_WORDS; ++i) {
        s += __popc(bits[i]);
    }

    return s;
}

__device__ __forceinline__ bool bit_empty_words(const uint32_t* bits) {
    for (int i = 0; i < LOCAL_WORDS; ++i) {
        if (bits[i]) return false;
    }

    return true;
}

__device__ __forceinline__ int first_bit_words(const uint32_t* bits) {
    for (int i = 0; i < LOCAL_WORDS; ++i) {
        uint32_t x = bits[i];

        if (x) return i * 32 + __ffs(x) - 1;
    }

    return -1;
}

__device__ __forceinline__ void clear_bit_word(uint32_t* bits, int v) {
    bits[v >> 5] &= ~(1u << (v & 31));
}

__device__ __forceinline__ void set_bit_word(uint32_t* bits, int v) {
    bits[v >> 5] |= 1u << (v & 31);
}

__device__ int greedy_color_bound_gpu(const uint32_t* P, const uint32_t* adj, int local_n) {
    uint32_t rem[LOCAL_WORDS];
    uint32_t avail[LOCAL_WORDS];

    for (int i = 0; i < LOCAL_WORDS; ++i) rem[i] = P[i];

    int colors = 0;

    while (!bit_empty_words(rem)) {
        ++colors;

        for (int i = 0; i < LOCAL_WORDS; ++i) avail[i] = rem[i];

        while (!bit_empty_words(avail)) {
            int v = first_bit_words(avail);

            if (v < 0 || v >= local_n) break;

            clear_bit_word(rem, v);
            clear_bit_word(avail, v);

            const uint32_t* row = adj + v * LOCAL_WORDS;

            for (int w = 0; w < LOCAL_WORDS; ++w) {
                avail[w] &= ~row[w];
            }
        }
    }

    return colors;
}

__device__ int select_branch_vertex_gpu(const uint32_t* P, const uint32_t* adj, int local_n) {
    int best_v = -1;
    int best_d = -1;

    for (int w = 0; w < LOCAL_WORDS; ++w) {
        uint32_t x = P[w];

        while (x) {
            int b = __ffs(x) - 1;
            int v = w * 32 + b;

            if (v < local_n) {
                int d = 0;
                const uint32_t* row = adj + v * LOCAL_WORDS;

                for (int k = 0; k < LOCAL_WORDS; ++k) {
                    d += __popc(P[k] & row[k]);
                }

                if (d > best_d) {
                    best_d = d;
                    best_v = v;
                }
            }

            x &= x - 1;
        }
    }

    return best_v;
}

__device__ __noinline__ void dfs_gpu(uint32_t* P, const uint32_t* adj, int local_n, int csize, int* local_best) {
    if (csize > *local_best) {
        *local_best = csize;
        atomicMax(&d_best_size, csize);
    }

    while (!bit_empty_words(P)) {
        int best_now = d_best_size;

        if (*local_best > best_now) best_now = *local_best;

        int cnt = bit_count_words(P);

        if (csize + cnt <= best_now) return;

        int cb = greedy_color_bound_gpu(P, adj, local_n);

        if (csize + cb <= best_now) return;

        int v = select_branch_vertex_gpu(P, adj, local_n);

        if (v < 0) return;

        clear_bit_word(P, v);

        uint32_t newP[LOCAL_WORDS];
        const uint32_t* row = adj + v * LOCAL_WORDS;

        for (int w = 0; w < LOCAL_WORDS; ++w) {
            newP[w] = P[w] & row[w];
        }

        dfs_gpu(newP, adj, local_n, csize + 1, local_best);
    }
}

__device__ bool has_edge_dev(int u, int v, const int* row, const int* col) {
    int l = row[u];
    int r = row[u + 1] - 1;

    while (l <= r) {
        int m = (l + r) >> 1;
        int x = col[m];

        if (x == v) return true;
        if (x < v) l = m + 1;
        else r = m - 1;
    }

    return false;
}

__global__ void maxclique_kernel(
    int n,
    const int* __restrict__ row,
    const int* __restrict__ col,
    const int* __restrict__ order,
    const int* __restrict__ rank,
    const int* __restrict__ core,
    int* __restrict__ next_task,
    unsigned char* __restrict__ overflow
) {
    __shared__ int task_shared;
    __shared__ int cand[MAX_LOCAL_CAND];
    __shared__ int cand_count;
    __shared__ int overflow_flag;
    __shared__ uint32_t adj[MAX_LOCAL_CAND * LOCAL_WORDS];

    while (true) {
        if (threadIdx.x == 0) {
            task_shared = atomicAdd(next_task, 1);
            cand_count = 0;
            overflow_flag = 0;
        }

        __syncthreads();

        int task = task_shared;

        if (task >= n) break;

        int root = order[task];
        int best_now = d_best_size;
        int active = core[root] + 1 > best_now;

        if (active) {
            for (int e = row[root] + threadIdx.x; e < row[root + 1]; e += blockDim.x) {
                int u = col[e];

                if (rank[u] > task && core[u] + 1 > best_now) {
                    int p = atomicAdd(&cand_count, 1);

                    if (p < MAX_LOCAL_CAND) {
                        cand[p] = u;
                    } else {
                        overflow_flag = 1;
                    }
                }
            }
        }

        __syncthreads();

        if (active && overflow_flag) {
            if (threadIdx.x == 0) overflow[task] = 1;
            __syncthreads();
            continue;
        }

        int local_n = cand_count;

        if (active && local_n + 1 > d_best_size) {
            for (int i = threadIdx.x; i < local_n * LOCAL_WORDS; i += blockDim.x) {
                adj[i] = 0;
            }

            __syncthreads();

            long long pairs = (long long)local_n * local_n;

            for (long long p = threadIdx.x; p < pairs; p += blockDim.x) {
                int i = (int)(p / local_n);
                int j = (int)(p - (long long)i * local_n);

                if (i != j) {
                    int u = cand[i];
                    int v = cand[j];

                    if (has_edge_dev(u, v, row, col)) {
                        atomicOr(adj + i * LOCAL_WORDS + (j >> 5), 1u << (j & 31));
                    }
                }
            }

            __syncthreads();

            if (threadIdx.x == 0) {
                if (local_n == 0) {
                    atomicMax(&d_best_size, 1);
                } else {
                    uint32_t P[LOCAL_WORDS];

                    for (int w = 0; w < LOCAL_WORDS; ++w) P[w] = 0;

                    for (int i = 0; i < local_n; ++i) set_bit_word(P, i);

                    int local_best = d_best_size;

                    dfs_gpu(P, adj, local_n, 1, &local_best);
                }
            }
        }

        __syncthreads();
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " graph.txt\n";
        return 1;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    CSRGraph g = read_graph(argv[1]);

    std::vector<int> core = core_decomposition(g);

    int degeneracy = 0;

    for (int c : core) degeneracy = std::max(degeneracy, c);

    std::vector<int> order(g.n);
    std::iota(order.begin(), order.end(), 0);

    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (core[a] != core[b]) return core[a] > core[b];
        if (g.deg[a] != g.deg[b]) return g.deg[a] > g.deg[b];
        return a < b;
    });

    std::vector<int> rank(g.n);

    for (int i = 0; i < g.n; ++i) rank[order[i]] = i;

    int initial = greedy_initial_clique(g, order, core);

    int device = 0;
    cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop, device));
    cuda_check(cudaSetDevice(device));
    cudaDeviceSetLimit(cudaLimitStackSize, 131072);

    int* d_row = nullptr;
    int* d_col = nullptr;
    int* d_order = nullptr;
    int* d_rank = nullptr;
    int* d_core = nullptr;
    int* d_next_task = nullptr;
    unsigned char* d_overflow = nullptr;

    cuda_check(cudaMalloc(&d_row, (size_t)(g.n + 1) * sizeof(int)));
    cuda_check(cudaMalloc(&d_col, (size_t)g.col.size() * sizeof(int)));
    cuda_check(cudaMalloc(&d_order, (size_t)g.n * sizeof(int)));
    cuda_check(cudaMalloc(&d_rank, (size_t)g.n * sizeof(int)));
    cuda_check(cudaMalloc(&d_core, (size_t)g.n * sizeof(int)));
    cuda_check(cudaMalloc(&d_next_task, sizeof(int)));
    cuda_check(cudaMalloc(&d_overflow, (size_t)g.n * sizeof(unsigned char)));

    cuda_check(cudaMemcpy(d_row, g.row.data(), (size_t)(g.n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(d_col, g.col.data(), (size_t)g.col.size() * sizeof(int), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(d_order, order.data(), (size_t)g.n * sizeof(int), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(d_rank, rank.data(), (size_t)g.n * sizeof(int), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(d_core, core.data(), (size_t)g.n * sizeof(int), cudaMemcpyHostToDevice));
    cuda_check(cudaMemset(d_overflow, 0, (size_t)g.n * sizeof(unsigned char)));
    cuda_check(cudaMemset(d_next_task, 0, sizeof(int)));
    cuda_check(cudaMemcpyToSymbol(d_best_size, &initial, sizeof(int)));

    int blocks = std::max(1, prop.multiProcessorCount * 16);

    if (g.n > 0) {
        maxclique_kernel<<<blocks, THREADS>>>(g.n, d_row, d_col, d_order, d_rank, d_core, d_next_task, d_overflow);
        cuda_check(cudaGetLastError());
        cuda_check(cudaDeviceSynchronize());
    }

    int best = initial;
    cuda_check(cudaMemcpyFromSymbol(&best, d_best_size, sizeof(int)));

    std::vector<unsigned char> overflow(g.n, 0);
    cuda_check(cudaMemcpy(overflow.data(), d_overflow, (size_t)g.n * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    int overflow_roots = 0;

    for (int task = 0; task < g.n; ++task) {
        if (!overflow[task]) continue;

        ++overflow_roots;

        int root = order[task];

        if (core[root] + 1 <= best) continue;

        std::vector<int> P;

        for (int e = g.row[root]; e < g.row[root + 1]; ++e) {
            int u = g.col[e];

            if (rank[u] > task && core[u] + 1 > best) P.push_back(u);
        }

        std::sort(P.begin(), P.end(), [&](int a, int b) {
            if (core[a] != core[b]) return core[a] > core[b];
            if (g.deg[a] != g.deg[b]) return g.deg[a] > g.deg[b];
            return a < b;
        });

        cpu_expand(g, P, 1, best);
    }

    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_order);
    cudaFree(d_rank);
    cudaFree(d_core);
    cudaFree(d_next_task);
    cudaFree(d_overflow);

    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "Vertices " << g.n << "\n";
    std::cout << "Edges " << g.m << "\n";
    std::cout << "Degeneracy " << degeneracy << "\n";
    std::cout << "MaxLocalCandidates " << MAX_LOCAL_CAND << "\n";
    std::cout << "InitialClique " << initial << "\n";
    std::cout << "MaxCliqueSize " << best << "\n";
    std::cout << "OverflowRoots " << overflow_roots << "\n";
    std::cout << "Device " << prop.name << "\n";
    std::cout << std::fixed << std::setprecision(6) << "Time " << sec << "\n";
    std::cout << "CSV," << g.n << "," << g.m << "," << degeneracy << "," << MAX_LOCAL_CAND << "," << initial << "," << best << "," << overflow_roots << "," << std::setprecision(6) << sec << "\n";

    return 0;
}