#!/usr/bin/env python3
"""
Generate random graphs in DIMACS edge format for Maximum Clique benchmarks.

Usage:
    python3 generate_graphs.py              # default set of instances
    python3 generate_graphs.py 100 0.5      # n=100 vertices, density=0.5
"""

import sys
import random
import os

SEED_BASE = 2025


def generate_dimacs(n: int, density: float, seed: int, filename: str) -> None:
    rng = random.Random(seed)
    edges = []
    for u in range(1, n + 1):
        for v in range(u + 1, n + 1):
            if rng.random() < density:
                edges.append((u, v))

    with open(filename, "w") as f:
        f.write(f"c Random graph: n={n}, density={density:.2f}, seed={seed}\n")
        f.write(f"p edge {n} {len(edges)}\n")
        for u, v in edges:
            f.write(f"e {u} {v}\n")

    # Estimate clique lower bound via greedy
    adj = {i: set() for i in range(1, n + 1)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    best = 1
    for start in range(1, min(n + 1, 21)):  # try 20 random starts
        rng2 = random.Random(seed + start)
        clique = [start]
        candidates = list(adj[start])
        rng2.shuffle(candidates)
        for v in candidates:
            if all(v in adj[u] for u in clique):
                clique.append(v)
        if len(clique) > best:
            best = len(clique)

    print(f"  Created {filename}  (n={n}, density={density:.2f}, "
          f"edges={len(edges)}, greedy clique≥{best})")


def main() -> None:
    out_dir = os.path.dirname(os.path.abspath(__file__))
    graphs_dir = os.path.join(out_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    if len(sys.argv) == 3:
        configs = [(int(sys.argv[1]), float(sys.argv[2]))]
    else:
        # (n, density) pairs covering a range of problem difficulties
        configs = [
            (50,  0.5),
            (50,  0.9),
            (100, 0.5),
            (100, 0.8),
            (150, 0.5),
            (150, 0.8),
            (200, 0.5),
            (200, 0.8),
        ]

    print("Generating random graph instances (DIMACS format)...")
    for n, d in configs:
        tag = f"rand_n{n}_d{int(d*100)}"
        filename = os.path.join(graphs_dir, f"{tag}.dimacs")
        generate_dimacs(n, d, seed=SEED_BASE + n + int(d * 100), filename=filename)
    print("Done.")


if __name__ == "__main__":
    main()
