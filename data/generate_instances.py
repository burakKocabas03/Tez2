#!/usr/bin/env python3
"""
Generate random TSP instances in TSPLIB EUC_2D format.

Usage:
    python3 generate_instances.py              # creates all default sizes
    python3 generate_instances.py 200 500 1000 # creates specific sizes
"""

import sys
import random
import os
import math

GRID_SIZE = 10_000   # coordinate range [0, GRID_SIZE]
SEED_BASE = 2025


def generate_tsp(n: int, seed: int, filename: str) -> None:
    rng = random.Random(seed)
    coords = [(rng.uniform(0, GRID_SIZE), rng.uniform(0, GRID_SIZE))
              for _ in range(n)]

    with open(filename, "w") as f:
        f.write(f"NAME : random{n}\n")
        f.write(f"COMMENT : Random EUC_2D instance with {n} cities (seed={seed})\n")
        f.write("TYPE : TSP\n")
        f.write(f"DIMENSION : {n}\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(coords, start=1):
            f.write(f"{i} {x:.4f} {y:.4f}\n")
        f.write("EOF\n")

    # Estimate nearest-neighbour tour length for reference
    visited = [False] * n
    tour_len = 0.0
    curr = 0
    visited[0] = True
    for _ in range(n - 1):
        best_d = math.inf
        best_j = -1
        for j in range(n):
            if not visited[j]:
                dx = coords[curr][0] - coords[j][0]
                dy = coords[curr][1] - coords[j][1]
                d = math.sqrt(dx * dx + dy * dy)
                if d < best_d:
                    best_d = d
                    best_j = j
        tour_len += best_d
        visited[best_j] = True
        curr = best_j
    # Close tour
    dx = coords[curr][0] - coords[0][0]
    dy = coords[curr][1] - coords[0][1]
    tour_len += math.sqrt(dx * dx + dy * dy)

    print(f"  Created {filename}  ({n} cities, NN tour ≈ {tour_len:.1f})")


def main() -> None:
    if len(sys.argv) > 1:
        sizes = [int(s) for s in sys.argv[1:]]
    else:
        sizes = [100, 500, 1000, 5000]

    out_dir = os.path.dirname(os.path.abspath(__file__))
    print("Generating random TSP instances...")
    for n in sizes:
        filename = os.path.join(out_dir, f"random{n}.tsp")
        generate_tsp(n, seed=SEED_BASE + n, filename=filename)
    print("Done.")


if __name__ == "__main__":
    main()
