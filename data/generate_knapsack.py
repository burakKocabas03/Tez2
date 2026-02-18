#!/usr/bin/env python3
"""
Generate random 0/1 Knapsack instances.

Instance format (one per file):
  N W
  w1 v1
  w2 v2
  ...

Usage:
    python3 generate_knapsack.py              # default set of instances
    python3 generate_knapsack.py 1000 50000   # n=1000 items, W=50000
"""

import sys
import random
import os

SEED_BASE = 2025
MAX_WEIGHT = 1000   # max weight per item
MAX_VALUE  = 1000   # max value  per item


def generate_instance(n: int, W: int, seed: int, filename: str) -> None:
    rng = random.Random(seed)
    items = [(rng.randint(1, MAX_WEIGHT), rng.randint(1, MAX_VALUE))
             for _ in range(n)]

    with open(filename, "w") as f:
        f.write(f"{n} {W}\n")
        for w, v in items:
            f.write(f"{w} {v}\n")

    # Greedy lower bound (value/weight ratio)
    sorted_items = sorted(items, key=lambda x: x[1] / x[0], reverse=True)
    total_w, total_v = 0, 0
    for w, v in sorted_items:
        if total_w + w <= W:
            total_w += w
            total_v += v

    dp_cells = n * (W + 1)
    print(f"  Created {filename}  (n={n}, W={W}, "
          f"DP cells={dp_cells:,}, greedy value≥{total_v})")


def main() -> None:
    out_dir = os.path.dirname(os.path.abspath(__file__))
    ks_dir = os.path.join(out_dir, "knapsack")
    os.makedirs(ks_dir, exist_ok=True)

    if len(sys.argv) == 3:
        configs = [(int(sys.argv[1]), int(sys.argv[2]))]
    else:
        # (n_items, capacity) — DP cost = n * W
        configs = [
            (500,    50_000),    # 25M cells
            (1_000,  50_000),    # 50M cells
            (2_000,  50_000),    # 100M cells
            (1_000, 100_000),    # 100M cells
            (2_000, 100_000),    # 200M cells
            (5_000,  50_000),    # 250M cells
        ]

    print("Generating 0/1 Knapsack instances...")
    for n, W in configs:
        tag      = f"ks_n{n}_W{W}"
        filename = os.path.join(ks_dir, f"{tag}.txt")
        generate_instance(n, W, seed=SEED_BASE + n + W, filename=filename)
    print("Done.")


if __name__ == "__main__":
    main()
