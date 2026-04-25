#!/usr/bin/env python3
"""
WP4/WP5 — Analyze benchmark results and generate thesis figures.
Reads CSV files from wp4_results/ and produces publication-quality charts.

Usage:
    python3 wp4_analyze.py                          # CPU-only results
    python3 wp4_analyze.py --cuda cuda_results.csv  # merge CUDA results
"""

import os
import sys
import csv
import math
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "wp4_results")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "..", "figures", "wp4")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'figure.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

COLORS = {
    'serial': '#2196F3',
    1: '#9E9E9E',
    2: '#FF9800',
    4: '#4CAF50',
    8: '#E91E63',
    'cuda': '#7C4DFF',
}


def load_csv(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def group_stats(rows, key_fields, time_field='time_s'):
    """Group by key_fields, compute mean and std of time."""
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in key_fields)
        groups[key].append(float(row[time_field]))
    stats = {}
    for key, times in groups.items():
        stats[key] = {
            'mean': np.mean(times),
            'std': np.std(times, ddof=1) if len(times) > 1 else 0.0,
            'min': np.min(times),
            'max': np.max(times),
            'n': len(times),
        }
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
#  TSP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def analyze_tsp():
    rows = load_csv(os.path.join(RESULTS_DIR, "tsp_results.csv"))
    if not rows:
        print("  [SKIP] No TSP results found")
        return

    stats = group_stats(rows, ['instance', 'n', 'impl', 'threads'])
    instances = sorted(set(r['instance'] for r in rows),
                       key=lambda x: int([r for r in rows if r['instance'] == x][0]['n']))

    # --- Figure: Thread scaling per instance ---
    fig, axes = plt.subplots(1, len(instances), figsize=(4 * len(instances), 4.5),
                              squeeze=False, sharey=False)
    for idx, inst in enumerate(instances):
        ax = axes[0][idx]
        n = [r for r in rows if r['instance'] == inst][0]['n']
        serial_key = (inst, n, 'serial', '1')
        if serial_key not in stats:
            continue
        serial_mean = stats[serial_key]['mean']

        threads = []
        speedups = []
        speedup_errs = []
        for t in [1, 2, 4, 8]:
            key = (inst, n, 'openmp', str(t))
            if key in stats:
                threads.append(t)
                sp = serial_mean / stats[key]['mean']
                speedups.append(sp)
                rel_err = stats[key]['std'] / stats[key]['mean'] if stats[key]['mean'] > 0 else 0
                speedup_errs.append(sp * rel_err)

        ax.errorbar(threads, speedups, yerr=speedup_errs, marker='o', linewidth=2,
                    capsize=4, color='#E91E63', label='Measured')
        ax.plot([1, 8], [1, 8], '--', color='gray', alpha=0.5, label='Ideal (linear)')
        ax.set_xlabel('Threads')
        ax.set_title(f'{inst}\n(n={n})', fontweight='bold')
        ax.set_xticks([1, 2, 4, 8])
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.set_ylabel('Speedup (vs serial)')
            ax.legend(fontsize=9)

    fig.suptitle('TSP — OpenMP Thread Scaling (Simulated Annealing)', fontweight='bold', y=1.02)
    fig.savefig(os.path.join(FIGURES_DIR, 'tsp_thread_scaling.png'))
    plt.close()
    print("  [OK] tsp_thread_scaling.png")

    # --- Figure: Execution time bar chart (all instances, serial vs best OMP) ---
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(instances))
    w = 0.35
    serial_times = []
    serial_errs = []
    omp_times = []
    omp_errs = []
    for inst in instances:
        n = [r for r in rows if r['instance'] == inst][0]['n']
        sk = (inst, n, 'serial', '1')
        serial_times.append(stats[sk]['mean'])
        serial_errs.append(stats[sk]['std'])
        best_omp_t = None
        best_omp_mean = float('inf')
        for t in [1, 2, 4, 8]:
            ok = (inst, n, 'openmp', str(t))
            if ok in stats and stats[ok]['mean'] < best_omp_mean:
                best_omp_mean = stats[ok]['mean']
                best_omp_t = ok
        if best_omp_t:
            omp_times.append(stats[best_omp_t]['mean'])
            omp_errs.append(stats[best_omp_t]['std'])
        else:
            omp_times.append(0)
            omp_errs.append(0)

    ax.bar(x - w/2, serial_times, w, yerr=serial_errs, label='Serial',
           color='#2196F3', edgecolor='black', linewidth=0.5, capsize=3)
    ax.bar(x + w/2, omp_times, w, yerr=omp_errs, label='OpenMP (best)',
           color='#FF9800', edgecolor='black', linewidth=0.5, capsize=3)

    for i in range(len(instances)):
        if omp_times[i] > 0:
            sp = serial_times[i] / omp_times[i]
            ax.text(x[i] + w/2, omp_times[i] + omp_errs[i] + max(serial_times) * 0.02,
                    f'{sp:.1f}×', ha='center', fontweight='bold', fontsize=9, color='green')

    ax.set_xticks(x)
    labels = []
    for inst in instances:
        n = [r for r in rows if r['instance'] == inst][0]['n']
        labels.append(f'{inst}\n(n={n})')
    ax.set_xticklabels(labels)
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('TSP — Serial vs OpenMP Execution Time', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.savefig(os.path.join(FIGURES_DIR, 'tsp_exec_time.png'))
    plt.close()
    print("  [OK] tsp_exec_time.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAX CLIQUE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def analyze_clique():
    rows = load_csv(os.path.join(RESULTS_DIR, "clique_results.csv"))
    if not rows:
        print("  [SKIP] No Clique results found")
        return

    stats = group_stats(rows, ['instance', 'n', 'density', 'impl', 'threads'])
    instances = sorted(set(r['instance'] for r in rows),
                       key=lambda x: (int([r for r in rows if r['instance'] == x][0]['n']),
                                      [r for r in rows if r['instance'] == x][0]['density']))

    # --- Thread scaling ---
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), squeeze=False)
    plot_idx = 0
    for inst in instances:
        if plot_idx >= 8:
            break
        row0 = [r for r in rows if r['instance'] == inst][0]
        n, dens = row0['n'], row0['density']
        ax = axes[plot_idx // 4][plot_idx % 4]

        serial_key = (inst, n, dens, 'serial', '1')
        if serial_key not in stats:
            plot_idx += 1
            continue
        serial_mean = stats[serial_key]['mean']

        threads, speedups = [], []
        for t in [1, 2, 4, 8]:
            key = (inst, n, dens, 'openmp', str(t))
            if key in stats:
                threads.append(t)
                speedups.append(serial_mean / stats[key]['mean'])

        ax.plot(threads, speedups, 'o-', color='#E91E63', linewidth=2)
        ax.plot([1, 8], [1, 8], '--', color='gray', alpha=0.5)
        ax.set_xlabel('Threads')
        ax.set_title(f'n={n}, d={dens}', fontweight='bold', fontsize=10)
        ax.set_xticks([1, 2, 4, 8])
        ax.grid(alpha=0.3)
        if plot_idx % 4 == 0:
            ax.set_ylabel('Speedup')
        plot_idx += 1

    fig.suptitle('Maximum Clique — OpenMP Thread Scaling', fontweight='bold', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIGURES_DIR, 'clique_thread_scaling.png'))
    plt.close()
    print("  [OK] clique_thread_scaling.png")

    # --- Bar chart: Serial vs OMP-8T ---
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(instances))
    w = 0.35
    serial_t, omp8_t = [], []
    for inst in instances:
        row0 = [r for r in rows if r['instance'] == inst][0]
        n, dens = row0['n'], row0['density']
        sk = (inst, n, dens, 'serial', '1')
        ok = (inst, n, dens, 'openmp', '8')
        serial_t.append(stats[sk]['mean'] if sk in stats else 0)
        omp8_t.append(stats[ok]['mean'] if ok in stats else 0)

    ax.bar(x - w/2, serial_t, w, label='Serial', color='#2196F3', edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, omp8_t, w, label='OpenMP (8T)', color='#FF9800', edgecolor='black', linewidth=0.5)

    for i in range(len(instances)):
        if omp8_t[i] > 0 and serial_t[i] > 0:
            sp = serial_t[i] / omp8_t[i]
            ax.text(x[i] + w/2, omp8_t[i] + max(serial_t) * 0.02,
                    f'{sp:.1f}×', ha='center', fontweight='bold', fontsize=8, color='green')

    labels = []
    for inst in instances:
        row0 = [r for r in rows if r['instance'] == inst][0]
        labels.append(f"n={row0['n']}\nd={row0['density']}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Maximum Clique — Serial vs OpenMP (8 threads)', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.savefig(os.path.join(FIGURES_DIR, 'clique_exec_time.png'))
    plt.close()
    print("  [OK] clique_exec_time.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  KNAPSACK ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def analyze_knapsack():
    rows = load_csv(os.path.join(RESULTS_DIR, "knapsack_results.csv"))
    if not rows:
        print("  [SKIP] No Knapsack results found")
        return

    stats = group_stats(rows, ['instance', 'n', 'W', 'dp_cells', 'impl', 'threads'])
    instances = sorted(set(r['instance'] for r in rows),
                       key=lambda x: int([r for r in rows if r['instance'] == x][0]['dp_cells']))

    # --- Thread scaling for each instance ---
    n_inst = len(instances)
    cols = min(n_inst, 3)
    fig_rows = math.ceil(n_inst / cols)
    fig, axes = plt.subplots(fig_rows, cols, figsize=(5 * cols, 4.5 * fig_rows), squeeze=False)
    for idx, inst in enumerate(instances):
        ax = axes[idx // cols][idx % cols]
        row0 = [r for r in rows if r['instance'] == inst][0]
        n, W, dc = row0['n'], row0['W'], row0['dp_cells']

        serial_key = (inst, n, W, dc, 'serial', '1')
        if serial_key not in stats:
            continue
        serial_mean = stats[serial_key]['mean']

        threads, speedups = [], []
        for t in [1, 2, 4, 8]:
            key = (inst, n, W, dc, 'openmp', str(t))
            if key in stats:
                threads.append(t)
                speedups.append(serial_mean / stats[key]['mean'])

        ax.plot(threads, speedups, 'o-', color='#E91E63', linewidth=2, label='Measured')
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline (1×)')
        ax.set_xlabel('Threads')
        ax.set_title(f'n={n}, W={int(W):,}', fontweight='bold', fontsize=10)
        ax.set_xticks([1, 2, 4, 8])
        ax.set_ylim(0, max(max(speedups, default=1) * 1.2, 1.5))
        ax.grid(alpha=0.3)
        if idx % cols == 0:
            ax.set_ylabel('Speedup')
        if idx == 0:
            ax.legend(fontsize=9)

    for idx in range(n_inst, fig_rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle('Knapsack — OpenMP Thread Scaling (Memory-Bound DP)', fontweight='bold', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIGURES_DIR, 'knapsack_thread_scaling.png'))
    plt.close()
    print("  [OK] knapsack_thread_scaling.png")

    # --- Throughput chart ---
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(instances))
    w = 0.35
    serial_tp, omp8_tp = [], []
    for inst in instances:
        row0 = [r for r in rows if r['instance'] == inst][0]
        n, W, dc = row0['n'], row0['W'], row0['dp_cells']
        dp = int(dc)
        sk = (inst, n, W, dc, 'serial', '1')
        ok = (inst, n, W, dc, 'openmp', '8')
        serial_tp.append(dp / stats[sk]['mean'] / 1e6 if sk in stats else 0)
        omp8_tp.append(dp / stats[ok]['mean'] / 1e6 if ok in stats else 0)

    ax.bar(x - w/2, serial_tp, w, label='Serial', color='#2196F3', edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, omp8_tp, w, label='OpenMP (8T)', color='#FF9800', edgecolor='black', linewidth=0.5)

    labels = []
    for inst in instances:
        row0 = [r for r in rows if r['instance'] == inst][0]
        labels.append(f"n={row0['n']}\nW={int(row0['W']):,}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Throughput (M cells/s)')
    ax.set_title('Knapsack — DP Throughput (Serial vs OpenMP)', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.savefig(os.path.join(FIGURES_DIR, 'knapsack_throughput.png'))
    plt.close()
    print("  [OK] knapsack_throughput.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
def print_summary():
    print("\n" + "=" * 70)
    print("  SUMMARY — Mean Execution Times (seconds)")
    print("=" * 70)

    for problem, filename, key_fields in [
        ("TSP", "tsp_results.csv", ['instance', 'n', 'impl', 'threads']),
        ("Clique", "clique_results.csv", ['instance', 'n', 'density', 'impl', 'threads']),
        ("Knapsack", "knapsack_results.csv", ['instance', 'n', 'W', 'dp_cells', 'impl', 'threads']),
    ]:
        rows = load_csv(os.path.join(RESULTS_DIR, filename))
        if not rows:
            continue
        stats = group_stats(rows, key_fields)
        print(f"\n  ── {problem} ──")
        print(f"  {'Instance':<28} {'Impl':<8} {'Threads':>7} {'Mean(s)':>10} {'Std':>10} {'Runs':>5}")
        for key in sorted(stats.keys()):
            s = stats[key]
            inst = key[0]
            impl = key[-2]
            threads = key[-1]
            print(f"  {inst:<28} {impl:<8} {threads:>7} {s['mean']:>10.6f} {s['std']:>10.6f} {s['n']:>5}")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("WP4/WP5 — Benchmark Analysis")
    print("=" * 50)
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Figures dir: {FIGURES_DIR}")
    print()

    print("Analyzing TSP...")
    analyze_tsp()

    print("Analyzing Max Clique...")
    analyze_clique()

    print("Analyzing Knapsack...")
    analyze_knapsack()

    print_summary()

    print(f"\nAll figures saved to: {FIGURES_DIR}/")
