#!/usr/bin/env python3
"""
WP4/WP5 Final Analysis — CPU + CUDA combined charts.
"""

import os
import csv
import math
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "wp4_results")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "..", "figures", "final")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'figure.dpi': 200, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.15,
})

C_SERIAL = '#2196F3'
C_OMP    = '#FF9800'
C_CUDA   = '#4CAF50'


def load_csv(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def mean_std(vals):
    a = np.array(vals, dtype=float)
    return a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0


# ═══════════════════════════════════════════════════════════════════
#  LOAD ALL DATA
# ═══════════════════════════════════════════════════════════════════
cpu_tsp = load_csv(os.path.join(RESULTS_DIR, "tsp_results.csv"))
cuda_tsp = load_csv(os.path.join(RESULTS_DIR, "cuda_tsp_results.csv"))
cpu_clique = load_csv(os.path.join(RESULTS_DIR, "clique_results.csv"))
cuda_clique = load_csv(os.path.join(RESULTS_DIR, "cuda_clique_results.csv"))
cpu_ks = load_csv(os.path.join(RESULTS_DIR, "knapsack_results.csv"))
cuda_ks = load_csv(os.path.join(RESULTS_DIR, "cuda_knapsack_results.csv"))


def get_times(rows, filt):
    return [float(r['time_s']) for r in rows if all(r.get(k) == str(v) for k, v in filt.items())]


# ═══════════════════════════════════════════════════════════════════
#  FIGURE 1: TSP — Serial vs OMP vs CUDA (bar chart)
# ═══════════════════════════════════════════════════════════════════
instances_tsp = ['berlin52', 'random100', 'random500', 'random1000', 'random5000']
n_vals_tsp = [52, 100, 500, 1000, 5000]

fig, ax = plt.subplots(figsize=(12, 5.5))
x = np.arange(len(instances_tsp))
w = 0.25

serial_m, serial_e = [], []
omp_m, omp_e = [], []
cuda_m, cuda_e = [], []

for inst, n in zip(instances_tsp, n_vals_tsp):
    # Serial
    t = get_times(cpu_tsp, {'instance': inst, 'impl': 'serial', 'threads': '1'})
    if t:
        m, s = mean_std(t)
        serial_m.append(m); serial_e.append(s)
    else:
        serial_m.append(0); serial_e.append(0)

    # OMP best (8 threads)
    t = get_times(cpu_tsp, {'instance': inst, 'impl': 'openmp', 'threads': '8'})
    if t:
        m, s = mean_std(t)
        omp_m.append(m); omp_e.append(s)
    else:
        omp_m.append(0); omp_e.append(0)

    # CUDA best (pick best chain count by lowest mean time)
    best_cuda_m = float('inf')
    best_cuda_s = 0
    for chains in ['128', '256', '512', '1024']:
        t = get_times(cuda_tsp, {'instance': inst, 'chains': chains})
        if t:
            m, s = mean_std(t)
            if m < best_cuda_m:
                best_cuda_m = m
                best_cuda_s = s
    if best_cuda_m < float('inf'):
        cuda_m.append(best_cuda_m); cuda_e.append(best_cuda_s)
    else:
        cuda_m.append(0); cuda_e.append(0)

b1 = ax.bar(x - w, serial_m, w, yerr=serial_e, label='Serial (CPU)', color=C_SERIAL, edgecolor='black', linewidth=0.5, capsize=3)
b2 = ax.bar(x, omp_m, w, yerr=omp_e, label='OpenMP 8T (CPU)', color=C_OMP, edgecolor='black', linewidth=0.5, capsize=3)
b3 = ax.bar(x + w, cuda_m, w, yerr=cuda_e, label='CUDA (T4 GPU)', color=C_CUDA, edgecolor='black', linewidth=0.5, capsize=3)

for i in range(len(instances_tsp)):
    if omp_m[i] > 0 and serial_m[i] > 0:
        sp = serial_m[i] / omp_m[i]
        ax.text(x[i], omp_m[i] + omp_e[i] + max(serial_m) * 0.02, f'{sp:.1f}×', ha='center', fontweight='bold', fontsize=8, color='#E65100')
    if cuda_m[i] > 0 and serial_m[i] > 0:
        sp = serial_m[i] / cuda_m[i]
        ax.text(x[i] + w, cuda_m[i] + cuda_e[i] + max(serial_m) * 0.02, f'{sp:.1f}×', ha='center', fontweight='bold', fontsize=8, color='#1B5E20')

ax.set_xticks(x)
ax.set_xticklabels([f'{inst}\n(n={n})' for inst, n in zip(instances_tsp, n_vals_tsp)], fontsize=9)
ax.set_ylabel('Execution Time (seconds)')
ax.set_title('TSP — Serial vs OpenMP vs CUDA Execution Time', fontweight='bold', fontsize=13)
ax.legend()
ax.grid(axis='y', alpha=0.3)
fig.savefig(os.path.join(FIGURES_DIR, 'fig_tsp_all.png'))
plt.close()
print("  [OK] fig_tsp_all.png")

# ═══════════════════════════════════════════════════════════════════
#  FIGURE 2: CUDA chain scaling (TSP)
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 5, figsize=(18, 4), squeeze=False, sharey=False)
for idx, (inst, n) in enumerate(zip(instances_tsp, n_vals_tsp)):
    ax = axes[0][idx]
    chains_list = [128, 256, 512, 1024]
    times = []
    for c in chains_list:
        t = get_times(cuda_tsp, {'instance': inst, 'chains': str(c)})
        if t:
            times.append(mean_std(t)[0])
        else:
            times.append(0)
    ax.plot(chains_list, times, 'o-', color=C_CUDA, linewidth=2)
    ax.set_xlabel('CUDA Chains')
    ax.set_title(f'{inst}\n(n={n})', fontweight='bold', fontsize=10)
    ax.set_xticks(chains_list)
    ax.grid(alpha=0.3)
    if idx == 0:
        ax.set_ylabel('Time (s)')
fig.suptitle('TSP CUDA — Chain Count Scaling', fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, 'fig_tsp_cuda_scaling.png'))
plt.close()
print("  [OK] fig_tsp_cuda_scaling.png")

# ═══════════════════════════════════════════════════════════════════
#  FIGURE 3: Max Clique — Serial vs OMP vs CUDA
# ═══════════════════════════════════════════════════════════════════
clique_instances = ['rand_n50_d50', 'rand_n50_d90', 'rand_n100_d50', 'rand_n100_d80', 'rand_n150_d50', 'rand_n200_d50']
clique_labels = ['n=50\nd=0.49', 'n=50\nd=0.90', 'n=100\nd=0.50', 'n=100\nd=0.81', 'n=150\nd=0.50', 'n=200\nd=0.50']

fig, ax = plt.subplots(figsize=(12, 5.5))
x = np.arange(len(clique_instances))
w = 0.25

s_m, o_m, c_m = [], [], []
s_e, o_e, c_e = [], [], []

for inst in clique_instances:
    t = get_times(cpu_clique, {'instance': inst, 'impl': 'serial', 'threads': '1'})
    m, s = mean_std(t) if t else (0, 0)
    s_m.append(m); s_e.append(s)

    t = get_times(cpu_clique, {'instance': inst, 'impl': 'openmp', 'threads': '8'})
    m, s = mean_std(t) if t else (0, 0)
    o_m.append(m); o_e.append(s)

    t = get_times(cuda_clique, {'instance': inst})
    m, s = mean_std(t) if t else (0, 0)
    c_m.append(m); c_e.append(s)

ax.bar(x - w, s_m, w, yerr=s_e, label='Serial', color=C_SERIAL, edgecolor='black', linewidth=0.5, capsize=3)
ax.bar(x, o_m, w, yerr=o_e, label='OpenMP 8T', color=C_OMP, edgecolor='black', linewidth=0.5, capsize=3)
ax.bar(x + w, c_m, w, yerr=c_e, label='CUDA (T4)', color=C_CUDA, edgecolor='black', linewidth=0.5, capsize=3)

for i in range(len(clique_instances)):
    if o_m[i] > 0 and s_m[i] > 0:
        sp = s_m[i] / o_m[i]
        ax.text(x[i], o_m[i] + max(s_m) * 0.03, f'{sp:.1f}×', ha='center', fontweight='bold', fontsize=8, color='#E65100')
    if c_m[i] > 0 and s_m[i] > 0:
        sp = s_m[i] / c_m[i]
        ax.text(x[i] + w, c_m[i] + max(s_m) * 0.03, f'{sp:.1f}×', ha='center', fontweight='bold', fontsize=8, color='#1B5E20')

ax.set_xticks(x)
ax.set_xticklabels(clique_labels, fontsize=9)
ax.set_ylabel('Execution Time (seconds)')
ax.set_title('Maximum Clique — Serial vs OpenMP vs CUDA', fontweight='bold', fontsize=13)
ax.legend()
ax.grid(axis='y', alpha=0.3)
fig.savefig(os.path.join(FIGURES_DIR, 'fig_clique_all.png'))
plt.close()
print("  [OK] fig_clique_all.png")

# ═══════════════════════════════════════════════════════════════════
#  FIGURE 4: Knapsack — Serial vs OMP vs CUDA
# ═══════════════════════════════════════════════════════════════════
ks_instances = ['ks_n500_W50000', 'ks_n1000_W50000', 'ks_n1000_W100000', 'ks_n2000_W50000', 'ks_n2000_W100000', 'ks_n5000_W50000']
ks_labels = ['n=500\nW=50K', 'n=1K\nW=50K', 'n=1K\nW=100K', 'n=2K\nW=50K', 'n=2K\nW=100K', 'n=5K\nW=50K']

fig, ax = plt.subplots(figsize=(12, 5.5))
x = np.arange(len(ks_instances))
w = 0.25

s_m, o_m, c_m = [], [], []
s_e, o_e, c_e = [], [], []

for inst in ks_instances:
    t = get_times(cpu_ks, {'instance': inst, 'impl': 'serial', 'threads': '1'})
    m, s = mean_std(t) if t else (0, 0)
    s_m.append(m); s_e.append(s)

    t = get_times(cpu_ks, {'instance': inst, 'impl': 'openmp', 'threads': '8'})
    m, s = mean_std(t) if t else (0, 0)
    o_m.append(m); o_e.append(s)

    t = get_times(cuda_ks, {'instance': inst})
    m, s = mean_std(t) if t else (0, 0)
    c_m.append(m); c_e.append(s)

ax.bar(x - w, s_m, w, yerr=s_e, label='Serial', color=C_SERIAL, edgecolor='black', linewidth=0.5, capsize=3)
ax.bar(x, o_m, w, yerr=o_e, label='OpenMP 8T', color=C_OMP, edgecolor='black', linewidth=0.5, capsize=3)
ax.bar(x + w, c_m, w, yerr=c_e, label='CUDA (T4)', color=C_CUDA, edgecolor='black', linewidth=0.5, capsize=3)

for i in range(len(ks_instances)):
    if c_m[i] > 0 and s_m[i] > 0:
        sp = s_m[i] / c_m[i]
        ax.text(x[i] + w, c_m[i] + max(s_m) * 0.03, f'{sp:.0f}×', ha='center', fontweight='bold', fontsize=8, color='#1B5E20')

ax.set_xticks(x)
ax.set_xticklabels(ks_labels, fontsize=9)
ax.set_ylabel('Execution Time (seconds)')
ax.set_title('Knapsack — Serial vs OpenMP vs CUDA (GPU 20-50× faster)', fontweight='bold', fontsize=13)
ax.legend()
ax.grid(axis='y', alpha=0.3)
fig.savefig(os.path.join(FIGURES_DIR, 'fig_knapsack_all.png'))
plt.close()
print("  [OK] fig_knapsack_all.png")

# ═══════════════════════════════════════════════════════════════════
#  FIGURE 5: Knapsack Throughput — Serial vs OMP vs CUDA (log scale)
# ═══════════════════════════════════════════════════════════════════
ks_dp_cells = [25000500, 50001000, 100001000, 100002000, 200002000, 250005000]

fig, ax = plt.subplots(figsize=(12, 5.5))
x = np.arange(len(ks_instances))
w = 0.25

s_tp, o_tp, c_tp = [], [], []
for i, inst in enumerate(ks_instances):
    dc = ks_dp_cells[i]
    t = get_times(cpu_ks, {'instance': inst, 'impl': 'serial', 'threads': '1'})
    s_tp.append(dc / mean_std(t)[0] / 1e6 if t else 0)
    t = get_times(cpu_ks, {'instance': inst, 'impl': 'openmp', 'threads': '8'})
    o_tp.append(dc / mean_std(t)[0] / 1e6 if t else 0)
    t = get_times(cuda_ks, {'instance': inst})
    c_tp.append(dc / mean_std(t)[0] / 1e6 if t else 0)

ax.bar(x - w, s_tp, w, label='Serial', color=C_SERIAL, edgecolor='black', linewidth=0.5)
ax.bar(x, o_tp, w, label='OpenMP 8T', color=C_OMP, edgecolor='black', linewidth=0.5)
ax.bar(x + w, c_tp, w, label='CUDA (T4)', color=C_CUDA, edgecolor='black', linewidth=0.5)

for i in range(len(ks_instances)):
    ax.text(x[i] + w, c_tp[i] * 1.05, f'{c_tp[i]:,.0f}', ha='center', fontsize=7, fontweight='bold', color='#1B5E20')

ax.set_xticks(x)
ax.set_xticklabels(ks_labels, fontsize=9)
ax.set_ylabel('Throughput (M DP cells / second)')
ax.set_title('Knapsack Throughput — GPU achieves 10,000-20,000 M cells/s', fontweight='bold', fontsize=13)
ax.set_yscale('log')
ax.legend()
ax.grid(axis='y', alpha=0.3, which='both')
fig.savefig(os.path.join(FIGURES_DIR, 'fig_knapsack_throughput.png'))
plt.close()
print("  [OK] fig_knapsack_throughput.png")

# ═══════════════════════════════════════════════════════════════════
#  FIGURE 6: Overall Speedup Summary
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5.5))

problems = ['TSP\n(n=1000)', 'Max Clique\n(n=100, d=0.81)', 'Max Clique\n(n=200, d=0.50)', 'Knapsack\n(n=2000, W=100K)', 'Knapsack\n(n=5000, W=50K)']

# Serial times
s_tsp1000 = mean_std(get_times(cpu_tsp, {'instance': 'random1000', 'impl': 'serial', 'threads': '1'}))[0]
s_clq100 = mean_std(get_times(cpu_clique, {'instance': 'rand_n100_d80', 'impl': 'serial', 'threads': '1'}))[0]
s_clq200 = mean_std(get_times(cpu_clique, {'instance': 'rand_n200_d50', 'impl': 'serial', 'threads': '1'}))[0]
s_ks2000 = mean_std(get_times(cpu_ks, {'instance': 'ks_n2000_W100000', 'impl': 'serial', 'threads': '1'}))[0]
s_ks5000 = mean_std(get_times(cpu_ks, {'instance': 'ks_n5000_W50000', 'impl': 'serial', 'threads': '1'}))[0]

# OMP-8T times
o_tsp1000 = mean_std(get_times(cpu_tsp, {'instance': 'random1000', 'impl': 'openmp', 'threads': '8'}))[0]
o_clq100 = mean_std(get_times(cpu_clique, {'instance': 'rand_n100_d80', 'impl': 'openmp', 'threads': '8'}))[0]
o_clq200 = mean_std(get_times(cpu_clique, {'instance': 'rand_n200_d50', 'impl': 'openmp', 'threads': '8'}))[0]
o_ks2000 = mean_std(get_times(cpu_ks, {'instance': 'ks_n2000_W100000', 'impl': 'openmp', 'threads': '8'}))[0]
o_ks5000 = mean_std(get_times(cpu_ks, {'instance': 'ks_n5000_W50000', 'impl': 'openmp', 'threads': '8'}))[0]

# CUDA best times
c_tsp1000 = mean_std(get_times(cuda_tsp, {'instance': 'random1000', 'chains': '1024'}))[0]
c_clq100 = mean_std(get_times(cuda_clique, {'instance': 'rand_n100_d80'}))[0]
c_clq200 = mean_std(get_times(cuda_clique, {'instance': 'rand_n200_d50'}))[0]
c_ks2000 = mean_std(get_times(cuda_ks, {'instance': 'ks_n2000_W100000'}))[0]
c_ks5000 = mean_std(get_times(cuda_ks, {'instance': 'ks_n5000_W50000'}))[0]

omp_speedups = [s_tsp1000/o_tsp1000, s_clq100/o_clq100, s_clq200/o_clq200, s_ks2000/o_ks2000, s_ks5000/o_ks5000]
cuda_speedups = [s_tsp1000/c_tsp1000, s_clq100/c_clq100, s_clq200/c_clq200, s_ks2000/c_ks2000, s_ks5000/c_ks5000]

xp = np.arange(len(problems))
wp = 0.32
b1 = ax.bar(xp - wp/2, omp_speedups, wp, label='OpenMP 8T', color=C_OMP, edgecolor='black', linewidth=0.5)
b2 = ax.bar(xp + wp/2, cuda_speedups, wp, label='CUDA (T4 GPU)', color=C_CUDA, edgecolor='black', linewidth=0.5)

for bar, sp in zip(b1, omp_speedups):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{sp:.1f}×', ha='center', fontweight='bold', fontsize=9, color='#E65100')
for bar, sp in zip(b2, cuda_speedups):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{sp:.1f}×', ha='center', fontweight='bold', fontsize=9, color='#1B5E20')

ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Baseline (1× = serial)')
ax.set_xticks(xp)
ax.set_xticklabels(problems, fontsize=9)
ax.set_ylabel('Speedup (relative to serial)')
ax.set_title('Overall Speedup Summary — OpenMP vs CUDA', fontweight='bold', fontsize=13)
ax.legend(loc='upper left')
ax.grid(axis='y', alpha=0.3)
fig.savefig(os.path.join(FIGURES_DIR, 'fig_speedup_summary.png'))
plt.close()
print("  [OK] fig_speedup_summary.png")


# ═══════════════════════════════════════════════════════════════════
#  PRINT SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("  FINAL SUMMARY — CPU vs CUDA")
print("=" * 80)
print(f"  {'Problem':<30} {'Serial':>10} {'OMP-8T':>10} {'CUDA':>10} {'OMP sp':>8} {'CUDA sp':>8}")
print("-" * 80)

data = [
    ('TSP berlin52', 'berlin52', 'berlin52', '1024'),
    ('TSP random500', 'random500', 'random500', '1024'),
    ('TSP random1000', 'random1000', 'random1000', '1024'),
    ('TSP random5000', 'random5000', 'random5000', '128'),
    ('Clique n100 d50', 'rand_n100_d50', 'rand_n100_d50', None),
    ('Clique n100 d80', 'rand_n100_d80', 'rand_n100_d80', None),
    ('Clique n200 d50', 'rand_n200_d50', 'rand_n200_d50', None),
    ('Knapsack n1000 W50K', 'ks_n1000_W50000', 'ks_n1000_W50000', None),
    ('Knapsack n2000 W100K', 'ks_n2000_W100000', 'ks_n2000_W100000', None),
    ('Knapsack n5000 W50K', 'ks_n5000_W50000', 'ks_n5000_W50000', None),
]

for label, cpu_inst, cuda_inst, chains in data:
    st = get_times(cpu_tsp if 'TSP' in label else cpu_clique if 'Clique' in label else cpu_ks,
                   {'instance': cpu_inst, 'impl': 'serial', 'threads': '1'})
    ot = get_times(cpu_tsp if 'TSP' in label else cpu_clique if 'Clique' in label else cpu_ks,
                   {'instance': cpu_inst, 'impl': 'openmp', 'threads': '8'})
    if chains:
        ct = get_times(cuda_tsp, {'instance': cuda_inst, 'chains': chains})
    else:
        ct = get_times(cuda_clique if 'Clique' in label else cuda_ks, {'instance': cuda_inst})

    sm = mean_std(st)[0] if st else 0
    om = mean_std(ot)[0] if ot else 0
    cm = mean_std(ct)[0] if ct else 0

    osp = f'{sm/om:.1f}×' if om > 0 else 'N/A'
    csp = f'{sm/cm:.1f}×' if cm > 0 else 'N/A'

    print(f"  {label:<30} {sm:>10.4f} {om:>10.4f} {cm:>10.4f} {osp:>8} {csp:>8}")

print("=" * 80)
