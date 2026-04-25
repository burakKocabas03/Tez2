#!/usr/bin/env python3
"""Generate all figures for the interim report."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

out_dir = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'figure.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

SERIAL_COLOR = '#2196F3'
OPENMP_COLOR = '#FF9800'
CUDA_COLOR   = '#4CAF50'

# ═══════════════════════════════════════════════════════════════════
# Figure 2: TSP Execution Time
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4.5))
labels = ['Serial', 'OpenMP\n(8 threads)', 'CUDA\n(256 chains)']
times  = [0.016, 0.005, 0.023]
colors = [SERIAL_COLOR, OPENMP_COLOR, CUDA_COLOR]
bars = ax.bar(labels, times, color=colors, width=0.55, edgecolor='black', linewidth=0.5)
for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
            f'{t:.3f}s', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_ylabel('Execution Time (seconds)')
ax.set_title('Figure 2. TSP Performance — berlin52 (n=52)', fontweight='bold')
ax.set_ylim(0, 0.032)
ax.grid(axis='y', alpha=0.3)
# Speedup annotations
ax.annotate('3.2× speedup', xy=(1, 0.005), xytext=(1.5, 0.013),
            fontsize=10, color='green', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='green'))
fig.savefig(os.path.join(out_dir, 'fig2_tsp_time.png'))
plt.close()

# ═══════════════════════════════════════════════════════════════════
# Figure 3: MCP Execution Time
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
instances = ['n=100\nd=0.5', 'n=150\nd=0.5', 'n=200\nd=0.5', 'n=50\nd=0.9']
serial_t  = [0.003, 0.018, 0.086, 0.040]
openmp_t  = [0.002, 0.008, 0.030, 0.030]
speedups  = [s/o for s, o in zip(serial_t, openmp_t)]

x = np.arange(len(instances))
w = 0.32
b1 = ax.bar(x - w/2, serial_t, w, label='Serial', color=SERIAL_COLOR, edgecolor='black', linewidth=0.5)
b2 = ax.bar(x + w/2, openmp_t, w, label='OpenMP (8 threads)', color=OPENMP_COLOR, edgecolor='black', linewidth=0.5)

for i, (bar, sp) in enumerate(zip(b2, speedups)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{sp:.1f}×', ha='center', va='bottom', fontweight='bold', fontsize=10, color='green')

ax.set_xticks(x)
ax.set_xticklabels(instances)
ax.set_ylabel('Execution Time (seconds)')
ax.set_title('Figure 3. Maximum Clique Performance — Serial vs OpenMP', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
fig.savefig(os.path.join(out_dir, 'fig3_clique_time.png'))
plt.close()

# ═══════════════════════════════════════════════════════════════════
# Figure 4: Knapsack Execution Time (THE KEY FIGURE)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5))
labels = ['Serial', 'OpenMP\n(8 threads)', 'CUDA\n(T4 GPU)']
times  = [0.174, 0.212, 0.007]
colors = [SERIAL_COLOR, OPENMP_COLOR, CUDA_COLOR]
bars = ax.bar(labels, times, color=colors, width=0.55, edgecolor='black', linewidth=0.5)
for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{t:.3f}s', ha='center', va='bottom', fontweight='bold', fontsize=12)

ax.annotate('24× speedup!', xy=(2, 0.007), xytext=(1.5, 0.12),
            fontsize=13, color='red', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.annotate('0.8× (slower!)', xy=(1, 0.212), xytext=(0.2, 0.22),
            fontsize=10, color='#D32F2F', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#D32F2F'))

ax.set_ylabel('Execution Time (seconds)')
ax.set_title('Figure 4. Knapsack Performance — n=2000, W=100,000\n(Memory-Bandwidth-Bound Workload)', fontweight='bold')
ax.set_ylim(0, 0.28)
ax.grid(axis='y', alpha=0.3)
fig.savefig(os.path.join(out_dir, 'fig4_knapsack_time.png'))
plt.close()

# ═══════════════════════════════════════════════════════════════════
# Figure 5: Knapsack Throughput (log scale)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5))
labels     = ['Serial', 'OpenMP\n(8 threads)', 'CUDA\n(T4 GPU)']
throughput = [1152, 945, 27297]
colors     = [SERIAL_COLOR, OPENMP_COLOR, CUDA_COLOR]
bars = ax.bar(labels, throughput, color=colors, width=0.55, edgecolor='black', linewidth=0.5)
for bar, t in zip(bars, throughput):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.08,
            f'{t:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.set_ylabel('Throughput (Million DP cells / second)')
ax.set_title('Figure 5. Knapsack Throughput — n=2000, W=100,000\n(GPU: 24× higher throughput due to GDDR6 bandwidth)', fontweight='bold')
ax.set_yscale('log')
ax.set_ylim(500, 60000)
ax.grid(axis='y', alpha=0.3, which='both')
fig.savefig(os.path.join(out_dir, 'fig5_knapsack_throughput.png'))
plt.close()

# ═══════════════════════════════════════════════════════════════════
# Figure 1: Overall Speedup Summary (all 3 problems)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))

problems   = ['TSP\n(berlin52)', 'Max Clique\n(n=200, d=0.5)', 'Knapsack\n(n=2000, W=100K)']
omp_speed  = [3.2, 2.9, 0.82]
cuda_speed = [0.7, 0.86, 24.0]

x = np.arange(len(problems))
w = 0.32
b1 = ax.bar(x - w/2, omp_speed,  w, label='OpenMP (8 threads)', color=OPENMP_COLOR, edgecolor='black', linewidth=0.5)
b2 = ax.bar(x + w/2, cuda_speed, w, label='CUDA (T4 GPU)',      color=CUDA_COLOR,   edgecolor='black', linewidth=0.5)

for bar, sp in zip(b1, omp_speed):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{sp}×', ha='center', va='bottom', fontweight='bold', fontsize=10, color='#E65100')
for bar, sp in zip(b2, cuda_speed):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{sp}×', ha='center', va='bottom', fontweight='bold', fontsize=10, color='#1B5E20')

ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Baseline (1× = serial)')
ax.set_xticks(x)
ax.set_xticklabels(problems)
ax.set_ylabel('Speedup (relative to serial)')
ax.set_title('Figure 1. Overall Speedup Summary — OpenMP vs CUDA', fontweight='bold')
ax.legend(loc='upper left')
ax.set_ylim(0, 28)
ax.grid(axis='y', alpha=0.3)
fig.savefig(os.path.join(out_dir, 'fig1_speedup_summary.png'))
plt.close()

print("All figures generated:")
for f in sorted(os.listdir(out_dir)):
    if f.endswith('.png'):
        print(f"  {f}")
