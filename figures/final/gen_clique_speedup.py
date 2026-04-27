#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

instances = ['n=50\nd=0.49', 'n=50\nd=0.90', 'n=100\nd=0.50', 'n=100\nd=0.81', 'n=150\nd=0.50', 'n=200\nd=0.50']

serial_mean = [0.000294, 0.064219, 0.006400, 4.831634, 0.028369, 0.162873]
omp8_mean   = [0.000767, 0.060715, 0.003753, 2.035176, 0.011926, 0.097248]
cuda_mean   = [0.001656, 0.072064, 0.004019, 1.699146, 0.012030, 0.065025]

omp_speedup  = [s / o for s, o in zip(serial_mean, omp8_mean)]
cuda_speedup = [s / c for s, c in zip(serial_mean, cuda_mean)]

plt.rcParams.update({'font.size': 11, 'font.family': 'serif', 'figure.dpi': 200,
                     'savefig.bbox': 'tight', 'savefig.pad_inches': 0.15})

fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(instances))
w = 0.32

b1 = ax.bar(x - w/2, omp_speedup, w, label='OpenMP (8 threads)', color='#FF9800', edgecolor='black', linewidth=0.5)
b2 = ax.bar(x + w/2, cuda_speedup, w, label='CUDA (T4 GPU)', color='#4CAF50', edgecolor='black', linewidth=0.5)

for bar, sp in zip(b1, omp_speedup):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{sp:.2f}×', ha='center', fontweight='bold', fontsize=9, color='#E65100')
for bar, sp in zip(b2, cuda_speedup):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{sp:.2f}×', ha='center', fontweight='bold', fontsize=9, color='#1B5E20')

ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Baseline (1× = serial)')
ax.set_xticks(x)
ax.set_xticklabels(instances, fontsize=10)
ax.set_ylabel('Speedup (relative to serial)')
ax.set_title('Maximum Clique — OpenMP vs CUDA Relative Speedup', fontweight='bold', fontsize=13)
ax.legend(loc='upper left')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(max(omp_speedup), max(cuda_speedup)) * 1.25)

fig.savefig('/Users/burakkocabas/Tez2/figures/final/fig_clique_speedup.png')
plt.close()
print("[OK] fig_clique_speedup.png")
