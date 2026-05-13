#!/usr/bin/env python3
"""
Benchmark Visualization — TSP / MCP / Knapsack
================================================
Reads CSV results from benchmark/results/ and produces comparison charts
saved to figures/ directory.

Charts produced per problem:
  1. Execution Time (grouped bar)
  2. Speedup vs Serial
  3. Solution Quality comparison
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#e94560',
    'axes.labelcolor': '#eaeaea',
    'axes.titlecolor': '#eaeaea',
    'text.color': '#eaeaea',
    'xtick.color': '#aaaaaa',
    'ytick.color': '#aaaaaa',
    'grid.color': '#2a2a4a',
    'grid.alpha': 0.5,
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'legend.facecolor': '#16213e',
    'legend.edgecolor': '#e94560',
    'legend.labelcolor': '#eaeaea',
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.facecolor': '#1a1a2e',
})

COLORS = {
    'serial':  '#e94560',  # coral red
    'openmp':  '#0f3460',  # navy (with bright border)
    'cpu_opt': '#53d769',  # green
}
COLORS_BRIGHT = {
    'serial':  '#ff6b81',
    'openmp':  '#5b9bd5',
    'cpu_opt': '#7deca5',
}
VARIANT_LABELS = {
    'serial': 'Serial (WP1)',
    'openmp': 'OpenMP (WP2)',
    'cpu_opt': 'CPU-Opt (WP4)',
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'benchmark', 'results')
FIGURES = os.path.join(ROOT, 'figures')
os.makedirs(FIGURES, exist_ok=True)


def add_bar_values(ax, bars, fmt='{:.4f}'):
    """Add value labels above bars."""
    for bar in bars:
        h = bar.get_height()
        if h > 0 and np.isfinite(h):
            ax.text(bar.get_x() + bar.get_width() / 2., h,
                    fmt.format(h), ha='center', va='bottom',
                    fontsize=7, color='#eaeaea', fontweight='bold')


# ============================================================================
#  TSP Plots
# ============================================================================
def plot_tsp():
    csv_path = os.path.join(RESULTS, 'tsp_results.csv')
    if not os.path.exists(csv_path):
        print("[SKIP] tsp_results.csv not found")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("[SKIP] tsp_results.csv is empty")
        return

    datasets = df['dataset'].unique()
    variants = ['serial', 'openmp', 'cpu_opt']

    # ── 1. Execution Time ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(datasets))
    w = 0.25

    for i, var in enumerate(variants):
        sub = df[df['variant'] == var].set_index('dataset')
        vals = [sub.loc[d, 'time_s'] if d in sub.index else 0 for d in datasets]
        bars = ax.bar(x + i * w, vals, w, label=VARIANT_LABELS[var],
                      color=COLORS_BRIGHT[var], edgecolor=COLORS[var], linewidth=1.2)
        add_bar_values(ax, bars, '{:.3f}')

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('TSP — Execution Time Comparison')
    ax.set_xticks(x + w)
    ax.set_xticklabels(datasets, rotation=0)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(axis='y', linestyle='--')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'tsp_execution_time.png'))
    plt.close(fig)
    print("  ✓ tsp_execution_time.png")

    # ── 2. Speedup ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    serial_times = df[df['variant'] == 'serial'].set_index('dataset')['time_s']

    for i, var in enumerate(['openmp', 'cpu_opt']):
        sub = df[df['variant'] == var].set_index('dataset')
        speedups = []
        for d in datasets:
            if d in serial_times.index and d in sub.index:
                speedups.append(serial_times[d] / sub.loc[d, 'time_s'])
            else:
                speedups.append(0)
        bars = ax.bar(x + i * 0.35, speedups, 0.35,
                      label=VARIANT_LABELS[var],
                      color=COLORS_BRIGHT[var], edgecolor=COLORS[var], linewidth=1.2)
        add_bar_values(ax, bars, '{:.1f}x')

    ax.axhline(y=1, color='#e94560', linestyle='--', alpha=0.7, label='Serial baseline')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Speedup (vs Serial)')
    ax.set_title('TSP — Speedup vs Serial')
    ax.set_xticks(x + 0.175)
    ax.set_xticklabels(datasets, rotation=0)
    ax.legend()
    ax.grid(axis='y', linestyle='--')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'tsp_speedup.png'))
    plt.close(fig)
    print("  ✓ tsp_speedup.png")

    # ── 3. Solution Quality ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, var in enumerate(variants):
        sub = df[df['variant'] == var].set_index('dataset')
        vals = [sub.loc[d, 'cost'] if d in sub.index else 0 for d in datasets]
        bars = ax.bar(x + i * w, vals, w, label=VARIANT_LABELS[var],
                      color=COLORS_BRIGHT[var], edgecolor=COLORS[var], linewidth=1.2)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Tour Cost (lower is better)')
    ax.set_title('TSP — Solution Quality Comparison')
    ax.set_xticks(x + w)
    ax.set_xticklabels(datasets, rotation=0)
    ax.legend()
    ax.grid(axis='y', linestyle='--')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'tsp_solution_quality.png'))
    plt.close(fig)
    print("  ✓ tsp_solution_quality.png")


# ============================================================================
#  MCP Plots
# ============================================================================
def plot_mcp():
    csv_path = os.path.join(RESULTS, 'mcp_results.csv')
    if not os.path.exists(csv_path):
        print("[SKIP] mcp_results.csv not found")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("[SKIP] mcp_results.csv is empty")
        return

    datasets = df['dataset'].unique()
    variants = df['variant'].unique()

    # ── 1. Execution Time ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(datasets))
    w = 0.25

    for i, var in enumerate(variants):
        sub = df[df['variant'] == var].set_index('dataset')
        vals = [sub.loc[d, 'time_s'] if d in sub.index else 0 for d in datasets]
        color = COLORS_BRIGHT.get(var, '#888888')
        edge = COLORS.get(var, '#444444')
        bars = ax.bar(x + i * w, vals, w, label=VARIANT_LABELS.get(var, var),
                      color=color, edgecolor=edge, linewidth=1.2)
        add_bar_values(ax, bars, '{:.4f}')

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('MCP — Execution Time Comparison')
    ax.set_xticks(x + w * (len(variants) - 1) / 2)
    ax.set_xticklabels(datasets, rotation=0)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(axis='y', linestyle='--')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'mcp_execution_time.png'))
    plt.close(fig)
    print("  ✓ mcp_execution_time.png")

    # ── 2. Speedup (only where serial is available) ──────────────────
    serial_data = df[df['variant'] == 'serial'].set_index('dataset')
    if not serial_data.empty:
        common_datasets = [d for d in datasets if d in serial_data.index]
        if common_datasets:
            fig, ax = plt.subplots(figsize=(10, 6))
            x2 = np.arange(len(common_datasets))

            for i, var in enumerate(['openmp', 'cpu_opt']):
                sub = df[df['variant'] == var].set_index('dataset')
                speedups = []
                for d in common_datasets:
                    if d in sub.index:
                        speedups.append(serial_data.loc[d, 'time_s'] / sub.loc[d, 'time_s'])
                    else:
                        speedups.append(0)
                color = COLORS_BRIGHT.get(var, '#888888')
                edge = COLORS.get(var, '#444444')
                bars = ax.bar(x2 + i * 0.35, speedups, 0.35,
                              label=VARIANT_LABELS.get(var, var),
                              color=color, edgecolor=edge, linewidth=1.2)
                add_bar_values(ax, bars, '{:.1f}x')

            ax.axhline(y=1, color='#e94560', linestyle='--', alpha=0.7, label='Serial baseline')
            ax.set_xlabel('Dataset')
            ax.set_ylabel('Speedup (vs Serial)')
            ax.set_title('MCP — Speedup vs Serial')
            ax.set_xticks(x2 + 0.175)
            ax.set_xticklabels(common_datasets, rotation=0)
            ax.legend()
            ax.grid(axis='y', linestyle='--')
            fig.tight_layout()
            fig.savefig(os.path.join(FIGURES, 'mcp_speedup.png'))
            plt.close(fig)
            print("  ✓ mcp_speedup.png")

    # ── 3. Clique Size ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(datasets))

    for i, var in enumerate(variants):
        sub = df[df['variant'] == var].set_index('dataset')
        vals = [int(sub.loc[d, 'clique_size']) if d in sub.index else 0 for d in datasets]
        color = COLORS_BRIGHT.get(var, '#888888')
        edge = COLORS.get(var, '#444444')
        bars = ax.bar(x + i * w, vals, w, label=VARIANT_LABELS.get(var, var),
                      color=color, edgecolor=edge, linewidth=1.2)
        add_bar_values(ax, bars, '{:.0f}')

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Maximum Clique Size')
    ax.set_title('MCP — Solution Quality (Clique Size)')
    ax.set_xticks(x + w * (len(variants) - 1) / 2)
    ax.set_xticklabels(datasets, rotation=0)
    ax.legend()
    ax.grid(axis='y', linestyle='--')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'mcp_clique_size.png'))
    plt.close(fig)
    print("  ✓ mcp_clique_size.png")

    # ── 4. Nodes Explored ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, var in enumerate(variants):
        sub = df[df['variant'] == var].set_index('dataset')
        vals = [sub.loc[d, 'nodes'] if d in sub.index else 0 for d in datasets]
        color = COLORS_BRIGHT.get(var, '#888888')
        edge = COLORS.get(var, '#444444')
        bars = ax.bar(x + i * w, vals, w, label=VARIANT_LABELS.get(var, var),
                      color=color, edgecolor=edge, linewidth=1.2)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Nodes Explored')
    ax.set_title('MCP — Search Space (Nodes Explored)')
    ax.set_xticks(x + w * (len(variants) - 1) / 2)
    ax.set_xticklabels(datasets, rotation=0)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(axis='y', linestyle='--')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'mcp_nodes_explored.png'))
    plt.close(fig)
    print("  ✓ mcp_nodes_explored.png")


# ============================================================================
#  Knapsack Plots
# ============================================================================
def plot_knapsack():
    csv_path = os.path.join(RESULTS, 'knapsack_results.csv')
    if not os.path.exists(csv_path):
        print("[SKIP] knapsack_results.csv not found")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("[SKIP] knapsack_results.csv is empty")
        return

    # Extract item count from dataset name for cleaner labels
    df['items'] = df['n'].astype(int)
    df['short_name'] = df['items'].apply(lambda x: f'n={x}')

    datasets = df['dataset'].unique()
    short_names = [df[df['dataset'] == d]['short_name'].iloc[0] for d in datasets]
    variants = ['serial', 'openmp', 'cpu_opt']

    # ── 1. Execution Time ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(datasets))
    w = 0.25

    for i, var in enumerate(variants):
        sub = df[df['variant'] == var].set_index('dataset')
        vals = [sub.loc[d, 'time_s'] if d in sub.index else 0 for d in datasets]
        bars = ax.bar(x + i * w, vals, w, label=VARIANT_LABELS[var],
                      color=COLORS_BRIGHT[var], edgecolor=COLORS[var], linewidth=1.2)
        add_bar_values(ax, bars, '{:.4f}')

    ax.set_xlabel('Dataset (number of items)')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Knapsack — Execution Time Comparison')
    ax.set_xticks(x + w)
    ax.set_xticklabels(short_names, rotation=0)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(axis='y', linestyle='--')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'knapsack_execution_time.png'))
    plt.close(fig)
    print("  ✓ knapsack_execution_time.png")

    # ── 2. Speedup vs Serial ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    serial_times = df[df['variant'] == 'serial'].set_index('dataset')['time_s']

    for i, var in enumerate(['openmp', 'cpu_opt']):
        sub = df[df['variant'] == var].set_index('dataset')
        speedups = []
        for d in datasets:
            if d in serial_times.index and d in sub.index:
                sp = serial_times[d] / sub.loc[d, 'time_s']
                speedups.append(sp)
            else:
                speedups.append(0)
        bars = ax.bar(x + i * 0.35, speedups, 0.35,
                      label=VARIANT_LABELS[var],
                      color=COLORS_BRIGHT[var], edgecolor=COLORS[var], linewidth=1.2)
        add_bar_values(ax, bars, '{:.2f}x')

    ax.axhline(y=1, color='#e94560', linestyle='--', alpha=0.7, label='Serial baseline')
    ax.set_xlabel('Dataset (number of items)')
    ax.set_ylabel('Speedup (vs Serial)')
    ax.set_title('Knapsack — Speedup vs Serial')
    ax.set_xticks(x + 0.175)
    ax.set_xticklabels(short_names, rotation=0)
    ax.legend()
    ax.grid(axis='y', linestyle='--')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'knapsack_speedup.png'))
    plt.close(fig)
    print("  ✓ knapsack_speedup.png")

    # ── 3. Correctness Check ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))

    known_df = df[df['known_optimal'].notna() & (df['known_optimal'] != '')]
    if not known_df.empty:
        known_df = known_df.copy()
        known_df['known_optimal'] = known_df['known_optimal'].astype(int)
        known_df['match'] = known_df['optimal_value'].astype(int) == known_df['known_optimal']

        # Build a table-like display
        check_data = []
        for d in datasets:
            row = {'Dataset': df[df['dataset'] == d]['short_name'].iloc[0]}
            for var in variants:
                subset = known_df[(known_df['dataset'] == d) & (known_df['variant'] == var)]
                if not subset.empty:
                    row[VARIANT_LABELS[var]] = '✓' if subset.iloc[0]['match'] else '✗'
                else:
                    row[VARIANT_LABELS[var]] = '—'
            check_data.append(row)

        check_df = pd.DataFrame(check_data)
        ax.axis('off')
        table = ax.table(
            cellText=check_df.values,
            colLabels=check_df.columns,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.6)

        # Style the table
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor('#e94560')
            if row == 0:
                cell.set_facecolor('#e94560')
                cell.set_text_props(color='white', fontweight='bold')
            else:
                cell.set_facecolor('#16213e')
                cell.set_text_props(color='#eaeaea')
                txt = cell.get_text().get_text()
                if txt == '✓':
                    cell.set_text_props(color='#53d769', fontweight='bold')
                elif txt == '✗':
                    cell.set_text_props(color='#ff4444', fontweight='bold')

        ax.set_title('Knapsack — Correctness Verification (vs Known Optimals)', pad=20)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURES, 'knapsack_correctness.png'))
        plt.close(fig)
        print("  ✓ knapsack_correctness.png")


# ============================================================================
#  Summary Plot (all problems combined)
# ============================================================================
def plot_summary():
    """Create a combined summary figure showing key metrics for all problems."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ── TSP ────────────────────────────────────────────────────────
    csv = os.path.join(RESULTS, 'tsp_results.csv')
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        if not df.empty:
            ax = axes[0]
            datasets = df['dataset'].unique()
            x = np.arange(len(datasets))
            w = 0.25
            for i, var in enumerate(['serial', 'openmp', 'cpu_opt']):
                sub = df[df['variant'] == var].set_index('dataset')
                vals = [sub.loc[d, 'time_s'] if d in sub.index else 0 for d in datasets]
                ax.bar(x + i * w, vals, w, label=VARIANT_LABELS[var],
                       color=COLORS_BRIGHT[var], edgecolor=COLORS[var], linewidth=1)
            ax.set_title('TSP — Execution Time')
            ax.set_xticks(x + w)
            ax.set_xticklabels(datasets, rotation=15, fontsize=8)
            ax.set_ylabel('Time (s)')
            ax.set_yscale('log')
            ax.legend(fontsize=8)
            ax.grid(axis='y', linestyle='--')

    # ── MCP ────────────────────────────────────────────────────────
    csv = os.path.join(RESULTS, 'mcp_results.csv')
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        if not df.empty:
            ax = axes[1]
            datasets = df['dataset'].unique()
            variants = df['variant'].unique()
            x = np.arange(len(datasets))
            w = 0.8 / max(len(variants), 1)
            for i, var in enumerate(variants):
                sub = df[df['variant'] == var].set_index('dataset')
                vals = [sub.loc[d, 'time_s'] if d in sub.index else 0 for d in datasets]
                ax.bar(x + i * w, vals, w, label=VARIANT_LABELS.get(var, var),
                       color=COLORS_BRIGHT.get(var, '#888'), edgecolor=COLORS.get(var, '#444'), linewidth=1)
            ax.set_title('MCP — Execution Time')
            ax.set_xticks(x + w * (len(variants) - 1) / 2)
            ax.set_xticklabels(datasets, rotation=15, fontsize=8)
            ax.set_ylabel('Time (s)')
            ax.set_yscale('log')
            ax.legend(fontsize=8)
            ax.grid(axis='y', linestyle='--')

    # ── Knapsack ──────────────────────────────────────────────────
    csv = os.path.join(RESULTS, 'knapsack_results.csv')
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        if not df.empty:
            ax = axes[2]
            df['short_name'] = df['n'].astype(int).apply(lambda x: f'n={x}')
            datasets = df['dataset'].unique()
            short_names = [df[df['dataset'] == d]['short_name'].iloc[0] for d in datasets]
            x = np.arange(len(datasets))
            w = 0.25
            for i, var in enumerate(['serial', 'openmp', 'cpu_opt']):
                sub = df[df['variant'] == var].set_index('dataset')
                vals = [sub.loc[d, 'time_s'] if d in sub.index else 0 for d in datasets]
                ax.bar(x + i * w, vals, w, label=VARIANT_LABELS[var],
                       color=COLORS_BRIGHT[var], edgecolor=COLORS[var], linewidth=1)
            ax.set_title('Knapsack — Execution Time')
            ax.set_xticks(x + w)
            ax.set_xticklabels(short_names, rotation=15, fontsize=8)
            ax.set_ylabel('Time (s)')
            ax.set_yscale('log')
            ax.legend(fontsize=8)
            ax.grid(axis='y', linestyle='--')

    fig.suptitle('Benchmark Summary — Serial vs OpenMP vs CPU-Optimized',
                 fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, 'benchmark_summary.png'))
    plt.close(fig)
    print("  ✓ benchmark_summary.png")


# ============================================================================
#  Main
# ============================================================================
if __name__ == '__main__':
    print("================================================================")
    print("  Generating benchmark plots...")
    print(f"  Results: {RESULTS}")
    print(f"  Figures: {FIGURES}")
    print("================================================================")

    print("\n── TSP ──")
    plot_tsp()

    print("\n── MCP ──")
    plot_mcp()

    print("\n── Knapsack ──")
    plot_knapsack()

    print("\n── Summary ──")
    plot_summary()

    print("\n================================================================")
    print("  All plots generated!")
    print(f"  Check: {FIGURES}/")
    print("================================================================")
