#!/usr/bin/env python
"""
Regenerate fig4_progression with full v1-v9 narrative.

The original v6 paper figure showed v1-v5. The revised paper extends the
methodology through v9, so the figure must reflect that.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['savefig.dpi'] = 300


def main():
    versions = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9']
    accuracies     = [None,  50.75, 56.18, 60.42, 59.40, 61.77, 68.46, None,  62.23]
    wilson_lower   = [None,  47.06, 47.98, 42.53, 55.63, 59.75, 67.0,  None,  61.09]
    n_signals      = [2,     728,   152,   26,    684,   2260,  2911,  None,  4266]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    valid_acc = [(v, a, l) for v, a, l in zip(versions, accuracies, wilson_lower) if a is not None]
    vs, accs, los = zip(*valid_acc)
    accs = np.array(accs)
    los = np.array(los)
    yerr = accs - los
    colors = ['#2ecc71' if l > 50 else '#e74c3c' for l in los]

    axes[0].bar(vs, accs, color=colors, alpha=0.85, edgecolor='black', linewidth=1.0)
    axes[0].errorbar(vs, accs, yerr=yerr, fmt='none', color='black',
                     capsize=8, capthick=2, linewidth=2)
    axes[0].axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Chance (50%)')
    axes[0].set_ylabel('Direction Accuracy (%)', fontsize=12)
    axes[0].set_xlabel('Validation Version', fontsize=12)
    axes[0].set_title('Validation Progression v1→v9\n(95% Wilson CI lower bound)',
                      fontsize=13, fontweight='bold')
    axes[0].set_ylim(40, 78)
    axes[0].legend(loc='lower right')

    annotations = {
        'v1': '(invalid n=2)',
        'v8': '(walk-forward,\nSharpe 0.95)',
    }
    for i, v in enumerate(vs):
        if v in annotations:
            axes[0].annotate(annotations[v], xy=(i, accs[i]),
                             xytext=(0, 12), textcoords='offset points',
                             ha='center', fontsize=8, color='#444444')

    valid_n = [(v, n) for v, n in zip(versions, n_signals) if n is not None]
    vs_n, ns = zip(*valid_n)
    bar_colors = []
    for v in vs_n:
        if v in ['v8']:
            bar_colors.append('#9b59b6')
        elif v in ['v6', 'v9']:
            bar_colors.append('#27ae60')
        else:
            bar_colors.append('#3498db')

    axes[1].bar(vs_n, ns, color=bar_colors, alpha=0.85, edgecolor='black')
    axes[1].set_yscale('log')
    axes[1].set_ylabel('Total Test Signals (log scale)', fontsize=12)
    axes[1].set_xlabel('Validation Version', fontsize=12)
    axes[1].set_title('Sample Size Progression\n(more samples → tighter CI)',
                      fontsize=13, fontweight='bold')
    for i, n in enumerate(ns):
        axes[1].text(i, n * 1.25, f'{int(n):,}', ha='center', fontsize=9)

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color='#3498db', alpha=0.85,
                       label='CV-only validation'),
        plt.Rectangle((0, 0), 1, 1, color='#27ae60', alpha=0.85,
                       label='Headline result (v6) / Rigorous (v9)'),
        plt.Rectangle((0, 0), 1, 1, color='#9b59b6', alpha=0.85,
                       label='Walk-forward profitability (v8)'),
    ]
    axes[1].legend(handles=legend_elements, loc='upper left', fontsize=8)

    plt.tight_layout()
    out_dir = 'results/figures'
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f'{out_dir}/fig4_progression.pdf', bbox_inches='tight')
    plt.savefig(f'{out_dir}/fig4_progression.png', bbox_inches='tight')
    plt.close()
    print(f"Wrote: {out_dir}/fig4_progression.{{pdf,png}}")


if __name__ == '__main__':
    main()
