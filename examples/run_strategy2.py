#!/usr/bin/env python
"""
Strategy 2 Example: Mapper Algorithm for Exchange Flow Networks.

Builds a Mapper graph on synthetic exchange data with embedded
manipulation patterns and detects:
  - Bottleneck nodes (high centrality exchanges)
  - Communities (coordinated groups)
  - Ring trading patterns (cycles)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.mapper_algorithm import MapperAlgorithm, ExchangeFlowAnalyzer


def generate_realistic_exchange_data(n_legitimate=20, n_suspicious=5, seed=42):
    """
    Generate realistic exchange flow features with embedded manipulation.

    Features:
      - total_volume: 24h trading volume
      - velocity: trades per hour
      - in_out_ratio: deposits / withdrawals
      - unique_counterparts: distinct trading partners
      - avg_trade_size: mean trade size
      - max_to_avg_ratio: largest trade / average (whale indicator)
    """
    np.random.seed(seed)

    legitimate = np.column_stack([
        np.random.lognormal(mean=15, sigma=1.5, size=n_legitimate),
        np.random.gamma(shape=2, scale=50, size=n_legitimate),
        np.random.normal(loc=1.0, scale=0.2, size=n_legitimate),
        np.random.gamma(shape=3, scale=30, size=n_legitimate),
        np.random.lognormal(mean=8, sigma=0.5, size=n_legitimate),
        np.random.gamma(shape=2, scale=2, size=n_legitimate),
    ])

    suspicious = np.column_stack([
        np.random.lognormal(mean=14, sigma=0.3, size=n_suspicious),
        np.random.gamma(shape=10, scale=80, size=n_suspicious),
        np.random.normal(loc=1.0, scale=0.05, size=n_suspicious),
        np.random.gamma(shape=1, scale=3, size=n_suspicious),
        np.random.lognormal(mean=8, sigma=0.2, size=n_suspicious),
        np.random.gamma(shape=10, scale=5, size=n_suspicious),
    ])

    X = np.vstack([legitimate, suspicious])
    columns = ['total_volume', 'velocity', 'in_out_ratio',
               'unique_counterparts', 'avg_trade_size', 'max_to_avg_ratio']

    labels = ['legit'] * n_legitimate + ['suspicious'] * n_suspicious
    names = ([f'Exchange_L{i:02d}' for i in range(n_legitimate)]
             + [f'Exchange_S{i:02d}' for i in range(n_suspicious)])

    df = pd.DataFrame(X, columns=columns, index=names)
    df['true_label'] = labels
    return df


def visualize_mapper_graph(analyzer, true_labels, save_path='data/persistence/mapper_graph.png'):
    """Plot the Mapper graph using networkx."""
    try:
        import networkx as nx
    except ImportError:
        print("  [Warning] networkx required for visualization")
        return

    G = analyzer.mapper.to_networkx()
    if len(G.nodes) == 0:
        print("  [Warning] Empty Mapper graph")
        return

    suspicious_nodes = []
    for node_id, attrs in analyzer.mapper.nodes.items():
        members = attrs['members']
        sus_count = sum(1 for m in members if true_labels[m] == 'suspicious')
        if sus_count > 0:
            suspicious_nodes.append(node_id)

    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42, k=2)

    sizes = [analyzer.mapper.nodes[n]['size'] * 200 for n in G.nodes()]
    colors = ['red' if n in suspicious_nodes else 'lightblue' for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=colors,
                          alpha=0.8, edgecolors='black', linewidths=1.5, ax=ax)

    edge_weights = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.6, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    ax.set_title('Exchange Flow Mapper Graph\n(red = contains suspicious exchanges)',
                 fontsize=12)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved Mapper graph: {save_path}")


def run_strategy_2():
    """Run Strategy 2 demo with synthetic data."""

    print(f"\n{'=' * 70}")
    print(f"Strategy 2: Mapper Algorithm for Exchange Flow Networks")
    print(f"{'=' * 70}\n")

    print("Step 1: Generating exchange flow data...")
    df = generate_realistic_exchange_data(n_legitimate=20, n_suspicious=5)
    true_labels = df['true_label'].tolist()
    feature_df = df.drop(columns=['true_label'])
    print(f"  Generated {len(df)} exchanges (20 legit + 5 suspicious)")

    print(f"\n  Sample data:")
    print(feature_df.describe().round(2))

    print(f"\nStep 2: Building Mapper graph...")
    analyzer = ExchangeFlowAnalyzer(mapper_kwargs={
        'n_intervals': 8,
        'overlap': 0.4,
        'lens_function': 'l2_norm',
    })
    analyzer.fit(feature_df)
    print(f"  Built Mapper graph with {len(analyzer.mapper.nodes)} nodes, "
          f"{len(analyzer.mapper.edges)} edges")

    print(f"\nStep 3: Detecting anomalies...")
    report = analyzer.report()

    print(f"\n  Graph Summary:")
    for k, v in report['graph_summary'].items():
        if isinstance(v, float):
            print(f"    {k}: {v:.2f}")
        else:
            print(f"    {k}: {v}")

    print(f"\n  Bottleneck nodes (high centrality = critical exchanges):")
    if report['bottleneck_nodes']:
        for node_id, score in report['bottleneck_nodes']:
            members = analyzer.mapper.nodes[node_id]['members']
            sus_in_node = sum(1 for m in members if true_labels[m] == 'suspicious')
            print(f"    Node {node_id}: centrality={score:.3f}, "
                  f"size={len(members)}, suspicious={sus_in_node}")
    else:
        print(f"    None detected")

    print(f"\n  Communities found: {report['n_communities']}")
    print(f"  Community sizes: {report['community_sizes']}")
    print(f"  Ring patterns (cycles): {report['ring_patterns_found']}")

    print(f"\nStep 4: Visualizing Mapper graph...")
    os.makedirs('data/persistence', exist_ok=True)
    visualize_mapper_graph(analyzer, true_labels)

    print(f"\nStep 5: Anomaly detection performance...")
    suspicious_in_bottleneck = 0
    total_in_bottleneck = 0
    for node_id, _ in report['bottleneck_nodes']:
        members = analyzer.mapper.nodes[node_id]['members']
        total_in_bottleneck += len(members)
        suspicious_in_bottleneck += sum(1 for m in members if true_labels[m] == 'suspicious')

    if total_in_bottleneck > 0:
        precision = suspicious_in_bottleneck / total_in_bottleneck
        recall = suspicious_in_bottleneck / 5
        print(f"  Suspicious flagged in bottleneck nodes: {suspicious_in_bottleneck}/5")
        print(f"  Precision: {precision:.2%}")
        print(f"  Recall: {recall:.2%}")

    return analyzer, report


if __name__ == '__main__':
    analyzer, report = run_strategy_2()
