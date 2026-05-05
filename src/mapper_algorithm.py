"""
Mapper Algorithm: Build simplified topological networks from data.

Used for Strategy 2: Exchange flow networks.
Identifies bottleneck structures, communities, and ring-trading patterns.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class MapperAlgorithm:
    """Construct Mapper graph from high-dimensional data."""

    def __init__(self,
                 n_intervals=10,
                 overlap=0.3,
                 clusterer=None,
                 lens_function='pca',
                 standardize=True):
        """
        Args:
            n_intervals: Number of overlapping intervals for the cover
            overlap: Fractional overlap between adjacent intervals (0-1)
            clusterer: sklearn-style clusterer (defaults to DBSCAN)
            lens_function: 'pca' (1D), 'pca_2d', 'l2_norm', or callable
            standardize: Z-score features before clustering (recommended)
        """
        self.n_intervals = n_intervals
        self.overlap = overlap
        self.clusterer = clusterer if clusterer is not None else DBSCAN(eps=0.8, min_samples=2)
        self.lens_function = lens_function
        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None

        self.lens_values = None
        self.cover_intervals = None
        self.nodes = {}
        self.edges = []

    def _apply_lens(self, X):
        """Project data to 1D or 2D using the lens function."""
        if callable(self.lens_function):
            return self.lens_function(X)
        elif self.lens_function == 'pca':
            return PCA(n_components=1).fit_transform(X).flatten()
        elif self.lens_function == 'pca_2d':
            return PCA(n_components=2).fit_transform(X)
        elif self.lens_function == 'l2_norm':
            return np.linalg.norm(X, axis=1)
        else:
            raise ValueError(f"Unknown lens: {self.lens_function}")

    def _create_cover(self, lens_values):
        """Create overlapping intervals covering the lens range."""
        lens_min = lens_values.min()
        lens_max = lens_values.max()
        lens_range = lens_max - lens_min

        interval_length = lens_range / self.n_intervals * (1 + self.overlap)
        step = lens_range / self.n_intervals

        intervals = []
        for i in range(self.n_intervals):
            start = lens_min + i * step - (interval_length - step) / 2
            end = start + interval_length
            intervals.append((start, end))

        return intervals

    def fit(self, X):
        """
        Build the Mapper graph.

        Args:
            X: (n_samples, n_features) array

        Returns:
            self with nodes and edges populated
        """
        X = np.asarray(X, dtype=float)
        if self.standardize:
            X = self.scaler.fit_transform(X)
        self.lens_values = self._apply_lens(X)

        if self.lens_values.ndim == 1:
            self.cover_intervals = self._create_cover(self.lens_values)
        else:
            raise NotImplementedError("2D Mapper not yet supported in this version")

        self.nodes = {}
        node_membership = {}
        node_id = 0

        for interval_idx, (low, high) in enumerate(self.cover_intervals):
            in_interval = (self.lens_values >= low) & (self.lens_values <= high)
            indices = np.where(in_interval)[0]

            if len(indices) < 2:
                continue

            X_subset = X[indices]
            try:
                labels = self.clusterer.fit_predict(X_subset)
            except Exception:
                labels = np.zeros(len(indices), dtype=int)

            unique_labels = [l for l in set(labels) if l >= 0]

            for label in unique_labels:
                cluster_mask = labels == label
                cluster_indices = indices[cluster_mask]

                self.nodes[node_id] = {
                    'interval': interval_idx,
                    'label': label,
                    'members': set(cluster_indices.tolist()),
                    'size': len(cluster_indices),
                    'lens_mean': float(self.lens_values[cluster_indices].mean()),
                    'centroid': X[cluster_indices].mean(axis=0).tolist(),
                }

                for idx in cluster_indices:
                    node_membership.setdefault(int(idx), []).append(node_id)

                node_id += 1

        self.edges = []
        seen_edges = set()
        for sample_idx, node_ids in node_membership.items():
            if len(node_ids) < 2:
                continue
            for i in range(len(node_ids)):
                for j in range(i + 1, len(node_ids)):
                    edge = tuple(sorted([node_ids[i], node_ids[j]]))
                    if edge in seen_edges:
                        continue
                    seen_edges.add(edge)
                    overlap_size = len(self.nodes[edge[0]]['members']
                                       & self.nodes[edge[1]]['members'])
                    self.edges.append({
                        'source': edge[0], 'target': edge[1], 'weight': overlap_size,
                    })

        return self

    def to_networkx(self):
        """Export Mapper graph to networkx (if installed)."""
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("Install networkx: pip install networkx")

        G = nx.Graph()
        for node_id, attrs in self.nodes.items():
            G.add_node(node_id, **{k: v for k, v in attrs.items() if k != 'members'})
        for edge in self.edges:
            G.add_edge(edge['source'], edge['target'], weight=edge['weight'])
        return G

    def graph_summary(self):
        """Compute network-level statistics."""
        return {
            'n_nodes': len(self.nodes),
            'n_edges': len(self.edges),
            'avg_node_size': np.mean([n['size'] for n in self.nodes.values()]) if self.nodes else 0,
            'max_node_size': max([n['size'] for n in self.nodes.values()]) if self.nodes else 0,
            'avg_edge_weight': np.mean([e['weight'] for e in self.edges]) if self.edges else 0,
        }


class ExchangeFlowAnalyzer:
    """Detect manipulation patterns via Mapper on exchange flow networks."""

    def __init__(self, mapper_kwargs=None):
        self.mapper_kwargs = mapper_kwargs or {'n_intervals': 8, 'overlap': 0.3}
        self.mapper = None
        self.anomaly_scores = None

    def fit(self, exchange_features):
        """
        Args:
            exchange_features: (n_exchanges, n_features) DataFrame or array
                Features: total_volume, velocity, in_out_ratio, etc.
        """
        if isinstance(exchange_features, pd.DataFrame):
            X = exchange_features.values
            self.exchange_ids = exchange_features.index.tolist()
        else:
            X = np.asarray(exchange_features)
            self.exchange_ids = list(range(len(X)))

        self.mapper = MapperAlgorithm(**self.mapper_kwargs)
        self.mapper.fit(X)
        return self

    def detect_bottlenecks(self, top_k=3):
        """Identify nodes with high betweenness centrality (bottlenecks)."""
        try:
            import networkx as nx
            G = self.mapper.to_networkx()
            if len(G.nodes) == 0:
                return []
            centrality = nx.betweenness_centrality(G)
            sorted_nodes = sorted(centrality.items(), key=lambda x: -x[1])
            return sorted_nodes[:top_k]
        except ImportError:
            print("[Warning] networkx not installed -- skipping centrality")
            return []

    def detect_communities(self):
        """Find connected components / communities."""
        try:
            import networkx as nx
            G = self.mapper.to_networkx()
            return list(nx.connected_components(G))
        except ImportError:
            return []

    def detect_ring_patterns(self, min_cycle_length=3):
        """Find cycles in the Mapper graph (potential ring trading)."""
        try:
            import networkx as nx
            G = self.mapper.to_networkx()
            cycles = nx.cycle_basis(G)
            return [c for c in cycles if len(c) >= min_cycle_length]
        except ImportError:
            return []

    def report(self):
        """Generate anomaly detection report."""
        summary = self.mapper.graph_summary()
        bottlenecks = self.detect_bottlenecks()
        communities = self.detect_communities()
        rings = self.detect_ring_patterns()

        return {
            'graph_summary': summary,
            'bottleneck_nodes': bottlenecks,
            'n_communities': len(communities),
            'community_sizes': sorted([len(c) for c in communities], reverse=True),
            'ring_patterns_found': len(rings),
            'rings': [list(r) for r in rings[:5]],
        }


def demo_with_synthetic_data():
    """Run a demo with synthetic exchange flow data."""
    print("[Mapper Demo] Generating synthetic exchange flow data")

    np.random.seed(42)
    n_exchanges = 30

    cluster1 = np.random.randn(10, 4) + np.array([5, 5, 0, 0])
    cluster2 = np.random.randn(10, 4) + np.array([-5, -5, 0, 0])
    cluster3 = np.random.randn(8, 4) + np.array([0, 0, 5, 5])
    suspicious = np.random.randn(2, 4) * 0.1 + np.array([0, 0, 0, 0])

    X = np.vstack([cluster1, cluster2, cluster3, suspicious])
    feature_names = ['total_volume', 'velocity', 'in_out_ratio', 'unique_counterparts']
    df = pd.DataFrame(X, columns=feature_names)
    df.index = [f'Exchange_{i}' for i in range(len(df))]

    print(f"  Created {len(df)} exchanges with {len(feature_names)} features")

    print("[Mapper Demo] Building Mapper graph")
    analyzer = ExchangeFlowAnalyzer(mapper_kwargs={'n_intervals': 6, 'overlap': 0.4})
    analyzer.fit(df)

    print("[Mapper Demo] Running anomaly detection")
    report = analyzer.report()

    print(f"\nGraph Summary:")
    for k, v in report['graph_summary'].items():
        print(f"  {k}: {v}")

    print(f"\nBottlenecks (top centrality nodes):")
    for node_id, score in report['bottleneck_nodes']:
        print(f"  Node {node_id}: centrality={score:.3f}")

    print(f"\nCommunities found: {report['n_communities']}")
    print(f"Community sizes: {report['community_sizes']}")
    print(f"\nRing patterns detected: {report['ring_patterns_found']}")

    return analyzer, report


if __name__ == '__main__':
    demo_with_synthetic_data()
