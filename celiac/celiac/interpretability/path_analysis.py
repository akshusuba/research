"""
Multi-hop path analysis for knowledge graph interpretability.

Extracts and ranks paths between nodes to explain predictions.
"""

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict
from dataclasses import dataclass
import heapq


@dataclass
class Path:
    """Represents a path in the knowledge graph."""
    nodes: List[Tuple[str, int, str]]  # [(type, idx, name), ...]
    edges: List[Tuple[str, str, str]]  # [(src_type, rel, dst_type), ...]
    score: float = 0.0
    length: int = 0

    def __post_init__(self):
        self.length = len(self.edges)

    def __repr__(self):
        path_str = " → ".join([f"{n[2]}" for n in self.nodes])
        return f"Path(score={self.score:.4f}, length={self.length}): {path_str}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'nodes': self.nodes,
            'edges': self.edges,
            'score': self.score,
            'length': self.length,
            'path_string': " → ".join([f"{n[2]}" for n in self.nodes]),
        }


def build_adjacency_dict(data: HeteroData) -> Dict[Tuple[str, int], List[Tuple[Tuple[str, int], str]]]:
    """
    Build adjacency dictionary for path finding.

    Returns:
        Dict mapping (node_type, node_idx) -> [(neighbor_type, neighbor_idx, relation), ...]
    """
    adj = defaultdict(list)

    for edge_type in data.edge_types:
        src_type, rel, dst_type = edge_type
        if not hasattr(data[edge_type], 'edge_index'):
            continue

        edge_index = data[edge_type].edge_index

        for i in range(edge_index.size(1)):
            src_idx = edge_index[0, i].item()
            dst_idx = edge_index[1, i].item()

            # Add both directions for undirected traversal
            adj[(src_type, src_idx)].append(((dst_type, dst_idx), rel))
            adj[(dst_type, dst_idx)].append(((src_type, src_idx), f"rev_{rel}"))

    return adj


def find_paths(
    data: HeteroData,
    source: Tuple[str, int],
    target: Tuple[str, int],
    max_length: int = 3,
    max_paths: int = 100,
) -> List[Path]:
    """
    Find all paths between source and target up to max_length.

    Uses BFS to enumerate paths.

    Args:
        data: HeteroData object
        source: (node_type, node_idx) for source
        target: (node_type, node_idx) for target
        max_length: Maximum path length
        max_paths: Maximum number of paths to return

    Returns:
        List of Path objects
    """
    adj = build_adjacency_dict(data)

    # Get node names for readability
    node_names = {}
    for node_type in data.node_types:
        names = data[node_type].get('node_names', [])
        for i in range(data[node_type].num_nodes):
            name = names[i] if i < len(names) else f'{node_type}_{i}'
            node_names[(node_type, i)] = name

    paths = []

    # BFS with path tracking
    # State: (current_node, path_nodes, path_edges, visited)
    queue = [(source, [(source[0], source[1], node_names.get(source, str(source)))], [], {source})]

    while queue and len(paths) < max_paths:
        current, path_nodes, path_edges, visited = queue.pop(0)

        if current == target and len(path_edges) > 0:
            paths.append(Path(
                nodes=path_nodes,
                edges=path_edges,
                score=0.0,  # Will be scored later
            ))
            continue

        if len(path_edges) >= max_length:
            continue

        # Explore neighbors
        for (neighbor, rel) in adj.get(current, []):
            if neighbor not in visited:
                new_visited = visited | {neighbor}
                new_edges = path_edges + [(current[0], rel, neighbor[0])]
                new_nodes = path_nodes + [(neighbor[0], neighbor[1], node_names.get(neighbor, str(neighbor)))]
                queue.append((neighbor, new_nodes, new_edges, new_visited))

    return paths


def score_paths(
    paths: List[Path],
    model: nn.Module,
    data: HeteroData,
    edge_weights: Optional[Dict[Tuple[str, str, str], float]] = None,
    length_penalty: float = 0.8,
) -> List[Path]:
    """
    Score paths based on model predictions and path properties.

    Scoring considers:
    - Model's edge prediction scores along the path
    - Path length (shorter = better)
    - Edge type importance weights

    Args:
        paths: List of paths to score
        model: Trained model for edge scoring
        data: HeteroData object
        edge_weights: Optional dict of edge type importance weights
        length_penalty: Penalty factor per hop (e.g., 0.8 means 20% penalty per hop)

    Returns:
        List of paths sorted by score (descending)
    """
    model.eval()

    if edge_weights is None:
        edge_weights = {}

    for path in paths:
        score = 1.0

        # Length penalty
        score *= (length_penalty ** path.length)

        # Edge type weights
        for edge_type in path.edges:
            weight = edge_weights.get(edge_type, 1.0)
            score *= weight

        path.score = score

    # Sort by score
    paths.sort(key=lambda p: p.score, reverse=True)

    return paths


def extract_top_paths(
    data: HeteroData,
    source_type: str,
    target_type: str,
    model: Optional[nn.Module] = None,
    max_hops: int = 3,
    top_k: int = 10,
    source_indices: Optional[List[int]] = None,
    target_indices: Optional[List[int]] = None,
) -> List[Path]:
    """
    Extract and rank top paths between node types.

    Args:
        data: HeteroData object
        source_type: Source node type
        target_type: Target node type
        model: Optional trained model for scoring
        max_hops: Maximum path length
        top_k: Number of top paths to return
        source_indices: Specific source nodes to consider
        target_indices: Specific target nodes to consider

    Returns:
        List of top-k paths
    """
    # Get node indices
    if source_indices is None:
        source_indices = list(range(min(10, data[source_type].num_nodes)))
    if target_indices is None:
        target_indices = list(range(min(10, data[target_type].num_nodes)))

    all_paths = []

    for src_idx in source_indices:
        for tgt_idx in target_indices:
            source = (source_type, src_idx)
            target = (target_type, tgt_idx)

            paths = find_paths(data, source, target, max_length=max_hops, max_paths=20)
            all_paths.extend(paths)

    # Score paths
    if model is not None:
        all_paths = score_paths(all_paths, model, data)
    else:
        # Score by length only
        for path in all_paths:
            path.score = 1.0 / (path.length + 1)
        all_paths.sort(key=lambda p: p.score, reverse=True)

    return all_paths[:top_k]


def visualize_path(
    path: Path,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 4),
) -> None:
    """
    Visualize a single path as a diagram.

    Args:
        path: Path object to visualize
        output_path: Path to save figure
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=figsize)

    # Node positions
    n_nodes = len(path.nodes)
    x_positions = np.linspace(0.1, 0.9, n_nodes)
    y_pos = 0.5

    # Node type colors
    type_colors = {
        'gene': '#4CAF50',
        'microbe': '#2196F3',
        'metabolite': '#FF9800',
        'phenotype': '#E91E63',
    }

    # Draw nodes
    for i, (node_type, node_idx, node_name) in enumerate(path.nodes):
        color = type_colors.get(node_type, '#9E9E9E')

        # Node circle
        circle = plt.Circle((x_positions[i], y_pos), 0.06, color=color, alpha=0.8)
        ax.add_patch(circle)

        # Node label
        label = node_name[:20] + '...' if len(node_name) > 20 else node_name
        ax.text(x_positions[i], y_pos - 0.15, label, ha='center', va='top', fontsize=9)
        ax.text(x_positions[i], y_pos + 0.12, node_type, ha='center', va='bottom', fontsize=8, alpha=0.7)

    # Draw edges
    for i, (src_type, rel, dst_type) in enumerate(path.edges):
        x_start = x_positions[i] + 0.06
        x_end = x_positions[i + 1] - 0.06

        ax.annotate(
            '', xy=(x_end, y_pos), xytext=(x_start, y_pos),
            arrowprops=dict(arrowstyle='->', color='gray', lw=2)
        )

        # Edge label
        rel_label = rel.replace('rev_', '←')[:15]
        ax.text((x_start + x_end) / 2, y_pos + 0.05, rel_label,
               ha='center', va='bottom', fontsize=8, alpha=0.8)

    # Score label
    ax.text(0.5, 0.1, f'Path Score: {path.score:.4f}', ha='center', fontsize=10, transform=ax.transAxes)

    # Legend
    legend_patches = [mpatches.Patch(color=c, label=t) for t, c in type_colors.items()]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=8)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(f'Knowledge Graph Path (Length: {path.length})', fontsize=12)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    else:
        plt.show()

    plt.close()


def find_motifs(
    data: HeteroData,
    motif_pattern: List[Tuple[str, str, str]],
    max_instances: int = 100,
) -> List[List[Tuple[str, int]]]:
    """
    Find instances of a specific path motif in the graph.

    Args:
        data: HeteroData object
        motif_pattern: List of edge types defining the motif
        max_instances: Maximum instances to return

    Returns:
        List of node sequences matching the motif
    """
    adj = build_adjacency_dict(data)
    instances = []

    # Start from all nodes of the first edge type's source
    first_edge = motif_pattern[0]
    src_type = first_edge[0]

    for start_idx in range(data[src_type].num_nodes):
        # DFS to find matching paths
        stack = [((src_type, start_idx), [(src_type, start_idx)], 0)]

        while stack and len(instances) < max_instances:
            current, path, edge_idx = stack.pop()

            if edge_idx >= len(motif_pattern):
                instances.append(path)
                continue

            expected_edge = motif_pattern[edge_idx]
            expected_rel = expected_edge[1]
            expected_dst_type = expected_edge[2]

            for (neighbor, rel) in adj.get(current, []):
                if rel == expected_rel and neighbor[0] == expected_dst_type:
                    stack.append((neighbor, path + [neighbor], edge_idx + 1))

        if len(instances) >= max_instances:
            break

    return instances


def get_path_statistics(paths: List[Path]) -> Dict[str, Any]:
    """Get statistics about a collection of paths."""
    if not paths:
        return {}

    lengths = [p.length for p in paths]
    scores = [p.score for p in paths]

    # Count edge types
    edge_type_counts = defaultdict(int)
    for path in paths:
        for edge_type in path.edges:
            edge_type_counts[edge_type] += 1

    # Count node types
    node_type_counts = defaultdict(int)
    for path in paths:
        for node_type, _, _ in path.nodes:
            node_type_counts[node_type] += 1

    return {
        'num_paths': len(paths),
        'avg_length': np.mean(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'avg_score': np.mean(scores),
        'edge_type_counts': dict(edge_type_counts),
        'node_type_counts': dict(node_type_counts),
    }
