"""
Convert PrimeKG subgraph to PyTorch Geometric format.
"""

import json
import torch
from torch_geometric.data import HeteroData
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import csv


def convert_to_pyg(
    nodes: Dict,
    edges: List,
    output_path: Optional[str] = None,
) -> HeteroData:
    """
    Convert subgraph to PyG HeteroData format.

    Args:
        nodes: Dict mapping (type, id) -> node_info
        edges: List of (src_key, relation, dst_key) tuples
        output_path: Optional path to save the HeteroData object

    Returns:
        PyG HeteroData object
    """
    data = HeteroData()

    # Group nodes by type
    nodes_by_type = defaultdict(list)
    for key, info in nodes.items():
        node_type = info['type']
        nodes_by_type[node_type].append((key, info))

    # Create node type to local index mapping
    node_to_idx = {}  # (type, id) -> local_index
    for node_type, node_list in nodes_by_type.items():
        for local_idx, (key, info) in enumerate(node_list):
            node_to_idx[key] = local_idx

        # Store number of nodes for each type
        data[node_type].num_nodes = len(node_list)

        # Store node names as metadata
        names = [info['name'] for key, info in node_list]
        data[node_type].node_names = names

    # Group edges by type
    edges_by_type = defaultdict(list)
    for src_key, relation, dst_key in edges:
        src_type = nodes[src_key]['type']
        dst_type = nodes[dst_key]['type']

        # Normalize relation name for PyG (remove special characters)
        rel_normalized = relation.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')

        edge_type = (src_type, rel_normalized, dst_type)

        src_idx = node_to_idx[src_key]
        dst_idx = node_to_idx[dst_key]

        edges_by_type[edge_type].append((src_idx, dst_idx))

    # Create edge indices
    for edge_type, edge_list in edges_by_type.items():
        if edge_list:
            src_indices = [e[0] for e in edge_list]
            dst_indices = [e[1] for e in edge_list]

            data[edge_type].edge_index = torch.tensor(
                [src_indices, dst_indices],
                dtype=torch.long
            )

    # Print summary
    print(f"Created HeteroData with:")
    print(f"  Node types: {data.node_types}")
    print(f"  Edge types: {data.edge_types}")
    for node_type in data.node_types:
        print(f"  {node_type}: {data[node_type].num_nodes} nodes")
    for edge_type in data.edge_types:
        print(f"  {edge_type}: {data[edge_type].edge_index.size(1)} edges")

    # Save if path provided
    if output_path:
        torch.save(data, output_path)
        print(f"Saved to {output_path}")

    return data


def load_primekg_subgraph(
    subgraph_dir: str = 'data/primekg',
    load_cached: bool = True,
) -> HeteroData:
    """
    Load PrimeKG celiac subgraph as PyG HeteroData.

    Args:
        subgraph_dir: Directory containing subgraph files
        load_cached: Whether to load cached .pt file if available

    Returns:
        PyG HeteroData object
    """
    subgraph_dir = Path(subgraph_dir)

    # Check for cached PyG file
    pt_path = subgraph_dir / 'ced_subgraph.pt'
    if load_cached and pt_path.exists():
        print(f"Loading cached HeteroData from {pt_path}")
        return torch.load(pt_path)

    # Load from JSON files
    nodes_path = subgraph_dir / 'ced_subgraph_nodes.json'
    edges_path = subgraph_dir / 'ced_subgraph_edges.json'

    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError(
            f"Subgraph files not found in {subgraph_dir}. "
            "Run extract_ced_subgraph() first."
        )

    with open(nodes_path, 'r') as f:
        nodes_serialized = json.load(f)

    with open(edges_path, 'r') as f:
        edges_serialized = json.load(f)

    # Reconstruct nodes dict
    nodes = {}
    for key_str, info in nodes_serialized.items():
        parts = key_str.split('::')
        key = (parts[0], parts[1])
        nodes[key] = info

    # Reconstruct edges list
    edges = []
    for e in edges_serialized:
        src_key = (e['src_type'], e['src_id'])
        dst_key = (e['dst_type'], e['dst_id'])
        edges.append((src_key, e['relation'], dst_key))

    # Convert to PyG
    data = convert_to_pyg(nodes, edges, output_path=str(pt_path))

    return data


def convert_curated_kg_to_pyg(
    data_dir: str = 'data/processed/pyg',
    output_path: Optional[str] = None,
) -> HeteroData:
    """
    Convert the curated CeD knowledge graph to PyG HeteroData.

    This loads from the existing CSV files in data/processed/pyg/.

    Args:
        data_dir: Directory containing node and edge CSV files
        output_path: Optional path to save the HeteroData

    Returns:
        PyG HeteroData object
    """
    data_dir = Path(data_dir)
    data = HeteroData()

    # Load metadata
    metadata_path = data_dir / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # Load node files
    node_files = list(data_dir.glob('nodes_*.csv'))
    node_to_idx = {}

    for node_file in node_files:
        node_type = node_file.stem.replace('nodes_', '')

        with open(node_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            continue

        # Map node IDs to local indices
        for local_idx, row in enumerate(rows):
            node_id = row.get('id', row.get('node_id', str(local_idx)))
            node_to_idx[(node_type, node_id)] = local_idx

        data[node_type].num_nodes = len(rows)

        # Store node names if available
        if 'name' in rows[0]:
            data[node_type].node_names = [r['name'] for r in rows]

        print(f"Loaded {len(rows)} {node_type} nodes")

    # Load edge files
    edge_files = list(data_dir.glob('edges_*.csv'))

    for edge_file in edge_files:
        with open(edge_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            continue

        # Determine edge type from filename or first row
        filename_parts = edge_file.stem.replace('edges_', '').split('_')

        if len(filename_parts) >= 3:
            src_type = filename_parts[0]
            relation = '_'.join(filename_parts[1:-1])
            dst_type = filename_parts[-1]
        else:
            # Try to get from row data
            first_row = rows[0]
            src_type = first_row.get('src_type', 'gene')
            relation = first_row.get('relation', 'related_to')
            dst_type = first_row.get('dst_type', 'phenotype')

        edge_type = (src_type, relation, dst_type)

        # Build edge index
        src_indices = []
        dst_indices = []

        for row in rows:
            src_id = row.get('src', row.get('source', row.get('src_id')))
            dst_id = row.get('dst', row.get('target', row.get('dst_id')))

            src_key = (src_type, str(src_id))
            dst_key = (dst_type, str(dst_id))

            if src_key in node_to_idx and dst_key in node_to_idx:
                src_indices.append(node_to_idx[src_key])
                dst_indices.append(node_to_idx[dst_key])

        if src_indices:
            data[edge_type].edge_index = torch.tensor(
                [src_indices, dst_indices],
                dtype=torch.long
            )
            print(f"Loaded {len(src_indices)} {edge_type} edges")

    # Save if path provided
    if output_path:
        torch.save(data, output_path)
        print(f"Saved to {output_path}")

    return data


def add_node_features(
    data: HeteroData,
    feature_dim: int = 64,
    init: str = 'xavier',
) -> HeteroData:
    """
    Add learnable node features to HeteroData.

    Args:
        data: HeteroData object
        feature_dim: Feature dimension
        init: Initialization method ('xavier', 'normal', 'ones')

    Returns:
        HeteroData with node features
    """
    for node_type in data.node_types:
        num_nodes = data[node_type].num_nodes

        if init == 'xavier':
            x = torch.empty(num_nodes, feature_dim)
            torch.nn.init.xavier_uniform_(x)
        elif init == 'normal':
            x = torch.randn(num_nodes, feature_dim)
        elif init == 'ones':
            x = torch.ones(num_nodes, feature_dim)
        else:
            raise ValueError(f"Unknown init: {init}")

        data[node_type].x = x

    return data
