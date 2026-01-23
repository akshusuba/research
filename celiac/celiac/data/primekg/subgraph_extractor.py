"""
Celiac disease subgraph extraction from PrimeKG.

Extracts a focused subgraph relevant to celiac disease and gut-brain axis research.
"""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import json


# Celiac disease seed nodes for subgraph extraction
CED_SEED_NODES = {
    # Diseases
    'disease': [
        'celiac disease',
        'gluten sensitivity',
        'dermatitis herpetiformis',
        'gluten ataxia',
        'celiac disease associated conditions',
    ],
    # Key genes (HLA, transglutaminases, tryptophan pathway, cytokines)
    'gene/protein': [
        'HLA-DQA1', 'HLA-DQB1', 'HLA-DQ2', 'HLA-DQ8',
        'TGM2', 'TGM3', 'TGM6',  # Transglutaminases
        'TPH1', 'TPH2',  # Tryptophan hydroxylase
        'SLC6A4',  # Serotonin transporter
        'IL15', 'IL21', 'IFNG', 'TNF',  # Cytokines
        'MICA', 'MICB',  # Stress molecules
        'CD4', 'CD8A',  # T cell markers
        'CXCR3', 'CCR9',  # Homing receptors
        'IDO1', 'IDO2',  # Indoleamine dioxygenases
    ],
    # Neurological phenotypes
    'effect/phenotype': [
        'ataxia', 'cerebellar ataxia', 'gait ataxia',
        'peripheral neuropathy', 'polyneuropathy',
        'cognitive impairment', 'brain fog',
        'depression', 'anxiety',
        'epilepsy', 'seizures',
        'headache', 'migraine',
        'encephalopathy',
    ],
    # Gut-brain related pathways
    'biological_process': [
        'tryptophan metabolic process',
        'serotonin biosynthetic process',
        'immune response',
        'inflammatory response',
        'T cell activation',
        'antigen processing and presentation',
    ],
}

# Neurological phenotype HPO IDs for more precise matching
NEURO_HPO_IDS = [
    'HP:0001251',  # Ataxia
    'HP:0007340',  # Gait ataxia
    'HP:0002460',  # Distal muscle weakness
    'HP:0009830',  # Peripheral neuropathy
    'HP:0001250',  # Seizures
    'HP:0002376',  # Cognitive impairment
    'HP:0000739',  # Anxiety
    'HP:0000716',  # Depression
    'HP:0002315',  # Headache
    'HP:0002076',  # Migraine
]


class SubgraphExtractor:
    """Extract celiac-relevant subgraph from PrimeKG."""

    def __init__(self, primekg_path: str):
        """
        Args:
            primekg_path: Path to PrimeKG CSV file
        """
        self.primekg_path = primekg_path
        self.nodes = {}  # (type, id) -> node_info
        self.edges = []  # List of edge tuples
        self.node_name_to_id = defaultdict(list)  # name -> [(type, id), ...]

    def load_primekg(self, show_progress: bool = True) -> None:
        """Load PrimeKG into memory."""
        print(f"Loading PrimeKG from {self.primekg_path}...")

        with open(self.primekg_path, 'r') as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader):
                # Extract node info
                x_type = row.get('x_type', 'unknown')
                x_id = row.get('x_id', row.get('x_index', str(i)))
                x_name = row.get('x_name', '').lower()

                y_type = row.get('y_type', 'unknown')
                y_id = row.get('y_id', row.get('y_index', str(i)))
                y_name = row.get('y_name', '').lower()

                relation = row.get('relation', 'related_to')

                # Store nodes
                x_key = (x_type, x_id)
                y_key = (y_type, y_id)

                if x_key not in self.nodes:
                    self.nodes[x_key] = {'type': x_type, 'id': x_id, 'name': x_name}
                    self.node_name_to_id[x_name].append(x_key)

                if y_key not in self.nodes:
                    self.nodes[y_key] = {'type': y_type, 'id': y_id, 'name': y_name}
                    self.node_name_to_id[y_name].append(y_key)

                # Store edge
                self.edges.append((x_key, relation, y_key))

                if show_progress and (i + 1) % 500000 == 0:
                    print(f"  Loaded {i + 1:,} edges...")

        print(f"Loaded {len(self.nodes):,} nodes and {len(self.edges):,} edges")

    def find_seed_nodes(
        self,
        seed_dict: Dict[str, List[str]] = None,
    ) -> Set[Tuple[str, str]]:
        """
        Find seed nodes by name matching.

        Args:
            seed_dict: Dict mapping node type -> list of names to match

        Returns:
            Set of (type, id) tuples for matched nodes
        """
        if seed_dict is None:
            seed_dict = CED_SEED_NODES

        seed_nodes = set()

        for node_type, names in seed_dict.items():
            for name in names:
                name_lower = name.lower()

                # Exact match
                if name_lower in self.node_name_to_id:
                    for key in self.node_name_to_id[name_lower]:
                        if key[0] == node_type or node_type in key[0]:
                            seed_nodes.add(key)

                # Partial match
                for node_name, keys in self.node_name_to_id.items():
                    if name_lower in node_name or node_name in name_lower:
                        for key in keys:
                            if key[0] == node_type or node_type in key[0]:
                                seed_nodes.add(key)

        print(f"Found {len(seed_nodes)} seed nodes")
        return seed_nodes

    def extract_k_hop_subgraph(
        self,
        seed_nodes: Set[Tuple[str, str]],
        k: int = 2,
        max_nodes: int = 50000,
    ) -> Tuple[Dict, List]:
        """
        Extract k-hop neighborhood around seed nodes.

        Args:
            seed_nodes: Set of (type, id) seed nodes
            k: Number of hops
            max_nodes: Maximum nodes to include

        Returns:
            Tuple of (nodes_dict, edges_list)
        """
        print(f"Extracting {k}-hop subgraph from {len(seed_nodes)} seeds...")

        # Build adjacency list for efficient traversal
        adj = defaultdict(set)
        for src, rel, dst in self.edges:
            adj[src].add((dst, rel))
            adj[dst].add((src, rel))  # Undirected for traversal

        # BFS to find k-hop neighborhood
        visited = set(seed_nodes)
        frontier = set(seed_nodes)

        for hop in range(k):
            next_frontier = set()
            for node in frontier:
                for neighbor, rel in adj[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)

                        if len(visited) >= max_nodes:
                            print(f"  Reached max nodes ({max_nodes}) at hop {hop + 1}")
                            break
                if len(visited) >= max_nodes:
                    break

            frontier = next_frontier
            print(f"  Hop {hop + 1}: {len(visited)} nodes")

            if len(visited) >= max_nodes:
                break

        # Extract nodes
        subgraph_nodes = {key: self.nodes[key] for key in visited if key in self.nodes}

        # Extract edges (only between subgraph nodes)
        subgraph_edges = []
        for src, rel, dst in self.edges:
            if src in visited and dst in visited:
                subgraph_edges.append((src, rel, dst))

        print(f"Subgraph: {len(subgraph_nodes)} nodes, {len(subgraph_edges)} edges")
        return subgraph_nodes, subgraph_edges

    def filter_by_relevance(
        self,
        nodes: Dict,
        edges: List,
        min_degree: int = 1,
    ) -> Tuple[Dict, List]:
        """
        Filter subgraph by relevance criteria.

        Args:
            nodes: Node dictionary
            edges: Edge list
            min_degree: Minimum node degree to keep

        Returns:
            Filtered (nodes, edges)
        """
        # Compute degrees
        degree = defaultdict(int)
        for src, rel, dst in edges:
            degree[src] += 1
            degree[dst] += 1

        # Filter nodes by degree
        filtered_nodes = {
            key: info for key, info in nodes.items()
            if degree[key] >= min_degree
        }

        # Filter edges
        filtered_edges = [
            (src, rel, dst) for src, rel, dst in edges
            if src in filtered_nodes and dst in filtered_nodes
        ]

        print(f"After filtering: {len(filtered_nodes)} nodes, {len(filtered_edges)} edges")
        return filtered_nodes, filtered_edges


def extract_ced_subgraph(
    primekg_path: str,
    output_dir: str = 'data/primekg',
    k_hops: int = 2,
    max_nodes: int = 50000,
    min_degree: int = 2,
    seed_dict: Optional[Dict[str, List[str]]] = None,
) -> Tuple[Dict, List]:
    """
    Extract celiac disease-relevant subgraph from PrimeKG.

    Args:
        primekg_path: Path to PrimeKG CSV
        output_dir: Directory to save output
        k_hops: Number of hops from seed nodes
        max_nodes: Maximum nodes to include
        min_degree: Minimum node degree
        seed_dict: Custom seed nodes (default: CED_SEED_NODES)

    Returns:
        Tuple of (nodes_dict, edges_list)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract subgraph
    extractor = SubgraphExtractor(primekg_path)
    extractor.load_primekg()

    seed_nodes = extractor.find_seed_nodes(seed_dict)
    nodes, edges = extractor.extract_k_hop_subgraph(seed_nodes, k=k_hops, max_nodes=max_nodes)
    nodes, edges = extractor.filter_by_relevance(nodes, edges, min_degree=min_degree)

    # Save subgraph
    nodes_path = output_dir / 'ced_subgraph_nodes.json'
    edges_path = output_dir / 'ced_subgraph_edges.json'

    # Convert keys to strings for JSON serialization
    nodes_serializable = {f"{k[0]}::{k[1]}": v for k, v in nodes.items()}
    edges_serializable = [
        {'src_type': src[0], 'src_id': src[1], 'relation': rel, 'dst_type': dst[0], 'dst_id': dst[1]}
        for src, rel, dst in edges
    ]

    with open(nodes_path, 'w') as f:
        json.dump(nodes_serializable, f, indent=2)

    with open(edges_path, 'w') as f:
        json.dump(edges_serializable, f, indent=2)

    print(f"Saved subgraph to {output_dir}")
    return nodes, edges


def get_subgraph_stats(nodes: Dict, edges: List) -> Dict:
    """Get statistics about the subgraph."""
    node_types = defaultdict(int)
    relation_types = defaultdict(int)

    for key, info in nodes.items():
        node_types[info['type']] += 1

    for src, rel, dst in edges:
        relation_types[rel] += 1

    return {
        'num_nodes': len(nodes),
        'num_edges': len(edges),
        'node_types': dict(node_types),
        'relation_types': dict(relation_types),
    }
