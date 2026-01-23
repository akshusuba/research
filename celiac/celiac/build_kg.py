"""
Build the Celiac Gut-Brain Knowledge Graph.
Combines data from multiple sources into a unified graph structure.
Outputs PyTorch Geometric compatible format.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict

from celiac.config import (
    RAW_DIR, PROCESSED_DIR,
    CED_MICROBIOME_TAXA, NEUROLOGICAL_HPO_TERMS, KEY_GENES,
    KEY_METABOLITES, EVIDENCE_WEIGHTS, NODE_TYPES, EDGE_TYPES
)
from celiac.data_fetcher import (
    fetch_geo_series_matrix, parse_geo_series_matrix,
    fetch_all_gene_phenotype_edges, get_curated_edges,
    fetch_monarch_phenotype_genes
)


class KnowledgeGraph:
    """
    Heterogeneous Knowledge Graph for Celiac Gut-Brain Axis.
    """

    def __init__(self):
        # Node storage: {node_type: {node_id: {attributes}}}
        self.nodes: Dict[str, Dict[str, Dict]] = defaultdict(dict)

        # Edge storage: {(src_type, rel, dst_type): [(src_id, dst_id, {attributes})]}
        self.edges: Dict[Tuple[str, str, str], List[Tuple[str, str, Dict]]] = defaultdict(list)

        # Node ID to index mapping per type
        self.node_to_idx: Dict[str, Dict[str, int]] = defaultdict(dict)

        # Statistics
        self.stats = {}

    def add_node(self, node_type: str, node_id: str, **attributes) -> None:
        """Add a node to the graph."""
        if node_id not in self.nodes[node_type]:
            self.nodes[node_type][node_id] = attributes
            self.node_to_idx[node_type][node_id] = len(self.node_to_idx[node_type])
        else:
            # Update attributes
            self.nodes[node_type][node_id].update(attributes)

    def add_edge(self, src_type: str, src_id: str, relation: str,
                 dst_type: str, dst_id: str, **attributes) -> None:
        """Add an edge to the graph."""
        # Ensure nodes exist
        if src_id not in self.nodes[src_type]:
            self.add_node(src_type, src_id)
        if dst_id not in self.nodes[dst_type]:
            self.add_node(dst_type, dst_id)

        edge_type = (src_type, relation, dst_type)
        self.edges[edge_type].append((src_id, dst_id, attributes))

    def get_node_idx(self, node_type: str, node_id: str) -> Optional[int]:
        """Get the index of a node."""
        return self.node_to_idx[node_type].get(node_id)

    def compute_stats(self) -> Dict:
        """Compute graph statistics."""
        self.stats = {
            "node_counts": {t: len(nodes) for t, nodes in self.nodes.items()},
            "edge_counts": {str(t): len(edges) for t, edges in self.edges.items()},
            "total_nodes": sum(len(nodes) for nodes in self.nodes.values()),
            "total_edges": sum(len(edges) for edges in self.edges.values()),
        }
        return self.stats

    def print_stats(self) -> None:
        """Print graph statistics."""
        stats = self.compute_stats()
        print("\n" + "="*50)
        print("KNOWLEDGE GRAPH STATISTICS")
        print("="*50)
        print(f"\nTotal nodes: {stats['total_nodes']}")
        print(f"Total edges: {stats['total_edges']}")
        print("\nNodes by type:")
        for node_type, count in stats['node_counts'].items():
            print(f"  {node_type}: {count}")
        print("\nEdges by type:")
        for edge_type, count in stats['edge_counts'].items():
            print(f"  {edge_type}: {count}")


def build_knowledge_graph(
    fetch_geo: bool = True,
    fetch_monarch: bool = True,
    use_curated: bool = True,
    expand_phenotype_genes: bool = True,
    output_dir: Optional[Path] = None
) -> KnowledgeGraph:
    """
    Build the complete knowledge graph.

    Args:
        fetch_geo: Whether to fetch and process GEO data
        fetch_monarch: Whether to fetch Monarch gene-phenotype edges
        use_curated: Whether to use curated microbe-metabolite edges
        expand_phenotype_genes: Whether to expand genes from phenotypes
        output_dir: Directory to save intermediate files

    Returns:
        KnowledgeGraph object
    """
    if output_dir is None:
        output_dir = PROCESSED_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    kg = KnowledgeGraph()

    print("\n" + "="*60)
    print("BUILDING CELIAC GUT-BRAIN KNOWLEDGE GRAPH")
    print("="*60)

    # =========================================================================
    # 1. Add Microbe nodes (CeD-specific)
    # =========================================================================
    print("\n[1/5] Adding Microbe nodes...")
    for taxon, info in CED_MICROBIOME_TAXA.items():
        kg.add_node(
            "microbe", taxon,
            direction=info["direction"],
            source=info["source"],
            ced_specific=True
        )
    print(f"  Added {len(CED_MICROBIOME_TAXA)} CeD-associated microbes")

    # =========================================================================
    # 2. Add Metabolite nodes and Microbe→Metabolite edges
    # =========================================================================
    print("\n[2/5] Adding Metabolites and Microbe→Metabolite edges...")
    if use_curated:
        mm_edges, mg_edges = get_curated_edges(output_dir)

        for edge in mm_edges:
            # Add metabolite node
            kg.add_node("metabolite", edge["metabolite"])

            # Add edge
            kg.add_edge(
                "microbe", edge["microbe"],
                edge["relation"],
                "metabolite", edge["metabolite"],
                evidence=edge["evidence"],
                source=edge["source"],
                weight=EVIDENCE_WEIGHTS.get(edge["evidence"], 0.5)
            )

        print(f"  Added {len(set(e['metabolite'] for e in mm_edges))} metabolites")
        print(f"  Added {len(mm_edges)} microbe→metabolite edges")

    # =========================================================================
    # 3. Add Gene nodes and Metabolite→Gene edges
    # =========================================================================
    print("\n[3/5] Adding Genes and Metabolite→Gene edges...")

    # Add key genes
    for gene in KEY_GENES:
        kg.add_node("gene", gene, key_gene=True)

    if use_curated:
        for edge in mg_edges:
            # Add gene node if not exists
            kg.add_node("gene", edge["gene"])

            # Add edge
            kg.add_edge(
                "metabolite", edge["metabolite"],
                edge["relation"],
                "gene", edge["gene"],
                evidence=edge["evidence"],
                source=edge["source"],
                weight=EVIDENCE_WEIGHTS.get(edge["evidence"], 0.5)
            )

        print(f"  Added {len(mg_edges)} metabolite→gene edges")

    # =========================================================================
    # 4. Add Phenotype nodes and Gene→Phenotype edges
    # =========================================================================
    print("\n[4/5] Adding Phenotypes and Gene→Phenotype edges...")

    # Add neurological phenotypes
    for hpo_id, label in NEUROLOGICAL_HPO_TERMS.items():
        kg.add_node("phenotype", hpo_id, label=label, neurological=True)

    if fetch_monarch:
        # Get all genes in the graph so far
        all_genes = list(kg.nodes["gene"].keys())

        gp_edges_file = output_dir / "gene_phenotype_edges.csv"

        # Check if we already have the edges cached
        if gp_edges_file.exists():
            print(f"  Loading cached gene-phenotype edges from {gp_edges_file}")
            with open(gp_edges_file, 'r') as f:
                reader = csv.DictReader(f)
                gp_edges = list(reader)
        else:
            gp_edges = fetch_all_gene_phenotype_edges(all_genes, gp_edges_file)

        # Add edges
        neuro_edge_count = 0
        for edge in gp_edges:
            # Add phenotype node
            kg.add_node(
                "phenotype", edge["target"],
                label=edge.get("target_label", ""),
                neurological=(edge["target"] in NEUROLOGICAL_HPO_TERMS)
            )

            # Add edge
            kg.add_edge(
                "gene", edge["source"],
                "associated_with",
                "phenotype", edge["target"],
                evidence="monarch",
                source="Monarch Initiative",
                weight=EVIDENCE_WEIGHTS["medium"]
            )

            if edge["target"] in NEUROLOGICAL_HPO_TERMS:
                neuro_edge_count += 1

        print(f"  Added {len(gp_edges)} gene→phenotype edges")
        print(f"  ({neuro_edge_count} to neurological phenotypes)")

    # =========================================================================
    # 5. Expand genes from neurological phenotypes
    # =========================================================================
    if expand_phenotype_genes:
        print("\n[5/5] Expanding genes from neurological phenotypes...")
        new_genes = 0
        new_edges = 0

        for hpo_id in list(NEUROLOGICAL_HPO_TERMS.keys())[:5]:  # Top 5 phenotypes
            genes = fetch_monarch_phenotype_genes(hpo_id)

            for gene_info in genes[:20]:  # Top 20 genes per phenotype
                gene_symbol = gene_info.get("label", "")
                if gene_symbol and gene_symbol not in kg.nodes["gene"]:
                    kg.add_node("gene", gene_symbol, source="monarch_expansion")
                    new_genes += 1

                if gene_symbol:
                    kg.add_edge(
                        "gene", gene_symbol,
                        "associated_with",
                        "phenotype", hpo_id,
                        evidence="monarch",
                        source="Monarch Initiative",
                        weight=EVIDENCE_WEIGHTS["medium"]
                    )
                    new_edges += 1

        print(f"  Added {new_genes} new genes from phenotype expansion")
        print(f"  Added {new_edges} gene→phenotype edges")
    else:
        print("\n[5/5] Skipping phenotype gene expansion")

    # =========================================================================
    # Print final statistics
    # =========================================================================
    kg.print_stats()

    return kg


def save_kg_for_pyg(kg: KnowledgeGraph, output_dir: Path) -> Dict[str, Path]:
    """
    Save knowledge graph in PyTorch Geometric compatible format.

    Creates:
    - node_<type>.csv: Node features
    - edge_<src>_<rel>_<dst>.csv: Edge lists
    - metadata.json: Graph metadata

    Returns dict of file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}

    print("\n" + "="*50)
    print("SAVING GRAPH FOR PYTORCH GEOMETRIC")
    print("="*50)

    # Save node files
    for node_type, nodes in kg.nodes.items():
        node_file = output_dir / f"nodes_{node_type}.csv"
        with open(node_file, 'w', newline='') as f:
            if nodes:
                # Get all possible keys
                all_keys = set()
                for attrs in nodes.values():
                    all_keys.update(attrs.keys())
                all_keys = sorted(all_keys)

                writer = csv.writer(f)
                writer.writerow(["node_id", "node_idx"] + list(all_keys))

                for node_id, attrs in nodes.items():
                    idx = kg.node_to_idx[node_type][node_id]
                    row = [node_id, idx] + [attrs.get(k, "") for k in all_keys]
                    writer.writerow(row)

        files[f"nodes_{node_type}"] = node_file
        print(f"  Saved {len(nodes)} {node_type} nodes to {node_file.name}")

    # Save edge files
    for edge_type, edges in kg.edges.items():
        src_type, rel, dst_type = edge_type
        edge_file = output_dir / f"edges_{src_type}_{rel}_{dst_type}.csv"

        with open(edge_file, 'w', newline='') as f:
            if edges:
                # Get all possible attribute keys
                all_keys = set()
                for _, _, attrs in edges:
                    all_keys.update(attrs.keys())
                all_keys = sorted(all_keys)

                writer = csv.writer(f)
                writer.writerow(["src_id", "src_idx", "dst_id", "dst_idx"] + list(all_keys))

                for src_id, dst_id, attrs in edges:
                    src_idx = kg.node_to_idx[src_type][src_id]
                    dst_idx = kg.node_to_idx[dst_type][dst_id]
                    row = [src_id, src_idx, dst_id, dst_idx] + [attrs.get(k, "") for k in all_keys]
                    writer.writerow(row)

        files[f"edges_{src_type}_{rel}_{dst_type}"] = edge_file
        print(f"  Saved {len(edges)} {rel} edges to {edge_file.name}")

    # Save metadata
    metadata = {
        "node_types": list(kg.nodes.keys()),
        "edge_types": [list(et) for et in kg.edges.keys()],
        "stats": kg.compute_stats(),
        "node_counts": {t: len(n) for t, n in kg.nodes.items()},
        "edge_counts": {f"{s}_{r}_{d}": len(e) for (s, r, d), e in kg.edges.items()},
    }

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    files["metadata"] = metadata_file
    print(f"  Saved metadata to {metadata_file.name}")

    return files


def create_pyg_dataset(kg: KnowledgeGraph, output_file: Path) -> None:
    """
    Create a PyTorch Geometric HeteroData object and save it.
    Requires torch and torch_geometric to be installed.
    """
    try:
        import torch
        from torch_geometric.data import HeteroData
    except ImportError:
        print("  PyTorch Geometric not installed. Saving CSV format only.")
        print("  Install with: pip install torch torch_geometric")
        return

    data = HeteroData()

    # Add nodes (using simple one-hot + any numeric features)
    for node_type, nodes in kg.nodes.items():
        n = len(nodes)
        # Simple: just use node index as feature (will be replaced with embeddings)
        data[node_type].x = torch.arange(n).unsqueeze(1).float()
        data[node_type].num_nodes = n

    # Add edges
    for edge_type, edges in kg.edges.items():
        src_type, rel, dst_type = edge_type
        if not edges:
            continue

        src_indices = []
        dst_indices = []
        weights = []

        for src_id, dst_id, attrs in edges:
            src_idx = kg.node_to_idx[src_type].get(src_id)
            dst_idx = kg.node_to_idx[dst_type].get(dst_id)
            if src_idx is not None and dst_idx is not None:
                src_indices.append(src_idx)
                dst_indices.append(dst_idx)
                weights.append(attrs.get("weight", 1.0))

        if src_indices:
            edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
            data[src_type, rel, dst_type].edge_index = edge_index
            data[src_type, rel, dst_type].edge_weight = torch.tensor(weights, dtype=torch.float)

    # Save
    torch.save(data, output_file)
    print(f"  Saved PyG HeteroData to {output_file}")


if __name__ == "__main__":
    # Build the knowledge graph
    kg = build_knowledge_graph(
        fetch_geo=False,  # Skip GEO for now (focus on graph structure)
        fetch_monarch=True,
        use_curated=True,
        expand_phenotype_genes=True
    )

    # Save for PyG
    pyg_dir = PROCESSED_DIR / "pyg"
    files = save_kg_for_pyg(kg, pyg_dir)

    # Try to create PyG dataset
    create_pyg_dataset(kg, pyg_dir / "hetero_data.pt")

    print("\n" + "="*50)
    print("KNOWLEDGE GRAPH BUILD COMPLETE")
    print("="*50)
    print(f"\nFiles saved to: {pyg_dir}")
    print("\nNext steps:")
    print("  1. Review the CSV files to verify graph structure")
    print("  2. Install PyG: pip install torch torch_geometric")
    print("  3. Run the GNN training script")
