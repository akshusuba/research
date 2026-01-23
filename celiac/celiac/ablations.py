"""
Ablation experiments for the Celiac Gut-Brain GNN.
Tests the importance of different node types and graph structure.
"""

import torch
import json
import copy
from pathlib import Path
from typing import Dict, Tuple, Optional
from torch_geometric.data import HeteroData

from celiac.config import PROCESSED_DIR, MODELS_DIR, FIGURES_DIR
from celiac.train import load_kg_from_csv, train_model


def remove_node_type(data: HeteroData, node_type: str) -> HeteroData:
    """
    Remove a node type and all its connected edges from the graph.
    Returns a new HeteroData object.
    """
    new_data = HeteroData()

    # Copy nodes except the removed type
    for nt in data.node_types:
        if nt != node_type:
            new_data[nt].num_nodes = data[nt].num_nodes
            if hasattr(data[nt], 'x'):
                new_data[nt].x = data[nt].x.clone()
            if hasattr(data[nt], 'node_ids'):
                new_data[nt].node_ids = data[nt].node_ids.copy()

    # Copy edges that don't involve the removed node type
    for edge_type in data.edge_types:
        src_type, rel, dst_type = edge_type
        if src_type != node_type and dst_type != node_type:
            if hasattr(data[edge_type], 'edge_index'):
                new_data[edge_type].edge_index = data[edge_type].edge_index.clone()
            if hasattr(data[edge_type], 'edge_weight'):
                new_data[edge_type].edge_weight = data[edge_type].edge_weight.clone()

    return new_data


def keep_only_edge_types(data: HeteroData, keep_edges: list) -> HeteroData:
    """
    Keep only specified edge types (and their connected node types).
    """
    new_data = HeteroData()

    # Find which node types are needed
    needed_node_types = set()
    for src, rel, dst in keep_edges:
        needed_node_types.add(src)
        needed_node_types.add(dst)

    # Copy needed nodes
    for nt in data.node_types:
        if nt in needed_node_types:
            new_data[nt].num_nodes = data[nt].num_nodes
            if hasattr(data[nt], 'x'):
                new_data[nt].x = data[nt].x.clone()
            if hasattr(data[nt], 'node_ids'):
                new_data[nt].node_ids = data[nt].node_ids.copy()

    # Copy specified edges
    for edge_type in data.edge_types:
        if edge_type in keep_edges:
            if hasattr(data[edge_type], 'edge_index'):
                new_data[edge_type].edge_index = data[edge_type].edge_index.clone()
            if hasattr(data[edge_type], 'edge_weight'):
                new_data[edge_type].edge_weight = data[edge_type].edge_weight.clone()

    return new_data


def run_ablation_experiments(
    data_dir: Path = PROCESSED_DIR / "pyg",
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Run all ablation experiments.

    Returns dict of {experiment_name: {auroc, auprc, ...}}
    """
    results = {}
    target_edge_type = ("gene", "associated_with", "phenotype")

    # Common training params
    train_params = dict(
        hidden_channels=64,
        num_layers=2,
        dropout=0.3,
        lr=0.01,
        epochs=100,
        patience=20,
        seed=seed,
        verbose=verbose
    )

    # Load full graph
    print("\n" + "="*60)
    print("LOADING FULL KNOWLEDGE GRAPH")
    print("="*60)
    full_data = load_kg_from_csv(data_dir)
    print(f"Node types: {list(full_data.node_types)}")
    print(f"Edge types: {list(full_data.edge_types)}")

    # =========================================================================
    # Experiment 1: Full graph (baseline)
    # =========================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 1: Full Graph (Baseline)")
    print("="*60)

    _, history = train_model(full_data, target_edge_type, **train_params)
    results["full_graph"] = {
        "auroc": history["test_auroc"],
        "auprc": history["test_auprc"],
        "best_epoch": history["best_epoch"]
    }

    # =========================================================================
    # Experiment 2: Remove metabolite nodes
    # =========================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 2: Remove Metabolite Nodes")
    print("="*60)

    no_metabolite_data = remove_node_type(full_data, "metabolite")
    print(f"Node types: {list(no_metabolite_data.node_types)}")
    print(f"Edge types: {list(no_metabolite_data.edge_types)}")

    if target_edge_type in no_metabolite_data.edge_types:
        _, history = train_model(no_metabolite_data, target_edge_type, **train_params)
        results["no_metabolite"] = {
            "auroc": history["test_auroc"],
            "auprc": history["test_auprc"],
            "best_epoch": history["best_epoch"]
        }
    else:
        print("  Target edge type not available after ablation")
        results["no_metabolite"] = {"auroc": None, "auprc": None}

    # =========================================================================
    # Experiment 3: Remove microbe nodes
    # =========================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 3: Remove Microbe Nodes")
    print("="*60)

    no_microbe_data = remove_node_type(full_data, "microbe")
    print(f"Node types: {list(no_microbe_data.node_types)}")
    print(f"Edge types: {list(no_microbe_data.edge_types)}")

    if target_edge_type in no_microbe_data.edge_types:
        _, history = train_model(no_microbe_data, target_edge_type, **train_params)
        results["no_microbe"] = {
            "auroc": history["test_auroc"],
            "auprc": history["test_auprc"],
            "best_epoch": history["best_epoch"]
        }
    else:
        print("  Target edge type not available after ablation")
        results["no_microbe"] = {"auroc": None, "auprc": None}

    # =========================================================================
    # Experiment 4: Direct gene-phenotype only
    # =========================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 4: Direct Gene-Phenotype Only")
    print("="*60)

    direct_data = keep_only_edge_types(full_data, [target_edge_type])
    print(f"Node types: {list(direct_data.node_types)}")
    print(f"Edge types: {list(direct_data.edge_types)}")

    if target_edge_type in direct_data.edge_types:
        _, history = train_model(direct_data, target_edge_type, **train_params)
        results["direct_only"] = {
            "auroc": history["test_auroc"],
            "auprc": history["test_auprc"],
            "best_epoch": history["best_epoch"]
        }
    else:
        print("  Target edge type not available")
        results["direct_only"] = {"auroc": None, "auprc": None}

    # =========================================================================
    # Experiment 5: Vary number of layers
    # =========================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 5: Layer Ablation (1, 2, 3 layers)")
    print("="*60)

    for num_layers in [1, 3]:
        print(f"\n--- {num_layers} layer(s) ---")
        layer_params = train_params.copy()
        layer_params["num_layers"] = num_layers

        _, history = train_model(full_data, target_edge_type, **layer_params)
        results[f"layers_{num_layers}"] = {
            "auroc": history["test_auroc"],
            "auprc": history["test_auprc"],
            "best_epoch": history["best_epoch"]
        }

    # Add the 2-layer result from baseline
    results["layers_2"] = results["full_graph"]

    # =========================================================================
    # Save results
    # =========================================================================
    print("\n" + "="*60)
    print("ABLATION RESULTS SUMMARY")
    print("="*60)

    print(f"\n{'Experiment':<25} {'AUROC':<10} {'AUPRC':<10}")
    print("-" * 45)
    for exp_name, metrics in results.items():
        auroc = f"{metrics['auroc']:.4f}" if metrics['auroc'] else "N/A"
        auprc = f"{metrics['auprc']:.4f}" if metrics['auprc'] else "N/A"
        print(f"{exp_name:<25} {auroc:<10} {auprc:<10}")

    # Save to JSON
    results_file = MODELS_DIR / "ablation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_file}")

    return results


if __name__ == "__main__":
    results = run_ablation_experiments()
