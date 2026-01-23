"""
Extended ablation experiments for the Celiac Gut-Brain GNN.

Tests the importance of:
- Different node types (gene, microbe, metabolite, phenotype)
- Different edge types
- Network depth (1-4 layers)
- Hidden dimensions (32, 64, 128, 256)
- Attention heads (for HGT)
- Negative sampling ratios
"""

import torch
import json
import copy
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any, Callable
from dataclasses import dataclass, field, asdict
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np

from celiac.config import PROCESSED_DIR, MODELS_DIR, FIGURES_DIR
from celiac.train import load_kg_from_csv, train_model


@dataclass
class AblationConfig:
    """Configuration for an ablation experiment."""
    name: str
    description: str
    remove_node_types: List[str] = field(default_factory=list)
    remove_edge_types: List[Tuple[str, str, str]] = field(default_factory=list)
    keep_edge_types: Optional[List[Tuple[str, str, str]]] = None
    num_layers: int = 2
    hidden_channels: int = 64
    dropout: float = 0.3
    num_heads: int = 4  # For HGT
    neg_sampling_ratio: float = 1.0


# Predefined ablation configurations
ABLATION_SUITE = {
    # Full graph baseline
    'full_graph': AblationConfig(
        name='full_graph',
        description='Full graph with all node and edge types',
    ),

    # Node type ablations
    'no_microbe': AblationConfig(
        name='no_microbe',
        description='Remove microbe nodes',
        remove_node_types=['microbe'],
    ),
    'no_metabolite': AblationConfig(
        name='no_metabolite',
        description='Remove metabolite nodes',
        remove_node_types=['metabolite'],
    ),
    'no_microbe_metabolite': AblationConfig(
        name='no_microbe_metabolite',
        description='Remove both microbe and metabolite nodes (direct gene-phenotype only)',
        remove_node_types=['microbe', 'metabolite'],
    ),

    # Layer depth ablations
    'layers_1': AblationConfig(
        name='layers_1',
        description='Single GNN layer',
        num_layers=1,
    ),
    'layers_2': AblationConfig(
        name='layers_2',
        description='Two GNN layers (default)',
        num_layers=2,
    ),
    'layers_3': AblationConfig(
        name='layers_3',
        description='Three GNN layers',
        num_layers=3,
    ),
    'layers_4': AblationConfig(
        name='layers_4',
        description='Four GNN layers',
        num_layers=4,
    ),

    # Hidden dimension ablations
    'hidden_32': AblationConfig(
        name='hidden_32',
        description='Hidden dimension 32',
        hidden_channels=32,
    ),
    'hidden_64': AblationConfig(
        name='hidden_64',
        description='Hidden dimension 64 (default)',
        hidden_channels=64,
    ),
    'hidden_128': AblationConfig(
        name='hidden_128',
        description='Hidden dimension 128',
        hidden_channels=128,
    ),
    'hidden_256': AblationConfig(
        name='hidden_256',
        description='Hidden dimension 256',
        hidden_channels=256,
    ),

    # Dropout ablations
    'dropout_0': AblationConfig(
        name='dropout_0',
        description='No dropout',
        dropout=0.0,
    ),
    'dropout_0.3': AblationConfig(
        name='dropout_0.3',
        description='Dropout 0.3 (default)',
        dropout=0.3,
    ),
    'dropout_0.5': AblationConfig(
        name='dropout_0.5',
        description='Dropout 0.5',
        dropout=0.5,
    ),

    # Negative sampling ablations
    'neg_ratio_1': AblationConfig(
        name='neg_ratio_1',
        description='1:1 positive:negative ratio (default)',
        neg_sampling_ratio=1.0,
    ),
    'neg_ratio_3': AblationConfig(
        name='neg_ratio_3',
        description='1:3 positive:negative ratio',
        neg_sampling_ratio=3.0,
    ),
    'neg_ratio_5': AblationConfig(
        name='neg_ratio_5',
        description='1:5 positive:negative ratio',
        neg_sampling_ratio=5.0,
    ),
}


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


def apply_ablation_config(
    data: HeteroData,
    config: AblationConfig
) -> HeteroData:
    """
    Apply ablation configuration to create modified graph.

    Args:
        data: Original HeteroData
        config: Ablation configuration

    Returns:
        Modified HeteroData
    """
    result = data

    # Remove node types
    for node_type in config.remove_node_types:
        if node_type in result.node_types:
            result = remove_node_type(result, node_type)

    # Remove edge types
    for edge_type in config.remove_edge_types:
        if edge_type in result.edge_types:
            result = remove_edge_type(result, edge_type)

    # Keep only specified edge types
    if config.keep_edge_types is not None:
        result = keep_only_edge_types(result, config.keep_edge_types)

    return result


def remove_edge_type(data: HeteroData, edge_type: Tuple[str, str, str]) -> HeteroData:
    """Remove a specific edge type from the graph."""
    new_data = HeteroData()

    # Copy all nodes
    for nt in data.node_types:
        new_data[nt].num_nodes = data[nt].num_nodes
        if hasattr(data[nt], 'x'):
            new_data[nt].x = data[nt].x.clone()
        if hasattr(data[nt], 'node_ids'):
            new_data[nt].node_ids = data[nt].node_ids.copy()

    # Copy edges except the removed type
    for et in data.edge_types:
        if et != edge_type:
            if hasattr(data[et], 'edge_index'):
                new_data[et].edge_index = data[et].edge_index.clone()
            if hasattr(data[et], 'edge_weight'):
                new_data[et].edge_weight = data[et].edge_weight.clone()

    return new_data


def run_extended_ablation_suite(
    data_dir: Path = PROCESSED_DIR / "pyg",
    ablations: Optional[List[str]] = None,
    seeds: List[int] = [0, 1, 2, 42, 123],
    verbose: bool = True,
    save_results: bool = True,
) -> pd.DataFrame:
    """
    Run extended ablation suite with multiple seeds.

    Args:
        data_dir: Path to data directory
        ablations: List of ablation names to run (None = all)
        seeds: List of random seeds
        verbose: Print progress
        save_results: Save results to file

    Returns:
        DataFrame with results
    """
    if ablations is None:
        ablations = list(ABLATION_SUITE.keys())

    target_edge_type = ("gene", "associated_with", "phenotype")

    # Load full graph
    print("\n" + "=" * 60)
    print("LOADING FULL KNOWLEDGE GRAPH")
    print("=" * 60)
    full_data = load_kg_from_csv(data_dir)

    all_results = []

    for ablation_name in ablations:
        config = ABLATION_SUITE[ablation_name]

        print(f"\n{'=' * 60}")
        print(f"ABLATION: {ablation_name}")
        print(f"Description: {config.description}")
        print("=" * 60)

        # Apply ablation
        ablated_data = apply_ablation_config(full_data, config)

        # Check if target edge type still exists
        if target_edge_type not in ablated_data.edge_types:
            print(f"  Target edge type not available after ablation")
            continue

        # Run with multiple seeds
        seed_results = {'auroc': [], 'auprc': [], 'best_epoch': []}

        for seed in seeds:
            if verbose:
                print(f"  Seed {seed}...")

            train_params = dict(
                hidden_channels=config.hidden_channels,
                num_layers=config.num_layers,
                dropout=config.dropout,
                lr=0.01,
                epochs=100,
                patience=20,
                seed=seed,
                verbose=False
            )

            try:
                _, history = train_model(ablated_data, target_edge_type, **train_params)
                seed_results['auroc'].append(history['test_auroc'])
                seed_results['auprc'].append(history['test_auprc'])
                seed_results['best_epoch'].append(history['best_epoch'])
            except Exception as e:
                print(f"    Error: {e}")
                continue

        if seed_results['auroc']:
            result = {
                'ablation': ablation_name,
                'description': config.description,
                'auroc_mean': np.mean(seed_results['auroc']),
                'auroc_std': np.std(seed_results['auroc']),
                'auprc_mean': np.mean(seed_results['auprc']),
                'auprc_std': np.std(seed_results['auprc']),
                'best_epoch_mean': np.mean(seed_results['best_epoch']),
                'num_seeds': len(seed_results['auroc']),
            }
            all_results.append(result)

            if verbose:
                print(f"  AUROC: {result['auroc_mean']:.4f} ± {result['auroc_std']:.4f}")
                print(f"  AUPRC: {result['auprc_mean']:.4f} ± {result['auprc_std']:.4f}")

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Save results
    if save_results:
        results_file = MODELS_DIR / "extended_ablation_results.csv"
        df.to_csv(results_file, index=False)
        print(f"\nSaved results to {results_file}")

    return df


def create_ablation_figure(
    results_df: pd.DataFrame,
    metric: str = 'auroc',
    output_path: Optional[str] = None,
) -> None:
    """
    Create ablation results visualization.

    Args:
        results_df: DataFrame from run_extended_ablation_suite
        metric: Metric to plot ('auroc' or 'auprc')
        output_path: Path to save figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    # Sort by mean performance
    df_sorted = results_df.sort_values(f'{metric}_mean', ascending=True)

    # Create bar chart with error bars
    y_pos = np.arange(len(df_sorted))
    means = df_sorted[f'{metric}_mean'].values
    stds = df_sorted[f'{metric}_std'].values

    bars = ax.barh(y_pos, means, xerr=stds, capsize=3, color='steelblue', alpha=0.8)

    # Highlight the baseline
    baseline_idx = df_sorted[df_sorted['ablation'] == 'full_graph'].index
    if len(baseline_idx) > 0:
        baseline_pos = list(df_sorted['ablation']).index('full_graph')
        bars[baseline_pos].set_color('darkgreen')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted['ablation'])
    ax.set_xlabel(f'{metric.upper()} (mean ± std)')
    ax.set_title(f'Ablation Study Results - {metric.upper()}')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {output_path}")
    else:
        plt.show()

    plt.close()


def run_layer_depth_analysis(
    data: HeteroData,
    target_edge_type: Tuple[str, str, str],
    max_layers: int = 5,
    seeds: List[int] = [42],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Analyze effect of GNN layer depth.

    Args:
        data: HeteroData
        target_edge_type: Target edge type for link prediction
        max_layers: Maximum number of layers to test
        seeds: Random seeds

    Returns:
        DataFrame with results per layer count
    """
    results = []

    for num_layers in range(1, max_layers + 1):
        print(f"\nTesting {num_layers} layer(s)...")

        layer_results = {'auroc': [], 'auprc': []}

        for seed in seeds:
            train_params = dict(
                hidden_channels=64,
                num_layers=num_layers,
                dropout=0.3,
                lr=0.01,
                epochs=100,
                patience=20,
                seed=seed,
                verbose=False
            )

            try:
                _, history = train_model(data, target_edge_type, **train_params)
                layer_results['auroc'].append(history['test_auroc'])
                layer_results['auprc'].append(history['test_auprc'])
            except Exception as e:
                print(f"  Error with {num_layers} layers: {e}")

        if layer_results['auroc']:
            results.append({
                'num_layers': num_layers,
                'auroc_mean': np.mean(layer_results['auroc']),
                'auroc_std': np.std(layer_results['auroc']),
                'auprc_mean': np.mean(layer_results['auprc']),
                'auprc_std': np.std(layer_results['auprc']),
            })

            if verbose:
                print(f"  Layers {num_layers}: AUROC = {results[-1]['auroc_mean']:.4f} ± {results[-1]['auroc_std']:.4f}")

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Run basic ablations
    results = run_ablation_experiments()

    # Run extended ablations if pandas available
    try:
        extended_df = run_extended_ablation_suite(
            ablations=['full_graph', 'no_microbe', 'no_metabolite', 'layers_1', 'layers_3'],
            seeds=[42],  # Single seed for quick test
        )
        print("\nExtended ablation results:")
        print(extended_df.to_string())
    except Exception as e:
        print(f"Extended ablations failed: {e}")
