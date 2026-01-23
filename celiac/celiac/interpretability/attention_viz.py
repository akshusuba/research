"""
Attention weight visualization for heterogeneous GNNs.

Visualize which edges and nodes the model pays attention to.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


def extract_attention_weights(
    model: nn.Module,
    data: HeteroData,
    layer_idx: int = -1,
) -> Dict[Tuple[str, str, str], torch.Tensor]:
    """
    Extract attention weights from a trained model.

    Works with models that use attention mechanisms (GAT, HGT, etc.)

    Args:
        model: Trained GNN model
        data: HeteroData object
        layer_idx: Which layer to extract attention from (-1 = last)

    Returns:
        Dict mapping edge_type -> attention weights tensor
    """
    model.eval()
    attention_weights = {}

    # Hook to capture attention weights
    captured_attention = {}

    def attention_hook(name):
        def hook(module, input, output):
            if hasattr(output, 'attention_weights'):
                captured_attention[name] = output.attention_weights
            elif isinstance(output, tuple) and len(output) > 1:
                # Some implementations return (output, attention)
                captured_attention[name] = output[1]
        return hook

    # Register hooks on attention layers
    hooks = []
    for name, module in model.named_modules():
        if 'attn' in name.lower() or 'attention' in name.lower():
            hook = module.register_forward_hook(attention_hook(name))
            hooks.append(hook)

    # Forward pass
    with torch.no_grad():
        try:
            _ = model(data)
        except Exception as e:
            print(f"Forward pass failed: {e}")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Process captured attention
    for name, attn in captured_attention.items():
        if isinstance(attn, dict):
            attention_weights.update(attn)
        elif isinstance(attn, torch.Tensor):
            attention_weights[name] = attn

    return attention_weights


def get_top_attention_edges(
    attention_weights: Dict[Tuple[str, str, str], torch.Tensor],
    data: HeteroData,
    top_k: int = 20,
    edge_type: Optional[Tuple[str, str, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Get edges with highest attention weights.

    Args:
        attention_weights: Dict from extract_attention_weights
        data: HeteroData object
        top_k: Number of top edges to return
        edge_type: Specific edge type to analyze (None = all)

    Returns:
        List of dicts with edge info and attention score
    """
    top_edges = []

    edge_types = [edge_type] if edge_type else list(attention_weights.keys())

    for et in edge_types:
        if et not in attention_weights:
            continue

        attn = attention_weights[et]
        if attn is None or len(attn) == 0:
            continue

        # Handle multi-head attention (average across heads)
        if attn.dim() > 1:
            attn = attn.mean(dim=-1)

        # Get edge index
        if et in data.edge_types:
            edge_index = data[et].edge_index
        else:
            continue

        # Get top-k indices
        k = min(top_k, len(attn))
        top_values, top_indices = torch.topk(attn, k)

        # Get node names if available
        src_type, rel, dst_type = et
        src_names = data[src_type].get('node_names', [f'{src_type}_{i}' for i in range(data[src_type].num_nodes)])
        dst_names = data[dst_type].get('node_names', [f'{dst_type}_{i}' for i in range(data[dst_type].num_nodes)])

        for idx, val in zip(top_indices.tolist(), top_values.tolist()):
            src_idx = edge_index[0, idx].item()
            dst_idx = edge_index[1, idx].item()

            src_name = src_names[src_idx] if src_idx < len(src_names) else f'{src_type}_{src_idx}'
            dst_name = dst_names[dst_idx] if dst_idx < len(dst_names) else f'{dst_type}_{dst_idx}'

            top_edges.append({
                'edge_type': et,
                'src_idx': src_idx,
                'dst_idx': dst_idx,
                'src_name': src_name,
                'dst_name': dst_name,
                'attention': val,
            })

    # Sort by attention weight
    top_edges.sort(key=lambda x: x['attention'], reverse=True)

    return top_edges[:top_k]


def visualize_attention_heatmap(
    attention_weights: Dict[Tuple[str, str, str], torch.Tensor],
    data: HeteroData,
    edge_type: Tuple[str, str, str],
    max_nodes: int = 30,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
) -> None:
    """
    Create attention heatmap for a specific edge type.

    Args:
        attention_weights: Dict from extract_attention_weights
        data: HeteroData object
        edge_type: Edge type to visualize
        max_nodes: Maximum nodes to show (for readability)
        output_path: Path to save figure
        figsize: Figure size
    """
    if edge_type not in attention_weights:
        print(f"No attention weights for {edge_type}")
        return

    attn = attention_weights[edge_type]
    if attn is None:
        return

    # Handle multi-head attention
    if attn.dim() > 1:
        attn = attn.mean(dim=-1)

    edge_index = data[edge_type].edge_index
    src_type, rel, dst_type = edge_type

    # Get number of nodes
    num_src = data[src_type].num_nodes
    num_dst = data[dst_type].num_nodes

    # Limit nodes for visualization
    num_src = min(num_src, max_nodes)
    num_dst = min(num_dst, max_nodes)

    # Create attention matrix
    attn_matrix = np.zeros((num_src, num_dst))

    for i in range(edge_index.size(1)):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        if src < num_src and dst < num_dst:
            attn_matrix[src, dst] = attn[i].item() if i < len(attn) else 0

    # Get node names
    src_names = data[src_type].get('node_names', [f'{src_type}_{i}' for i in range(num_src)])[:num_src]
    dst_names = data[dst_type].get('node_names', [f'{dst_type}_{i}' for i in range(num_dst)])[:num_dst]

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(attn_matrix, cmap='Blues', aspect='auto')

    # Labels
    ax.set_xticks(np.arange(num_dst))
    ax.set_yticks(np.arange(num_src))
    ax.set_xticklabels(dst_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(src_names, fontsize=8)

    ax.set_xlabel(f'{dst_type}')
    ax.set_ylabel(f'{src_type}')
    ax.set_title(f'Attention Weights: {src_type} → {rel} → {dst_type}')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_node_attention(
    model: nn.Module,
    data: HeteroData,
    node_type: str,
    node_idx: int,
    output_path: Optional[str] = None,
) -> None:
    """
    Visualize what a specific node attends to.

    Args:
        model: Trained model
        data: HeteroData object
        node_type: Type of the node to analyze
        node_idx: Index of the node
        output_path: Path to save figure
    """
    attention_weights = extract_attention_weights(model, data)

    # Find all edges involving this node
    incoming_attention = defaultdict(list)
    outgoing_attention = defaultdict(list)

    for edge_type, attn in attention_weights.items():
        if attn is None:
            continue

        src_type, rel, dst_type = edge_type
        edge_index = data[edge_type].edge_index

        # Handle multi-head
        if attn.dim() > 1:
            attn = attn.mean(dim=-1)

        for i in range(edge_index.size(1)):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()

            if src_type == node_type and src == node_idx:
                outgoing_attention[dst_type].append((dst, attn[i].item()))
            if dst_type == node_type and dst == node_idx:
                incoming_attention[src_type].append((src, attn[i].item()))

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Incoming attention
    ax = axes[0]
    ax.set_title(f'Incoming Attention to {node_type}[{node_idx}]')

    y_pos = 0
    for src_type, attns in incoming_attention.items():
        attns.sort(key=lambda x: x[1], reverse=True)
        for idx, weight in attns[:10]:
            ax.barh(y_pos, weight, color='steelblue', alpha=0.7)
            ax.text(weight + 0.01, y_pos, f'{src_type}[{idx}]', va='center', fontsize=8)
            y_pos += 1

    ax.set_xlim(0, 1)
    ax.set_ylabel('Source Nodes')
    ax.set_xlabel('Attention Weight')

    # Outgoing attention
    ax = axes[1]
    ax.set_title(f'Outgoing Attention from {node_type}[{node_idx}]')

    y_pos = 0
    for dst_type, attns in outgoing_attention.items():
        attns.sort(key=lambda x: x[1], reverse=True)
        for idx, weight in attns[:10]:
            ax.barh(y_pos, weight, color='coral', alpha=0.7)
            ax.text(weight + 0.01, y_pos, f'{dst_type}[{idx}]', va='center', fontsize=8)
            y_pos += 1

    ax.set_xlim(0, 1)
    ax.set_ylabel('Target Nodes')
    ax.set_xlabel('Attention Weight')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    else:
        plt.show()

    plt.close()
