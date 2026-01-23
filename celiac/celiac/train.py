"""
Training script for Celiac Gut-Brain GNN.
Handles data loading, training loop, and evaluation.
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
import csv
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

from celiac.config import PROCESSED_DIR, MODELS_DIR, FIGURES_DIR
from celiac.models import CeliacGNN, create_model_from_data


def load_kg_from_csv(data_dir: Path) -> HeteroData:
    """
    Load knowledge graph from CSV files into HeteroData.
    """
    data = HeteroData()

    # Load metadata
    metadata_file = data_dir / "metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Load nodes
    for node_type in metadata["node_types"]:
        node_file = data_dir / f"nodes_{node_type}.csv"
        if node_file.exists():
            with open(node_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            num_nodes = len(rows)
            data[node_type].num_nodes = num_nodes

            # Store node IDs for reference
            data[node_type].node_ids = [row["node_id"] for row in rows]

            # Create simple feature (will be replaced by learned embeddings)
            data[node_type].x = torch.arange(num_nodes).unsqueeze(1).float()

    # Load edges
    for edge_type_list in metadata["edge_types"]:
        src_type, rel, dst_type = edge_type_list
        edge_file = data_dir / f"edges_{src_type}_{rel}_{dst_type}.csv"

        if edge_file.exists():
            with open(edge_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            if rows:
                src_indices = [int(row["src_idx"]) for row in rows]
                dst_indices = [int(row["dst_idx"]) for row in rows]

                edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
                data[src_type, rel, dst_type].edge_index = edge_index

                # Edge weights if available
                if "weight" in rows[0]:
                    weights = [float(row.get("weight", 1.0)) for row in rows]
                    data[src_type, rel, dst_type].edge_weight = torch.tensor(weights)

    return data


def create_link_prediction_splits(
    data: HeteroData,
    target_edge_type: Tuple[str, str, str],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    neg_sampling_ratio: float = 1.0,
    seed: int = 42
) -> Tuple[HeteroData, HeteroData, HeteroData]:
    """
    Create train/val/test splits for link prediction.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        neg_sampling_ratio=neg_sampling_ratio,
        edge_types=[target_edge_type],
        rev_edge_types=None,
    )

    train_data, val_data, test_data = transform(data)

    return train_data, val_data, test_data


def train_epoch(
    model: CeliacGNN,
    data: HeteroData,
    optimizer: torch.optim.Optimizer,
    target_edge_type: Tuple[str, str, str],
    device: torch.device
) -> float:
    """
    Train for one epoch.
    Returns average loss.
    """
    model.train()
    optimizer.zero_grad()

    # Get node embeddings
    z_dict = model.encode(data.to(device))

    # Get supervision edges
    src_type, rel, dst_type = target_edge_type
    edge_label_index = data[target_edge_type].edge_label_index
    edge_label = data[target_edge_type].edge_label

    # Decode
    if dst_type == "phenotype" and src_type == "gene":
        pred = model.decode_gene_phenotype(z_dict, edge_label_index)
    elif dst_type == "phenotype" and src_type == "microbe":
        pred = model.decode_microbe_phenotype(z_dict, edge_label_index)
    else:
        raise ValueError(f"Unknown target edge type: {target_edge_type}")

    # Loss
    loss = F.binary_cross_entropy_with_logits(pred, edge_label.float())
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(
    model: CeliacGNN,
    data: HeteroData,
    target_edge_type: Tuple[str, str, str],
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model on validation/test set.
    Returns dict of metrics.
    """
    model.eval()

    z_dict = model.encode(data.to(device))

    src_type, rel, dst_type = target_edge_type
    edge_label_index = data[target_edge_type].edge_label_index
    edge_label = data[target_edge_type].edge_label

    if dst_type == "phenotype" and src_type == "gene":
        pred = model.decode_gene_phenotype(z_dict, edge_label_index)
    elif dst_type == "phenotype" and src_type == "microbe":
        pred = model.decode_microbe_phenotype(z_dict, edge_label_index)
    else:
        raise ValueError(f"Unknown target edge type: {target_edge_type}")

    pred_probs = torch.sigmoid(pred).cpu().numpy()
    labels = edge_label.cpu().numpy()

    # Handle edge case where all labels are the same
    if len(np.unique(labels)) < 2:
        return {"auroc": 0.5, "auprc": 0.5}

    auroc = roc_auc_score(labels, pred_probs)
    auprc = average_precision_score(labels, pred_probs)

    return {"auroc": auroc, "auprc": auprc}


def train_model(
    data: HeteroData,
    target_edge_type: Tuple[str, str, str],
    hidden_channels: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    lr: float = 0.01,
    epochs: int = 100,
    patience: int = 20,
    seed: int = 42,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Tuple[CeliacGNN, Dict]:
    """
    Full training pipeline.

    Returns:
        Trained model and training history.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if verbose:
        print(f"\nTraining on {device}")
        print(f"Target edge type: {target_edge_type}")

    # Create splits
    train_data, val_data, test_data = create_link_prediction_splits(
        data, target_edge_type, seed=seed
    )

    # Create model
    model = create_model_from_data(
        train_data,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr)

    # Training loop
    history = {
        "train_loss": [],
        "val_auroc": [],
        "val_auprc": [],
    }

    best_val_auroc = 0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(epochs):
        loss = train_epoch(model, train_data, optimizer, target_edge_type, device)
        val_metrics = evaluate(model, val_data, target_edge_type, device)

        history["train_loss"].append(loss)
        history["val_auroc"].append(val_metrics["auroc"])
        history["val_auprc"].append(val_metrics["auprc"])

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: Loss={loss:.4f}, "
                  f"Val AUROC={val_metrics['auroc']:.4f}, "
                  f"Val AUPRC={val_metrics['auprc']:.4f}")

        # Early stopping
        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            best_epoch = epoch
            patience_counter = 0
            # Save best model state
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(best_state)

    # Final evaluation on test set
    test_metrics = evaluate(model, test_data, target_edge_type, device)
    history["test_auroc"] = test_metrics["auroc"]
    history["test_auprc"] = test_metrics["auprc"]
    history["best_epoch"] = best_epoch

    if verbose:
        print(f"\nBest epoch: {best_epoch + 1}")
        print(f"Test AUROC: {test_metrics['auroc']:.4f}")
        print(f"Test AUPRC: {test_metrics['auprc']:.4f}")

    return model, history


def plot_training_history(history: Dict, save_path: Optional[Path] = None) -> None:
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history["train_loss"])
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")

    # Validation metrics
    axes[1].plot(history["val_auroc"], label="AUROC")
    axes[1].plot(history["val_auprc"], label="AUPRC")
    axes[1].axhline(y=history["test_auroc"], color='r', linestyle='--',
                    label=f"Test AUROC: {history['test_auroc']:.3f}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation Metrics")
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()


def run_experiment(
    data_dir: Path = PROCESSED_DIR / "pyg",
    target_task: str = "gene_phenotype",  # or "microbe_phenotype"
    **train_kwargs
) -> Tuple[CeliacGNN, Dict]:
    """
    Run a complete experiment.
    """
    print("\n" + "="*60)
    print(f"RUNNING EXPERIMENT: {target_task}")
    print("="*60)

    # Load data
    print("\nLoading knowledge graph...")
    data = load_kg_from_csv(data_dir)
    print(f"  Node types: {list(data.node_types)}")
    print(f"  Edge types: {list(data.edge_types)}")

    # Determine target edge type
    if target_task == "gene_phenotype":
        target_edge_type = ("gene", "associated_with", "phenotype")
    elif target_task == "microbe_phenotype":
        # This is a virtual edge type for prediction
        # We'll need to create it from the graph structure
        target_edge_type = ("gene", "associated_with", "phenotype")  # Fallback
    else:
        raise ValueError(f"Unknown task: {target_task}")

    # Check if target edge type exists
    if target_edge_type not in data.edge_types:
        print(f"\n  Warning: {target_edge_type} not in graph")
        print(f"  Available: {list(data.edge_types)}")
        return None, {}

    # Train model
    model, history = train_model(data, target_edge_type, **train_kwargs)

    # Save model
    model_path = MODELS_DIR / f"{target_task}_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved model to {model_path}")

    # Plot
    plot_path = FIGURES_DIR / f"{target_task}_training.png"
    plot_training_history(history, plot_path)

    # Save history
    history_path = MODELS_DIR / f"{target_task}_history.json"
    with open(history_path, 'w') as f:
        json.dump({k: v if not isinstance(v, np.ndarray) else v.tolist()
                  for k, v in history.items()}, f, indent=2)

    return model, history


if __name__ == "__main__":
    # Check if data exists
    data_dir = PROCESSED_DIR / "pyg"
    if not (data_dir / "metadata.json").exists():
        print("Knowledge graph not found. Building first...")
        from celiac.build_kg import build_knowledge_graph, save_kg_for_pyg
        kg = build_knowledge_graph()
        save_kg_for_pyg(kg, data_dir)

    # Run experiment
    model, history = run_experiment(
        data_dir=data_dir,
        target_task="gene_phenotype",
        hidden_channels=64,
        num_layers=2,
        dropout=0.3,
        lr=0.01,
        epochs=100,
        patience=20,
        verbose=True
    )
