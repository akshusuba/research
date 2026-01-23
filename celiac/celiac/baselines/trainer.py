"""
Unified training interface for all baseline models.

Provides consistent training, evaluation, and comparison across:
- KGE models (TransE, DistMult, RotatE, ComplEx)
- GNN baselines (R-GCN, CompGCN, HGT)
- Non-graph (Node2Vec)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit
from typing import Dict, List, Optional, Tuple, Type, Callable, Any
import copy

from .kge_models import (
    TransEModel, DistMultModel, RotatEModel, ComplExModel, KGELinkPredictor
)
from .gnn_baselines import RGCNModel, CompGCNModel, HGTModel
from .node2vec_baseline import Node2VecModel, Node2VecMLP

from ..evaluation.metrics import compute_all_metrics


# Registry of all available baselines
BASELINE_REGISTRY = {
    # KGE models
    'TransE': {'class': TransEModel, 'type': 'kge'},
    'DistMult': {'class': DistMultModel, 'type': 'kge'},
    'RotatE': {'class': RotatEModel, 'type': 'kge'},
    'ComplEx': {'class': ComplExModel, 'type': 'kge'},
    # GNN baselines
    'R-GCN': {'class': RGCNModel, 'type': 'gnn'},
    'CompGCN': {'class': CompGCNModel, 'type': 'gnn'},
    'HGT': {'class': HGTModel, 'type': 'gnn'},
    # Non-graph
    'Node2Vec': {'class': Node2VecModel, 'type': 'node2vec'},
    'Node2Vec-MLP': {'class': Node2VecMLP, 'type': 'node2vec'},
}


class BaselineTrainer:
    """
    Unified trainer for all baseline models.

    Handles:
    - Model instantiation
    - Training with early stopping
    - Evaluation with full metrics
    - Negative sampling
    """

    def __init__(
        self,
        model_name: str,
        data: HeteroData,
        target_edge_type: Tuple[str, str, str],
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 0.01,
        weight_decay: float = 1e-5,
        device: str = 'auto',
        **model_kwargs,
    ):
        """
        Args:
            model_name: Name of baseline model (from BASELINE_REGISTRY)
            data: PyG HeteroData object
            target_edge_type: Edge type for link prediction task
            hidden_channels: Hidden dimension
            num_layers: Number of layers (for GNN models)
            dropout: Dropout rate
            learning_rate: Learning rate
            weight_decay: Weight decay
            device: Device ('auto', 'cuda', 'cpu')
            **model_kwargs: Additional model-specific arguments
        """
        if model_name not in BASELINE_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(BASELINE_REGISTRY.keys())}")

        self.model_name = model_name
        self.model_info = BASELINE_REGISTRY[model_name]
        self.data = data
        self.target_edge_type = target_edge_type
        self.hidden_channels = hidden_channels
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Create model
        self.model = self._create_model(
            num_layers=num_layers,
            dropout=dropout,
            **model_kwargs
        )
        self.model = self.model.to(self.device)

        # Training state
        self.optimizer = None
        self.best_model_state = None
        self.training_history = []

    def _create_model(self, **kwargs) -> nn.Module:
        """Create model instance."""
        model_class = self.model_info['class']
        model_type = self.model_info['type']

        if model_type == 'kge':
            return KGELinkPredictor(
                data=self.data,
                model_class=model_class,
                embedding_dim=self.hidden_channels,
            )
        elif model_type == 'gnn':
            return model_class(
                data=self.data,
                hidden_channels=self.hidden_channels,
                **kwargs
            )
        elif model_type == 'node2vec':
            return model_class(
                data=self.data,
                embedding_dim=self.hidden_channels,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _sample_negatives(
        self,
        edge_index: torch.Tensor,
        num_nodes_src: int,
        num_nodes_dst: int,
        num_neg: int,
    ) -> torch.Tensor:
        """Sample negative edges."""
        # Create set of positive edges for fast lookup
        pos_edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))

        neg_src = []
        neg_dst = []

        while len(neg_src) < num_neg:
            src = torch.randint(0, num_nodes_src, (num_neg * 2,))
            dst = torch.randint(0, num_nodes_dst, (num_neg * 2,))

            for s, d in zip(src.tolist(), dst.tolist()):
                if (s, d) not in pos_edges:
                    neg_src.append(s)
                    neg_dst.append(d)
                    if len(neg_src) >= num_neg:
                        break

        return torch.stack([
            torch.tensor(neg_src[:num_neg]),
            torch.tensor(neg_dst[:num_neg])
        ], dim=0)

    def train(
        self,
        train_data: HeteroData,
        val_data: HeteroData,
        epochs: int = 200,
        patience: int = 20,
        neg_sampling_ratio: float = 1.0,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            train_data: Training data with positive edges
            val_data: Validation data
            epochs: Maximum epochs
            patience: Early stopping patience
            neg_sampling_ratio: Ratio of negative to positive samples
            verbose: Print progress

        Returns:
            Training history dict
        """
        # Handle Node2Vec separately (requires pre-training)
        if self.model_info['type'] == 'node2vec':
            return self._train_node2vec(train_data, val_data, epochs, verbose)

        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Get edge data
        src_type, rel, dst_type = self.target_edge_type
        pos_edge_index = train_data[self.target_edge_type].edge_index.to(self.device)

        # Get number of nodes
        num_nodes_src = train_data[src_type].num_nodes if hasattr(train_data[src_type], 'num_nodes') else pos_edge_index[0].max().item() + 1
        num_nodes_dst = train_data[dst_type].num_nodes if hasattr(train_data[dst_type], 'num_nodes') else pos_edge_index[1].max().item() + 1

        best_val_metric = 0
        patience_counter = 0
        history = {'train_loss': [], 'val_auroc': [], 'val_auprc': []}

        for epoch in range(epochs):
            # Training
            self.model.train()
            self.optimizer.zero_grad()

            # Sample negatives
            num_neg = int(pos_edge_index.size(1) * neg_sampling_ratio)
            neg_edge_index = self._sample_negatives(
                pos_edge_index.cpu(), num_nodes_src, num_nodes_dst, num_neg
            ).to(self.device)

            # Forward pass
            if self.model_info['type'] == 'gnn':
                z_dict = self.model(train_data.to(self.device))
                pos_scores = self.model.decode(z_dict, self.target_edge_type, pos_edge_index)
                neg_scores = self.model.decode(z_dict, self.target_edge_type, neg_edge_index)
            else:  # KGE
                pos_scores = self.model(self.target_edge_type, pos_edge_index)
                neg_scores = self.model(self.target_edge_type, neg_edge_index)

            # Loss
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_scores, torch.ones_like(pos_scores)
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_scores, torch.zeros_like(neg_scores)
            )
            loss = pos_loss + neg_loss

            loss.backward()
            self.optimizer.step()

            # Validation
            val_metrics = self.evaluate(val_data)
            val_auroc = val_metrics['auroc']

            history['train_loss'].append(loss.item())
            history['val_auroc'].append(val_auroc)
            history['val_auprc'].append(val_metrics['auprc'])

            # Early stopping
            if val_auroc > best_val_metric:
                best_val_metric = val_auroc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Val AUROC={val_auroc:.4f}")

            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        self.training_history = history
        return history

    def _train_node2vec(
        self,
        train_data: HeteroData,
        val_data: HeteroData,
        epochs: int,
        verbose: bool,
    ) -> Dict[str, Any]:
        """Train Node2Vec model."""
        if verbose:
            print("Training Node2Vec embeddings...")

        # Train embeddings
        self.model.train_embeddings(
            epochs=epochs,
            batch_size=128,
            lr=self.learning_rate,
            device=str(self.device),
        )

        # Evaluate
        val_metrics = self.evaluate(val_data)
        if verbose:
            print(f"Node2Vec Val AUROC: {val_metrics['auroc']:.4f}")

        return {'val_auroc': [val_metrics['auroc']]}

    @torch.no_grad()
    def evaluate(
        self,
        data: HeteroData,
        return_predictions: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate the model.

        Args:
            data: Data to evaluate on
            return_predictions: Whether to return predictions

        Returns:
            Dict of metrics
        """
        self.model.eval()
        data = data.to(self.device)

        src_type, rel, dst_type = self.target_edge_type

        # Get positive edges
        if hasattr(data[self.target_edge_type], 'edge_label_index'):
            edge_index = data[self.target_edge_type].edge_label_index
            labels = data[self.target_edge_type].edge_label
        else:
            # Use all edges as positives, sample negatives
            pos_edge_index = data[self.target_edge_type].edge_index
            num_nodes_src = data[src_type].num_nodes if hasattr(data[src_type], 'num_nodes') else pos_edge_index[0].max().item() + 1
            num_nodes_dst = data[dst_type].num_nodes if hasattr(data[dst_type], 'num_nodes') else pos_edge_index[1].max().item() + 1

            neg_edge_index = self._sample_negatives(
                pos_edge_index.cpu(), num_nodes_src, num_nodes_dst, pos_edge_index.size(1)
            ).to(self.device)

            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            labels = torch.cat([
                torch.ones(pos_edge_index.size(1)),
                torch.zeros(neg_edge_index.size(1))
            ]).to(self.device)

        # Get predictions
        if self.model_info['type'] == 'gnn' or self.model_info['type'] == 'node2vec':
            z_dict = self.model(data)
            scores = self.model.decode(z_dict, self.target_edge_type, edge_index)
        else:  # KGE
            scores = self.model(self.target_edge_type, edge_index)

        scores = torch.sigmoid(scores)

        # Compute metrics
        metrics = compute_all_metrics(labels.cpu(), scores.cpu())

        if return_predictions:
            return metrics, {'scores': scores.cpu(), 'labels': labels.cpu()}
        return metrics

    def get_embeddings(self) -> Dict[str, torch.Tensor]:
        """Get learned node embeddings."""
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, 'get_embeddings'):
                return self.model.get_embeddings()
            elif self.model_info['type'] == 'gnn' or self.model_info['type'] == 'node2vec':
                return self.model(self.data.to(self.device))
            else:
                raise NotImplementedError(f"get_embeddings not available for {self.model_name}")


def train_all_baselines(
    data: HeteroData,
    target_edge_type: Tuple[str, str, str],
    train_data: HeteroData,
    val_data: HeteroData,
    test_data: HeteroData,
    baselines: Optional[List[str]] = None,
    hidden_channels: int = 64,
    epochs: int = 200,
    patience: int = 20,
    device: str = 'auto',
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Train and evaluate all baseline models.

    Args:
        data: Full HeteroData object
        target_edge_type: Edge type for link prediction
        train_data: Training split
        val_data: Validation split
        test_data: Test split
        baselines: List of baseline names (None = all)
        hidden_channels: Hidden dimension
        epochs: Max epochs
        patience: Early stopping patience
        device: Device
        verbose: Print progress

    Returns:
        Dict mapping model name -> {'trainer', 'train_history', 'test_metrics'}
    """
    if baselines is None:
        baselines = list(BASELINE_REGISTRY.keys())

    results = {}

    for model_name in baselines:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training {model_name}")
            print('='*60)

        trainer = BaselineTrainer(
            model_name=model_name,
            data=data,
            target_edge_type=target_edge_type,
            hidden_channels=hidden_channels,
            device=device,
        )

        history = trainer.train(
            train_data=train_data,
            val_data=val_data,
            epochs=epochs,
            patience=patience,
            verbose=verbose,
        )

        test_metrics = trainer.evaluate(test_data)

        if verbose:
            print(f"\nTest Results for {model_name}:")
            print(f"  AUROC: {test_metrics['auroc']:.4f}")
            print(f"  AUPRC: {test_metrics['auprc']:.4f}")
            print(f"  Hits@10: {test_metrics['hits@10']:.4f}")

        results[model_name] = {
            'trainer': trainer,
            'train_history': history,
            'test_metrics': test_metrics,
        }

    return results
