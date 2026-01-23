"""
Visualization utilities for Celiac Gut-Brain GNN.
Creates t-SNE plots of learned embeddings.
"""

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path
from typing import Dict, Optional

from celiac.config import PROCESSED_DIR, MODELS_DIR, FIGURES_DIR
from celiac.train import load_kg_from_csv
from celiac.models import create_model_from_data


def extract_embeddings(model, data, device='cpu') -> Dict[str, np.ndarray]:
    """
    Extract learned node embeddings from the trained model.
    """
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        z_dict = model.encode(data.to(device))

    embeddings = {}
    for node_type, z in z_dict.items():
        embeddings[node_type] = z.cpu().numpy()

    return embeddings


def plot_tsne_embeddings(
    embeddings: Dict[str, np.ndarray],
    node_labels: Optional[Dict[str, list]] = None,
    save_path: Optional[Path] = None,
    perplexity: int = 10,
    random_state: int = 42
) -> None:
    """
    Create t-SNE visualization of node embeddings.
    """
    # Combine all embeddings
    all_embeddings = []
    all_types = []
    all_labels = []

    type_colors = {
        'microbe': '#e41a1c',      # Red
        'metabolite': '#377eb8',   # Blue
        'gene': '#4daf4a',         # Green
        'phenotype': '#984ea3'     # Purple
    }

    for node_type, emb in embeddings.items():
        all_embeddings.append(emb)
        all_types.extend([node_type] * len(emb))
        if node_labels and node_type in node_labels:
            all_labels.extend(node_labels[node_type])
        else:
            all_labels.extend([''] * len(emb))

    X = np.vstack(all_embeddings)

    # Run t-SNE
    print(f"Running t-SNE on {X.shape[0]} nodes with {X.shape[1]} dimensions...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, X.shape[0] - 1),
        random_state=random_state,
        n_iter=1000
    )
    X_2d = tsne.fit_transform(X)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each node type
    offset = 0
    for node_type, emb in embeddings.items():
        n = len(emb)
        indices = range(offset, offset + n)
        color = type_colors.get(node_type, '#999999')

        ax.scatter(
            X_2d[indices, 0],
            X_2d[indices, 1],
            c=color,
            label=f'{node_type} (n={n})',
            alpha=0.7,
            s=50
        )
        offset += n

    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlabel('t-SNE dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE dimension 2', fontsize=12)
    ax.set_title('Learned Node Embeddings (t-SNE)', fontsize=14)

    # Remove axes spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved t-SNE plot to {save_path}")

    plt.show()


def plot_gene_pathway_clustering(
    embeddings: Dict[str, np.ndarray],
    gene_ids: list,
    save_path: Optional[Path] = None,
    perplexity: int = 10,
    random_state: int = 42
) -> None:
    """
    Create t-SNE visualization showing gene clustering by pathway.
    """
    # Define pathway groups
    pathway_genes = {
        'Serotonin': ['TPH1', 'TPH2', 'SLC6A4', 'HTR1A', 'HTR2A', 'HTR3A', 'HTR4'],
        'SCFA': ['FFAR2', 'FFAR3', 'HDAC1', 'HDAC2', 'HDAC3'],
        'Kynurenine': ['IDO1', 'TDO2', 'KYNU', 'KMO'],
        'GABA': ['GAD1', 'GAD2', 'GABRA1', 'GABRB2', 'SLC6A1'],
        'Dopamine': ['TH', 'DDC', 'DRD1', 'DRD2', 'SLC6A3'],
        'Immune': ['IL1B', 'IL6', 'IL10', 'TNF', 'IFNG', 'TGFB1'],
        'Celiac': ['HLA-DQA1', 'HLA-DQB1', 'TG', 'CTLA4', 'IL2', 'IL21']
    }

    # Map genes to pathways
    gene_to_pathway = {}
    for pathway, genes in pathway_genes.items():
        for gene in genes:
            gene_to_pathway[gene] = pathway

    # Get gene embeddings
    if 'gene' not in embeddings:
        print("No gene embeddings found")
        return

    gene_emb = embeddings['gene']

    # Assign pathways
    pathways = []
    for gene_id in gene_ids:
        pathways.append(gene_to_pathway.get(gene_id, 'Other'))

    # Run t-SNE on genes only
    print(f"Running t-SNE on {len(gene_ids)} genes...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, len(gene_ids) - 1),
        random_state=random_state,
        n_iter=1000
    )
    X_2d = tsne.fit_transform(gene_emb)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    pathway_colors = {
        'Serotonin': '#e41a1c',
        'SCFA': '#377eb8',
        'Kynurenine': '#4daf4a',
        'GABA': '#984ea3',
        'Dopamine': '#ff7f00',
        'Immune': '#ffff33',
        'Celiac': '#a65628',
        'Other': '#999999'
    }

    for pathway in pathway_colors.keys():
        indices = [i for i, p in enumerate(pathways) if p == pathway]
        if indices:
            ax.scatter(
                X_2d[indices, 0],
                X_2d[indices, 1],
                c=pathway_colors[pathway],
                label=f'{pathway} (n={len(indices)})',
                alpha=0.7,
                s=60
            )

    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlabel('t-SNE dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE dimension 2', fontsize=12)
    ax.set_title('Gene Embeddings Colored by Pathway', fontsize=14)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved pathway clustering plot to {save_path}")

    plt.show()


def create_visualizations(
    data_dir: Path = PROCESSED_DIR / "pyg",
    model_path: Path = MODELS_DIR / "gene_phenotype_model.pt",
    output_dir: Path = FIGURES_DIR
) -> None:
    """
    Create all visualization figures.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)

    # Load data
    print("\nLoading knowledge graph...")
    data = load_kg_from_csv(data_dir)

    # Load model
    print("Loading trained model...")
    model = create_model_from_data(data, hidden_channels=32, num_layers=2, dropout=0.3)

    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"  Loaded weights from {model_path}")
    else:
        print(f"  Warning: Model file not found at {model_path}")
        print("  Using randomly initialized model (for testing)")

    # Extract embeddings
    print("\nExtracting embeddings...")
    embeddings = extract_embeddings(model, data)
    for node_type, emb in embeddings.items():
        print(f"  {node_type}: {emb.shape}")

    # Get node labels
    gene_ids = data['gene'].node_ids if hasattr(data['gene'], 'node_ids') else []

    # Create t-SNE plot of all node types
    print("\nCreating t-SNE visualization (all nodes)...")
    plot_tsne_embeddings(
        embeddings,
        save_path=output_dir / "tsne_all_nodes.png",
        perplexity=15
    )

    # Create pathway clustering plot (genes only)
    print("\nCreating gene pathway clustering plot...")
    plot_gene_pathway_clustering(
        embeddings,
        gene_ids,
        save_path=output_dir / "tsne_gene_pathways.png",
        perplexity=10
    )

    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    create_visualizations()
