"""
Main entry point for Celiac Gut-Brain Knowledge Graph project.
"""

import argparse
from pathlib import Path

from celiac.config import PROCESSED_DIR, MODELS_DIR, FIGURES_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Celiac Gut-Brain Knowledge Graph Pipeline"
    )

    parser.add_argument(
        "command",
        choices=["check", "build", "train", "all"],
        help="Command to run"
    )
    parser.add_argument(
        "--task",
        default="gene_phenotype",
        choices=["gene_phenotype", "microbe_phenotype"],
        help="Link prediction task"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=64,
        help="Hidden layer size"
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=2,
        help="Number of GNN layers"
    )

    args = parser.parse_args()

    if args.command == "check":
        print("Running data availability check...")
        import subprocess
        subprocess.run(["python3", "check_data_availability.py"])

    elif args.command == "build":
        print("Building knowledge graph...")
        from celiac.build_kg import build_knowledge_graph, save_kg_for_pyg

        kg = build_knowledge_graph(
            fetch_geo=False,
            fetch_monarch=True,
            use_curated=True,
            expand_phenotype_genes=True
        )

        pyg_dir = PROCESSED_DIR / "pyg"
        save_kg_for_pyg(kg, pyg_dir)

        print(f"\nKnowledge graph saved to {pyg_dir}")

    elif args.command == "train":
        print("Training GNN model...")
        from celiac.train import run_experiment

        model, history = run_experiment(
            data_dir=PROCESSED_DIR / "pyg",
            target_task=args.task,
            hidden_channels=args.hidden,
            num_layers=args.layers,
            epochs=args.epochs,
            verbose=True
        )

        print(f"\nTest AUROC: {history.get('test_auroc', 'N/A'):.4f}")
        print(f"Test AUPRC: {history.get('test_auprc', 'N/A'):.4f}")

    elif args.command == "all":
        print("Running full pipeline...")

        # Step 1: Build KG
        print("\n" + "="*60)
        print("STEP 1: Building Knowledge Graph")
        print("="*60)
        from celiac.build_kg import build_knowledge_graph, save_kg_for_pyg

        kg = build_knowledge_graph()
        pyg_dir = PROCESSED_DIR / "pyg"
        save_kg_for_pyg(kg, pyg_dir)

        # Step 2: Train
        print("\n" + "="*60)
        print("STEP 2: Training GNN")
        print("="*60)
        from celiac.train import run_experiment

        model, history = run_experiment(
            data_dir=pyg_dir,
            target_task=args.task,
            hidden_channels=args.hidden,
            num_layers=args.layers,
            epochs=args.epochs,
            verbose=True
        )

        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        print(f"\nResults:")
        print(f"  Test AUROC: {history.get('test_auroc', 'N/A'):.4f}")
        print(f"  Test AUPRC: {history.get('test_auprc', 'N/A'):.4f}")
        print(f"\nOutputs:")
        print(f"  Model: {MODELS_DIR}")
        print(f"  Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
