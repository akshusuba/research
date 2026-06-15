#!/usr/bin/env python
"""Graph-removal ablation on a real spatial dataset (the falsification test).

Runs the identical GNN on the intact spatial kNN graph, a degree-matched
random-edge graph, and an empty graph, plus a blind MLP reference, under the
same cross-section split. If the GNN's edge truly carries the signal, intact
should dominate shuffled/empty (which collapse toward the MLP).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatialgnn.ablation import print_summary, run_graph_ablation
from spatialgnn.config import RESULTS_DIR, ModelConfig, SplitConfig, TrainConfig
from spatialgnn.data.real import load_h5ad_spatial
from spatialgnn.splits import make_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--label-key", required=True)
    ap.add_argument("--sample-key", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--device", default=None, help="cuda|cpu (default: cuda if available)")
    ap.add_argument("--mode", default=None,
                    help="cross_slice|within_slice|stratified "
                         "(default: cross_slice if --sample-key else within_slice)")
    ap.add_argument("--out", default="real_graph_ablation.json")
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds = load_h5ad_spatial(args.h5ad, label_key=args.label_key,
                           sample_key=args.sample_key, k_neighbors=args.k)
    mode = args.mode or ("cross_slice" if args.sample_key else "within_slice")
    scfg = SplitConfig(mode=mode)
    mcfg = ModelConfig(encoder="sage")
    tcfg = TrainConfig(epochs=args.epochs, patience=40, device=device)

    print(f"=== graph-removal ablation: mode={mode} device={device} ===")
    res = run_graph_ablation(ds, make_split, scfg, mcfg, tcfg, seeds=args.seeds)
    print_summary(res)

    path = os.path.join(RESULTS_DIR, args.out)
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
