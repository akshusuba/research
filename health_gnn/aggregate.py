"""Collect all results/*.json into a single leaderboard."""
import os
import glob
import json

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    recs = []
    for p in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        if os.path.basename(p) == "leaderboard.json":
            continue
        with open(p) as f:
            recs.append(json.load(f))
    recs.sort(key=lambda r: r["name"])

    print(f"\n{'experiment':<22}{'task':<14}{'metric':<8}"
          f"{'GNN':>8}{'XGB':>8}{'MLP':>8}{'margin':>9}  win")
    print("-" * 85)
    wins = 0
    for r in recs:
        m = r["primary_metric"]
        g, x, ml = r["gnn"][m], r["xgboost"][m], r["mlp"][m]
        wins += int(r["gnn_wins"])
        print(f"{r['name']:<22}{r['task']:<14}{m:<8}"
              f"{g:>8.4f}{x:>8.4f}{ml:>8.4f}{r['margin_vs_best_baseline']:>+9.4f}"
              f"  {'YES' if r['gnn_wins'] else 'NO'}")
    print("-" * 85)
    print(f"GNN wins both baselines in {wins}/{len(recs)} experiments.")
    with open(os.path.join(RESULTS, "leaderboard.json"), "w") as f:
        json.dump(recs, f, indent=2)


if __name__ == "__main__":
    main()
