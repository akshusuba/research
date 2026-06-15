"""Fetch EGFR (CHEMBL203) bioactivity from ChEMBL.

Keeps records with a canonical SMILES and a pchembl_value.
Writes a deduplicated CSV: smiles, pchembl_value.
"""
import sys
import time
import pandas as pd

TARGET = "CHEMBL203"  # EGFR (Epidermal growth factor receptor), human
OUT = "/home/elrarun/code/research/molgnn/data/egfr_raw.csv"


def fetch():
    from chembl_webresource_client.new_client import new_client

    activity = new_client.activity
    print(f"Querying ChEMBL activities for target {TARGET} ...", flush=True)
    # Filter on the API side: standard_type IC50/Ki etc. captured via pchembl_value presence.
    res = activity.filter(
        target_chembl_id=TARGET,
        pchembl_value__isnull=False,
    ).only(
        ["molecule_chembl_id", "canonical_smiles", "pchembl_value",
         "standard_type", "standard_relation", "target_organism"]
    )

    rows = []
    t0 = time.time()
    for i, r in enumerate(res):
        rows.append(r)
        if i % 1000 == 0 and i > 0:
            print(f"  fetched {i} records ({time.time()-t0:.0f}s)", flush=True)
    print(f"Total raw records: {len(rows)} ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def main():
    df = fetch()
    df.to_csv(OUT.replace(".csv", "_full.csv"), index=False)
    print("Columns:", list(df.columns), flush=True)
    print("Organisms:", df["target_organism"].value_counts().to_dict() if "target_organism" in df else "n/a", flush=True)

    df = df.dropna(subset=["canonical_smiles", "pchembl_value"]).copy()
    df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df = df.dropna(subset=["pchembl_value"])

    # Restrict to '=' relations where available (drop censored > / < to reduce noise)
    if "standard_relation" in df.columns:
        before = len(df)
        df = df[df["standard_relation"].isin(["=", None]) | df["standard_relation"].isna()]
        print(f"After relation filter: {len(df)} (from {before})", flush=True)

    # Aggregate duplicate measurements per molecule by median pchembl
    agg = (df.groupby("canonical_smiles")["pchembl_value"]
             .median().reset_index())
    agg.columns = ["smiles", "pchembl_value"]
    agg.to_csv(OUT, index=False)
    print(f"Unique molecules with pchembl: {len(agg)}", flush=True)
    for thr in (6.0, 6.5, 7.0):
        pos = (agg["pchembl_value"] >= thr).sum()
        print(f"  thr={thr}: active={pos} ({pos/len(agg)*100:.1f}%), inactive={len(agg)-pos}", flush=True)


if __name__ == "__main__":
    main()
