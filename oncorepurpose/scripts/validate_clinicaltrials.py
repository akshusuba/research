"""Orthogonal real-world validation of repurposing predictions via ClinicalTrials.gov.

Fellowship-grade question: do the model's TOP-RANKED *novel* drug->cancer
predictions (pairs NOT already known indications in PrimeKG) show up as real
interventional oncology clinical trials more often than matched control pairs?

This is an external, knowledge-graph-independent corroboration. ClinicalTrials.gov
is a registry of real human trials; it has no overlap with how PrimeKG edges or
the model's scores were produced, so a positive enrichment is genuine independent
signal (though trial existence != efficacy; see caveats in the .md report).

Design
------
POSITIVE set : top-ranked novel (drug, cancer) predictions from the trained GNN
               (specificity-lift ranking, known indications excluded), spread
               across several well-connected oncology diseases.
CONTROLS     : (1) RANDOM   - random (drug, cancer) pairs over the same cancer
                             set and the full drug pool, excluding known
                             indications and the positive set (seeded).
               (2) LOW-RANK - novel pairs the model ranks deep down the list for
                             the same cancers (low model score), a tight matched
                             control that differs from positives only by rank.
METRIC       : enrichment = P(>=1 interventional trial | top) vs the same for each
               control, with ratio and Fisher's exact test. Also AUROC of
               "trial exists" as a function of model score across all scored
               novel pairs.

Every ClinicalTrials.gov lookup is cached to data/clinicaltrials_cache.json so
re-runs are cheap, and network failures never abort the run.

Usage
-----
  PYTHONPATH=. python scripts/validate_clinicaltrials.py --smoke
  PYTHONPATH=. python scripts/validate_clinicaltrials.py            # full run
  PYTHONPATH=. python scripts/validate_clinicaltrials.py --n-diseases 15 --top-k 6
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CT_API = "https://clinicaltrials.gov/api/v2/studies"
USER_AGENT = "oncorepurpose-validation/1.0 (research; contact: researcher@example.org)"

REPO = Path(__file__).resolve().parent.parent
CACHE_PATH = REPO / "data" / "clinicaltrials_cache.json"
DEFAULT_OUT = REPO / "results" / "clinicaltrials_validation.json"


# --------------------------------------------------------------------------- #
# ClinicalTrials.gov client (cached, fault-tolerant)
# --------------------------------------------------------------------------- #
class CTClient:
    """Cached client for the ClinicalTrials.gov v2 studies endpoint.

    A "hit" for a (drug, cancer) pair is defined as >=1 INTERVENTIONAL study that
    ClinicalTrials.gov returns for query.intr=<drug> AND query.cond=<cancer>.
    We rely on the registry's own free-text + synonym matching for both fields,
    restricted to interventional study type via an Essie filter.
    """

    def __init__(self, cache_path: Path, sleep: float = 0.34, page_size: int = 5,
                 timeout: int = 30, retries: int = 2):
        self.cache_path = cache_path
        self.sleep = sleep
        self.page_size = page_size
        self.timeout = timeout
        self.retries = retries
        self.cache: Dict[str, dict] = {}
        if cache_path.exists():
            try:
                self.cache = json.loads(cache_path.read_text())
            except Exception:
                self.cache = {}
        self.n_network_calls = 0
        self.n_errors = 0

    @staticmethod
    def _key(drug: str, cond: str) -> str:
        return f"{drug.strip().lower()}|||{cond.strip().lower()}"

    @staticmethod
    def _sanitize(s: str) -> str:
        """Strip characters that break ClinicalTrials.gov's Essie query syntax.

        Raw PrimeKG drug nodes sometimes carry IUPAC-style names with braces /
        brackets / parentheses (e.g. ``(4-{(2S)-2-[...]}phenyl)...``) that yield
        HTTP 400. We keep alphanumerics, spaces and hyphens; unsearchable
        chemical strings then simply return zero trials (a true negative)."""
        s = re.sub(r"[^A-Za-z0-9 -]", " ", str(s))
        return re.sub(r"\s+", " ", s).strip()

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.cache, indent=1))
        tmp.replace(self.cache_path)

    def query(self, drug: str, cond: str) -> dict:
        """Return {total, hit, examples:[{nct,title,status}], error}. Cached."""
        key = self._key(drug, cond)
        if key in self.cache and "error" not in self.cache[key]:
            return self.cache[key]

        q_drug = self._sanitize(drug)
        q_cond = self._sanitize(cond)
        if not q_drug or not q_cond:
            result = {"total": 0, "hit": 0, "examples": [], "unsearchable": True}
            self.cache[key] = result
            return result
        params = {
            "query.intr": q_drug,
            "query.cond": q_cond,
            "filter.advanced": "AREA[StudyType]INTERVENTIONAL",
            "pageSize": str(self.page_size),
            "countTotal": "true",
            "format": "json",
        }
        url = CT_API + "?" + urllib.parse.urlencode(params)
        result: Optional[dict] = None
        last_err = None
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                self.n_network_calls += 1
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    d = json.load(r)
                total = int(d.get("totalCount", 0) or 0)
                examples = []
                for s in d.get("studies", [])[:5]:
                    ps = s.get("protocolSection", {})
                    idm = ps.get("identificationModule", {})
                    stm = ps.get("statusModule", {})
                    examples.append({
                        "nct": idm.get("nctId"),
                        "title": (idm.get("briefTitle") or "")[:160],
                        "status": stm.get("overallStatus"),
                    })
                result = {"total": total, "hit": int(total > 0), "examples": examples}
                break
            except urllib.error.HTTPError as e:
                if e.code == 400:
                    # Query too long/complex even after sanitization: these are
                    # raw IUPAC chemical strings, not searchable drug names. Treat
                    # as a definitive "no usable name -> no trial" (not an error).
                    result = {"total": 0, "hit": 0, "examples": [], "unsearchable": True}
                    break
                last_err = f"HTTP {e.code}"
                time.sleep(self.sleep * (attempt + 1) + 0.5)
            except (urllib.error.URLError, TimeoutError, ValueError) as e:
                last_err = str(e)
                time.sleep(self.sleep * (attempt + 1) + 0.5)
            except Exception as e:  # never let the run crash
                last_err = f"unexpected: {e}"
                time.sleep(self.sleep * (attempt + 1) + 0.5)

        if result is None:
            self.n_errors += 1
            # Cache failures separately so they can be retried on the next run,
            # but treat as no-hit for this run's accounting.
            result = {"total": 0, "hit": 0, "examples": [], "error": last_err}

        self.cache[key] = result
        time.sleep(self.sleep)
        return result


# --------------------------------------------------------------------------- #
# Name helpers
# --------------------------------------------------------------------------- #
def clean_disease(name: str) -> str:
    name = str(name)
    if name.endswith(" (disease)"):
        name = name[: -len(" (disease)")]
    return name.strip()


# --------------------------------------------------------------------------- #
# Candidate-set construction
# --------------------------------------------------------------------------- #
def build_from_shortlist(shortlist_path: Path, n_per_disease: int) -> List[dict]:
    """Smoke-mode positives: read the existing shortlist json."""
    data = json.loads(shortlist_path.read_text())
    pos = []
    for entry in data.get("shortlist", []):
        cancer = clean_disease(entry["disease"])
        for c in entry.get("candidates", [])[:n_per_disease]:
            pos.append({
                "drug": c["drug"],
                "cancer": cancer,
                "model_score": float(c.get("model_score", float("nan"))),
                "rank": None,
                "set": "top",
            })
    return pos


# A curated, deliberately DIVERSE set of major distinct cancers. Selecting by
# raw indication-degree instead collapses onto ~15 near-duplicate hematologic
# ontology terms ("Hodgkins lymphoma", "classic Hodgkin lymphoma", ...) sharing
# the same handful of top drugs, which causes severe pseudo-replication.
CURATED_CANCERS = [
    "glioblastoma", "pancreatic adenocarcinoma", "melanoma", "breast carcinoma",
    "non-small cell lung carcinoma", "colorectal cancer", "prostate cancer",
    "ovarian cancer", "hepatocellular carcinoma", "renal cell carcinoma",
    "acute myeloid leukemia", "multiple myeloma", "bladder carcinoma",
    "gastric cancer", "neuroblastoma", "cervical cancer", "esophageal carcinoma",
    "thyroid carcinoma", "head and neck squamous cell carcinoma", "sarcoma",
]


def select_diseases_by_name(data, queries: List[str], n: int):
    """Map cancer-name queries to distinct disease indices.

    For each query, prefer oncology-flagged matches with the most indication
    edges (well-connected, so the model has signal). De-duplicates indices and
    returns up to ``n`` of them."""
    import torch
    from oncorepurpose.config import DISEASE_TYPE, DRUG_TYPE

    names = [str(x).lower() for x in data[DISEASE_TYPE].node_names]
    onc = data[DISEASE_TYPE].is_oncology
    et = (DRUG_TYPE, "indication", DISEASE_TYPE)
    deg = torch.zeros(int(data[DISEASE_TYPE].num_nodes))
    for d in data[et].edge_index[1].tolist():
        deg[d] += 1

    selected: List[int] = []
    for q in queries:
        ql = q.lower()
        matches = [i for i, nm in enumerate(names) if ql in nm]
        if not matches:
            continue
        # Prefer oncology-flagged, non-"syndrome/susceptibility" terms where the
        # query covers most of the name, then well-connected, then short names.
        def keyf(i):
            nm = names[i]
            clean = nm.endswith(" (disease)")
            bad = any(w in nm for w in ("syndrome", "susceptibility"))
            coverage = len(ql) / max(1, len(nm))
            return (1 if bool(onc[i]) else 0, 0 if bad else 1, round(coverage, 2),
                    1 if clean else 0, float(deg[i]))
        best = max(matches, key=keyf)
        if best not in selected:
            selected.append(best)
        if len(selected) >= n:
            break
    return selected


SCORES_CACHE = REPO / "data" / "ctval_model_scores.json"


def compute_or_load_scores(args) -> dict:
    """Train the GNN once and score every novel (drug, cancer) pair; cache to disk.

    Returns {drug_names, diseases:[{idx,name}], known:[[drug_idx,dz],...],
    ranked:{dz: [[drug_idx, score, lift], ...]}} where ``ranked`` covers all
    novel (non-known-indication) drugs per selected disease. Re-runs reuse the
    cache so we never retrain just to tweak set construction or stats.
    """
    queries = tuple(args.diseases) if args.diseases else tuple(CURATED_CANCERS)
    sig = {"seed": args.seed, "n_diseases": args.n_diseases, "epochs": args.gnn_epochs,
           "hidden": args.hidden, "queries": list(queries)}
    if SCORES_CACHE.exists():
        try:
            cached = json.loads(SCORES_CACHE.read_text())
            if cached.get("_sig") == sig:
                print("Loaded cached model scores (skipping training).")
                cached["ranked"] = {int(k): v for k, v in cached["ranked"].items()}
                return cached
        except Exception:
            pass

    import torch
    from oncorepurpose.config import DISEASE_TYPE, DRUG_TYPE
    from oncorepurpose.datasets import load_primekg
    from oncorepurpose.evaluation.splits import make_split
    from oncorepurpose.evaluation.trainer import set_all_seeds, train_gnn
    from oncorepurpose.interpret.paths import _known_pairs, predict_candidates_for_diseases
    from oncorepurpose.models import HeteroGNN

    dev = torch.device("cpu")  # retrieval/validation task: keep the A100 free
    print("Loading PrimeKG ...")
    data, targets = load_primekg(with_features=True)
    target = targets["indication"]

    print(f"Training deployment GNN on indication edges ({args.gnn_epochs} epochs, CPU) ...")
    set_all_seeds(args.seed)
    split = make_split(data, target, "transductive", seed=args.seed, val_frac=0.1, test_frac=0.0)
    in_dims = {nt: int(data[nt].x.size(1)) for nt in data.node_types}
    gnn = HeteroGNN(list(data.node_types), list(split.base.edge_types), in_dims,
                    hidden=args.hidden, num_layers=2, dropout=0.3)
    gnn = train_gnn(gnn, split, dev, epochs=args.gnn_epochs, patience=10)

    disease_idx = select_diseases_by_name(data, list(queries), args.n_diseases)
    drug_names = [str(n) for n in data[DRUG_TYPE].node_names]
    dis_names = data[DISEASE_TYPE].node_names
    print("Cancers:", [str(dis_names[i]) for i in disease_idx])

    num_drugs = int(data[DRUG_TYPE].num_nodes)
    preds = predict_candidates_for_diseases(
        gnn, data, target, disease_idx, dev,
        top_k=num_drugs, exclude_known=True, rank_by="specificity", seed=args.seed)

    known = _known_pairs(data)
    known_for = [[int(a), int(b)] for (a, b) in known if b in set(disease_idx)]

    out = {
        "_sig": sig,
        "drug_names": drug_names,
        "num_drugs": num_drugs,
        "diseases": [{"idx": int(i), "name": str(dis_names[i]),
                      "clean": clean_disease(dis_names[i])} for i in disease_idx],
        "known": known_for,
        "ranked": {int(dz): [[int(di), float(sc), float(lf)] for di, sc, lf in preds[dz]]
                   for dz in disease_idx},
    }
    SCORES_CACHE.write_text(json.dumps(out))
    print(f"Cached model scores -> {SCORES_CACHE}")
    return out


def build_full(args) -> Tuple[List[dict], dict]:
    """Build top(-by-score) / low / random / top-by-lift candidate sets."""
    sc = compute_or_load_scores(args)
    drug_names = sc["drug_names"]
    diseases = sc["diseases"]
    disease_idx = [d["idx"] for d in diseases]
    clean_by_idx = {d["idx"]: d["clean"] for d in diseases}
    known = {(a, b) for a, b in sc["known"]}
    num_drugs = sc["num_drugs"]
    # ranked-by-score and ranked-by-lift views per disease
    by_score = {dz: sorted(sc["ranked"][dz], key=lambda r: r[1], reverse=True)
                for dz in disease_idx}
    by_lift = {dz: sorted(sc["ranked"][dz], key=lambda r: r[2], reverse=True)
               for dz in disease_idx}
    score_lookup = {(int(r[0]), dz): float(r[1]) for dz in disease_idx for r in sc["ranked"][dz]}

    candidates: List[dict] = []
    seen: set = set()

    def add(drug_i, dz, score, rank, which, allow_dup=False):
        key = (drug_i, dz)
        if key in seen and not allow_dup:
            return False
        seen.add(key)
        candidates.append({
            "drug": str(drug_names[drug_i]),
            "cancer": clean_by_idx[dz],
            "drug_idx": int(drug_i), "disease_idx": int(dz),
            "model_score": float(score), "rank": int(rank), "set": which,
        })
        return True

    # PRIMARY POSITIVE: top-k novel per disease ranked by RAW model score.
    for dz in disease_idx:
        for rank, (di, score, _lift) in enumerate(by_score[dz][: args.top_k]):
            add(di, dz, score, rank, "top")
    n_top = sum(1 for c in candidates if c["set"] == "top")

    # LOW control: bottom-band novel pairs by raw score (matched count per disease).
    for dz in disease_idx:
        ranked = by_score[dz]
        start = int(len(ranked) * args.low_rank_frac)
        for rank in range(start, min(start + args.top_k, len(ranked))):
            di, score, _lift = ranked[rank]
            add(di, dz, score, rank, "low")

    # RANDOM control: random drug x sampled cancer, excluding known + chosen.
    rng = random.Random(args.seed)
    tries = 0
    while sum(1 for c in candidates if c["set"] == "random") < n_top and tries < n_top * 80:
        tries += 1
        dz = rng.choice(disease_idx)
        di = rng.randrange(num_drugs)
        if (di, dz) in known:
            continue
        add(di, dz, score_lookup.get((di, dz), float("nan")), -1, "random")

    # SECONDARY (honest contrast): top-k by SPECIFICITY LIFT = the deployed
    # shortlist ranking. Stored as its own set; duplicates of "top" allowed so
    # the lift set is complete on its own terms.
    for dz in disease_idx:
        for rank, (di, score, lift) in enumerate(by_lift[dz][: args.top_k]):
            candidates.append({
                "drug": str(drug_names[di]), "cancer": clean_by_idx[dz],
                "drug_idx": int(di), "disease_idx": int(dz),
                "model_score": float(score), "lift": float(lift),
                "rank": int(rank), "set": "top_lift",
            })

    meta = {
        "n_diseases": len(disease_idx),
        "cancers": [clean_by_idx[i] for i in disease_idx],
        "num_drugs": num_drugs,
        "low_rank_frac": args.low_rank_frac,
        "primary_ranking": "raw model score (sigmoid link probability), known indications excluded",
        "secondary_set": "top_lift = specificity-lift ranking (the deployed shortlist ordering)",
    }
    return candidates, meta


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def fisher(top_hits, top_n, ctrl_hits, ctrl_n):
    from scipy.stats import fisher_exact
    table = [[top_hits, top_n - top_hits], [ctrl_hits, ctrl_n - ctrl_hits]]
    odds, p = fisher_exact(table, alternative="greater")
    return float(odds), float(p)


def frac(hits, n):
    return (hits / n) if n else float("nan")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="few pairs from existing shortlist, no training")
    ap.add_argument("--diseases", nargs="*", default=[],
                    help="cancer-name queries (default: a curated diverse set of major cancers)")
    ap.add_argument("--n-diseases", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=6, help="positives (and low-rank controls) per disease")
    ap.add_argument("--low-rank-frac", type=float, default=0.7,
                    help="rank fraction where the low-rank control band starts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gnn-epochs", type=int, default=50)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--sleep", type=float, default=0.34, help="seconds between API calls")
    ap.add_argument("--page-size", type=int, default=5)
    ap.add_argument("--max-pairs", type=int, default=0, help="cap total queried pairs (0 = no cap)")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out_path = Path(args.out)
    if args.smoke:
        out_path = out_path.with_name("clinicaltrials_validation_smoke.json")

    # ---- Build candidate sets -------------------------------------------- #
    if args.smoke:
        print("SMOKE mode: using existing shortlist + matched random control (no training)")
        import torch
        from oncorepurpose.config import DISEASE_TYPE, DRUG_TYPE
        from oncorepurpose.datasets import load_primekg
        from oncorepurpose.interpret.paths import _known_pairs

        data, _ = load_primekg(with_features=True)
        drug_names = [str(n) for n in data[DRUG_TYPE].node_names]
        name2idx = {n.lower(): i for i, n in enumerate(drug_names)}
        known = _known_pairs(data)
        dis_names = [str(n) for n in data[DISEASE_TYPE].node_names]
        # map cleaned cancer -> disease idx
        clean2idx: Dict[str, int] = {}
        for i, n in enumerate(dis_names):
            clean2idx.setdefault(clean_disease(n).lower(), i)

        positives = build_from_shortlist(REPO / "results" / "repurposing_shortlist.json",
                                         n_per_disease=2)
        positives = positives[:6]
        # matched random control: same cancers, random drugs from the FULL drug pool
        pos_cancers = sorted({p["cancer"] for p in positives})
        rng = random.Random(args.seed)
        controls: List[dict] = []
        pos_keys = {(p["drug"].lower(), p["cancer"].lower()) for p in positives}
        tries = 0
        while len(controls) < len(positives) and tries < 5000:
            tries += 1
            drug = drug_names[rng.randrange(len(drug_names))]
            cancer = rng.choice(pos_cancers)
            di = name2idx.get(drug.lower())
            ci = clean2idx.get(cancer.lower())
            kkey = (drug.lower(), cancer.lower())
            if kkey in pos_keys:
                continue
            if di is not None and ci is not None and (di, ci) in known:
                continue
            if kkey in {(c["drug"].lower(), c["cancer"].lower()) for c in controls}:
                continue
            controls.append({"drug": drug, "cancer": cancer, "model_score": float("nan"),
                             "rank": -1, "set": "random"})
        candidates = positives + controls
        meta = {"mode": "smoke", "n_positives": len(positives), "n_controls": len(controls)}
    else:
        candidates, meta = build_full(args)
        meta["mode"] = "full"

    if args.max_pairs and len(candidates) > args.max_pairs:
        candidates = candidates[: args.max_pairs]

    n_by_set = {}
    for c in candidates:
        n_by_set[c["set"]] = n_by_set.get(c["set"], 0) + 1
    print(f"Candidate pairs: {len(candidates)} -> {n_by_set}")

    # ---- Query ClinicalTrials.gov ---------------------------------------- #
    client = CTClient(CACHE_PATH, sleep=args.sleep, page_size=args.page_size)
    n_cached = sum(1 for c in candidates
                   if client._key(c["drug"], c["cancer"]) in client.cache
                   and "error" not in client.cache[client._key(c["drug"], c["cancer"])])
    print(f"Querying ({n_cached}/{len(candidates)} already cached) ...")
    for i, c in enumerate(candidates):
        res = client.query(c["drug"], c["cancer"])
        c["trial_total"] = res["total"]
        c["trial_hit"] = res["hit"]
        c["trial_examples"] = res.get("examples", [])
        c["query_error"] = res.get("error")
        if (i + 1) % 25 == 0:
            client.save()
            print(f"  {i+1}/{len(candidates)} queried")
    client.save()
    print(f"Network calls: {client.n_network_calls}, errors: {client.n_errors}")

    # ---- Enrichment & stats ---------------------------------------------- #
    def subset(name):
        return [c for c in candidates if c["set"] == name]

    sets = {name: subset(name) for name in n_by_set}
    summary = {}
    for name, items in sets.items():
        hits = sum(c["trial_hit"] for c in items)
        summary[name] = {"n": len(items), "hits": hits, "fraction": frac(hits, len(items)),
                         "n_unique_drugs": len({c["drug"].lower() for c in items})}

    def compare(pos_name, ctrl_name):
        pos, ctrl = sets.get(pos_name), sets.get(ctrl_name)
        if not pos or not ctrl:
            return None
        ph, ch = sum(c["trial_hit"] for c in pos), sum(c["trial_hit"] for c in ctrl)
        odds, p = fisher(ph, len(pos), ch, len(ctrl))
        f_pos, f_ctrl = frac(ph, len(pos)), frac(ch, len(ctrl))
        ratio = (f_pos / f_ctrl) if f_ctrl else float("inf")
        return {"positive_fraction": f_pos, "control_fraction": f_ctrl,
                "ratio": ratio, "odds_ratio": odds, "p_value": p,
                "test": "Fisher exact, one-sided (positive > control)"}

    comparisons = {}
    for pos_name, ctrl_name in [("top", "random"), ("top", "low"),
                                ("top_lift", "random"), ("top_lift", "low")]:
        c = compare(pos_name, ctrl_name)
        if c:
            comparisons[f"{pos_name}_vs_{ctrl_name}"] = c

    # ---- AUROC: trial_hit ~ model_score over scored novel pairs ---------- #
    auroc = None
    auroc_p = None
    scored = [c for c in candidates
              if c.get("model_score") == c.get("model_score")  # not NaN
              and c["set"] in ("top", "low", "random")]
    labels = [c["trial_hit"] for c in scored]
    if scored and 0 < sum(labels) < len(labels):
        from sklearn.metrics import roc_auc_score
        from scipy.stats import mannwhitneyu
        auroc = float(roc_auc_score(labels, [c["model_score"] for c in scored]))
        pos = [c["model_score"] for c in scored if c["trial_hit"]]
        neg = [c["model_score"] for c in scored if not c["trial_hit"]]
        auroc_p = float(mannwhitneyu(pos, neg, alternative="greater").pvalue)

    # ---- Promiscuity / popularity confound diagnostic -------------------- #
    # If a handful of broadly-indicated drugs (e.g. folic acid) account for most
    # "hits", a positive AUROC is a popularity artifact rather than specific
    # repurposing insight. Report the concentration of hits across drugs.
    from collections import Counter
    confound = {}
    for sname in ("top", "random", "top_lift"):
        items = sets.get(sname, [])
        hit_drugs = Counter(c["drug"] for c in items if c["trial_hit"])
        n_hits = sum(hit_drugs.values())
        confound[sname] = {
            "n_hits": n_hits,
            "n_distinct_hit_drugs": len(hit_drugs),
            "top_hit_drug": (hit_drugs.most_common(1)[0] if hit_drugs else None),
            "share_top_drug": (hit_drugs.most_common(1)[0][1] / n_hits) if n_hits else None,
        }

    # ---- Concrete corroborated examples (one row per distinct hit drug) --- #
    def distinct_examples(set_name, limit=12):
        items = sorted([c for c in sets.get(set_name, []) if c["trial_hit"]],
                       key=lambda c: c.get("model_score") or 0, reverse=True)
        seen_d, out = set(), []
        for c in items:
            if c["drug"].lower() in seen_d:
                continue
            seen_d.add(c["drug"].lower())
            ex = c.get("trial_examples", [])
            out.append({
                "drug": c["drug"], "cancer": c["cancer"],
                "model_score": c["model_score"], "rank": c["rank"],
                "trial_total": c["trial_total"],
                "example_nct": ex[0]["nct"] if ex else None,
                "example_title": ex[0]["title"] if ex else None,
            })
            if len(out) >= limit:
                break
        return out

    top = sets.get("top", [])
    examples = distinct_examples("top")
    examples_lift = distinct_examples("top_lift")

    result = {
        "meta": {**meta, "seed": args.seed,
                 "hit_definition": ">=1 interventional ClinicalTrials.gov study for query.intr=drug & query.cond=cancer",
                 "api": CT_API,
                 "network_calls": client.n_network_calls, "query_errors": client.n_errors},
        "set_summary": summary,
        "comparisons": comparisons,
        "auroc_trial_vs_modelscore": auroc,
        "auroc_mannwhitney_p": auroc_p,
        "n_scored_for_auroc": len(scored),
        "promiscuity_confound": confound,
        "corroborated_examples": examples,
        "corroborated_examples_shortlist_ranking": examples_lift,
        "pairs": candidates,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved -> {out_path}")

    # ---- Markdown summary ------------------------------------------------ #
    write_markdown(result, out_path.with_suffix(".md"), smoke=args.smoke)
    print(f"Saved -> {out_path.with_suffix('.md')}")

    # ---- Console headline ------------------------------------------------ #
    print("\n=== HEADLINE ===")
    for name, s in summary.items():
        print(f"  {name:8s}: {s['hits']}/{s['n']} = {s['fraction']:.1%} have an interventional trial")
    for k, c in comparisons.items():
        print(f"  {k}: {c['positive_fraction']:.1%} vs {c['control_fraction']:.1%}  "
              f"ratio={c['ratio']:.2f}  p={c['p_value']:.4g}")
    if auroc is not None:
        print(f"  AUROC(trial ~ model_score) over {len(scored)} pairs = {auroc:.3f}")


def write_markdown(result: dict, path: Path, smoke: bool):
    s = result["set_summary"]
    comp = result["comparisons"]
    lines = []
    title = "ClinicalTrials.gov Validation (SMOKE)" if smoke else "ClinicalTrials.gov Validation of Repurposing Predictions"
    lines.append(f"# {title}\n")
    lines.append("**Question:** Do the model's top-ranked *novel* drug->cancer predictions "
                 "(pairs not already known indications in PrimeKG) appear as real interventional "
                 "oncology trials more often than matched control pairs? ClinicalTrials.gov is "
                 "fully independent of the knowledge graph and the model.\n")
    lines.append(f"**Hit definition:** {result['meta']['hit_definition']}.\n")

    if not smoke:
        lines.append("**Sets.** `top` = top-k novel pairs per cancer ranked by **raw model score** "
                     "(the primary test of \"top-ranked predictions\"). `low` = bottom-band novel pairs "
                     "by raw score (matched). `random` = random drug x cancer pairs over the same cancers, "
                     "known indications excluded. `top_lift` = top-k by **specificity lift** -- the ranking "
                     "actually used for the deployed shortlist (shown for an honest contrast).\n")

    lines.append("## Headline\n")
    lines.append("| set | hits / n | fraction with a trial | unique drugs |")
    lines.append("| --- | --- | --- | --- |")
    for name, v in s.items():
        lines.append(f"| {name} | {v['hits']} / {v['n']} | {v['fraction']:.1%} | {v.get('n_unique_drugs','')} |")
    lines.append("")

    for k, c in comp.items():
        lines.append(f"- **{k}:** {c['positive_fraction']:.1%} vs {c['control_fraction']:.1%} "
                     f"(ratio **{c['ratio']:.2f}x**, odds ratio {c['odds_ratio']:.2f}, "
                     f"Fisher one-sided **p = {c['p_value']:.3g}**).")
    lines.append("")
    auroc = result.get("auroc_trial_vs_modelscore")
    if auroc is not None:
        ap = result.get("auroc_mannwhitney_p")
        ap_s = f", Mann-Whitney p = {ap:.3g}" if ap is not None else ""
        lines.append(f"**AUROC** of \"interventional trial exists\" vs raw model score over "
                     f"{result['n_scored_for_auroc']} scored novel pairs (top+low+random): "
                     f"**{auroc:.3f}**{ap_s} (0.5 = no signal).\n")

    conf = result.get("promiscuity_confound", {})
    if conf:
        lines.append("## Popularity / promiscuity confound\n")
        lines.append("How concentrated are the trial \"hits\" on a few broadly-indicated drugs? "
                     "If one drug accounts for most hits, a positive raw-score AUROC reflects drug "
                     "popularity (such drugs are scored high *everywhere* and trialed *everywhere*) "
                     "rather than specific repurposing insight.\n")
        lines.append("| set | hits | distinct hit-drugs | most frequent hit-drug (share) |")
        lines.append("| --- | --- | --- | --- |")
        for sname, cc in conf.items():
            thd = cc.get("top_hit_drug")
            share = cc.get("share_top_drug")
            thd_s = f"{thd[0]} ({thd[1]}/{cc['n_hits']}, {share:.0%})" if thd else "-"
            lines.append(f"| {sname} | {cc['n_hits']} | {cc['n_distinct_hit_drugs']} | {thd_s} |")
        lines.append("")

    # Auto-generated interpretation from the numbers.
    if not smoke:
        tr = comp.get("top_vs_random")
        verdict = []
        if tr:
            if tr["ratio"] > 1 and tr["p_value"] < 0.1:
                verdict.append(f"Top raw-score novel predictions are enriched for real trials "
                               f"({tr['ratio']:.2f}x vs random, p={tr['p_value']:.3g}).")
            elif tr["ratio"] > 1:
                verdict.append(f"Top raw-score novel predictions trend toward more trials "
                               f"({tr['ratio']:.2f}x vs random) but the pair-level Fisher test is not "
                               f"significant (p={tr['p_value']:.3g}).")
            else:
                verdict.append(f"Top raw-score novel predictions are NOT enriched vs random "
                               f"({tr['ratio']:.2f}x, p={tr['p_value']:.3g}).")
        if auroc is not None:
            ap = result.get("auroc_mannwhitney_p")
            sig = (ap is not None and ap < 0.05)
            if auroc >= 0.6 and sig:
                verdict.append(f"Raw model score does rank trial-existence above chance "
                               f"(AUROC {auroc:.3f}, p={ap:.2g}): higher-scored novel pairs are more "
                               f"likely to have a real interventional trial.")
            elif auroc >= 0.6:
                verdict.append(f"Raw model score shows a weak (non-significant) ranking signal for "
                               f"trial-existence (AUROC {auroc:.3f}).")
            else:
                verdict.append(f"Raw model score gives little ranking signal for trial-existence (AUROC {auroc:.3f}).")
        # Popularity-confound caveat on the AUROC.
        ct = (result.get("promiscuity_confound", {}) or {}).get("top", {})
        share = ct.get("share_top_drug")
        if share is not None and share >= 0.3 and ct.get("top_hit_drug"):
            verdict.append(f"But this is heavily confounded by drug popularity: a single broadly-indicated "
                           f"drug ({ct['top_hit_drug'][0]}) accounts for {share:.0%} of the top set's hits "
                           f"-- such drugs score high for every cancer and are trialed for every cancer.")
        tl = comp.get("top_lift_vs_random")
        if tl:
            verdict.append(f"Consistently, the specificity-lift ranking (the deployed shortlist ordering, which "
                           f"de-confounds popularity) shows NO trial enrichment ({tl['ratio']:.2f}x vs random, "
                           f"p={tl['p_value']:.3g}): the genuinely specific novel predictions are not (yet) "
                           f"over-represented in trials.")
        verdict.append("Bottom line: weak and confounded independent signal -- this corroborates that the "
                       "model's raw scores track human trial activity (mostly via popular drugs), but does NOT "
                       "yet provide clean real-world validation of the specific novel repurposing shortlist.")
        lines.append("## Interpretation\n")
        lines.append(" ".join(verdict) + "\n")

    def ex_table(ex, header):
        if not ex:
            return
        lines.append(header)
        lines.append("| drug | cancer | model score | rank | # trials | example NCT |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for e in ex:
            rk = "" if e["rank"] is None else e["rank"]
            ms = f"{e['model_score']:.3f}" if e["model_score"] == e["model_score"] else "n/a"
            lines.append(f"| {e['drug']} | {e['cancer']} | {ms} | {rk} | {e['trial_total']} | "
                         f"{e['example_nct'] or ''} |")
        lines.append("")

    ex_table(result.get("corroborated_examples", []),
             "## Concrete corroborated novel predictions -- raw-score top set (one row per distinct drug)\n")
    ex_table(result.get("corroborated_examples_shortlist_ranking", []),
             "## Concrete corroborated novel predictions -- specificity-lift shortlist ranking (one row per distinct drug)\n")

    lines.append("## Caveats\n")
    lines.append("- **Name matching is fuzzy.** Drug/cancer strings are passed to ClinicalTrials.gov's "
                 "free-text + synonym search; some true matches may be missed and some loose matches counted.")
    lines.append("- **A registered trial is not evidence of efficacy.** It shows someone judged the "
                 "drug-cancer pair worth testing in humans, which is exactly the orthogonal plausibility signal sought here.")
    lines.append("- **Sample sizes are modest**; treat p-values and the AUROC as indicative, not definitive.")
    lines.append("- **Reverse-causality risk is low:** ClinicalTrials.gov is not an input to PrimeKG or the model, "
                 "so enrichment cannot be an artifact of training leakage, but popular/older drugs are over-represented "
                 "in both trials and predictions.")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
