#!/usr/bin/env python
"""Extend the OncoEvidence learning track to 8 parts.

Adds two new self-contained notebooks based on the recent improvements:
  Part 7 -- prospective NAMED case studies (builds on Part 5's temporal split)
  Part 8 -- popularity-DECONFOUNDED orthogonal validation (ClinicalTrials.gov)

It also inserts the matching Finding 7 / Finding 8 sections into the master
self-contained notebook (and refreshes the Part 4 copy), and relabels the track
from "of 6" to "of 8" everywhere. Pure-JSON notebook manipulation (no nbformat
dependency). Idempotent: re-running removes any prior Finding 7/8 insertions first.

Run:
    .venv/bin/python scripts/build_parts_7_8.py
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

NB = Path("notebooks")
MASTER = NB / "oncoevidence_colab.ipynb"
PART4 = NB / "oncoevidence_part4_full.ipynb"
PART5 = NB / "oncoevidence_part5_prospective.ipynb"
PART6 = NB / "oncoevidence_part6_faithfulness.ipynb"
PART7 = NB / "oncoevidence_part7_named_cases.ipynb"
PART8 = NB / "oncoevidence_part8_deconfounded.ipynb"


def load(p):
    return json.loads(Path(p).read_text())


def dump(p, nb):
    Path(p).write_text(json.dumps(nb, indent=1))
    print(f"wrote {p}: {len(nb['cells'])} cells")


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


def src(cell):
    return "".join(cell["source"])


# --------------------------------------------------------------------------- #
# Shared prose
# --------------------------------------------------------------------------- #
TRACK8 = (
    "This is one of eight notebooks. Parts 1 to 3 build the core pipeline (foundations, "
    "mechanisms, evidence and recovery), Part 4 is the full self-contained notebook, and "
    "Parts 5 to 8 are advanced extensions (prospective validation, mechanism faithfulness, "
    "prospective named cases, and popularity-deconfounded trial validation). Each part runs "
    "on its own and is Colab-ready; set the runtime to GPU under Runtime, Change runtime "
    "type. The toolbox cells (L1, L2, and so on) are the same building blocks in every part "
    "and match the full Part 4 notebook, so as we work through the track we are assembling "
    "the complete system one piece at a time.\n\n"
    "Everything here is hypothesis-generating research, not medical advice.")

OLD_TRACK_FRAGMENT = ("Part 4 is the full self-contained notebook, and Parts 5 and 6 are "
                      "advanced extensions (prospective validation and mechanism faithfulness).")
NEW_TRACK_FRAGMENT = ("Part 4 is the full self-contained notebook, and Parts 5 to 8 are "
                      "advanced extensions (prospective validation, mechanism faithfulness, "
                      "prospective named cases, and popularity-deconfounded trial validation).")


# --------------------------------------------------------------------------- #
# Finding 7 -- prospective named case studies (reuses Finding 5 objects)
# --------------------------------------------------------------------------- #
F7_MD = r"""
## 11. Finding 7: prospective NAMED case studies

Finding 5 gave a single number (a prospective AUROC). A number is easy to wave away, so
here we make it concrete. Using the very same time-split, we take the model trained only on
pre-$T$ structure and, for each held-out *future* indication, score every drug against that
cancer and read off where the real drug landed. A high rank means the system would have
surfaced a genuine, later-established indication near the top of a blind screen, before that
indication was in the graph.

What we recorded on the full run (cutoff $T=2005$, single seed): of 101 future indications,
the GNN placed 11 in the top 50 of about 7{,}956 candidate drugs (a 17.3x enrichment over a
random screen), versus 1 for the structure-blind control. The named hits are recognizable
post-2005 oncology approvals the model had no indication edge for: romidepsin and pralatrexate
for cutaneous T-cell lymphoma, nilotinib for CML, trabectedin for ovarian cancers, plerixafor
for myeloma. The graph's advantage is concentrated where a shortlist matters, the very top of
the ranking; across the bulk of pairs the structure-blind control is competitive, which we
report rather than hide.

This cell reuses the temporal split and resolved years from Finding 5 (run that cell first).
In fast mode it trains briefly on a small sample, so the live ranks are far noisier than the
recorded full-run numbers above; the point here is the method and the named-case framing.
"""

F7_CODE = r'''
# --- Finding 7: prospective NAMED case studies ----------------------------------
# Builds on Finding 5: same time-split, but instead of one AUROC we rank every drug
# against each held-out cancer using only pre-T structure and report where the real
# future indication landed.
import numpy as np

# future indications = pairs whose first-evidence year is after the cutoff T
future_pairs = [(d, c) for (d, c), y in pair_years.items() if y > T]
print(f"future (held-out) indications: {len(future_pairs)} | cutoff T={T}")


@torch.no_grad()
def rank_future(model, base, future_pairs):
    """Rank each future drug among all drugs for its cancer, excluding drugs already
    known as a PAST indication of that cancer in the message-passing graph `base`."""
    model.eval()
    z = model.encode(base)
    num_drugs = int(data_st[DRUG_TYPE].num_nodes)
    all_drugs = torch.arange(num_drugs, device=z[DRUG_TYPE].device)
    known_by_dis = {}
    ei_b = base[target_st].edge_index
    for col in range(ei_b.size(1)):
        known_by_dis.setdefault(int(ei_b[1, col]), set()).add(int(ei_b[0, col]))
    out = {}
    for c in sorted({cc for _, cc in future_pairs}):
        eli = torch.stack([all_drugs, torch.full((num_drugs,), c, device=all_drugs.device)])
        sc = torch.sigmoid(model.decode(z, target_st, eli)).cpu().numpy()
        pool = np.ones(num_drugs, dtype=bool)
        for d in known_by_dis.get(c, ()):
            pool[d] = False
        psize = int(pool.sum())
        for (d, cc) in future_pairs:
            if cc == c:
                out[(d, c)] = (int(((sc > sc[d]) & pool).sum()) + 1, psize)
    return out


if len(future_pairs) >= 3:
    set_all_seeds(0)
    sp_t = temporal_split(data_st, target_st, pair_years, T, onco_set_t, seed=0)
    gnn = HeteroGNN(list(data_st.node_types), list(sp_t.base.edge_types), in_dims_st).to(DEVICE)
    gnn = train_gnn(gnn, sp_t, DEVICE, epochs=GNN_EPOCHS, patience=PATIENCE)
    set_all_seeds(0)
    mlp = FeatureMLP(list(data_st.node_types), in_dims_st).to(DEVICE)
    mlp = train_mlp(mlp, sp_t, DEVICE, epochs=MLP_EPOCHS, patience=PATIENCE)

    gr = rank_future(gnn, sp_t.base, future_pairs)
    mr = rank_future(mlp, sp_t.base, future_pairs)

    K = 50
    rows = []
    for (d, c) in future_pairs:
        rows.append((rxnames_t[d], dnames_t[c].replace(" (disease)", ""),
                     pair_years[(d, c)], gr[(d, c)][0], gr[(d, c)][1], mr[(d, c)][0]))
    rows.sort(key=lambda r: r[3])
    pool_med = int(np.median([r[4] for r in rows]))
    g_top = sum(1 for r in rows if r[3] <= K)
    m_top = sum(1 for r in rows if r[5] <= K)
    enr = (g_top / len(rows)) / (K / max(1, pool_med))
    print(f"\nGNN put {g_top}/{len(rows)} future indications in the top-{K} of ~{pool_med} "
          f"candidates ({enr:.1f}x a random screen); structure-blind MLP put {m_top}.")
    print("(recorded full run, T=2005: GNN 11/101 in top-50 = 17.3x; MLP 1/101 = 1.6x)\n")
    print(f"{'drug':<22}{'cancer':<40}{'yr':>5}{'GNN rk':>8}{'MLP rk':>8}")
    print("-" * 83)
    for r in rows[:15]:
        print(f"{r[0][:21]:<22}{r[1][:39]:<40}{r[2]:>5}{r[3]:>8}{r[5]:>8}")
else:
    print("Not enough resolved future pairs in this sample; disable FAST_MODE for the full run.")
'''


# --------------------------------------------------------------------------- #
# Finding 8 -- popularity-deconfounded orthogonal validation (self-contained)
# --------------------------------------------------------------------------- #
F8_MD = r"""
## 12. Finding 8: a popularity-deconfounded orthogonal check

The repurposing shortlist (Section 8) raises an obvious question: do the model's novel
predictions show up as real human trials? A naive check (do top-scored novel pairs have
ClinicalTrials.gov entries more than random pairs) is badly confounded by popularity on two
sides. Some drugs are trialed for every cancer, and some cancers are trialed with every drug,
so a positive result can mean "popular drug" or "popular cancer" rather than "right drug for
this cancer". We remove both effects and test only what is left: the drug $\times$ cancer
interaction.

We fit an additive two-way model, $\mathrm{score}(d,c)=\mu+\alpha_d+\beta_c+\varepsilon_{dc}$,
subtract the drug effect $\alpha_d$ and the cancer effect $\beta_c$, and ask whether the
residual $\varepsilon_{dc}$ still predicts a real trial. We also report the simpler
within-drug AUROC, which removes drug popularity only.

What we recorded on the full run (about 100 drugs by 18 cancers, roughly 1{,}800 pairs): the
within-drug AUROC is a strong-looking 0.821 ($p=5\times10^{-4}$), but once the cancer effect
is also removed the interaction AUROC is 0.475 ($p=0.85$), that is, no signal. The apparent
effect was cancer popularity (cervical cancer, for instance, scores about 0.998 for almost
every drug and is widely trialed), not pairwise specificity. This is an honest, fully
characterized negative, and it stands in deliberate contrast to the prospective result
(Findings 5 and 7), where specific predictive signal does appear.

This cell makes live ClinicalTrials.gov calls (cached to disk). In fast mode it uses only a
handful of drugs and cancers, so the live numbers are noisy; the recorded full-run numbers
above are the ones to read.
"""

F8_CODE = r'''
# --- Finding 8: popularity-DECONFOUNDED orthogonal validation -------------------
import numpy as np, urllib.parse, urllib.request, json as _json, time, re as _re
from pathlib import Path as _Path
from sklearn.metrics import roc_auc_score

# Ranking drugs by name needs real semantic features (the fast-mode hashing features
# make the top-scored "drugs" arbitrary chemicals with no trials). Reuse Finding 5's
# ST-feature graph if present, else load it here so this part stands alone.
try:
    data_v, target_v = data_st, target_st
except NameError:
    data_v, _targets_v = load_primekg(with_features=True, force_fallback_features=False)
    target_v = _targets_v["indication"]

# A small bounded demo (live ClinicalTrials.gov). The recorded full run uses ~100
# drugs x 18 cancers; here we keep it tiny so the notebook finishes quickly.
DEMO_CANCERS = ["glioblastoma", "melanoma", "breast carcinoma", "ovarian cancer",
                "prostate cancer", "colorectal cancer"]
N_FOCUS  = 8 if FAST_MODE else 40
CAP_DRUGS = 12 if FAST_MODE else 80

# 1) Train a transductive deployment GNN on indication edges and score drugs x cancers.
set_all_seeds(0)
sp = make_split(data_v, target_v, "transductive", seed=0, val_frac=0.1, test_frac=0.0)
in_dims_v = {nt: int(data_v[nt].x.size(1)) for nt in data_v.node_types}
dep = HeteroGNN(list(data_v.node_types), list(sp.base.edge_types), in_dims_v).to(DEVICE)
dep = train_gnn(dep, sp, DEVICE, epochs=GNN_EPOCHS, patience=PATIENCE)

dnames_v  = list(data_v[DISEASE_TYPE].node_names)
rxnames_v = list(data_v[DRUG_TYPE].node_names)
onc = data_v[DISEASE_TYPE].is_oncology
name2dz = {}
for q in DEMO_CANCERS:
    for i, nm in enumerate(dnames_v):
        if q in str(nm).lower() and bool(onc[i]):
            name2dz[q] = i
            break
dz_ids = list(dict.fromkeys(name2dz.values()))

def _known_dd(data):
    known = set()
    for et in data.edge_types:
        s, r, d = et
        if {s, d} == {DRUG_TYPE, DISEASE_TYPE} and any(k in r for k in ("indication", "contra", "off-label")):
            ei = data[et].edge_index
            for a, b in zip(ei[0].tolist(), ei[1].tolist()):
                known.add((a, b) if s == DRUG_TYPE else (b, a))
    return known
known = _known_dd(data_v)

@torch.no_grad()
def score_all(model, dz):
    z = model.encode(sp.base)
    nd = int(data_v[DRUG_TYPE].num_nodes)
    eli = torch.stack([torch.arange(nd, device=z[DRUG_TYPE].device),
                       torch.full((nd,), dz, device=z[DRUG_TYPE].device)])
    return torch.sigmoid(model.decode(z, target_v, eli)).cpu().numpy()

scores_by_dz = {dz: score_all(dep, dz) for dz in dz_ids}

# focus drugs = union of each cancer's top-N novel (not-already-indicated) drugs
focus, seen = [], set()
for dz in dz_ids:
    cnt = 0
    for d in np.argsort(-scores_by_dz[dz]):
        d = int(d)
        if (d, dz) in known:
            continue
        if d not in seen:
            seen.add(d); focus.append(d)
        cnt += 1
        if cnt >= N_FOCUS:
            break
focus = focus[:CAP_DRUGS]

# 2) Query ClinicalTrials.gov (cached) for each (focus drug, demo cancer) novel pair.
CT_CACHE = _Path("data/clinicaltrials_cache.json")
ctc = _json.loads(CT_CACHE.read_text()) if CT_CACHE.exists() else {}

def ct_hit(drug, cond):
    key = drug.strip().lower() + "|||" + cond.strip().lower()
    if key in ctc and "error" not in ctc[key]:
        return int(ctc[key]["hit"])
    qd = _re.sub(r"[^A-Za-z0-9 -]", " ", str(drug)).strip()
    qc = _re.sub(r"[^A-Za-z0-9 -]", " ", str(cond)).strip()
    if not qd or not qc:
        ctc[key] = {"hit": 0, "total": 0}; return 0
    params = {"query.intr": qd, "query.cond": qc,
              "filter.advanced": "AREA[StudyType]INTERVENTIONAL",
              "countTotal": "true", "pageSize": "1", "format": "json"}
    url = "https://clinicaltrials.gov/api/v2/studies?" + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "oncoevidence/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            tot = int(_json.load(r).get("totalCount", 0) or 0)
        ctc[key] = {"hit": int(tot > 0), "total": tot}
    except Exception:
        return 0
    time.sleep(0.34)
    return ctc[key]["hit"]

dz_clean = {dz: str(dnames_v[dz]).replace(" (disease)", "") for dz in dz_ids}
S = np.full((len(focus), len(dz_ids)), np.nan)
H = np.full_like(S, np.nan)
for i, d in enumerate(focus):
    for j, dz in enumerate(dz_ids):
        if (d, dz) in known:
            continue
        S[i, j] = scores_by_dz[dz][d]
        H[i, j] = ct_hit(rxnames_v[d], dz_clean[dz])
CT_CACHE.parent.mkdir(exist_ok=True)
CT_CACHE.write_text(_json.dumps(ctc))
mask = ~np.isnan(S)

# 3) Remove drug + cancer popularity (two-way fixed effects); test the residual.
mu = float(np.nanmean(S)); a = np.zeros(S.shape[0]); b = np.zeros(S.shape[1])
Sc = np.where(mask, S, np.nan)
for _ in range(200):
    a = np.nan_to_num(np.nanmean(Sc - mu - b[None, :], axis=1))
    b = np.nan_to_num(np.nanmean(Sc - mu - a[:, None], axis=0))
R = Sc - mu - a[:, None] - b[None, :]
r = R[mask]; h = H[mask].astype(int)

conc = comp = 0.0   # within-drug AUROC (controls drug popularity only)
for i in range(S.shape[0]):
    sr = S[i, mask[i]]; hr = H[i, mask[i]].astype(int)
    pos = sr[hr == 1]; neg = sr[hr == 0]
    for sp_ in pos:
        comp += len(neg); conc += np.sum(sp_ > neg) + 0.5 * np.sum(sp_ == neg)
within = conc / comp if comp else float("nan")
inter = roc_auc_score(h, r) if 0 < h.sum() < len(h) else float("nan")

print(f"demo pairs={int(mask.sum())}  trial-hit fraction={np.nanmean(H):.3f}")
print(f"within-drug AUROC (drug popularity removed)  = {within:.3f}")
print(f"interaction AUROC (drug AND cancer removed)   = {inter:.3f}")
print("\n(recorded full run: within-drug 0.821 [p=5e-4] but interaction 0.475 [p=0.85] -->")
print(" the apparent signal is cancer popularity, not pairwise specificity: an honest negative.)")
'''


# --------------------------------------------------------------------------- #
# 1) Build Part 7 and Part 8 from Part 5's canonical building blocks
# --------------------------------------------------------------------------- #
def build_parts():
    p5 = load(PART5)
    cells = p5["cells"]
    # locate the Finding 5 markdown ("## 9. Finding 5")
    f5_md = next(i for i, c in enumerate(cells)
                 if c["cell_type"] == "markdown" and "Finding 5" in src(c))
    setup_through_primekg = copy.deepcopy(cells[:f5_md])         # header .. PrimeKG inspect
    finding5_pair = copy.deepcopy(cells[f5_md:f5_md + 2])        # Finding 5 md + code

    # ---- Part 7 -------------------------------------------------------------
    p7_header = md(
        "# OncoEvidence, Part 7 of 8: Prospective named cases\n\n"
        "From a prospective AUROC to concrete, named predictions. (Extension.)\n\n" + TRACK8 + "\n\n"
        "What Part 7 covers:\n\n"
        "- A recap of the Finding 5 time-split, which this part reruns so it stands alone.\n"
        "- Turning that single number into named stories: for each held-out future indication, "
        "rank every drug against the cancer using only pre-cutoff structure and read off where the "
        "real drug landed.\n"
        "- The result: the graph surfaces real, later-established indications (recorded full run: "
        "11x as many in the top 50 as a structure-blind control), including recognizable post-2005 "
        "approvals the model had no edge for.\n\n"
        "This part trains models and reuses Finding 5's resolved publication years, so a GPU helps "
        "and the first run is slower.")
    p7_footer = md(
        "## Recap and next: Part 8\n\n"
        "We turned the prospective AUROC into named cases: trained only on pre-cutoff structure, the "
        "graph ranks real future indications near the top of a blind screen, far above a "
        "structure-blind control. That is the strongest evidence that the model is genuinely "
        "predictive rather than merely self-consistent. Part 8 "
        "(`oncoevidence_part8_deconfounded.ipynb`) turns a skeptical eye on the other direction: it "
        "tests the shortlist against ClinicalTrials.gov while carefully removing popularity "
        "confounds, and reports an honest negative.")
    p7 = {"cells": [p7_header] + setup_through_primekg[1:] + finding5_pair
                    + [md(F7_MD), code(F7_CODE), p7_footer],
          "metadata": p5["metadata"], "nbformat": 4, "nbformat_minor": 0}
    dump(PART7, p7)

    # ---- Part 8 -------------------------------------------------------------
    p8_header = md(
        "# OncoEvidence, Part 8 of 8: Popularity-deconfounded validation\n\n"
        "Does the shortlist hold up against real trials, once popularity is removed? (Extension.)\n\n"
        + TRACK8 + "\n\n"
        "What Part 8 covers:\n\n"
        "- The confound: a naive ClinicalTrials.gov check is driven by popularity on two sides "
        "(popular drugs and popular cancers are trialed for everything).\n"
        "- The fix: fit an additive drug-effect plus cancer-effect model to the scores, subtract "
        "both, and test whether the drug-by-cancer interaction residual still predicts a real "
        "trial.\n"
        "- The result, reported honestly: the within-drug signal looks strong (recorded 0.821) but "
        "vanishes once cancer popularity is also removed (interaction 0.475, no signal). The "
        "apparent effect was popularity, not pairwise specificity, a clean negative that contrasts "
        "with the prospective wins in Parts 5 and 7.\n\n"
        "This part trains a deployment GNN and makes live ClinicalTrials.gov calls (cached), so a "
        "GPU helps and the first run touches the network.")
    p8_footer = md(
        "## End of the 8-part track\n\n"
        "That completes the extended track. Parts 1 to 3 build the pipeline, Part 4 is the full "
        "self-contained reference, and Parts 5 to 8 stress-test the system on the questions that "
        "decide whether it is trustworthy: is it predictive over time (Parts 5 and 7), is its "
        "mechanism reasoning faithful (Part 6), and does an independent registry corroborate the "
        "specific shortlist once popularity is removed (Part 8, an honest negative). Taken "
        "together, the graph earns its place not by out-ranking a tuned tabular model, but by "
        "producing a traceable, time-predictive, faithfully-grounded mechanism, and we are equally "
        "explicit about where the external signal does not (yet) hold.")
    p8 = {"cells": [p8_header] + setup_through_primekg[1:] + [md(F8_MD), code(F8_CODE), p8_footer],
          "metadata": p5["metadata"], "nbformat": 4, "nbformat_minor": 0}
    dump(PART8, p8)


# --------------------------------------------------------------------------- #
# 2) Insert Finding 7 / 8 into the master (idempotent) and refresh Part 4
# --------------------------------------------------------------------------- #
def update_master():
    nb = load(MASTER)
    cells = nb["cells"]

    def is_new(c):
        s = src(c)
        return ("Finding 7: prospective NAMED" in s or "Finding 8: a popularity" in s
                or "Finding 7: prospective NAMED case" in s
                or "Finding 8: popularity-DECONFOUNDED" in s
                or "## 11. Finding 7" in s or "## 12. Finding 8" in s)

    cells = [c for c in cells if not is_new(c)]
    wrap = next(i for i, c in enumerate(cells)
                if c["cell_type"] == "markdown" and src(c).lstrip().startswith("## Wrap-up"))
    new_cells = [md(F7_MD), code(F7_CODE), md(F8_MD), code(F8_CODE)]
    cells[wrap:wrap] = new_cells

    # update wrap-up prose: six -> eight findings + two bullets
    w = cells[wrap + len(new_cells)]
    s = src(w)
    s = s.replace("The six findings again:", "The eight findings again:")
    if "7. Prospective named cases" not in s:
        s = s.replace(
            "6. A counterfactual test shows the mechanism head is faithful: deleting the true mechanism edge hurts far more than deleting a random one.",
            "6. A counterfactual test shows the mechanism head is faithful: deleting the true mechanism edge hurts far more than deleting a random one.\n"
            "7. Prospective named cases: trained on pre-2005 structure, the graph surfaces real later approvals (romidepsin, pralatrexate, nilotinib) near the top of a blind screen, 11x as often as a structure-blind control.\n"
            "8. An honest, fully characterized negative: once both drug and cancer popularity are removed, the ClinicalTrials interaction signal vanishes (AUROC 0.475), so the raw scores track popularity, not pairwise specificity.")
    w["source"] = s.splitlines(keepends=True)

    nb["cells"] = cells
    dump(MASTER, nb)
    # Part 4 is a verbatim copy of the master.
    Path(PART4).write_text(json.dumps(nb, indent=1))
    print(f"wrote {PART4}: {len(nb['cells'])} cells (copy of master)")


# --------------------------------------------------------------------------- #
# 3) Relabel the existing track notebooks from "of 6" to "of 8"
# --------------------------------------------------------------------------- #
def relabel_existing():
    targets = [NB / f"oncoevidence_part{i}_{n}.ipynb" for i, n in [
        (1, "foundations"), (2, "mechanisms"), (3, "evidence_and_recovery"),
        (5, "prospective"), (6, "faithfulness")]]
    for p in targets:
        nb = load(p)
        changed = 0
        for c in nb["cells"]:
            if c["cell_type"] != "markdown":
                continue
            s = src(c)
            o = s
            s = s.replace(" of 6:", " of 8:")
            s = s.replace("one of six notebooks", "one of eight notebooks")
            s = s.replace(OLD_TRACK_FRAGMENT, NEW_TRACK_FRAGMENT)
            s = s.replace("6-part track", "8-part track")
            if s != o:
                c["source"] = s.splitlines(keepends=True)
                changed += 1
        # Part 6 was the old final notebook: repoint its closing cell to Part 7.
        if p.name.endswith("part6_faithfulness.ipynb"):
            last = nb["cells"][-1]
            if "End of the 8-part track" in src(last) or "End of the" in src(last):
                last["source"] = md(
                    "## Recap and next: Part 7\n\n"
                    "We showed the mechanism reasoning is faithful: deleting the curated mechanism "
                    "edge, and only that edge, collapses the prediction. That covers trust in the "
                    "explanation. Parts 7 and 8 go after the two remaining questions a reviewer "
                    "would ask. Part 7 (`oncoevidence_part7_named_cases.ipynb`) turns the "
                    "prospective result into concrete named predictions, and Part 8 "
                    "(`oncoevidence_part8_deconfounded.ipynb`) checks the shortlist against "
                    "ClinicalTrials.gov with popularity carefully removed.")["source"]
                changed += 1
        dump(p, nb)
        print(f"  relabeled {p.name}: {changed} cells touched")


if __name__ == "__main__":
    update_master()      # master + Part 4 first (so Part 4 has Findings 7/8)
    build_parts()        # Part 7 + Part 8
    relabel_existing()   # Parts 1,2,3,5,6 -> "of 8"
    print("done.")
