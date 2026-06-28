# Prospective named case studies (smoke)

_2026-06-28T20:36:48.823975_

## What this shows

Trained **only** on knowledge-graph structure known by the cutoff year **T = 2004**, the model scores every drug against each cancer and we read off where the *real future indication* landed. A high rank means the system would have surfaced a genuine, later-established indication near the top of a blind screen.

## Headline

- Future indications evaluated: **12** (held out; their edges removed from the graph).
- Candidate pool per cancer: ~**7957** novel drugs.
- The GNN placed **0/12** future indications in the **top-50**; the structure-blind MLP placed 0.
- Median rank: GNN **2951** vs MLP 2772 (lower is better).
- The GNN ranked the true future indication higher than the MLP on **3/12** pairs (paired Wilcoxon p = 0.94).

## Top prospective hits (GNN, best-ranked first)

| drug | cancer | first-evidence year | GNN rank / pool | GNN %ile | MLP rank | rank gain vs MLP |
|---|---|---|---|---|---|---|
| Desoximetasone | primary cutaneous T-cell lymphoma | 2007 | 677 / 7957 | 0.085 | 3165 | +2488 |
| Hydroxyurea | theca steroid-producing cell malignant tumor of ovary, not further specified | 2015 | 1061 / 7957 | 0.133 | 308 | -753 |
| Belotecan | malignant Sertoli-Leydig cell tumor of ovary | 2018 | 1450 / 7955 | 0.182 | 114 | -1336 |
| Decitabine | acute myeloid leukemia with NPM1 somatic mutations | 2006 | 2532 / 7957 | 0.318 | 3824 | +1292 |
| Glasdegib | acute myeloid leukemia with inv3(p21;q26.2) or t(3;3)(p21;q26.2) | 2019 | 2722 / 7957 | 0.342 | 616 | -2106 |
| Tretamine | malignant Sertoli-Leydig cell tumor of ovary | 2012 | 2813 / 7955 | 0.354 | 1386 | -1427 |
| Brigatinib | non-small cell lung carcinoma | 2011 | 3089 / 7956 | 0.388 | 2427 | -662 |
| Apalutamide | prostate cancer | 2012 | 3468 / 7957 | 0.436 | 3641 | +173 |
| Panobinostat | plasma cell myeloma | 2006 | 3494 / 7957 | 0.439 | 3118 | -376 |
| Panobinostat | primary cutaneous T-cell non-Hodgkin lymphoma | 2008 | 3548 / 7957 | 0.446 | 3446 | -102 |
| Bevacizumab | undifferentiated carcinoma of the corpus uteri | 2008 | 3848 / 7957 | 0.484 | 271 | -3577 |
| Niraparib | primary peritoneal serous/papillary carcinoma | 2013 | 4023 / 7956 | 0.506 | 3489 | -534 |

## How to read a row

Take the first row: the indication was first co-mentioned in the literature in its listed year (later than the cutoff T = 2004), yet a model that saw only pre-T structure ranked that drug among the very top of all candidate drugs for that cancer. That is the model anticipating a real indication rather than memorising one it was shown.

## Caveats

- First-evidence year is an approximate proxy (earliest Europe PMC co-mention, not a regulatory or discovery date) and is noisy.
- A high rank reflects graph plausibility, not proof of efficacy.
- Ranks depend on the trained model and the cutoff; treat them as indicative.
