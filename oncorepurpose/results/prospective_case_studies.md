# Prospective named case studies (full)

_2026-06-28T20:41:06.995126_

## What this shows

Trained **only** on knowledge-graph structure known by the cutoff year **T = 2005**, the model scores every drug against each cancer and we read off where the *real future indication* landed. A high rank means the system would have surfaced a genuine, later-established indication near the top of a blind screen.

## Headline

- Future indications evaluated: **101** (held out; their edges removed from the graph).
- Candidate pool per cancer: ~**7956** novel drugs, so a random screen would put only 50/7956 (~0.6%) of true future indications in the top-50.
- The GNN placed **11/101** future indications in the **top-50** -- a **17.3x enrichment** over a random screen. The structure-blind MLP placed 1 (1.6x), so the graph yields **11x** as many top-50 prospective hits.

_Honest nuance:_ the graph's advantage is concentrated where a shortlist actually matters -- the very top of the ranking. Across the *bulk* of pairs the structure-blind control is competitive (median rank GNN 3019 vs MLP 1482; GNN ranks higher on 34/101 pairs). For prospective triage, precision at the top is the metric that counts, and there the graph wins decisively.

## Top prospective hits (GNN, best-ranked first)

| drug | cancer | first-evidence year | GNN rank / pool | GNN %ile | MLP rank | rank gain vs MLP |
|---|---|---|---|---|---|---|
| Trabectedin | hereditary site-specific ovarian cancer syndrome | 2007 | 2 / 7957 | 0.000 | 3040 | +3038 |
| Trabectedin | yolk sac tumor | 2008 | 5 / 7953 | 0.001 | 3036 | +3031 |
| Trabectedin | malignant sex cord stromal tumor of ovary | 2010 | 5 / 7956 | 0.001 | 3039 | +3034 |
| Trabectedin | ovarian mucinous adenocarcinoma | 2006 | 5 / 7955 | 0.001 | 3038 | +3033 |
| Prednicarbate | primary cutaneous T-cell non-Hodgkin lymphoma | 2016 | 6 / 7957 | 0.001 | 1097 | +1091 |
| Methoxsalen | primary cutaneous T-cell non-Hodgkin lymphoma | 2006 | 10 / 7957 | 0.001 | 2154 | +2144 |
| Daunorubicin | acute myeloid leukemia with CEBPA somatic mutations | 2006 | 41 / 7957 | 0.005 | 1927 | +1886 |
| Daunorubicin | acute myeloid leukemia with NPM1 somatic mutations | 2006 | 41 / 7957 | 0.005 | 1929 | +1888 |
| Thiotepa | malignant dysgerminomatous germ cell tumor of ovary | 2019 | 42 / 7956 | 0.005 | 17 | -25 |
| Daunorubicin | acute myeloid leukemia with inv3(p21;q26.2) or t(3;3)(p21;q26.2) | 2009 | 45 / 7956 | 0.006 | 1926 | +1881 |
| Paclitaxel | malignant dysgerminomatous germ cell tumor of ovary | 2014 | 48 / 7956 | 0.006 | 255 | +207 |
| Paclitaxel | maligant granulosa cell tumor of ovary | 2010 | 54 / 7957 | 0.007 | 259 | +205 |
| Romidepsin | primary cutaneous aggressive epidermotropic CD8+ T-cell lymphoma | 2010 | 63 / 7957 | 0.008 | 1726 | +1663 |
| Romidepsin | primary cutaneous CD4+ small/medium-sized pleomorphic T-cell lymphoma | 2008 | 63 / 7957 | 0.008 | 1726 | +1663 |
| Flumethasone | primary cutaneous T-cell non-Hodgkin lymphoma | 2013 | 95 / 7957 | 0.012 | 529 | +434 |
| Pralatrexate | primary cutaneous CD4+ small/medium-sized pleomorphic T-cell lymphoma | 2009 | 321 / 7957 | 0.040 | 2235 | +1914 |
| Nilotinib | chronic myelogenous leukemia, BCR-ABL1 positive | 2006 | 342 / 7956 | 0.043 | 1098 | +756 |
| Hydroxyurea | theca steroid-producing cell malignant tumor of ovary, not further specified | 2015 | 361 / 7957 | 0.045 | 128 | -233 |
| Azacitidine | acute myeloid leukemia with CEBPA somatic mutations | 2006 | 362 / 7957 | 0.045 | 962 | +600 |
| Plerixafor | plasma cell myeloma | 2007 | 428 / 7956 | 0.054 | 1797 | +1369 |
| Pazopanib | cancer | 2006 | 438 / 7956 | 0.055 | 462 | +24 |
| Azacitidine | unclassified acute myeloid leukemia | 2006 | 452 / 7952 | 0.057 | 956 | +504 |
| Azacitidine | acute myeloid leukemia with inv3(p21;q26.2) or t(3;3)(p21;q26.2) | 2009 | 460 / 7956 | 0.058 | 961 | +501 |
| Sorafenib | fibrolamellar hepatocellular carcinoma | 2008 | 536 / 7957 | 0.067 | 1093 | +557 |
| Sorafenib | nonpapillary renal cell carcinoma | 2006 | 540 / 7957 | 0.068 | 1093 | +553 |

## How to read a row

Take the first row: the indication was first co-mentioned in the literature in its listed year (later than the cutoff T = 2005), yet a model that saw only pre-T structure ranked that drug among the very top of all candidate drugs for that cancer. That is the model anticipating a real indication rather than memorising one it was shown.

## Caveats

- First-evidence year is an approximate proxy (earliest Europe PMC co-mention, not a regulatory or discovery date) and is noisy.
- A high rank reflects graph plausibility, not proof of efficacy.
- Ranks depend on the trained model and the cutoff; treat them as indicative.
