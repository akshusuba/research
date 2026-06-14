"""Configuration: paths, PrimeKG schema constants, target relations, oncology selection."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

for _d in (DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# PrimeKG raw knowledge graph (Harvard Dataverse).
PRIMEKG_KG_CSV = DATA_DIR / "kg.csv"
PRIMEKG_KG_URL = "https://dataverse.harvard.edu/api/access/datafile/6180620"

# Cached processed artifacts.
HETERODATA_PT = DATA_DIR / "primekg_hetero.pt"
FEATURE_CACHE = MODELS_DIR / "primekg_text_features.pt"

# PrimeKG drug<->disease therapeutic relations (the prediction targets).
# In PrimeKG these appear as the `relation` column values below.
INDICATION_REL = "indication"
CONTRAINDICATION_REL = "contraindication"
OFFLABEL_REL = "off-label use"
THERAPEUTIC_RELS = (INDICATION_REL, CONTRAINDICATION_REL, OFFLABEL_REL)

DRUG_TYPE = "drug"
DISEASE_TYPE = "disease"

# Sentence-transformer used for shared node features (fed to both GNN and XGBoost).
TEXT_MODEL = "all-MiniLM-L6-v2"

# Keywords used to flag oncology / neoplasm diseases when an ontology subtree
# is unavailable. Matched case-insensitively against the disease name.
ONCOLOGY_KEYWORDS = (
    "cancer", "carcinoma", "sarcoma", "neoplasm", "tumor", "tumour",
    "leukemia", "leukaemia", "lymphoma", "melanoma", "glioma", "glioblastoma",
    "myeloma", "blastoma", "adenoma", "malignant", "metastat", "oncocytoma",
    "mesothelioma", "astrocytoma", "teratoma", "carcinosarcoma",
)

DEFAULT_SEEDS = (0, 1, 2, 42, 123)
