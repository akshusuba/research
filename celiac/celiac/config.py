"""
Configuration for the Celiac Gut-Brain Knowledge Graph project.
"""

from pathlib import Path

# Directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Create directories
for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# GEO Dataset
GEO_ACCESSION = "GSE164883"

# CeD-associated taxa from meta-analyses
CED_MICROBIOME_TAXA = {
    # Enriched in CeD
    "Bacteroides": {"direction": "enriched", "source": "Arcila-Galvis 2022"},
    "Escherichia": {"direction": "enriched", "source": "Arcila-Galvis 2022"},
    "Staphylococcus": {"direction": "enriched", "source": "Arcila-Galvis 2022"},
    "Neisseria": {"direction": "enriched", "source": "Arcila-Galvis 2022"},
    "Haemophilus": {"direction": "enriched", "source": "Arcila-Galvis 2022"},
    "Prevotella": {"direction": "enriched", "source": "Arcila-Galvis 2022"},
    "Klebsiella": {"direction": "enriched", "source": "Caminero 2019"},
    "Pseudomonas": {"direction": "enriched", "source": "Caminero 2019"},
    # Depleted in CeD
    "Bifidobacterium": {"direction": "depleted", "source": "Arcila-Galvis 2022"},
    "Lactobacillus": {"direction": "depleted", "source": "Arcila-Galvis 2022"},
    "Streptococcus": {"direction": "depleted", "source": "Arcila-Galvis 2022"},
    "Faecalibacterium": {"direction": "depleted", "source": "Arcila-Galvis 2022"},
    "Blautia": {"direction": "depleted", "source": "Wacklin 2014"},
    "Dorea": {"direction": "depleted", "source": "Wacklin 2014"},
}

# Neurological HPO terms relevant to celiac
NEUROLOGICAL_HPO_TERMS = {
    "HP:0001251": "Ataxia",
    "HP:0001271": "Polyneuropathy",
    "HP:0002354": "Memory impairment",
    "HP:0000708": "Atypical behavior",
    "HP:0002076": "Migraine",
    "HP:0001250": "Seizures",
    "HP:0100543": "Cognitive impairment",
    "HP:0002078": "Truncal ataxia",
    "HP:0007340": "Lower limb muscle weakness",
    "HP:0009830": "Peripheral neuropathy",
    "HP:0001288": "Gait disturbance",
    "HP:0001260": "Dysarthria",
    "HP:0002080": "Intention tremor",
    "HP:0001272": "Cerebellar atrophy",
}

# Key CeD and gut-brain genes
KEY_GENES = [
    # Celiac-specific
    "TGM6",      # Transglutaminase 6 - gluten ataxia
    "TGM2",      # Transglutaminase 2 - celiac autoantigen
    "HLA-DQA1",  # HLA risk allele
    "HLA-DQB1",  # HLA risk allele
    "IL15",      # Inflammatory cytokine
    "IFNG",      # Interferon gamma
    # Tryptophan/serotonin pathway
    "TPH1",      # Tryptophan hydroxylase (serotonin synthesis)
    "TPH2",      # Tryptophan hydroxylase 2
    "IDO1",      # Indoleamine dioxygenase (kynurenine pathway)
    "SLC6A4",    # Serotonin transporter
    "HTR1A",     # Serotonin receptor 1A
    "HTR2A",     # Serotonin receptor 2A
    # SCFA receptors
    "FFAR2",     # Free fatty acid receptor 2 (GPR43)
    "FFAR3",     # Free fatty acid receptor 3 (GPR41)
    # Inflammatory
    "TLR4",      # Toll-like receptor 4
    "IL6",       # Interleukin 6
    "TNF",       # Tumor necrosis factor
    "NFKB1",     # NF-kB
]

# Key metabolites for gut-brain axis
KEY_METABOLITES = [
    # Tryptophan pathway
    "tryptophan",
    "serotonin",
    "kynurenine",
    "kynurenic_acid",
    "quinolinic_acid",
    "indole",
    "indole-3-propionic_acid",
    # SCFAs
    "acetate",
    "propionate",
    "butyrate",
    # Other
    "GABA",
    "glutamate",
    "dopamine",
]

# Evidence weights
EVIDENCE_WEIGHTS = {
    "high": 1.0,      # Strong, replicated evidence
    "medium": 0.6,    # Consistent but moderate support
    "low": 0.3,       # Early or indirect support
}

# Node type definitions
NODE_TYPES = ["gene", "microbe", "metabolite", "phenotype"]

# Edge type definitions (source_type, relation, target_type)
EDGE_TYPES = [
    ("microbe", "produces", "metabolite"),
    ("metabolite", "modulates", "gene"),
    ("gene", "associated_with", "phenotype"),
    ("gene", "coexpressed_with", "gene"),
    ("microbe", "cooccurs_with", "microbe"),
]
